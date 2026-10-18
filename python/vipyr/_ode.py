# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vp.ode: initial value problems y' = f(t, y, p), solved by lib/ode.

The integrators are Viv (lib/ode/: Dormand-Prince 5(4), Radau IIA of order
5, variable-order BDF, and AUTO, which starts explicit and moves to Radau
when it detects stiffness). The right-hand side must be Viv too, because the
integrator is compiled together with it and calls it directly:

    rhs="robertson"                  a problem of lib/ode/problems.viv
    rhs="model.viv:decay"            a def of your own .viv file
    rhs=kernels.decay                a def of a module from vp.load
    rhs=decay                        a Python function in py2viv's numeric
                                     subset, translated to Viv once

Whatever the form, the def's signature is `(t, y, p)`: time, state (an
array of n) and parameters (an array, `params=`), returning dy/dt as an
array of n. `events=` names a def of the same form returning an array of g
values, one per event, in the same file. The Jacobian is exact by default:
the driver writes one reverse-mode `grad` per row of the rhs. `jac="fd"`
takes forward differences instead; `jac="name"` uses a def of yours.

For each distinct (rhs, events, jac) the module writes a small driver .viv
that imports the problem and lib/ode/solve.viv, compiles it through vipyr's
cache (once per change), and calls it: `solve` for one initial value
problem, `batch` for many parameter sets and initial states on one output
grid in one call.

What a Viv rhs may not do, from Viv's function values: it is a top-level def
(not a closure made at run time), and its parameters come in through `p`,
not through variables captured from Python. A file passed by path is copied
beside the driver, so it may import Viv's library modules but not other
files beside it.
"""
from __future__ import annotations

import hashlib
import inspect
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import numpy as np

from . import _core, _spec

METHODS = {"dopri5": 0, "rk45": 0, "radau": 1, "bdf": 2, "auto": 3}
STATUS = {1: "The solver successfully reached the end of the integration interval.",
          2: "A termination event occurred.",
          -1: "Required step size is less than spacing between numbers.",
          -2: "The maximum number of steps was taken before the end of the interval.",
          -3: "The solution stopped being finite (an infinity or NaN appeared)."}
EPS = np.finfo(float).eps
JAC_FD, JAC_GIVEN = 0, 1

# lib/ode/problems.viv: name -> (rhs def, exact Jacobian def or None, events def or None)
BUILTIN = {
    "robertson": ("robertson", "robertson_jac", None),
    "vdpol": ("vdpol", "vdpol_jac", None),
    "hires": ("hires", "hires_jac", None),
    "oregonator": ("oregonator", "oregonator_jac", None),
    "e5": ("e5", "e5_jac", None),
    "kepler": ("kepler", None, None),
    "lotka": ("lotka", None, None),
    "ball": ("ball", None, "ball_hits"),
    "sir": ("sir", "sir_jac", None),
    "pk": ("pk", None, None),
}

_lock = threading.Lock()
_drivers: dict = {}
# py2viv's translations in this process, by the Python source translated
_translated: dict = {}


class OdeProblemError(ValueError):
    """The rhs, events or jac given cannot be made into a Viv problem."""


def _driver_dir() -> Path:
    forced = os.environ.get("VIPYR_CACHE")
    base = Path(forced) if forced else Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "vipyr"
    d = base / "ode"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _py2viv(funcs) -> str:
    """Python functions in py2viv's subset, as one Viv module's text."""
    src = ["import math", "import numpy as np", ""]
    for f in funcs:
        try:
            src.append(textwrap.dedent(inspect.getsource(f)))
        except (OSError, TypeError) as e:
            raise OdeProblemError("vp.ode: the source of %r is not available to translate (%s)" % (f, e)) from None
    key = "\n".join(src)
    hit = _translated.get(key)
    if hit is not None:
        return hit
    tool = _spec.repo_root() / "tools" / "py2viv" / "py2viv.py"
    if not tool.is_file():
        raise OdeProblemError("vp.ode: a Python rhs needs tools/py2viv/py2viv.py from a Viv checkout")
    tmp = _driver_dir() / ("py_%s.py" % hashlib.sha256("\n".join(src).encode()).hexdigest()[:16])
    tmp.write_text("\n".join(src), encoding="utf-8")
    r = subprocess.run([sys.executable, str(tool), str(tmp)], capture_output=True, text=True)
    if r.returncode != 0:
        raise OdeProblemError("vp.ode: py2viv refused the Python rhs:\n" + (r.stderr or r.stdout).strip())
    _translated[key] = r.stdout
    return r.stdout


def _problem(rhs, events, jac):
    """(import line text, files to place beside the driver, rhs name, jac name or
    None, events name or None)."""
    if isinstance(rhs, str) and ":" not in rhs and not rhs.endswith(".viv"):
        if rhs not in BUILTIN:
            raise OdeProblemError("vp.ode: unknown problem %r; the built-in ones are %s, or give "
                                  "'file.viv:def'" % (rhs, ", ".join(sorted(BUILTIN))))
        name, jname, ename = BUILTIN[rhs]
        if isinstance(events, str):
            ename = events
        elif events is not None:
            raise OdeProblemError("vp.ode: events for a built-in problem name a def of lib/ode/problems.viv")
        if jac not in (None, "exact", "fd"):
            jname = jac
        names = [n for n in (name, jname, ename) if n]
        return "from ode.problems import %s" % ", ".join(dict.fromkeys(names)), {}, name, jname, ename
    if callable(rhs) and getattr(rhs, "__vipyr_source__", None) is None and not _vipyr_def(rhs):
        funcs = [rhs] + ([events] if callable(events) else [])
        text = _py2viv(funcs)
        mod = "odepy_" + hashlib.sha256(text.encode()).hexdigest()[:12]

        def viv_name(n):
            # py2viv keeps a name, or prefixes one Viv reserves (`py_f` becomes `u_py_f`)
            m = re.search(r"^def (\w*?%s)\(" % re.escape(n), text, re.M)
            if m is None:
                raise OdeProblemError("vp.ode: py2viv's translation has no def for %s" % n)
            return m.group(1)
        name = viv_name(rhs.__name__)
        ename = viv_name(events.__name__) if callable(events) else events
        jname = viv_name(jac) if jac not in (None, "exact", "fd") else None
        names = [n for n in (name, jname, ename) if n]
        return "from %s import %s" % (mod, ", ".join(dict.fromkeys(names))), {mod + ".viv": text}, name, jname, ename
    # a .viv file: "path.viv:def", or a def of a module vp.load made
    if callable(rhs):
        src, name = _vipyr_def(rhs)
    else:
        path, _, name = str(rhs).rpartition(":")
        if not path or not name:
            raise OdeProblemError("vp.ode: give rhs as 'file.viv:def'; got %r" % (rhs,))
        src = Path(path).expanduser().resolve()
    if not src.is_file():
        raise OdeProblemError("vp.ode: no file %s" % src)
    text = src.read_text(encoding="utf-8")
    mod = "odeuser_" + re.sub(r"\W", "_", src.stem)
    ename = events.__name__ if callable(events) else events
    jname = jac if jac not in (None, "exact", "fd") else None
    for n in (name, ename, jname):
        if n and not re.search(r"^def\s+%s\s*\(" % re.escape(n), text, re.M):
            raise OdeProblemError("vp.ode: %s has no top-level def %s" % (src, n))
    names = [n for n in (name, jname, ename) if n]
    return "from %s import %s" % (mod, ", ".join(dict.fromkeys(names))), {mod + ".viv": text}, name, jname, ename


def _vipyr_def(f):
    """(source .viv, def name) of a function of a module vp.load made, else None."""
    f = getattr(f, "__vipyr_raw__", f)
    mod = sys.modules.get(getattr(f, "__module__", None) or "")
    file = getattr(f, "__vipyr_source__", None)
    if file is None:
        g = getattr(f, "__globals__", {})
        if "__vipyr__" in g and str(g.get("__file__", "")).endswith(".viv"):
            file = g["__file__"]
    if file is None and mod is not None and hasattr(mod, "__vipyr__") and str(mod.__file__).endswith(".viv"):
        file = mod.__file__
    if file is None:
        return None
    return Path(file), getattr(f, "__name__", None)


DRIVER = '''###
post: vp.ode driver (written by vipyr/_ode.py)
agent: vipyr
###

{imports}
from ode.solve import ode_batch_flat, ode_solve_flat, no_events, no_jac

{jacdefs}
def solve(t0: Num, t1: Num, y0: Arr, p: Arr, method: Int, rtol: Num, atol: Arr, t_eval: Arr, dirs: Arr,
          term: Arr, first_step: Num, max_step: Num, max_steps: Int, jac_mode: Int) {{
    return ode_solve_flat({rhs}, {jac}, {ev}, t0, t1, y0, p, method, rtol, atol, t_eval, dirs, term, first_step,
                          max_step, max_steps, jac_mode)
}}

def batch(t0: Num, t1: Num, Y0: Arr, P: Arr, method: Int, rtol: Num, atol: Arr, t_eval: Arr, dirs: Arr,
          term: Arr, first_step: Num, max_step: Num, max_steps: Int, jac_mode: Int) {{
    return ode_batch_flat({rhs}, {jac}, {ev}, t0, t1, Y0, P, method, rtol, atol, t_eval, dirs, term, first_step,
                          max_step, max_steps, jac_mode)
}}
'''

AD_JAC = '''def _ode_row(y: Arr, t: Num, p: Arr, i) -> Num {{
    return {rhs}(t, y, p)[i]
}}

def _ode_jac(t: Num, y: Arr, p: Arr) -> Arr {{
    n = len(y)
    J = zeros(n, n)
    for i to n {{
        J[i, :] = grad(_ode_row, y, t, p, i)
    }}
    return J
}}
'''


def driver(rhs, events=None, jac=None, fast=False):
    """The compiled driver module for one problem: (module, jac mode, key).
    The key is the SHA-256 prefix of the driver and the problem's source."""
    imports, files, name, jname, ename = _problem(rhs, events, jac)
    mode = JAC_FD if jac == "fd" else JAC_GIVEN
    jacdefs = ""
    if mode == JAC_GIVEN and not jname:
        jacdefs, jname = AD_JAC.format(rhs=name), "_ode_jac"
    text = DRIVER.format(imports=imports, jacdefs=jacdefs, rhs=name, jac=jname or "no_jac", ev=ename or "no_events")
    key = hashlib.sha256((text + "\0" + "\0".join(k + "\0" + v for k, v in sorted(files.items()))).encode()
                         ).hexdigest()[:16]
    with _lock:
        hit = _drivers.get((key, fast))
        if hit is not None:
            return hit, mode, key
        d = _driver_dir() / key
        d.mkdir(parents=True, exist_ok=True)
        for fname, ftext in files.items():
            p = d / fname
            if not p.is_file() or p.read_text(encoding="utf-8") != ftext:
                p.write_text(ftext, encoding="utf-8")
        drv = d / ("odedrv_%s.viv" % key)
        if not drv.is_file() or drv.read_text(encoding="utf-8") != text:
            drv.write_text(text, encoding="utf-8")
        mod = _core.load(drv, fast=fast)
        _drivers[(key, fast)] = mod
        return mod, mode, key


def _raw(mod, name):
    f = getattr(mod, name)
    return getattr(f, "__vipyr_raw__", f)


def _common(t_span, n, method, rtol, atol, events_given, direction, terminal, first_step, max_step, max_steps):
    if method not in METHODS:
        raise ValueError("vp.ode: method must be one of %s; got %r" % (", ".join(METHODS), method))
    try:
        t0, t1 = (float(t_span[0]), float(t_span[1]))
    except (TypeError, IndexError, ValueError):
        raise TypeError("vp.ode: t_span must be (t0, t1); got %r" % (t_span,)) from None
    if not (math.isfinite(t0) and math.isfinite(t1)):
        raise ValueError("vp.ode: t_span must be finite; got %r" % (t_span,))
    rtol = float(rtol)
    if not rtol > 0:
        raise ValueError("vp.ode: rtol must be positive; got %r" % rtol)
    rtol = max(rtol, 100 * EPS)          # as SciPy: below 100 eps the error test means nothing
    atol = np.broadcast_to(np.asarray(atol, dtype=float), (n,)).copy()
    if np.any(atol < 0):
        raise ValueError("vp.ode: atol must be non-negative")
    if events_given:
        m = None if direction is None else np.atleast_1d(np.asarray(direction, dtype=float))
        k = None if terminal is None else np.atleast_1d(np.asarray(terminal, dtype=float))
        m_ev = len(m) if m is not None else (len(k) if k is not None else 1)
        dirs = np.zeros(m_ev) if m is None else np.broadcast_to(m, (m_ev,)).astype(float)
        term = np.zeros(m_ev) if k is None else np.broadcast_to(k, (m_ev,)).astype(float)
    else:
        dirs, term = np.zeros(0), np.zeros(0)
    fs = 0.0 if first_step is None else float(first_step)
    ms = math.inf if max_step is None else float(max_step)
    return t0, t1, rtol, atol, dirs, term, fs, ms, int(max_steps)


def _solve(rhs, t_span, y0, *, method="auto", rtol=1e-3, atol=1e-6, params=None, t_eval=None, events=None,
           direction=None, terminal=None, jac=None, first_step=None, max_step=None, max_steps=1_000_000, fast=False):
    """vp.ode.solve, before the result object is made (vipyr/_runtime.py wraps it)."""
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    if y0.ndim != 1:
        raise ValueError("vp.ode: y0 must be one state (rank 1); use vp.ode.batch for many")
    n = len(y0)
    t0, t1, rtol, atol, dirs, term, fs, ms, mx = _common(t_span, n, method, rtol, atol, events is not None,
                                                         direction, terminal, first_step, max_step, max_steps)
    p = np.atleast_1d(np.asarray([] if params is None else params, dtype=float))
    te = np.zeros(0) if t_eval is None else np.atleast_1d(np.asarray(t_eval, dtype=float))
    if te.size and (np.any(np.diff(te) * (t1 - t0) < 0) or min(te.min(), te.max()) < min(t0, t1)
                    or max(te.min(), te.max()) > max(t0, t1)):
        raise ValueError("vp.ode: t_eval must be sorted in the direction of integration and lie in t_span")
    mod, mode, key = driver(rhs, events, jac, fast)
    t, y, tev, yev, iev, st = _raw(mod, "solve")(t0, t1, y0, p, METHODS[method], rtol, atol, te, dirs, term, fs, ms,
                                                  mx, mode)
    y = np.asarray(y).reshape(-1, n).T
    yev = np.asarray(yev).reshape(-1, n)
    iev = np.asarray(iev).astype(int)
    status = int(st[0])
    m_ev = len(dirs)
    return {
        "t": np.asarray(t),
        "y": y,
        "t_events": [np.asarray(tev)[iev == k] for k in range(m_ev)] if events is not None else None,
        "y_events": [yev[iev == k] for k in range(m_ev)] if events is not None else None,
        "status": status,
        "message": STATUS.get(status, "status %d" % status),
        "success": status > 0,
        "stats": {"nfev": int(st[1]), "njev": int(st[2]), "nlu": int(st[3]), "steps": int(st[4]),
                  "rejected": int(st[5])},
        "stiff_at": float(st[6]) if st[6] >= 0 else None,
        "_problem": key,
    }


def _batch(rhs, t_span, y0, params, *, t_eval, method="auto", rtol=1e-3, atol=1e-6, events=None, direction=None,
           terminal=None, jac=None, first_step=None, max_step=None, max_steps=1_000_000, fast=False):
    """vp.ode.batch, before the result object is made."""
    P = np.asarray(params, dtype=float)
    if P.ndim == 1:
        P = P[:, None]
    if P.ndim != 2:
        raise ValueError("vp.ode: params must be (B, k), one row per solve")
    Y0 = np.asarray(y0, dtype=float)
    if Y0.ndim == 1:
        Y0 = np.broadcast_to(Y0, (P.shape[0], len(Y0)))
    if Y0.ndim != 2 or Y0.shape[0] != P.shape[0]:
        raise ValueError("vp.ode: y0 must be one state or one row per row of params; got %s for %d rows"
                         % (Y0.shape, P.shape[0]))
    nb, n = Y0.shape
    t0, t1, rtol, atol, dirs, term, fs, ms, mx = _common(t_span, n, method, rtol, atol, events is not None,
                                                         direction, terminal, first_step, max_step, max_steps)
    te = np.atleast_1d(np.asarray(t_eval, dtype=float))
    if te.size == 0:
        raise ValueError("vp.ode: a batch needs t_eval, the common output times")
    mod, mode, key = driver(rhs, events, jac, fast)
    Y, st, first = _raw(mod, "batch")(t0, t1, np.ascontiguousarray(Y0), np.ascontiguousarray(P), METHODS[method],
                                      rtol, atol, te, dirs, term, fs, ms, mx, mode)
    Y = np.asarray(Y).reshape(nb, len(te), n).transpose(0, 2, 1)
    st = np.asarray(st).reshape(nb, -1)
    return {
        "t": te,
        "y": Y,
        "status": st[:, 0].astype(int),
        "success": bool(np.all(st[:, 0] > 0)),
        "first_event": np.asarray(first) if events is not None else None,
        "stats": {"nfev": st[:, 1].astype(int), "njev": st[:, 2].astype(int), "nlu": st[:, 3].astype(int),
                  "steps": st[:, 4].astype(int), "rejected": st[:, 5].astype(int)},
        "_problem": key,
    }
