# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Kernel collection K1 against its references (SciPy and mpmath).

Each of the eight kernels in k1.export.toml is called through a Python
callable and checked against its reference, with the tolerance the manifest
states. The callable comes from one of two backends:

  export   the module `viv export kernels/k1/k1.export.toml` writes (k1.py,
           which builds libk1 from k1.c). Used when it can be imported: put
           its directory on K1_DIST, or on sys.path.
  native   each call writes a small Viv program with the inputs as literals,
           runs it with the `viv` binary and reads the results back from its
           printed lines (`fmt` prints the shortest text that reads back as
           the same double).

With the export backend, the results of the fixed cases are also hashed
(k1.results_hash) and compared with kernels/k1/k1.exact_hashes.json, which
`--record` writes after the library's results and the interpreter's are the
same bits for every kernel.

The fixed inputs are also written as programs in cases/ (one per kernel,
`python test_k1.py --write-cases`), so the Viv side can be run by hand:

  viv run -q kernels/k1/cases/case_boys.viv

Usage:

  python kernels/k1/test_k1.py               # table of worst errors; exit 1 on a failure
  python kernels/k1/test_k1.py --record      # export against the interpreter, bit for bit;
                                             # writes k1.exact_hashes.json
  python kernels/k1/test_k1.py --engine c    # the cases as native C (-O2 -ffp-contract=off),
                                             # and their output against the interpreter's, byte for byte
  python -m pytest kernels/k1/test_k1.py

Environment: VIV (the viv binary), CC (default cc), K1_BACKEND (export or
native), K1_DIST. Random cases with Hypothesis run only if it is installed.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
try:
    import tomllib                    # Python 3.11 and newer
except ModuleNotFoundError:           # 3.9 and 3.10: the backport, `pip install tomli`
    import tomli as tomllib
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import scipy.special as ssp
import mpmath

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CASES = HERE / "cases"
CASES_JSON = HERE / "k1.cases.json"          # the fixed cases, for the exported test
HASHES = HERE / "k1.exact_hashes.json"
MANIFEST = tomllib.loads((HERE / "k1.export.toml").read_text())
KERNELS = MANIFEST["kernels"]
U = np.finfo(float).eps / 2          # unit roundoff, 1.1e-16
EXACT_FLAGS = ["-O2", "-ffp-contract=off"]


def _find_viv() -> str:
    for cand in (os.environ.get("VIV"),
                 REPO / "target" / "release" / "viv",
                 REPO.parent.parent / "viv" / "target" / "release" / "viv",
                 shutil.which("viv")):
        if cand and Path(cand).is_file():
            return str(cand)
    raise RuntimeError("no viv binary: set VIV, or build target/release/viv")


class K1Error(ValueError):
    """A Viv run-time error (a `stop`, an index out of range) in a kernel."""


# ---------------------------------------------------------------------------
# Rendering a call as a Viv program
# ---------------------------------------------------------------------------

def _num(x) -> str:
    x = float(x)
    if not np.isfinite(x):
        raise ValueError("Viv literals are finite; NaN and infinity cannot be passed to the native backend")
    s = repr(x)
    return s if any(c in s for c in ".en") else s + ".0"


def _lit(value, vtype: str) -> str:
    if vtype == "Int":
        if isinstance(value, (bool, float)) or int(value) != value:
            raise TypeError(f"an Int argument needs an integer, got {value!r}")
        return str(int(value))
    if vtype == "Num":
        return _num(value)
    if vtype == "Complex":
        z = complex(value)
        return f"complex({_num(z.real)}, {_num(z.imag)})"
    if vtype == "Arr":
        a = np.asarray(value, dtype=float)
        if a.ndim == 1:
            return "zeros(0)" if a.size == 0 else "[" + ", ".join(_num(v) for v in a) + "]"
        if a.ndim == 2:
            if a.shape[1] == 0 or a.shape[0] == 0:
                return f"zeros({a.shape[0]}, {a.shape[1]})"
            return "[" + ", ".join("[" + ", ".join(_num(v) for v in row) + "]" for row in a) + "]"
        raise TypeError("K1's array arguments are rank 1 or 2")
    raise TypeError(f"no literal for Viv type {vtype}")


def _module_of(kernel: str) -> str:
    path, _, _ = KERNELS[kernel]["viv"].partition(":")
    return path.removesuffix(".viv").replace("/", ".")


PRINTERS = """\
def k1_num(c: Num, name: Str, x: Num) -> Num {
    print(c, name, fmt(x))
    return 0.0
}

def k1_cplx(c: Num, name: Str, z: Complex) -> Num {
    print(c, name + "@re", fmt(re(z)))
    print(c, name + "@im", fmt(im(z)))
    return 0.0
}

def k1_arr(c: Num, name: Str, M: Arr) -> Num {
    s = shape(M)
    if len(s) == 1 {
        print(c, name + "#shape", fmt(s[0]))
    }
    else {
        print(c, name + "#shape", fmt(s[0]), fmt(s[1]))
    }
    v = ravel(M)
    for i to len(v) {
        print(c, name, fmt(v[i]))
    }
    return 0.0
}
"""


def render(kernel: str, calls: list[tuple]) -> str:
    """A Viv program that makes each call in turn and prints its results."""
    spec = KERNELS[kernel]
    lines = [f"# generated by kernels/k1/test_k1.py for `{kernel}`; do not edit",
             f"from {_module_of(kernel)} import {kernel}", "", PRINTERS]
    printer = {"Num": "k1_num", "Complex": "k1_cplx", "Arr": "k1_arr"}
    names = [r["name"] for r in spec["results"]]
    for c, args in enumerate(calls):
        lits = ", ".join(_lit(a, s["type"]) for a, s in zip(args, spec["args"]))
        outs = [f"k1_{c}_{n.lower()}" for n in names]
        lines.append(f"{', '.join(outs)} = {kernel}({lits})")
        for out, r in zip(outs, spec["results"]):
            lines.append(f"junk_{out} = {printer[r['type']]}({c}, \"{r['name']}\", {out})")
    return "\n".join(lines) + "\n"


def parse(kernel: str, text: str, ncalls: int) -> list:
    """The printed lines back into each call's results, in the def's order."""
    spec = KERNELS[kernel]
    raw: list[dict] = [dict() for _ in range(ncalls)]
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        c, name, vals = int(parts[0]), parts[1], parts[2:]
        raw[c].setdefault(name, []).extend(float(v) for v in vals)
    results = []
    for c in range(ncalls):
        got = []
        for r in spec["results"]:
            n = r["name"]
            if r["type"] == "Num":
                got.append(raw[c][n][0])
            elif r["type"] == "Complex":
                got.append(complex(raw[c][n + "@re"][0], raw[c][n + "@im"][0]))
            else:
                shp = tuple(int(d) for d in raw[c][n + "#shape"])
                got.append(np.array(raw[c].get(n, []), dtype=float).reshape(shp))
        results.append(got[0] if len(got) == 1 else tuple(got))
    return results


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class NativeK1:
    """The kernels through the `viv` binary: the interpreter, or C via cc."""

    def __init__(self, engine: str = "interp"):
        self.viv = _find_viv()
        self.engine = engine
        self.name = f"native ({engine})"

    def run_text(self, text: str) -> str:
        # the program must sit inside the repository so `from kernels.k1...` resolves
        fd, path = tempfile.mkstemp(prefix=".k1_", suffix=".viv", dir=CASES)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
            return self.run_file(Path(path))
        finally:
            os.unlink(path)

    def run_file(self, path: Path) -> str:
        if self.engine == "interp":
            p = subprocess.run([self.viv, "run", "-q", str(path)], capture_output=True, text=True, cwd=REPO)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                csrc, exe = Path(tmp) / "k.c", Path(tmp) / "k.bin"
                q = subprocess.run([self.viv, str(path), "-o", str(csrc)], capture_output=True, text=True, cwd=REPO)
                if q.returncode:
                    raise K1Error(q.stderr.strip())
                q = subprocess.run([os.environ.get("CC", "cc"), *EXACT_FLAGS, str(csrc), "-o", str(exe), "-lm"],
                                   capture_output=True, text=True)
                if q.returncode:
                    raise RuntimeError("cc failed:\n" + q.stderr)
                p = subprocess.run([str(exe)], capture_output=True, text=True)
        if p.returncode:
            msg = (p.stderr or p.stdout).strip()
            raise K1Error(msg.splitlines()[0] if msg else f"exit status {p.returncode}")
        return p.stdout

    def batch(self, kernel: str, calls: list[tuple]) -> list:
        return parse(kernel, self.run_text(render(kernel, calls)), len(calls))

    def __getattr__(self, kernel: str):
        if kernel not in KERNELS:
            raise AttributeError(kernel)
        return lambda *args: self.batch(kernel, [args])[0]


class ExportK1:
    """The module `viv export` writes; one Python call per kernel call."""

    def __init__(self, module):
        self.mod = module
        self.name = "export"

    def batch(self, kernel: str, calls: list[tuple]) -> list:
        f = getattr(self.mod, kernel)
        return [f(*args) for args in calls]

    def __getattr__(self, kernel: str):
        return getattr(self.mod, kernel)


def load_backend(engine: str = "interp"):
    want = os.environ.get("K1_BACKEND", "")
    if want != "native":
        if os.environ.get("K1_DIST"):
            sys.path.insert(0, os.environ["K1_DIST"])
        try:
            import k1  # noqa: F401  (written by `viv export`)
            return ExportK1(k1)
        except ImportError:
            if want == "export":
                raise
    return NativeK1(engine)


# ---------------------------------------------------------------------------
# Fixed inputs
# ---------------------------------------------------------------------------

def _stable(rng, n, shift=0.5):
    A = rng.standard_normal((n, n))
    return A - (max(np.linalg.eigvals(A).real) + shift) * np.eye(n)


def _spd(rng, n):
    M = rng.standard_normal((n, n))
    return M @ M.T + n * np.eye(n)


def fixed_cases() -> dict[str, list[tuple]]:
    rng = np.random.default_rng(20261007)
    grid = [complex(x, y) for x in np.linspace(-6, 6, 13) for y in (-1.0, -0.4, 0.0, 0.3, 1.0, 2.5, 4.0)]
    ts = [0.0, 1e-12, 1e-6, 0.05, 0.3, 1.0, 2.5, 7.0, 15.0, 25.0, 34.3, 34.4, 34.5, 50.0, 80.0, 120.0, 200.0]
    boys = [(16, t) for t in ts] + [(0, 0.7), (1, 40.0), (5, 3.3), (10, 199.9)]
    en = [np.array([3.0, 4.0]), np.array([3.0, 4.0, 12.0, 84.0]) * 1e300,
          np.array([3.0, 4.0, 12.0, 84.0]) * 1e-300, np.array([5e-324, 1e-320, 2e-310]),
          np.array([1e200, 1.0, 1e-200]), np.array([3.834e-20, 3.8e-20, 1.0]),
          np.array([1.304e19, 1.0]), np.array([-7.5]), np.zeros(3), np.zeros(0)]
    en += [rng.standard_normal(n) * 10.0 ** rng.integers(-150, 150) for n in (1, 2, 7, 33, 100)]
    en += [rng.standard_normal(64) * 10.0 ** rng.integers(-300, 300, 64)]   # every scale at once
    schur = [(np.array([[2.0]]),), (np.array([[0.0, -1.0], [1.0, 0.0]]),),
             (np.array([[4.0, -2.0, 1.0], [3.0, 6.0, -4.0], [2.0, 1.0, 8.0]]),),
             (np.triu(rng.standard_normal((5, 5))),), (np.zeros((3, 3)),)]
    S = rng.standard_normal((6, 6))
    schur.append((S + S.T,))
    schur += [(rng.standard_normal((n, n)),) for n in (2, 4, 7, 9, 12)]
    care = [(np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.0], [1.0]]), np.eye(2), np.array([[1.0]]))]
    for n, m in ((2, 1), (3, 2), (4, 1), (6, 3), (8, 2)):
        A = rng.standard_normal((n, n))
        care.append((A, rng.standard_normal((n, m)), _spd(rng, n), _spd(rng, m)))
    lyap = [(np.array([[-1.0, 2.0], [0.0, -3.0]]), np.array([[1.0, 0.5], [0.5, 2.0]]))]
    for n in (1, 3, 5, 8, 10):
        Q = rng.standard_normal((n, n))
        lyap.append((_stable(rng, n), Q + Q.T))
    lyap.append((_stable(rng, 4), rng.standard_normal((4, 4))))     # Q need not be symmetric
    return {
        "boys": boys,
        "plasma_z": [(z,) for z in grid],
        "faddeeva_w": [(z,) for z in grid],
        "enorm": [(x,) for x in en],
        "schur": schur,
        "solve_care": care,
        "solve_lyapunov": lyap,
        "lqr": care,
    }


# ---------------------------------------------------------------------------
# Checks: each returns {metric: worst error}; the manifest gives each metric's tolerance
# ---------------------------------------------------------------------------

mpmath.mp.dps = 50


def _boys_ref(n, t):
    if t == 0.0:
        return mpmath.mpf(1) / (2 * n + 1)
    t = mpmath.mpf(t)
    a = n + mpmath.mpf(1) / 2
    return mpmath.gammainc(a, 0, t) / (2 * t ** a)


def _rel(a, b):
    return abs(a - b) / abs(b) if b != 0 else abs(a - b)


def check_boys(calls, outs):
    worst = 0.0
    for (n, t), F in zip(calls, outs):
        assert F.shape == (n + 1,)
        for k in range(n + 1):
            ref = _boys_ref(k, t)
            worst = max(worst, float(abs((mpmath.mpf(F[k]) - ref) / ref)))
    return {"rel": worst}


def check_faddeeva_w(calls, outs):
    return {"rel": max(_rel(w, ssp.wofz(z)) for (z,), w in zip(calls, outs))}


def check_plasma_z(calls, outs):
    ez = ep = 0.0
    for (z,), (Z, Zp) in zip(calls, outs):
        Zr = 1j * np.sqrt(np.pi) * ssp.wofz(z)
        ez = max(ez, _rel(Z, Zr))
        ep = max(ep, _rel(Zp, -2.0 * (1.0 + z * Zr)))
    return {"rel": ez, "rel_prime": ep}


def check_enorm(calls, outs):
    worst = vs = 0.0
    for (x,), r in zip(calls, outs):
        ref = mpmath.sqrt(mpmath.fsum(mpmath.mpf(v) ** 2 for v in x))
        if ref == 0:
            worst = max(worst, abs(r))
        else:
            worst = max(worst, float(abs((mpmath.mpf(r) - ref) / ref)))
        big = np.max(np.abs(x)) if len(x) else 0.0
        if 1e-150 < big < 1e150:      # SciPy's sum of squares neither overflows nor underflows
            vs = max(vs, _rel(r, sla.norm(x)))
    return {"rel": worst, "vs_scipy": vs}


def _quasi_triangular(T, tol):
    n = len(T)
    if n and np.max(np.abs(np.tril(T, -2))) > 0.0:
        return False
    sub = [abs(T[i + 1, i]) > tol for i in range(n - 1)]
    return not any(a and b for a, b in zip(sub, sub[1:]))


def _block_eigs(T):
    n, out, k = len(T), [], 0
    while k < n:
        if k + 1 < n and T[k + 1, k] != 0.0:
            out += list(np.linalg.eigvals(T[k:k + 2, k:k + 2]))
            k += 2
        else:
            out.append(complex(T[k, k]))
            k += 1
    return np.array(out)


def check_schur(calls, outs):
    res = orth = eig = 0.0
    for (A,), (T, Z) in zip(calls, outs):
        n = len(A)
        scale = max(np.max(np.abs(A)), np.finfo(float).tiny)
        res = max(res, np.max(np.abs(A - Z @ T @ Z.T)) / scale)
        orth = max(orth, np.max(np.abs(Z.T @ Z - np.eye(n))))
        assert _quasi_triangular(T, 0.0), "T is not quasi-triangular"
        mine = _block_eigs(T)
        ref = list(sla.eigvals(A))
        for lam in mine:          # match each eigenvalue to its nearest unused reference
            j = int(np.argmin([abs(lam - r) for r in ref]))
            eig = max(eig, abs(lam - ref.pop(j)) / scale)
        # SciPy's own factor reaches the same residual level; T itself may differ
        Ts, Zs = sla.schur(A, output="real")
        assert np.max(np.abs(A - Zs @ Ts @ Zs.T)) / scale < 1e-12
    return {"residual": res, "orthogonality": orth, "eigenvalues": eig}


def _care_rel(A, B, Q, R, X):
    G = B @ np.linalg.solve(R, B.T)
    r = A.T @ X + X @ A - X @ G @ X + Q
    m = lambda M: np.max(np.abs(M))
    return m(r) / (2 * m(A) * m(X) + m(X) ** 2 * m(G) + m(Q))


def check_solve_care(calls, outs):
    res = ref = 0.0
    for (A, B, Q, R), X in zip(calls, outs):
        res = max(res, _care_rel(A, B, Q, R, X))
        Xs = sla.solve_continuous_are(A, B, Q, R)
        ref = max(ref, np.max(np.abs(X - Xs)) / np.max(np.abs(Xs)))
    return {"residual": res, "vs_reference": ref}


def check_solve_lyapunov(calls, outs):
    res = ref = 0.0
    for (A, Q), X in zip(calls, outs):
        m = lambda M: np.max(np.abs(M))
        res = max(res, m(A @ X + X @ A.T - Q) / (2 * m(A) * m(X) + m(Q)))
        Xs = sla.solve_continuous_lyapunov(A, Q)
        ref = max(ref, m(X - Xs) / m(Xs))
    return {"residual": res, "vs_reference": ref}


def check_lqr(calls, outs):
    worst = 0.0
    for (A, B, Q, R), (K, P) in zip(calls, outs):
        Ps = sla.solve_continuous_are(A, B, Q, R)
        Ks = sla.solve(R, B.T @ Ps)
        worst = max(worst, np.max(np.abs(K - Ks)) / np.max(np.abs(Ks)))
        assert np.max(np.linalg.eigvals(A - B @ K).real) < 0, "closed loop is not stable"
    return {"rel": worst}


CHECKS = {k: globals()["check_" + k] for k in KERNELS}

# Inputs each kernel must refuse with an exception, not a wrong answer.
BAD_INPUTS = {
    "boys": [(17, 1.0), (3, -0.5), (-1, 1.0)],
    "enorm": [(np.ones((2, 2)),)],
    "schur": [(np.ones((2, 3)),)],
    "solve_care": [(np.eye(2), np.ones((3, 1)), np.eye(2), np.eye(1)),
                   (np.eye(2), np.ones((2, 1)), np.eye(2), np.eye(1))],    # A = I has no stabilizing X
    "solve_lyapunov": [(np.eye(2), np.eye(3))],
    "lqr": [(np.ones((2, 3)), np.ones((2, 1)), np.eye(2), np.eye(1))],
}


def evaluate(backend, cases=None) -> list[tuple]:
    """Rows of (kernel, metric, worst error, tolerance, passed, ncases)."""
    cases = cases or fixed_cases()
    rows = []
    for kernel, calls in cases.items():
        outs = backend.batch(kernel, calls)
        tol = KERNELS[kernel]["tolerance"]
        for metric, err in CHECKS[kernel](calls, outs).items():
            rows.append((kernel, metric, float(err), tol[metric], bool(err <= tol[metric]), len(calls)))
    return rows


def check_refusals(backend) -> list[str]:
    # the exported module raises VivError (VivStop for an adapter's `stop`)
    raised = (K1Error, ValueError, TypeError) + ((backend.mod.VivError,) if isinstance(backend, ExportK1) else ())
    problems = []
    for kernel, calls in BAD_INPUTS.items():
        for args in calls:
            try:
                getattr(backend, kernel)(*args)
                problems.append(f"{kernel}{tuple(np.shape(a) for a in args)} returned instead of raising")
            except raised:
                pass
    return problems


# ---------------------------------------------------------------------------
# The case programs in cases/
# ---------------------------------------------------------------------------

def case_text(kernel: str) -> str:
    return render(kernel, fixed_cases()[kernel])


def _json_arg(value, vtype: str):
    if vtype == "Complex":
        z = complex(value)
        return [z.real, z.imag]
    if vtype == "Arr":
        return np.asarray(value, dtype=float).tolist()
    if vtype == "Int":
        return int(value)
    return float(value)


def cases_json() -> str:
    """The fixed cases as `viv export` copies them beside the library: each
    kernel's calls, a complex number as [re, im], an array as nested lists."""
    import json
    out = {}
    for kernel, calls in fixed_cases().items():
        specs = KERNELS[kernel]["args"]
        out[kernel] = [[_json_arg(a, s["type"]) for a, s in zip(args, specs)] for args in calls]
    return json.dumps(out, indent=1) + "\n"


def write_cases():
    CASES.mkdir(exist_ok=True)
    for kernel in KERNELS:
        (CASES / f"case_{kernel}.viv").write_text(case_text(kernel))
    CASES_JSON.write_text(cases_json())


def run_cases(engine: str):
    """Run cases/*.viv in the named engine; the parsed results per kernel and the raw output."""
    b = NativeK1(engine)
    got, raw = {}, {}
    for kernel, calls in fixed_cases().items():
        out = b.run_file(CASES / f"case_{kernel}.viv")
        raw[kernel] = out
        got[kernel] = parse(kernel, out, len(calls))
    return got, raw


# ---------------------------------------------------------------------------
# pytest
# ---------------------------------------------------------------------------

_ROWS = None


def _rows():
    global _ROWS
    if _ROWS is None:
        _ROWS = evaluate(load_backend())
    return _ROWS


def _make_test(kernel):
    def test():
        bad = [r for r in _rows() if r[0] == kernel and not r[4]]
        assert not bad, bad
    test.__name__ = f"test_{kernel}"
    return test


for _k in KERNELS:
    globals()[f"test_{_k}"] = _make_test(_k)


def test_bad_input_raises():
    assert check_refusals(load_backend()) == []


def test_case_programs_are_current():
    for kernel in KERNELS:
        assert (CASES / f"case_{kernel}.viv").read_text() == case_text(kernel), \
            f"cases/case_{kernel}.viv is stale: python test_k1.py --write-cases"
    assert CASES_JSON.read_text() == cases_json(), "k1.cases.json is stale: python test_k1.py --write-cases"


def recorded_hashes() -> dict:
    import json
    return json.loads(HASHES.read_text()) if HASHES.exists() else {}


def check_hashes(backend) -> list[tuple]:
    """With the export backend: each kernel's results hash against the
    recorded one. Rows of (kernel, hash, matches)."""
    if not isinstance(backend, ExportK1):
        return []
    want = recorded_hashes()
    rows = []
    for kernel, calls in fixed_cases().items():
        h = backend.mod.results_hash(kernel, backend.batch(kernel, calls))
        rows.append((kernel, h, want.get(kernel) == h))
    return rows


def test_exact_results_match_the_recorded_hashes():
    rows = check_hashes(load_backend())
    assert all(ok for _, _, ok in rows), [r for r in rows if not r[2]]


def record(export) -> int:
    """The library's results against the interpreter's, bit for bit (both
    hashed with k1.results_hash); the hashes go to k1.exact_hashes.json."""
    import json
    interp_out, _ = run_cases("interp")
    hashes, ok = {}, True
    for kernel, calls in fixed_cases().items():
        h_lib = export.mod.results_hash(kernel, export.batch(kernel, calls))
        h_int = export.mod.results_hash(kernel, interp_out[kernel])
        same = h_lib == h_int
        ok &= same
        hashes[kernel] = h_lib
        print(f"{kernel:16} {len(calls):3d} calls  sha256 {h_lib[:16]}  interpreter {'same' if same else 'DIFFERENT ' + h_int[:16]}")
    if ok:
        HASHES.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
        print("wrote", HASHES.relative_to(REPO))
    return 0 if ok else 1


try:
    from hypothesis import given, settings, strategies as st

    _finite = st.floats(min_value=-1e300, max_value=1e300, allow_nan=False, allow_infinity=False)

    @settings(max_examples=50, deadline=None)
    @given(st.integers(0, 16), st.floats(0.0, 200.0))
    def test_boys_random(n, t):
        rows = CHECKS["boys"]([(n, t)], [load_backend().boys(n, t)])
        assert rows["rel"] <= KERNELS["boys"]["tolerance"]["rel"]

    @settings(max_examples=50, deadline=None)
    @given(st.floats(-6.0, 6.0), st.floats(-1.0, 4.0))
    def test_faddeeva_random(x, y):
        z = complex(x, y)
        rows = CHECKS["faddeeva_w"]([(z,)], [load_backend().faddeeva_w(z)])
        assert rows["rel"] <= KERNELS["faddeeva_w"]["tolerance"]["rel"]

    @settings(max_examples=50, deadline=None)
    @given(st.lists(_finite, min_size=1, max_size=100))
    def test_enorm_random(xs):
        x = np.array(xs)
        rows = CHECKS["enorm"]([(x,)], [load_backend().enorm(x)])
        assert rows["rel"] <= KERNELS["enorm"]["tolerance"]["rel"]

    @settings(max_examples=25, deadline=None)
    @given(st.integers(1, 10), st.integers(0, 2**32 - 1))
    def test_schur_random(n, seed):
        A = np.random.default_rng(seed).standard_normal((n, n))
        rows = CHECKS["schur"]([(A,)], [load_backend().schur(A)])
        tol = KERNELS["schur"]["tolerance"]
        assert all(rows[k] <= tol[k] for k in rows)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engine", choices=["interp", "c"], default="interp")
    ap.add_argument("--write-cases", action="store_true")
    ap.add_argument("--record", action="store_true", help="the export against the interpreter; write the hashes")
    args = ap.parse_args()
    if args.write_cases:
        write_cases()
        print(f"wrote {len(KERNELS)} programs to {CASES.relative_to(REPO)}")
        return 0

    backend = load_backend(args.engine)
    print(f"backend: {backend.name}")
    if args.record:
        if not isinstance(backend, ExportK1):
            print("--record needs the exported module (K1_DIST)")
            return 1
        return record(backend)
    if isinstance(backend, NativeK1):
        stale = [k for k in KERNELS if (CASES / f"case_{k}.viv").read_text() != case_text(k)]
        if stale:
            print("stale case programs:", ", ".join(stale), "(python test_k1.py --write-cases)")
            return 1
        got, raw = run_cases(args.engine)
        cases = fixed_cases()
        rows = []
        for kernel, calls in cases.items():
            tol = KERNELS[kernel]["tolerance"]
            for metric, err in CHECKS[kernel](calls, got[kernel]).items():
                rows.append((kernel, metric, float(err), tol[metric], bool(err <= tol[metric]), len(calls)))
        if args.engine == "c":
            _, raw_i = run_cases("interp")
            same = [k for k in KERNELS if raw[k] == raw_i[k]]
            print(f"C output byte-equal to the interpreter's: {len(same)} of {len(KERNELS)} kernels"
                  + ("" if len(same) == len(KERNELS) else f" (differ: {sorted(set(KERNELS) - set(same))})"))
    else:
        rows = evaluate(backend)

    print(f"{'kernel':16} {'metric':14} {'cases':>5} {'max error':>10} {'tolerance':>10}  pass")
    for kernel, metric, err, tol, ok, n in rows:
        print(f"{kernel:16} {metric:14} {n:5d} {err:10.2e} {tol:10.1e}  {'yes' if ok else 'NO'}")
    refusals = check_refusals(backend)
    nbad = sum(len(v) for v in BAD_INPUTS.values())
    print(f"bad inputs raising an exception: {nbad - len(refusals)} of {nbad}")
    for p in refusals:
        print("  ", p)
    hashes = check_hashes(backend)
    for kernel, h, ok in hashes:
        print(f"{kernel:16} sha256 {h[:16]}  {'the recorded bits' if ok else 'NOT the recorded bits'}")
    return 0 if all(r[4] for r in rows) and not refusals and all(ok for _, _, ok in hashes) else 1


if __name__ == "__main__":
    sys.exit(main())
