# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vipyr: Viv as a Python package. `pip install vipyr`, then `import vipyr as vp`.

Viv is the language (.viv files, the `viv` command); vipyr is how Python
calls it. The wheel carries the `viv` compiler, so nothing else is installed.

Your own Viv, called from Python:

    import vipyr as vp
    k = vp.load("kernels.viv")          # compiled once, cached
    k.sumsq(np.arange(4.0))

    import vipyr.auto                   # or vp.install()
    import kernels                      # finds kernels.viv beside your .py files

Each top-level def of the file whose parameters are annotated (Num, Int,
Bool, Complex, Arr, [Int]) becomes a Python function; a file with a
`<name>.export.toml` beside it is exported by that manifest instead. A
compile error is a SyntaxError located in the .viv file; a run-time error
is the module's VivError (VivIndexError is also an IndexError, and so on).

A function also takes a pandas or polars DataFrame: columns go to parameters
and record fields by name (`columns=` renames), and a result of the input's
length comes back as a frame of the same library (vipyr.frames).
`vp.check_point_in_time(fn, df, by=, ts=)` checks that a feature function
reads no row later than the row it is computed for (vipyr.pit);
`vp.check_parity(fn, df, by=, ts=)` replays serving request by request and
checks the features equal the training batch's bit for bit (vipyr.parity).
`vp.compile` turns a fitted scikit-learn pipeline into one C99 file.

A Python function, translated to Viv by the strict translator py2viv
(exact or refused) and compiled: `fast = vp.translate(fn)` or `@vp.jit`;
the first call is checked against the Python function, `fast.viv_source`
is the Viv (vipyr.translate).

In IPython or Jupyter, `%load_ext vipyr` adds the `%%viv NAME` cell magic,
which compiles a cell of Viv defs into the module NAME and records each call
in `.viv_runs.jsonl` (vipyr.magic).

Viv's scientific programs, called with keyword arguments:

    r = vp.scf.run(atoms=[8, 1, 1], coords=xyz, basis="cc-pvdz")
    r.e_tot, r.mo_energy, r.converged       # hartree
    r.check()                               # the program's own tests and recorded bits
    r.run_id                                # the run record's content address

Each one is a Viv program (or one kernel of a collection), compiled once
through the same cache. Its keyword arguments, their ranges and its results
come from the program's `<name>.export.toml` manifest and its `#:` lines;
vipyr/_registry.py adds only the Python names, the fields and the examples.
`vp.functions()` lists them all; https://viv-lang.org/vipyr is the same
reference as web pages.

The core distribution carries special, linalg, control, ode, optimize,
finance, features, statmech, randla and power. The domain families are distributions of their own, found
through the `vipyr.packages` entry points (vipyr._packages): scf and dft
(`pip install vipyr[chem]`), nuclear (`vipyr[nuclear]`), plasma
(`vipyr[plasma]`), xc (`vipyr[xc]`, MPL-2.0), and two that list programs
still to come, symreg and hardware. In a checkout of the Viv repository they
are found in packages/ without an install.
"""
import importlib as _importlib

from ._core import (VivBuildError, VivCompileError, build, compiler, compiler_identity, load,
                    viv_root, write_stub)
from ._core import stats as cache_stats  # the compile cache's counters (vp.stats is the family)
from ._hook import install, uninstall
from .parity import ParityError, check_parity
from .pit import PointInTimeError, check_point_in_time
from ._runtime import CheckReport, ParameterError, Result, record_runs
from .translate import VivTranslationError, VivTranslationMismatch, jit, translate
from .digits import DigitsReport, digits
from .atlas import atlas

__version__ = "0.0.1"

# the core distribution's families; the others come from vipyr._packages
FAMILIES = ["special", "linalg", "control", "ode", "optimize", "gpt", "gguf", "stats", "finance", "features", "statmech", "randla", "power"]

# submodules reached as attributes without an import of their own
_LAZY = FAMILIES + ["compile", "frames", "magic", "docs"]


def families():
    """Every family this installation can reach: the core's and those of the
    distributions found (installed, or in the checkout)."""
    from ._packages import registered
    return FAMILIES + sorted(f for f in registered() if f not in FAMILIES)


def functions(kind=None):
    """{qualified name: spec} of every function vipyr can reach, its
    distributions' included; `kind="coming"` for the listed, unbuilt ones,
    `kind="built"` for the rest."""
    fs = {}
    for fam in families():
        fs.update(getattr(__getattr__(fam), "__vipyr_functions__", {}))
    if kind == "built":
        return {k: v for k, v in fs.items() if v["kind"] != "coming"}
    if kind:
        return {k: v for k, v in fs.items() if v["kind"] == kind}
    return dict(fs)


def load_ipython_extension(ip):
    """`%load_ext vipyr`: the %%viv, %viv_export and %viv_runs magics."""
    from .magic import load_ipython_extension as load
    load(ip)


def __getattr__(name):
    if name in _LAZY:
        return _importlib.import_module("." + name, __name__)
    if not name.startswith("_"):
        from . import _packages
        target = _packages.registered().get(name)
        if target is not None:
            mod = _importlib.import_module(target)
            globals()[name] = mod
            return mod
        msg = _packages.missing_message(name)
        if msg:
            raise ImportError(msg, name=name)
    raise AttributeError("module 'vipyr' has no attribute %r" % name)


def __dir__():
    return sorted(set(list(globals()) + _LAZY + families()))


__all__ = ["load", "install", "uninstall", "write_stub", "build", "compiler", "compiler_identity", "viv_root",
           "VivCompileError", "VivBuildError", "cache_stats", "__version__",
           "check_point_in_time", "PointInTimeError", "check_parity", "ParityError",
           "translate", "jit", "VivTranslationError", "VivTranslationMismatch",
           "functions", "families", "record_runs", "Result", "CheckReport", "ParameterError", "FAMILIES",
           "digits", "DigitsReport", "atlas"]
