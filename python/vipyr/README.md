<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../../www/public/brand/vipyr/vipyr-logo-dark.svg">
    <img src="../../www/public/brand/vipyr/vipyr-logo-light.svg" alt="vipyr" width="286">
  </picture>
</p>

# vipyr

vipyr is Viv as a Python package: `import vipyr as vp`, like
`import numpy as np`. Viv is the language (`.viv` files, the `viv` command,
viv-lang.org); vipyr is the package, and its platform wheel carries the `viv`
compiler and installs the `viv` command.

vipyr is not on PyPI yet, so `pip install vipyr` does not install it. Use it
from a checkout. After `cargo build --release -p viv` at the repository root,
either put the sources on the path with no install,

```sh
PYTHONPATH=python python3 -c "import vipyr as vp"
```

or build and install the wheel into an environment that already has NumPy:

```sh
pip install --no-build-isolation ./python                   # the core, with the compiler
pip install --no-build-isolation ./packages/vipyr-chem      # an extra: vp.scf, vp.dft
```

The `pip install vipyr[...]` lines below are the names the distributions will
have on PyPI.

It does two things.

**Your own Viv as Python functions.**

```python
import vipyr as vp
k = vp.load("kernels.viv")          # compiled to C once, cached under __pycache__/vipyr/
k.sumsq(np.arange(4.0))

import vipyr.auto                   # or vp.install(): the import hook
import kernels                      # finds kernels.viv beside your .py files
```

`%load_ext vipyr` adds the `%%viv NAME` cell magic in Jupyter;
`vp.check_point_in_time` and `vp.check_parity` check feature functions on a
DataFrame; `vipyr compile model.pkl -o model.c` writes a fitted scikit-learn
pipeline as one C99 file; `build-backend = "vipyr.build"` with
`[tool.vipyr] modules = [...]` builds a wheel of your own `.viv` modules.

**Viv's scientific programs, called with keyword arguments.**

```python

r = vp.scf.run(atoms=[8, 1, 1], coords=xyz, basis="cc-pvdz")   # coords in angstrom
r.e_tot, r.mo_energy, r.converged       # hartree
r.check()                               # the program's own tests, and its recorded bits
r.run_id                                # the content address of the run record

vp.dft.pw(material="si").gap         # eV
vp.nuclear.skyrme_hf(spacing=0.05).r_ch
vp.plasma.landau(setup="k05").gamma
vp.nuclear.shell_model(interaction="usdb", z=2, n=2).energies     # MeV
vp.statmech.ising(temperature=2.0).mag_abs
T, Z = vp.linalg.schur(A=A)
r = vp.ode.solve("robertson", (0, 1e5), [1, 0, 0], method="radau", params=[0.04, 1e4, 3e7])
popt, pcov = vp.optimize.curve_fit(model, x, y, p0)                  # model: a Python function py2viv translates

import vipyr as vp                      # the same package under the alias the newer docs use
vp.stats.t.ppf(0.975, 10)               # SciPy's distributions, names and parameters
vp.stats.ols(y, X).summary()            # statsmodels' table; NIST's certified regressions held
vp.stats.glm(y, X, family="binomial", link="probit")   # R's glm.fit, step for step
```

The reference, one page per function, is https://viv-lang.org/vipyr.

## What it is

Each function is a Viv program (or one kernel of a collection) compiled to
a C library by `viv export` and loaded through `vipyr`'s cache: the first
call in a process compiles it or loads the library built before, and later
calls reuse it. A call

- checks its keyword arguments against the program's manifest before
  anything is compiled or run, and raises `vp.ParameterError` (a
  `ValueError` whose `.param` names the argument) or a `TypeError` that
  names it;
- returns a frozen dataclass: the program's results as fields (each field's
  doc gives its units), plus `params`, `tests` (the program's test blocks
  for this run), `log`, `run_id` and `record`;
- `result.check()` returns a `CheckReport`: the test blocks, and for a run at
  a deck's settings (or a kernel's fixed cases) whether the result hash is
  the one recorded when the library, a native build and Viv's interpreter
  agreed bit for bit.

The run record is vipyr's (`vipyr/magic.py`, `viv.cell-run/1`):
`vp.record_runs("runs.jsonl")` or `$VIPYR_RUNS` appends one JSON line
per call, and a rerun with the same inputs whose output differs is reported.

## Where the API comes from

One table, `_registry.py`, lists the functions: the Python name, the field,
the manifest or source file, renamed keywords (`kmesh` for the program's
`NKMESH`) and the example. Everything else is read from the program:

| what | from |
|---|---|
| parameters, types, ranges, enum choices, shapes, groups | the manifest, `<name>.export.toml` |
| each parameter's help and default (per deck) | the `#:` lines, through the docstring `viv export` writes |
| the summary | the program's intent lens or `#|` block, the same way |
| the results and their meanings | the manifest and the `#:` lines |
| what it is checked against | the manifest's `[reference]` or `[kernels.<name>]` table |
| the recorded hashes | `<name>.exact_hashes.json` beside the manifest |

`python -m vipyr.docs` derives all of it into `spec.json` (signatures and
docstrings at import) and `www/src/data/vipyr.json` (the /vipyr pages);
`python -m vipyr.docs --check` and python/tests/test_vipyr_programs.py fail when
either is stale.

Families: `scf`, `dft`, `nuclear`, `plasma`, `special`, `linalg`,
`control`, `ode`, `optimize`, `finance`, `features`, `statmech`, `randla`. Listed and not built yet (calling one
raises `NotImplementedError`): `symreg.sisso`, `symreg.ai_feynman`,
`hardware.riscv`.

## Distributions

The core distribution, vipyr, holds the compiler, the tools, and `special`,
`linalg`, `control`, `ode`, `optimize`, `randla`, `finance`, `features`, `statmech`. Each domain is a
distribution of its own under `packages/`, installed as an extra:

| distribution | install | families | license |
|---|---|---|---|
| vipyr-chem | `pip install vipyr[chem]` | `scf`, `dft` | MIT OR Apache-2.0 |
| vipyr-nuclear | `pip install vipyr[nuclear]` | `nuclear` | MIT OR Apache-2.0 (BIGSTICK's MIT notice in NOTICE) |
| vipyr-plasma | `pip install vipyr[plasma]` | `plasma` | MIT OR Apache-2.0 |
| vipyr-xc | `pip install vipyr[xc]` | `xc` | MPL-2.0 (Libxc's, xc/LICENSE) |
| vipyr-symreg, vipyr-hardware | placeholders | `symreg`, `hardware` | MIT OR Apache-2.0 |

From a checkout, `pip install --no-build-isolation ./packages/<distribution>`
installs one of them, and with no install at all they are on the path already
(below). Once vipyr is on PyPI, `pip install vipyr[all]` will install the four
extras. A distribution registers its families in the entry-point group `vipyr.packages`, and `vp.scf` imports
`vipyr_chem.scf` on first use; without it, `vp.scf` raises
`ImportError: vp.scf needs vipyr-chem: pip install vipyr[chem]`.

**In a checkout nothing is installed.** `python/vipyr/_packages.py` reads
the `vipyr.packages` entry points of every `packages/*/pyproject.toml` and
`ports/*/pyproject.toml` (not `_template`) and puts those directories on
`sys.path`, so `PYTHONPATH=python python3 -c "import vipyr as vp; vp.scf"`
works from the repository root, and the tests and the site build use the
sources as they are. `VIPYR_DEV=0` turns this off, which makes the checkout
behave like a core-only install.

Ports of upstream codes go under `ports/<name>/` as `vipyr-<name>`, under
the upstream's license; `ports/_template/` lists the files each one needs.

## Writing a function's reference page

The page at viv-lang.org/vipyr/<family>/<name> and `help(vp.<family>.<name>)`
(numpydoc) are generated from the same spec. Parameters, types, ranges and
results come from the program's `<name>.export.toml` and its `#:` lines.
The prose comes from a file beside the manifest, `<name>.docs.toml`
(`viv export` reads the manifest strictly, so the prose lives apart):

```toml
summary = "one line"
description = """paragraphs, markdown"""
notes = """the method: equations ($...$, $$...$$), algorithm, cost, limits"""
see_also = ["dft.pw"]

[[references]]                 # numbered in order; upstream code first for a port
text = "Author, Title, Journal 1, 2 (2000)."
doi = "10.xxxx/yyyy"           # or url = "..."

[[examples]]                   # run by `python -m vipyr.docs`; the output is shown
caption = "what it shows"
code = '''import vipyr as vp ...'''

[params.coords]                # per parameter: doc, units, range
units = "angstrom"
range = "what happens at the edges"
```

A kernel or a def of a shared file names its file in `_registry.py`
(`docs=`) and keeps its entries under `[<entry>]`. Sections nobody has
written show "Not yet documented." `study/ports/pyscf/vivscf.docs.toml`
(`vp.scf.run`) is the worked example.

## Differential equations: `vp.ode`

`vp.ode.solve(rhs, t_span, y0, method=, rtol=, atol=, params=, t_eval=,
events=, direction=, terminal=, jac=)` is `solve_ivp` with the integrator
(lib/ode: Dormand-Prince 5(4), Radau IIA, BDF, auto) and the right-hand side
compiled together; `vp.ode.batch(rhs, t_span, y0, params, t_eval=)` runs
one solve per row of `params` in one call. The rhs is a Viv def
`f(t: Num, y: Arr, p: Arr) -> Arr`: a problem of lib/ode/problems.viv by
name, `"file.viv:def"`, a def of a module from `vp.load`, or a Python
function in py2viv's subset. vipyr/_ode.py writes a small driver per
problem (with an exact Jacobian from `grad`) and compiles it through the
cache. Its parameters reach it through `p`, never through captured Python
variables: a compiled def cannot close over Python state. The reference and
the comparison with SciPy are lib/ode/ode.docs.toml and
lib/ode/bench_scipy.json.

## Fitting and optimization: `vp.optimize`

`vp.optimize.least_squares(fun, x0, jac=, args=, ftol=, xtol=, gtol=,
x_scale=, max_nfev=)` is SciPy's `least_squares(method="lm")`,
`vp.optimize.curve_fit(f, xdata, ydata, p0, sigma=, absolute_sigma=)`
returns the parameters and their covariance (`popt, pcov = ...` unpacks as
SciPy's), and `vp.optimize.root(fun, x0)` is `root(method="lm")`. The
solver is lib/minpack's port of MINPACK's lmder; vipyr/_optimize.py writes
a driver per function (a Viv def, `"file.viv:def"`, or a Python function
py2viv translates) with an exact Jacobian from `grad`, or MINPACK's
forward differences with `jac="fd"`. No bounds (MINPACK has none). The
reference is lib/minpack/optimize.docs.toml; the comparison with NIST's
certified values and with SciPy is lib/minpack/bench_optimize.json.

## Requirements

NumPy. A platform wheel carries the `viv` compiler, every program of its
distribution precompiled (`<pkg>/_prebuilt`, vipyr/_prebuilt.py) and their
sources (`<pkg>/_root`), so it needs no C compiler; a user's own .viv is
compiled with $CC, or runs in Viv's interpreter when there is none
(vipyr/_interp.py). A checkout reads the sources from the repository
(`$VIV_ROOT` points elsewhere) and compiles on first use; so does an sdist.
tools/wheels/build_all.sh builds the wheels, .github/workflows/wheels.yml
the platform matrix, tools/wheels/smoke_nocc.py checks an install. pandas for
`vp.features`. Python 3.11, or 3.9 and 3.10 with `tomli`.

## Tests

    python -m pytest python/tests

Every function's example runs as written and returns its documented fields;
bad arguments are refused naming the parameter; the specs are current; the
site has a page per function. The first run compiles vivscf.c (5 MB of C,
about a minute).

## The name on PyPI

Nothing has been registered on PyPI: the owner reserves `vipyr` before a
release.
