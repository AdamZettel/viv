<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 The Viv authors -->
# AGENTS.md

The entry point for an agent working in this repository: what is here, where
it lives, how to build and check it, and the rules a change follows.

## What Viv and vipyr are

Viv is a small language for numerical programs. It compiles to C99 and to
WebAssembly, and every program runs in three engines (an interpreter,
compiled C and compiled wasm) that must print the same bytes. The compiler
is written in Rust on knotkit, a kit for content-addressed, multi-target
compilers, and also ships as wasm, so viv-lang.org runs it in the browser.

vipyr is Viv as a Python package. A user keeps their Python program and
moves a slow function into a `.viv` file beside it; vipyr compiles it on
first import. vipyr also ships ported, checked engines (Hartree-Fock against
PySCF, plane-wave DFT, statistics, control, ODEs) as Python functions.
Section "Writing Viv for Python callers" below is the working guide.

## Where everything is listed

`node tools/inventory.mjs ls` lists every asset: site pages (each dynamic
route expanded from its data), posts, claims, channels, gallery entries,
scenes, showcases, explorer programs, ladders, PDE solvers, engines, Python
and Viv pairs, docs topics, vipyr functions and families, code atlases, every
`.viv` file git knows, vipyr distributions (`packages/`),
ports, `viv` verbs, the language feature catalog, test
suites and every gate check (read from `gate.CHECKS`). Each record has a
kind, an id, a title, its source paths, its site address, the reference it
is checked against (when the data names one) and its status.

```sh
node tools/inventory.mjs kinds               # the kinds and their counts
node tools/inventory.mjs ls                  # everything, one line each
node tools/inventory.mjs ls program          # one kind
node tools/inventory.mjs ls --grep pyscf     # records that mention a word
node tools/inventory.mjs ls vipyr --json     # JSON lines, for a script
node tools/inventory.mjs                     # write inventory.json and INVENTORY.md
```

`ls` and `kinds` build the list from the checkout every time, so nothing
needs regenerating after a change. inventory.json (the records) and
INVENTORY.md (the same as tables) are build products: the site's `npm run
stage` writes them for /inventory, and they are not committed. Site-derived
kinds (posts, claims, gallery entries, vipyr functions) read
`www/src/data`, so they follow the last `npm run stage`. (A `viv ls` verb
may come later; for now the node script is the way.)

## The repository map

Top level (`node tools/inventory.mjs ls dir` prints this list from the same
descriptions):

| directory | what it holds |
|---|---|
| `lang/` | the languages on knotkit: `viv/` (the compiler), `skein/` (a teaching Scheme), `weft/` (a weak-form language that compiles to Viv), `oa/` (an operator-algebra probe) |
| `crates/knotkit/` | the compiler kit Viv is built on |
| `lib/` | Viv's library: the modules programs import |
| `python/` | vipyr, the Python package (`python/vipyr`), and its tests (`python/tests`) |
| `packages/` | the vipyr extras as their own distributions (chem, hardware, nuclear, plasma, symreg, xc) |
| `www/` | viv-lang.org, an Astro site; the compiler runs in the browser as wasm |
| `tools/` | the gate, merge, checks, generators and harnesses |
| `study/` | studies and measurements: comparisons with Python, port ledgers, memory checks |
| `posts/` | the posts: each a `.viv` program with a header, published when its tests pass |
| `channels/` | the syllabus of each posts channel |
| `claims/` | claims: a bounded statement, the program that checks it, and its vipyr twin |
| `scf/`, `dft/`, `xc/`, `plasma/`, `pde/`, `stat/` | application programs and their checks: Hartree-Fock against PySCF, plane-wave DFT, Libxc-derived functionals, Vlasov-Poisson, one program per PDE, Ising |
| `engines/` | programs with a declared interface that read input files and run as wasm |
| `ports/` | ports of published codes with their upstream license and comparison |
| `kernels/`, `catalog/` | kernel libraries exported for Python, and per-entity DataFrame features |
| `examples/`, `gallery/` | worked examples and the vipyr gallery scripts |
| `show/`, `ladders/` | the programs behind the showcase pages and the ladders |
| `bugs/` | the bug ledger and the test of each fixed bug |
| `bench/`, `tests/`, `docs/` | integral timings, C inputs for lib/cc, the tutorial |
| `server/mcp/` | the MCP endpoint, served at /mcp by a Netlify Function |
| `third_party/`, `LICENSES/` | vendored code (SQLite) and the license texts |
| `claude/` | the Claude Code plugin for vipyr |

The large ones, one level down:

- `lang/viv/`: `src/` (lexer, parser, types, `interp/`, `lower/` and the C
  and wasm emission, the `viv` command in `main.rs` and `agent.rs`),
  `std/` (the prelude), `features/` (the catalog `viv features` prints,
  each entry with an example the tests run), `tests/` (cargo tests),
  `examples/`, `schemas/` (the `--json` envelopes), `agent/` (the guide
  `viv help --agent` prints).
- `lib/`: `hf/`, `hfref/`, `scf/`, `basis/` (Hartree-Fock and its reference
  tables), `skyrme/`, `nuclear/` (nuclear structure), `minpack/`, `ode/`,
  `eigen/`, `randla/`, `stats/`, `power/`, `cas/`, `gpt/`, `gguf/`, `cc/`
  and `rv32/` (a C compiler to RV32IM and its simulator), `algo/`,
  `nprandom/` (NumPy 1.24.4's random draws), plus single-file modules
  (`lqr.viv`, `schur.viv`, `lobpcg.viv`, ...).
- `www/`: `src/pages/` (routes), `src/components/`, `src/layouts/`,
  `src/lib/`, `src/data/` (generated by the staging; do not edit by hand),
  `src/styles/`, `scripts/` (the `stage` steps and the `check` steps),
  `public/` (the wasm compiler, workers, staged programs).
- `tools/`: `gate/` (gate.py holds every check), `merge.sh`, `py2viv/` (the
  strict Python to Viv translator), `fuzz/` (vivsmith), `memcheck/`,
  `claims/`, `jupyter/` (a Viv kernel), `licenses/`, `wheels/`, and
  the per-feature case generators (`arrays/`, `ints/`, `strings/`, ...).
- `python/`: `vipyr/` (load, the import hook, frames, magics, translate,
  compile, catalog, atlas) and `tests/`.
- `study/`: `benchmarks/viv/` (the benchmark programs the site serves),
  `compare/` (the pairs behind /compare), `ports/` (port ledgers),
  `memcheck/`, `release/`.

## Build and test

```sh
cargo build --release -p viv          # the compiler: target/release/viv
target/release/viv run prog.viv       # run a program and its test blocks
target/release/viv check prog.viv     # parse and type-check only
tools/three-engines.sh <dir>          # interpreter, C and wasm must print the same bytes
sh www/build-wasm.sh                  # the browser compiler: www/public/viv/viv.wasm
python3 -m pytest -q python/tests/test_vipyr.py   # one vipyr suite
```

The gate (`tools/gate.sh`, checks in `tools/gate/gate.py`) is the fixed list
of checks a change must pass; CONTRIBUTING.md describes the tiers:

```sh
tools/gate.sh quick                   # language checks, before every commit you report
tools/gate.sh quick --only <check>    # one check while you iterate
tools/gate.sh premerge                # what a merge runs
tools/gate.sh <tier> --list           # what a tier runs
```

Branches reach `main` through `tools/merge.sh <branch>`, never `git merge`:
it merges in a separate worktree, runs the premerge tier on the merged tree
and moves `main` only when every check passed.

The site:

```sh
cd www
npm run stage                         # regenerate src/data, public/ and the inventory
npm run check                         # the site's checks
PUBLIC_SITE=1 npx astro build && node scripts/csp.mjs   # the public build and its header check
npm run aispeak                       # the prose lint
```

A new inline script changes the Content-Security-Policy hashes:
`node scripts/csp.mjs --write` updates netlify.toml, and a new top-level
route needs its own X-Frame-Options rule there.

## House rules

- **Stage named paths.** Never `git add -A` or `git add .`: worktrees carry
  a `www/node_modules` symlink and staged build products that must not be
  committed. `git add path/one path/two`.
- **Work in a worktree** for a branch (`git worktree add -b <branch>
  ../viv-wt/<branch>`), and symlink `www/node_modules` to the main
  checkout's rather than installing again.
- **The gate decides.** A failed check is fixed at its cause; do not drop a
  check or narrow what it covers. A shared machine runs one heavy build or
  gate at a time: run the targeted check (`--only`), not the full tier.
- **No flaky tests.** A check gates only on deterministic results; timings
  and network calls are reported, never pass/fail. Randomized checks fix and
  print their seed. A check that fails without a code change is quarantined
  and fixed, not retried until it passes.
- **Three engines agree.** A difference between the interpreter, C and wasm
  is a bug in one of them, never a tolerance to widen.
- **Every fixed bug has a test** in `bugs/ledger.jsonl`, added when the bug
  is found.
- **A new feature has a catalog entry** in `lang/viv/features/` with an
  example; a change to the language starts with a written proposal.
- **Code style** is the house style in CONTRIBUTING.md: names as the problem
  uses them, a comprehension when a loop builds one value, defs under 40 code
  lines.
- **Prose is plain.** Say what a thing does and give the number with its
  source. American English. No colons or dashes in titles and headings, no
  antithesis ("not X but Y"), no boasting about speed. `npm run aispeak`
  in `www/` checks the site's text and posts, and the
  deploy runs it with `--strict`.
- **Provenance.** Text the owner wrote is marked: on the site with
  `data-by="user"`, in a post by having no `agent:` line in its header.
  The lint skips it and an agent does not rewrite it; an agent's own text
  carries `agent: claude` (posts) and is linted.
- **No local paths in served artifacts.** Nothing under `www/dist`, the
  wasm modules, or committed result files may contain a home directory, a
  user name or `Documents/`; paths are repository-relative. Check with
  `strings www/public/viv/viv.wasm | grep -c Documents` (0).
- **Generated files** (`www/src/data/*`, and the uncommitted
  `inventory.json` and `INVENTORY.md`) are rewritten by their generator,
  not by hand.
- **Numbers cite their result file** and say whether the run was timed
  under load.

## Tools for reading a program

- `viv help --agent`: the working guide for an agent, with every verb.
- `viv features [--search words]`: what the language has; look before
  writing.
- `viv explain`, `viv map`, `viv lsfunc`, `viv whatis`, `viv src`: a
  program's intent, its code map, its table of contents, one name, one def.
- `viv atlas <file.viv> [--text]`: a whole program's call graph, by module,
  with the calls one run made (an HTML page, or text); in vipyr it is
  `vp.atlas`.
- The MCP endpoint (server/mcp/README.md) serves read-only tools at
  viv-lang.org/mcp from the site's Netlify Function.

## GPU kernels

`viv cuda <file.viv>` writes CUDA C from a Viv kernel (examples in
`examples/cuda/`); the emulator in the interpreter runs the same kernels on
the CPU, and tests/cuda.rs checks what `viv cuda` writes.

## Writing Viv for Python callers

The examples here import vipyr under its short name, `vp` (as in `vp.atlas`).

Agents write Viv; Python calls it. vipyr is Viv as a Python package
(`import vipyr as vp`; source in python/vipyr; not on PyPI yet, so from a
checkout use `PYTHONPATH=python` or `pip install --no-build-isolation ./python`). A
user keeps their Python program and moves a slow function into a `.viv` file
beside it; vipyr compiles that file to a shared library on first import and
caches it.

```python
import vipyr.auto            # installs the import hook
import kernels               # compiles kernels.viv beside this file, once
kernels.sumsq(x)             # x: a NumPy array
```

`vp.load("kernels.viv")` does the same without the hook.

### The workflow

0. `vipyr help --agent` prints the working guide for an agent, led by
   `vipyr catalog`: the validated engines, each with its reference, claims
   and the timings measured beside plain Python where they exist. Every
   command takes `--json` (`viv help json` describes the object it prints),
   and `viv skill install` writes the Claude Code skill (claude/ holds the
   plugin).
1. Look up what Viv has before writing: `viv features`,
   `viv features --search <words>`, `viv features <name>`. (The wheel
   installs the `viv` command; `vipyr viv ...` is the same, the bundled
   compiler with Viv's library on its path.)
2. Write `kernels.viv`: one top-level `def` per function Python should call,
   with every parameter annotated, and `test` blocks that pin the results.
3. `viv check kernels.viv` (front end: parse and types), then
   `viv run kernels.viv` (runs the test blocks, prints PASS or FAIL).
4. Call it from Python. `vipyr stubs kernels.viv` writes `kernels.pyi` so
   editors and type checkers see the signatures.
5. For a library, build the wheel with the backend in `vipyr.build` (below):
   installers then need neither Viv nor a C compiler.

### What gets exported

Each top-level def whose parameters are all annotated with one of these:

| Viv | Python argument | Python result |
|---|---|---|
| `Num` | `float` (an `int` is converted) | `float` |
| `Int` | `int` (not `bool`) | `int` |
| `Bool` | `bool` | `bool` |
| `Complex` (or `Cplx`) | `complex` | `complex` |
| `Arr` | anything `np.asarray` takes, as float64 | `ndarray` of float64 |
| `[Int]` | anything `np.asarray` takes, as int64 | `ndarray` of int64 |

The result type comes from the compiled def: annotate it (`-> Num`) or not.
`return a, b` gives a Python tuple. A def is left out, with a note, when a
parameter has no annotation or a type that cannot cross (`Str`, `CArr`, a
record, a map, an optional), when it returns nothing, or when its name starts
with `_` (private). `__all__` lists the exported names; the notes are in
`module.__vipyr__.notes` and in the output of `vipyr build kernels.viv`.

The `#|` comment lines above a def (and `#||`) become its Python docstring and
the stub's.

For range checks, ranks and documented references, write a manifest
`kernels.export.toml` beside the file (kernels/k1/k1.export.toml is the
example). With a manifest, only the kernels
it lists are exported, and their arguments are checked against it.

### Errors

| what happened | Python exception |
|---|---|
| parse or type error in the .viv file (or a module it imports) | `vipyr.VivCompileError`, a `SyntaxError` with the .viv file, line and column |
| no def can be exported | `vipyr.VivCompileError` |
| no C compiler | the module runs in Viv's interpreter (same functions and results, slower; a one-time notice); a program manifest raises `vipyr.VivBuildError` (`VIPYR_INTERP=0`: always raise) |
| the C does not build | `vipyr.VivBuildError`, an `ImportError` |
| an argument of the wrong type | `TypeError`, before any work |
| a non-integral float for an `Int`, an array of rank above 8 | `ValueError`, before any work |
| index out of range | `module.VivIndexError`, also an `IndexError` |
| a bad value, such as arrays of mismatched shapes | `module.VivValueError`, also a `ValueError` |
| overflow, division by zero, recursion, memory | `VivOverflowError`, `VivZeroDivisionError`, `VivRecursionError`, `VivMemoryError` |
| `stop("...")` in the def | `module.VivStop` |
| a call while another is running in the process | `RuntimeError` |

Every run-time error is a subclass of `module.VivError`, with Viv's message
and the source line. The next call works after an error.

### Python first: `vp.translate`

`vp.translate(fn)` (or `@vp.jit`) runs a Python function through py2viv, the
strict translator (exact or refused, tools/py2viv/SUBSET.md), compiles it and
checks the first call against `fn`. `.viv_source` is the Viv: a way in for a
reader who knows the Python (python/vipyr/translate.py).

### Exact and fast

`vipyr.load(path)` and the import hook build with `-O2 -ffp-contract=off`:
the library's results are the interpreter's, bit for bit, on every platform.
`vipyr.load(path, fast=True)` (or `vipyr.install(fast=True)`) builds with
`-O3 -ffp-contract=fast`, which may change the last bits. Each build has its
own cache entry.

### The cache

Under `__pycache__/vipyr/` beside the source, or `~/.cache/vipyr` when
that is not writable (`$VIPYR_CACHE` overrides). An entry is keyed by the
SHA-256 of the source, the compiler binary, `$CC` and the flags, and stays
valid while every module the program imported hashes the same and no module
appears where an import looked first. Delete the directory to clear it.

### Pitfalls

- A literal `0` is a `Num`. Write `t: Int = 0` for an Int accumulator.
- Counted loops are `for i to n` (0 to n-1), not `0..n` or `range(n)`.
- `append` returns the grown array: `out = append(out, x)`.
- Arrays are values. A def that assigns into its `Arr` argument changes its
  own copy; the caller's NumPy array is never written. Return the result.
- The file name is the module name: `my_kernels.viv`, not `my-kernels.viv`.
- A `kernels.py` beside `kernels.viv` wins the import; the hook only finds
  names nothing else provides.
- Two modules of one name (`./minpack/enorm.viv` beside the file and the
  library's `lib/minpack/enorm.viv`) are refused; rename one.
- The names `build`, `version`, `tests`, `results_hash` and `normalize`
  belong to the generated module; a def of one of them is left out.
- One call at a time per process; a call runs in the calling thread.
- Test blocks run under `viv run`, not when Python imports the module.

### Libraries: shipping compiled kernels

```toml
[build-system]
requires = ["vipyr"]
build-backend = "vipyr.build"

[project]
name = "mypkg"
version = "1.0"
dependencies = ["numpy"]

[tool.vipyr]
packages = ["mypkg"]
modules = ["mypkg/kernels.viv"]
```

`pip wheel .` gives a `py3-none-<platform>` wheel holding `mypkg/kernels.py`,
its shared library and `kernels.pyi`.

### Reference

- `viv features`: the language, by area, with runnable examples
- python/vipyr: the package; python/tests: what it promises, as tests
- https://viv-lang.org/reference and https://viv-lang.org/docs

### DataFrames (python/vipyr/frames.py, vipyr/pit.py)

A loaded module's function takes a pandas or polars DataFrame: columns go to
parameters and record fields by name (`columns={"param": "col"}` renames),
and a result of the input's length comes back as a frame of the same
library (pandas keeps the index; `as_frame=False` gives the numpy dict).
The conversion rules (zero copy for float64/int64/datetime64[ns], datetime
to UTC int64 ns, categorical/string to codes rebuilt on output, bool to 0/1,
nulls refused unless the def declares `<name>_valid: [Int]`) are in the
module docstring of frames.py, and each has a test in
python/tests/test_frames.py. `vipyr.check_point_in_time(fn, df, by=, ts=,
outcomes=)` is the leakage check: it alters rows after t and outcomes at t
and requires features at or before t to stay the same. The worked example
is examples/datasci/card_velocity (README: parity with a pandas
groupby().apply() reference on 1e6 rows, the check catching an
assistant-style polars version).

### Notebooks: `%%viv` (python/vipyr/magic.py)

In IPython or Jupyter, `%load_ext vipyr` adds three magics:

- `%%viv NAME [--fast]`: the cell's Viv defs become the module `NAME` in the
  notebook. The cell is compiled through `vipyr.load`, so an unchanged cell
  comes from the cache and DataFrame arguments work as above. A compile
  error names the cell and its line (`<%%viv NAME>, line 7`, counting the
  `%%viv` line as line 1); a run-time error's `-->` line does the same.
- `%viv_export NAME -o FILE.c`: the module as one C file, the one
  `viv export` wrote and the shared library is built from.
- `%viv_runs [PATH | off | on]`: where run records go, or stop and restart
  recording.

Every successful call of a cell module's function appends a JSON line to
`.viv_runs.jsonl` beside the notebook (`$VIPYR_RUNS` or `%viv_runs PATH`
moves it): the function, the SHA-256 of the cell's source, the compiler's
identity, a hash per input (per column for a DataFrame, plus a pandas
index), the output hash, the time and the wall time. The `id` is a
content address over what ran, the inputs hash and the outputs hash. Equal runs share an id.

The rerun check: a call with the same source, compiler, flags and input
hashes as an earlier record in the log, whose output hash differs, prints
`vipyr: rerun mismatch` with both record ids, and the new record carries
`mismatch_of`. Viv that reads only its arguments repeats bit for bit, so the
warning means a hidden input. Hashing reads every input once per call, about
as long as one pass over the data; in a loop of many calls, `%viv_runs off`.

A cell's `import` finds Viv's library modules but not `.viv` files beside
the notebook (the cell is compiled from `__pycache__/vipyr/cells/`); put
such code in a file and use `vipyr.load`. The example is
examples/datasci/notebook_demo.ipynb; python/tests/test_magic.py runs it.

### Deploying a scikit-learn pipeline (python/vipyr/compile)

`vipyr compile model.pkl --data holdout.csv [--features f.viv] -o m.c`
writes the pipeline as readable Viv (`m.viv`), exports it, and adds a CSV
command line and `--selftest` to one C99 file. Trees, forests and boosting
match scikit-learn to the bit (probabilities through `exp` to 1 ulp),
linear models to a stated tolerance. examples/deploy/README.md has the
supported steps, the limits and a worked example; python/tests/test_compile.py
checks each family.
