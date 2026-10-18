# Changelog

Viv uses 0.x version numbers until the language is stable: any release
(0.0.2, 0.1) may change the language, and each one lists what changed here.

## 0.0.1 (preview)

### Contracts and methods

`contract name(params) -> outputs { ... }` declares a problem, its intent,
the methods that solve it, what they must agree on (`within relative T`,
`within absolute T`, `exactly`) and the default. A
call runs the default; `name(...) using m`, `using name = m` and `viv run
--using name=m` choose a method, `settings m(name = e)` gives a method its
settings, and
`every method` (or `every <label> method`, a labeled group) runs them all
and stops at the first disagreement. The
compiler checks each method's signature against the contract.
`integrate` and `find_root` are contracts and now run their default method
(Gauss, the safeguarded secant) instead of the median of three; printed
values are unchanged. `lib/eigen/lowest.viv` holds `lowest_eigenpairs`
(iterative: `lanczos`; dense: `householder_ql`, `jacobi`), which
`vp.nuclear.shell_model` uses; its new keyword `eigensolver=` picks
"lanczos", "householder_ql", "jacobi", "every" or "every dense". `--set` takes
enum values (`--set eigensolver=LowestEigenpairs.Dense`).

### vipyr: Viv as a Python package

The two Python packages, vivlang (load, the import hook, the build backend,
the cell magic, scikit-learn to C99) and vivpy (the scientific programs),
are one package, vipyr: `pip install vipyr`, then `import vipyr as vp`. The
language keeps its name. The tools are at the top level (`vp.load`,
`vp.install`, `import vipyr.auto`, `%load_ext vipyr`), the programs are its
families (`vp.scf.run`, `vp.dft.pw`, ...), the build backend is
`vipyr.build` with `[tool.vipyr]`, the cache is `__pycache__/vipyr/` and the
run-record variable is `$VIPYR_RUNS`. The wheel installs the `vipyr` and
`viv` commands. New: `vp.statmech.ising` and `vp.nuclear.shell_model`, from
their manifests. Nothing was published under the old names.

### Release 0.0.1 of the exported programs

The first programs a Python user can run without installing Viv: `pwdft`
(plane-wave DFT), `vivscf` (Hartree-Fock, the PySCF port), `skyrmehf`
(Skyrme Hartree-Fock of oxygen-16), `vlasov` (Landau damping) and the kernel
collection K1, each one C file and one Python file written by `viv export`.
The files are GitHub release
assets, not committed. They need a C99 compiler (Clang or GCC; MinGW-w64 on
Windows should work and is untested; MSVC is unsupported) and Python 3.9 or
newer with NumPy.

Changes to the numbers every program computes, made for this release (every
recorded exact hash was recorded again):

- `exp` is correctly rounded in exact mode in all three engines: the double
  nearest e^x for every input, subnormal results and the last binade before
  overflow included (it was musl's, within one ulp; `exp(1.0)` printed
  2.7182818284590455). Checked against mpmath on 2,025,786 inputs.
- `sum(a)` of a whole array adds in NumPy's pairwise order in every exact
  engine (it added left to right), the order `mean`, `var`, `std` and the
  axis reductions already used; `mean(a)` is now `sum(a) / numel(a)` to the
  bit. Fast mode is not bound by the order.

The first public preview. It collects the work from May to October 2026.

### The language

- Numbers are 64-bit floats; `Int` is a checked 64-bit integer, with
  `[Int]` arrays, enums with exhaustive `match`, and optionals (`T?`).
- Arrays up to rank 4 with NumPy's broadcasting rules and about 240 builtins
  under NumPy's names: reductions with `axis=` and `keepdims=`, slices,
  `einsum`, sorting with `key=`, linear solves, eigensolvers, FFT.
- Records with typed fields, nesting, arrays of records, record update and
  record arithmetic.
- Strings with Python's semantics, f-strings and `fmt`; maps and map
  comprehensions; growable arrays.
- Comprehensions, if-expressions, negative indices, `in`, unpacking, multiple
  returns, keyword arguments with defaults, lambdas.
- Modules and imports in Python's forms (`import cas`, `from nn import
  rmsnorm`), with private names.
- `test` blocks and the assert builtins as syntax; a failed test makes the
  process exit nonzero.
- Automatic differentiation as a compiler transform: `deriv`, `grad` and
  `value_and_grad`, including over records of arrays.
- Contracts that run several strategies for one objective and fail when they
  disagree.
- Phrases: mathematics written as words (`integrate x^2 from 1 to 2`),
  including a program's own phrases.
- Named steps, and lenses that keep a program's intent, debug views and
  architecture rules beside its code.
- Figures as program output: `figure`, `plot`, `scatter`, `bars`, `plot2d`,
  `plot3d` and scenes.
- `linspace(a, b, n)` gives NumPy's answer for every n: `[a]` for n = 1 and
  an empty array for n = 0 (it used to return both endpoints), and a
  negative n is an error, in every engine.
- `@` of two vectors stops with an error that names `dot(u, v)` and
  `outer(u, v)`, the two products it could have meant.

### Engines and tools

- Three engines that print identical bytes: a tree-walking interpreter, a C99
  backend and a direct WebAssembly backend, with one libm
  (fdlibm via musl, with Viv's own correctly rounded `exp`) shared by all
  three.
- Fast mode: the platform's libm, fused multiply-add, BLAS and LAPACK in a
  native build, checked number by number against the exact engines.
- Performance tier 1: integer loop counters and indices, bounds checks proven
  away, a cheaper recursion guard; per-iteration memory release in loops;
  separate compilation.
- The compiler as a wasm module, so the language runs in a browser tab.
- `viv features`, a catalog of the language generated from the compiler's
  tables, each entry with an example tested in three engines.
- `viv graph`, `viv story` and the explorer: a program shown as its defs and
  steps, and a run placed in that graph.
- `viv atlas`: a program's defs by module with the calls between them and the
  counts of one run, as one HTML file that opens with no network (the site's
  code map component, bundled), as text (`--text`) or as JSON (`--json`);
  `vp.atlas` in vipyr.
- py2viv: numeric Python with NumPy translated to Viv, exactly or refused with
  the reason.
- Content addresses for every definition (`viv hash`, `viv diff`).
- The gate (`tools/gate.sh quick|full|nightly`), `tools/merge.sh`, and a bug
  ledger whose fixed bugs each have a test.

### Libraries and programs

- `lib/hf`: restricted and unrestricted Hartree-Fock, checked against PySCF
  2.5.0 on eleven cases (energies within 1.3e-11 hartree).
- `lib/minpack` (Levenberg-Marquardt, checked against the Fortran), linear
  algebra (Schur form, Riccati equations, LQR), `lib/nn` and `lib/optim` (a
  small GPT), `lib/nprandom` (NumPy's random generators, draw for draw),
  `lib/cas` (stage 0 of a computer algebra system).
- `dft/`: a plane-wave DFT code, six materials, checked against NumPy
  references and published results.
- `pde/`: 15 PDE solvers, each tested against an exact solution, a conserved
  quantity and a measured order of accuracy.
- `study/`: Python/Viv comparisons, the Viv programs of three published
  benchmark suites, and the PySCF port.

### The website

- viv-lang.org, built from this repository: the workbench, the tour, the
  reference, the explorer, the comparisons, the DFT page and the posts.

### Known limitations

See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md): 31 open bugs at this
writing, the features the Rosetta Code study found missing (arbitrary-precision
integers, generators, map literals, standard input, exceptions, sets), and the
benchmark programs Viv cannot run.
