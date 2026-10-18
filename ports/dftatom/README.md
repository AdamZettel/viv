# dftatom, in Viv (vipyr-dftatom)

**dftatom** is by **Ondřej Čertík, John E. Pask and Jiří Vackář**:
<https://github.com/certik/dftatom> (MIT license). If you use this package, cite

<!-- aispeak-ok: the paper's title -->
> O. Čertík, J. E. Pask and J. Vackář, "dftatom: A robust and general Schrödinger and Dirac
> solver for atomic structure calculations", Comput. Phys. Commun. 184, 1777 (2013),
> doi:10.1016/j.cpc.2013.02.014.

dftatom solves the radial Schrödinger and Dirac equations for any potential on an exponential
mesh, and the self-consistent Kohn-Sham atom in the local density approximation, nonrelativistic
(LDA) or relativistic (RLDA), for Z = 1 to 92. This directory is its port to Viv: the same
algorithms, written as Viv and checked against upstream's Fortran. With the same math library it
returns the same doubles as the Fortran on every case of `COMPARISON.md`, and it reproduces the
NIST atomic reference data to the accuracy upstream states (1e-6 Hartree in total energies, 2e-6
in eigenvalues).

## Use

From Python, with vipyr (`vp.dftatom` is this package's `vipyr.packages` entry point; in a
checkout vipyr finds it without an install, and `import vipyr_dftatom` is the same module):

```python
import vipyr as vp
r = vp.dftatom.atom(Z=26)                        # iron, LDA, N = 6000
print(r.etot, r.ekin, r.ecoul, r.eenuc, r.exc)   # Hartree
for o in r.orbitals:
    print(o.n, o.l, o.kappa, o.occupation, o.energy)
r.mesh, r.density, r.v_tot                       # Bohr, electrons / Bohr^3, Hartree

r = vp.dftatom.atom(Z=92, relativistic=True, r_min=1e-8, a=6.2e7, N=18082)
assert r.converged and r.check()
```

The keywords, with the program's values: `Z=26`, `relativistic=False`, `r_min=1e-7`,
`r_max=50`, `a=2.7e6` (last mesh interval over the first), `N=6000` (intervals), `c=137.0359895`,
`reigen_eps=1e-10`, `reigen_max_iter=100`, `mixing_eps=5e-9`, `mixing_alpha=0.5`,
`mixing_max_iter=200`, `perturb=True`. For the 1e-8 Hartree accuracy upstream documents, take
`N` from upstream's `tests/lda/data8.dat` (nonrelativistic, `r_min=1e-7`, `a=2.7e6`) or
`tests/rlda/data8.dat` (relativistic, `r_min=1e-8`, `a=6.2e7`); the cases in
`oracle/upstream_reference.json` list them.

From Viv:

```
viv run ports/dftatom/atom.viv                  # iron; the settings are its top-level lines
viv run ports/dftatom/tests.viv                 # upstream's hydrogen-like uranium tests
```

and `from ports.dftatom.dftatom import atom_scf, solve_radial_eigenproblem, ...` in a program
of your own: the radial solvers take any potential on any mesh, as upstream's do.

## Files

| File | What |
|---|---|
| `dftatom.viv` | the port: mesh, integrators, eigenvalue search, Poisson, LDA/RLDA, SCF |
| `states.viv` | the ground configurations, generated from upstream's `states.f90` |
| `atom.viv`, `atom.export.toml`, `atom.cases.json` | the exported program, its manifest and its cases |
| `dftatom.exact_hashes.json` | the result hashes, the same from the library, native and interpreter runs |
| `tests.viv` | upstream's `Z92n7all` tests |
| `test_dftatom.py` | the port against the Fortran, upstream's converged tables and NIST |
| `oracle/` | the Fortran driver, the scripts that ran it and fetched NIST, and their output |
| `vipyr_dftatom/`, `pyproject.toml` | the Python package |
| `LICENSE`, `AUTHORS`, `CITATION.cff`, `UPSTREAM.md`, `COMPARISON.md` | attribution and the comparison |
