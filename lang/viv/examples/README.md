# Viv examples — and their validated results

Every program here is written in **Viv**, compiled by `knotkit` to portable
C99, then compiled to a native binary. Each scientific result is validated
against an independent reference (the C++ `viv` interpreter, published
quantum-chemistry energies, or physical conservation laws).

## Building and running

```sh
# Viv source -> C
cargo run -p viv -- lang/viv/examples/<file>.viv -o /tmp/out.c

# C -> native binary
cc -std=c99 -O3 /tmp/out.c -lm -o /tmp/out
/tmp/out

# ...with OpenMP (for `parallel for`). On macOS with Homebrew libomp:
cc -std=c99 -O3 -Xpreprocessor -fopenmp \
   -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib \
   /tmp/out.c -lomp -lm -o /tmp/out
OMP_NUM_THREADS=8 /tmp/out
```

## Quantum chemistry

| Example | What it computes | Result | Reference |
|---|---|---|---|
| `h2_sto3g.viv` | RHF SCF, H₂, STO-3G (s only) | **−1.11671 Eₕ** | Szabo & Ostlund / interpreter |
| `scf_mcmurchie.viv` | Same, via the general McMurchie–Davidson engine | **−1.11671 Eₕ** | (engine validation) |
| `scf_water.viv` | **RHF SCF, H₂O, STO-3G (s + p orbitals)** | **−74.94208 Eₕ** | Crawford projects (−74.9420787); agrees to ~1 µEₕ |
| `scf_water_631gd.viv` | **RHF SCF, H₂O, 6-31G\* (s + p + d, 19 functions)** | **−76.0192 Eₕ** | variational ladder + s/p anchor (below) |
| `scf_sto3g.viv` | **Molecule-driven RHF/STO-3G** (element library H–Ne + Mg, DIIS, ERI symmetry, two-guess SCF) + **MP2** correlation | molecule database (below) | published HF/STO-3G + MP2 energies |
| `geomopt.viv` | **Geometry optimization** of a diatomic (numerical-gradient Newton on the SCF energy) | H₂ R_e = 1.346 bohr, HF = 1.806 bohr | HF/STO-3G equilibrium bond lengths |
| `geomopt_water.viv` | **Polyatomic geometry optimization** of water (2D internal-coord Newton) | r_OH = 1.870 bohr (0.989 Å), ∠HOH = 100.0°, E = −74.9659 | HF/STO-3G water equilibrium |
| `hchain_sto3g.viv` | RHF SCF, linear Hₙ chain (scalable N), parallel ERIs | H₈ = −4.06498, H₂₀ = −4.79091 | C++ interpreter (exact) |

### Molecule database (`scf_sto3g.viv` + `tests/molecule_db.rs`)

`scf_sto3g.viv` is a *general* STO-3G driver: it assembles the basis from a
per-element library given only the atoms (`MOL_BEGIN..MOL_END` block). The
database harness substitutes that block per molecule, compiles, runs, and
compares the total energy to a published value. Production-grade upgrades:

- **DIIS** (Pulay) Fock extrapolation — converges N₂ in **9 cycles** vs ~52 for damped iteration.
- **8-fold ERI permutational symmetry** — ~8× fewer two-electron integral evaluations (the parallel hot loop); identical energies.
- **Two-guess SCF** — runs from both the core and GWH guesses and keeps the lower (ground-state) energy. Neither guess is universally safe: the core guess sends N₂ to a wrong excited solution, while GWH gives the Be atom the wrong 1s²2p² occupation. Both are valid SCF fixed points, so the variational principle picks the true ground state.

| Molecule | E (kit-viv) / Eₕ | reference / Eₕ | err / mEₕ | reference source |
|---|---|---|---|---|
| He   | −2.80778   | −2.807784   |  0.00 | STO-3G atom |
| Be   | −14.35190  | −14.351880  | −0.02 | STO-3G atom |
| Ne   | −126.60500 | −126.604522 | −0.48 | STO-3G atom |
| H₂   | −1.11671   | −1.1167     | −0.01 | Szabo & Ostlund (firm) |
| LiH  | −7.86201   | −7.8633     |  1.29 | lit. HF/STO-3G |
| HeH⁺ | −2.84184   | −2.8418     | −0.04 | lit. HF/STO-3G |
| HF   | −98.57080  | −98.5708    |  0.00 | lit. HF/STO-3G |
| H₂O  | −74.94210  | −74.9420787 | −0.02 | Crawford projects (firm) |
| N₂   | −107.49600 | −107.4960   |  0.00 | lit. HF/STO-3G |
| CO   | −111.22500 | −111.2238   | −1.20 | lit. HF/STO-3G (geometry-sensitive) |
| CH₄  | −39.72690  | −39.7269    |  0.00 | lit. HF/STO-3G |
| Mg   | −197.02    | ~−197.0     | — | STO-3G atom (period-3 probe) |

The element library covers the full second period **H–Ne** (validated tightly by
the He/Be/Ne closed-shell atom energies plus the molecules above). **Period 3**
adds a 3s/3p shell; the Mg atom (−197.02) validates that machinery against the
known STO-3G value. Other period-3 exponents are deliberately left unpopulated
rather than reconstructed from memory — they should be copied from a basis-set
source, after which the atom-energy harness validates them on arrival (an Ar
attempt with reconstructed exponents was ~0.7 % off, which is exactly why).
Run: `cargo test -p viv --test molecule_db -- --nocapture`.

**Correlation — MP2.** On top of the converged RHF reference, `scf_sto3g.viv` also
computes the **MP2 correlation energy**: it transforms the AO two-electron integrals
to the MO basis (four O(N⁵) quarter-transforms) and sums the pair energies. Water/STO-3G
gives **E_corr = −0.0491496 Eₕ**, matching the Crawford project-#4 reference
(−0.049149636) to <1 µEₕ; the database reports E_corr for every molecule. (He and Ne
correctly give exactly 0 — a minimal basis leaves them no virtual orbitals.)

`scf_water.viv` / `scf_water_631gd.viv` are the headline: a real polyatomic
calculation with **s, p, and d** angular-momentum integrals (Hermite expansion +
Hermite Coulomb recurrences + Boys functions), reproducing a published HF/STO-3G
energy to microhartree, then extending to a polarized split-valence basis.

**Validating the basis hierarchy** — at one fixed (Crawford) geometry, each basis
improvement lowers the HF energy monotonically toward the HF limit, exactly as
the variational principle requires:

| Basis | functions | E(SCF) / Eₕ |
|---|---|---|
| STO-3G (s,p) | 7 | −74.9421 (exact vs Crawford −74.9420787) |
| 6-31G (split-valence s,p) | 13 | −75.9356 |
| 6-31G\* (+ d on O) | 19 | −76.0192 |
| HF basis-set limit | — | ~−76.067 |

## Finite elements

| Example | What it computes | Result |
|---|---|---|
| `fem_poisson.viv` | 2D Poisson −∇²u = f on the unit square, linear (P1) triangles, dense solve | RMS nodal error **O(h²)** (ratios 3.45 → 3.74 → 3.87 → 4 as the mesh halves); u(½,½) → 0.0625 |
| `fem_poisson_records.viv` | The same FEM using **records** (a `Mesh` value, per-element `Tri` values) | identical result — records are zero-cost bundling |

The convergence *rate* is the validation: against a manufactured solution
`u = x(1-x)y(1-y)`, P1 elements give a second-order error that quarters each time
the mesh is refined, exactly as theory requires. Reuses the `solve_linear` from
the SCF work. `tests/fem_poisson.rs` checks the rate and the records equivalence.

The records version is the worked example for the **record** type: a `record Mesh { node_x: Arr, node_y: Arr, elem: Arr }` is
passed to functions as one value (array fields are shared handles, so no copy of
the big arrays), and each element's geometry is gathered into a `record Tri`
instead of nine loose scalar arguments.

## Molecular dynamics

| Example | What it does | Result |
|---|---|---|
| `md_polymer.viv` | Lennard-Jones bead-spring polymer, velocity-Verlet, parallel forces | energy conserved (~0.5% bounded); chain coils, R_g 11.5 → 6.0 |

## Performance (12-core laptop, same source both ways)

| Workload | Interpreter | kit-viv native (1 thread) | speedup | OpenMP (12 threads) |
|---|---|---|---|---|
| H₈ SCF | 46.6 s | 0.082 s | ~570× | — |
| H₂₀ SCF | — | 2.67 s | — | 0.35 s (7.6×) |
| MD polymer (N=40, 6000 steps) | 33.3 s | 0.046 s | ~720× | — |

## Automatic differentiation

| Example | What it does | Result |
|---|---|---|
| `autodiff_forward.viv` | **Forward-mode AD** via dual numbers (records) — exact derivatives, one pass per input | f′(2)=9.926737; ∇g(1.5,0.5)=(2.5585, 5.4255) |
| `autodiff_reverse.viv` | **Reverse-mode AD** via a tape — all gradients in one backward pass; drives gradient descent | ∇g matches forward; descent → (3, −1) |
| `autodiff_deriv.viv` | **Automatic forward-mode AD as a compiler feature**: `deriv(fn(x) -> expr, x0)` dualizes the expression — no dual arithmetic written by hand | f′(2)=9.926737 (auto); d/dx √(1+x²)\|₃=0.948683; Newton √2=1.41421 |

The two hand-written modes agree (a cross-check), with no finite-difference error.
Forward mode is a plain library over a `Dual { v, d }` record; reverse mode is a
flat-array tape ("micrograd in Viv") whose backward sweep gives every input's
gradient in a single pass.

`deriv` goes one step further — it is **AD as an IR transform**, the project's
thesis (JAX's idea) realized in the compiler. `deriv(fn(x) -> expr, x0)` takes a
scalar-expression closure and emits the *exact* derivative by dualizing the
expression at lower-time (chain rule over `+ - * /`, `exp`, `sqrt`, and `pow` with
a constant exponent); the generated C is the inline chain-rule computation, with no
runtime `Dual` struct. It reuses the same closure-argument machinery as `simpson`.
Run: `cargo test -p viv --test autodiff`.

## Language-feature examples

`area.viv` (functions, arithmetic) and `control.viv` (recursion, `while`,
`for`, `break`/`continue`) are minimal feature demos.
