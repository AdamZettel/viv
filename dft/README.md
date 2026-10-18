# pwdft — plane-wave density-functional theory, in Viv

A from-scratch Kohn–Sham DFT code: plane-wave basis, separable norm-conserving
pseudopotentials, LDA, self-consistent field, band structures. Six materials
(silicon, germanium, diamond, gallium arsenide, aluminum, copper), each checked
against an independent numpy implementation and against published LDA band
energies.

```sh
cargo build --release -p viv

dft/run.sh si            # silicon, dense engine
dft/run.sh si it         # silicon, iterative engine
dft/run.sh cu it         # copper: 1363 plane waves x 128 k-points, ~3 min
dft/check.sh             # five materials, every test; fails on any FAIL
dft/check.sh cu
dft/build-wasm.sh        # Viv -> C -> wasm: the modules the website runs (needs emscripten)
dft/build-wasm.sh --check   # were the committed modules built from the current sources?
```

`run.sh` compiles a deck, the engine and a check file to one C program and runs
it. `build-wasm.sh` hands that same C to emscripten. The same files also run in
the interpreter (`viv run --use ...`).

<https://viv-lang.org/dft> offers both in a browser tab: the **compiled** module
(`dft/wasm/<mat>.wasm`, ~210 KB, no interpreter in it) and the **interpreter**
(`viv.wasm`, reading the source). Every file is readable at
<https://viv-lang.org/dft/source>.

## Two engines

| | `pwdft.viv` (dense) | `pwdft_it.viv` (iterative) |
|---|---|---|
| H at each k-point | built as a matrix, diagonalized by `eig_herm` | never formed: applied by FFT |
| eigensolver | Householder + implicit QL | LOBPCG (`lib/lobpcg`), Teter–Payne–Allan preconditioner |
| cost per k-point | O(N³) in plane waves | O(bands · N log N) |
| checked against | a numpy twin, step for step: same energy to 1e-8 Ha **and the same iteration count** | that same *dense* numpy reference (a different algorithm as well as a different language), to 1e-7 Ha and 1e-5 eV |
| copper | not run: too slow | 180 s native, 258 s as wasm in a browser |

Shared by both, in `pw_common.viv`: geometry, the GTH closed forms, the FFT
grid, occupations and the Fermi level, Anderson mixing. `kmesh.viv` builds the
k-mesh.

Seconds, on one core of an Apple M2 Pro. "wasm" columns are what a browser tab
does: the compiled module, and the interpreter reading the source.

| | cutoff | plane waves | k-points | dense | iterative | wasm, compiled | wasm, interpreted |
|---|---|---|---|---|---|---|---|
| Si | 10 Ha | 420 | 16 | 55 | 7 | 8 | 20 |
| Ge | 14 | 778 | 16 | — | 10 | 12 | 33 |
| C | 30 | 620 | 16 | — | 9 | 11 | 27 |
| GaAs | 14 | 778 | 16 | — | 9 | 12 | 35 |
| Al | 10 | 178 | 128 | 24 | 6 | 7 | 22 |
| Cu | 50 | 1363 | 128 | — | 180 | 258 | — |

The dense engine is kept for silicon and aluminum, where it is cheap enough to
be the numpy reference's step-for-step twin. Copper as compiled wasm passes all
13 of its tests in a browser, in 4.3 minutes and 370 MB.

## Files

| File | What it is |
|---|---|
| `pwdft.viv`, `pwdft_it.viv` | the two engines |
| `pw_common.viv`, `kmesh.viv` | what they share; the symmetric k-mesh |
| `../lib/lobpcg.viv` | the iterative engine's eigensolver, a library of its own: `pwdft_it.viv` hands it H at one k-point and the preconditioner, both as functions |
| `<mat>.viv` | an input deck: ordinary Viv that sets the globals the engine reads. **Generated.** |
| `<mat>_check.viv`, `<mat>_it_check.viv` | what that material must reproduce on each engine, and its band-structure figure. **Generated**, by running the numpy reference |
| `run.sh`, `check.sh` | compile-and-run one; run them all |
| `<module>.intent`, `<module>.debug`, `lens.sh` | the lenses of kmesh, pw_common, pwdft and pwdft_it: what each def and step computes and why, and what to check while it runs (see Reading the code, below) |
| `ref/notebook_si.py` | the numbers the silicon notebook post (`posts/2026-09-25-silicon-bands-from-the-crystal-up.viv`) asserts, from the numpy reference at the post's smaller settings |
| `build-wasm.sh`, `wasm/` | Viv → C → wasm for the website, the committed modules, and the manifest of C hashes that shows whether they match the current sources |
| `pseudo/make_pseudo.py` | writes pseudopotential parameters as Viv source, from CP2K's table |
| `ref/pwdft_ref.py`, `ref/pwdft_iter.py` | the independent numpy implementations, dense and iterative |
| `ref/make_deck.py`, `ref/materials.py` | the generator, and the one table of materials it reads |
| `ref/verify_formulas.py` | every closed form against something that does not share its derivation |
| `ref/cutoff_convergence.py`, `ref/cutoff_kinks.py`, `ref/kmesh_symmetry.py`, `ref/cu_smearing.py`, `ref/cu_convergence.py`, `ref/check_iter.py` | the measurements behind the choices below |
| `results/cu.out` | copper's recorded native run, which the website draws |

## Reading the code

Two ways in, besides the code itself.

**The lenses.** `kmesh.intent`, `pw_common.intent`,
`pwdft.intent` and `pwdft_it.intent` say, per def and per step, what it
computes and why, in plain words and with no code. `.debug` files list the
values to watch and the conditions that must hold while it runs. Every entry
carries the body hash of its def, so an entry whose def has changed since it
was last read is reported as stale; `dft/check.sh` runs `dft/lens.sh check`
first and fails on any stale, missing or uncited entry.

```sh
dft/lens.sh view pwdft              # the dense engine as pseudocode
dft/lens.sh view pwdft_it all       # intent and code interleaved
dft/lens.sh explain pwdft           # the intent in the order the calls run
dft/lens.sh check                   # stale stamps, missing text, uncited numbers
dft/lens.sh stamp pwdft build_h     # after re-reading an entry: record the new hash
```

The modules need an input deck to build, so `lens.sh` checks each one as part
of silicon's program (its header explains how). Only the dense engine,
`pwdft.viv`, has `step` blocks. A step adds code to the generated C, and
`pw_common.viv`, `kmesh.viv` and `pwdft_it.viv` are compiled into the
committed wasm modules, so steps there would make `build-wasm.sh --check`
report them stale. Their two long defs, `anderson_step` and `setup_k`, get
steps when the wasm is next rebuilt; until then `lens.sh` lets them through
by name. Their intent files still describe the order of the work in the
module text.

**The notebook post**, `posts/2026-09-25-silicon-bands-from-the-crystal-up.viv`:
a small silicon calculation (5 Ha cutoff, 4 k-points, plain linear mixing)
written out cell by cell, from the crystal to the band structure, in the
browser tab in a few seconds. Its numbers are checked against
`ref/notebook_si.py`, and its last cells compare it with the 10 Ha run here.

## How it is checked

Three independent layers, because a DFT code can be wrong in ways that still
draw a plausible band structure.

**1. The formulas** (`ref/verify_formulas.py`). The GTH projector and
local-potential closed forms are tested against brute-force radial quadrature
of their *real-space definitions*, and the Teter-93 Pade LDA against **libxc**.
A closed form copied from a paper can be wrong and still run; this way an error
has to be made twice, independently, to survive.

**2. The implementation** (the first tests in every check file). Total energy,
Fermi level, and band energies at L, Γ and X against the numpy reference. The
check files are *generated by running that reference*, so the numbers a
deck is checked against are always the reference's current output. The numpy iterative solver, which is
copper's reference, itself agrees with numpy dense diagonalization to 1e-12 Ha
where both can run (`ref/check_iter.py`).

**3. The physics** (the rest of each check file). Published LDA values, with
tolerances that reflect the spread between papers rather than this code's
precision:

| silicon | this code | asserted |
|---|---|---|
| Γ₁ᵥ, valence-band bottom | −11.98 eV | −11.95 ± 0.15 |
| indirect gap | 0.47 eV | 0.5 ± 0.1 (measured: 1.17) |
| conduction minimum | 0.85 of Γ–X | 0.85 ± 0.1 |
| X₄ᵥ | −2.88 eV | −2.87 ± 0.12 |
| L₃′ᵥ | −1.21 eV | −1.2 ± 0.1 |

| copper, eV from E_F | this code | published LDA | measured |
|---|---|---|---|
| Γ₁ | −9.63 | −9.79 | −8.60 |
| Γ₁₂ | −2.25 | −2.27 | −2.78 |
| X₅ | −1.40 | −1.40 | −2.01 |
| L₃ | −1.59 | −1.63 | −2.25 |
| L₂′ | −1.16 | −1.12 | −0.85 |
| X₅ − X₁ | 3.59 | 3.70 | 3.17 |

Published and measured: Marini, Onida & Del Sole, *Phys. Rev. Lett.* **88**,
016403 (2002), Table I — read from the paper, not recalled. Reproducing LDA's
*failures* (silicon's gap at under half the measured one, copper's d bands
0.5–0.7 eV too shallow) is a sharper test than getting a right answer would be.

One number is shown and **not asserted**: copper's L₁ − L₂′ gap is 4.87 eV here
against 5.40 in that paper, whose pseudopotential keeps 3s and 3p in the
valence. Why that one empty state differs by 0.5 eV has not been traced, so
the check file claims no tolerance for it.

## Choices, and the measurement behind each

- **Cutoffs are measured, not inherited.** `ref/cutoff_convergence.py` runs each
  material up a ladder of cutoffs; the one in `materials.py` is the first whose
  asserted band energies sit within ~0.02 eV of the top of the ladder. The first
  version of these decks used 6 Ha for silicon, germanium and aluminum. It drew
  plausible bands, agreed with the numpy reference to 1e-8 Ha (they shared the
  cutoff), and had **germanium's band order wrong**: at 6 Ha the s-like Γ₂′ state
  sits 0.88 eV above the valence-band top and germanium passes for a
  semiconductor with its gap at L; converged, it sits 0.015 eV *below* — LDA
  germanium has no gap, which is the published result. What exposed it was
  tripling the band path's resolution: the conduction minimum moved one point
  off L, because at 6 Ha the energy *at* L sat 21 meV off the curve through its
  neighbors (`ref/cutoff_kinks.py`). Agreement between two implementations says
  nothing about a parameter they share.
- **Every band at every path point is pinned**, not only L, Γ and X: 61 points
  × the valence bands and two above, against the numpy reference.
- **The k-mesh is the four-shift fcc mesh.** A single `(½ ½ ½)`-shifted mesh in
  primitive fcc coordinates is not closed under the cubic point group; with no
  explicit symmetrization the density is only as symmetric as the mesh, and
  silicon's valence-band *triplet* came out split by 56 meV. The four shifts
  are that shift's orbit: split 0, and 16 k-points land within 6.4 mHa of 128.
  Every semiconductor check file tests the triplet. `ref/kmesh_symmetry.py`
- **Metals use 4×4×4 (128 k-points) and a 0.01 Ha Fermi–Dirac width.**
  Aluminum's E_F is off by 0.7 eV at 2×2×2 and 0.05 at 4×4×4. At 0.02 Ha the
  smearing tail reaches copper's d bands and pushes E_F up 0.26 eV; from 0.01
  to 0.005 it moves 0.04. `ref/cu_smearing.py`
- **Copper's cutoff is 50 Ha.** The d bands move ~0.1 eV from 50 to 60 Ha and
  ~0.05 from 60 to 70. `ref/cu_convergence.py`
- **The FFT grid side is the next 2·3·5-smooth number ≥ 4M + 2.** Any side that
  large is exact; these are the ones Viv's FFT has a dedicated butterfly for.
- **LOBPCG builds its search direction directly** from the `w` and `p` columns
  of the Ritz coefficients, never as `xn − (xn xᴴ) x`. That subtraction cancels
  to the size of the step, and the iteration, after converging to 1e-6, then
  *diverged* by ×5 a step. Converged bands are soft-locked; search directions
  are normalized before the rank-revealing orthonormalization (un-normalized,
  it discards them at a residual of ~1e-5 and the iteration stalls).
- **SCF convergence requires the occupied bands' residuals**, not only the
  density and the energy.

## Pseudopotentials

Goedecker–Teter–Hutter / Hartwigsen–Goedecker–Hutter: norm-conserving,
separable, and *analytic* — a handful of numbers per element, with closed-form
Fourier transforms for both the local part and the projectors. That makes them
the right first choice for a code with no file I/O: a pseudopotential arrives
as a few array assignments.

Parameters are **never typed by hand.** `pseudo/make_pseudo.py` reads CP2K's
`GTH_POTENTIALS` table (the copy PySCF ships) and emits Viv source:

```sh
python3 dft/pseudo/make_pseudo.py Si
python3 dft/pseudo/make_pseudo.py Ga As        # two species
python3 dft/pseudo/make_pseudo.py Cu:q11       # an explicit valence
```

While writing this code the silicon parameters were recalled from memory
correctly and the *copper* ones were not (r = 0.4314 where the table says
0.4237) — an error of tens of meV that no test of the code itself would have
caught.

## Regenerating

```sh
python3 dft/ref/make_deck.py Si Ge C GaAs Al Cu
```

Needs numpy, scipy and PySCF (for the pseudopotential table). Reference results
are remembered in `ref/.cache/`, keyed on the material's settings *and the text
of both numpy implementations*, so rewording a test is instant and changing
anything that could change a number is not.

## What it does not do

- **Total energies comparable between structures.** The ion–ion (Ewald) term is
  left out. It is a constant for fixed atoms and does not move a band.
- **GGA.** LDA only; PBE needs density gradients.
- **Spin, or spin–orbit coupling.**
- **Symmetry reduction.** The full mesh is computed, halved only by time
  reversal — which is why the density is symmetric without being symmetrized,
  and why copper needs 128 k-points.
- **Forces, or lattices other than fcc.**
