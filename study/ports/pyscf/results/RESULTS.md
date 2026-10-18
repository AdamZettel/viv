# P0: lib/hf against PySCF 2.5.0, results

Written by `python3 study/ports/harness/port.py report pyscf` from results/verify.json,
results/final.json and results/provisional.json. A time marked `*` was taken while the
1-minute load average reached 2 or more (provisional); unmarked times stayed below it.

Machine: Apple M2 Pro, 12 CPUs. PySCF 2.5.0, Python 3.11.5, NumPy 1.24.4; Apple clang version 17.0.0 (clang-1700.4.4.1); Viv at d2b8174.

## Verification

Both sides from the core-Hamiltonian guess, CDIIS over 8 matrices starting at the second
iteration; PySCF at conv_tol 1e-12, conv_tol_grad 1e-8, Viv at e_tol 1e-12, grad_tol 1e-9.
J/K builds: PySCF makes cycles + 2 (the guess's and the conv_check cycle's); Viv one per iteration.

| case | AOs | PySCF energy | Viv - PySCF | max orbital-energy gap | PySCF cycles (J/K builds) | Viv iterations |
|---|---|---|---|---|---|---|
| water_sto3g | 7 | -74.96306312972919 | -1.4e-13 | 1.9e-09 | 8 (10) | 9 |
| water_631g | 13 | -75.98394849810563 | 4.3e-14 | 1.1e-10 | 13 (15) | 15 |
| water_ccpvdz | 24 | -76.02676567311998 | 7.1e-14 | 2.3e-09 | 13 (15) | 15 |
| water_ccpvtz | 58 | -76.05711408311929 | -7.4e-13 | 2.0e-09 | 15 (17) | 16 |
| benzene_ccpvdz | 114 | -230.72082519861564 | -4.0e-13 | 4.6e-10 | 26 (28) | 22 |
| water1_631gs | 18 | -76.00909910859744 | 1.3e-13 | 1.7e-09 | 18 (20) | 15 |
| water2_631gs | 36 | -152.0122289223226 | -3.4e-13 | 8.1e-10 | 14 (16) | 16 |
| water4_631gs | 72 | -303.99398803770885 | -2.3e-13 | 6.3e-10 | 16 (18) | 21 |
| water8_631gs | 144 | -608.0105045827227 | 1.3e-11 | 6.4e-10 | 23 (25) | 25 |
| watercat_631g_uhf | 13 | -75.58051984404865 | 1.4e-14 | 3.0e-09 | 17 (19) | 16 |
| watercat_ccpvdz_uhf | 24 | -75.63187060619029 | -8.5e-14 | 2.7e-09 | 17 (19) | 16 |

## Timing, whole SCF (integrals included)

Single thread. PySCF: `scf.RHF(mol).kernel()` (or UHF) on a fresh SCF object, in process after a
warm-up, median of the runs, no checkpoint file (the `chk` column keeps PySCF's default HDF5
checkpoint, written every cycle). Viv: the in-program stage clocks around `scf_with` in the native
binary (`cc -O2`, checks on) and the release build (`-O3 -mcpu=native -DVIV_RELEASE`), median of
the runs; wasm: process time minus the baseline program's (the molecule only). The gate is
native / PySCF <= 1.5.

| case | AOs | J/K | PySCF | PySCF chk | Viv native | release | wasm | native / PySCF | release / PySCF | per J/K build, native / PySCF | iterations Viv / PySCF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| water_sto3g | 7 | in core | 2.3 ms | 8.1 ms | 2.4 ms | 2.1 ms | 16.1 ms | 1.0x | 0.9x | 0.2x | 8 / 7 |
| water_631g | 13 | in core | 4.6 ms | 13.4 ms | 6.8 ms * | 5.7 ms * | 24.9 ms * | 1.5x | 1.3x | 1.1x | 12 / 11 |
| water_ccpvdz | 24 | in core | 9.3 ms | 18.2 ms | 20.8 ms * | 16.4 ms * | 62.2 ms * | 2.2x | 1.8x | 1.9x | 12 / 11 |
| water_ccpvtz | 58 | in core | 100.5 ms | 108.6 ms | 265.2 ms | 198.0 ms * | 974.3 ms | 2.6x | 2.0x | 2.2x | 12 / 11 |
| benzene_ccpvdz | 114 | direct | 15.6 s * | 15.7 s * | 55.7 s * | 43.1 s * | 111.5 s * | 3.6x | 2.8x | 3.6x | 14 / 12 |
| water1_631gs | 18 | in core | 5.8 ms | 13.7 ms | 11.8 ms * | 9.7 ms * | 38.4 ms * | 2.0x | 1.7x | 1.4x | 11 / 10 |
| water2_631gs | 36 | in core | 36.0 ms | 45.5 ms | 119.8 ms * | 97.5 ms * | 270.4 ms * | 3.3x | 2.7x | 2.2x | 12 / 11 |
| water4_631gs | 72 | in core | 389.7 ms | 401.7 ms | 1.37 s * | 1.11 s * | 3.20 s * | 3.5x | 2.8x | 2.2x | 13 / 12 |
| water8_631gs | 144 | direct | 26.5 s * | 26.7 s | 245.1 s * | 188.7 s * | - | 9.2x | 7.1x | 9.8x | 16 / 15 |
| watercat_631g_uhf | 13 | in core | 5.8 ms | 14.5 ms | 10.5 ms | 8.0 ms | 34.4 ms | 1.8x | 1.4x | 2.3x | 12 / 11 |
| watercat_ccpvdz_uhf | 24 | in core | 12.7 ms | 21.6 ms | 63.1 ms | 45.1 ms | 122.3 ms | 5.0x | 3.6x | 5.5x | 12 / 11 |

PySCF in core on the direct cases (Viv cannot hold their packed ERIs): benzene_ccpvdz 2.22 s; water8_631gs 5.62 s.

## Fast mode (2026-10-05, branch `fast`, provisional)

PySCF, native, release and the fast build (release plus fast mode)
timed in one pass, `port.py time pyscf --engines
original,native,release,fast`, with the load between 13 and 98: every cell is
provisional (`*`). Each fast run's energy matched the oracle within 1e-8 Eh.
Correctness is judged on the exact builds (Verification above).

| case | AOs | PySCF | Viv native | release | fast | native / PySCF | release / PySCF | fast / PySCF | fast / release | J/K: fast / release |
|---|---|---|---|---|---|---|---|---|---|---|
| water_sto3g | 7 | 2.4 ms* | 2.4 ms* | 2.1 ms* | 2.0 ms* | 1.0x | 0.9x | 0.8x | 0.94 | 0.99 |
| water_631g | 13 | 4.8 ms* | 6.8 ms* | 5.9 ms* | 5.5 ms* | 1.4x | 1.2x | 1.1x | 0.92 | 0.98 |
| water_ccpvdz | 24 | 9.7 ms* | 25.1 ms* | 16.9 ms* | 16.5 ms* | 2.6x | 1.7x | 1.7x | 0.97 | 0.97 |
| water_ccpvtz | 58 | 102.9 ms* | 488.2 ms* | 421.0 ms* | 538.2 ms* | 4.7x | 4.1x | 5.2x | 1.28 | 1.74 |
| benzene_ccpvdz | 114 | 17.83 s* | 112.79 s* | 71.53 s* | 60.79 s* | 6.3x | 4.0x | 3.4x | 0.85 | 0.85 |
| water1_631gs | 18 | 6.6 ms* | 13.4 ms* | 10.7 ms* | 9.8 ms* | 2.0x | 1.6x | 1.5x | 0.92 | 0.95 |
| water2_631gs | 36 | 39.3 ms* | 138.9 ms* | 109.8 ms* | 106.9 ms* | 3.5x | 2.8x | 2.7x | 0.97 | 0.86 |
| water4_631gs | 72 | 416.6 ms* | 1.59 s* | 1.29 s* | 1.22 s* | 3.8x | 3.1x | 2.9x | 0.95 | 0.96 |
| watercat_631g_uhf | 13 | 6.3 ms* | 11.5 ms* | 6.6 ms* | 9.7 ms* | 1.8x | 1.0x | 1.5x | 1.47 | 0.99 |
| watercat_ccpvdz_uhf | 24 | 14.0 ms* | 35.8 ms* | 25.1 ms* | 26.0 ms* | 2.6x | 1.8x | 1.9x | 1.04 | 0.94 |

## The gate

| case | native / PySCF | <= 1.5x | stages PySCF runs in Python where Viv is slower | taken |
|---|---|---|---|---|
| water_sto3g | 1.0x | yes | none | final |
| water_631g | 1.5x | yes | none | provisional (load reached 2.1) |
| water_ccpvdz | 2.2x | no | none | provisional (load reached 2.1) |
| water_ccpvtz | 2.6x | no | diis, other | final |
| benzene_ccpvdz | 3.6x | no | diis, other | provisional (load reached 2.7) |
| water1_631gs | 2.0x | no | none | provisional (load reached 2.1) |
| water2_631gs | 3.3x | no | diis, other | provisional (load reached 3.1) |
| water4_631gs | 3.5x | no | diis, other | provisional (load reached 2.0) |
| water8_631gs | 9.2x | no | diis, other | provisional (load reached 2.5) |
| watercat_631g_uhf | 1.8x | no | none | final |
| watercat_ccpvdz_uhf | 5.0x | no | diis | final |

## By stage

Exclusive times (a stage's time leaves out the stages inside it). PySCF's diagonalization
row is `scipy.linalg.eigh(F, S)` each iteration, which orthogonalizes as it solves; Viv
orthogonalizes once and diagonalizes `X^T F X`. Viv's DIIS row holds the commutator step
(PySCF computes its error vector inside CDIIS.update).

### water_sto3g (7 AOs, J/K in core)

| stage | PySCF | PySCF runs it in | Viv native | Viv release | native / PySCF | release / native |
|---|---|---|---|---|---|---|
| one-electron integrals | 0.2 ms | libcint (C) via Python | 0.7 ms | 0.7 ms | 3.5x | 0.98 |
| two-electron integrals (in core) | 0.5 ms | libcint (C) | 1.4 ms | 1.2 ms | 2.7x | 0.89 |
| guess | 0.0 ms | Python + LAPACK | 0.0 ms | 0.0 ms | 8.5x | 0.86 |
| orthogonalization (once) | 0.0 ms | (inside eigh) | 0.0 ms | 0.0 ms | - | 0.79 |
| J/K builds | 0.3 ms | libcvhf (C) | 0.1 ms | 0.0 ms | 0.2x | 0.43 |
| DIIS (error vector and extrapolation) | 0.3 ms | Python + NumPy | 0.0 ms | 0.0 ms | 0.1x | 0.88 |
| diagonalization | 0.2 ms | LAPACK dsygvd (generalized, with S) | 0.0 ms | 0.0 ms | 0.2x | 0.80 |
| energy | 0.3 ms | Python + NumPy | 0.0 ms | 0.0 ms | 0.0x | 0.26 |
| the rest (Fock, density, occupation, loop) | 0.4 ms | Python + NumPy | 0.1 ms | 0.0 ms | 0.2x | 0.48 |
| total | 2.3 ms | | 2.4 ms | 2.1 ms | 1.0x | 0.88 |

### water_631g (13 AOs, J/K in core)

| stage | PySCF | PySCF runs it in | Viv native | Viv release | native / PySCF | release / native |
|---|---|---|---|---|---|---|
| one-electron integrals | 0.3 ms | libcint (C) via Python | 0.9 ms | 0.9 ms | 3.6x | 0.94 |
| two-electron integrals (in core) | 1.6 ms | libcint (C) | 4.6 ms | 4.1 ms | 2.9x | 0.89 |
| guess | 0.0 ms | Python + LAPACK | 0.1 ms | 0.1 ms | 20.6x | 0.76 |
| orthogonalization (once) | 0.0 ms | (inside eigh) | 0.0 ms | 0.0 ms | - | 0.71 |
| J/K builds | 0.7 ms | libcvhf (C) | 0.7 ms | 0.3 ms | 1.0x | 0.48 |
| DIIS (error vector and extrapolation) | 0.6 ms | Python + NumPy | 0.1 ms | 0.1 ms | 0.2x | 0.92 |
| diagonalization | 0.4 ms | LAPACK dsygvd (generalized, with S) | 0.2 ms | 0.1 ms | 0.5x | 0.63 |
| energy | 0.5 ms | Python + NumPy | 0.0 ms | 0.0 ms | 0.0x | 0.37 |
| the rest (Fock, density, occupation, loop) | 0.5 ms | Python + NumPy | 0.1 ms | 0.1 ms | 0.3x | 0.53 |
| total | 4.6 ms | | 6.8 ms | 5.7 ms | 1.5x | 0.84 |

### water_ccpvdz (24 AOs, J/K in core)

| stage | PySCF | PySCF runs it in | Viv native | Viv release | native / PySCF | release / native |
|---|---|---|---|---|---|---|
| one-electron integrals | 0.3 ms | libcint (C) via Python | 1.7 ms | 1.5 ms | 5.3x | 0.91 |
| two-electron integrals (in core) | 3.7 ms | libcint (C) | 12.4 ms | 10.3 ms | 3.3x | 0.83 |
| guess | 0.0 ms | Python + LAPACK | 0.3 ms | 0.2 ms | 76.1x | 0.58 |
| orthogonalization (once) | 0.0 ms | (inside eigh) | 0.2 ms | 0.1 ms | - | 0.58 |
| J/K builds | 2.5 ms | libcvhf (C) | 4.3 ms | 3.3 ms | 1.7x | 0.76 |
| DIIS (error vector and extrapolation) | 0.6 ms | Python + NumPy | 0.3 ms | 0.3 ms | 0.4x | 0.93 |
| diagonalization | 0.9 ms | LAPACK dsygvd (generalized, with S) | 1.1 ms | 0.5 ms | 1.2x | 0.48 |
| energy | 0.5 ms | Python + NumPy | 0.0 ms | 0.0 ms | 0.0x | 0.57 |
| the rest (Fock, density, occupation, loop) | 0.6 ms | Python + NumPy | 0.3 ms | 0.2 ms | 0.5x | 0.50 |
| total | 9.3 ms | | 20.8 ms | 16.4 ms | 2.2x | 0.79 |

### water_ccpvtz (58 AOs, J/K in core)

| stage | PySCF | PySCF runs it in | Viv native | Viv release | native / PySCF | release / native |
|---|---|---|---|---|---|---|
| one-electron integrals | 0.7 ms | libcint (C) via Python | 3.9 ms | 3.6 ms | 5.7x | 0.92 |
| two-electron integrals (in core) | 30.0 ms | libcint (C) | 105.2 ms | 81.6 ms | 3.5x | 0.78 |
| guess | 0.0 ms | Python + LAPACK | 4.1 ms | 2.0 ms | 566.8x | 0.49 |
| orthogonalization (once) | 0.0 ms | (inside eigh) | 2.7 ms | 1.5 ms | - | 0.54 |
| J/K builds | 62.6 ms | libcvhf (C) | 130.0 ms | 100.7 ms | 2.1x | 0.77 |
| DIIS (error vector and extrapolation) | 1.3 ms | Python + NumPy | 2.7 ms | 2.6 ms | 2.0x | 0.96 |
| diagonalization | 4.1 ms | LAPACK dsygvd (generalized, with S) | 12.6 ms | 4.7 ms | 3.1x | 0.37 |
| energy | 0.7 ms | Python + NumPy | 0.1 ms | 0.1 ms | 0.2x | 0.99 |
| the rest (Fock, density, occupation, loop) | 1.0 ms | Python + NumPy | 2.4 ms | 1.2 ms | 2.3x | 0.51 |
| total | 100.5 ms | | 265.2 ms | 198.0 ms | 2.6x | 0.75 |

### benzene_ccpvdz (114 AOs, direct)

| stage | PySCF | PySCF runs it in | Viv native | Viv release | native / PySCF | release / native |
|---|---|---|---|---|---|---|
| one-electron integrals | 5.6 ms | libcint (C) via Python | 54.4 ms | 46.8 ms | 9.7x | 0.86 |
| two-electron integrals (in core) | 0.0 ms | libcint (C) | 20.1 ms | 17.3 ms | - | 0.86 |
| guess | 0.0 ms | Python + LAPACK | 27.6 ms | 13.8 ms | 2728.3x | 0.50 |
| orthogonalization (once) | 0.0 ms | (inside eigh) | 19.4 ms | 10.5 ms | - | 0.54 |
| J/K builds | 15.6 s | libcvhf (C) | 55.5 s | 43.0 s | 3.6x | 0.77 |
| DIIS (error vector and extrapolation) | 5.5 ms | Python + NumPy | 18.2 ms | 17.8 ms | 3.3x | 0.98 |
| diagonalization | 17.6 ms | LAPACK dsygvd (generalized, with S) | 89.8 ms | 32.9 ms | 5.1x | 0.37 |
| energy | 2.1 ms | Python + NumPy | 0.4 ms | 0.4 ms | 0.2x | 0.84 |
| the rest (Fock, density, occupation, loop) | 3.5 ms | Python + NumPy | 13.6 ms | 6.9 ms | 3.9x | 0.51 |
| total | 15.6 s | | 55.7 s | 43.1 s | 3.6x | 0.77 |

### water1_631gs (18 AOs, J/K in core)

| stage | PySCF | PySCF runs it in | Viv native | Viv release | native / PySCF | release / native |
|---|---|---|---|---|---|---|
| one-electron integrals | 0.3 ms | libcint (C) via Python | 1.2 ms | 1.2 ms | 4.3x | 0.92 |
| two-electron integrals (in core) | 2.3 ms | libcint (C) | 8.0 ms | 6.8 ms | 3.4x | 0.84 |
| guess | 0.0 ms | Python + LAPACK | 0.2 ms | 0.1 ms | 40.2x | 0.65 |
| orthogonalization (once) | 0.0 ms | (inside eigh) | 0.1 ms | 0.1 ms | - | 0.64 |
| J/K builds | 1.1 ms | libcvhf (C) | 1.4 ms | 1.0 ms | 1.3x | 0.75 |
| DIIS (error vector and extrapolation) | 0.5 ms | Python + NumPy | 0.2 ms | 0.2 ms | 0.3x | 0.93 |
| diagonalization | 0.6 ms | LAPACK dsygvd (generalized, with S) | 0.5 ms | 0.3 ms | 0.8x | 0.56 |
| energy | 0.5 ms | Python + NumPy | 0.0 ms | 0.0 ms | 0.0x | 0.46 |
| the rest (Fock, density, occupation, loop) | 0.5 ms | Python + NumPy | 0.2 ms | 0.1 ms | 0.4x | 0.54 |
| total | 5.8 ms | | 11.8 ms | 9.7 ms | 2.0x | 0.82 |

### water2_631gs (36 AOs, J/K in core)

| stage | PySCF | PySCF runs it in | Viv native | Viv release | native / PySCF | release / native |
|---|---|---|---|---|---|---|
| one-electron integrals | 0.7 ms | libcint (C) via Python | 3.8 ms | 3.4 ms | 5.5x | 0.90 |
| two-electron integrals (in core) | 21.7 ms | libcint (C) | 89.0 ms | 75.1 ms | 4.1x | 0.84 |
| guess | 0.0 ms | Python + LAPACK | 1.0 ms | 0.5 ms | 229.8x | 0.55 |
| orthogonalization (once) | 0.0 ms | (inside eigh) | 0.7 ms | 0.4 ms | - | 0.56 |
| J/K builds | 9.7 ms | libcvhf (C) | 20.1 ms | 15.6 ms | 2.1x | 0.78 |
| DIIS (error vector and extrapolation) | 0.8 ms | Python + NumPy | 0.9 ms | 0.9 ms | 1.1x | 0.96 |
| diagonalization | 1.8 ms | LAPACK dsygvd (generalized, with S) | 3.1 ms | 1.3 ms | 1.7x | 0.41 |
| energy | 0.6 ms | Python + NumPy | 0.0 ms | 0.0 ms | 0.1x | 0.73 |
| the rest (Fock, density, occupation, loop) | 0.7 ms | Python + NumPy | 0.7 ms | 0.4 ms | 1.1x | 0.56 |
| total | 36.0 ms | | 119.8 ms | 97.5 ms | 3.3x | 0.81 |

### water4_631gs (72 AOs, J/K in core)

| stage | PySCF | PySCF runs it in | Viv native | Viv release | native / PySCF | release / native |
|---|---|---|---|---|---|---|
| one-electron integrals | 2.8 ms | libcint (C) via Python | 19.6 ms | 16.9 ms | 7.0x | 0.86 |
| two-electron integrals (in core) | 215.4 ms | libcint (C) | 980.2 ms | 817.2 ms | 4.6x | 0.83 |
| guess | 0.0 ms | Python + LAPACK | 7.1 ms | 3.5 ms | 913.6x | 0.50 |
| orthogonalization (once) | 0.0 ms | (inside eigh) | 4.8 ms | 2.6 ms | - | 0.54 |
| J/K builds | 160.2 ms | libcvhf (C) | 326.0 ms | 251.9 ms | 2.0x | 0.77 |
| DIIS (error vector and extrapolation) | 2.1 ms | Python + NumPy | 4.4 ms | 4.3 ms | 2.1x | 0.97 |
| diagonalization | 6.8 ms | LAPACK dsygvd (generalized, with S) | 23.8 ms | 8.8 ms | 3.5x | 0.37 |
| energy | 1.0 ms | Python + NumPy | 0.2 ms | 0.2 ms | 0.2x | 0.89 |
| the rest (Fock, density, occupation, loop) | 1.5 ms | Python + NumPy | 4.2 ms | 2.2 ms | 2.7x | 0.54 |
| total | 389.7 ms | | 1.37 s | 1.11 s | 3.5x | 0.81 |

### water8_631gs (144 AOs, direct)

| stage | PySCF | PySCF runs it in | Viv native | Viv release | native / PySCF | release / native |
|---|---|---|---|---|---|---|
| one-electron integrals | 14.4 ms | libcint (C) via Python | 165.7 ms | 120.3 ms | 11.5x | 0.73 |
| two-electron integrals (in core) | 0.0 ms | libcint (C) | 0.1 ms | 0.1 ms | - | 0.56 |
| guess | 0.0 ms | Python + LAPACK | 104.7 ms | 69.4 ms | 12252.7x | 0.66 |
| orthogonalization (once) | 0.0 ms | (inside eigh) | 73.3 ms | 49.7 ms | - | 0.68 |
| J/K builds | 26.5 s | libcvhf (C) | 244.0 s | 188.0 s | 9.2x | 0.77 |
| DIIS (error vector and extrapolation) | 11.9 ms | Python + NumPy | 288.1 ms | 203.1 ms | 24.2x | 0.70 |
| diagonalization | 33.5 ms | LAPACK dsygvd (generalized, with S) | 462.0 ms | 291.4 ms | 13.8x | 0.63 |
| energy | 3.2 ms | Python + NumPy | 0.7 ms | 0.6 ms | 0.2x | 0.92 |
| the rest (Fock, density, occupation, loop) | 6.6 ms | Python + NumPy | 41.2 ms | 26.6 ms | 6.3x | 0.65 |
| total | 26.5 s | | 245.1 s | 188.7 s | 9.2x | 0.77 |

### watercat_631g_uhf (13 AOs, J/K in core)

| stage | PySCF | PySCF runs it in | Viv native | Viv release | native / PySCF | release / native |
|---|---|---|---|---|---|---|
| one-electron integrals | 0.3 ms | libcint (C) via Python | 1.1 ms | 0.9 ms | 3.5x | 0.88 |
| two-electron integrals (in core) | 1.6 ms | libcint (C) | 5.4 ms | 4.2 ms | 3.3x | 0.79 |
| guess | 0.0 ms | Python + LAPACK | 0.1 ms | 0.1 ms | 16.5x | 0.75 |
| orthogonalization (once) | 0.0 ms | (inside eigh) | 0.1 ms | 0.1 ms | - | 0.70 |
| J/K builds | 1.1 ms | libcvhf (C) | 2.3 ms | 1.6 ms | 2.1x | 0.70 |
| DIIS (error vector and extrapolation) | 0.7 ms | Python + NumPy | 0.6 ms | 0.5 ms | 0.9x | 0.79 |
| diagonalization | 0.8 ms | LAPACK dsygvd (generalized, with S) | 0.8 ms | 0.5 ms | 0.9x | 0.69 |
| energy | 0.6 ms | Python + NumPy | 0.0 ms | 0.0 ms | 0.0x | 0.62 |
| the rest (Fock, density, occupation, loop) | 0.7 ms | Python + NumPy | 0.2 ms | 0.2 ms | 0.3x | 0.64 |
| total | 5.8 ms | | 10.5 ms | 8.0 ms | 1.8x | 0.76 |

### watercat_ccpvdz_uhf (24 AOs, J/K in core)

| stage | PySCF | PySCF runs it in | Viv native | Viv release | native / PySCF | release / native |
|---|---|---|---|---|---|---|
| one-electron integrals | 0.4 ms | libcint (C) via Python | 1.9 ms | 1.7 ms | 5.3x | 0.87 |
| two-electron integrals (in core) | 3.7 ms | libcint (C) | 28.6 ms | 21.7 ms | 7.8x | 0.76 |
| guess | 0.0 ms | Python + LAPACK | 0.7 ms | 0.5 ms | 63.9x | 0.69 |
| orthogonalization (once) | 0.0 ms | (inside eigh) | 0.4 ms | 0.3 ms | - | 0.69 |
| J/K builds | 4.7 ms | libcvhf (C) | 23.8 ms | 15.7 ms | 5.1x | 0.66 |
| DIIS (error vector and extrapolation) | 0.8 ms | Python + NumPy | 2.6 ms | 1.9 ms | 3.2x | 0.71 |
| diagonalization | 1.8 ms | LAPACK dsygvd (generalized, with S) | 4.2 ms | 2.8 ms | 2.4x | 0.66 |
| energy | 0.6 ms | Python + NumPy | 0.0 ms | 0.0 ms | 0.1x | 0.84 |
| the rest (Fock, density, occupation, loop) | 0.8 ms | Python + NumPy | 0.8 ms | 0.5 ms | 0.9x | 0.63 |
| total | 12.7 ms | | 63.1 ms | 45.1 ms | 5.0x | 0.71 |

## Cold

One fresh process. PySCF: `python3 pyscf_side.py cold <case>` (start, imports, molecule, SCF). Viv:
`viv` to C, `cc -O2` and the run.

| case | PySCF | Viv compile (viv + cc) | Viv run (process) | Viv total |
|---|---|---|---|---|
| water_sto3g | 343.4 ms | 5.44 s | 4.9 ms | 5.44 s |
| water_631g | 347.8 ms | 5.15 s | 9.5 ms | 5.16 s |
| water_ccpvdz | 350.5 ms | 5.27 s | 23.8 ms | 5.29 s |
| water_ccpvtz | 447.3 ms | 21.1 s | 269.7 ms | 21.3 s |
| benzene_ccpvdz | 16.1 s | 20.9 s | 56.1 s | 77.0 s |
| water1_631gs | 357.8 ms | 5.46 s | 14.9 ms | 5.47 s |
| water2_631gs | 381.1 ms | 5.56 s | 122.9 ms | 5.68 s |
| water4_631gs | 751.6 ms | 5.43 s | 1.37 s | 6.81 s |
| water8_631gs | 26.9 s | 4.13 s | 245.4 s | 249.6 s |
| watercat_631g_uhf | 344.4 ms | 3.97 s | 13.5 ms | 3.99 s |
| watercat_ccpvdz_uhf | 352.2 ms | 3.87 s | 66.3 ms | 3.93 s |
