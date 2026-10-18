# Hartree-Fock in Viv, locked against PySCF 2.5.0

This directory holds the programs that check `lib/hf` as a whole: the lock
ladder (energies from computed integrals against PySCF), the metamorphic
suites, the coverage gate and the bridge to the legacy integral code.

## Water, before and after

The library's user-facing layer was rebuilt on records. Water in 6-31G, RHF, before: the basis came from PySCF's packs
in lib/hfref, the options were a 32-slot array, and the result was a flat
vector read at offsets.

    #! use lib/hf/hf
    #! use lib/hfref/water_631g

    opts = hf_opts(water631g_atoms(), 0, 0)
    opts[HF_OPT_SWEEP] = 1
    r = hf_energy(water631g_atoms(), water631g_shells(), water631g_prims(), opts)
    E = r[0]
    converged = hf_tail(r, 13, 1, DEN_T_CONV)
    iterations = hf_tail(r, 13, 1, DEN_T_ITER)

After: the molecule is written as PySCF takes it, the atoms as text and the
basis by its name (lib/hf/mole, over lib/xyz's parser and the tables of
lib/basis, generated from the Basis Set Exchange), and the result is a
record. The program uses the table of the set it names
(`lib/basis/pople631g`), and links no other; `lib/basis/all` gives every set.

    #! use lib/basis/pople631g
    #! use lib/hf/mole
    #! use lib/hf/scf

    mol = molecule("O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", "6-31g")
    r = rhf(mol)
    show r.energy
    show r.converged
    show r.iterations

The PySCF lines it replaces:

    from pyscf import gto, scf
    mol = gto.M(atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", basis="6-31g")
    r = scf.RHF(mol).run()
    print(r.e_tot)

The atoms may also be a whole XYZ file or one atom per line; the basis name
is read by PySCF's rule (any case, `-`, `_` and spaces ignored: "6-31G*",
"631gs"); `molecule_in(atoms, basis, "bohr")` reads bohr. Options change by
a record update, and so do the charge and spin:

    r = scf_with(mol, ScfOptions { guess: GS_CORE, level_shift: 0.2, ..scf_defaults() })
    ion = Molecule { charge: 1, spin: 1, ..mol }
    show rohf(ion).energy
    show uhf(ion).alpha.mo_energy

The array form is still there for a program that builds geometries in a
loop or wants one table linked instead of eight:
`molecule_from([8, 1, 1], [[0, 0, 0], [0, -0.757, 0.587], [0, 0.757, 0.587]], basis_631g())`
with `#! use lib/basis/pople631g`. `molecule_text.viv` checks that the two forms
build the same molecule, bit for bit, for every molecule and basis of the
lock ladder.

## Reading order

1. `lib/hf/scf.viv`: `rhf`, `uhf`, `rohf`, `scf_with`, the two calls behind
   them (`integrals(mol, o)` for the setup, `scf(setup)` for the loop), and
   the table of what an `ScfResult` holds. `lib/hf/mole.viv` (the string form),
   `lib/hf/molecule.viv` and `lib/basis/`: the molecule and its basis. `lib/hf/opts.viv`: every option and its default.
2. The contracts: the packs (`atoms`, `shells`, `prims`), the
   normalization convention, packed-8 ERI storage, the stage signatures,
   every registry (sections 11 and 12), and the records (section 15).
3. `lib/hf/basis.viv`, `lib/hf/sweep.viv`: the basis tables and the sweep
   drivers that run a registry and blame the member that dissents.
4. `lib/hf/integrals.viv`: the contract page. One registry per integral
   stage, one contract def per stage, each tolerance next to it.
5. `lib/hf/hf.viv`: the same pipeline with every integral member run and
   held to agree (`cross_check`), and the lock helpers the programs here use.
6. The SCF: `lib/hf/scf_loop.viv`, the one loop for RHF, UHF and ROHF, and
   the stages it picks from (`guess`, `orthogonalize`, `eigen`, `jk`,
   `accelerate`, `density`, `coupling`); `lib/hf/rhf.viv`, `uhf.viv` and
   `rohf.viv` hold each method's checks against PySCF.
7. `lib/hf/fuzz.viv`: seeded random bases and molecules, and the
   transformations the metamorphic suites apply.
8. The programs below.

## The programs

| Program | What it holds |
|---|---|
| `lock_water.viv` | Water STO-3G and 6-31G from the packs: all five ERI members swept and held to PySCF's full sets; RHF at every member of the guess, accelerator, orthogonalizer, eigensolver and J/K registries; the direct SCF; H2O+ ROHF/UHF and water spin -2 UHF/ROHF; H and He atoms. |
| `lock_631gs.viv` | Water and HF in 6-31G*, 6D and 5D, all five ERI members, full integral sets; PySCF's density as a fixed point; all four J/K members at it. |
| `lock_ccpvdz.viv` | Water cc-pVDZ (6D, 5D), def2-SVP (6D, 5D), def2-TZVP (5D), against checksums, the pair diagonal and 256 samples. |
| `lock_breadth.viv` | NH3, CH4, N2 in 6-31G*, 6D and 5D, against the sampled oracles. |
| `lock_open.viv` | OH and O2 in 6-31G*: UHF and ROHF at PySCF's states; the stable O2 ROHF state through the fixed-point import of PySCF's density. |
| `meta_integrals.viv` | Seeds 1-32 of random bases (l <= 3, exponents 0.05-50): translation, rotation, permutation, scaling, `T = -1/2 Laplacian_B S`, linearity of V in Z, positivity, Schwarz monotonicity, member agreement; 40-digit mpmath references. |
| `meta_scf.viv` | Seeds 1-32 of random O/H molecules: E invariant under rotation, translation and permutation; closed-shell UHF = RHF; spin flip; E_UHF <= E_ROHF; coupling and orthogonalizer independence; He2 at 20 bohr and N2 x N2 at 200 bohr. |
| `coverage.viv` | Per stage and per input class, the members that agree with the consensus; every class with at least 2 (all 55 ERI classes with l <= 3 by MD, HGP and Rys); a planted 1e-10 ERI error blamed on its member and entry. |
| `story_water.viv` | One RHF run on water 6-31G from the core guess, for `viv story scf/story_water.viv`: the SCF told in the words of `lib/hf/*.intent`, one line per iteration with the energy, its change, the commutator norm and the DIIS history size (`lib/hf/scf_loop.debug`). |
| `molecule_packs.viv` | Every molecule of the ladder built from the lib/basis tables against PySCF's packs in lib/hfref: atoms, shells and exponents bit for bit, normalized coefficients within 1e-14; water's RHF and H2O+ ROHF energies from the tables. |
| `legacy_bridge.viv` | `lib/gto_md` wrapped as extra registry members: integrals within 1e-12 and the legacy RHF within 1e-10 of `lib/hf`, water STO-3G and 6-31G. |

Every energy is from integrals computed by `lib/hf`, never from the stored
ones, and is held within 1e-9 of PySCF 2.5.0 (`conv_tol` 1e-12):

| System | 6D (Cartesian) | 5D (spherical) | Program |
|---|---|---|---|
| Water STO-3G RHF | -74.9630631297292 | | lock_water |
| Water 6-31G RHF | -75.98394849810573 | | lock_water |
| H2O+ 6-31G ROHF | -75.57839637958979 | | lock_water |
| Water spin -2 6-31G UHF / ROHF | -75.72639690903668 / -75.72365493633157 | | lock_water |
| Water 6-31G* RHF | -76.01049616963635 | -76.0090991085974 | lock_631gs |
| HF 6-31G* RHF | -100.00286265223036 | -100.000746998235 | lock_631gs |
| Water cc-pVDZ RHF | -76.027107008870573 | -76.026765673119627 | lock_ccpvdz |
| Water def2-SVP RHF | -75.96221113003139 | -75.96097516698583 | lock_ccpvdz |
| Water def2-TZVP RHF | | -76.05899070651952 | lock_ccpvdz |
| NH3 6-31G* RHF | -56.183883221995956 | -56.183242048528086 | lock_breadth |
| CH4 6-31G* RHF | -40.19514034833892 | -40.194810317717504 | lock_breadth |
| N2 6-31G* RHF | -108.94266234798268 | -108.94186885969204 | lock_breadth |
| OH 6-31G* UHF | -75.38214937611997 | -75.38093753158927 | lock_open |
| OH 6-31G* ROHF | -75.3782251872344 | -75.37702555164822 | lock_open |
| O2 6-31G* UHF | | -149.61231729065165 | lock_open |
| O2 6-31G* ROHF, default / stable | | -149.59185715618136 / -149.59202183229573 | lock_open |

## Running the checks

    scf/check.sh quick      lib/hf/{integrals,hf,fuzz}, their tests, and scf/{lock_water,meta_integrals,meta_scf,coverage}
    scf/check.sh full       every lib/hf module, every lib/hf/tests file and every scf program

A module's tests that compare it with PySCF's numbers (lib/hfref) or run the
sweep drivers live in `lib/hf/tests/<module>.viv`, so that a program using
the module loads neither; its other tests stay beside its code.

Each program goes through `tools/both-engines.sh` (interpreter and native
build, outputs compared line for line), in parallel, `JOBS` at a time
(default: the CPU count). The report is one line per program. On a 12-core
machine the quick tier took 2 min 30 s of wall time. One program at a time:

    tools/both-engines.sh scf lock_open
    tools/race.sh bench/hf_scf_bench.viv     native water cc-pVDZ 5D energy, gate 1.0 s

The interpreter stops a program after 2e9 steps (`STEP_BUDGET` in
`lang/viv/src/interp.rs`), so every program here is sized to stay under it:
the metamorphic suites use small bases (three shells, up to two primitives)
and production members for the SCF runs, and each seed's work is done once
per program and read by the tests.

## Regenerating the references

    python3 tools/hf_reference.py            every lib/hfref system module
    python3 tools/hf_reference.py o2_631gs   one module
    python3 tools/boys_reference.py          lib/hfref/boys.viv (mpmath)
    python3 tools/pyscf_suite.py             lib/hfref/pyscf_tests.viv

They need PySCF 2.5.0 (and mpmath for the Boys table). Nobody edits a
generated module by hand. Two sets of reference values live in the
programs instead: the stable O2 6-31G* ROHF densities in `lock_open.viv`
(the PySCF settings are in the comment above them), and the 40-digit
mpmath values in `meta_integrals.viv`.

## What the random bases showed

On molecular bases every integral member meets the 1e-12 contract. On the
random bases of `meta_integrals` (f functions, exponents near 50, centers 1
to 4 bohr apart) McMurchie-Davidson stays within 2e-16 of 40-digit mpmath
values, while Obara-Saika's V is off by up to 2.2e-12, HGP by up to 5.4e-12
and Rys by up to 3.4e-11. The member sweep there holds the members to 1e-11,
and Rys is swept class by class in `coverage.viv` on a milder basis
(exponents 1.3 and 0.35), where all three families agree within 1e-12.
