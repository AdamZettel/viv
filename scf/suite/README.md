# PySCF suite replay

Each program here replays a group of the assertions in PySCF v2.5.0's own
unit tests, harvested into `tools/pyscf_suite.tsv` and, as Viv constants,
`lib/hfref/pyscf_tests.viv`. The integrals are computed from the basis packs
and the SCF runs on them through `lib/hf/hf`. The only numbers taken from
PySCF are the rows' targets (and, for the Schwarz row, the two sides it
compares).

    tools/hf_score.sh          run every program in both engines, then score
    tools/hf_score.sh -v       also list every row

| Program | Rows |
|---|---|
| `h2o.viv` | test_h2o.py: water 6-31G RHF, UHF, H2O+ ROHF, the core guess, E_nuc |
| `small.viv` | He and H atoms, He2+, the atom-free model, (11\|11), a fractional charge, cart2sph factors, the Schwarz row |
| `states.viv` | stretched water (the saddle-point default state), spin -2 water, follow_state, remove_linear_dep_, a hand-set occupation |
| `veff.viv` | get_veff on a stack of random densities (the `+ 0j` rows) |
| `vdz_rhf.viv` | water cc-pVDZ RHF: J/K members, zero start, gradient, Cartesian, one electron |
| `vdz_mats.viv` | J, K, V, E and the level shift of numpy's random matrices |
| `vdz_uhf.viv` | water cc-pVDZ UHF on the closed shell, Cartesian UHF |
| `vdz_open.viv` | H2O+ ROHF, the triplet's gradient and determinant overlap |
| `vdz_direct_r/u/ro.viv` | the max_memory=0 rows, with the direct J/K member |
| `n2.viv` | N2 and N2+ in cc-pVDZ |
| `ghost_intor.viv` | a ghost atom, benzene overlaps, He-Ne packed ERIs |
| `native/bz.viv` | benzene cc-pVDZ (114 AOs), native only, timed, not gated |

`ref/stb.viv` holds the basis packs lib/hfref does not carry; it is written
by `python3 scf/suite/gen_ref.py` (geometry and basis only). `ref/sts.viv`
holds the row scorer, numpy's MT19937 stream (so the tests' random matrices
are drawn entry for entry) and small linear-algebra helpers.

Rows tagged `path` in the manifest depend on PySCF's iteration path (EDIIS,
a rollback, a dynamic level shift, max_cycle=0 from the minao guess); they
are listed apart and never counted as failures. `sts_line` prints the
decade of each gap, not the value, so the output of the two engines stays
comparable byte for byte.
