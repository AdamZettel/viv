# Third-party code and data

Viv itself is licensed under MIT OR Apache-2.0 ([LICENSE-MIT](LICENSE-MIT),
[LICENSE-APACHE](LICENSE-APACHE)); [LICENSING.md](LICENSING.md) explains the
scheme, and REUSE.toml records each file's license. The files below come from other projects, or
are derived from them, and keep their own licenses and notices. Nothing here is
under the GPL.

## Code in this repository

| path | what | origin | license | notice kept |
|---|---|---|---|---|
| `lang/viv/src/vmath/`, `lang/viv/src/vmath.c` | Viv's libm (sin, exp, pow, ...) | the `libm` crate 0.2.8, a port of musl's libm, from FreeBSD msun and Sun's fdlibm | MIT OR Apache-2.0 (libm crate); fdlibm's permissive notice | yes: the Sun Microsystems notice heads each function taken from fdlibm; the rest (cosh, sinh, tanh, hypot, scalbn, expo2) are musl's own code, MIT |
| `lib/minpack/ref/` | MINPACK Fortran (lmdif, lmder, qrfac, ...), the oracle for the Viv port in `lib/minpack/` | MINPACK, Argonne National Laboratory, via netlib | Minpack Copyright Notice (1999), University of Chicago (BSD-style) | yes: `lib/minpack/ref/disclaimer` |
| `lib/minpack/` (`enorm`, `fit`, `lmder`, `lmpar`, `qrfac`, `qrsolv`, `mgh`, `mgh_ref` and their `.intent`/`.debug` files), `kernels/k1/norms.viv` | the Viv port of MINPACK's lmder and the routines it calls (the Fortran's operations in the Fortran's order), the MGH test problems of `file17.f` and the netlib reference results; `kernels/k1/norms.viv` adapts `enorm` | MINPACK, J. J. Moré, B. S. Garbow and K. E. Hillstrom, Argonne National Laboratory, via netlib | Minpack Copyright Notice (1999), University of Chicago (`LicenseRef-Minpack`) | yes: the SPDX lines of each file and REUSE.toml; the acknowledgment below |
| `xc/` (the generated `xc/lda/*.viv`, `xc/gga/*.viv`, `xc/catalog.json`, `xc/tests/exact.json`) | viv-xc: exchange-correlation functionals translated from Libxc's Maple scripts by `tools/vivxc/` | Libxc 6.1.0, M.A.L. Marques, S. Lehtola et al. (`maple/**/*.mpl`; parameter values from its `src/*.c`) | MPL-2.0 (`xc/LICENSE`); `xc/core.viv`, `xc/tests/oracle.py` and `tools/vivxc/` are MIT OR Apache-2.0 | yes: every generated file names its Libxc source file and version; `xc/README.md` gives the citation |
| `study/benchmarks/upstream/Microbenchmarks/` (the license only), `study/benchmarks/viv/micro/` | the Julia micro benchmarks, ported to Viv | JuliaLang/Microbenchmarks @ 7da11c8 | MIT | yes: `LICENSE.md` |
| `study/benchmarks/upstream/benchmarksgame/` (the licenses only), `study/benchmarks/viv/bg/` | Benchmarks Game programs, ported to Viv | benchmarksgame-team (salsa.debian.org) @ 4029666 | Revised BSD (3-clause) | yes: `LICENSE.md`, `LICENSE-programs` |
| `study/benchmarks/upstream/numpy-benchmarks/` (the license and the 12 kernels the site quotes), `study/benchmarks/viv/npb/` | numpy-benchmarks kernels, and their translations to Viv by py2viv | serge-sans-paille/numpy-benchmarks @ deef91c | BSD 3-clause | yes: `LICENSE` |
| `study/benchmarks/viv/nas/` | NAS Parallel Benchmarks 3.4.3 (EP, IS, CG, MG, FT), ported to Viv | NASA Advanced Supercomputing Division | the NPB notice (`LicenseRef-NPB`) | the upstream sources are not distributed here |
| `study/benchmarks/upstream/nas/NPB-CPP/` (the license only) | a C++ port of NPB, kept for the record | GMAP @ 5bc1e2c | MIT | yes: `LICENSE.md` |
| `study/benchmarks/upstream/polybench-c-4.2.1/` (the license only), `study/benchmarks/viv/polybench/` | PolyBench/C 4.2.1, ported to Viv | Ohio State University | OSU Software Distribution License (`LicenseRef-PolyBench-OSU`) | yes: `LICENSE.txt` |
| `study/benchmarks/upstream/polybench-python/` (the license only) | PolyBench/Python | UDC-GAC @ ac77577 | Apache-2.0 | yes: `LICENSE` |
| `study/benchmarks/upstream/PolyBench.jl/` (the license only) | PolyBench.jl | M. Reisinger @ a13e08d | MIT | yes: `LICENSE.md` |
| `study/ports/pyscf/upstream/` | PySCF's `LICENSE` and `NOTICE` only | PySCF 2.5.0 | Apache-2.0 | yes. No PySCF source is copied: `lib/hf` is written independently and PySCF runs as an external oracle |
| `study/ports/sisso/sisso.viv`, `python/vipyr/sisso.py` | SISSO regression (feature construction, SIS, L0), ported to Viv; the Fortran is the oracle of `study/ports/sisso/compare_fortran.py` | SISSO 3.5, Runhai Ouyang, github.com/rouyang2017/SISSO @ bc18cae; Ouyang et al., Phys. Rev. Materials 2, 083802 (2018) | Apache-2.0 | yes: `study/ports/sisso/upstream/LICENSE` (the repository has no NOTICE); the derived file names its source routines and states that it was changed |
| `lib/nuclear/shell/` (`shell.viv`, `shellmodel.viv`, `shellmodel.export.toml`, `bigstick_compare.py`) | nuclear shell model (M-scheme configuration interaction) in Viv; no Fortran is copied, the design follows BIGSTICK and BIGSTICK is the oracle | BIGSTICK 8.0.0, C. W. Johnson, W. E. Ormand, P. G. Krastev et al. (github.com/cwjsdsu/BigstickPublick @ 63e16b5) | MIT | yes: the BIGSTICK notice heads `shell.viv`; `shellmodel.viv`, the manifest and `bigstick_compare.py` name it |
| `www/public/brand/vipyr/` (the vipyr lockup SVGs and PNGs) | the wordmark "vipyr": glyphs of Outfit SemiBold converted to outlines; no font file is shipped | Outfit, The Outfit Project Authors (github.com/Outfitio/Outfit-Fonts) | SIL Open Font License 1.1 | yes: `www/public/brand/vipyr/FONT-LICENSE-Outfit-OFL.txt` |
| `lib/cc/`, `tests/cc/` | a C compiler written in Viv (lexer, parser, checks, RV32IM code generator); no C is copied: the order of features, the grammar and the stack-machine code generation follow chibicc, and the early tests are chibicc's | chibicc, Rui Ueyama, github.com/rui314/chibicc (commits 0522e2d .. aacc0cf, 6c0a429, daa7398, 6b88bcb, f30f781) | MIT | yes: the MIT notice through REUSE.toml and each file's SPDX lines |
| `lib/rv32/tests/riscv/` | the riscv-tests rv32ui and rv32um suites, preprocessed into Viv programs by `tools/rv32/riscv_tests.py` | riscv-software-src/riscv-tests @ 7b68f7b, The Regents of the University of California | BSD 3-clause | yes: the copyright line and license in each generated file's SPDX lines |
| `study/idioms/micrograd/` (`micrograd.viv`, `micrograd_ad.viv`, `hand/engine.viv`, `moons.viv`) | micrograd's engine and MLP ported to Viv, and the moons data its demo trains on | karpathy/micrograd @ 7bc720e | MIT, 2020 Andrej Karpathy | yes: each file's SPDX lines and REUSE.toml |
| `study/idioms/makemore/` (`makemore.viv`, `makemore_ad.viv`, `hand/backward.viv`, `names_subset.viv`) | makemore's bigram and MLP models ported to Viv, and the first 300 lines of its `names.txt` | karpathy/makemore @ 988aa59 | MIT, 2022 Andrej Karpathy | yes: each file's SPDX lines and REUSE.toml |
| `study/idioms/thealgorithms-python/` (`backtracking`, `dp`, `graphs`, `maths`, `sorting`, `strings`, `structures`) | algorithms from TheAlgorithms/Python ported to Viv | TheAlgorithms/Python @ b3233c9 | MIT, 2016-2022 TheAlgorithms and contributors | yes: each file's SPDX lines and REUSE.toml |
| `study/idioms/ddpm/` (`ddpm.viv`, `ddpm_ad.viv`, `hand/backward.viv`) | labml's annotated DDPM (`labml_nn/diffusion/ddpm`) ported to Viv | labmlai/annotated_deep_learning_paper_implementations @ 33ab022 | MIT, 2020 Varuna Jayasiri | yes: each file's SPDX lines and REUSE.toml |
| `study/idioms/cleanrl-ppo/` (`ppo.viv`, `ppo_ad.viv`, `hand/backward.viv`) | CleanRL's PPO for CartPole (`cleanrl/ppo.py`) ported to Viv | vwxyzjn/cleanrl @ fe8d8a0 | MIT, 2019 CleanRL developers | yes: each file's SPDX lines and REUSE.toml |
| `third_party/sqlite/` | SQLite 3.53.4 amalgamation (`sqlite3.c`, `sqlite3.h`), linked into the `viv` binary and compiled beside an exported program that uses the `db_*` builtins | sqlite.org, `sqlite-amalgamation-3530400.zip` (SHA3-256 in the README) | public domain | yes: the dedication heads `sqlite3.c`; `third_party/sqlite/README.md` |
| `lang/viv/src/mathtext_metrics.rs`, `www/src/lib/mathtextMetrics.js` | glyph metrics (depth, height, italic correction, skew, width) of the KaTeX fonts, for the math in figure labels; generated by `tools/mathtext/gen_metrics.mjs` | KaTeX 0.18.7 `src/fontMetricsData.js`, Khan Academy and other contributors | MIT | yes: the header of each file |
| `lang/viv/atlas/viewer.js`, `lang/viv/atlas/viewer.css` | the viewer `viv atlas` writes into each page, bundled by `www/scripts/atlas-viewer-build.mjs`: React and React DOM, and KaTeX with its stylesheet and its fonts as woff2 | React 19, Meta Platforms, Inc. and affiliates; KaTeX 0.18.7, Khan Academy and other contributors | MIT | yes: React's `@license` comments at the end of viewer.js; KaTeX's copyright in the banner of viewer.js and above its rules in viewer.css |

This product includes software developed by the University of Chicago, as
Operator of Argonne National Laboratory (MINPACK, in `lib/minpack/`).

## Data

| path | what | origin | license or status |
|---|---|---|---|
| `study/idioms/cnn/digits.viv` | the first 1,536 images of scikit-learn's `load_digits` | the UCI Optical Recognition of Handwritten Digits data set, E. Alpaydin and C. Kaynak (1998), doi:10.24432/C50P49 | CC BY 4.0 (`LICENSES/CC-BY-4.0.txt`); the pixels are divided by 16 |
| `lib/basis/*.viv` | Gaussian basis sets (STO-3G, 6-31G family, cc-pVDZ, cc-pVTZ, ...) | Basis Set Exchange, via `tools/basis_tables.py` | published scientific data; BSE asks that the basis set's original paper be cited |
| `dft/*.viv` pseudopotential arrays, `dft/pseudo/make_pseudo.py` | GTH pseudopotential parameters | CP2K's `GTH_POTENTIALS` table as PySCF ships it; Goedecker, Teter and Hutter (1996), Hartwigsen, Goedecker and Hutter (1998) | published parameters; only the numbers are used, no CP2K or PySCF code |
| `engines/examples/*.cif`, `*.poscar`, `*.xyz` | crystal and molecule geometries (Si, GaAs, water, ...) | standard textbook structures | facts, no license |
| `lib/nuclear/shell/data/sd.sps`, `usdb.int`, and their text in `interactions.viv` | the sd shell and the USDB interaction | BIGSTICK's `examples/sd` (MIT, notice in `interactions.viv`); Brown and Richter, Physical Review C 74, 034315 (2006) | published parameters; MIT for the files as BIGSTICK ships them |
| `lib/nuclear/shell/data/ckpot.int`, `p.sps`, and their text in `interactions.viv` | the p shell and the Cohen-Kurath (8-16)POT interaction, in isospin form | S. Cohen and D. Kurath, Nucl. Phys. 73, 1 (1965); the numbers read from KSHELL's `ckpot.snt` and converted by `data/snt_to_iso.py` | published parameters; only the numbers are used, no KSHELL code |

## Reference programs that run outside the repository

The idiom studies under `study/idioms/` compare a Viv port with an original
program. The originals (karpathy/micrograd, karpathy/makemore,
KellerJordan/modded-nanogpt, CleanRL, labml's annotated DDPM,
pytorch/examples, TheAlgorithms/Python, TheAlgorithms/Julia) are imported from
local clones at run time and are not copied here; the `ref*.py` and `ref.jl`
drivers in this repository are new code. The Viv ports of five of them are
listed in the table above with their upstream notices. `nanogpt-speedrun/reference.py` is a
minimal float64 rewrite of parts of modded-nanogpt's `train_gpt.py` (MIT).

## Site dependencies

The website's npm dependencies are listed in `www/package-lock.json`. One of
them, `sharp` (used by Astro for image processing at build time), installs
platform binaries of libvips under LGPL-3.0-or-later. They run only during the
site build and are not part of the published site or of the compiler.
