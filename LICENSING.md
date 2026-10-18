<!--
SPDX-License-Identifier: MIT OR Apache-2.0
SPDX-FileCopyrightText: 2026 The Viv authors
-->
# Licensing

Viv is licensed under MIT OR Apache-2.0: you may use it under either one.
Some parts of the repository come from other projects or are translated from
them, and those keep their authors' licenses. This page says which parts
those are and what each license asks of you. [THIRD_PARTY.md](THIRD_PARTY.md)
lists every borrowed file with its origin.

## How each file says its license

The repository follows the [REUSE](https://reuse.software) convention:

- A source file that can hold a comment names its license and copyright in
  two lines: `SPDX-License-Identifier` and `SPDX-FileCopyrightText`. In most
  files they are the first lines. In a `.viv` file they are the last lines,
  so that line numbers in error messages, expected outputs and the docs stay
  what they were, and a program still opens with its own text.
- Files that cannot hold a comment (JSON, images, fonts, wasm, expected
  outputs), files a generator writes, and code copied unchanged from another
  project are covered by [REUSE.toml](REUSE.toml) instead.
- Each license's full text is in [LICENSES/](LICENSES/), once, named by its
  SPDX id.

`python3 tools/licenses/check.py` checks all of this: every tracked file has a
license and a copyright, every license named has its text in `LICENSES/`, and
no file is under the GPL. `tools/gate.sh premerge` runs it as `licenses`. To
give new files their header, run `python3 tools/licenses/add_headers.py`.

## What is licensed how

| Part | License | What it means for you |
|---|---|---|
| The compiler, the runtime, `lib/`, `python/vipyr`, the site, the tools | MIT OR Apache-2.0 | Use, change and ship it, in open or closed products. Keep the copyright notice. |
| `xc/` and the `vipyr-xc` package: exchange-correlation functionals translated from Libxc | MPL-2.0 | File-level copyleft. You can combine these files with code under any license. If you ship changed versions of these files, you publish those files' source under the MPL. |
| `ports/dftatom` (the `vipyr-dftatom` package) | MIT, copyright Ondřej Čertík, Jiří Vackář and John Pask | Like Viv's own MIT: keep the notice. |
| `lib/nuclear/shell` (shipped in `vipyr-nuclear`) | MIT, copyright Lawrence Livermore National Security and San Diego State University (BIGSTICK) | Keep the notice; `packages/vipyr-nuclear/NOTICE` carries it. |
| `lang/viv/src/vmath/`, `lang/viv/src/vmath.c` (Viv's libm) | MIT, plus Sun's fdlibm notice (`SunPro`) on the functions that come from fdlibm | Keep both notices. Every exported program includes this code, so its notice block lists it. |
| `third_party/sqlite/` | public domain (SQLite's blessing) | No conditions. |
| `lib/minpack/ref/` | the MINPACK license (University of Chicago) | Keep the notice. These Fortran files are the test oracle for the Viv port; they are not compiled into Viv. |
| The MINPACK port in `lib/minpack/` (`enorm`, `fit`, `lmder`, `lmpar`, `qrfac`, `qrsolv`, `mgh`, `mgh_ref`) and `kernels/k1/norms.viv` | the MINPACK license (University of Chicago) | BSD-style: keep the notice, and include the Argonne acknowledgment in the documentation of anything you ship with it (THIRD_PARTY.md has the sentence). `viv export` lists it in NOTICES. |
| `study/benchmarks/upstream/` | each suite's own license (MIT, BSD-3-Clause, Apache-2.0, the NAS and PolyBench notices) | Benchmark programs kept unchanged to compare against. They are not part of the compiler. |
| The site's fonts and the vipyr wordmark | SIL Open Font License 1.1 | The fonts can be used and shipped with anything; they cannot be sold on their own. |

No code in this repository is under the GPL, and the checker keeps it that
way.

## Packages and copyleft

Each Python distribution names its license in its `pyproject.toml` and ships
its license files in the wheel, under `<name>.dist-info/licenses/`:

- `vipyr`: MIT OR Apache-2.0. Its platform wheel carries the `viv` compiler,
  which links SQLite and Viv's libm, so the wheel also carries every text in
  `LICENSES/`, this page and THIRD_PARTY.md.
- `vipyr-chem`, `vipyr-plasma`, `vipyr-hardware`, `vipyr-symreg`:
  MIT OR Apache-2.0.
- `vipyr-nuclear`: MIT OR Apache-2.0, with BIGSTICK's MIT notice in `NOTICE`
  for the shell model.
- `vipyr-xc`: MPL-2.0. This is the one copyleft package. Installing it and
  calling it from your code puts no conditions on your code. If you change
  its files and ship them, you share those files' source.
- Ports (`ports/<name>`, published as `vipyr-<name>`): the upstream license.

## Ports carry the upstream license

A port translates another project's code into Viv, so it is a derived work
and keeps the original license. Each port directory holds the upstream
`LICENSE` unchanged, an `AUTHORS` file with the upstream authors first, an
`UPSTREAM.md` naming the commit it was ported from, and a `CITATION.cff`.
Each file of the port names the upstream license and both copyright holders
in its SPDX lines. Which licenses allow a port, and what to do with the others: GPL code is never translated,
only run as an outside oracle.

## Exported C

`viv export` (and `vipyr.compile`, which calls it) reads the SPDX lines of
every module that contributes code to a program, follows imports, and adds
the runtime and libm the C backend puts in every program. The exported C file
then starts with a comment that names each part, its license and its
copyright, and a `NOTICES` file is written next to it with the full text of
each license. A module without SPDX lines takes its license from the
REUSE.toml of the checkout it lives in; your own program, with no SPDX lines,
is listed as `NOASSERTION`. For example, exporting the dftatom port gives:

```c
/* SPDX-License-Identifier: MIT AND (MIT OR Apache-2.0) AND (MIT AND SunPro)
   SPDX-FileCopyrightText: 2006-2012 Ondřej Čertík, Jiří Vackář, John Pask
   SPDX-FileCopyrightText: 2026 The Viv authors
   SPDX-FileCopyrightText: 1993 Sun Microsystems, Inc.
   SPDX-FileCopyrightText: 2005-2020 Rich Felker et al. (musl)

   This file is compiled from these components; NOTICES, beside it, has
   the full text of each license.

     ports/dftatom/atom.viv
         MIT; 2006-2012 Ondřej Čertík, Jiří Vackář, John Pask; 2026 The Viv authors
     ...
     Viv's libm (lang/viv/src/vmath.c: fdlibm as musl carries it)
         MIT AND SunPro; 1993 Sun Microsystems, Inc.; ...
*/
```

If you ship the exported program, ship `NOTICES` with it (or keep the comment
in the C file); that satisfies the notice requirements of MIT, Apache-2.0 and
SunPro.

Two licenses change what the export does:

- **GPL or AGPL.** The export stops with an error naming the module. A GPL
  module would put the whole program under the GPL, and Viv's exports are
  meant to drop into any product.
- **MPL or LGPL.** The export goes ahead and prints a warning naming each such
  file. Whoever ships the program must keep those files' source available
  under their license, including any changes; the rest of the program keeps
  its own license. `vipyr.compile` turns the warning into a Python warning.

If the program uses the `db_*` builtins, `sqlite3.c` is copied next to the C
file, and SQLite's blessing is listed too.
