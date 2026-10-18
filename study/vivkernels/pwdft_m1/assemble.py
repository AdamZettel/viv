# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""PROTOTYPE: turn the C that main's Viv
compiler writes for the Si program into one self-contained pwdft.c.

    python3 assemble.py program.c pwdft.c

It does by text what `viv export` would emit:
  1. each parameter's top-level assignment in main (`    ECUT = 10.0;`) is
     replaced by a read of the parameter struct (`    ECUT = vk_param.ecut;`);
     the line must occur exactly once, or this script stops;
  2. prelude.c (with pwdft.h pasted in) goes before the program and entry.c
     after it, so the result is one file that `cc -shared` builds with nothing
     but libc.
"""
import re, sys
from pathlib import Path

PARAMS = {"ECUT": "ecut", "NKMESH": "nkmesh", "ALAT": "alat"}

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
here = Path(__file__).resolve().parent
c = src.read_text()
for name, field in PARAMS.items():
    pat = re.compile(r"^    %s = [-0-9.e()]+;$" % name, re.M)
    hits = pat.findall(c)
    if len(hits) != 1:
        sys.exit(f"assemble.py: expected one top-level assignment of {name}, found {len(hits)}")
    c = pat.sub(f"    {name} = vk_param.{field};", c)
out = [
    "/* PROTOTYPE pwdft.c: assembled by study/vivkernels/pwdft_m1/assemble.py from\n"
    "   prelude.c, the C Viv wrote for the Si program, and entry.c. */\n",
    (here / "prelude.c").read_text().replace('#include "pwdft.h"\n', (here / "pwdft.h").read_text()),
    "\n/* ---- the program, as the Viv compiler wrote it (parameters patched) ---- */\n",
    c,
    "\n/* ---- the exported entry ---- */\n",
    (here / "entry.c").read_text(),
]
dst.write_text("".join(out))
