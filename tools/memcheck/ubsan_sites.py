#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Name the distinct UBSan findings of a run by the C function they are in.

    ubsan_sites.py prog.c stderr.txt

Prints `function: message` for each distinct report, joined by "; ". The
function is the nearest definition above the reported line of the generated
C (a line at column 0 with a parenthesis that ends in `{` or is followed by
a line starting with `{`), so a finding in
the runtime is named by its runtime function and one in the program by its
def (Viv defs keep their names in C, `main` is the top level).
"""
import re
import sys

lines = open(sys.argv[1]).read().split("\n")
seen = []
for m in re.finditer(r"\.c:(\d+):\d+: runtime error: (.*)", open(sys.argv[2], errors="replace").read()):
    ln = int(m.group(1))
    msg = re.sub(r"-?\d+(\.\d+)?(e[-+]?\d+)?", "N", m.group(2)).strip()
    fn = "?"
    for i in range(ln - 1, -1, -1):
        t = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if t and not t[0].isspace() and t[0] not in "#/*}" and "(" in t and (t.rstrip().endswith("{") or nxt.startswith("{")):
            head = t.split("(")[0].split()
            fn = head[-1].lstrip("*") if head else "?"
            break
    key = f"{fn}: {msg}"
    if key not in seen:
        seen.append(key)
print("; ".join(seen))
