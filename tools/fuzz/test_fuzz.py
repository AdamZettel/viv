#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vivsmith's own tests: the generator is a function of its seed, its programs
pass the front end, and the reducer shrinks a failure while keeping it.

    python3 tools/fuzz/test_fuzz.py

The reducer is driven by a stand-in for run.check(): a program "fails" when the
interpreter runs it and it still prints the marker line, so the test needs only
the interpreter (target/release/viv, or VIV=...)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vivsmith  # noqa: E402
import reduce as red  # noqa: E402
from run import run, VIV, clean, first_error  # noqa: E402

bad = 0


def ok(what, cond):
    global bad
    print(("pass  " if cond else "FAIL  ") + what)
    bad += not cond


a, _ = vivsmith.generate(42)
b, _ = vivsmith.generate(42)
c, _ = vivsmith.generate(43)
ok("the same seed gives the same program", a == b)
ok("another seed gives another program", a != c)

import re  # noqa: E402
for seed in range(7, 40):
    # the first seed with a top-level Int to print
    src = vivsmith.generate(seed)[0].splitlines()
    var = next((m.group(1) for l in src for m in [re.match(r"^(v\d+): Int = ", l)] if m), None)
    if var:
        break
MARK = f'print("MARK", {var})'


def stub(src, engines=(), san=False):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "p.viv")
        open(p, "w").write(src)
        rc, out = run([VIV, "run", p], cwd=d)
        if rc != 0 or first_error(clean(out)):
            return "interp", first_error(out)
        return ("differs", "native: line 1") if "MARK" in out else (None, None)


if var:
    src.insert(len(src) - 1, MARK)
if MARK in src:
    r = red.Reducer("differs", "native: line 1", (), False, 400, src, checker=stub)
    small = r.reduce(src)
    ok(f"the reducer keeps the failure ({len(src)} -> {len(small)} lines, {r.tests} tests)", MARK in small)
    ok("the reduced program is much smaller", len(small) <= max(6, len(src) // 4))
else:
    ok("a seed below 40 declares a top-level Int (the reducer test needs one)", False)
sys.exit(1 if bad else 0)
