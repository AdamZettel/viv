# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The installed-wheel check: run every vipyr example with no C compiler on
PATH, require that nothing is compiled (every library function comes from
the wheel's precompiled libraries), and that a user's own .viv runs in the
interpreter with the compiled build's bits. Run it from a directory outside
the checkout, with PATH holding no cc/clang/gcc and CC unset:

    env -i HOME=$HOME PATH=$VENV/bin:/bin $VENV/bin/python tools/wheels/smoke_nocc.py
"""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import time

import numpy as np

for c in ("cc", "clang", "gcc"):
    assert shutil.which(c) is None, "a C compiler is visible: %s" % shutil.which(c)
assert not os.environ.get("CC"), "CC is set"

import vipyr as vp
from vipyr import _core, _interp, _prebuilt

assert _core._repo_root() is None, "vipyr is running from a checkout, not an installed wheel"
fails = 0
for q, s in sorted(vp.functions(kind="built").items()):
    if not s.get("example"):
        continue
    t = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(s["example"], "<%s>" % q, "exec"), {"__name__": "__ex__"})
        print("ok   %-26s %6.2fs" % (q, time.time() - t))
    except Exception as e:
        fails += 1
        print("FAIL %-26s %s: %s" % (q, type(e).__name__, str(e)[:300]))
if "dftatom" in vp.families():
    r = vp.dftatom.atom(Z=6)
    print("ok   dftatom.atom(Z=6)          etot=%.10f" % r.etot)
print("compiles: %d   precompiled libraries used: %d" % (_core.stats["compiles"], _prebuilt.stats["prebuilt_hits"]))
if _core.stats["compiles"]:
    fails += 1

with tempfile.TemporaryDirectory() as d:
    src = os.path.join(d, "mine.viv")
    with open(src, "w") as f:
        f.write("def ewma(x: Arr, a: Num) -> Arr {\n    y = zeros(len(x))\n    s = x[0]\n    for i to len(x) {\n"
                "        s = a * x[i] + (1.0 - a) * s\n        y[i] = s\n    }\n    return y\n}\n"
                "def count_above(x: Arr, t: Num) -> Int {\n    n: Int = 0\n    for v in x {\n"
                "        if v > t {\n            n = n + 1\n        }\n    }\n    return n\n}\n")
    m = vp.load(src)
    x = np.random.default_rng(1).random(10000)
    ref = np.empty_like(x)
    s = x[0]
    for i in range(len(x)):
        s = 0.1 * x[i] + (1.0 - 0.1) * s
        ref[i] = s
    ok = np.array_equal(m.ewma(x, 0.1), ref) and m.count_above(x, 0.5) == int(np.sum(x > 0.5))
    try:
        import pandas as pd
        ok = ok and np.array_equal(np.asarray(m.ewma(pd.DataFrame({"x": x, "a": [0.1] * len(x)}).iloc[:, :1], a=0.1)), ref)
    except ImportError:
        pass
    print("%s interpreter fallback for a user's .viv (%d calls): results equal Python's bit for bit"
          % ("ok  " if ok else "FAIL", _interp.stats["interp_calls"]))
    fails += not ok
    # vp.translate from the wheel's py2viv, run by the interpreter
    py = os.path.join(d, "mine_py.py")
    with open(py, "w") as f:
        f.write("import numpy as np\n\ndef running_max(x):\n    out = np.zeros(len(x))\n    m = x[0]\n"
                "    for i in range(len(x)):\n        if x[i] > m:\n            m = x[i]\n        out[i] = m\n"
                "    return out\n")
    sys.path.insert(0, d)
    import mine_py
    t = vp.translate(mine_py.running_max)
    ok = np.array_equal(t(x[:2000]), mine_py.running_max(x[:2000]))
    print("%s vp.translate with the wheel's py2viv, in the interpreter: checked against Python"
          % ("ok  " if ok else "FAIL"))
    fails += not ok
sys.exit(1 if fails else 0)
