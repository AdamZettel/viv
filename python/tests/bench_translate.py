# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vp.translate against the Python it came from, on 1e6 and 1e7 elements.

    VIV=target/release/viv python3 python/tests/bench_translate.py [1e6 1e7]

Each translated function is checked against the Python on its first call
(that call is not timed), then both are timed once more on the same input.
Prints a Markdown table: Python seconds, Viv seconds, speedup."""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "translate_cases"))
import vipyr as vp  # noqa: E402
import numeric as N  # noqa: E402


def cases(n):
    rng = np.random.default_rng(5)
    x = rng.random(n)
    keys = np.sort(rng.integers(0, max(1, n // 100), n))
    return [
        ("ewma (recurrence)", N.ewma, (x, 0.1)),
        ("count_above (int result)", N.count_above, (x, 0.5)),
        ("smooth (helper + constant)", N.smooth, (x,)),
        ("group_cumsum (sorted keys)", N.group_cumsum, (keys, x)),
        ("group_mean (sorted keys)", N.group_mean, (keys.astype(float), x)),
        ("cube_roots (Newton per row)", N.cube_roots, (x * 1000.0,)),
        ("poly_hash (int arithmetic)", N.poly_hash, (rng.integers(0, 1 << 20, n),)),
    ]


def main(sizes):
    print("| function | n | Python s | Viv s | speedup |")
    print("|---|---|---|---|---|")
    for n in sizes:
        for label, fn, args in cases(n):
            t = vp.translate(fn)
            t(*args)                                   # translate, compile, check
            t0 = time.perf_counter()
            fn(*args)
            tp = time.perf_counter() - t0
            t0 = time.perf_counter()
            t(*args)
            tv = time.perf_counter() - t0
            print("| %s | %.0e | %.3f | %.4f | %.0fx |" % (label, n, tp, tv, tp / tv), flush=True)


if __name__ == "__main__":
    main([int(float(a)) for a in sys.argv[1:]] or [10 ** 6, 10 ** 7])
