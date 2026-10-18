# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Timing for gallery/data/plot_feature_catalog.py: three catalog features on
a synthetic event log, vipyr against the pandas reference (a groupby().apply()
with a Python loop per row, catalog/catalog.py). Warm median of 3 runs for
vipyr, one run for pandas. Writes gallery/data/features_bench.json; the
gallery page prints that file, since a page must print the same thing on
every run.

    VIV=target/release/viv PYTHONPATH=python:catalog python3 gallery/data/bench_features.py
"""
import json
import os
import platform
import statistics
import subprocess
import time
import warnings

warnings.filterwarnings("ignore")
import catalog  # noqa: E402
import pandas as pd  # noqa: E402
import vipyr as vp  # noqa: E402

N = 200_000
ev = catalog.make_events(N, seed=1)
acc = catalog.make_accounts(N // 4, seed=1)
out = {"rows_events": len(ev), "rows_accounts": len(acc), "features": {}}
for name, df, ref in (
    ("sessions", ev, catalog.sessions_reference),
    ("velocity", ev, catalog.velocity_reference),
    ("delinquency", acc, catalog.delinquency_reference),
):
    fn = getattr(catalog, name)
    fn(df)
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        fn(df)
        ts.append(time.perf_counter() - t)
    t = time.perf_counter()
    ref(df)
    tp = time.perf_counter() - t
    out["features"][name] = {
        "vipyr_s": round(statistics.median(ts), 4),
        "pandas_s": round(tp, 3),
    }
    print(name, out["features"][name])
chip = subprocess.run(
    ["sysctl", "-n", "machdep.cpu.brand_string"],
    capture_output=True,
    text=True,
).stdout.strip()
out["machine"] = chip or platform.processor()
out["load_average"] = round(os.getloadavg()[0], 1)
out["versions"] = "pandas %s, Python %s" % (
    pd.__version__,
    platform.python_version(),
)
json.dump(out, open("gallery/data/features_bench.json", "w"), indent=1)
