# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Plant one bug at a time in the catalog's Viv kernels and see which check
finds it: the pandas reference, the point-in-time check, the parity check.

    VIV=../target/release/viv python3 mutants.py

Each mutant is a one-line edit of a kernel, the kind of slip a person or an
assistant makes. The table this prints is in README.md.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402

import catalog  # noqa: E402
import vipyr  # noqa: E402
from vipyr.pit import _equal  # noqa: E402

# (entry, file, text, replacement, what the slip is)
MUTANTS = [
    ("velocity", "velocity.viv", "while ts[left_1h] < t - HOUR", "while ts[left_1h] <= t - HOUR",
     "1-hour window open on the left: a row exactly 1 hour back is dropped"),
    ("velocity", "velocity.viv", "n1[k] = i - left_1h", "n1[k] = same - left_1h - 1",
     "the other rows at the same instant are counted"),
    ("ewma", "ewma.viv", "                m = 0.0 / 0.0\n                if w > 0.0 {",
     "                for q from i to same {\n                    s += value[q]\n                    w += 1.0\n"
     "                }\n                m = 0.0 / 0.0\n                if w > 0.0 {",
     "the row's own value is in its mean"),
    ("ewma", "ewma.viv", "if reset[k] != 0 {\n                        cleared = true",
     "if reset[k] != 0 {\n                        cleared = true\n                        mean[k] = 0.0 / 0.0",
     "a reset row reads its own reset"),
    ("since_last", "since_last.viv", "                a = never\n                if have_hit {",
     "                for q from i to same {\n                    if hit[q] != 0 {\n"
     "                        have_hit = true\n                        last_hit = t\n                    }\n"
     "                }\n                a = never\n                if have_hit {",
     "an event of the type counts for itself (0 seconds since the last one)"),
    ("delinquency", "delinquency.viv", "                if head < len(due) {\n                    now = (t - due[head]) // DAY",
     "                pre: Int = 0\n                for q from i to same {\n                    pre += paid[q]\n"
     "                }\n                if head < len(due) and pre == 0 {\n                    now = (t - due[head]) // DAY",
     "a payment at the row's instant clears its own days past due"),
    ("streaks", "streaks.viv", "lo = d - WINDOW_DAYS + 1", "lo = d - WINDOW_DAYS",
     "the 90-day window holds 91 days"),
    ("asof", "asof.viv", "(strict == 0 and right.ts[j] == t)", "(strict == 0 and right.ts[j] <= t + 1000000000)",
     "the join reads right rows up to one second after the left row"),
]


def run(n=4000, seed=7, verbose=True):
    rows = []
    for name, fname, old, new, what in MUTANTS:
        tmp = Path(tempfile.mkdtemp(prefix="viv-mutant-"))
        for f in catalog.HERE.glob("*.viv"):
            shutil.copy(f, tmp / f.name)
        src = (tmp / fname).read_text()
        assert old in src, "mutant %r: text not found in %s" % (what, fname)
        (tmp / fname).write_text(src.replace(old, new, 1))
        mod = fname[:-4]
        saved = catalog._mods.get(mod)
        catalog._mods[mod] = vipyr.load(tmp / fname)
        try:
            e = catalog.ENTRIES[name]
            df = e.make(n, seed)
            got, ref = e.fn(df), e.reference(df)
            differ = np.zeros(len(df), bool)
            for c in ref.columns:
                x, y = np.asarray(got[c].to_numpy()), np.asarray(ref[c].to_numpy())
                if x.dtype.kind == "M":
                    x, y = x.view(np.int64), y.astype("datetime64[ns]").view(np.int64)
                if e.exact:
                    differ |= ~_equal(x, y)
                else:
                    differ |= ~(np.isclose(x, y, rtol=1e-13, atol=0) | (np.isnan(x) & np.isnan(y)))
            ref_rows = int(differ.sum())
            try:
                vipyr.check_point_in_time(e.fn, df, by=e.by, ts=e.ts, outcomes=e.outcomes, cuts=5)
                pit = "passes"
            except vipyr.PointInTimeError:
                pit = "FAILS"
            try:
                vipyr.check_parity(e.fn, df, by=e.by, ts=e.ts, outcomes=e.outcomes, lookback=e.lookback,
                                     requests=None)
                par = "passes"
            except vipyr.ParityError:
                par = "FAILS"
        finally:
            catalog._mods[mod] = saved
            shutil.rmtree(tmp, ignore_errors=True)
        rows.append((name, what, ref_rows, len(df), pit, par))
        if verbose:
            print("| %s | %s | %s | %s | %s |" % (name, what,
                  "%d of %d rows differ" % (ref_rows, len(df)) if ref_rows else "equal", pit, par))
    return rows


if __name__ == "__main__":
    print("| entry | the slip | vs the reference | point-in-time | parity |")
    print("|---|---|---|---|---|")
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 4000)
