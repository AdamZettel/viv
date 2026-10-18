# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
Customer features from a transaction log
========================================

Compute per-customer features with state from an event log with the feature
catalog in `vp.features`: sessions (a new one after 30 minutes of silence),
transaction velocity (counts and sums over the last hour and day), and days
past due on a billing account. Each feature is a Viv function called on a
pandas DataFrame, reads only rows before the one it is computed for, and
equals a pandas reference written as a Python loop per row.
"""

# %%
# ## The data
#
# Synthetic, from `catalog/catalog.py`: about 22,000 events of 333 users over
# 120 days (`user`, `ts`, `amount`, `merchant`), with bursts of activity, and
# about 5,000 bill and payment rows of billing accounts (`account`, `ts`,
# `billed`, `paid`), some paid late. Rows may come in any order.

import json
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import vipyr as vp
from vipyr.figure import figure

sys.path.insert(0, "catalog")
import catalog

ev = catalog.make_events(20000, seed=1)
acc = catalog.make_accounts(5000, seed=1)
print("events:", ev.shape, "users:", ev.user.nunique())
print(ev.head(3).to_string())
print("billing rows:", acc.shape, "accounts:", acc.account.nunique())

# %%
# ## Sessions and velocity
#
# `vp.features.sessions(df=...)` and `vp.features.velocity(df=...)` return
# one row of features per input row, in the input's order. `n_1h` counts the
# user's events in the hour before this one; `sum_24h` adds their amounts
# over the day before.

sess = vp.features.sessions(df=ev).frame
vel = vp.features.velocity(df=ev).frame
feats = ev.join(sess).join(vel)
print(
    feats[
        [
            "user",
            "ts",
            "amount",
            "session_index",
            "n_prev_in_session",
            "n_1h",
            "n_24h",
            "sum_24h",
        ]
    ]
    .iloc[100:106]
    .to_string()
)

per_user = feats.groupby("user").session_index.max() + 1
counts = np.bincount(feats.n_1h.to_numpy())
print("events with 0, 1, 2, ... earlier events in the hour:", counts.tolist())
fig = figure("Events in the hour before each event", "n_1h", "events")
fig.bars(np.arange(len(counts)), counts, "events")
fig.yscale("log")
hist, edges = np.histogram(per_user, bins=12)
fig = figure("Sessions per user over 120 days", "sessions", "users")
fig.bars(0.5 * (edges[1:] + edges[:-1]), hist, "users")

# %%
# ## One user's timeline
#
# The busiest user's events over the first 30 days: each dot is an event at
# its amount, and each square is `sum_24h` at that event, the user's
# spending in the 24 hours before it. It is zero after a quiet day and adds
# up inside a burst; the event's own amount is not in it.

u = feats.user.value_counts().idxmax()
one = feats[(feats.user == u)].sort_values("ts")
days = (one.ts - one.ts.min()).dt.total_seconds().to_numpy() / 86400
m = days < 30
print(
    "user %d: %d events, %d sessions"
    % (u, len(one), one.session_index.max() + 1)
)
fig = figure(
    "User %d: amounts and spending in the previous 24 hours" % u,
    "day",
    "amount",
)
fig.scatter(days[m], one.amount.to_numpy()[m], "event amount")
fig.plot(days[m], one.sum_24h.to_numpy()[m], "sum_24h")
fig.style("sum_24h", "s2 noline square")

# %%
# ## Days past due
#
# `vp.features.delinquency` pays the oldest open bill first and carries any
# credit forward, and at every row reports the days the oldest unpaid bill is
# overdue (`dpd`), the worst in the last 12 months, and whether the account
# was ever cured after being 60 or more days late. In this synthetic ledger
# a bill is due the moment it is issued, so every account is late at some
# point, and the 12-month worst is rarely under 30 days.

dq = acc.join(vp.features.delinquency(df=acc).frame)
worst = dq.groupby("account").worst_dpd_12m.max()
bins = [0, 1, 30, 60, 90, 10**9]
labels = ["current", "1-29", "30-59", "60-89", "90+"]
n_bin = np.histogram(worst, bins=bins)[0]
for lab, n in zip(labels, n_bin):
    print(
        "worst days past due in the last 12 months, %-7s %4d accounts"
        % (lab + ":", n)
    )
print(
    "accounts ever cured after 60+ days late:",
    int(dq.groupby("account").ever_cured_60.max().sum()),
)
fig = figure("Accounts by worst days past due", "", "accounts")
fig.bars(np.arange(5), n_bin, "accounts")
for k, lab in enumerate(labels):
    fig.xtick(lab, k)

# %%
# ## How it's checked
#
# Each feature against its pandas reference on these rows, value for value,
# and `vp.check_point_in_time`, which cuts the log at five random times,
# recomputes the features from the rows up to each cut alone, and fails if
# any row before the cut changes (outcome columns, such as a payment, are
# hidden at their own instant). The timing table is
# read from `gallery/data/features_bench.json`, written by
# `gallery/data/bench_features.py` on 200,000 events and 50,000 billing rows.

from vipyr.pit import _equal

for name, df, ours in (
    ("sessions", ev, sess),
    ("velocity", ev, vel),
    ("delinquency", acc, dq[["dpd", "worst_dpd_12m", "ever_cured_60"]]),
):
    ref = getattr(catalog, name + "_reference")(df)
    same = bool(
        np.all(_equal(ours.reset_index(drop=True), ref.reset_index(drop=True)))
    )
    print(
        "%-12s equal to the pandas reference on all %d rows: %s"
        % (name, len(df), same)
    )
for name, df in (("sessions", ev), ("velocity", ev), ("delinquency", acc)):
    e = catalog.ENTRIES[name]
    rep = vp.check_point_in_time(
        e.fn, df, by=e.by, ts="ts", outcomes=e.outcomes
    )
    print(
        "%-12s point in time: %d rows checked at %d cuts, no leak"
        % (name, rep["rows_checked"], len(rep["cuts"]))
    )
print("check():", vp.features.velocity(df=ev).check().passed)

bench = json.load(open("gallery/data/features_bench.json"))
print("\n%-12s %10s %10s %8s" % ("seconds", "vipyr", "pandas", "ratio"))
for name, b in bench["features"].items():
    print(
        "%-12s %10.4f %10.3f %7.0fx"
        % (name, b["vipyr_s"], b["pandas_s"], b["pandas_s"] / b["vipyr_s"])
    )
print(
    "%s rows of events; measured on an %s, load average %.0f; %s"
    % (
        bench["rows_events"],
        bench["machine"],
        bench["load_average"],
        bench["versions"],
    )
)
