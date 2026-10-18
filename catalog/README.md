# The feature catalog

Per-entity features with state, each written once in Viv and called on a
pandas or polars DataFrame. Every entry comes with:

- a Viv function (`<entry>.viv`) whose `#|` lines and steps are its intent
  entry (`viv lens check catalog/*.viv` passes);
- an exact definition, below, that says which rows a feature reads: window
  edges, rows at the same instant, how far back it needs to look;
- a pandas reference in `catalog.py`, a `groupby().apply()` with a Python loop
  per row that states the definition and nothing else;
- tests in `python/tests/test_catalog.py`: equality with the reference on
  synthetic data, `vipyr.check_point_in_time`, and `vipyr.check_parity`.

```python
import catalog                         # catalog/catalog.py
feats = catalog.velocity(df)           # df: user, ts, amount, merchant; any row order
e = catalog.ENTRIES["velocity"]        # fn, reference, by, ts, outcomes, lookback, kind
```

The catalog functions take rows in any order. They sort into the canonical
order (key, ts, then the other input columns, so rows at the same instant are
always taken in one order), call the kernel and return the result in the
input's row order; a pandas result keeps the input's index. A frame already
in that order goes straight to the kernel. The kernels themselves also take a
frame directly when it is sorted by (key, ts):
`catalog.module("sessions").sessions(df, gap=...)`.

Timestamps reach Viv as int64 nanoseconds since the epoch, UTC. "Day" means a
UTC calendar day.

## The rule every feature follows

A feature at a row with time t reads only rows of the same key with a time
before t, plus the row's own time and the attributes known when the row
happens. Columns listed as `outcomes` (a payment amount, a reset flag) are
known only after the row's instant, so the row's own outcomes are not
inputs, and neither are those of other rows at the same instant.

## Entries

### 1. sessions (`sessions.viv`)

Input `user`, `ts`. Parameter `gap` (default 30 minutes).

A session is a run of a user's events in which no two consecutive events are
more than `gap` apart. A gap of exactly `gap` keeps the session open; a gap
longer than `gap` starts a new one. Rows at the same instant share a session.

| output | definition |
|---|---|
| `session_start` | time of the session's first event; with `user`, the session's id |
| `session_index` | number of the user's sessions that started before this one |
| `secs_since_start` | `(ts - session_start)` in seconds |
| `n_prev_in_session` | events of the session with a time before `ts` (same instant excluded) |

Serving needs the user's whole history in principle, since a session can stay
open indefinitely: lookback is unbounded. There is no global session number,
because one numbered across users changes when an earlier user gains a session
later (the point-in-time test shows the check catching it).

### 2. velocity (`velocity.viv`)

Input `user`, `ts`, `amount`, `merchant`. For a row at time t the window of
length W holds the user's rows j with

    t - W <= ts_j < t

The left edge is inclusive and the right edge exclusive: a row exactly W back
counts, the row itself and every row at the same instant do not.

| output | definition |
|---|---|
| `n_1h`, `n_24h` | rows in the 1-hour and 24-hour windows |
| `sum_24h` | `amount` summed over the 24-hour window, left to right in canonical order |
| `distinct_24h` | distinct merchants in the 24-hour window |

The sum is computed directly over the window for each instant, so it is the
same float a plain loop gives. Lookback: 24 hours, checked by the parity test
with a 24-hour serving store.

### 3. streaks (`streaks.viv`)

Input `user`, `ts`. A row's day is `floor(ts / 1 day)`; a day is active when
the user has an event on it. For a row on day D:

| output | definition |
|---|---|
| `streak_days` | consecutive active days ending on D, counting D (the row makes D active) |
| `longest_90d` | longest run of consecutive active days inside D-89 .. D; a run that began before D-89 is cut at D-89 |

Lookback: `longest_90d` needs 90 days, `streak_days` the whole run. The parity
test shows a 100-day streak served from a 90-day store coming out short.

### 4. gap-aware EWMA (`ewma.viv`)

Input `user`, `ts`, `value`, `reset`. Parameter `half_life` (default 1 day).
Outcomes: `value`, `reset`.

Each user carries a weighted sum S and weight W. Before the rows at instant t
are scored, both decay by `f = exp(-ln 2 * (t - t_prev) / half_life)`, with
t_prev the user's previous instant. The rows at t read

| output | definition |
|---|---|
| `ewma` | S / W; NaN when there is no value since the start or the last reset |
| `ewm_weight` | W, the decayed count of values behind the mean |

Then the instant updates the state: if any of its rows has `reset = 1`, S and
W are cleared, and every value at t is added with weight 1. A row never sees
its own value or its own reset. Lookback: unbounded.

Viv's `exp` and the platform's `math.exp` differ in the last bit for about 1
argument in 700, so this entry is compared with its reference to a relative
1e-13 (the largest difference seen is 3e-16). Its parity test is exact, as
for every entry: Viv against Viv.

### 5. time since last event of a type (`since_last.viv`)

Input `user`, `ts`, `kind`. Parameter `kind` (default `"purchase"`).

| output | definition |
|---|---|
| `secs_since_hit` | seconds since the user's latest event of that kind with a time before t; NaN if there is none ("never") |
| `secs_since_prev` | seconds since the user's latest event of any kind before t; NaN for the user's first instant |

Lookback: unbounded (the last event of a kind can be arbitrarily old).

### 6. delinquency state machine (`delinquency.viv`)

Input `account`, `ts`, `billed` (a new bill due at ts, cents), `paid` (a
payment at ts, cents). Outcomes: `billed`, `paid`.

At an instant the bills are added first, then the payments pay the oldest open
bills first; a payment beyond every open bill is kept as credit for the next
bills. `dpd(s) = floor((s - d) / 1 day)`, d the due time of the oldest bill
still open at s, or 0 when nothing is open. For a row at time t, before the
instant's own bills and payments:

| output | definition |
|---|---|
| `dpd` | dpd(t) |
| `worst_dpd_12m` | the largest dpd(s) over t - 365 days <= s <= t |
| `ever_cured_60` | 1 if at an earlier instant a payment took the account from dpd >= 60 to dpd = 0 |

The kernel keeps a queue of open bills. The reference is written another way,
from cumulative sums: with oldest-first payment and carried credit, bill k is
closed at time s exactly when the payments made by s add up to at least the
bills up to k. Lookback: unbounded (`ever_cured_60` is a lifetime flag).

### 7. as-of join (`asof.viv`)

Left `user`, `ts`; right `user`, `ts`, `value`. Option `strict`. For a left row
at time t the match is the right row of the same user with the largest time
such that

    ts_r <= t   (inclusive, the default)
    ts_r <  t   (strict)

Use strict when a right value stamped t is only published after t, like an
end-of-day figure. When several right rows share the matching time, the
largest value wins, so the answer does not depend on the right table's row
order.

| output | definition |
|---|---|
| `value` | the matched right value; NaN when there is none |
| `age_secs` | `t - ts_r` in seconds; NaN when there is none |

`catalog.asof(left, right, strict=False)` takes two frames. For the checks,
`catalog.asof_combined(left, right)` stacks both tables into one frame (a
`side` column tells them apart), so the point-in-time check alters right rows
after t as well as left ones, and the parity check serves each request from
both tables' history.

### 8. labels: churn (`labels.viv`), NOT A FEATURE

Input `user`, `ts`. Parameter `horizon` (default 14 days).

| output | definition |
|---|---|
| `churned` | 1.0 when the user has no event with t < ts_j <= t + horizon; 0.0 when it has one; NaN when t + horizon is past the last time in the data |

A model is trained to predict its label, so a label reads the future by
design. It is registered with `kind="label"`, and the tests require both
checks to refuse it: `check_point_in_time` raises `PointInTimeError` and
`check_parity` raises `ParityError`, each naming `churned`.

## The parity check

`vipyr.check_parity(fn, df, by, ts, outcomes=(), lookback=None,
requests=300, serve=None, batches=None)` (in `python/vipyr/parity.py`)
compares training with serving. Training runs `fn` on the whole frame. Serving
is replayed one request at a time, a request being one key at one instant t.
The serving store holds that key's rows with `ts < t`, or with
`t - lookback <= ts < t` when a lookback is given, plus the request's own rows
at t with their outcomes hidden (0/1 columns flipped, other numbers set to
0). `serve` (by default `fn`) computes the request rows' features from that
frame alone, and they must equal the training values bit for bit, NaN equal
to NaN. `batches="1D"` replays daily batch scoring instead.

What parity finds that the point-in-time check does not:

- an outcome that is not a 0/1 flag leaking into its own row (the
  point-in-time check only flips 0/1 outcomes);
- a feature that needs more history than the serving store keeps;
- a feature that reads other keys' rows;
- a leak a short time ahead. The point-in-time check tests a few cut times;
  parity checks every request instant it replays;
- a serving implementation that is different code from training (`serve=`),
  such as a window closed on the other side.

## What the checks find

`catalog/mutants.py` plants one slip at a time in a kernel and runs the three
checks (4,000 synthetic rows, every request replayed):

| entry | the slip | vs the reference | point-in-time | parity |
|---|---|---|---|---|
| velocity | 1-hour window open on the left: a row exactly 1 hour back is dropped | 24 of 4,456 rows differ | passes | passes |
| velocity | the other rows at the same instant are counted | 284 of 4,456 rows differ | passes | passes |
| ewma | the row's own value is in its mean | 4,456 of 4,456 rows differ | passes | FAILS |
| ewma | a reset row reads its own reset | 51 of 4,456 rows differ | FAILS | FAILS |
| since_last | an event of the kind counts for itself (0 seconds since the last one) | 489 of 4,456 rows differ | passes | passes |
| delinquency | a payment at the row's instant clears its own days past due | 3,500 of 4,018 rows differ | FAILS | FAILS |
| streaks | the 90-day window holds 91 days | 8 of 4,456 rows differ | passes | passes |
| asof | the join reads right rows up to one second after the left row | 98 of 6,456 rows differ | passes | FAILS |

Every slip is found by at least one check, and no single check finds them all.
Off-by-one window edges and same-instant mistakes do not read the future, so
only the reference finds them, and only if the data puts rows exactly on the
edges. The first version of the synthetic data did not: it had no row exactly
1 hour back and no streak long enough to cross the 90-day window, and the
reference passed both of those slips. `make_events` now puts 15% of burst rows
exactly on a window edge (30 minutes, 1 hour, 24 hours) or one second past it,
and gives one user in twenty a near-daily habit.

`test_every_planted_bug_is_found` runs this table on 2,500 rows.

## Speed

`catalog/bench.py` on about 10,000,000 rows. Viv runs on one core; polars
uses all of them.

Measured with the machine loaded by other jobs (load average 13 at the start,
29 at the end), so read these as rough. 10,949,762 rows, 166,666 users;
warm median of 3; the pandas reference timed on 109,585 rows and scaled.

| entry | Viv kernel, sorted frame | Viv, catalog wrapper | polars | pandas reference (scaled) |
|---|---|---|---|---|
| sessions | 0.51 s | 0.38 s | 7.3 s | 105 s |
| velocity | 0.54 s | 0.76 s | 1.7 s (no distinct count) | 108 s |
| asof (5,474,881 right rows) | n/a | 9.2 s | 2.0 s | 94 s |

The wrapper's time includes sorting into canonical order when the frame is
not in it (the as-of right table is sorted by value within an instant, a
three-key sort of 5 million rows) and putting the result back.

The polars versions do not give the reference's bits. polars divides a
column by a constant by multiplying with its reciprocal, so `secs_since_start`
and `age_secs` differ from `(t - s) / 1e9` in the last bit on some rows; the
rolling sum adds and subtracts as the window moves, so `sum_24h` drifts from
the sum taken over the window. polars has no rolling distinct count by time.
The counts, the session numbers and the matched values agree exactly.

## Running it

```sh
VIV=target/release/viv python3 -m pytest -q python/tests/test_catalog.py
cd catalog && VIV=../target/release/viv python3 mutants.py
cd catalog && VIV=../target/release/viv python3 bench.py 1000000
```
