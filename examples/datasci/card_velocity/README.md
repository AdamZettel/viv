# Card velocity: a feature you can check

Two fraud features per card transaction, written once in Viv
(`features.viv`) and called on a pandas or polars DataFrame:

- `n_prev_1h`: how many transactions the same card made in the hour before
  this one. The current transaction does not count, and neither do others at
  the same instant.
- `status`: the card's state just before the transaction. `0` ok, `1` watch
  (3 or more declines in the last 24 hours), `2` locked (a chargeback; it
  clears after 7 days with no decline and no chargeback).

Both use only rows of the same card with an earlier timestamp. A
transaction's own outcome (declined, charged back) is not known when it is
scored, so it is not an input to its own features.

```python
import vipyr
k = vipyr.load("features.viv")
feats = k.card_features(df)          # df sorted by (card, ts); columns by name
```

`df` holds `card` (int64), `ts` (datetime64), `declined` and `chargeback`
(bool). The result is a DataFrame of the same library with the input's index.
`velocity.viv_features(df)` also accepts unsorted rows: it sorts them, calls
the kernel and puts the result back in input order.

## What is checked

**Parity with a plain reference.** `velocity.pandas_reference` is the
definition written as a `groupby("card").apply(...)` with a Python loop per
row: slow, and easy to read. On 1,000,000 synthetic transactions (20,000
cards, all three states present) Viv and the reference agree on every row of
both features.

**No reads from the future.** `vipyr.check_point_in_time(fn, df, by="card",
ts="ts", outcomes=["declined", "chargeback"])` picks cut times t and runs the
feature function on altered copies of the data:

- every row after t gets a new later timestamp and shuffled values;
- the outcomes of the rows at exactly t are flipped;
- the rows are put in a random order.

The features of every row at or before t must not change. The Viv features
pass on 1,018,320 row-features over 5 cuts.

**The version an assistant tends to write.** `velocity.polars_llm` is the
polars code a coding assistant commonly produces for this task: rolling
windows per card (`rolling_sum_by(...).over("card")`), with `- 1` to drop the
current row. It runs and looks right. Both checks reject it:

| check | result |
|---|---|
| parity, `n_prev_1h` | 392 of 1,000,000 rows differ: polars windows include the current instant and leave out a row exactly one hour back |
| parity, `status` | 23,431 rows differ: the 24-hour decline count and the 7-day lock include the current transaction's own outcome |
| point-in-time | fails, because a `status` of 0 became 2 when only the outcomes at t were flipped |

The point-in-time check needs no reference implementation. It finds the
outcome leak from the polars function alone. The `n_prev_1h` off-by-one does
not read the future, so only the parity check finds it.

## Running it

```sh
VIV=../../../target/release/viv python3 compare.py           # 1e6 parity, 1e7 timing
VIV=../../../target/release/viv python3 compare.py --quick   # 1e5 and 1e6
VIV=target/release/viv python3 -m pytest -q python/tests/test_frames.py   # from the repo root
```

The tests in `python/tests/test_frames.py` run the parity check on 1e6 rows,
the point-in-time check on Viv (it must pass) and on the polars version (it
must fail), and a set of features made leaky on purpose.

## Speed

On 10,000,000 rows, warm median of 5 runs, one core, load below 3, frame
conversion included:

| | time |
|---|---|
| Viv, pandas frame in and out | 0.22 s |
| Viv, polars frame in and out | 0.22 s |
| polars `rolling_*_by` version (wrong answers) | 1.08 s |
| pandas `groupby().apply()` reference | 4.3 s per 1,000,000 rows |

The polars time is for different, incorrect work, so it is not a like-for-like
comparison. The pandas reference is about 200 times slower per row than Viv,
which is the cost of a Python loop inside `apply`.

The kernel reads `card` and `ts` without copying them: an int64 pandas column
and a `datetime64[ns]` column are passed as views of the frame's memory, and
so are polars columns without nulls.
