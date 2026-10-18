# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Train a churn model on card transactions, compile it to churn.c, check it,
and time it against scikit-learn.

    python examples/deploy/deploy.py [--rows 120000]

The model: two features per transaction from card_velocity/features.viv (the
card's transactions in the hour before, its decline/chargeback status), a
ColumnTransformer (StandardScaler on the numbers, OneHotEncoder on merchant
type and status) and a GradientBoostingClassifier. The label is synthetic:
whether the holder closes the card within 30 days of the transaction.

Training computes the features with the same Viv file the C file runs, so
the features scored in production are the features the model was fitted on.
Writes churn.c, churn.viv, holdout.csv and results.json beside this file.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "python"))
warnings.filterwarnings("ignore")

import vipyr  # noqa: E402
from vipyr.compile import compile_model, load_model  # noqa: E402

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

FEATURES = ROOT / "examples" / "datasci" / "card_velocity" / "features.viv"
MERCHANTS = ["grocery", "fuel", "travel", "online", "restaurant", "electronics"]


def transactions(n, seed):
    """n transactions sorted by (card, ts), about 50 per card over 30 days."""
    rng = np.random.default_rng(seed)
    cards = max(1, n // 50)
    card = np.sort(rng.integers(0, cards, n)) + seed * 10 ** 7
    t = rng.integers(0, 30 * 86400, n)
    burst = rng.random(n) < 0.35
    t = np.where(burst, np.roll(t, 1) + rng.integers(0, 1200, n), t)
    order = np.lexsort((t, card))
    declined = (rng.random(n) < np.where(burst, 0.3, 0.03))[order]
    return pd.DataFrame({
        "card": card[order].astype(np.int64),
        "ts": (t[order].astype(np.int64) + 1_767_225_600) * 10 ** 9,      # 2026-01-01, in ns
        "amount": np.round(rng.gamma(2.0, 30.0, n), 2),
        "merchant": rng.choice(MERCHANTS, n, p=[0.3, 0.15, 0.05, 0.3, 0.15, 0.05]),
        "declined": declined.astype(np.int64),
        "chargeback": (rng.random(n) < 0.003).astype(np.int64),
    })


def label(df, feats, seed):
    rng = np.random.default_rng(seed + 99)
    z = (-2.2 + 0.35 * feats["n_prev_1h"] + 0.9 * (feats["status"] == 1) + 1.6 * (feats["status"] == 2)
         + 0.004 * df["amount"] + 0.5 * (df["merchant"] == "travel") + rng.logistic(0, 1, len(df)))
    return (z > 0).astype(int).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=120_000, help="training rows (held-out: a quarter of that)")
    a = ap.parse_args()
    feats_mod = vipyr.load(FEATURES)

    def with_features(df):
        f = feats_mod.card_features(df[["card", "ts", "declined", "chargeback"]])
        return df.assign(n_prev_1h=np.asarray(f["n_prev_1h"]), status=np.asarray(f["status"]))

    train = with_features(transactions(a.rows, seed=1))
    hold_raw = transactions(a.rows // 4, seed=2)
    hold = with_features(hold_raw)
    y = label(train, train, 1)

    pipe = Pipeline([
        ("ct", ColumnTransformer([
            ("num", StandardScaler(), ["amount", "n_prev_1h"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["merchant", "status"]),
        ], sparse_threshold=0)),
        ("gbm", GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=0)),
    ])
    t0 = time.perf_counter()
    pipe.fit(train, y)
    print("fitted in %.1f s; held-out accuracy %.3f" % (time.perf_counter() - t0,
                                                      (pipe.predict(hold) == label(hold, hold, 2)).mean()))
    with open(HERE / "model.pkl", "wb") as f:
        pickle.dump(pipe, f)
    hold_raw.to_csv(HERE / "holdout.csv", index=False)

    # compile: the raw held-out rows are the check data (the features run inside)
    t0 = time.perf_counter()
    rep = compile_model(pipe, hold_raw, HERE / "churn.c", features=FEATURES, check_rows=300)
    rep["compile_total_s"] = time.perf_counter() - t0

    # scoring speed, in-process, warm, median of 5: the C library (features
    # included) against scikit-learn's predict_proba on the frame that already
    # holds the features
    lib = load_model(HERE / "churn.c")

    def best(f):
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            f()
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    n = len(hold_raw)
    t_c = best(lambda: lib.predict_proba(hold_raw))
    t_sk = best(lambda: pipe.predict_proba(hold))
    t_feat = best(lambda: with_features(hold_raw))
    t_sk_one = best(lambda: [pipe.predict_proba(hold.iloc[i:i + 1]) for i in range(200)]) / 200
    t_c_one = best(lambda: [lib.predict_proba(hold_raw.iloc[i:i + 1]) for i in range(200)]) / 200
    exe = HERE / "churn"
    csv = (HERE / "holdout.csv").read_bytes()
    t_cli = best(lambda: subprocess.run([str(exe)], input=csv, capture_output=True, check=True))
    rep["speed"] = {
        "rows": n, "load": os.getloadavg()[0],
        "c_library_rows_per_s": n / t_c, "sklearn_predict_proba_rows_per_s": n / t_sk,
        "sklearn_with_viv_features_rows_per_s": n / (t_sk + t_feat),
        "cli_csv_rows_per_s": n / t_cli,
        "single_row_c_us": t_c_one * 1e6, "single_row_sklearn_us": t_sk_one * 1e6,
    }
    for m in ("m2cgen", "onnxruntime", "skl2onnx"):
        try:
            __import__(m)
            rep["speed"][m] = "installed, not timed here"
        except ImportError:
            rep["speed"][m] = "not installed"
    (HERE / "results.json").write_text(json.dumps(rep, indent=2, default=str))
    s = rep["speed"]
    print("scoring %d rows (load %.1f): C library %.0f rows/s, scikit-learn predict_proba %.0f rows/s "
          "(%.0f with the Viv features), command line on CSV %.0f rows/s; one row: C %.0f us, scikit-learn %.0f us"
          % (n, s["load"], s["c_library_rows_per_s"], s["sklearn_predict_proba_rows_per_s"],
             s["sklearn_with_viv_features_rows_per_s"], s["cli_csv_rows_per_s"], s["single_row_c_us"],
             s["single_row_sklearn_us"]))


if __name__ == "__main__":
    main()
