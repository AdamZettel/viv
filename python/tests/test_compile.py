# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vipyr compile: scikit-learn pipelines as one C99 file (vipyr/compile).

Each case fits a small model, compiles it, and requires the library's
outputs on held-out rows to be scikit-learn's: bit for bit for the trees,
forests and boosting, within the embedded tolerance for linear models.
`python python/tests/test_compile.py` prints the table of what each family
achieves; pytest runs the same cases.
"""
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,  # noqa: E402
                              GradientBoostingRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # noqa: E402

from vipyr.compile import CompileError, compile_model, load_model  # noqa: E402
from vipyr.compile.emit import le_bound  # noqa: E402

N_TRAIN, N_TEST = 1500, 800


def frame(seed=0, classes=2, nan_frac=0.0, regression=False):
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_TEST
    df = pd.DataFrame({
        "a": rng.normal(0, 1, n), "b": rng.gamma(2.0, 3.0, n), "c": rng.uniform(-5, 5, n),
        "k": rng.integers(0, 4, n).astype(np.int64),
        "city": rng.choice(["oslo", "lima", "pune", "kiev"], n),
    })
    z = df.a * 1.5 - 0.2 * df.b + np.sin(df.c) + 0.3 * df.k + (df.city == "lima") * 0.8 + rng.normal(0, 0.5, n)
    if nan_frac:
        df.loc[rng.random(n) < nan_frac, "b"] = np.nan
    if regression:
        y = z.to_numpy()
    else:
        y = np.digitize(z, np.quantile(z, np.linspace(0, 1, classes + 1)[1:-1]))
    return df, y


def mixed(est, impute=False, drop=None):
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]) if impute else StandardScaler()
    ct = ColumnTransformer([("num", num, ["a", "b", "c"]), ("mm", MinMaxScaler(), ["k"]),
                            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=drop), ["city"])],
                           sparse_threshold=0)
    return Pipeline([("ct", ct), ("est", est)])


CASES = {
    "DecisionTreeClassifier": (lambda: mixed(DecisionTreeClassifier(max_depth=8, random_state=0)), dict(classes=3)),
    "DecisionTreeRegressor": (lambda: mixed(DecisionTreeRegressor(max_depth=8, random_state=0)), dict(regression=True)),
    "RandomForestClassifier": (lambda: mixed(RandomForestClassifier(30, max_depth=8, random_state=0)), dict(classes=3)),
    "RandomForestRegressor": (lambda: mixed(RandomForestRegressor(30, max_depth=8, random_state=0)), dict(regression=True)),
    "ExtraTreesClassifier": (lambda: mixed(ExtraTreesClassifier(30, max_depth=8, random_state=0)), dict()),
    "ExtraTreesRegressor": (lambda: mixed(ExtraTreesRegressor(30, max_depth=8, random_state=0)), dict(regression=True)),
    "GradientBoostingClassifier (binary)": (lambda: mixed(GradientBoostingClassifier(n_estimators=60, random_state=0)), dict()),
    "GradientBoostingClassifier (3 classes)": (lambda: mixed(GradientBoostingClassifier(n_estimators=40, random_state=0)), dict(classes=3)),
    "GradientBoostingRegressor": (lambda: mixed(GradientBoostingRegressor(n_estimators=60, random_state=0)), dict(regression=True)),
    "HistGradientBoostingClassifier (binary, NaN)": (lambda: mixed(HistGradientBoostingClassifier(max_iter=60, random_state=0)), dict(nan_frac=0.1)),
    "HistGradientBoostingClassifier (3 classes)": (lambda: mixed(HistGradientBoostingClassifier(max_iter=40, random_state=0)), dict(classes=3)),
    "HistGradientBoostingRegressor": (lambda: mixed(HistGradientBoostingRegressor(max_iter=60, random_state=0)), dict(regression=True)),
    "LogisticRegression (binary)": (lambda: mixed(LogisticRegression(max_iter=500)), dict()),
    "LogisticRegression (3 classes)": (lambda: mixed(LogisticRegression(max_iter=500)), dict(classes=3)),
    "LinearRegression": (lambda: mixed(LinearRegression(), drop="first"), dict(regression=True)),
    "Ridge (imputer)": (lambda: mixed(Ridge(alpha=1.0), impute=True), dict(regression=True, nan_frac=0.1)),
}
EXACT_FAMILIES = ("Tree", "Forest", "Trees", "Boosting")


def run_case(name, out_dir, build_check=False):
    make, kw = CASES[name]
    df, y = frame(**kw)
    m = make().fit(df[:N_TRAIN], y[:N_TRAIN])
    out = Path(out_dir) / (name.split(" ")[0].lower() + ".c")
    return compile_model(m, df[N_TRAIN:], out, check_rows=80, log=None, build_check=build_check)


@pytest.mark.parametrize("name", list(CASES))
def test_family(name, tmp_path):
    r = run_case(name, tmp_path)
    if any(f in name for f in EXACT_FAMILIES) and ("Regressor" in name or "Forest" in name or "Tree" in name):
        assert r["bit_identical"], r
    elif any(f in name for f in EXACT_FAMILIES):
        # probabilities through exp: Viv's exp is correctly rounded, the platform's may not be
        assert r["bit_identical"] or (r["max_ulps"] <= 2 and r["labels_identical"]), r
    else:
        assert r["max_abs_diff"] <= 1e-12 and r["labels_identical"] in (True, None), r


@pytest.mark.parametrize("name", ["DecisionTreeClassifier", "DecisionTreeRegressor", "RandomForestClassifier",
                                  "GradientBoostingClassifier (3 classes)",
                                  "HistGradientBoostingClassifier (binary, NaN)"])
def test_branches_and_tables_agree(name, tmp_path, monkeypatch):
    """The trees written as nested ifs (row by row, and in blocks of rows)
    and as node tables reach the same leaves: every output equal to the
    bit, NaNs routed alike."""
    emit_mod = sys.modules["vipyr.compile.emit"]    # the module: vipyr.compile.emit is also the function
    make, kw = CASES[name]
    df, y = frame(**kw)
    m = make().fit(df[:N_TRAIN], y[:N_TRAIN])
    got = {}
    for mode, block_nodes in (("branches", emit_mod.BLOCK_NODES), ("blocks", 0), ("tables", emit_mod.BLOCK_NODES)):
        monkeypatch.setattr(emit_mod, "BLOCK_NODES", block_nodes)
        out = tmp_path / ("m_%s.c" % mode)
        tc = "tables" if mode == "tables" else "branches"
        r = compile_model(m, df[N_TRAIN:], out, check_rows=40, log=None, build_check=False, tree_code=tc)
        assert ("as %s" % tc) in r["notes"][0], r["notes"]
        if mode == "blocks" and "DecisionTree" not in name:
            assert "ROW_BLOCK" in out.with_suffix(".viv").read_text()
        got[mode] = load_model(out).predict_all(df[N_TRAIN:])
    assert np.array_equal(got["branches"], got["tables"], equal_nan=True)
    assert np.array_equal(got["blocks"], got["tables"], equal_nan=True)


def test_selftest_and_cli_under_each_compiler(tmp_path):
    r = run_case("GradientBoostingClassifier (binary)", tmp_path, build_check=True)
    assert r["builds"] and all(b["selftest_status"] == 0 and b["cli_identical_to_library"] for b in r["builds"]), r


def test_le_bound_is_the_float32_comparison():
    """float32(x) <= t exactly when x <= le_bound(t), at the bound, one ulp
    either side, and at the float32 midpoints themselves."""
    rng = np.random.default_rng(3)
    ts = np.concatenate([rng.normal(0, 1, 3000), rng.normal(0, 1e6, 500), [0.0, -0.0, 1e-45, -1e-45, 3.4e38, -3.4e38]])
    xs = []
    for t in ts:
        b = le_bound(t)
        f = np.float32(t)
        for v in (b, np.nextafter(b, np.inf), np.nextafter(b, -np.inf), float(f), t,
                  float(np.nextafter(f, np.float32(np.inf), dtype=np.float32))):
            xs.append((t, float(v)))
    for t, x in xs:
        assert (np.float64(np.float32(x)) <= t) == (x <= le_bound(t)), (t, x)


def test_errors(tmp_path):
    df, y = frame()
    m = mixed(DecisionTreeClassifier(max_depth=3)).fit(df[:N_TRAIN], y[:N_TRAIN])
    compile_model(m, df[N_TRAIN:], tmp_path / "e.c", check_rows=20, log=None, build_check=False)
    lib = load_model(tmp_path / "e.c")
    with pytest.raises(KeyError):
        lib.predict(df.drop(columns=["city"]))
    # an unknown city is all zeros in the one-hot columns (handle_unknown="ignore"), as in scikit-learn
    odd = df[N_TRAIN:N_TRAIN + 5].copy()
    odd["city"] = "nowhere"
    assert np.array_equal(lib.predict_proba(odd), m.predict_proba(odd))
    with pytest.raises(CompileError):
        bad = Pipeline([("ct", ColumnTransformer([("cat", OneHotEncoder(), ["city"])])), ("est", DecisionTreeClassifier())])
        compile_model(bad.fit(df[:N_TRAIN], y[:N_TRAIN]), df[N_TRAIN:], tmp_path / "s.c", log=None, build_check=False)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        print("| model | outputs | bit-identical | rows differing | max diff | labels |")
        print("|---|---|---|---|---|---|")
        for name in CASES:
            r = run_case(name, d)
            print("| %s | %d x %d | %s | %d | %.2g (%.0f ulp) | %s |" % (
                name, r["rows_verified"], r["outputs_per_row"], "yes" if r["bit_identical"] else "no",
                r["rows_differing"], r["max_abs_diff"], r["max_ulps"],
                                                      {True: "same", False: "DIFFER", None: "-"}[r["labels_identical"]]))
            sys.stdout.flush()
