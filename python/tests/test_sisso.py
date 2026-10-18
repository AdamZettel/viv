# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Runhai Ouyang (SISSO, github.com/rouyang2017/SISSO)
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vipyr.sisso against the Fortran SISSO 3.5, from its recorded outputs.

study/ports/sisso/fortran_reference.json holds what the Fortran printed for
each case of study/ports/sisso/compare_fortran.py (which runs it live and
rewrites the file with --record). These tests need no Fortran: they run the
Viv port and require the same feature-space sizes, the same screened
subspace (names, in order), the same descriptors, and the coefficients,
intercepts, RMSE and MaxAE to the Fortran's ten printed digits.
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PORT = os.path.join(ROOT, "study", "ports", "sisso")
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, PORT)

if not os.environ.get("VIV") and os.path.exists(os.path.join(ROOT, "target", "release", "viv")):
    os.environ["VIV"] = os.path.join(ROOT, "target", "release", "viv")

from vipyr import sisso  # noqa: E402
import compare_fortran as cf  # noqa: E402

REFERENCE = json.load(open(os.path.join(PORT, "fortran_reference.json")))
REL = 1e-9

# the timing case takes seconds; the others well under one
FAST_CASES = ["example", "synthetic", "deep", "fcheavy"]


def close(a, b):
    return abs(a - b) <= REL * max(abs(a), abs(b)) + 1e-12


@pytest.fixture(scope="module")
def fits():
    return {name: cf.fit_viv(cf.CASES[name]()) for name in FAST_CASES}


@pytest.mark.parametrize("name", FAST_CASES)
def test_feature_spaces_match_the_fortran(fits, name):
    r, f = fits[name], REFERENCE[name]["fortran"][-1]
    assert [p for p in r.phi_sizes if p[0] != -1] == [p for p in f["phi"] if p]
    assert [s for s in r.sis_sizes if s] == [s for s in f["sis"] if s]


@pytest.mark.parametrize("name", FAST_CASES)
def test_subspace_matches_the_fortran(fits, name):
    assert fits[name].subspace == REFERENCE[name]["fortran"][-1]["subspace"]


@pytest.mark.parametrize("name", FAST_CASES)
def test_models_match_the_fortran(fits, name):
    r = fits[name]
    for d, f in enumerate(REFERENCE[name]["fortran"], start=1):
        m = r.models[d - 1]
        assert m.descriptors == f["descriptors"]
        assert [i + 1 for i in m.indices] == f["ids"]
        for got, want in zip(m.coefficients + [m.intercept, m.rmse, m.maxae],
                             f["coef"] + [f["c0"], f["rmse"], f["maxae"]]):
            assert close(got, want), (d, got, want)


def test_known_formulas_are_recovered(fits):
    # fcheavy: y = 1.7 x1 exp(-x0) - 0.4 |x2 - x3| / x4 + noise
    assert fits["fcheavy"].models[1].descriptors == ["(x1/exp(x0))", "(abs(x2-x3)/x4)"]
    # deep: y = (a + b)^2 / (a c) + noise, found at dimension 1
    assert fits["deep"].models[0].descriptors == ["(((a+b))^2/(a*c))"]


def test_predict_reproduces_the_training_fit(fits):
    case = cf.CASES["example"]()
    r = fits["example"]
    m = r.models[-1]
    pred = r.predict(case["df"])
    y = np.asarray(case["df"]["property"])
    # the model was fit to the subspace as the Fortran writes it (ten digits)
    assert np.sqrt(np.mean((pred - y) ** 2)) == pytest.approx(m.rmse, rel=1e-6)
    assert np.max(np.abs(pred - y)) == pytest.approx(m.maxae, rel=1e-6)


def test_units_keep_sums_consistent():
    rng = np.random.default_rng(7)
    df = {"a": rng.uniform(1, 2, 20), "b": rng.uniform(1, 2, 20), "t": rng.uniform(1, 2, 20)}
    df["y"] = df["a"] / df["t"] + 0.1 * df["b"]
    r = sisso.fit(df, target="y", units={"a": "m", "b": "m", "t": "s"}, rung=1, dim=1, ops=["+", "-", "*", "/"],
                  nf_sis=1000)
    for name in r.subspace:
        assert "a+t" not in name and "t+a" not in name and "a-t" not in name and "t-a" not in name
    assert r.models[0].descriptors == ["(a/t)"]


def test_parse_unit_and_ops():
    assert sisso.parse_unit("kg*m^2/s^2") == {"kg": 1, "m": 2, "s": -2}
    assert sisso.parse_unit("m s^-1") == {"m": 1, "s": -1}
    assert sisso.parse_unit(None) == {}
    assert sisso.parse_ops("(+)(-)(|-|)(^-1)") == [1, 2, 13, 7]
    with pytest.raises(ValueError):
        sisso.parse_ops(["tanh"])
