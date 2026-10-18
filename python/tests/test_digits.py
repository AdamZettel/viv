# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vipyr.digits (the digits lens, lang/viv/src/digits.rs) on known cases:
1 - cos(x) keeps about 16 - log10(2/x^2) digits, its rewrite keeps all but
the last; a Hilbert solve keeps about 16 - log10 cond(H)."""
import numpy as np

import vipyr as vp


def test_cancellation_loses_the_digits_the_subtraction_cancels():
    r = vp.digits("examples/digits/cancel.viv", attrib=0, samples=16)
    o = r.outputs
    for k, x in enumerate((1e-2, 1e-4, 1e-6)):
        naive, good = o[3 * k + 1], o[3 * k + 2]
        expected = 16 - np.log10(2 / x ** 2)
        assert abs(naive.stable - expected) < 1.0, (x, naive.stable, expected)
        assert good.stable > 14.5
    assert o[10].value == 0.0 and o[10].stable == 0.0  # 1 - cos(1e-8): no digit


def test_hilbert_tracks_the_condition_number():
    n = 8
    H = 1.0 / (np.arange(n)[:, None] + np.arange(n)[None, :] + 1.0)
    r = vp.digits("examples/digits/hilbert.viv", sets={"n": n}, attrib=0)
    assert abs(r.stable.min() - (16 - np.log10(np.linalg.cond(H)))) < 1.0


def test_switches_and_attribution():
    r = vp.digits("examples/digits/kahan.viv", sets={"n": 1000}, fma=False, mca="off", attrib=0)
    assert [o.mca for o in r.outputs] == [None, None, None]
    assert r.outputs[0].order == 15.95  # a loop is never reordered
    r = vp.digits("examples/digits/cancel.viv", attrib=4, samples=8)
    moved = [l for l in r.lines if any(e["digits"] < 5 for e in l.get("effects", []))]
    assert any("cos(x)" in l["code"] for l in moved)
