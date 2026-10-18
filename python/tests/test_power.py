# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vp.power.runpf from Python: the case forms, the methods, the errors.
The numbers themselves are held to MATPOWER in lib/power/matpower_ref.viv and
to SciPy in lib/power/compare.py."""
import numpy as np
import pytest

import vipyr as vp
from vipyr import _power


def test_a_dict_case_gives_the_named_case_result():
    named = vp.power.runpf("case30")
    c, _ = _power.load_case("case30")
    given = vp.power.runpf({"baseMVA": c["baseMVA"], "bus": c["bus"], "gen": c["gen"], "branch": c["branch"]})
    assert named.success and given.success
    assert np.array_equal(named.vm, given.vm) and np.array_equal(named.pf, given.pf)


def test_the_three_methods_agree_where_they_should():
    nr = vp.power.runpf("case118")
    fd = vp.power.runpf("case118", method="fdxb")
    dc = vp.power.runpf("case118", method="dc")
    assert nr.iterations == 3 and nr.mismatch < 1e-8
    assert np.max(abs(fd.vm - nr.vm)) < 1e-8 and np.max(abs(fd.va - nr.va)) < 1e-6
    assert np.all(dc.vm == 1.0) and np.max(abs(dc.va - nr.va)) < 10.0


def test_max_it_stops_early_and_says_so():
    r = vp.power.runpf("case300", max_it=1)
    assert not r.success and r.iterations == 1 and r.mismatch > 1e-8


def test_bad_arguments():
    with pytest.raises(ValueError):
        vp.power.runpf("case118", method="gauss")
    with pytest.raises(ValueError):
        vp.power.runpf("case99999")
    with pytest.raises(ValueError):
        vp.power.runpf({"bus": np.zeros((1, 13))})


def test_raw_reads_back_the_matpower_case():
    raw = _power.read_raw(
        _power.cases_dir() / "raw" / "case118_v34.raw")
    m = vp.power.runpf("case118")
    r = vp.power.runpf(raw)
    assert raw["rev"] == 34
    assert np.max(abs(r.vm - m.vm)) < 1e-9


def test_contingency_rows_and_errors():
    r = vp.power.contingency("case30", contingencies=[("branch", 0)])
    assert r.converged.shape == (2,) and r.flows.shape == (2, 41)
    with pytest.raises(ValueError):
        vp.power.contingency("case30", contingencies=[("bus", 0)])
    with pytest.raises(ValueError):
        vp.power.contingency("case30", limits={"vmax2": 1.0})


def test_short_circuit_needs_base_kv():
    with pytest.raises(ValueError):
        vp.power.short_circuit("case14")
