# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The shell-model program, called from Python through vipyr, against BIGSTICK.

    python3 -m pytest lib/nuclear/shell/test_shellmodel.py

bigstick_reference.json holds BIGSTICK 8.0.0's energies, spins and basis
dimensions for each case (bigstick_compare.py --record wrote it from the
oracle build described there). The tolerances:

- dimension: equal;
- energies: 1e-5 MeV. BIGSTICK stores its matrix elements in single
  precision, about 1e-7 relative, so on energies of 40 to 140 MeV its own
  values carry a few 1e-6 MeV of rounding; the largest difference over the
  seven cases is 4.2e-6 MeV;
- spins: 5e-5. BIGSTICK's J comes from a single-precision J^2 as well; the
  largest difference is 1.7e-5 (silicon-28).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "python"))
import vipyr  # noqa: E402

REF = json.loads((HERE / "bigstick_reference.json").read_text())
# case: interaction and valence numbers; the two heaviest are left to bigstick_compare.py
CASES = {
    "ne20": ("usdb", 2, 2),
    "na22": ("usdb", 3, 3),
    "c12": ("ckpot", 4, 4),
    "b10": ("ckpot", 3, 3),
    "mg24": ("usdb", 4, 4),
}


@pytest.fixture(scope="module")
def sm():
    return vipyr.load(HERE / "shellmodel.viv")


@pytest.mark.parametrize("name", list(CASES))
def test_against_bigstick(sm, name):
    inter, z, n = CASES[name]
    ref = REF[name]
    r = sm.run(interaction=inter, z=z, n=n, nstates=len(ref["energies"]), tol=1e-8)
    assert r["converged"]
    assert r["dim"] == ref["dim"]
    assert np.max(np.abs(r["energies"] - ref["energies"])) < 1e-5
    assert np.max(np.abs(r["j"] - ref["j"])) < 5e-5
    assert np.allclose(r["excitation"], r["energies"] - r["energies"][0], rtol=0, atol=1e-12)
    assert np.allclose(r["occ_p"].sum(axis=1), z, atol=1e-10)
    assert np.allclose(r["occ_n"].sum(axis=1), n, atol=1e-10)


def test_files_read_from_disk_give_the_same_bits(sm):
    shipped = sm.run(z=2, n=2, nstates=4)
    read = sm.run(sps_file=str(HERE / "data" / "sd.sps"), int_file=str(HERE / "data" / "usdb.int"),
                  z=2, n=2, nstates=4)
    assert np.array_equal(shipped["energies"], read["energies"])
    assert shipped["run_id"].startswith("shell-usdb-z2-n2")
    assert read["run_id"].startswith("shell-files-z2-n2")


def test_odd_nucleus_takes_m_one_half(sm):
    r = sm.run(z=1, n=2, nstates=3)
    assert "m2_1" in r["run_id"]
    assert np.all(np.abs(2 * r["j"] - np.round(2 * r["j"]) ) < 1e-6)
    assert np.all(np.round(2 * r["j"]) % 2 == 1)


def test_higher_m_drops_the_low_spins(sm):
    """With 2M = 4 only states of J >= 2 remain: the 0+ ground state of
    neon-20 is gone and the lowest state is its 2+ at 1.75 MeV."""
    low = sm.run(z=2, n=2, nstates=3)
    high = sm.run(z=2, n=2, nstates=1, twice_m=4)
    assert abs(high["energies"][0] - low["energies"][1]) < 1e-9
    assert abs(high["j"][0] - 2.0) < 1e-6


def test_bad_parameters_stop_with_the_programs_message(sm):
    with pytest.raises(sm.VivStop, match="parity"):
        sm.run(z=2, n=2, twice_m=1)
    with pytest.raises(sm.VivStop, match="go together"):
        sm.run(sps_file=str(HERE / "data" / "sd.sps"))


@pytest.mark.parametrize("name", ["ne20", "c12", "b10"])
def test_the_dense_methods_agree_with_lanczos_and_bigstick(sm, name):
    """The lowest_eigenpairs contract (lib/eigen/lowest):
    Householder + QL and cyclic Jacobi on H as a matrix give
    Lanczos's energies to 1e-10 MeV (the run's tol is 1e-8; the largest
    difference on these cases is 2.2e-13) and BIGSTICK's to 1e-5."""
    inter, z, n = CASES[name]
    lz = sm.run(interaction=inter, z=z, n=n, nstates=10)
    dn = sm.run(interaction=inter, z=z, n=n, nstates=10, eigensolver="householder_ql")
    jc = sm.run(interaction=inter, z=z, n=n, nstates=10, eigensolver="jacobi")
    assert np.max(np.abs(jc["energies"] - dn["energies"])) < 1e-10
    assert dn["converged"] and dn["iterations"] == dn["dim"]
    assert np.max(np.abs(lz["energies"] - dn["energies"])) < 1e-10
    assert np.max(np.abs(dn["energies"] - REF[name]["energies"])) < 1e-5
    assert np.max(np.abs(dn["j"] - REF[name]["j"])) < 5e-5
    assert dn["run_id"].endswith("-householder_ql") and jc["run_id"].endswith("-jacobi")


def test_every_eigensolver_runs_both_and_the_keyword_picks_one():
    import vipyr as vp
    every = vp.nuclear.shell_model(z=2, n=2, nstates=4, eigensolver="every")
    dense = vp.nuclear.shell_model(z=2, n=2, nstates=4, eigensolver="householder_ql")
    both = vp.nuclear.shell_model(z=2, n=2, nstates=4, eigensolver="every dense")
    assert np.max(np.abs(every.energies - dense.energies)) < 1e-10
    assert np.array_equal(both.energies, dense.energies)
    # the Lanczos settings do nothing for the other methods
    loose = vp.nuclear.shell_model(z=2, n=2, nstates=4, eigensolver="householder_ql", tol=1e-3, krylov=7)
    assert np.array_equal(loose.energies, dense.energies)
    with pytest.raises(ValueError):
        vp.nuclear.shell_model(z=2, n=2, eigensolver="arnoldi")


def test_dense_refuses_a_basis_it_cannot_store(sm):
    with pytest.raises(sm.VivStop, match="LOWEST_DENSE_MAX"):
        sm.run(z=4, n=4, nstates=2, eigensolver="householder_ql")
