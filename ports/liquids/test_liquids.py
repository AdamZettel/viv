# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The port of Allen and Tildesley's Lennard-Jones programs against upstream.

    python3 ports/liquids/test_liquids.py            (or python3 -m pytest ports/liquids/test_liquids.py)
    python3 ports/liquids/test_liquids.py --table    the tables of COMPARISON.md

It loads each program of ports/liquids through vipyr (python/vipyr), which
exports it by the manifest beside it and builds the library. $VIV names the
compiler (default target/release/viv).

References, in order of trust:
  - upstream: the Fortran of https://github.com/Allen-Tildesley/examples
    built by oracle/build.py with the port's random numbers (MINSTD) and
    Viv's libm, run by oracle/make_reference.py on the same cases
    (oracle/upstream_reference.json). With the same random numbers the
    port does upstream's arithmetic in upstream's order, so every run
    average, every error bar, every block average, every coordinate of the
    final configuration and every value of g(r), the correlation functions,
    the equation of state and the error analysis must be the same double:
    tolerance zero.
  - the equation of state of Thol et al. (2015, 2016), as eos_lj computes
    it (itself held to upstream's bits above): the run averages of longer
    runs must agree with it within NSIG standard errors of the run, the
    standard error being the block error. The known bias of the delta
    correction to the cut pressure (GUIDE.md: "too low by about 0.03" at
    rho = 0.75, because g(r_c) = 0.91, not 1) is allowed for in that check.
  - GUIDE.md's own test results, printed to 3 or 4 digits with their error
    in the last digit: the run averages must agree within the stated error
    plus NSIG of the run's own standard error.
  - the velocity Verlet energy fluctuation scales as dt^2 (its mean square
    as dt^4): the slope of log(conserved_msd) against log(dt) must be 4
    within 0.4, on the timesteps GUIDE.md tabulates.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.environ.setdefault("VIV", os.path.join(ROOT, "target", "release", "viv"))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools", "export"))
import vipyr  # noqa: E402
import check  # noqa: E402  (tools/export/check.py: result_hash)

REF = json.load(open(os.path.join(HERE, "oracle", "upstream_reference.json")))
FRAMES = json.load(open(os.path.join(HERE, "oracle", "upstream_frames.json")))["frames"]
HASHES = os.path.join(HERE, "liquids.exact_hashes.json")
NSIG = 4.0  # statistical tolerance, in standard errors

# the columns of upstream's block lines, as each program's results name them
COLUMNS = {
    "md_nve": ["e_cs", "p_cs", "e_full", "p_full", "t_kin", "t_config", "cv_cs", "conserved_msd"],
    "md_nvt": ["e_cs", "p_cs", "e_full", "p_full", "t_kin", "t_config", "conserved", "cv_cs", "cv_full",
               "conserved_msd"],
    "mc_nvt": ["move_ratio", "e_cut", "p_cut", "e_full", "p_full", "t_config", "cv_full"],
    "mc_npt": ["move_ratio", "volume_ratio", "density_avg", "e_cut", "p_cut", "e_full", "p_full", "t_config",
               "cp_cut", "cp_full", "volume_msd"],
}

_mods = {}


def program(name):
    if name not in _mods:
        _mods[name] = vipyr.load(os.path.join(HERE, name + ".viv"))
    return _mods[name]


def same(a, b):
    """Bit for bit, element by element: how many of the doubles differ."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    assert a.shape == b.shape, (a.shape, b.shape)
    return int(np.sum(a.view(np.uint64) != b.view(np.uint64)))


def compare_simulation(name):
    case = REF["simulations"][name]
    t0 = time.time()
    r = program(case["port"]).run(**case["params"])
    secs = time.time() - t0
    cols = COLUMNS[case["port"]]
    avg = [r[c] for c in cols]
    blocks = np.asarray(r["blocks"])
    final = np.asarray(case["final"]["rows"])
    mine = np.asarray(r["r_final"])
    if "v_final" in r:
        mine = np.hstack([mine, np.asarray(r["v_final"])])
    box = r.get("box_final", r["box"])
    diffs = {
        "run averages": same(avg, case["run_averages"]),
        "block averages": same(blocks, case["blocks"]),
        "final configuration": same(mine, final),
        "box": same([box], [case["final"]["box"]]),
    }
    return diffs, secs, r


def test_simulations_against_upstream():
    for name in REF["simulations"]:
        diffs, _, r = compare_simulation(name)
        bad = {k: v for k, v in diffs.items() if v}
        assert not bad, "%s: doubles that differ from upstream: %s" % (name, bad)


def test_errors_against_upstream():
    """The run errors: the same standard errors, upstream's formula."""
    for name, case in REF["simulations"].items():
        r = program(case["port"]).run(**case["params"])
        cols = COLUMNS[case["port"]]
        errs = []
        for c, up in zip(cols, case["run_errors"]):
            if c + "_err" in r:
                errs.append((r[c + "_err"], up))
        assert same([e[0] for e in errs], [e[1] for e in errs]) == 0, name


def test_eos_against_upstream():
    eos = program("eos_lj")
    for name, case in REF["eos"].items():
        r = eos.run(temperature=case["temperature"], density=case["density"])
        for key, up in case.items():
            if key in ("temperature", "density"):
                continue
            if key.startswith("a"):
                i, j, which = int(key[1]), int(key[2]), key.split("_")[1]
                mine = r["a_res_full_table" if which == "full" else "a_res_cs_table"][i][j]
            else:
                mine = r[key]
            assert same([mine], [up]) == 0, (name, key, mine, up)


def frames_array(key):
    rows = [row for fr in FRAMES for row in fr["rows"]]
    a = np.asarray(rows)
    return a[:, :3] if key == "r" else a[:, 3:]


def test_pair_distribution_against_upstream():
    n, box = FRAMES[0]["n"], FRAMES[0]["box"]
    r = program("pair_distribution").run(n=n, box=box, frames=frames_array("r"), dr=0.02)
    up = np.asarray(REF["pair_distribution"]["out"])
    assert same(r["r_mid"], up[:, 0]) == 0
    assert same(r["g"], up[:, 1]) == 0


def test_diffusion_against_upstream():
    n, box = FRAMES[0]["n"], FRAMES[0]["box"]
    d = REF["diffusion"]
    r = program("diffusion").run(n=n, box=box, frames_r=frames_array("r"), frames_v=frames_array("v"),
                                 nt=d["nt"], origin_interval=d["origin_interval"], delta=d["delta"])
    up = np.asarray(d["out"])
    assert same(r["vacf"], up[:, 1]) == 0
    assert same(r["rvcf"], up[:, 2]) == 0
    assert same(r["msd"], up[:, 3]) == 0


def test_error_calc_against_upstream():
    e = REF["error_calc"]
    kw = dict(x.strip().split("=") for x in e["namelist"].split(","))
    r = program("error_calc").run(**{k: int(v) for k, v in kw.items()})
    for key in ("si_exact", "err_exact", "err_brute", "mean", "var", "err_naive"):
        assert same([r[key]], [e[key]]) == 0, key
    lines = np.asarray(e["lines"])
    nfixed = len(r["err_fixed"])
    assert same(r["err_fixed"], lines[:nfixed, 2]) == 0
    assert same(r["si_fixed"], lines[:nfixed, 3]) == 0
    assert same(r["err_halving"], lines[nfixed:, 2]) == 0
    assert same(r["si_halving"], lines[nfixed:, 3]) == 0


# ---- statistics: longer runs against the equation of state and GUIDE.md

# GUIDE.md's test results at rho = 0.75, T = 1.0, N = 256 (value, error in
# the last digit as printed); None where GUIDE.md gives no error
GUIDE_MC_NVT = {"e_cut": (-3.332, 0.001), "p_cut": (0.651, 0.003), "e_full": (-3.734, 0.001),
                "p_full": (0.350, 0.003), "cv_full": (2.28, 0.01)}
GUIDE_MC_NPT = {"density": (0.7501, 0.0002), "e_cut": (-3.331, 0.001), "p_cut": (0.666, 0.002),
                "e_full": (-3.733, 0.001), "p_full": (0.364, 0.002)}
STATE = dict(n=256, density=0.75, temperature=1.0)
_equilibrated = {}


def liquid_start():
    """An equilibrated liquid at rho = 0.75, T = 1: md_nvt from the fcc
    lattice for 4 blocks of 1000 steps (20 time units; the lattice melts in
    the first few)."""
    if "md" not in _equilibrated:
        r = program("md_nvt").run(nblock=4, nstep=1000, **STATE)
        _equilibrated["md"] = dict(r_start=r["r_final"], v_start=r["v_final"], box_start=r["box"])
    return _equilibrated["md"]


def within(name, got, err, want, tol):
    assert abs(got - want) <= tol, "%s = %.5f +- %.5f, reference %.5f: off by %.5f > %.5f" % (
        name, got, err, want, abs(got - want), tol)


def test_md_nvt_against_the_cut_and_shifted_equation_of_state():
    """Thol et al. (2015) fitted the cut-and-shifted potential at r_c = 2.5,
    exactly what md_nvt simulates, so the run's E and P must agree with it
    within NSIG of the run's block errors."""
    s = liquid_start()
    r = program("md_nvt").run(nblock=10, nstep=2000, seed=2, **dict(STATE, **s))
    e = program("eos_lj").run(temperature=1.0, density=0.75)
    within("E_cs", r["e_cs"], r["e_cs_err"], e["e_cs"], NSIG * r["e_cs_err"])
    within("P_cs", r["p_cs"], r["p_cs_err"], e["p_cs"], NSIG * r["p_cs_err"])
    within("T_kin", r["t_kin"], r["t_kin_err"], 1.0, NSIG * r["t_kin_err"])


def test_md_nve_against_the_equation_of_state_at_its_temperature():
    """At constant energy T is measured, not set: the equation of state is
    taken at the run's mean kinetic temperature, and the tolerance adds the
    pressure's change over NSIG errors of that temperature
    (dP/dT from the equation of state). The configurational temperature is
    not checked against T: the cut-and-shifted force jumps at r_c, which
    adds a delta term to the Laplacian that upstream leaves out (GUIDE.md,
    "there should be a delta correction for the configurational
    temperature"); it comes out about 1% high."""
    s = liquid_start()
    r = program("md_nve").run(nblock=10, nstep=2000, **dict(STATE, **s))
    t = r["t_kin"]
    e = program("eos_lj").run(temperature=t, density=0.75)
    dt = 1e-3
    dpdt = (program("eos_lj").run(temperature=t + dt, density=0.75)["p_cs"] -
            program("eos_lj").run(temperature=t - dt, density=0.75)["p_cs"]) / (2 * dt)
    within("P_cs", r["p_cs"], r["p_cs_err"], e["p_cs"], NSIG * math.hypot(r["p_cs_err"], dpdt * r["t_kin_err"]))


def test_mc_nvt_against_guide():
    """Upstream's own mc_nvt result (GUIDE.md), which sits 0.012 below the
    equation of state's E_cut because the long-range correction assumes
    g(r) = 1 beyond r_c: the port must reproduce upstream's statistics,
    within NSIG of the two error bars combined."""
    r0 = program("mc_nvt").run(nblock=1, nstep=500, **STATE)
    r = program("mc_nvt").run(nblock=10, nstep=1000, seed=3, r_start=r0["r_final"], box_start=r0["box"],
                              **STATE)
    for key, (want, err) in GUIDE_MC_NVT.items():
        within(key, r[key], r[key + "_err"], want, NSIG * math.hypot(r[key + "_err"], err))


def test_mc_npt_against_guide():
    """Upstream's mc_npt at P = 0.69 (GUIDE.md), the same way."""
    r0 = program("mc_npt").run(nblock=1, nstep=500, pressure=0.69, **STATE)
    r = program("mc_npt").run(nblock=10, nstep=500, seed=4, pressure=0.69, r_start=r0["r_final"],
                              box_start=r0["box_final"], **STATE)
    r["density"] = r["density_avg"]
    for key, (want, err) in GUIDE_MC_NPT.items():
        within(key, r[key], r[key + "_err"], want, NSIG * math.hypot(r[key + "_err"], err))


DT_LAW = [0.008, 0.004, 0.002, 0.001]


def dt_law():
    """md_nve from the equilibrated liquid for 4 time units at each step
    size: the mean-square fluctuation of the energy per atom."""
    s = liquid_start()
    out = []
    for dt in DT_LAW:
        r = program("md_nve").run(nblock=1, nstep=int(round(4.0 / dt)), dt=dt, **dict(STATE, **s))
        out.append(r["conserved_msd"])
    return out


def test_energy_fluctuation_scales_as_dt_squared():
    """Velocity Verlet's energy error is O(dt^2), so its mean square is
    O(dt^4): the least-squares slope of log(MSD) on log(dt) must be 4 within
    0.4 (GUIDE.md's own table gives 4.2 over the same steps)."""
    msd = dt_law()
    slope = np.polyfit(np.log(DT_LAW), np.log(msd), 1)[0]
    assert abs(slope - 4.0) <= 0.4, (slope, msd)


MANIFESTS = ["md_nve", "md_nvt", "mc_nvt", "mc_npt", "eos_lj", "pair_distribution", "diffusion", "error_calc"]


def test_bits_against_recorded_hashes():
    """Every case of every manifest gives the result hash that
    tools/export/check.py --record wrote once the library, the native build
    and the interpreter agreed on it."""
    for name in MANIFESTS:
        rec = json.load(open(os.path.join(HERE, name + ".exact_hashes.json")))
        cases = json.load(open(os.path.join(HERE, name + ".cases.json")))
        for case, kw in cases.items():
            h = check.result_hash(program(name).run(**kw))
            assert h == rec[case], (name, case, h, rec[case])


def table():
    """The tables of COMPARISON.md."""
    print("| case | upstream program | doubles compared | differing |")
    print("|---|---|---|---|")
    for name, case in REF["simulations"].items():
        diffs, secs, r = compare_simulation(name)
        n = len(case["run_averages"]) + np.asarray(case["blocks"]).size + np.asarray(case["final"]["rows"]).size + 1
        print("| %s (%s) | %s | %d | %d |" % (name, case["namelist"], case["program"], n, sum(diffs.values())))


if __name__ == "__main__":
    if "--table" in sys.argv:
        table()
        sys.exit(0)
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for n in names:
        print("==", n)
        globals()[n]()
    print("all %d passed" % len(names))
