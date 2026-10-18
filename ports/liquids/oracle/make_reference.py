# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Runs the exact build of upstream (build.py) on the comparison cases and
writes what it printed to upstream_reference.json beside this file.

    python3 ports/liquids/oracle/build.py <examples checkout> <build dir>
    python3 ports/liquids/oracle/make_reference.py <examples checkout> <build dir>

Each simulation case runs upstream's initialize (fcc lattice; velocities
for molecular dynamics) and then the program, each seeded with
MINSTD_SEED, both reading namelists as upstream's README describes. The
reference keeps the "Run averages" and "Run errors" lines, every block's
line, and the final configuration (cnf.out), all printed with 17
significant digits by the exact build. The analysis cases run
pair_distribution and diffusion on configurations md_nve_lj saved after
each of its blocks (stored too, so the port reads the same numbers), and
error_calc and eos_lj on their own.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "upstream_reference.json")
FRAMES = os.path.join(HERE, "upstream_frames.json")
SEED = 1

# name -> (program, initialize namelist, program namelist, port program, port parameters)
SIMULATIONS = {
    "md_nve_small": ("md_nve_lj", "nc=3, velocities=.true.", "nblock=2, nstep=20", "md_nve", {"n": 108, "nblock": 2, "nstep": 20}),
    "md_nve_ll": ("md_nve_lj_ll", "nc=5, velocities=.true.", "nblock=2, nstep=10", "md_nve",
                  {"n": 500, "nblock": 2, "nstep": 10, "neighbors": "link"}),
    "md_nve_vl": ("md_nve_lj_vl", "nc=4, velocities=.true.", "nblock=2, nstep=50", "md_nve",
                  {"n": 256, "nblock": 2, "nstep": 50, "neighbors": "verlet"}),
    "md_nve_long": ("md_nve_lj", "nc=4, velocities=.true.", "nblock=5, nstep=400", "md_nve",
                    {"n": 256, "nblock": 5, "nstep": 400}),
    "md_nvt_small": ("md_nvt_lj", "nc=3, velocities=.true.", "nblock=2, nstep=20", "md_nvt", {"n": 108, "nblock": 2, "nstep": 20}),
    "mc_nvt_small": ("mc_nvt_lj", "nc=3", "nblock=2, nstep=5", "mc_nvt", {"n": 108, "nblock": 2, "nstep": 5}),
    "mc_nvt_long": ("mc_nvt_lj", "nc=4", "nblock=4, nstep=100", "mc_nvt", {"n": 256, "nblock": 4, "nstep": 100}),
    "mc_npt_small": ("mc_npt_lj", "nc=4", "nblock=2, nstep=5", "mc_npt", {"n": 256, "nblock": 2, "nstep": 5}),
}
EOS = {"liquid": (1.0, 0.75), "gas": (2.0, 0.1), "dense": (0.9, 0.85), "supercritical": (1.5, 0.5)}
# the trajectory for pair_distribution and diffusion: 30 blocks of 5 steps of 108 atoms
TRAJ_INIT = "nc=3, velocities=.true."
TRAJ_RUN = "nblock=30, nstep=5"
TRAJ_DT = 0.005
ERROR_CALC = "nstep=4096, nequil=1000, n_repeat=4"
NUM = r"-?\d\.\d+E[+-]\d+"


def nums(line):
    return [float(x) for x in re.findall(NUM, line)]


def run(binary, namelist, cwd):
    env = dict(os.environ, MINSTD_SEED=str(SEED))
    return subprocess.run([binary], input="&nml %s /\n" % namelist, capture_output=True, text=True, cwd=cwd,
                          env=env, check=True).stdout


def read_cnf(path):
    with open(path) as f:
        lines = f.read().split("\n")
    n = int(lines[0])
    box = float(lines[1])
    rows = [nums(line) for line in lines[2:2 + n]]
    return {"n": n, "box": box, "rows": rows}


def simulation(build, prog, init_nml, run_nml):
    with tempfile.TemporaryDirectory() as d:
        run(os.path.join(build, "initialize"), init_nml, d)
        out = run(os.path.join(build, prog), run_nml, d)
        lines = out.split("\n")
        avg = err = None
        blocks = []
        for i, line in enumerate(lines):
            if line.strip().startswith("Run averages"):
                avg = nums(line)
            elif line.strip().startswith("Run errors"):
                err = nums(line)
            elif re.match(r"^\s+\d+\s+-?\d\.\d+E", line):
                blocks.append(nums(line))
        cnf = read_cnf(os.path.join(d, "cnf.out"))
        return {"run_averages": avg, "run_errors": err, "blocks": blocks, "final": cnf}


def eos(build, t, rho):
    with tempfile.TemporaryDirectory() as d:
        out = run(os.path.join(build, "eos_lj"), "temperature=%r, density=%r" % (t, rho), d)
    section = None
    res = {}
    keys = {"Pressure P": "p", "Energy E/N": "e", "Heat capacity Cv/NkB": "cv", "Heat capacity Cp/NkB": "cp",
            "Chemical potential mu": "mu", "Activity z": "z"}
    for line in out.split("\n"):
        if line.startswith("Full Lennard-Jones"):
            section = "full"
        elif line.startswith("Lennard-Jones potential cut (but not shifted)"):
            section = "cut"
        elif line.startswith("Lennard-Jones potential cut-and-shifted"):
            section = "cs"
        for label, key in keys.items():
            if line.startswith(label) and section:
                res["%s_%s" % (key, section)] = nums(line)[0]
        m = re.match(r"^Ares(\d)(\d)", line)
        if m and section:
            res["a%s%s_%s" % (m.group(1), m.group(2), section)] = nums(line)[0]
    return res


def trajectory(build):
    """md_nve_lj's configuration after each block, renamed cnf.000, ...; and
    pair_distribution and diffusion run on them."""
    with tempfile.TemporaryDirectory() as d:
        run(os.path.join(build, "initialize"), TRAJ_INIT, d)
        run(os.path.join(build, "md_nve_lj"), TRAJ_RUN, d)
        nblock = int(re.search(r"nblock=(\d+)", TRAJ_RUN).group(1))
        frames = []
        for b in range(1, nblock + 1):
            shutil.copy(os.path.join(d, "cnf.%03d" % b), os.path.join(d, "f.%03d" % (b - 1)))
        for b in range(nblock):
            os.replace(os.path.join(d, "f.%03d" % b), os.path.join(d, "cnf.%03d" % b))
            frames.append(read_cnf(os.path.join(d, "cnf.%03d" % b)))
        # the last block's file, now a copy one number down: upstream's
        # programs read cnf.000, cnf.001, ... until a number is missing
        os.remove(os.path.join(d, "cnf.%03d" % nblock))
        run(os.path.join(build, "pair_distribution"), "dr=0.02", d)
        gr = [nums(line) for line in open(os.path.join(d, "pair_distribution.out")) if nums(line)]
        nstep = int(re.search(r"nstep=(\d+)", TRAJ_RUN).group(1))
        run(os.path.join(build, "diffusion"), "nt=20, origin_interval=4, delta=%r" % (nstep * TRAJ_DT), d)
        diff = [nums(line) for line in open(os.path.join(d, "diffusion.out")) if nums(line)]
    return frames, gr, diff


def error_calc(build):
    with tempfile.TemporaryDirectory() as d:
        out = run(os.path.join(build, "error_calc"), ERROR_CALC, d)
    res = {"lines": []}
    for line in out.split("\n"):
        for label, key in (("Exact value of SI", "si_exact"), ("Exact error estimate", "err_exact"),
                           ("Empirical error estimate", "err_brute"), ("Sample average value", "mean"),
                           ("Sample variance", "var"), ("Error estimate neglecting correlations", "err_naive")):
            if line.startswith(label):
                res[key] = nums(line)[0]
        m = re.match(r"^\s+(\d+)\s+(\d+)\s+(" + NUM + r")\s+(" + NUM + ")", line)
        if m:
            res["lines"].append([int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))])
    return res


def main():
    checkout, build_dir = sys.argv[1], sys.argv[2]
    build = os.path.abspath(os.path.join(build_dir, "exact"))
    ref = {
        "upstream": "https://github.com/Allen-Tildesley/examples",
        "commit": subprocess.run(["git", "-C", checkout, "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip(),
        "oracle": "ports/liquids/oracle/build.py's exact build: gfortran -O2 -fdefault-real-8 -fall-intrinsics "
                  "-ffp-contract=off -fno-fast-math, MINSTD random numbers, Viv's libm, 17-digit output",
        "gfortran": subprocess.run(["gfortran", "--version"], capture_output=True, text=True).stdout.split("\n")[0],
        "seed": SEED,
        "simulations": {},
        "eos": {},
    }
    for name, (prog, init_nml, run_nml, port, params) in SIMULATIONS.items():
        ref["simulations"][name] = dict(program=prog, initialize=init_nml, namelist=run_nml, port=port,
                                        params=params, **simulation(build, prog, init_nml, run_nml))
        print(name, ref["simulations"][name]["run_averages"][:3])
    for name, (t, rho) in EOS.items():
        ref["eos"][name] = dict(temperature=t, density=rho, **eos(build, t, rho))
    frames, gr, diff = trajectory(build)
    ref["pair_distribution"] = {"namelist": "dr=0.02", "out": gr}
    nstep = int(re.search(r"nstep=(\d+)", TRAJ_RUN).group(1))
    ref["diffusion"] = {"nt": 20, "origin_interval": 4, "delta": nstep * TRAJ_DT, "out": diff}
    ref["error_calc"] = dict(namelist=ERROR_CALC, **error_calc(build))
    with open(OUT, "w") as f:
        json.dump(ref, f, indent=1)
        f.write("\n")
    with open(FRAMES, "w") as f:
        json.dump({"source": "md_nve_lj (exact build), %s then %s, each block's configuration" % (TRAJ_INIT, TRAJ_RUN),
                   "frames": frames}, f)
        f.write("\n")
    print("wrote", OUT, "and", FRAMES)


if __name__ == "__main__":
    main()
