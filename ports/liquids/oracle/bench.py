# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Times the port against upstream's Fortran and upstream's Python on the
same runs, and writes ports/liquids/timings.json.

    python3 ports/liquids/oracle/bench.py <examples checkout> <build dir>

<build dir> is build.py's; the Fortran timed is its stock build (upstream's
sources and flags with -O2, RANDOM_NUMBER, the platform's libm). The
Python is upstream's python_examples, run as its README says (JSON
parameters on stdin) with this machine's python3 and NumPy. The port runs
through vipyr's exact build (cc -O2 -ffp-contract=off), and its fast build
(fast=True: -O3 -ffp-contract=fast, which may change the last bits). Each
time is the
wall time of the run alone (the starting configuration is written first,
outside the timing; vipyr's library is built and loaded first), the best of
REPEAT; the load average before each set is recorded.
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "..", "timings.json")
REPEAT = 2

RUNS = [
    # name, upstream program, initialize namelist, Fortran namelist, Python JSON, port program, port parameters
    ("md_nve N=256, 10 x 200 steps", "md_nve_lj", "nc=4, velocities=.true.", "nblock=10, nstep=200",
     {"nblock": 10, "nstep": 200}, "md_nve", {"n": 256, "nblock": 10, "nstep": 200}),
    ("md_nve link cells N=864, 10 x 200 steps", "md_nve_lj_ll", "nc=6, velocities=.true.", "nblock=10, nstep=200",
     None, "md_nve", {"n": 864, "nblock": 10, "nstep": 200, "neighbors": "link"}),
    ("md_nvt N=256, 10 x 200 steps", "md_nvt_lj", "nc=4, velocities=.true.", "nblock=10, nstep=200",
     None, "md_nvt", {"n": 256, "nblock": 10, "nstep": 200}),
    ("mc_nvt N=256, 10 x 100 sweeps", "mc_nvt_lj", "nc=4", "nblock=10, nstep=100",
     {"nblock": 10, "nstep": 100}, "mc_nvt", {"n": 256, "nblock": 10, "nstep": 100}),
]


def load():
    return os.getloadavg()[0]


def best(fn):
    times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    checkout, build_dir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    stock = os.path.join(build_dir, "stock")
    sys.path.insert(0, os.path.join(ROOT, "python"))
    import vipyr
    out = {"machine": "%s %s, %s" % (platform.system(), platform.machine(), platform.processor()),
           "python": sys.version.split()[0], "numpy": __import__("numpy").__version__,
           "gfortran": subprocess.run(["gfortran", "--version"], capture_output=True, text=True).stdout.split("\n")[0],
           "date": time.strftime("%Y-%m-%d"), "runs": []}
    for name, prog, init_nml, nml, py_json, port, params in RUNS:
        row = {"run": name, "load_before": round(load(), 1)}
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([os.path.join(stock, "initialize")], input="&nml %s /\n" % init_nml, text=True,
                           capture_output=True, cwd=d, check=True)
            shutil.copy(os.path.join(d, "cnf.inp"), os.path.join(d, "cnf.start"))

            def fortran():
                shutil.copy(os.path.join(d, "cnf.start"), os.path.join(d, "cnf.inp"))
                subprocess.run([os.path.join(stock, prog)], input="&nml %s /\n" % nml, text=True,
                               capture_output=True, cwd=d, check=True)
            row["fortran_s"] = round(best(fortran), 3)
            if py_json is not None:
                pydir = os.path.join(checkout, "python_examples")

                def python():
                    shutil.copy(os.path.join(d, "cnf.start"), os.path.join(d, "cnf.inp"))
                    subprocess.run([sys.executable, os.path.join(pydir, prog + ".py")], input=json.dumps(py_json),
                                   text=True, capture_output=True, cwd=d, check=True,
                                   env=dict(os.environ, PYTHONPATH=pydir))
                row["python_s"] = round(best(python), 3)
        mod = vipyr.load(os.path.join(HERE, "..", port + ".viv"))
        row["viv_s"] = round(best(lambda: mod.run(**params)), 3)
        fast = vipyr.load(os.path.join(HERE, "..", port + ".viv"), fast=True)
        row["viv_fast_s"] = round(best(lambda: fast.run(**params)), 3)
        row["load_after"] = round(load(), 1)
        out["runs"].append(row)
        print(row, flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
