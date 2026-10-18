# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vivscf.viv (one program, the molecule as parameters) against the generated
per-case timing programs viv/<case>_t.viv, natively, at port.toml's timing
settings.

    python3 study/ports/pyscf/vivscf_timing.py [case ...]

Both are built by the same compiler (VIV, default target/release/viv) in the
exact mode, and both run the same lib/hf path (`scf_with`). Each process is
timed whole (wall clock): the molecule, the integrals, the SCF and the exit.
Per case: one untimed run of each (the first exec of a new binary is slow on
macOS), then `reps` runs of each, old and new alternating so that a change of
load hits both; the median of each. Cases with `reps` in port.toml (benzene,
(H2O)8) run once each with no warm-up. Every run's energy must be within
timing_energy_tol of the oracle or the case is marked. Writes
results/vivscf_timing.json.
"""
import json
import os
import statistics
import subprocess
import sys
import time
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import vivscf_native  # noqa: E402
from test_vivscf_vs_pyscf import params, case_order  # noqa: E402

CONF = tomllib.load(open(os.path.join(HERE, "port.toml"), "rb"))
T = CONF["timing"]


def build_old(name):
    exe = os.path.join(vivscf_native.BUILD, "old_" + name + "_t")
    src = os.path.join(HERE, "viv", name + "_t.viv")
    env = dict(os.environ, VIV_JOBS=os.environ.get("VIV_JOBS", "2"))
    r = subprocess.run([vivscf_native.viv_bin(), "build", src, "-o", exe], cwd=REPO, capture_output=True, text=True, env=env)
    if r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    return exe


def timed(exe):
    t0 = time.perf_counter()
    r = subprocess.run([exe], capture_output=True, text=True)
    wall = time.perf_counter() - t0
    e = None
    for line in r.stdout.splitlines():
        k, _, v = line.partition(" ")
        if k in ("energy", "e_tot"):
            e = float(v)
    return wall, e


def main(names):
    if names:
        os.environ["VIVSCF_CASES"] = ",".join(names)
    out = {"settings": {"e_tol": T["viv_e_tol"], "grad_tol": T["viv_grad_tol"]}, "compiler": vivscf_native.viv_bin(), "rows": []}
    path = os.path.join(HERE, "results", "vivscf_timing.json")
    if os.path.exists(path):
        old = json.load(open(path))
        keep = {r["case"]: r for r in old.get("rows", [])}
    else:
        keep = {}
    for c in case_order():
        name = c["name"]
        o = json.load(open(os.path.join(HERE, "oracle", name + ".json")))
        p = params(c)
        p.update(e_tol=T["viv_e_tol"], grad_tol=T["viv_grad_tol"])
        new_exe = vivscf_native.build(dict(p, print_arrays=False))
        old_exe = build_old(name)
        reps = c.get("reps", T["reps_warm"])
        if "reps" not in c:
            timed(old_exe)
            timed(new_exe)
        tn, to, bad = [], [], []
        load0 = os.getloadavg()[0]
        for i in range(reps):
            for exe, acc, tag in ((old_exe, to, "old"), (new_exe, tn, "new")):
                w, e = timed(exe)
                acc.append(w)
                if e is None or abs(e - float(o["e_tot"])) >= T["timing_energy_tol"]:
                    bad.append(f"{tag} energy {e!r}")
        row = {"case": name, "nao": o["nao"], "reps": reps, "old_s": statistics.median(to), "new_s": statistics.median(tn),
               "old_runs": to, "new_runs": tn, "load_start": load0, "load_end": os.getloadavg()[0], "bad": bad}
        row["ratio"] = row["new_s"] / row["old_s"]
        keep[name] = row
        print(f"{name:22s} {o['nao']:4d} AOs  old {row['old_s']:9.4f} s  new {row['new_s']:9.4f} s  new/old {row['ratio']:.3f}  "
              f"load {load0:.1f}->{row['load_end']:.1f}" + (("  " + "; ".join(bad)) if bad else ""), flush=True)
    out["rows"] = [keep[k] for k in keep]
    out["taken"] = time.strftime("%Y-%m-%d %H:%M")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main(sys.argv[1:])
