# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""`viv digits` on four programs of the repository; writes
examples/digits/real_programs.json, which the gallery page prints.

    python3 examples/digits/real_programs.py               # runs them (minutes)
    python3 examples/digits/real_programs.py --reuse DIR   # reads DIR/<name>.json

Each run: reduction order (pairwise, left to right, reversed, 3 random), FMA
contraction, Monte Carlo arithmetic (16 samples), and the six most-executed
lines perturbed alone.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIV = ROOT / "target" / "release" / "viv"
PROGRAMS = [
    ("water RHF energy (STO-3G)", "water", "study/ports/pyscf/vivscf.viv", []),
    ("SISSO coefficients", "sisso", "study/ports/sisso/sisso.viv", []),
    ("2-D Ising, Metropolis, 16x16, 200 sweeps", "ising", "stat/ising/ising_mc.viv",
     ["sweeps=200", "thermalization=50"]),
    ("option prices", "options", "examples/options/digits_prices.viv", []),
]


def run(key, path, sets, reuse):
    if reuse:
        return json.loads((Path(reuse) / (key + ".json")).read_text())
    args = [str(VIV), "digits", "--attrib", "6", "--json"]
    for s in sets:
        args += ["--set", s]
    r = subprocess.run(args + [str(ROOT / path)], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def main(argv):
    reuse = argv[argv.index("--reuse") + 1] if "--reuse" in argv else None
    out = {"note": "written by examples/digits/real_programs.py", "programs": []}
    for name, key, path, sets in PROGRAMS:
        d = run(key, path, sets, reuse)
        # whole numbers that every run reproduces (sizes, indices) say nothing
        outs = [o for o in d["outputs"] if not (float(o["value"]).is_integer() and o["stable"] >= 15.95)]
        top = None
        for line in d["lines"]:
            for e in line.get("effects", []):
                if top is None or e["digits"] < top[0]:
                    top = (e["digits"], line)
        out["programs"].append({
            "name": name, "path": path, "sets": sets, "runs": d["runs"],
            "outputs": [{"label": o.get("label") or o["text"][:40], "value": o["value"], "order": o["order"], "fma": o["fma"],
                         "mca": o["mca"], "stable": o["stable"]} for o in outs],
            "failed": len(d["failed"]),
            "top_line": {"at": top[1]["at"], "code": top[1]["code"], "digits": top[0]} if top else None,
        })
    (ROOT / "examples" / "digits" / "real_programs.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv)
