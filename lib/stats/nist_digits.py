# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The digits lens on NIST's hardest linear regressions.

    python3 lib/stats/nist_digits.py     # writes lib/stats/nist_digits.json

Runs `viv digits` (reduction order, FMA contraction and Monte Carlo
arithmetic, 16 samples) on lib/stats/lens/nist_digits.viv for Filip, Longley and
Wampler1-5 with each method of the solver slot, and keeps the fewest stable
digits among the printed coefficients. Monte Carlo arithmetic perturbs the
whole program, the powers of x in the design matrix included, so the stable
digits measure the problem as much as the solver; the column beside them is
the certified digits of the same run (lib/stats/nist_lls.py's LRE).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import nist_lls  # noqa: E402

DEGREE = {"Filip": 10, "Longley": 0}
METHODS = {0: "qr, refined", 1: "qr", 2: "cholesky"}


def main():
    viv = os.environ.get("VIV", str(ROOT / "target" / "release" / "viv"))
    out = {"about": __doc__.split("\n\n")[1].replace("\n", " "), "runs": {}}
    for name in [
        "Filip",
        "Longley",
        "Wampler1",
        "Wampler2",
        "Wampler3",
        "Wampler4",
        "Wampler5",
    ]:
        y, X, cert = nist_lls.read(name)
        for m, label in METHODS.items():
            cmd = [
                viv,
                "digits",
                "--json",
                "--attrib",
                "0",
                "--trace",
                "0",
                "--set",
                'DATASET="%s"' % name,
                "--set",
                "DEGREE=%d" % DEGREE.get(name, 5),
                "--set",
                "METHOD=%d" % m,
                "lib/stats/lens/nist_digits.viv",
            ]
            d = json.loads(
                subprocess.run(
                    cmd, cwd=ROOT, check=True, capture_output=True, text=True
                ).stdout
            )
            vals = np.array([o["value"] for o in d["outputs"]])
            stable = min(o["stable"] for o in d["outputs"])
            out["runs"]["%s / %s" % (name, label)] = dict(
                stable=stable,
                certified=float(nist_lls.lre(vals, cert["coef"]).min()),
            )
            print(
                "%-10s %-12s stable %5.1f   certified %5.1f"
                % (
                    name,
                    label,
                    stable,
                    out["runs"]["%s / %s" % (name, label)]["certified"],
                ),
                flush=True,
            )
    (HERE / "nist_digits.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
