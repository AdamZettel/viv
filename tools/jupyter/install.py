# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Install the Viv kernel spec so Jupyter (and VS Code) can start the kernel.

    python tools/jupyter/install.py              for this user (jupyter kernelspec install --user)
    python tools/jupyter/install.py --prefix P   under P/share/jupyter/kernels (the tests use this)

The spec runs `<this python> -m vivkernel` with tools/jupyter on PYTHONPATH,
and points VIV_BIN at this repository's release build of viv. Build it
first with `cargo build --release -p viv`. Re-running the script replaces
the spec, which is how to move it after the repository moves.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", help="install under PREFIX instead of for this user")
    ap.add_argument("--viv", default=os.environ.get("VIV_BIN") or os.path.join(REPO, "target", "release", "viv"),
                    help="the viv binary the kernel runs (default: this repository's release build)")
    ap.add_argument("--engine", choices=["interp", "native"], default="interp",
                    help="the engine cells use unless they say otherwise")
    args = ap.parse_args()

    viv = os.path.abspath(args.viv)
    if not os.path.exists(viv):
        sys.exit(f"no viv binary at {viv}: build it with `cargo build --release -p viv`, or pass --viv")

    spec = {
        "argv": [sys.executable, "-m", "vivkernel", "-f", "{connection_file}"],
        "display_name": "Viv",
        "language": "viv",
        "interrupt_mode": "signal",
        "env": {"PYTHONPATH": HERE, "VIV_BIN": viv, "VIV_ENGINE": args.engine},
        "metadata": {"debugger": False},
    }
    with tempfile.TemporaryDirectory() as d:
        kdir = os.path.join(d, "viv")
        os.mkdir(kdir)
        with open(os.path.join(kdir, "kernel.json"), "w") as f:
            json.dump(spec, f, indent=2)
        logo = os.path.join(HERE, "logo-64x64.png")
        if os.path.exists(logo):
            shutil.copy(logo, kdir)
        where = ["--prefix", args.prefix] if args.prefix else ["--user"]
        subprocess.run(["jupyter", "kernelspec", "install", kdir, "--name", "viv", "--replace", *where], check=True)


if __name__ == "__main__":
    main()
