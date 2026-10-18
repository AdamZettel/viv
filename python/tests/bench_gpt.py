# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Training time of vp.gpt against the NumPy program
study/compare/ml/tiny_gpt.py and the PyTorch reimplementation of
python/tests/gpt_reference.py, at three sizes, same steps and same data.

    python python/tests/bench_gpt.py [--repeat 3]

Each time is the best of `--repeat` runs of the whole training loop (the
compile of lib/gpt/api.viv is done before timing). PyTorch runs on one
thread, and NumPy's BLAS is held to one thread too, as vp.gpt runs. The
machine's load average is printed with the table.
"""
import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_v] = "1"  # before NumPy loads its BLAS

import argparse  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import runpy  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import vipyr as vp  # noqa: E402
import gpt_reference as ref  # noqa: E402

TEXT = (ROOT / "LICENSE-MIT").read_text(encoding="utf-8")
SIZES = [
    ("small", vp.gpt.GptConfig(), 50),
    (
        "medium",
        vp.gpt.GptConfig(context=16, width=32, heads=4, layers=3, hidden=96, batch=8),
        50,
    ),
    (
        "large",
        vp.gpt.GptConfig(context=32, width=64, heads=4, layers=4, hidden=256, batch=16),
        20,
    ),
]


def best(f, repeat):
    times, out = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = f()
        times.append(time.perf_counter() - t0)
    return min(times), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    import torch

    torch.set_num_threads(1)
    with contextlib.redirect_stdout(io.StringIO()):
        twin = runpy.run_path(str(ROOT / "study" / "compare" / "ml" / "tiny_gpt.py"))
    vocab, ids = vp._gpt.tokenize(TEXT)
    seed = 7
    vp.gpt.train(TEXT, steps=1)  # compile both builds once, outside the timing
    vp.gpt.train(TEXT, steps=1, fast=True)
    print("load average before: %.2f %.2f %.2f" % os.getloadavg())
    print(
        "%-7s %8s %6s  %10s %10s %10s %10s   %s"
        % ("size", "params", "steps", "vp.gpt", "fast=True", "NumPy", "PyTorch", "largest loss difference")
    )
    for name, cfg, steps in SIZES:
        t_vp, r = best(
            lambda: vp.gpt.train(TEXT, config=cfg, steps=steps, seed=seed),
            a.repeat,
        )
        t_fast, _ = best(lambda: vp.gpt.train(TEXT, config=cfg, steps=steps, seed=seed, fast=True), a.repeat)
        c = r.model.config
        tc = twin["Config"](
            **{
                k: getattr(c, k)
                for k in (
                    "vocab",
                    "context",
                    "width",
                    "heads",
                    "layers",
                    "hidden",
                    "batch",
                )
            }
        )

        def numpy_run():
            model = twin["GPT"](tc, twin["Minstd"](seed))
            with contextlib.redirect_stdout(io.StringIO()):
                return np.array(twin["train"](model, ids.astype(int), steps))

        t_np, l_np = best(numpy_run, a.repeat)

        def torch_run():
            m = ref.TorchGpt(c, ref.init_arrays(c, seed))
            return ref.train(m, ids.astype(int), steps)

        t_t, l_t = best(torch_run, a.repeat)
        diff = max(np.abs(r.losses - l_np).max(), np.abs(r.losses - l_t).max())
        print(
            "%-7s %8d %6d  %9.3fs %9.3fs %9.3fs %9.3fs   %.1e"
            % (name, r.model.size, steps, t_vp, t_fast, t_np, t_t, diff)
        )
    print("load average after: %.2f %.2f %.2f" % os.getloadavg())


if __name__ == "__main__":
    main()
