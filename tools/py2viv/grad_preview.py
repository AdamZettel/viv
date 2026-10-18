#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""grad_preview.py: a preview of gradients through py2viv. A NumPy MLP loss
(examples/mlp_grad.py) is translated by py2viv and differentiated by Viv's value_and_grad in
three engines; the loss and both gradients are compared with a float64 torch twin
(loss.backward()) and with the complex-step reference diff.py uses.

    python3 tools/py2viv/grad_preview.py [--out DIR]

Needs torch (checked with `import torch`); without it, only the diff.py comparison runs.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import diff  # noqa: E402


def torch_twin():
    import torch
    torch.set_default_dtype(torch.float64)
    N, D, K, O = 16, 4, 8, 2
    X = torch.zeros(N, D)
    Y = torch.zeros(N, O)
    W1 = torch.zeros(D, K)
    W2 = torch.zeros(K, O)
    for i in range(N):
        for j in range(D):
            X[i, j] = math.sin(0.3 * i + 0.7 * j)
        for j in range(O):
            Y[i, j] = math.cos(0.5 * i - 0.2 * j)
    for i in range(D):
        for j in range(K):
            W1[i, j] = 0.5 * math.sin(1.3 * i + 0.4 * j + 0.1)
    for i in range(K):
        for j in range(O):
            W2[i, j] = 0.5 * math.cos(0.9 * i + 1.1 * j)
    W1.requires_grad_(True)
    W2.requires_grad_(True)
    H = torch.tanh(X @ W1)
    P = H @ W2
    d = P - Y
    loss = torch.sum(d * d) / d.numel()
    loss.backward()
    return {"loss": [loss.item()], "g1": W1.grad.reshape(-1).tolist(), "g2": W2.grad.reshape(-1).tolist(),
            "version": torch.__version__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "corpus_out", "grad_preview"))
    args = ap.parse_args()
    prog = os.path.join(HERE, "examples", "mlp_grad.py")
    rep = diff.differential(prog, args.out)
    print(diff.format_report(rep, max_rows=6))
    events = diff.parse_viv(open(os.path.join(args.out, "interp.out")).read())[0]
    viv = {"loss": events[0]["vals"][(0, "")], "g1": events[1]["vals"][(0, "")], "g2": events[2]["vals"][(0, "")]}
    try:
        tw = torch_twin()
    except ImportError:
        print("torch not installed: no torch comparison")
        return 0 if rep["verdict"].startswith("PASS") else 1
    print("torch %s float64 twin (loss.backward()) against Viv (identical in interp, C, wasm: %s):"
          % (tw["version"], "yes" if rep.get("engines_agree") else "NO"))
    worst = 0.0
    out = {}
    for k in ("loss", "g1", "g2"):
        a, b = tw[k], viv[k]
        scale = max(abs(x) for x in a)
        dev = max(abs(x - y) for x, y in zip(a, b))
        worst = max(worst, dev / scale)
        out[k] = {"n": len(a), "max_abs_dev": dev, "largest_entry": scale, "dev_over_largest": dev / scale,
                  "torch_first": a[0], "viv_first": b[0]}
        print("  %-5s %3d values: max |viv - torch| = %.2g, %.2g of the largest entry (torch %r, viv %r)"
              % (k, len(a), dev, dev / scale, a[0], b[0]))
    ok = worst <= 1e-14 and rep.get("engines_agree")
    print("verdict: %s (criterion: every value within 1e-14 of the largest entry, engines identical)" % ("PASS" if ok else "FAIL"))
    with open(os.path.join(args.out, "torch_compare.json"), "w") as f:
        json.dump(out, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
