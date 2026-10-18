# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""A PyTorch reimplementation of lib/gpt's model and optimizers, for
python/tests/test_gpt.py and bench/gpt/bench_gpt.py.

The model is built from a vp.gpt.Model's arrays (or from the same draws of
Viv's rand(): `init_arrays`), trained with autograd, a Muon written as
modded-nanogpt's (momentum, then five Newton-Schulz steps with its
coefficients) and torch.optim.Adam, in float64 on the CPU. Nothing here
calls Viv, so agreement with vp.gpt is a check of both.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

MUON_LR, MUON_MOMENTUM = 0.05, 0.95
ADAM_LR, ADAM_BETAS, ADAM_EPS = 0.05, (0.9, 0.99), 1e-8
LR_FLOOR = 0.1
NS_STEPS, NS_COEFFS = 5, (3.4445, -4.775, 2.0315)
RMS_EPS = 1e-8


class Minstd:
    """Viv's rand(): Park-Miller MINSTD, and randn by Box-Muller as lib/random."""

    def __init__(self, seed):
        self.state = math.floor(abs(seed)) % 2147483646 + 1

    def rand(self):
        self.state = 16807 * self.state % 2147483647
        return self.state / 2147483647

    def randn(self, rows, cols):
        out = np.empty((rows, cols))
        for i in range(rows):
            for j in range(cols):
                u1 = self.rand()
                u2 = self.rand()
                out[i, j] = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return out


def init_arrays(cfg, seed):
    """The initial matrices of lib/gpt's Gpt(cfg) after srand(seed), in for_leaves order, by name."""
    rng = Minstd(seed)
    hd = cfg.width // cfg.heads
    out = {
        "tok": rng.randn(cfg.vocab, cfg.width),
        "pos": rng.randn(cfg.context, cfg.width),
    }
    for i in range(cfg.layers):
        for k in range(cfg.heads):
            for n in ("Wq", "Wk", "Wv"):
                out["blocks[%d].attn.heads[%d].%s" % (i, k, n)] = rng.randn(cfg.width, hd) / math.sqrt(cfg.width)
        out["blocks[%d].attn.Wo" % i] = rng.randn(cfg.width, cfg.width) / math.sqrt(cfg.width)
        out["blocks[%d].mlp.up" % i] = rng.randn(cfg.width, cfg.hidden) / math.sqrt(cfg.width)
        out["blocks[%d].mlp.down" % i] = rng.randn(cfg.hidden, cfg.width) / math.sqrt(cfg.hidden)
    out["unembed"] = np.zeros((cfg.width, cfg.vocab))
    return out


def rmsnorm(x):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)


class TorchGpt:
    """The model as a dict of leaf tensors named as vp.gpt.Model's parameters()."""

    def __init__(self, cfg, arrays):
        self.cfg = cfg
        self.p = {n: torch.tensor(np.asarray(w, dtype=np.float64), requires_grad=True) for n, w in arrays.items()}

    def forward(self, idx):
        c, p = self.cfg, self.p
        x = p["tok"][idx] + p["pos"][: idx.shape[1]]
        for i in range(c.layers):
            n = rmsnorm(x)
            heads = [
                F.scaled_dot_product_attention(
                    n @ p["blocks[%d].attn.heads[%d].Wq" % (i, k)],
                    n @ p["blocks[%d].attn.heads[%d].Wk" % (i, k)],
                    n @ p["blocks[%d].attn.heads[%d].Wv" % (i, k)],
                    is_causal=True,
                )
                for k in range(c.heads)
            ]
            x = x + torch.cat(heads, dim=-1) @ p["blocks[%d].attn.Wo" % i]
            x = x + F.relu(rmsnorm(x) @ p["blocks[%d].mlp.up" % i]).square() @ p["blocks[%d].mlp.down" % i]
        return rmsnorm(x) @ p["unembed"]

    def loss(self, x, y):
        logits = self.forward(x)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def newton_schulz(g, steps=NS_STEPS, coeffs=NS_COEFFS):
    a, b, c = coeffs
    if g.shape[0] > g.shape[1]:
        return newton_schulz(g.T, steps, coeffs).T
    norm = g.norm()
    x = g / norm if norm > 0 else g
    for _ in range(steps):
        s = x @ x.T
        x = a * x + b * (s @ x) + c * ((s @ s) @ x)
    return x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                buf = self.state[p].setdefault("buf", torch.zeros_like(p))
                buf.mul_(group["momentum"]).add_(p.grad)
                rows, cols = p.shape
                p.sub_(group["lr"] * math.sqrt(max(1.0, rows / cols)) * newton_schulz(buf))


def batch(tokens, cfg, step, shift):
    windows = len(tokens) - cfg.context
    starts = [(step * cfg.batch + b) % windows for b in range(cfg.batch)]
    return torch.stack([tokens[s + shift : s + shift + cfg.context] for s in starts])


def train(model, tokens, steps):
    """Train in place; the loss of every step."""
    tokens = torch.as_tensor(np.asarray(tokens), dtype=torch.long)
    blocks = [t for n, t in model.p.items() if n.startswith("blocks")]
    others = [t for n, t in model.p.items() if not n.startswith("blocks")]
    opts = [
        Muon(blocks, MUON_LR, MUON_MOMENTUM),
        torch.optim.Adam(others, ADAM_LR, betas=ADAM_BETAS, eps=ADAM_EPS),
    ]
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lambda s: 1.0 - (1.0 - LR_FLOOR) * s / steps) for o in opts]
    losses = []
    for step in range(steps):
        x, y = batch(tokens, model.cfg, step, 0), batch(tokens, model.cfg, step, 1)
        loss = model.loss(x, y)
        for t in model.p.values():
            t.grad = None
        loss.backward()
        for o in opts + scheds:
            o.step()
        losses.append(loss.item())
    return np.array(losses)
