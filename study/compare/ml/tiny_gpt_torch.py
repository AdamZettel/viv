# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ imports
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

# @@ config
@dataclass
class Config:
    vocab: int
    context: int = 8
    width: int = 16
    heads: int = 2
    layers: int = 2
    hidden: int = 64
    batch: int = 4


MUON_LR, MUON_MOMENTUM = 0.05, 0.95
ADAM_LR, ADAM_BETAS = 0.05, (0.9, 0.99)
LR_FLOOR = 0.1
TEXT, PROMPT = "hello world! " * 5, "hello"
STEPS, SAMPLES = 30, 24
INIT_SEED, SAMPLE_SEED = 7, 11

# @@ random draws
class Minstd:
    """Viv's rand(), so the weights and the samples are the ones the Viv program draws."""

    def __init__(self, seed):
        self.state = math.floor(abs(seed)) % 2147483646 + 1

    def rand(self):
        self.state = 16807 * self.state % 2147483647
        return self.state / 2147483647

    def randn(self, rows, cols):
        u = [(self.rand(), self.rand()) for _ in range(rows * cols)]
        z = [math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2) for u1, u2 in u]
        return torch.tensor(z).reshape(rows, cols)

    def choice(self, p):
        return min(int(np.searchsorted(np.cumsum(p), self.rand(), side="right")), len(p) - 1)

# @@ parameters and init
@torch.no_grad()
def init_from(model, rng):
    """N(0, 1) embeddings, N(0, 1/fan_in) matrices, a zero output map, drawn in the Viv program's order."""
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.weight.copy_(rng.randn(*m.weight.shape))
        elif isinstance(m, nn.Linear) and m is not model.unembed:
            m.weight.copy_(rng.randn(m.in_features, m.out_features).T / math.sqrt(m.in_features))
    nn.init.zeros_(model.unembed.weight)

# @@ embedding
# nn.Embedding: model.tok and model.pos below

# @@ attention
class Head(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        size = cfg.width // cfg.heads
        self.q, self.k, self.v = (nn.Linear(cfg.width, size, bias=False) for _ in range(3))

    def forward(self, x):
        return F.scaled_dot_product_attention(self.q(x), self.k(x), self.v(x), is_causal=True)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = nn.ModuleList(Head(cfg) for _ in range(cfg.heads))
        self.out = nn.Linear(cfg.width, cfg.width, bias=False)

    def forward(self, x):
        return self.out(torch.cat([h(x) for h in self.heads], dim=-1))

# @@ block
def rmsnorm(x):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-8)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = Attention(cfg)
        self.up = nn.Linear(cfg.width, cfg.hidden, bias=False)
        self.down = nn.Linear(cfg.hidden, cfg.width, bias=False)

    def forward(self, x):
        x = x + self.attn(rmsnorm(x))
        return x + self.down(F.relu(self.up(rmsnorm(x))).square())

# @@ forward
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.width)
        self.pos = nn.Embedding(cfg.context, cfg.width)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.unembed = nn.Linear(cfg.width, cfg.vocab, bias=False)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        for block in self.blocks:
            x = block(x)
        return self.unembed(rmsnorm(x))

# @@ loss
# F.cross_entropy, in the training loop

# @@ backward
# autograd: loss.backward() in the training loop fills every p.grad

# @@ optimizer
def newton_schulz(g, steps=5, a=3.4445, b=-4.775, c=2.0315):
    if g.shape[0] > g.shape[1]:
        return newton_schulz(g.T, steps, a, b, c).T
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
                cols, rows = p.shape  # a Linear weight is stored (out, in)
                p.sub_(group["lr"] * math.sqrt(max(1.0, rows / cols)) * newton_schulz(buf.T).T)

# @@ training loop
def get_batch(tokens, cfg, step, shift):
    windows = len(tokens) - cfg.context
    starts = [(step * cfg.batch + b) % windows for b in range(cfg.batch)]
    return torch.stack([tokens[s + shift : s + shift + cfg.context] for s in starts])


def train(model, tokens, steps):
    matrices = [p for name, p in model.named_parameters() if name.startswith("blocks.")]
    others = [p for name, p in model.named_parameters() if not name.startswith("blocks.")]
    opts = [Muon(matrices, MUON_LR, MUON_MOMENTUM), torch.optim.Adam(others, ADAM_LR, betas=ADAM_BETAS)]
    schedules = [torch.optim.lr_scheduler.LambdaLR(o, lambda s: 1.0 - (1.0 - LR_FLOOR) * s / steps) for o in opts]
    losses = []
    for step in range(steps):
        x, y = get_batch(tokens, model.cfg, step, 0), get_batch(tokens, model.cfg, step, 1)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        model.zero_grad()
        loss.backward()
        for o in opts + schedules:
            o.step()
        losses.append(loss.item())
        print(f"step {step + 1}: loss {loss.item():.6f}")
    return losses

# @@ sampling
@torch.no_grad()
def generate(model, prompt, count, rng):
    out = list(prompt)
    for _ in range(count):
        logits = model(torch.tensor([out[-model.cfg.context :]]))
        out.append(rng.choice(F.softmax(logits[0, -1], dim=-1).numpy()))
    return out

# @@ text as tokens
vocab = sorted(set(TEXT))
stoi = {c: i for i, c in enumerate(vocab)}
encode = lambda text: [stoi[c] for c in text]
decode = lambda ids: "".join(vocab[i] for i in ids)

# @@ run
cfg = Config(vocab=len(vocab))
model = GPT(cfg)
init_from(model, Minstd(INIT_SEED))
count = sum(p.numel() for p in model.parameters())
print(f"{count} parameters, {len(vocab)} characters")
losses = train(model, torch.tensor(encode(TEXT)), STEPS)
written = decode(generate(model, encode(PROMPT), SAMPLES, Minstd(SAMPLE_SEED)))
print(written)

# @@ checks
assert count == 6560 and len(vocab) == 9
assert abs(losses[0] - math.log(9.0)) <= 1e-15
assert abs(losses[-1] - 0.199118) <= 1e-6
assert written == "hello world! hello world! hel"
