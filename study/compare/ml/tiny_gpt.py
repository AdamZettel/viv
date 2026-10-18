# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ imports
import math
from dataclasses import dataclass

import numpy as np

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


MUON_LR = 0.05
MUON_MOMENTUM = 0.95
ADAM_LR = 0.05
ADAM_BETA1, ADAM_BETA2 = 0.9, 0.99
ADAM_EPS = 1e-8
LR_FLOOR = 0.1  # the step size falls in a straight line to this fraction of its start
EPS = 1e-8
MASKED = -1e30

TEXT = "hello world! " * 5
PROMPT = "hello"
STEPS = 30
SAMPLES = 24
INIT_SEED = 7
SAMPLE_SEED = 11

# @@ random draws
class Minstd:
    """Viv's rand(): Park-Miller MINSTD. Using it here makes both programs draw the same numbers."""

    def __init__(self, seed):
        self.state = math.floor(abs(seed)) % 2147483646 + 1

    def rand(self):
        self.state = 16807 * self.state % 2147483647
        return self.state / 2147483647

    def randn(self, rows, cols):
        out = np.empty((rows, cols))
        for i in range(rows):
            for j in range(cols):
                u1, u2 = self.rand(), self.rand()
                out[i, j] = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return out

    def choice(self, p):
        return min(int(np.searchsorted(np.cumsum(p), self.rand(), side="right")), len(p) - 1)

# @@ parameters and init
class Module:
    """A layer: its own arrays in self.params, their gradients in self.grads, its sublayers in self.children."""

    def __init__(self):
        self.params, self.grads, self.children = {}, {}, {}

    def named_parameters(self, prefix=""):
        for name, p in self.params.items():
            yield prefix + name, p, self.grads.get(name)
        for name, child in self.children.items():
            yield from child.named_parameters(f"{prefix}{name}.")


def init_matrix(rng, rows, cols, scale=None):
    if scale is None:
        scale = 1.0 / math.sqrt(rows)
    return scale * rng.randn(rows, cols)


def flat(a):
    """(batch, time, features) -> (batch * time, features), for the weight gradients."""
    return a.reshape(-1, a.shape[-1])

# @@ embedding
class Embedding(Module):
    def __init__(self, cfg, rng):
        super().__init__()
        self.params["tok"] = init_matrix(rng, cfg.vocab, cfg.width, scale=1.0)
        self.params["pos"] = init_matrix(rng, cfg.context, cfg.width, scale=1.0)

    def forward(self, idx):
        self.idx = idx
        return self.params["tok"][idx] + self.params["pos"][: idx.shape[1]]

    def backward(self, dx):
        dtok = np.zeros_like(self.params["tok"])
        np.add.at(dtok, self.idx, dx)
        dpos = np.zeros_like(self.params["pos"])
        dpos[: dx.shape[1]] = dx.sum(axis=0)
        self.grads = {"tok": dtok, "pos": dpos}

# @@ attention
def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class Head(Module):
    def __init__(self, cfg, rng):
        super().__init__()
        size = cfg.width // cfg.heads
        for name in ("q", "k", "v"):
            self.params[name] = init_matrix(rng, cfg.width, size)

    def forward(self, x):
        p = self.params
        q, k, v = x @ p["q"], x @ p["k"], x @ p["v"]
        T = x.shape[1]
        scale = 1.0 / math.sqrt(q.shape[-1])
        s = (q @ k.transpose(0, 2, 1)) * scale
        s = np.where(np.triu(np.ones((T, T), dtype=bool), k=1), MASKED, s)
        w = softmax(s)
        self.cache = (x, q, k, v, w, scale)
        return w @ v

    def backward(self, dy):
        x, q, k, v, w, scale = self.cache
        p = self.params
        dv = w.transpose(0, 2, 1) @ dy
        dw = dy @ v.transpose(0, 2, 1)
        ds = w * (dw - (dw * w).sum(axis=-1, keepdims=True)) * scale
        dq, dk = ds @ k, ds.transpose(0, 2, 1) @ q
        self.grads = {"q": flat(x).T @ flat(dq), "k": flat(x).T @ flat(dk), "v": flat(x).T @ flat(dv)}
        return dq @ p["q"].T + dk @ p["k"].T + dv @ p["v"].T


class Attention(Module):
    def __init__(self, cfg, rng):
        super().__init__()
        self.heads = [Head(cfg, rng) for _ in range(cfg.heads)]
        self.children = {f"heads.{i}": h for i, h in enumerate(self.heads)}
        self.params["out"] = init_matrix(rng, cfg.width, cfg.width)

    def forward(self, x):
        self.z = np.concatenate([h.forward(x) for h in self.heads], axis=-1)
        return self.z @ self.params["out"]

    def backward(self, dy):
        self.grads = {"out": flat(self.z).T @ flat(dy)}
        dz = np.split(dy @ self.params["out"].T, len(self.heads), axis=-1)
        return sum(h.backward(d) for h, d in zip(self.heads, dz))

# @@ block
def rmsnorm(x):
    r = np.sqrt((x * x).mean(axis=-1, keepdims=True) + EPS)
    return x / r, r


def rmsnorm_backward(x, r, dy):
    dot = (dy * x).sum(axis=-1, keepdims=True)
    return dy / r - x * dot / (x.shape[-1] * r * r * r)


class MLP(Module):
    def __init__(self, cfg, rng):
        super().__init__()
        self.params["up"] = init_matrix(rng, cfg.width, cfg.hidden)
        self.params["down"] = init_matrix(rng, cfg.hidden, cfg.width)

    def forward(self, x):
        self.x = x
        self.u = x @ self.params["up"]
        self.h = np.maximum(self.u, 0.0) ** 2
        return self.h @ self.params["down"]

    def backward(self, dy):
        dh = dy @ self.params["down"].T
        du = dh * 2.0 * np.maximum(self.u, 0.0)
        self.grads = {"down": flat(self.h).T @ flat(dy), "up": flat(self.x).T @ flat(du)}
        return du @ self.params["up"].T


class Block(Module):
    def __init__(self, cfg, rng):
        super().__init__()
        self.attn, self.mlp = Attention(cfg, rng), MLP(cfg, rng)
        self.children = {"attn": self.attn, "mlp": self.mlp}

    def forward(self, x):
        n1, self.r1 = rmsnorm(x)
        x1 = x + self.attn.forward(n1)
        n2, self.r2 = rmsnorm(x1)
        self.x, self.x1 = x, x1
        return x1 + self.mlp.forward(n2)

    def backward(self, dy):
        dx1 = rmsnorm_backward(self.x1, self.r2, self.mlp.backward(dy)) + dy
        return rmsnorm_backward(self.x, self.r1, self.attn.backward(dx1)) + dx1

# @@ forward
class GPT(Module):
    def __init__(self, cfg, rng):
        super().__init__()
        self.cfg = cfg
        self.embed = Embedding(cfg, rng)
        self.blocks = [Block(cfg, rng) for _ in range(cfg.layers)]
        self.children = {"embed": self.embed, **{f"blocks.{i}": b for i, b in enumerate(self.blocks)}}
        # the output map starts at zero, so the first prediction is uniform
        self.params["unembed"] = np.zeros((cfg.width, cfg.vocab))

    def forward(self, idx):
        x = self.embed.forward(idx)
        for block in self.blocks:
            x = block.forward(x)
        self.x = x
        self.nf, self.rf = rmsnorm(x)
        return self.nf @ self.params["unembed"]

    # @@ loss
    def loss(self, logits, targets):
        """The mean cross-entropy of the next token, and its gradient with respect to the logits."""
        shape = logits.shape
        logits, targets = flat(logits), targets.reshape(-1)
        rows = np.arange(len(targets))
        top = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - top)
        total = e.sum(axis=1)
        loss = np.mean(np.log(total) + top[:, 0] - logits[rows, targets])
        dlogits = e / total[:, None]
        dlogits[rows, targets] -= 1.0
        return loss, (dlogits / len(targets)).reshape(shape)

    # @@ backward
    def backward(self, dlogits):
        self.grads = {"unembed": flat(self.nf).T @ flat(dlogits)}
        dx = rmsnorm_backward(self.x, self.rf, dlogits @ self.params["unembed"].T)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        self.embed.backward(dx)

# @@ optimizer
def newton_schulz(g, steps=5, a=3.4445, b=-4.775, c=2.0315):
    """A matrix with g's singular vectors and every singular value near 1 (Muon's direction)."""
    if g.shape[0] > g.shape[1]:
        return newton_schulz(g.T, steps, a, b, c).T
    norm = np.linalg.norm(g)
    x = g / norm if norm else g
    for _ in range(steps):
        s = x @ x.T
        x = a * x + b * (s @ x) + c * ((s @ s) @ x)
    return x


class MuonAdam:
    """Muon on the matrices inside the blocks, Adam on the embeddings and the output map."""

    def __init__(self):
        self.state = {}
        self.taken = 0

    def step(self, model, lr_scale):
        self.taken += 1
        for name, p, g in model.named_parameters():
            s = self.state.setdefault(name, {"m": 0.0, "v": 0.0})
            if name.startswith("blocks."):
                s["m"] = MUON_MOMENTUM * s["m"] + g
                rows, cols = p.shape
                p -= MUON_LR * lr_scale * math.sqrt(max(1.0, rows / cols)) * newton_schulz(s["m"])
            else:
                s["m"] = ADAM_BETA1 * s["m"] + (1.0 - ADAM_BETA1) * g
                s["v"] = ADAM_BETA2 * s["v"] + (1.0 - ADAM_BETA2) * g * g
                mean = s["m"] / (1.0 - ADAM_BETA1**self.taken)
                spread = np.sqrt(s["v"] / (1.0 - ADAM_BETA2**self.taken))
                p -= ADAM_LR * lr_scale * mean / (spread + ADAM_EPS)

# @@ training loop
def get_batch(tokens, cfg, step, shift):
    windows = len(tokens) - cfg.context
    starts = [(step * cfg.batch + b) % windows for b in range(cfg.batch)]
    return np.array([tokens[s + shift : s + shift + cfg.context] for s in starts])


def train(model, tokens, steps):
    opt = MuonAdam()
    losses = []
    for step in range(steps):
        x, y = get_batch(tokens, model.cfg, step, 0), get_batch(tokens, model.cfg, step, 1)
        loss, dlogits = model.loss(model.forward(x), y)
        model.backward(dlogits)
        opt.step(model, lr_scale=1.0 - (1.0 - LR_FLOOR) * step / steps)
        losses.append(loss)
        print(f"step {step + 1}: loss {loss:.6f}")
    return losses

# @@ sampling
def generate(model, prompt, count, rng):
    out = list(prompt)
    for _ in range(count):
        logits = model.forward(np.array([out[-model.cfg.context :]]))
        out.append(rng.choice(softmax(logits[0, -1])))
    return out

# @@ text as tokens
vocab = sorted(set(TEXT))
stoi = {c: i for i, c in enumerate(vocab)}


def encode(text):
    return [stoi[c] for c in text]


def decode(ids):
    return "".join(vocab[i] for i in ids)

# @@ run
cfg = Config(vocab=len(vocab))
model = GPT(cfg, Minstd(INIT_SEED))
count = sum(p.size for _, p, _ in model.named_parameters())
print(f"{count} parameters, {len(vocab)} characters")
losses = train(model, np.array(encode(TEXT)), STEPS)
written = decode(generate(model, encode(PROMPT), SAMPLES, Minstd(SAMPLE_SEED)))
print(written)

# @@ checks
assert count == 6560 and len(vocab) == 9
assert abs(losses[0] - math.log(9.0)) <= 1e-15
assert abs(losses[-1] - 0.199118) <= 1e-6
assert written == "hello world! hello world! hel"
