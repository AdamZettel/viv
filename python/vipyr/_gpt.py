# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vp.gpt: the GPT of lib/gpt/gpt.viv, trained and sampled from Python.

The model, its training step and its sampler are Viv (lib/gpt/gpt.viv, with
lib/nn, lib/optim and lib/polar); lib/gpt/api.viv gives them entry points
that take plain numbers and arrays, and this module compiles it once through
vipyr's cache. Python keeps the tokenizer and the model's names: a trained
model comes back as one flat vector of its matrices in `for_leaves` order,
and `Model` cuts it into named arrays laid out as the Viv records are,

    model.tok, model.pos, model.unembed
    model.blocks[l].attn.heads[h].Wq / .Wk / .Wv
    model.blocks[l].attn.Wo
    model.blocks[l].mlp.up / .down

and joins them back into that vector for `generate` and `loss`, so arrays
edited in Python are the ones the next call uses.
"""
from __future__ import annotations

import dataclasses
import hashlib
import threading
from dataclasses import dataclass, field

import numpy as np

from . import _core, _spec

API = "lib/gpt/api.viv"

_lock = threading.Lock()
_mod = {}


@dataclass(frozen=True)
class GptConfig:
    """The sizes of a model, as lib/gpt/gpt.viv's `GptConfig` record.

    vocab    tokens in the vocabulary; None: from the training text
    context  positions the model sees at once
    width    numbers per position in the residual stream
    heads    attention heads per block (width must be a multiple)
    layers   blocks
    hidden   width of the MLP's inner layer
    batch    sequences per training step
    """

    vocab: int | None = None
    context: int = 8
    width: int = 16
    heads: int = 2
    layers: int = 2
    hidden: int = 64
    batch: int = 4

    def sizes(self):
        """The numbers lib/gpt/api.viv's entry points take, in order (no batch)."""
        return [
            float(self.vocab),
            float(self.context),
            float(self.width),
            float(self.heads),
            float(self.layers),
            float(self.hidden),
        ]


@dataclass
class Head:
    Wq: np.ndarray
    Wk: np.ndarray
    Wv: np.ndarray


@dataclass
class Attention:
    heads: list
    Wo: np.ndarray


@dataclass
class Mlp:
    up: np.ndarray
    down: np.ndarray


@dataclass
class Block:
    attn: Attention
    mlp: Mlp


@dataclass
class Model:
    """A trained GPT: its matrices by name, its sizes and its characters.

    `vocab` is the list of characters a token id stands for (None when the
    model was trained on token ids).
    """

    tok: np.ndarray
    pos: np.ndarray
    blocks: list
    unembed: np.ndarray
    config: GptConfig
    vocab: list | None = field(default=None)

    def parameters(self):
        """(name, matrix) for every matrix, in lib/gpt's `for_leaves` order."""
        yield "tok", self.tok
        yield "pos", self.pos
        for i, b in enumerate(self.blocks):
            for k, h in enumerate(b.attn.heads):
                for n in ("Wq", "Wk", "Wv"):
                    yield "blocks[%d].attn.heads[%d].%s" % (i, k, n), getattr(h, n)
            yield "blocks[%d].attn.Wo" % i, b.attn.Wo
            yield "blocks[%d].mlp.up" % i, b.mlp.up
            yield "blocks[%d].mlp.down" % i, b.mlp.down
        yield "unembed", self.unembed

    @property
    def size(self):
        """The number of parameters."""
        return sum(w.size for _, w in self.parameters())

    def pack(self):
        """The matrices as one vector, row by row, in `for_leaves` order."""
        return np.concatenate([np.ascontiguousarray(w, dtype=float).ravel() for _, w in self.parameters()])

    @classmethod
    def unpack(cls, flat, config, vocab=None):
        """A model of the sizes `config` holding the matrices of `flat`."""
        flat = np.asarray(flat, dtype=float)
        c = config
        hd = c.width // c.heads
        at = [0]

        def take(rows, cols):
            w = flat[at[0] : at[0] + rows * cols].reshape(rows, cols).copy()
            at[0] += rows * cols
            return w

        tok = take(c.vocab, c.width)
        pos = take(c.context, c.width)
        blocks = []
        for _ in range(c.layers):
            heads = [Head(take(c.width, hd), take(c.width, hd), take(c.width, hd)) for _ in range(c.heads)]
            attn = Attention(heads, take(c.width, c.width))
            blocks.append(Block(attn, Mlp(take(c.width, c.hidden), take(c.hidden, c.width))))
        unembed = take(c.width, c.vocab)
        if at[0] != flat.size:
            raise ValueError("vp.gpt: a model of these sizes has %d parameters; the vector has %d" % (at[0], flat.size))
        return cls(tok, pos, blocks, unembed, c, vocab)


def tokenize(text):
    """(vocab, ids): the sorted distinct characters of `text`, and its ids."""
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    return vocab, np.array([stoi[ch] for ch in text], dtype=float)


def _api(fast=False):
    """lib/gpt/api.viv, compiled once per process (and cached across them);
    `fast` builds with the fast flags (results may differ in the last bits)."""
    with _lock:
        if fast not in _mod:
            mod = _core.load(_spec.source(API), fast=fast)
            _mod[fast] = {
                n: getattr(getattr(mod, n), "__vipyr_raw__", getattr(mod, n))
                for n in ("gpt_train", "gpt_generate", "gpt_attention", "gpt_loss")
            }
        return _mod[fast]


def _ids(model, text):
    """Token ids of a str (through the model's characters) or of an array."""
    if isinstance(text, str):
        if model.vocab is None:
            raise ValueError("vp.gpt: this model was trained on token ids; pass ids, not text")
        stoi = {ch: i for i, ch in enumerate(model.vocab)}
        missing = sorted(set(text) - set(stoi))
        if missing:
            raise ValueError("vp.gpt: characters not in the model's vocabulary: %r" % "".join(missing))
        return np.array([stoi[ch] for ch in text], dtype=float)
    ids = np.atleast_1d(np.asarray(text))
    if ids.ndim != 1 or (
        ids.size and (ids.min() < 0 or ids.max() >= model.config.vocab or np.any(ids != np.round(ids)))
    ):
        raise ValueError("vp.gpt: token ids must be integers in [0, %d)" % model.config.vocab)
    return ids.astype(float)


def _check_config(c, n):
    for name in ("context", "width", "heads", "layers", "hidden", "batch"):
        v = getattr(c, name)
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
            raise ValueError("vp.gpt: GptConfig.%s must be a positive int; got %r" % (name, v))
    if c.width % c.heads:
        raise ValueError("vp.gpt: width (%d) must be a multiple of heads (%d)" % (c.width, c.heads))
    if n <= c.context:
        raise ValueError("vp.gpt: the training text needs more than context = %d tokens; it has %d" % (c.context, n))


def _problem(*parts):
    h = hashlib.sha256(_spec.source(API).read_bytes())
    for p in parts:
        h.update(repr(p).encode())
    return h.hexdigest()[:16]


def _train(text, *, config=None, steps=30, seed=7, fast=False):
    """vp.gpt.train, before the result object is made."""
    if isinstance(text, str):
        vocab, ids = tokenize(text)
        nvocab = len(vocab)
    else:
        vocab = None
        ids = np.atleast_1d(np.asarray(text))
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids != np.round(ids)):
            raise ValueError("vp.gpt: tokens must be a 1-D array of non-negative integer ids")
        ids = ids.astype(float)
        nvocab = int(ids.max()) + 1
    c = config or GptConfig()
    if c.vocab is None:
        c = dataclasses.replace(c, vocab=nvocab)
    elif c.vocab < nvocab:
        raise ValueError("vp.gpt: GptConfig.vocab = %d, but the tokens need %d" % (c.vocab, nvocab))
    _check_config(c, len(ids))
    if int(steps) < 1:
        raise ValueError("vp.gpt: steps must be at least 1; got %r" % (steps,))
    losses, flat = _api(fast)["gpt_train"](ids, *c.sizes(), float(c.batch), float(steps), float(seed))
    model = Model.unpack(flat, c, vocab)
    return {
        "model": model,
        "losses": np.asarray(losses, dtype=float),
        "_problem": _problem(dataclasses.astuple(c), steps, seed, fast, ids.tobytes()),
    }


def _generate(model, prompt, *, n=100, temperature=1.0, seed=11):
    """vp.gpt.generate, before the result object is made."""
    if not isinstance(model, Model):
        raise TypeError("vp.gpt: model must be what vp.gpt.train returned (r.model); got %r" % type(model).__name__)
    ids = _ids(model, prompt)
    if ids.size == 0:
        raise ValueError("vp.gpt: the prompt needs at least one token")
    if not temperature >= 0:
        raise ValueError("vp.gpt: temperature must be >= 0; got %r" % (temperature,))
    out = np.asarray(
        _api()["gpt_generate"](
            model.pack(),
            *model.config.sizes(),
            ids,
            float(n),
            float(temperature),
            float(seed),
        )
    ).astype(int)
    text = None if model.vocab is None else "".join(model.vocab[i] for i in out)
    return {
        "tokens": out,
        "text": text,
        "_problem": _problem(
            dataclasses.astuple(model.config),
            n,
            temperature,
            seed,
            ids.tobytes(),
        ),
    }


def _attention(model, prompt):
    """vp.gpt.attention, before the result object is made."""
    if not isinstance(model, Model):
        raise TypeError("vp.gpt: model must be what vp.gpt.train returned (r.model); got %r" % type(model).__name__)
    ids = _ids(model, prompt)
    c = model.config
    if not 1 <= ids.size <= c.context:
        raise ValueError("vp.gpt: attention takes 1 to context = %d tokens; got %d" % (c.context, ids.size))
    W = np.asarray(_api()["gpt_attention"](model.pack(), *c.sizes(), ids)).reshape(
        c.layers, c.heads, ids.size, ids.size
    )
    labels = None if model.vocab is None else [model.vocab[int(i)] for i in ids]
    return {
        "weights": W,
        "labels": labels,
        "_problem": _problem(dataclasses.astuple(c), "attention", ids.tobytes()),
    }


def _check_attention(res):
    rows = res.weights.sum(axis=-1)
    ok = bool(np.abs(rows - 1.0).max() <= 1e-12)
    upper = bool(np.all(np.triu(res.weights, k=1) == 0.0))
    return (
        ("every row of weights sums to 1", ok, ""),
        ("no position looks ahead of itself", upper, ""),
    )


def _loss(model, tokens):
    """vp.gpt.loss, before the result object is made."""
    if not isinstance(model, Model):
        raise TypeError("vp.gpt: model must be what vp.gpt.train returned (r.model); got %r" % type(model).__name__)
    ids = _ids(model, tokens)
    if ids.size <= model.config.context:
        raise ValueError("vp.gpt: loss needs more than context = %d tokens" % model.config.context)
    value = float(_api()["gpt_loss"](model.pack(), *model.config.sizes(), ids))
    return {
        "loss": value,
        "_problem": _problem(dataclasses.astuple(model.config), ids.tobytes()),
    }


def _check_train(res):
    """What a training run is held to beyond lib/gpt's own tests."""
    first = res.losses[0]
    log_v = float(np.log(res.model.config.vocab))
    ok = abs(first - log_v) <= 1e-12
    finite = bool(np.all(np.isfinite(res.losses)))
    return (
        (
            "the first loss is log(vocab), because the output map starts at zero",
            ok,
            "" if ok else "%r vs %r" % (first, log_v),
        ),
        ("every loss is finite", finite, ""),
    )


def _check_generate(res):
    ok = res.tokens.ndim == 1
    return (("the tokens are one sequence", ok, ""),)


def _check_loss(res):
    ok = bool(np.isfinite(res.loss) and res.loss >= 0)
    return (("the loss is finite and non-negative", ok, ""),)
