# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vp.gguf.read and vp.gguf.tensor before their result objects are made:
GGUF files (ggml's and llama.cpp's model format), read by lib/gguf.

    f = vp.gguf.read("stories15M-q8_0.gguf")
    f.metadata["llama.embedding_length"]       # 288
    f.metadata["tokenizer.ggml.tokens"][:3]    # ['<unk>', '<s>', '</s>']
    [t.name for t in f.tensors][:2]            # ['token_embd.weight', 'output_norm.weight']
    w = vp.gguf.tensor("stories15M-q8_0.gguf", "blk.0.attn_q.weight").array   # (288, 288) float32

The metadata and the tensor table are read by lib/gguf/header.viv and
tensors.viv, the numbers by lib/gguf/quants.viv (F32, F16, BF16, Q8_0, Q4_0,
Q4_1, Q4_K, Q6_K), through lib/gguf/api.viv; this file reads the bytes with
NumPy and builds the Python values. Each number equals gguf-py's
`gguf.quants.dequantize` bit for bit (tools/gguf/check.sh), and comes back
as float32, the type ggml computes in, which holds it exactly.
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _core, _spec

API = "lib/gguf/api.viv"

# the metadata value types (docs/gguf.md), as lib/gguf/header.viv numbers them
_STRING, _FLOAT32, _FLOAT64, _BOOL, _ARRAY = 8, 6, 12, 7, 9
# ggml's type names by number (lib/gguf/tensors.viv's GGML_NAMES)
TYPE_NAMES = ["F32", "F16", "Q4_0", "Q4_1", "", "", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K", "Q4_K",
              "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S",
              "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M", "BF16", "", "", "", "TQ1_0", "TQ2_0", "",
              "", "", "MXFP4", "NVFP4", "Q1_0"]
# the first read of a file's head; doubled until the tensor table fits
_FIRST_READ = 1 << 20

_lock = threading.Lock()
_mod = None


def _api():
    global _mod
    with _lock:
        if _mod is None:
            _mod = _core.load(_spec.source(API))
        return _mod


@dataclass(frozen=True)
class TensorInfo:
    """One entry of the tensor table. `shape` is NumPy's order (slowest
    first), the reverse of GGUF's dims; `offset` is from the file's start."""
    name: str
    shape: tuple
    type: str
    offset: int
    n_bytes: int


class GgufFileMixin:
    """The method of vp.gguf.read's result (GgufFile)."""

    def tensor_info(self, name: str) -> TensorInfo:
        for t in self.tensors:
            if t.name == name:
                return t
        raise KeyError("vp.gguf: %s has no tensor %r" % (self.path, name))


class _Ints:
    """A cursor over file_index's flat Ints."""

    def __init__(self, a):
        self.a, self.i = a, 0

    def take(self):
        self.i += 1
        return int(self.a[self.i - 1])

    def takes(self, n):
        self.i += n
        return [int(x) for x in self.a[self.i - n:self.i]]

    def text(self):
        return "".join(chr(c) for c in self.takes(self.take()))


def _index(path):
    size = Path(path).stat().st_size
    n = min(size, _FIRST_READ)
    api = _api()
    while True:
        head = np.fromfile(path, dtype=np.uint8, count=n).astype(np.int64)
        try:
            return api.file_index(head)
        except api.VivStop as e:
            # the head ended inside the metadata or the table: read more
            if n >= size or "the bytes given end" not in str(e):
                raise
            n = min(size, 2 * n)


def _problem(path, *parts):
    """The run's content address: the Viv entry points, the file's bytes and
    the arguments."""
    h = hashlib.sha256(_spec.source(API).read_bytes())
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    for p in parts:
        h.update(repr(p).encode())
    return h.hexdigest()[:16]


def _read(path):
    """vp.gguf.read, before the result object is made."""
    path = str(path)
    ints, nums = _index(path)
    c = _Ints(ints)
    n_kv, n_tensors, alignment, data = c.takes(4)
    meta = {}
    for _ in range(n_kv):
        key = c.text()
        types = c.takes(c.take())
        count = c.take()
        leaf = types[-1]
        if leaf == _STRING:
            vals = [c.text() for _ in range(count)]
        elif leaf in (_FLOAT32, _FLOAT64):
            vals = [float(nums[c.take()]) for _ in range(count)]
        elif leaf == _BOOL:
            vals = [bool(c.take()) for _ in range(count)]
        else:
            vals = [c.take() for _ in range(count)]
        meta[key] = vals if types[0] == _ARRAY else vals[0]
    tensors = []
    for _ in range(n_tensors):
        name = c.text()
        dims = c.takes(c.take())
        t, off, nb = c.takes(3)
        tensors.append(TensorInfo(name, tuple(reversed(dims)), TYPE_NAMES[t], off, nb))
    return {"path": path, "metadata": meta, "tensors": tensors, "alignment": alignment, "data_offset": data,
            "_problem": _problem(path)}


def _tensor(path, name, *, file=None):
    """vp.gguf.tensor, before the result object is made."""
    path = str(path)
    if file is None:
        f = _read(path)
        infos = f["tensors"]
    else:
        if not hasattr(file, "tensors"):
            raise TypeError("vp.gguf: file must be what vp.gguf.read returned; got %r" % type(file).__name__)
        infos = file.tensors
    t = next((x for x in infos if x.name == name), None)
    if t is None:
        raise KeyError("vp.gguf: %s has no tensor %r" % (path, name))
    if TYPE_NAMES.index(t.type) not in _READ:
        raise NotImplementedError("vp.gguf: lib/gguf does not read %s tensors (%s)" % (t.type, name))
    raw = np.fromfile(path, dtype=np.uint8, count=t.n_bytes, offset=t.offset).astype(np.int64)
    n = int(np.prod(t.shape))
    x = _api().dequant(raw, TYPE_NAMES.index(t.type), n)
    return {"array": x.astype(np.float32).reshape(t.shape), "info": t, "_problem": _problem(path, name)}


# bytes per block and weights per block of the types lib/gguf reads
# (ggml.c's type_traits)
_BLOCK = {"F32": (4, 1), "F16": (2, 1), "BF16": (2, 1), "Q8_0": (34, 32), "Q4_0": (18, 32), "Q4_1": (20, 32),
          "Q4_K": (144, 256), "Q6_K": (210, 256)}


def _check_read(res):
    """What a read is held to beyond lib/gguf's own tests: the table
    describes a data section that fits the file."""
    size = Path(res.path).stat().st_size
    al = all(t.offset % res.alignment == 0 and t.offset >= res.data_offset for t in res.tensors)
    spans = sorted((t.offset, t.offset + t.n_bytes) for t in res.tensors)
    apart = all(a[1] <= b[0] for a, b in zip(spans, spans[1:])) and (not spans or spans[-1][1] <= size)
    bad = [t.name for t in res.tensors if t.type in _BLOCK
           and t.n_bytes != int(np.prod(t.shape)) // _BLOCK[t.type][1] * _BLOCK[t.type][0]]
    return (
        ("every tensor starts at a multiple of the alignment, in the data section", al, ""),
        ("the tensors do not overlap and end within the file", apart, ""),
        ("each tensor's size is its type's block size times its blocks", not bad, ", ".join(bad[:5])),
    )


def _check_tensor(res):
    shape = res.array.shape == res.info.shape and res.array.dtype == np.float32
    finite = bool(np.all(np.isfinite(res.array)))
    return (
        ("the array is float32 of the table's shape", shape, ""),
        ("every weight is finite", finite, ""),
    )


# the ggml types lib/gguf reads
_READ = {TYPE_NAMES.index(n) for n in ("F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_1", "Q4_K", "Q6_K")}
