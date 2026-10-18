# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The stories15M numbers of gallery/ml/plot_gguf_quantization.py: the two
model files (Q8_0 and Q4_0, 26.7 and 19.1 MB) are not in the repository, so
this script reads them with vp.gguf and writes what the page shows to
gallery/ml/gguf_stories15m.json; the page prints that file.

    PATH=target/release:$PATH PYTHONPATH=python python3 gallery/ml/gguf_stories15m.py

The files are llama2.c's stories15M in GGUF from huggingface.co/ggml-org/models
(tinyllamas/stories15M-q8_0.gguf and stories15M-q4_0.gguf), read from
$GGUF_MODELS (default ~/models/gguf).
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import vipyr as vp

MODELS = Path(os.environ.get("GGUF_MODELS", "~/models/gguf")).expanduser()
FILES = {"Q8_0": MODELS / "stories15M-q8_0.gguf", "Q4_0": MODELS / "stories15M-q4_0.gguf"}
NAME = "blk.0.attn_q.weight"
KEYS = ("general.architecture", "llama.block_count", "llama.embedding_length", "llama.feed_forward_length",
        "llama.attention.head_count", "llama.context_length")
# the histogram: bins of 0.01 from -0.6 to 0.6
EDGES = np.linspace(-0.6, 0.6, 121)
# the block of Q8_0 and Q4_0: 32 weights with one scale
BLOCK = 32
SIG = 7

out = {"files": {}, "tensor": NAME}
w = {}
for fmt, path in FILES.items():
    f = vp.gguf.read(path)
    kinds = {}
    for t in f.tensors:
        kinds[t.type] = kinds.get(t.type, 0) + 1
    out["files"][fmt] = {
        "file": path.name, "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tensors": len(f.tensors), "types": kinds, "data_offset": f.data_offset,
    }
    if fmt == "Q8_0":
        m = f.metadata
        out["metadata"] = {k: m[k] for k in KEYS}
        out["tokens"] = {"count": len(m["tokenizer.ggml.tokens"]), "first": m["tokenizer.ggml.tokens"][:6]}
    r = vp.gguf.tensor(path, NAME, file=f)
    assert r.check()
    w[fmt] = r.array
    out["shape"], out["dtype"] = list(r.array.shape), str(r.array.dtype)
    out["files"][fmt]["distinct"] = len(np.unique(r.array))
    out["files"][fmt]["histogram"] = np.histogram(r.array, EDGES)[0].tolist()
a = w["Q8_0"].reshape(-1, BLOCK).astype(np.float64)
b = w["Q4_0"].reshape(-1, BLOCK).astype(np.float64)
amax = np.abs(a).max(axis=1)
rms = np.sqrt(((b - a) ** 2).mean(axis=1))
out["edges"] = [round(float(x), 2) for x in EDGES]
out["blocks"] = {"amax": [float("%.*g" % (SIG, x)) for x in amax], "rms": [float("%.*g" % (SIG, x)) for x in rms],
                 "median_ratio": float(np.median(rms / amax))}
dst = Path(__file__).with_name("gguf_stories15m.json")
dst.write_text(json.dumps(out, indent=1) + "\n")
print("wrote", dst, "median rms / amax %.4f" % out["blocks"]["median_ratio"])
