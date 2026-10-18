# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
A quantized model file, read in Viv
===================================

`vp.gguf` reads GGUF, the file format of ggml and llama.cpp, with
`lib/gguf`: the header and metadata (`header.viv`), the tensor table
(`tensors.viv`) and ggml's number formats (`quants.viv`), each a rewrite of
ggml's code built one step at a time. The models are Karpathy's llama2.c
models trained on TinyStories, in GGUF from huggingface.co/ggml-org/models
(MIT). The smallest, stories260K, is in the repository
(`lib/gguf/tests/stories260K.gguf`) and is read below as the page runs.
The 15-million-parameter model comes in two files, one with 8-bit weights
(Q8_0) and one with 4-bit weights (Q4_0); they are not in the repository,
so `gallery/ml/gguf_stories15m.py` reads them with the same calls and
writes the numbers this page prints to `gallery/ml/gguf_stories15m.json`.
Every number equals what gguf-py's `dequantize` gives for the same bytes
(`tools/gguf/check.sh`).
"""

# %%
# ## What the file says about the model
#
# The metadata is a list of typed key-value pairs; the tokenizer's tokens
# are one of them, an array of strings. stories260K, read now:

import json
import os
from pathlib import Path

import numpy as np
import vipyr as vp
from vipyr.figure import figure

KEYS = ("general.architecture", "llama.block_count", "llama.embedding_length",
        "llama.feed_forward_length", "llama.attention.head_count", "llama.context_length")
small = vp.gguf.read(os.path.join(vp.viv_root(), "lib/gguf/tests/stories260K.gguf"))
m = small.metadata
for key in KEYS:
    print("%-30s %s" % (key, m[key]))
print("%-30s %d, the first: %s" % ("tokenizer.ggml.tokens", len(m["tokenizer.ggml.tokens"]),
                                    m["tokenizer.ggml.tokens"][:6]))
kinds = {}
for t in small.tensors:
    kinds[t.type] = kinds.get(t.type, 0) + 1
print("%d tensors, %s; the data starts at byte %d" % (len(small.tensors), kinds, small.data_offset))
q = vp.gguf.tensor(small.path, "blk.0.attn_q.weight", file=small)
print(q.info)
print("its weights: mean %.5f, standard deviation %.5f, largest magnitude %.5f"
      % (q.array.mean(), q.array.std(), np.abs(q.array).max()))
assert small.check() and q.check()

# %%
# The same for stories15M, from `gallery/ml/gguf_stories15m.json`. Each
# file names the type of every tensor: the norms stay F32 in both, and the
# Q4_0 file keeps the output matrix in Q8_0.

big = json.loads((Path(vp.viv_root()) / "gallery/ml/gguf_stories15m.json").read_text())
for key in KEYS:
    print("%-30s %s" % (key, big["metadata"][key]))
print("%-30s %d, the first: %s" % ("tokenizer.ggml.tokens", big["tokens"]["count"], big["tokens"]["first"]))
for fmt, f in big["files"].items():
    print("%s file (%s, %d bytes): %d tensors, %s; the data starts at byte %d"
          % (fmt, f["file"], f["bytes"], f["tensors"], f["types"], f["data_offset"]))

# %%
# ## One matrix in two formats
#
# Both formats cut a row into blocks of 32 weights and store one
# half-precision scale per block. Q8_0 keeps each weight as a signed byte
# (255 levels), Q4_0 as 4 bits (16 levels): a weight is a whole number of
# steps of its block's scale. The query matrix of the first block of
# stories15M:

name = big["tensor"]
q8, q4 = big["files"]["Q8_0"], big["files"]["Q4_0"]
print(name, tuple(big["shape"]), big["dtype"])
print("distinct values: Q8_0 %d, Q4_0 %d" % (q8["distinct"], q4["distinct"]))
edges = np.array(big["edges"])
mid = (edges[:-1] + edges[1:]) / 2
fig = figure("Weights of %s" % name, "weight", "count per bin of 0.01")
fig.plot(mid, np.array(q8["histogram"]), "Q8_0")
fig.plot(mid, np.array(q4["histogram"]), "Q4_0")
fig.style("Q8_0", "s1 nomarker")
fig.style("Q4_0", "s2 nomarker")

# %%
# ## The error, block by block
#
# The original float32 weights are not in either file, but Q8_0's copy is
# close to them: its step is the block's largest magnitude over 127, so
# each weight is off by about a 254th of it (half a step, plus the rounding
# of the scale to half precision). Q4_0's step is the largest
# magnitude over 8, so its error is about 16 times larger. Independent
# errors add in squares, so the root mean square of the difference between
# the two files is Q4_0's times sqrt(1 + 1/16^2) = 1.002.
# A rounding error spread evenly over one step has a root mean square of
# step / sqrt(12); for Q4_0 that is the block's largest magnitude over
# 8 sqrt(12), about 27.7.

amax = np.array(big["blocks"]["amax"])
rms = np.array(big["blocks"]["rms"])
print("%d blocks; rms error / largest magnitude: median %.4f, 1/(8 sqrt 12) = %.4f"
      % (len(rms), big["blocks"]["median_ratio"], 1 / (8 * np.sqrt(12))))
fig = figure("Q4_0 against Q8_0, per block of 32 (%s)" % name, "largest |weight| in the block", "rms difference")
fig.scatter(amax, rms, "a block")
x = np.array([0.0, amax.max()])
fig.plot(x, x / (8 * np.sqrt(12)), "step$/\\sqrt{12}$")
fig.style("step$/\\sqrt{12}$", "s2 nomarker")
