<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 The Viv authors -->
# lib/gguf: upstream

lib/gguf reads GGUF, the file format of ggml (github.com/ggml-org/ggml, MIT,
copyright the ggml authors; LICENSE.ggml here). It is a rewrite, not a line
for line port: the layouts and the arithmetic come from these files of ggml
at commit 3612c0af93f2cfc8f048295300a61187723b12ae (2026-10-14):

| Viv module | upstream |
|---|---|
| header.viv | docs/gguf.md (the file layout, the metadata value types) |
| tensors.viv | docs/gguf.md (tensor infos, alignment, the data section); include/ggml.h (`enum ggml_type`) |
| api.viv | (the entry points vipyr's `vp.gguf` calls; no upstream) |
| quants.viv | src/ggml-common.h (the block structs), src/ggml-quants.c (`dequantize_row_q8_0`, `_q4_0`, `_q4_1`, `_q4_K`, `_q6_K`, `get_scale_min_k4`) |

The reference for every test is gguf-py 0.19.0 (the Python package in
ggml's sister repository llama.cpp, `gguf-py/`, MIT): its `GGUFReader` for the
metadata and the tensor table, and `gguf.quants.dequantize` for the numbers.
lib/gguf matches it bit for bit on the files it was run on (tools/gguf/check.sh).

Not read: big-endian GGUF files, versions 1 and 2, the other quantization
types (Q5_0, Q5_1, Q2_K, Q3_K, Q5_K, the IQ and TQ families, MXFP4).

## The fixture stories260K.gguf

`tests/stories260K.gguf` is the stories260K model of llama2.c
(github.com/karpathy/llama2.c, MIT, Copyright (c) 2023 Andrej Karpathy;
`tests/LICENSE.llama2c`), a 260,000-parameter Llama-architecture model
trained on TinyStories, converted to GGUF and published at
huggingface.co/ggml-org/models (`tinyllamas/stories260K.gguf`, 1,185,376
bytes, sha256
270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d). It is
committed unchanged so that the docs' examples and the gallery read a real
model file without a download. Its tensors are F32.
