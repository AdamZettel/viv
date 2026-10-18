#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Write lib/gguf/tests/tiny.gguf and its reference, lib/gguf/tests/tiny.ref.

    ~/.venvs/ggml/bin/python tools/gguf/make_fixture.py

tiny.gguf is written by gguf-py's GGUFWriter (gguf 0.19.0): one metadata key
of every GGUF value type, an array of each element type, an array of arrays,
UTF-8 strings, a 64-byte alignment, and one small tensor of each type the
ladder reads (F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q4_K, Q6_K). The quantized
tensors are random blocks (random quants, scales and half-precision deltas),
so every bit of every block layout is exercised, not only the patterns a
quantizer happens to produce.

tiny.ref holds what gguf-py's dequantize gives for each tensor, in the order
the tensors are written, as little-endian float32: the bytes lib/gguf's tests
compare against, element for element.
"""
import os
import numpy as np
import gguf
from gguf import GGMLQuantizationType as T, GGUFValueType as V

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "lib", "gguf", "tests")
SEED = 20261014
ROWS = 8
COLS = 256          # one Q4_K / Q6_K super-block per row
ALIGN = 64


def half_bytes(rng, n, scale):
    """n random finite halves of magnitude about `scale`, as bytes."""
    return (rng.standard_normal(n) * scale).astype(np.float16).view(np.uint8).reshape(n, 2)


def random_blocks(rng, qtype, n_blocks):
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    raw = rng.integers(0, 256, size=(n_blocks, size), dtype=np.uint8)
    # the half-precision fields: where each layout keeps them (ggml-common.h)
    if qtype in (T.Q8_0, T.Q4_0):
        raw[:, 0:2] = half_bytes(rng, n_blocks, 0.01)
    # a minimum far larger than the scale: q * d + m then needs more than 24
    # bits, and single precision rounds it, so the fixture shows the rounding
    elif qtype == T.Q4_1:
        raw[:, 0:2] = half_bytes(rng, n_blocks, 0.0001)
        raw[:, 2:4] = half_bytes(rng, n_blocks, 1.0)
    elif qtype == T.Q4_K:
        raw[:, 0:2] = half_bytes(rng, n_blocks, 0.001)
        raw[:, 2:4] = half_bytes(rng, n_blocks, 1.0)
    elif qtype == T.Q6_K:
        raw[:, size - 2:size] = half_bytes(rng, n_blocks, 0.001)
    return raw


def main():
    rng = np.random.default_rng(SEED)
    path = os.path.join(OUT, "tiny.gguf")
    w = gguf.GGUFWriter(path, "tiny")
    w.add_custom_alignment(ALIGN)
    w.add_uint8("tiny.u8", 200)
    w.add_int8("tiny.i8", -100)
    w.add_uint16("tiny.u16", 60000)
    w.add_int16("tiny.i16", -30000)
    w.add_uint32("tiny.u32", 4000000000)
    w.add_int32("tiny.i32", -2000000000)
    w.add_float32("tiny.f32", 0.1)
    w.add_bool("tiny.bool", True)
    w.add_string("tiny.str", "héllo ▁world 🙂")
    w.add_uint64("tiny.u64", 2 ** 40 + 3)
    w.add_int64("tiny.i64", -(2 ** 50) - 7)
    w.add_float64("tiny.f64", 2.0 ** -1074)
    for name, vt, vals in [("u8", V.UINT8, [0, 1, 255]), ("i8", V.INT8, [-128, 0, 127]),
                           ("u16", V.UINT16, [0, 65535]), ("i16", V.INT16, [-32768, 32767]),
                           ("u32", V.UINT32, [0, 4294967295]), ("i32", V.INT32, [-2147483648, 2147483647]),
                           ("f32", V.FLOAT32, [1.5, -0.0, 3.4028234663852886e38]), ("bool", V.BOOL, [True, False, True]),
                           ("str", V.STRING, ["a", "", "▁the", "<0x0A>"]), ("u64", V.UINT64, [0, 2 ** 63 - 1]),
                           ("i64", V.INT64, [-(2 ** 63), 2 ** 63 - 1]), ("f64", V.FLOAT64, [0.1, -1e300])]:
        w.add_key_value(f"tiny.arr.{name}", vals, V.ARRAY, sub_type=vt)
    w.add_key_value("tiny.arr.nested", [[1, 2], [3]], V.ARRAY, sub_type=V.ARRAY)

    ref = []
    x = rng.standard_normal((ROWS, COLS)).astype(np.float32)
    w.add_tensor("t.f32", x)
    ref.append(x)
    h = rng.standard_normal((ROWS, COLS)).astype(np.float16)
    h.view(np.uint16)[0, :8] = [0x0001, 0x8001, 0x03FF, 0x0400, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00]
    w.add_tensor("t.f16", h)
    ref.append(h.astype(np.float32))
    b = (rng.standard_normal((ROWS, COLS)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    w.add_tensor("t.bf16", b.view(np.uint8), raw_dtype=T.BF16)
    ref.append(gguf.quants.dequantize(b.view(np.uint8), T.BF16))
    for name, qt in [("q8_0", T.Q8_0), ("q4_0", T.Q4_0), ("q4_1", T.Q4_1), ("q4_k", T.Q4_K), ("q6_k", T.Q6_K)]:
        block, size = gguf.GGML_QUANT_SIZES[qt]
        raw = random_blocks(rng, qt, ROWS * COLS // block).reshape(ROWS, COLS // block * size)
        w.add_tensor(f"t.{name}", raw, raw_dtype=qt)
        ref.append(gguf.quants.dequantize(raw, qt))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with open(os.path.join(OUT, "tiny.ref"), "wb") as f:
        for r in ref:
            f.write(np.ascontiguousarray(r, dtype="<f4").tobytes())
    print(path, os.path.getsize(path), "bytes")


if __name__ == "__main__":
    main()
