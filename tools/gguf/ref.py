#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""What gguf-py reads from a GGUF file, printed as lib/gguf prints it.

    ~/.venvs/ggml/bin/python tools/gguf/ref.py meta <file.gguf>
    ~/.venvs/ggml/bin/python tools/gguf/ref.py tensors <file.gguf>
    ~/.venvs/ggml/bin/python tools/gguf/ref.py values <file.gguf> <dir> <TYPES>

`meta`: GGUFReader's fields, as lib/gguf/header.viv's print_meta prints
them. `tensors`: its tensor table, as lib/gguf/tensors.viv's print_tensors
prints it. `values`: gguf.quants.dequantize of each tensor whose type is in
the comma-separated TYPES, written to <dir>/<k>.f32 (k: its place in the
table) as little-endian binary32s, and the line tools/gguf/values.viv prints
for it when every number agrees.
"""
import os
import sys
import numpy as np
import gguf
from gguf import GGUFValueType as V


def g17(x):
    return format(float(x), ".17g")


def meta(r):
    h = r.fields
    print(f"version {h['GGUF.version'].parts[-1][0]} tensors {h['GGUF.tensor_count'].parts[-1][0]} keys {h['GGUF.kv_count'].parts[-1][0]}")
    for key, f in r.fields.items():
        if key.startswith("GGUF."):
            continue
        names = ",".join(V(t).name for t in f.types)
        leaf = V(f.types[-1])
        vals = [f.parts[i] for i in f.data]
        print(f"{key} {names} {len(vals)}")
        if leaf == V.STRING:
            for p in vals:
                print(f"  {bytes(p).decode('utf-8')!r}")
        elif leaf in (V.FLOAT32, V.FLOAT64):
            for p in vals:
                print(f"  {g17(p[0])}")
        elif leaf == V.BOOL:
            for p in vals:
                print(f"  {int(bool(p[0]))}")
        else:
            for p in vals:
                v = int(p[0])
                # lib/gguf keeps a UINT64 above 2^63 - 1 as its bit pattern
                print(f"  {v - 2 ** 64 if v >= 2 ** 63 else v}")


def tensors(r):
    print(f"alignment {r.alignment} data {r.data_offset}")
    for t in r.tensors:
        dims = ",".join(str(int(d)) for d in t.shape)
        print(f"{t.name} {t.tensor_type.name} {dims} {t.data_offset - r.data_offset} {t.n_bytes}")


def values(r, out, types):
    for k, t in enumerate(r.tensors):
        if t.tensor_type.name not in types:
            continue
        x = gguf.quants.dequantize(t.data, t.tensor_type).astype("<f4").reshape(-1)
        x.tofile(os.path.join(out, f"{k}.f32"))
        print(f"{t.name} {t.tensor_type.name} {x.size} differ 0")


if __name__ == "__main__":
    what, path = sys.argv[1], sys.argv[2]
    if what == "values":
        values(gguf.GGUFReader(path), sys.argv[3], sys.argv[4].split(","))
    else:
        {"meta": meta, "tensors": tensors}[what](gguf.GGUFReader(path))
