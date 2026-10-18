#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Write the reference for Viv's binary readers.

    python3 tools/bytes/gen_cases.py <dir>

<dir>/cases.bin: 2048 seeded random bytes, then every
binary16 class written out (zeros, subnormals, the normal range's ends,
infinities, NaNs), so le_f16 meets each of them.
<dir>/cases.txt: what lang/viv/tests/bytes/cases.viv must print, computed with
Python's struct module (`<B`, `<b`, `<H`, `<h`, `<I`, `<i`, `<Q`, `<q`, `<e`,
`<f`, `<d`; bfloat16 as the top half of a `<f`), one line per reader and
offset, numbers as format(x, ".17g"); and fround (round to binary32) of a
double made from each 8 bytes, against struct's `<f` round trip.
"""
import os
import random
import struct
import sys

N_RANDOM = 2048
SEED = 20261014

HALVES = [0x0000, 0x8000, 0x0001, 0x8001, 0x0200, 0x03FF, 0x83FF, 0x0400, 0x3C00, 0xBC00,
          0x3BFF, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0x7C01, 0xFE00, 0x3555, 0x5640]

# (name, width, struct format or None for bfloat16)
READERS = [("le_u8", 1, "<B"), ("le_i8", 1, "<b"), ("le_u16", 2, "<H"), ("le_i16", 2, "<h"),
           ("le_u32", 4, "<I"), ("le_i32", 4, "<i"), ("le_u64", 8, "<Q"), ("le_i64", 8, "<q"),
           ("le_f16", 2, "<e"), ("le_bf16", 2, None), ("le_f32", 4, "<f"), ("le_f64", 8, "<d")]


def text(v):
    if isinstance(v, str):
        return v
    return str(v) if isinstance(v, int) else format(v, ".17g")


def fround(x):
    """The nearest binary32: struct's `<f` packs with a C cast, and raises
    where the cast gives an infinity from a finite double."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return float("inf") if x > 0 else float("-inf")


def main():
    out = sys.argv[1]
    rng = random.Random(SEED)
    data = bytes(rng.randrange(256) for _ in range(N_RANDOM))
    data += b"".join(struct.pack("<H", h) for h in HALVES)
    lines = []
    for o in range(len(data)):
        for name, w, fmt in READERS:
            if o + w > len(data):
                continue
            if fmt is None:
                v = struct.unpack("<f", b"\0\0" + data[o:o + 2])[0]
            else:
                v = struct.unpack_from(fmt, data, o)[0]
            if name == "le_u64" and v > 2 ** 63 - 1:
                v = "over"
            lines.append(f"{name} {o} {text(v)}")
            if name == "le_f64":
                x = struct.unpack_from("<f", data, o)[0] * (1.0 + struct.unpack_from("<I", data, o + 4)[0] / 4294967296.0)
                lines.append(f"fround {o} {text(fround(x))}")
    ulp = 1.0 / 16777216.0
    lines.append(f"fround tie {text(fround(1.0 + ulp))} {text(fround(1.0 + 3.0 * ulp))} {text(fround(-1.0 - ulp))}")
    with open(os.path.join(out, "cases.bin"), "wb") as f:
        f.write(data)
    with open(os.path.join(out, "cases.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
