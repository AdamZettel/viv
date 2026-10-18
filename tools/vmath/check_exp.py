#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Hold Viv's `exp` to correct rounding, and its three engines to one another.

    python3 tools/vmath/check_exp.py [--n 400000] [--engines N] [--viv target/release/viv]

1. The inputs: N dense ones (uniform on [-745.2, 709.8], on [-2, 2], on
   [-1e-3, 1e-3], log-uniform magnitudes from 1e-300 to 700, on the two ends
   [-746, -708] and [709, 709.79], and random bit patterns), and the hard cases: x = +-m 2^-k for small m and k from 20 to
   40 with their neighbors (there e^x = 1 + x + x^2/2 + ... sits within
   2^-80 or less of a midpoint between two doubles; x = 2^-26 is the classic
   one), x next to every multiple of ln2/128 (exp_cr's table boundaries), the
   ends of the fast range (-708, 709), the subnormal results (-745.13 to
   -708.4), overflow (709.78), and the specials.
2. The C engine's exp: lang/viv/src/vmath.c compiled with the flags `viv`
   uses, so `exp` is the #define every compiled program gets.
3. The reference: mpmath at 300 bits, rounded to the nearest double by
   CPython's float() of 70 significant digits (correctly rounded, subnormals
   included). Every input must give exactly that double.
4. The three engines on --engines of the inputs (every hard case, then dense
   ones): one Viv program printing fmt(exp(x)) per input, run by the
   interpreter, by the C backend and by the direct wasm backend in node; the
   three outputs must be the same bytes, and each line the reference.

The Rust twin's bits equal the C twin's on every input of
`cargo test -p viv --test vmath` (240,000 and the cases file), so step 2
stands for the interpreter's own exp too. Exits nonzero on any difference.
"""
import argparse
import math
import os
import random
import struct
import subprocess
import sys
import tempfile

import mpmath

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def f2b(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def b2f(b):
    return struct.unpack("<d", struct.pack("<Q", b & 0xFFFFFFFFFFFFFFFF))[0]


def nb(x, k):
    """the double k steps from x (k may be negative)"""
    if x == 0.0:
        return b2f(abs(k)) * (1 if k > 0 else -1)
    b = f2b(x)
    return b2f(b + k if x > 0 else b - k)


def hard_cases():
    xs = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 10.0, -10.0, 100.0, -100.0, 700.0, -700.0,
          math.log(2.0), -math.log(2.0), 709.0, -708.0, 708.9999999999999, -707.9999999999999,
          709.782712893384, 709.7827128933841, 709.79, -745.1332191019411, -745.1332191019412,
          -745.13, -745.2, -744.44007192138122, -720.0, -730.5, -740.0, -708.3964185322641,
          5.551115123125783e-17, -5.551115123125783e-17, 5.551115123125782e-17, 1e-300, -1e-300,
          5e-324, -5e-324, 2.2250738585072014e-308,
          # Lefevre and Muller's worst case for exp in double precision
          7.5417527749959590085206221e-10]
    for m in range(1, 65):
        for k in range(20, 41):
            for s in (1.0, -1.0):
                x = s * m * 2.0 ** -k
                xs += [x, nb(x, 1), nb(x, -1), nb(x, 2), nb(x, -2)]
    L = math.log(2.0) / 128
    for j in range(-128 * 12, 128 * 12 + 1):
        c = j * L + L / 2  # the rounding boundary of k = round(x 128/ln2)
        xs += [c, nb(c, 1), nb(c, -1), j * L]
    for x in (-708.0, 709.0):
        xs += [nb(x, k) for k in range(-3, 4)]
    return [x for x in xs if -746.0 < x < 710.0]


def dense(n, seed=20261007):
    r = random.Random(seed)
    xs = []
    while len(xs) < n:
        u = r.random()
        if u < 0.3:
            xs.append(r.uniform(-745.2, 709.8))
        elif u < 0.55:
            xs.append(r.uniform(-2.0, 2.0))
        elif u < 0.7:
            xs.append(r.uniform(-1e-3, 1e-3))
        elif u < 0.85:
            xs.append((1 if r.random() < 0.5 else -1) * 10.0 ** r.uniform(-300, math.log10(700)))
        elif u < 0.92:
            xs.append(r.uniform(-746.0, -708.0))
        elif u < 0.95:
            xs.append(r.uniform(709.0, 709.79))
        else:
            # any finite bit pattern whose result is neither 0 nor inf
            while True:
                x = b2f(r.getrandbits(64))
                if x == x and -746.0 < x < 710.0:
                    xs.append(x)
                    break
    return xs


def reference(x):
    if x == 0.0:
        return 1.0
    y = mpmath.exp(mpmath.mpf(x))
    return float(mpmath.nstr(y, 70, strip_zeros=False, min_fixed=1, max_fixed=0))


def c_engine(xs, tmp):
    src = open(os.path.join(ROOT, "lang", "viv", "src", "vmath.c")).read()
    src += r"""
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char** argv) {
    FILE* f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f) / 8; fseek(f, 0, SEEK_SET);
    double* x = malloc(8 * n); if (fread(x, 8, n, f) != (size_t)n) return 2; fclose(f);
    for (long i = 0; i < n; i++) x[i] = exp(x[i]);
    f = fopen(argv[2], "wb"); fwrite(x, 8, n, f); fclose(f);
    return 0;
}
"""
    c = os.path.join(tmp, "e.c")
    open(c, "w").write(src)
    exe = os.path.join(tmp, "e")
    subprocess.run(["cc", "-O2", "-ffp-contract=off", c, "-o", exe, "-lm"], check=True)
    inp, out = os.path.join(tmp, "x.bin"), os.path.join(tmp, "y.bin")
    open(inp, "wb").write(struct.pack(f"<{len(xs)}d", *xs))
    subprocess.run([exe, inp, out], check=True)
    d = open(out, "rb").read()
    return list(struct.unpack(f"<{len(xs)}d", d))


def three_engines(xs, refs, viv, tmp):
    prog = os.path.join(tmp, "exp_cases.viv")
    with open(prog, "w") as f:
        f.write("# exp of each input, as the shortest digits that read back\n")
        f.write("xs = [" + ", ".join(repr(x) for x in xs) + "]\n")
        f.write("for i to len(xs) {\n    print(fmt(exp(xs[i])))\n}\n")
    want = "".join(repr(y) + "\n" for y in refs)
    outs = {}
    p = subprocess.run([viv, "run", "-q", prog], capture_output=True, text=True)
    outs["interp"] = p.stdout
    exe = os.path.join(tmp, "exp_cases")
    subprocess.run([viv, "build", prog, "-o", exe], check=True, capture_output=True)
    outs["native"] = subprocess.run([exe], capture_output=True, text=True).stdout
    wasm = os.path.join(tmp, "exp_cases.wasm")
    subprocess.run([viv, prog, "-o", wasm], check=True, capture_output=True)
    outs["wasm"] = subprocess.run(["node", os.path.join(ROOT, "tools", "run-wasm.mjs"), wasm],
                                  capture_output=True, text=True).stdout
    bad = []
    for name, o in outs.items():
        if o != want:
            got, exp_ = o.splitlines(), want.splitlines()
            k = next((i for i in range(min(len(got), len(exp_))) if got[i] != exp_[i]), min(len(got), len(exp_)))
            bad.append(f"{name}: differs from the reference at line {k + 1} of {len(exp_)}"
                       f" (x = {xs[k]!r}: {got[k] if k < len(got) else 'nothing'} vs {exp_[k] if k < len(exp_) else 'nothing'})")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400000)
    ap.add_argument("--engines", type=int, default=20000)
    ap.add_argument("--viv", default=os.path.join(ROOT, "target", "release", "viv"))
    a = ap.parse_args()
    mpmath.mp.prec = 300
    hard = hard_cases()
    xs = hard + dense(a.n)
    with tempfile.TemporaryDirectory() as tmp:
        ys = c_engine(xs, tmp)
        wrong = []
        refs = []
        for i, (x, y) in enumerate(zip(xs, ys)):
            r = reference(x)
            refs.append(r)
            if f2b(r) != f2b(y):
                wrong.append((x, y, r))
        print(f"C engine (vmath.c `exp`): {len(xs)} inputs ({len(hard)} hard cases), "
              f"{len(wrong)} not correctly rounded")
        for x, y, r in wrong[:20]:
            print(f"  exp({x!r}) = {y!r}, the nearest double is {r!r}")
        bad = []
        if a.engines:
            k = min(len(xs), max(a.engines, len(hard)))
            bad = three_engines(xs[:k], refs[:k], a.viv, tmp)
            print(f"three engines (interpreter, C, wasm) on {k} inputs: "
                  + ("identical, every line the reference" if not bad else "DIFFER"))
            for b in bad:
                print("  " + b)
    sys.exit(1 if wrong or bad else 0)


if __name__ == "__main__":
    main()
