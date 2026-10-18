#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Replay lib/random.viv (and lib/stats.viv) in Python, draw for draw.

Viv's rand() is Park-Miller MINSTD, specified exactly, so a Python program can
reproduce every number a Viv program draws. lib/random.viv documents each of its
draws as a formula over rand(); this file is those formulas in Python, and the
numbers it prints are the ones lib/random.viv's tests hold. lib/stats.viv's
mean and std of the first six normals are replayed the same way.

    python3 tools/random_replay.py           print the reference blocks
    python3 tools/random_replay.py --write   write them into the two files
    python3 tools/random_replay.py --check   compare them with the blocks in
                                             lib/random.viv and lib/stats.viv,
                                             then run both files' tests with
                                             `viv run`

MINSTD (the interpreter's `srand` and `rand` arms, and the C runtime's twin):
    srand(seed):  s = floor(|seed|) mod 2147483646 + 1
    rand():       s = 16807 s mod 2147483647;  return s / 2147483647
Every product stays below 2^53, so the state is exact in a double and in a
Python float alike.

The integer draws (randint, shuffle, randperm, choice) replay exactly. randn
goes through log and cos, which Viv computes with its own libm and Python with
the platform's; the two can differ in the last bit, so its test allows 1e-14.
"""
import math
import os
import subprocess
import sys

M = 2147483647.0
state = 1.0


def srand(seed):
    global state
    state = math.floor(abs(seed)) % 2147483646.0 + 1.0


def rand():
    global state
    state = (16807.0 * state) % M
    return state / M


def randn():
    # Box-Muller, cosine branch; u1 is drawn first
    u1 = rand()
    u2 = rand()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * 3.141592653589793 * u2)


def randn_matrix(rows, cols):
    return [[randn() for _ in range(cols)] for _ in range(rows)]


def randint(lo, hi):
    return lo + math.floor(rand() * (hi - lo))


def arange(n):
    return [float(i) for i in range(n)]


def shuffle(a):
    # Fisher-Yates from the top index down: i = n-1, ..., 1 swaps with j in [0, i]
    n = len(a)
    for k in range(n - 1):
        i = n - 1 - k
        j = math.floor(rand() * (i + 1))
        a[i], a[j] = a[j], a[i]


def randperm(n):
    a = arange(n)
    shuffle(a)
    return a


def choice(p):
    # inverse CDF: the first k whose running sum of p passes one uniform draw
    u = rand()
    cum = 0.0
    for k, pk in enumerate(p):
        cum = cum + pk
        if u < cum:
            return float(k)
    return float(len(p) - 1)


def mean(a):
    s = 0.0
    for x in a:
        s += x
    return s / len(a)


def std(a):
    m = mean(a)
    s = 0.0
    for x in a:
        s += (x - m) * (x - m)
    return math.sqrt(s / len(a))


def fmt(x):
    return repr(float(x))


def viv_array(xs):
    return "[" + ", ".join(fmt(x) for x in xs) + "]"


def reference_block():
    """The same draws, in the same order, as lib/random.viv's test program."""
    srand(7.0)
    normals = [randn() for _ in range(6)]
    ints = [randint(-3.0, 4.0) for _ in range(12)]
    perm = randperm(10)
    a = arange(6)
    shuffle(a)
    p = [0.1, 0.2, 0.3, 0.4]
    picks = [choice(p) for _ in range(12)]
    Z = randn_matrix(2, 3)
    flat = [x for row in Z for x in row]
    after = rand()
    random_lines = [
        "REF_NORMALS = " + viv_array(normals),
        "REF_INTS = " + viv_array(ints),
        "REF_PERM = " + viv_array(perm),
        "REF_SHUFFLED = " + viv_array(a),
        "REF_PICKS = " + viv_array(picks),
        "REF_MATRIX = " + viv_array(flat),
        "REF_AFTER = " + fmt(after),
    ]
    stats_lines = [
        "SAMPLE = " + viv_array(normals),
        "REF_MEAN = " + fmt(mean(normals)),
        "REF_STD = " + fmt(std(normals)),
    ]
    return {"random": "\n".join(random_lines) + "\n", "stats": "\n".join(stats_lines) + "\n"}


OPEN = "    # ---- replayed by tools/random_replay.py ----\n"
CLOSE = "    # ---- end of the replayed block ----\n"


def main():
    blocks = reference_block()
    if "--check" not in sys.argv and "--write" not in sys.argv:
        for name, block in blocks.items():
            sys.stdout.write(f"# lib/{name}.viv\n" + block)
        return 0
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    viv = os.environ.get("VIV", os.path.join(root, "target", "release", "viv"))
    bad = 0
    for name, block in blocks.items():
        lib = os.path.join(root, "lib", name + ".viv")
        src = open(lib).read()
        # the block sits inside the file's first test, indented by four spaces
        start = src.index(OPEN) + len(OPEN)
        end = src.index(CLOSE)
        want = "".join("    " + line + "\n" for line in block.splitlines())
        if src[start:end] != want:
            if "--write" in sys.argv:
                open(lib, "w").write(src[:start] + want + src[end:])
                print(f"lib/{name}.viv: reference block written")
            else:
                sys.stdout.write(f"lib/{name}.viv's reference block differs from this replay:\n" + block)
                bad = 1
                continue
        run = subprocess.run([viv, "run", "-q", lib], capture_output=True, text=True)
        sys.stdout.write(run.stdout + run.stderr)
        if run.returncode != 0 or "FAIL" in run.stdout:
            bad = 1
        else:
            print(f"lib/{name}.viv: the reference block is this replay, and its tests pass")
    return bad


if __name__ == "__main__":
    sys.exit(main())
