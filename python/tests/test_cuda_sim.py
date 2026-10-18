# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Viv's GPU kernels (examples/cuda) against the same kernels
written in numba and run by numba's CUDA simulator: an independent
implementation of the CUDA execution model (Python threads, real barriers,
shared arrays). Every value must be equal bit for bit.

    NUMBA_ENABLE_CUDASIM=1 python -m pytest python/tests/test_cuda_sim.py
    python python/tests/test_cuda_sim.py          (without pytest)

Skips when numba is absent. The simulator kernels call Viv's own libm
(lang/viv/src/vmath.c, built here into a small shared library), as the
device build does; how many elements Python's libm would round differently
is printed, not required.
"""
import ctypes
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")

import numpy as np  # noqa: E402

try:
    import pytest
except ImportError:  # the script form below runs without it
    pytest = None

try:
    from numba import cuda, float64
except ImportError:
    cuda = None
    if pytest is not None:
        pytest.skip("numba is not installed", allow_module_level=True)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

EXAMPLES = ROOT / "examples" / "cuda"
SPAN, LEAF, LANES = 8192, 128, 8
LANE_SLOTS, MAX_NODES = 1024, 255
THREADS = 256

_lib = None
_mods = {}


def vmath():
    """Viv's exp, as a C function Python can call."""
    global _lib
    if _lib is None:
        d = Path(tempfile.mkdtemp(prefix="viv_vmath_"))
        (d / "v.c").write_text(f'#include "{ROOT}/lang/viv/src/vmath.c"\ndouble vv_exp(double x) {{ return viv_exp_cr(x); }}\n')
        so = d / "libvv.so"
        subprocess.run([os.environ.get("CC", "cc"), "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-w", str(d / "v.c"), "-o", str(so), "-lm"],
                       check=True)
        _lib = ctypes.CDLL(str(so))
        _lib.vv_exp.restype = ctypes.c_double
        _lib.vv_exp.argtypes = [ctypes.c_double]
    return _lib


def viv(name):
    """An example compiled by vipyr (C, the exact build)."""
    if name not in _mods:
        import vipyr as vp
        _mods[name] = vp.load(str(EXAMPLES / f"{name}.viv"))
    return _mods[name]


def bits(a):
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def wave(n, k, scale):
    return np.array([scale * math.sin(k * i + 0.25) for i in range(n)])


def data(n, k):
    return np.array([math.sin(k * i) * math.exp(0.001 * (i % 97)) + 1.0e-3 * math.cos(3.0 * i) for i in range(n)])


# ---------------------------------------------------------------- elementwise

if cuda is not None:
    @cuda.jit
    def damped_sim(a, x, z, out):
        i = cuda.grid(1)
        if i < out.shape[0]:
            out[i] = a * x[i] + vmath().vv_exp(-z[i]) / (1.0 + z[i] * z[i])

    @cuda.jit
    def damped_sim_pylibm(a, x, z, out):
        i = cuda.grid(1)
        if i < out.shape[0]:
            out[i] = a * x[i] + math.exp(-z[i]) / (1.0 + z[i] * z[i])


def run_elementwise(kernel, a, x, z):
    out = np.zeros(len(x))
    blocks = (len(x) + THREADS - 1) // THREADS
    kernel[blocks, THREADS](a, x, z, out)
    return out


def test_elementwise_tree_matches_the_simulator():
    for n, a in [(1000, 2.5), (257, -0.5), (1, 3.0)]:
        x, z = wave(n, 0.37, 4.0), wave(n, 0.11, 3.0)
        mine = viv("elementwise").damped(a, x, z)
        sim = run_elementwise(damped_sim, a, x, z)
        assert np.array_equal(bits(mine), bits(sim)), f"n={n}"
        other = run_elementwise(damped_sim_pylibm, a, x, z)
        print(f"elementwise n={n}: equal to the simulator; {int(np.sum(bits(other) != bits(sim)))} of {n} differ with Python's libm exp")


# ---------------------------------------------------------------- parallel for

if cuda is not None:
    @cuda.jit
    def heat_sim(u, r, out):
        i = cuda.grid(1)
        n = u.shape[0]
        if i < n:
            if i == 0 or i == n - 1:
                out[i] = u[i]
            else:
                out[i] = u[i] + r * (u[i - 1] - 2.0 * u[i] + u[i + 1])


def test_parallel_for_matches_the_simulator():
    n, r, steps = 101, 0.4, 50
    u0 = np.array([math.exp(-((i - 50.0) / 8.0) ** 2) for i in range(n)])
    mine, sim = u0.copy(), u0.copy()
    m = viv("heat")
    for _ in range(steps):
        mine = m.heat_loop(mine, r)
        out = np.zeros(n)
        heat_sim[(n + THREADS - 1) // THREADS, THREADS](sim, r, out)
        sim = out
        assert np.array_equal(bits(mine), bits(sim))
    print(f"heat: {steps} steps of {n} points equal to the simulator")


# ---------------------------------------------------------------- the canonical sum

def pairwise_plan(m):
    """examples/cuda/pairwise.viv's plan, written again in Python."""
    starts, lens, lefts, rights = [0], [m], [-1], [-1]
    k = 0
    while k < len(starts):
        if lens[k] > LEAF:
            half = lens[k] // 2
            half -= half % LANES
            lefts[k] = len(starts)
            starts.append(starts[k]); lens.append(half); lefts.append(-1); rights.append(-1)
            rights[k] = len(starts)
            starts.append(starts[k] + half); lens.append(lens[k] - half); lefts.append(-1); rights.append(-1)
        k += 1
    count = len(starts)
    height = [0] * count
    for q in reversed(range(count)):
        if lefts[q] >= 0:
            height[q] = 1 + max(height[lefts[q]], height[rights[q]])
    newid, leaves = [0] * count, 0
    for q in range(count):
        if lefts[q] < 0:
            newid[q] = leaves
            leaves += 1
    level_end, nxt = [], leaves
    for h in range(height[0]):
        for q in range(count):
            if height[q] == h + 1:
                newid[q] = nxt
                nxt += 1
        level_end.append(nxt)
    leaf_start, leaf_len = [0] * leaves, [0] * leaves
    left, right = [0] * (count - leaves), [0] * (count - leaves)
    for q in range(count):
        if lefts[q] < 0:
            leaf_start[newid[q]], leaf_len[newid[q]] = starts[q], lens[q]
        else:
            left[newid[q] - leaves], right[newid[q] - leaves] = newid[lefts[q]], newid[rights[q]]
    ints = lambda v: np.array(v, dtype=np.int64)  # noqa: E731
    return ints(leaf_start), ints(leaf_len), ints(left), ints(right), ints(level_end)


if cuda is not None:
    @cuda.jit
    def pairwise_sim(x, w, partial, first, leaf_start, leaf_len, left, right, level_end):
        lanes = cuda.shared.array(LANE_SLOTS, float64)
        nodes = cuda.shared.array(MAX_NODES, float64)
        t = cuda.threadIdx.x
        leaf, lane = t // LANES, t % LANES
        base = (first + cuda.blockIdx.x) * SPAN
        leaves = leaf_start.shape[0]
        if leaf < leaves:
            lo, n = base + leaf_start[leaf], leaf_len[leaf]
            if n >= LANES:
                acc = x[lo + lane] * w[lo + lane]
                k = LANES
                while k < n - n % LANES:
                    acc = acc + x[lo + k + lane] * w[lo + k + lane]
                    k += LANES
                lanes[t] = acc
        cuda.syncthreads()
        if leaf < leaves and lane == 0:
            lo, n = base + leaf_start[leaf], leaf_len[leaf]
            if n < LANES:
                s = 0.0
                for q in range(n):
                    s = s + x[lo + q] * w[lo + q]
            else:
                s = ((lanes[t] + lanes[t + 1]) + (lanes[t + 2] + lanes[t + 3])) + ((lanes[t + 4] + lanes[t + 5]) + (lanes[t + 6] + lanes[t + 7]))
                r = n - n % LANES
                while r < n:
                    s = s + x[lo + r] * w[lo + r]
                    r += 1
            nodes[leaf] = s
        for h in range(level_end.shape[0]):
            cuda.syncthreads()
            i = leaves + t
            if h > 0:
                i = level_end[h - 1] + t
            if i < level_end[h]:
                nodes[i] = nodes[left[i - leaves]] + nodes[right[i - leaves]]
        cuda.syncthreads()
        if t == 0:
            partial[first + cuda.blockIdx.x] = nodes[leaves + left.shape[0] - 1]


def sim_dot(x, w):
    n = len(x)
    full, rest = divmod(n, SPAN)
    partial = np.zeros(full + (1 if rest else 0))
    for first, blocks, m in [(0, full, SPAN), (full, 1, rest)]:
        if blocks and m:
            plan = pairwise_plan(m)
            pairwise_sim[blocks, LANES * len(plan[0])](x, w, partial, first, *plan)
    total = 0.0
    for p in partial:
        total = total + p
    return total


SIZES = [1, 7, 8, 100, 128, 129, 1000, 8192, 8200, 17000]


def pairwise(f):
    """NumPy's pairwise sum (numpy/_core/src/umath/loops_utils.h.src,
    pairwise_sum), written again in Python."""
    n = len(f)
    if n < LANES:
        s = 0.0
        for v in f:
            s += v
        return s
    if n <= LEAF:
        r = list(f[:LANES])
        i = LANES
        while i < n - n % LANES:
            for j in range(LANES):
                r[j] += f[i + j]
            i += LANES
        s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        while i < n:
            s += f[i]
            i += 1
        return s
    n2 = n // 2
    n2 -= n2 % LANES
    return pairwise(f[:n2]) + pairwise(f[n2:])


def canonical(f):
    """Viv's `sum` (interp/arrays.rs, npsum): from 0.0, span by span of 8192."""
    s = 0.0
    for b in range(0, len(f), SPAN):
        s += pairwise(f[b:b + SPAN])
    return s


def test_canonical_sum_matches_the_simulator():
    m = viv("pairwise")
    differ = []
    for n in SIZES:
        x = data(n, 0.731)
        mine, sim = m.gpu_sum(x), sim_dot(x, np.ones(n))
        assert bits(mine) == bits(sim) == bits(canonical(list(x))), f"n={n}: {mine!r} {sim!r}"
        # numpy.sum of a contiguous vector is one pairwise tree over all of it
        assert bits(np.sum(x)) == bits(pairwise(list(x)))
        if bits(np.sum(x)) != bits(mine):
            differ.append(n)
    assert all(n > SPAN for n in differ)
    print(f"sum: Viv's kernel = the simulator's = Viv's sum, bit for bit, at n = {SIZES}; numpy.sum differs at n = {differ} (one tree, no spans)")


def test_canonical_dot_matches_the_simulator():
    m = viv("pairwise")
    for n in [3, 64, 777, 8192, 16384, 17000]:
        u, v = data(n, 0.731), data(n, 0.173)
        mine, sim = m.gpu_dot(u, v), sim_dot(u, v)
        assert bits(mine) == bits(sim) == bits(canonical(list(u * v))), f"n={n}"
        if n <= SPAN:
            assert bits(mine) == bits(np.sum(u * v))
    print("dot: Viv's kernel = the simulator's = Viv's sum(u * v), bit for bit")


def test_the_plans_agree():
    m = viv("pairwise")
    for n in [5, 129, 1000, 8192]:
        for a, b in zip(m.pairwise_plan(n), pairwise_plan(n)):
            assert np.array_equal(np.asarray(a), b)



# ---------------------------------------------------------------- device-resident arrays

if cuda is not None:
    @cuda.jit
    def heat2d_sim(u, v, w, r):
        i = cuda.grid(1)
        n = u.shape[0]
        if i < n:
            h = n // w
            row = i // w
            col = i % w
            if row == 0 or col == 0 or row == h - 1 or col == w - 1:
                v[i] = u[i]
            else:
                v[i] = u[i] + r * (u[i - 1] + u[i + 1] + u[i - w] + u[i + w] - 4.0 * u[i])


if cuda is not None:
    @cuda.jit
    def heat2d_grid_sim(u, v, w, h, r):
        col, row = cuda.grid(2)
        if col < w and row < h:
            i = row * w + col
            if row == 0 or col == 0 or row == h - 1 or col == w - 1:
                v[i] = u[i]
            else:
                v[i] = u[i] + r * (u[i - 1] + u[i + 1] + u[i - w] + u[i + w] - 4.0 * u[i])


def heat2d_initial(w, h):
    """examples/cuda/heat2d.viv's `initial`, with the same operations."""
    u = np.zeros(w * h)
    for row in range(h):
        for col in range(w):
            x = col / (w - 1)
            y = row / (h - 1)
            u[row * w + col] = 16.0 * x * (1.0 - x) * y * (1.0 - y)
    return u


def heat2d_numpy(u, w, r, steps):
    """The step with NumPy's 2-D slices, the kernel's sum in its order."""
    h = len(u) // w
    U = u.reshape(h, w).copy()
    for _ in range(steps):
        V = U.copy()
        V[1:h - 1, 1:w - 1] = U[1:h - 1, 1:w - 1] + r * (U[1:h - 1, 0:w - 2] + U[1:h - 1, 2:w] + U[0:h - 2, 1:w - 1]
                                                         + U[2:h, 1:w - 1] - 4.0 * U[1:h - 1, 1:w - 1])
        U = V
    return U.ravel()


def test_device_resident_heat2d_matches_the_simulator():
    """`on device u, v` in Viv against device arrays in numba: two
    `cuda.to_device` copies in, a launch and a swap per step, one copy back."""
    w, h, r, steps = 24, 16, 0.2, 50
    u0 = heat2d_initial(w, h)
    mine = viv("heat2d").heat2d_flat_device(u0, w, r, steps)
    du, dv = cuda.to_device(u0), cuda.to_device(u0)
    blocks = (w * h + THREADS - 1) // THREADS
    for _ in range(steps):
        heat2d_sim[blocks, THREADS](du, dv, w, r)
        du, dv = dv, du
    sim = du.copy_to_host()
    assert np.array_equal(bits(mine), bits(sim))
    print(f"heat2d: {steps} steps of {w}x{h} on the device equal to the simulator")


def test_heat2d_on_a_2d_grid_matches_the_simulator():
    """The 2-D grid's kernel (`schedule heat2d(threads = [TX, TY], cover =
    [w, h])`) against numba's `cuda.grid(2)` with the same block shape, on
    grids that are not a multiple of the block along either axis."""
    tx, ty = 32, 8
    for w, h, steps in [(24, 16, 50), (37, 29, 20)]:
        u0 = heat2d_initial(w, h)
        mine = viv("heat2d").heat2d_device(u0, w, 0.2, steps)
        du, dv = cuda.to_device(u0), cuda.to_device(u0)
        grid = ((w + tx - 1) // tx, (h + ty - 1) // ty)
        for _ in range(steps):
            heat2d_grid_sim[grid, (tx, ty)](du, dv, w, h, 0.2)
            du, dv = dv, du
        assert np.array_equal(bits(mine), bits(du.copy_to_host())), f"{w}x{h}"
        print(f"heat2d on a 2-D grid: {steps} steps of {w}x{h} equal to the simulator")


def test_device_resident_heat2d_matches_numpy():
    """1000 steps, and a grid that is not a multiple of the block, against NumPy's slices."""
    for w, h, steps in [(24, 16, 1000), (37, 29, 200)]:
        u0 = heat2d_initial(w, h)
        mine = viv("heat2d").heat2d_device(u0, w, 0.2, steps)
        assert np.array_equal(bits(mine), bits(heat2d_numpy(u0, w, 0.2, steps))), f"{w}x{h}"

# ---------------------------------------------------------------- matrix product

TILE, MM_THREADS = 16, (16, 16)
MODULUS, HALF, A_MULT, B_MULT = 2003, 1001.0, 7919, 104729
MM_SHAPES = [(20, 13, 29), (33, 40, 17), (64, 64, 64), (1, 70, 3)]

if cuda is not None:
    @cuda.jit
    def matmul_naive_sim(a, b, c, m, k, n):
        col, row = cuda.grid(2)
        if row < m and col < n:
            acc = 0.0
            for p in range(k):
                x = a[row * k + p]
                if x != 0.0:
                    acc = acc + x * b[p * n + col]
            c[row * n + col] = acc

    @cuda.jit
    def matmul_tiled_sim(a, b, c, m, k, n):
        # written apart from Viv's: the running sum stays in a register across
        # barriers, as a CUDA programmer would keep it
        sa = cuda.shared.array(TILE * TILE, float64)
        sb = cuda.shared.array(TILE * TILE, float64)
        tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
        col = cuda.blockIdx.x * TILE + tx
        row = cuda.blockIdx.y * TILE + ty
        acc = 0.0
        for q in range((k + TILE - 1) // TILE):
            p0 = q * TILE
            sa[ty * TILE + tx] = a[row * k + p0 + tx] if row < m and p0 + tx < k else 0.0
            sb[ty * TILE + tx] = b[(p0 + ty) * n + col] if p0 + ty < k and col < n else 0.0
            cuda.syncthreads()
            for p in range(TILE):
                x = sa[ty * TILE + p]
                if x != 0.0:
                    acc = acc + x * sb[p * TILE + tx]
            cuda.syncthreads()
        if row < m and col < n:
            c[row * n + col] = acc


def mm_filled(rows, cols, mult):
    """examples/cuda/matmul.viv's `filled`, with the same operations."""
    i = np.arange(rows * cols, dtype=np.int64)
    return ((i * mult) % MODULUS) / HALF - 1.0


def test_matmul_matches_the_simulator():
    """Both kernels of examples/cuda/matmul.viv (the emulator, in Viv's C
    build) against the same kernels in numba, and against the CPU's `@`."""
    mod = viv("matmul")
    for m, k, n in MM_SHAPES:
        a, b = mm_filled(m, k, A_MULT), mm_filled(k, n, B_MULT)
        want = bits(mod.reference(m, k, n))
        assert np.array_equal(bits(mod.product_naive(m, k, n)), want), f"naive {m}x{k}x{n}"
        assert np.array_equal(bits(mod.product_tiled(m, k, n)), want), f"tiled {m}x{k}x{n}"
        for kern, threads, grid in [(matmul_naive_sim, MM_THREADS, ((n + 15) // 16, (m + 15) // 16)),
                                    (matmul_tiled_sim, (TILE, TILE), ((n + TILE - 1) // TILE, (m + TILE - 1) // TILE))]:
            c = np.zeros(m * n)
            kern[grid, threads](a, b, c, m, k, n)
            assert np.array_equal(bits(c), want), f"{kern} {m}x{k}x{n}"
        print(f"matmul {m}x{k} @ {k}x{n}: naive and tiled equal to the simulator and to @")


if __name__ == "__main__":
    if cuda is None:
        print("skip: numba is not installed")
        sys.exit(0)
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
