# Option kernels: benchmark results

Machine: Apple M2 Pro (arm64), macOS 26.1, 12 cores, 17 GB. Python 3.11.5,
NumPy 1.24.4, SciPy 1.12.0, Numba 0.57.1, Apple clang. Each time is the
median of 5 warm runs (one untimed run first, so Numba has compiled and
Viv's library is loaded). Every run started with the 1-minute load average
below 4 (between 1.8 and 2.7) and ran alone. Reproduce with
`python examples/options/bench.py iv|crr|pde|asian`; the raw numbers are in
`bench_results.json`.

Each column runs the same algorithm on the same inputs. Pure Python runs a
slice of the work (1/50 of the quotes or paths, 1/100 of the options) and
its time is multiplied up; the other columns run the full size.

| benchmark | size | pure Python | NumPy | Numba | Viv exact | Viv fast |
|---|---|---:|---:|---:|---:|---:|
| implied vol | 1e6 quotes | 7.79 s | 4.52 s | 0.585 s | 0.863 s | 0.568 s |
| CRR American put | 1e3 options, N = 1000 | 118 s | 10.8 s | 0.571 s | 0.897 s | 0.688 s |
| Crank-Nicolson American put | 2000 x 1000 grid | 1.38 s | 1.04 s | 0.029 s | 0.042 s | 0.036 s |
| MC arithmetic Asian | 1e6 paths x 12 dates | 22.0 s | 0.660 s | 0.258 s | 1.42 s | 1.22 s |

What each column is:

- **NumPy.** Implied vol: the same Newton iteration on all quotes at once,
  a fixed 60 steps. CRR: one option at a time, each time step a whole-array
  operation over the nodes. Crank-Nicolson: the right-hand side as array
  operations; the Brennan-Schwartz sweep is a recurrence and stays a Python
  loop, which is why NumPy barely beats pure Python there. Asian: Philox on
  uint64 arrays over all paths at once.
- **Numba.** `@njit` of the pure-Python source (no `fastmath`, no
  `parallel`), warm.
- **Viv exact** is what `import bs` gives: `-O2 -ffp-contract=off`, results
  identical to the interpreter's. **Viv fast** is `vipyr.load(...,
  fast=True)`: `-O3 -ffp-contract=fast`.

## Where Numba wins, and why

Numba is faster than Viv exact on all four, and faster than Viv fast on
three; Viv fast ties it on implied vol.

- **Bounds checks.** Every Viv array read and write is checked (an index
  out of range raises `VivIndexError` in Python). Numba does not check
  indices unless asked (`boundscheck=True`). The CRR and PDE inner loops
  are mostly indexed loads and stores.
- **Checked integers.** Viv's `Int` arithmetic stops on overflow, and its
  `//` and `%` check for division by zero. Numba's int64 arithmetic wraps.
- **No integer bit operations in Viv yet.** Philox needs 32-bit xor and
  shifts. Viv's `bxor` works on numbers (doubles holding 32-bit patterns),
  so each round converts between `Int` and `Num`, and the high word of a
  product comes from a checked floor division. Numba compiles the same
  steps to native uint64 instructions. The Asian benchmark, which is
  mostly Philox, has the largest gap (5.5x), and NumPy's vectorized Philox
  also beats Viv there.
- **Libm.** Viv calls its own `exp`, `log`, `cos` (the same bits on every
  platform); Numba calls the platform's. This share of the gap was not
  measured separately.

The exact build pays for reproducibility; the fast build lets the C
compiler contract and reorder, and comes within 1.25x of Numba on three of
the four (and ties it on implied vol).

## Agreement between columns

Viv exact and Numba give the same bits on the Crank-Nicolson put (the same
operations in the same order, with no library call in the loop). CRR
differs by 1.5e-12 (the tree's `exp`), implied vol by 6.5e-12 (NumPy's
fixed step count and scipy's `ndtr` against Viv's Cody CDF), and the Asian
plain estimate by 5e-14 (`log`, `cos`, `sin`).

## Lending arrays instead of copying them

A Viv def that only reads an array argument now gets the caller's NumPy
buffer itself. Summing a 240 MB array (`total(a: Arr)`): the library's peak
memory grows by 0 MB and the call takes 26 ms; a twin that writes one
element of its argument is copied, grows the peak by 240 MB and takes
49 ms.
