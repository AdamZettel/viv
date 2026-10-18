---
title: Runge-Kutta 4 on an oscillator
order: 4
output: identical
---
- `np.array([y[1], -w * w * y[0]])` is the array literal `[y[1], -w * w * y[0]]`:
  Viv arrays need no library, and `+`, `-`, `*` and `/` between an array and a
  number, or two arrays of one shape, work elementwise as NumPy's do.
- `rk4_step` is the same five lines in both.
- `for _ in range(steps)` names its counter in Viv (`for k to steps`).
- A variable assigned inside a loop at the top level is a global in both, so
  the test reads the last `err`.
- `np.cos` on a number is the builtin `cos`.
- `assert err < 1e-6` is `assert_lt(err, 1e-6)`, which prints both numbers
  when it fails.
