---
title: Bisection
order: 2
output: identical
---
- The loop, the tuple assignment `lo, flo = mid, fmid` and the two results
  `return root, steps` are the same in both.
- `raise ValueError(...)` is `stop("...")`. Python's caller could catch it;
  nothing catches a Viv `stop`.
- A `def` is a value in both: `bisect(cubic, 2.0, 3.0, 1e-10)` passes the
  function by name.
- `x ** 3` is `x ^ 3`.
- The loop `for tol in [1e-2, 1e-4, 1e-8]` walks an array literal in Viv as it
  walks a list in Python.
