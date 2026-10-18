---
title: Slices, copies and aliases
order: 3
output: differs
reason: One line differs and is left out of the comparison. A NumPy slice is a view, so writing into it changes the array it came from (d[1] becomes 99.0); a Viv slice is a copy, so d[1] stays 0.0.
py_drop: ^d\[1\] after
viv_drop: ^d\[1\] after
---
- A slice is a copy in Viv and a view in NumPy. This is the one place where the
  same code computes something different; the program shows it on `d`.
- Plain assignment shares the array in both: after `c = a`, a write through `c`
  is seen through `a`.
- `a[-1]` and `a[-3:]` count from the end in both. An index before the start
  stops the run in Viv, as it raises in NumPy; a slice bound outside the array
  stops it too, where NumPy clamps.
- `np.arange(10.0)` has no builtin twin; the Viv side fills `zeros(N)` in a loop.
- `np.argmax` and `np.cumsum` are loops in Viv.
- `a[5:8] = 0.0` writes a number into every element of a slice in both.
- NumPy prints arrays as `[2. 3. 4.]`; the Python side prints `tolist()` and the
  Viv side prints `fmt(a)`, which both write `[2.0, 3.0, 4.0]`.
