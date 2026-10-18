---
title: Functions
order: 4
output: identical
---
- Viv has no default arguments and no keyword arguments. The Viv side writes the
  general function with every parameter (`greet_with`) and a short one that
  fills in the default (`greet`).
- A docstring works the same way: a bare string as the first statement of a
  `def` documents it and does nothing when it runs.
- `return lo, hi, mean` and `lo, hi, mean = f(xs)` are the same in both. In Viv
  the unpacking must be a statement of its own: `print(f(xs))` is refused.
- A lambda is `fn(x) -> expr` instead of `lambda x: expr`. A `def` name and a
  lambda are both values that can be passed and called.
- `min`, `max` and `sum` take an array in Viv as they take a list in Python.
- Python's `print(lo, hi, mean)` writes each float in full; the Viv side uses
  `fmt(...)` to get the same digits (see Printing and formatting).
