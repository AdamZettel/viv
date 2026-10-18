---
title: Tuples and unpacking
order: 2
output: identical
---
- `return q, r` and `q, r = f(a, b)` read the same in both. Viv has no tuple
  value: a function returns several values and the caller unpacks them on a
  line of its own. `divmod_floor(-7, 2) == (-4, 1)` has no Viv form, so the test
  unpacks first.
- `a, b = b, a` swaps in both, and `v[0], v[2] = v[2], v[0]` swaps two elements:
  every value on the right is computed before any target is written.
- `x, y, z = point` unpacks an array of exactly three elements; another length
  stops the run, as Python raises `ValueError`.
- `for _ in range(n)` has no `_` in Viv; the loop names its counter.
- The generator expression `sum((x - mean) ** 2 for x in xs)` is one array
  expression in Viv, `sum((xs - mean) ^ 2)`: arithmetic between an array and a
  number applies to every element.
