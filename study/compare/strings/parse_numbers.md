---
title: Numbers from text
order: 5
output: identical
---
- `float` and `int` read text as CPython does: surrounding spaces, exponents,
  `inf`, and `int(s, 16)` for another base. In Viv both give a double.
- A dict of strings is a `map()` in Viv; `fields[key] = value` and `f["temp"]`
  read and write it as in Python.
- `key, value = part.split("=")` unpacks a two-element list in both. In Viv a
  list of another length stops the run with its line, as Python raises
  `ValueError`.
- `names.append(x)` is `names = append(names, x)`, and the empty `[]` becomes a
  list of strings or an array of numbers from what is appended to it.
- `for name, t in zip(names, temps)` is the same in both.
- Python prints `float("  2.5e-3 ")` as `0.0025`; Viv's `print` would write the
  same six digits here, but the Viv side uses `fmt` to match Python in general.
