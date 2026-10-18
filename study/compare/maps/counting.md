---
title: Counting with number keys
order: 2
output: identical
---
- A map's keys are numbers or strings, one kind per map, and so are its values.
  `print(sums)` writes `{2: 1, 3: 2, ...}` as Python writes a dict with int keys.
- `sorted(sums)` is `keys(sums)` followed by `sort(...)` on its own line: `keys`
  gives a new array of the keys in insertion order.
- `range(1, SIDES + 1)` is `for a from 1 to SIDES + 1`: the end is left out in
  both.
- `x not in first_at` is the same in both.
- `dict(first_at)` as a copy is `copy(first_at)`. Plain assignment shares one
  map in both languages.
- Python 3.11 needs single quotes for a string inside an f-string field
  (`{'#' * n}`); Viv reads `{"#" * n}` inside a double-quoted f-string, as
  Python 3.12 does.
- `max(sums, key=...)` is `max(keys(sums), key=...)`: a Viv map is not a
  sequence, so its keys are asked for. Both keep the first key with the largest
  count.
