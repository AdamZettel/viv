---
title: Word frequency
order: 1
output: identical
---
- A dict is `map()` in Viv. `m[k] = v` stores and `m[k]` reads, and
  `counts.get(w, 0)` is `get(counts, w, 0)`.
- `print(counts)` writes the map as Python writes a dict,
  `{'the': 4, 'cat': 1, ...}`, and the keys stay in the order they were first
  stored, as a dict's do.
- `w in counts` is the same in both; `counts.pop("and")` is
  `remove(counts, "and")`; `list(counts)` and `counts.values()` are `keys(counts)`
  and `values(counts)`.
- Sorting the words by count: Python's key is a tuple, `(-counts[w], w)`. Viv's
  keys are numbers or strings, so the Viv side sorts twice: by word, then by
  count with `key=fn(w) -> -counts[w]`, a stable sort that keeps the words of
  one count in order.
- A map is a handle in Viv: a function that stores into a map it was given
  stores into the caller's map, as with a dict.
