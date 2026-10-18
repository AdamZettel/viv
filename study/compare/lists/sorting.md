---
title: Sorting
order: 2
output: identical
---
- Viv's `sort(a)` sorts in place and gives back nothing, like Python's
  `a.sort()`. For a sorted copy (`sorted(nums)`), copy first:
  `ordered = copy(nums)`, then `sort(ordered)` on its own line.
- `sorted(nums, reverse=true)` is a sorted copy, largest first, as Python's
  `reverse=True` sorts: equal elements keep their order.
- `sorted` works on a list of strings in Viv, as in Python.
- Viv's keys are numbers or strings, not tuples. Sorting one list by another
  takes two stable sorts where Python passes
  `key=lambda i: (-scores[i], names[i])`: first by name (`argsort` gives the
  positions), then by score; a stable sort keeps the name order among equal
  scores.
- `list(a)` as a copy is `copy(a)`. Writing into an array parameter writes into
  the caller's array in Viv, so the copy matters there as it does in Python.
