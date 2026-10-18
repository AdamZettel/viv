---
title: Tower of Hanoi
order: 3
output: identical
---
- Both sides return the list of moves instead of appending to a global list.
  In Viv that is the only way: a function cannot grow a global array, because
  `moves = append(moves, x)` inside a `def` makes `moves` a local.
- `return []` gives an empty list; Viv learns that it holds strings from the
  `append` of an f-string, and `moves + hanoi(...)` joins two lists of strings
  as Python's `+` does.
- `f"disk {n}"` needs `:d` in Viv to print `1` rather than `1.0`.
- `2 ** MAX_DISKS - 1` is `2 ^ MAX_DISKS - 1`; power binds tighter than minus
  in both.
