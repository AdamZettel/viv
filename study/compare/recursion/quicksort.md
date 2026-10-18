---
title: Quicksort in place
order: 5
output: identical
---
- A list passed to a Python function and an array passed to a Viv function are
  both shared: `partition` swaps elements of the caller's array in either
  language.
- `a[i], a[j] = a[j], a[i]` swaps two elements in both.
- A Viv function with no `return` gives back 0, which the caller ignores;
  `quicksort` needs no return statement.
- `for j in range(lo, hi)` is `for j from lo to hi`.
- The generator in `all(... for k in ...)` becomes a loop of assertions inside
  the test.
