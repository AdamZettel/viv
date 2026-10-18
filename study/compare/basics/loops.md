---
title: Loops
order: 3
output: identical
---
- `for i in range(n)` is `for i to n`; `range(a, b, s)` is `for i from a to b by s`,
  and a negative step counts down in both. `reversed(range(n))` is
  `for i to n reversed`.
- Viv's `print` always ends the line; there is no `end=" "`. The Viv side builds
  the line in a string and prints it once.
- `viv lint` refuses an integer literal other than 0, 1 and 2 as a loop bound, so
  the Viv side names its bounds (`FIRST`, `LAST`, `STEP`). Python has no such rule;
  the Python side names the ones that mean something.
- `enumerate`, `zip`, the comprehension `[p * p for p in primes]` and the
  if-expression `n // 2 if n % 2 == 0 else 3 * n + 1` are the same in both.
  (`primes * primes` would also multiply element by element in Viv.)
- A comprehension's `for` takes Viv's loop heads: `for c from 1 to TABLE`
  where Python writes `for c in range(1, TABLE)`.
- `continue` and `break` are the same. `found.append(k)` is
  `found = append(found, k)`: an append gives back a new array, written by
  rebinding the name.
- A comparison of two arrays is elementwise in Viv, so a test of equality is
  `all(found == [3, 5])`.
