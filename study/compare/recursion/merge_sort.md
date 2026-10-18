---
title: Merge sort
order: 4
output: identical
---
- The algorithm is line for line the same. `out.append(x)` is
  `out = append(out, x)` and `out.extend(a[i:])` is `out = extend(out, a[i:])`.
- `i = j = 0` has no Viv form; the Viv side writes two assignments.
- Slices `xs[:mid]` and `xs[mid:]` are copies in both (Python lists copy on
  slicing; Viv arrays always do), so the sort never changes `data`.
- The test compares arrays with `all(a == b)`, since `==` between two arrays is
  elementwise in Viv; `sorted(data)` becomes `copy` then `sort`.
