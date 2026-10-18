---
title: Growing a list
order: 1
output: identical
---
- `squares.append(x)` is `squares = append(squares, x)`. The append gives back
  the grown array and the program rebinds the name; a second name taken
  before the append keeps the old array.
- The comprehensions are the same in both. A comprehension of numbers makes a
  Viv array; `squares[:HEAD] / 2` would divide every element too.
- `squares[-1]` and `squares[:-1]` count from the end in both.
- `extend` is a function that returns the joined array, as `append` is.
- `v in squares` and `a if c else b` read the same in both. On an array of
  numbers `in` compares as `==` does, so a NaN is in no array, as in NumPy.
- A list of lists is a matrix in Viv. Appending a row to `[]` makes it 1 by 3,
  `shape(table)` gives its size, and `for row in rows(table)` walks its rows
  (`for row in table` would walk its numbers).
- Viv's `print` of an array writes whole numbers without `.0`; `fmt(halves)`
  writes each element in Python's form, `[0.0, 0.5, 2.0, 4.5]`.
