---
title: Factorial
order: 1
output: identical
---
- The recursive function reads the same in both, with braces for the body.
- Python's integers have no size limit. A Viv number is a double, which holds
  every integer up to 2^53 exactly; 20! is exact because its odd part is small,
  but 23! and beyond are rounded. The table stops at 20 so both sides print the
  same digits.
- `f"{factorial(n):d}"` prints the double as a whole number, with no `.0`.
- `str(n)` of a Python int is its digits; Viv's `str` of a number is Python's
  `repr` of a float (`2.43290200817664e+18`), so the Viv side counts the digits
  of `fmt(x, "d")`.
- Recursion goes 2500 calls deep in every Viv engine, and a deeper call stops
  the run with a message; Python's default limit is 1000.
