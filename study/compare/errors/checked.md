---
title: Exceptions against checked results
order: 1
output: identical
---
- Viv has no exceptions and no `try`. A function that can fail returns a flag,
  its value and a message (`return false, 0, "why"`), and the caller tests the
  flag. The message here is written to match the one Python's `int` raises.
- `int("abc")` stops a Viv run with CPython's message, and nothing can catch
  it, so `parse_age` checks the text with `isdigit` before converting.
- `stop("...")` is Viv's `raise` for a condition that should never happen: it
  ends the run with `error:`, the message and the line, and exit status 1, in
  every engine. The program's check passes, so it does not run here.
- Dividing a double by zero in Viv gives `inf` or `nan`, as in C, where Python
  raises `ZeroDivisionError`; `ratio` tests for zero first. (Viv's `%` and `//`
  by zero stop the run, as Python's raise.)
- Viv has no `nan` constant; `float("nan")` works in both.
- `stock[item]` for a missing key stops a Viv run, as it raises `KeyError` in
  Python; `item in stock` tests first.
