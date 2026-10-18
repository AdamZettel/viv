---
title: Caesar cipher
order: 4
output: identical
---
- `ord` and `chr` are the same functions, and `%` gives a non-negative
  remainder for a positive divisor in both, so a negative shift wraps around.
- Python's chained `"a" <= c <= "z"` is two comparisons joined by `and` in Viv.
  Strings compare by code point in both.
- `for k in range(ALPHABET)` is `for k to ALPHABET`.
- `"".join(shift_char(c, k) for c in text)` and `print("key", k, "gives", guess)`
  are the same in both: `print` puts a space between its arguments.
