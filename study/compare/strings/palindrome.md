---
title: Palindromes
order: 3
output: identical
---
- `"".join(c.lower() for c in s if c.isalnum())` and
  `"yes" if is_palindrome(p) else "no"` are the same in both.
- `c.isalnum()` and `c.lower()` are Python's, Unicode included, so `"Été"`
  becomes `été` on both sides.
- `t[::-1]` reverses a string in both. (It does not reverse a Viv array of
  numbers: a slice step there must be 1 or more.)
- `return False` is `return false`: Viv's booleans are lower case.
- `i, j = 0, len(t) - 1` is the same statement in both.
