---
title: Word count
order: 2
output: identical
---
- `splitlines`, `split`, `strip` with a set of characters, and `lower` are the
  same calls, and `"\n".join(LINES)` is `join("\n", LINES)`.
- A trailing comma after the last element of an array literal is allowed in
  both.
- `[clean(w) for w in words]`, `max(cleaned, key=len)` and
  `enumerate(lines, start=1)` are the same in both.
- A number in a Viv f-string is written as `str` writes a float (`1.0`), so the
  Viv side asks for `{n:d}` where Python's int prints `1`.
