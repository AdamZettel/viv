---
title: String basics
order: 1
output: identical
---
- The string methods are Python's, with Python's names and results: `strip`,
  `upper`, `split`, `find`, `replace`, `count`, `startswith`. `t.upper()` in Viv
  is the call `upper(t)`; any function can be called with a dot this way.
- `join` is the exception to the dot form: Python's `"-".join(words)` is Viv's
  `join("-", words)`, the separator first.
- Indices and slices count characters as Python's do, including negative
  indices and a step of -1 (`t[::-1]`). Arrays of numbers take negative indices
  too; their slice step must be 1 or more.
- `print(words)` writes a list of strings the way Python does,
  `['The', 'quick', 'brown', 'fox']`.
- f-string alignment (`{w:<8}`, `{w:>8}`, `{w:^8}`) works for strings in both.
- `in` tests for a substring in both languages.
