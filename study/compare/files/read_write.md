---
title: Reading and writing text files
order: 1
output: identical
---
- `write_text(path, s)`, `read_text(path)` and `read_lines(path)` are builtins
  that behave as `Path.write_text`, `Path.read_text` and
  `read_text().splitlines()`: UTF-8, newlines read as `\n`, and `write_text`
  gives back the number of characters written.
- Viv has no `with open(...)` and no file objects; a file is read or written
  whole. Reading a missing file stops the run with Python's words,
  `No such file or directory: '...'`.
- In the browser there is no file system: the wasm runtime keeps a table of
  files, so a program can write a file and read it back in the browser.
- `print(text, end="")` becomes `print(text.rstrip("\n"))`, since Viv's `print`
  always ends the line.
- `scores.index(max(scores))` is `max([k for k to len(scores)], key=fn(k) -> scores[k])`:
  `list.index` exists in Viv for lists of strings, not for arrays of numbers.
  Both give the first position of the largest score.
- The report is the same comprehension over `zip(names, scores)` in both.
- A string times a number repeats it in both: `"*" * (s // 10)`.
