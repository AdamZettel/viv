---
title: Simpson's rule
order: 3
output: identical
---
- Python passes `math.sin` itself; a Viv builtin is not a value (a function parameter takes a `def` name or a `fn`, and `sin` is
  refused), so the Viv side wraps it: `fn(x) -> sin(x)`.
- The conditional expressions `(4 if i % 2 == 1 else 2)` and
  `f"..." if prev else "     -"` are the same in both.
- `prev = None` and `if prev:` have no Viv form (there is no `None`); the Viv side
  starts `prev` at 0 and tests `prev > 0`.
- Viv has a builtin `simpson(f, a, b, n)`. The program names its own
  `simpson_rule` so both can be called, and the test checks that they agree.
  (A program's own `def simpson` would replace the builtin everywhere in it.)
- The ratio column is about 16 in both: halving `h` divides the error by 2^4.
