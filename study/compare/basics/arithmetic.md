---
title: Arithmetic and conditionals
order: 2
output: differs
reason: Two lines print differently and are left out of the comparison. Python's round(2.5) is 2, because Python rounds a tie to the even neighbor; Viv's round(2.5) is 3, because Viv rounds a tie away from zero. Python prints a boolean as True or False, and Viv prints it as true or false.
py_drop: ^round\(2\.5\)|^comparisons:
viv_drop: ^round\(2\.5\)|^comparisons:
---
- A Viv number is a 64-bit float unless the program declares an `Int`; this
  program uses numbers only. `a // b` and `a % b` on numbers follow Python's rules for floats (the remainder takes the sign of
  `b`), so `-17 // 5` is `-4` and `-17 % 5` is `3` on both sides.
- Power is `^` in Viv and `**` in Python. Both are right-associative, so
  `2 ^ 3 ^ 2` is 512.
- `sqrt`, `exp`, `floor` and the rest are builtins in Viv; Python imports them
  from `math`. Viv has no `pi`: the program names the constant.
- `elif` is `else if`, and the Viv code here puts `else` on its own line after
  the closing brace.
- The conditional expression `"even" if a % 2 == 0 else "odd"` is the same in
  both; only the branch chosen is computed.
- Python chains comparisons (`0 < a < 20`); Viv writes `0 < a and a < 20`.
- Python keeps ints and floats apart when it prints: `17 // 5` is the int `3`
  and `17 / 5` is the float `3.4`. Viv's `print` writes any number with six
  significant digits and drops a trailing `.0`, so the whole numbers match
  Python's ints, and the Viv side wraps the fractions in `fmt(...)` to get
  Python's full digits.
