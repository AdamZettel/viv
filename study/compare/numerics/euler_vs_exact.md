---
title: Euler against the exact solution
order: 5
output: identical
---
- `errors[-1]` counts from the end in both; inside the loop the Viv side reads
  `errors[k]`.
- `errors.append(e)` is `errors = append(errors, e)`.
- `N0 * 2 ** k` is `N0 * 2 ^ k`, and the step count is a double in Viv; the
  `:4d` format prints it as a whole number.
- The parameter `f` of `euler` and the global `def f` have the same name in
  both programs; inside `euler` the parameter is the one called.
- `math.exp` is the builtin `exp`. Viv computes it with its own libm, the same
  in every engine; the printed digits agree with the platform's here.
