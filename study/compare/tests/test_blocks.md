---
title: Tests
order: 1
output: identical
keep_pass: yes
---
- A Viv `test "name" { ... }` block is a test function and its runner in one:
  it runs where it stands, and prints `PASS name`, or `FAIL name` with the first
  failing assertion's detail. A failure makes the process exit nonzero.
- The Python side writes the runner itself, in a few lines, so the two print
  the same verdicts. pytest or unittest would find the `test_` functions and
  print their own report instead.
- A test name in Viv is a string with spaces; in Python it is a function name.
- `assert cond` is `assert_true(cond)`, and `math.isclose(a, b, abs_tol=t)` is
  `assert_near(a, b, t)`. `assert_near`, `assert_lt`, `assert_gt` and
  `assert_allclose` print the numbers they compared when they fail.
- `all(x > 0 for x in w)` works in Viv too; `all(w > 0)` is the array form: the
  comparison makes an array of 1.0 and 0.0 and `all` reads it.
- `normalize` is one array expression in Viv, `xs / sum(xs)`, where the Python
  list needs a comprehension.
- A top-level `assert_true` outside any test prints its own line; the pairs on
  this site put their checks in test blocks.
