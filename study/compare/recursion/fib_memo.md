---
title: Fibonacci with a memo
order: 2
output: identical
---
- The memo is a dict in Python and a `map()` in Viv; `n in memo` is the same in
  both. A function stores into the global map without any declaration, because
  a map is a handle.
- Viv has no `global` statement, and a function cannot assign a global
  number: an assignment makes the name local for the whole body, and reading
  it first is refused with a note. The Viv side counts calls in a one-element
  array, `calls = [0]`, and a function writes into it with `calls[0] += 1`.
- `fib(78)` is 8944394323791464, below 2^53, so the double holds it exactly.
  Past `fib(78)` the Viv values would be rounded; Python's ints would not.
- `print(... fib_counted(SMALL), "after", calls ...)` in Python evaluates the
  call before reading `calls`. The Viv side stores the result first, which
  makes the order plain.
