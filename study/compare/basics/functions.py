# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ a function with a default argument
def greet(name, greeting="hello"):
    """A greeting for one person."""
    return f"{greeting}, {name}!"

print(greet("Ada"))
print(greet("Grace", "hi"))
print(greet("Alan", greeting="good morning"))

# @@ several results
def min_max_mean(xs):
    lo = min(xs)
    hi = max(xs)
    return lo, hi, sum(xs) / len(xs)

lo, hi, mean = min_max_mean([4.0, 1.0, 7.0, 2.0])
print(lo, hi, mean)

# @@ a function as an argument
def apply_twice(f, x):
    return f(f(x))

def square(x):
    return x * x

print(apply_twice(square, 3.0))
print(apply_twice(lambda x: x + 10.0, 1.0))

# @@ a lambda that captures a variable
scale = 2.5
scaled = lambda x: scale * x
print(scaled(4.0))

# @@ checks
assert greet("x") == "hello, x!"
assert apply_twice(square, 2.0) == 16.0
