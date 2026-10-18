# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ return several values
def divmod_floor(a, b):
    q = a // b
    return q, a - q * b

q, r = divmod_floor(17, 5)
print(q, r)

# @@ swap without a temporary
a, b = 1, 2
a, b = b, a
print(a, b)

# @@ Fibonacci by pairs
def fib_pair(n):
    prev, cur = 0, 1
    for _ in range(n):
        prev, cur = cur, prev + cur
    return prev, cur

f10, f11 = fib_pair(10)
print(f10, f11)

# @@ statistics in one pass
def mean_and_spread(xs):
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return mean, var ** 0.5, n

mean, sd, n = mean_and_spread([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
print(f"n = {n}, mean = {mean}, sd = {sd}")

# @@ unpack a list of known length
point = [0.5, -1.5, 2.0]
x, y, z = point
print(x + y + z)

# @@ swap two elements of a list
v = [10, 20, 30]
v[0], v[2] = v[2], v[0]
print(v)

# @@ checks
assert divmod_floor(-7, 2) == (-4, 1)
assert (f10, f11) == (55, 89)
