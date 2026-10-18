# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ imports
import math

# @@ variables and the four operations
a = 17
b = 5
print(a + b, a - b, a * b)
print(a / b)

# @@ floor division, remainder, power
print(a // b, a % b, -a // b, -a % b)
print(2 ** 10, 2 ** -1, 2 ** 3 ** 2)

# @@ math functions
print(f"{math.sqrt(2.0):.6f} {math.exp(1.0):.6f} {math.pi:.6f}")
print(abs(-3.5), max(a, b), min(a, b))
print(math.floor(-3.5), math.ceil(-3.5))

# @@ rounding a tie
print("round(2.5) =", round(2.5))

# @@ conditionals
def sign_word(x):
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"

for x in [3, -2, 0]:
    print(x, sign_word(x))

# @@ a value chosen by a condition
parity = "even" if a % 2 == 0 else "odd"
print(a, "is", parity)
print("comparisons:", 0 < a < 20, a > 0 and b > 10, not a == b)

# @@ checks
assert a == (a // b) * b + a % b
assert math.isclose(math.sqrt(2.0) ** 2, 2.0)
