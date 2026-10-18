# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ recursive
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# @@ with a loop
def factorial_loop(n):
    p = 1
    for k in range(2, n + 1):
        p *= k
    return p

# @@ a table
LAST = 20
STRIDE = 5
for n in range(0, LAST + 1, STRIDE):
    print(f"{n:2d}! = {factorial(n):d}")

# @@ the number of digits of 20!
print(len(str(factorial(LAST))), "digits")

# @@ checks
for n in range(LAST + 1):
    assert factorial(n) == factorial_loop(n)
