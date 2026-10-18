# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ plain recursion
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

# @@ remembered in a dict
memo = {}

def fib_memo(n):
    if n < 2:
        return n
    if n in memo:
        return memo[n]
    v = fib_memo(n - 1) + fib_memo(n - 2)
    memo[n] = v
    return v

# @@ count the calls
calls = 0

def fib_counted(n):
    global calls
    calls += 1
    if n < 2:
        return n
    return fib_counted(n - 1) + fib_counted(n - 2)

SMALL = 20
print("fib(20) =", fib_counted(SMALL), "after", calls, "calls")

# @@ large n with the memo
BIG = 78
print(f"fib(78) = {fib_memo(BIG):d}, memo holds {len(memo)} values")

# @@ checks
assert fib(SMALL) == fib_memo(SMALL)
