# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ a count from zero
N = 5
for i in range(N):
    print(i, end=" ")
print()

# @@ a start, an end and a step
for i in range(3, 20, 4):
    print(i, end=" ")
print()
for i in range(10, 0, -3):
    print(i, end=" ")
print()

# @@ reversed
for i in reversed(range(N)):
    print(i, end=" ")
print()

# @@ over the items of a list, with their index
primes = [2, 3, 5, 7, 11]
for k, p in enumerate(primes):
    print(f"primes[{k}] = {p}")

# @@ two lists side by side
squares = [p * p for p in primes]
for p, s in zip(primes, squares):
    print(p, s)

# @@ a while loop
n = 27
steps = 0
while n != 1:
    n = n // 2 if n % 2 == 0 else 3 * n + 1
    steps += 1
print("Collatz steps from 27:", steps)

# @@ break and continue
LIMIT = 30
found = []
for k in range(2, LIMIT):
    if k % 2 == 0:
        continue
    if k * k > LIMIT:
        break
    found.append(k)
print("odd k with k*k <= 30:", found)

# @@ nested loops
TABLE = 4
for r in range(1, TABLE):
    print(" ".join(f"{r * c:3d}" for c in range(1, TABLE)))

# @@ checks
assert steps == 111
assert found == [3, 5]
