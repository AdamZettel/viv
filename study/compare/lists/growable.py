# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ start empty and append
squares = []
N = 10
for i in range(N):
    squares.append(i * i)
print(squares)

# @@ a comprehension with a condition
evens = [s for s in squares if s % 2 == 0]
print(evens, len(evens))

# @@ a comprehension that maps
halves = [s / 2 for s in squares[:4]]
print(halves)

# @@ extend, and drop the last element
more = [100, 121]
squares.extend(more)
print(len(squares), squares[-1])
squares = squares[:-1]
print(len(squares), squares[-1])

# @@ membership and sums
for v in [49, 50]:
    print(v, "is in the list" if v in squares else "is not in the list")
print(sum(squares), sum(squares) / len(squares))

# @@ rows onto a table
table = []
for i in range(1, 4):
    table.append([i, i * i, i ** 3])
for row in table:
    print(row)

# @@ checks
assert evens == [0, 4, 16, 36, 64]
assert len(table) == 3 and len(table[0]) == 3
