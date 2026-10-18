# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ count the sums of two dice
SIDES = 6
sums = {}
for a in range(1, SIDES + 1):
    for b in range(1, SIDES + 1):
        sums[a + b] = sums.get(a + b, 0) + 1
print(sums)

# @@ a bar per key, in key order
for total in sorted(sums):
    print(f"{total:2d} {'#' * sums[total]}")

# @@ the most likely sum
best = max(sums, key=lambda k: sums[k])
print("most likely:", best, "with", sums[best], "of", SIDES * SIDES)

# @@ first occurrence of each value
data = [4, 7, 4, 1, 7, 7, 9]
first_at = {}
for i, x in enumerate(data):
    if x not in first_at:
        first_at[x] = i
print(first_at)

# @@ a copy is a second map
backup = dict(first_at)
first_at[100] = 0
print(len(first_at), len(backup))

# @@ checks
assert sum(sums.values()) == SIDES * SIDES
assert first_at[7] == 1
