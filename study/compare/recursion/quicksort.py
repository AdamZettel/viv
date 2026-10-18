# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ partition around the last element
def partition(a, lo, hi):
    pivot = a[hi]
    i = lo
    for j in range(lo, hi):
        if a[j] < pivot:
            a[i], a[j] = a[j], a[i]
            i += 1
    a[i], a[hi] = a[hi], a[i]
    return i

# @@ sort the two sides
def quicksort(a, lo, hi):
    if lo < hi:
        p = partition(a, lo, hi)
        quicksort(a, lo, p - 1)
        quicksort(a, p + 1, hi)

# @@ sort a list in place
data = [3.5, -1.0, 7.25, 0.0, 2.0, 7.25, -8.5, 4.0]
quicksort(data, 0, len(data) - 1)
print(data)

# @@ checks
assert all(data[k] <= data[k + 1] for k in range(len(data) - 1))
