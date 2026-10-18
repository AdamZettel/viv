# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ merge two sorted lists
def merge(a, b):
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out

# @@ split, sort each half, merge
def merge_sort(xs):
    if len(xs) <= 1:
        return xs
    mid = len(xs) // 2
    return merge(merge_sort(xs[:mid]), merge_sort(xs[mid:]))

# @@ run it
data = [38, 27, 43, 3, 9, 82, 10, 3, 55, -4]
print(merge_sort(data))
print(data)

# @@ checks
assert merge_sort(data) == sorted(data)
assert merge_sort([]) == []
