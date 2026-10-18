# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ a sorted copy
nums = [5.5, -1.0, 3.25, 9.0, 0.0, 3.25]
ordered = sorted(nums)
print(ordered)
print(nums)

# @@ sort in place, largest first
nums.sort(reverse=True)
print(nums)

# @@ sort strings
words = ["pear", "Apple", "fig", "banana"]
print(sorted(words))

# @@ sort one list by another
names = ["ann", "bob", "cy", "dee"]
scores = [72, 95, 88, 72]
order = sorted(range(len(names)), key=lambda i: (-scores[i], names[i]))
for i in order:
    print(f"{names[i]:4} {scores[i]}")

# @@ insertion sort by hand
def insertion_sort(a):
    a = list(a)
    for i in range(1, len(a)):
        v = a[i]
        j = i
        while j > 0 and a[j - 1] > v:
            a[j] = a[j - 1]
            j -= 1
        a[j] = v
    return a

print(insertion_sort([3, 1, 4, 1, 5, 9, 2, 6]))

# @@ checks
assert insertion_sort(nums) == sorted(nums)
