# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ imports
import numpy as np

# @@ slices of an array
N = 10
STEP = 3
a = np.arange(float(N))
print(a[2:5].tolist(), a[:3].tolist(), a[7:].tolist(), a[::STEP].tolist())

# @@ the last elements
print(a[-1], a[-3:].tolist())

# @@ a slice is a view in NumPy and a copy in Viv
d = np.zeros(4)
part = d[1:3]
part[0] = 99.0
print("d[1] after a write through the slice:", d[1])

# @@ assignment shares the array
c = a
c[0] = -1.0
print(a[0])

# @@ write into a slice
a[5:8] = 0.0
print(a.tolist())

# @@ where is the largest value
x = np.array([3.0, 8.5, -2.0, 8.5, 1.0])
k = int(np.argmax(x))
print(k, x[k])

# @@ running sums
running = np.cumsum(x)
print(running.tolist())

# @@ checks
assert a[0] == -1.0
assert running[-1] == np.sum(x)
