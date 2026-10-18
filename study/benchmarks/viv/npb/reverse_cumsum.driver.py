# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
#from: http://stackoverflow.com/questions/16541618/perform-a-reverse-cumulative-sum-on-a-numpy-array
#pythran export reverse_cumsum(float[])
#setup: import numpy as np ; np.random.seed(0); r = np.random.rand(1000000)
#run: reverse_cumsum(r)
import numpy as np
def reverse_cumsum(x):
    return np.cumsum(x[::-1])[::-1]


# ---- harness ----
import numpy as np

_a0 = np.array([float(t) for t in open('in0.txt').read().split()])

reps = int(open('reps.txt').read())
res = reverse_cumsum(_a0)
print(np.sum(res))
print(np.sum(res * res))
_chk = 0.0
for _i in range(reps):
    res = reverse_cumsum(_a0)
    _chk = _chk + res[0]
print(_chk)
