# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
#from: https://stackoverflow.com/questions/55854611/efficient-way-of-vectorizing-distance-calculation/55877642#55877642
#setup: import numpy as np ; np.random.seed(0); N = 80; x = np.random.rand(N,N); y = np.random.rand(N,N)
#run: l1norm(x, y)

#pythran export l1norm(float64[][], float64[:,:])
import numpy as np
def l1norm(x, y):
    return np.sum(np.abs(x[:, None, :] - y), axis=-1)


# ---- harness ----
import numpy as np

_a0 = np.array([float(t) for t in open('in0.txt').read().split()]).reshape(80, 80)
_a1 = np.array([float(t) for t in open('in1.txt').read().split()]).reshape(80, 80)

reps = int(open('reps.txt').read())
res = l1norm(_a0, _a1)
print(np.sum(res))
print(np.sum(res * res))
_chk = 0.0
for _i in range(reps):
    res = l1norm(_a0, _a1)
    _chk = _chk + res[0, 0]
print(_chk)
