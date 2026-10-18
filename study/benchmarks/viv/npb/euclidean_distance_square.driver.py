# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
#from:  https://stackoverflow.com/questions/50658884/why-this-numba-code-is-6x-slower-than-numpy-code
#setup: import numpy as np; np.random.seed(0); x1 = np.random.random((1, 512)); x2 = np.random.random((10000, 512))
#run: euclidean_distance_square(x1, x2)
#pythran export euclidean_distance_square(float64[1,:], float64[:,:])
import numpy as np
def euclidean_distance_square(x1, x2):
    return -2*np.dot(x1, x2.T) + np.sum(np.square(x1), axis=1)[:, np.newaxis] + np.sum(np.square(x2), axis=1)


# ---- harness ----
import numpy as np

_a0 = np.array([float(t) for t in open('in0.txt').read().split()]).reshape(1, 512)
_a1 = np.array([float(t) for t in open('in1.txt').read().split()]).reshape(10000, 512)

reps = int(open('reps.txt').read())
res = euclidean_distance_square(_a0, _a1)
print(np.sum(res))
print(np.sum(res * res))
_chk = 0.0
for _i in range(reps):
    res = euclidean_distance_square(_a0, _a1)
    _chk = _chk + res[0, 0]
print(_chk)
