# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
#from: http://stackoverflow.com/questions/7741878/how-to-apply-numpy-linalg-norm-to-each-row-of-a-matrix/7741976#7741976
#setup: import numpy as np ; np.random.seed(0); N = 1000; x = np.random.rand(N,N)
#run: l2norm(x)

#pythran export l2norm(float64[][])
import numpy as np
def l2norm(x):
    return np.sqrt(np.sum(np.abs(x)**2, 1))


# ---- harness ----
import numpy as np

_a0 = np.array([float(t) for t in open('in0.txt').read().split()]).reshape(1000, 1000)

reps = int(open('reps.txt').read())
res = l2norm(_a0)
print(np.sum(res))
print(np.sum(res * res))
_chk = 0.0
for _i in range(reps):
    res = l2norm(_a0)
    _chk = _chk + res[0]
print(_chk)
