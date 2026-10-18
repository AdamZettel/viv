# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
#from: http://people.duke.edu/~ccc14/sta-663-2016/03A_Numbers.html#Example:-Calculating-pairwise-distance-matrix-using-broadcasting-and-vectorization
#setup: import numpy as np ; X = np.linspace(0,10,20000).reshape(200,100)
#run: pairwise(X)

#pythran export pairwise(float [][])

import numpy as np
def pairwise(pts):
    return np.sum((pts[None,:] - pts[:, None])**2, -1)**0.5


# ---- harness ----
import numpy as np

_a0 = np.array([float(t) for t in open('in0.txt').read().split()]).reshape(200, 100)

reps = int(open('reps.txt').read())
res = pairwise(_a0)
print(np.sum(res))
print(np.sum(res * res))
_chk = 0.0
for _i in range(reps):
    res = pairwise(_a0)
    _chk = _chk + res[0, 0]
print(_chk)
