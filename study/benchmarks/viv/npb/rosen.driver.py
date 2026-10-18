# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
#setup: import numpy as np; r = np.arange(1000000, dtype=float)
#run: rosen(r)
import numpy as np

#pythran export rosen(float[])

def rosen(x):
    t0 = 100 * (x[1:] - x[:-1] ** 2) ** 2
    t1 = (1 - x[:-1]) ** 2
    return np.sum(t0 + t1)


# ---- harness ----
import numpy as np

_a0 = np.array([float(t) for t in open('in0.txt').read().split()])

reps = int(open('reps.txt').read())
res = rosen(_a0)
print(res)
_chk = 0.0
for _i in range(reps):
    res = rosen(_a0)
    _chk = _chk + res
print(_chk)
