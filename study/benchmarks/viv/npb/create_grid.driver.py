# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
#from: http://stackoverflow.com/questions/13815719/creating-grid-with-numpy-performance
#pythran export create_grid(float [])
#setup: import numpy as np ; N = 800 ; x = np.arange(0,1,1./N)
#run: create_grid(x)
import numpy as np
def create_grid(x):
    N = x.shape[0]
    z = np.zeros((N, N, 3))
    z[:,:,0] = x.reshape(-1,1)
    z[:,:,1] = x
    fast_grid = z.reshape(N*N, 3)
    return fast_grid


# ---- harness ----
import numpy as np

_a0 = np.array([float(t) for t in open('in0.txt').read().split()])

reps = int(open('reps.txt').read())
res = create_grid(_a0)
print(np.sum(res))
print(np.sum(res * res))
_chk = 0.0
for _i in range(reps):
    res = create_grid(_a0)
    _chk = _chk + res[0, 0]
print(_chk)
