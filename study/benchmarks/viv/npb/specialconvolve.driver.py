# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
#from: http://stackoverflow.com/questions/2196693/improving-numpy-performance
#pythran export specialconvolve(uint32 [][])
#setup: import numpy as np ; r = np.arange(100*10000, dtype=np.uint32).reshape(1000,1000)
#run: specialconvolve(r)

def specialconvolve(a):
    # sorry, you must pad the input yourself
    rowconvol = a[1:-1,:] + a[:-2,:] + a[2:,:]
    colconvol = rowconvol[:,1:-1] + rowconvol[:,:-2] + rowconvol[:,2:] - 9*a[1:-1,1:-1]
    return colconvol


# ---- harness ----
import numpy as np

_a0 = np.array([int(t) for t in open('in0.txt').read().split()]).reshape(1000, 1000)

reps = int(open('reps.txt').read())
res = specialconvolve(_a0)
print(np.sum(res))
print(np.sum(res * res))
_chk = 0.0
for _i in range(reps):
    res = specialconvolve(_a0)
    _chk = _chk + res[0, 0]
print(_chk)
