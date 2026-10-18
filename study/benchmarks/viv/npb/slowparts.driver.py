# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
#from: https://groups.google.com/forum/#!topic/parakeet-python/p-flp2kdE4U
#setup: import numpy as np ;d = 10 ;re = 5 ;params = (d, re, np.ones((2*d, d+1, re)), np.ones((d, d+1, re)),  np.ones((d, 2*d)), np.ones((d, 2*d)), np.ones((d+1, re, d)), np.ones((d+1, re, d)), 1)
#run: slowparts(*params)

#pythran export slowparts(int, int, float [][][], float [][][], float [][], float [][], float [][][], float [][][], int)
from numpy import zeros, power, tanh
def slowparts(d, re, preDz, preWz, SRW, RSW, yxV, xyU, resid):
    """ computes the linear algebra intensive part of the gradients of the grae
    """
    fprime = lambda x: 1 - power(tanh(x), 2)

    partialDU = zeros((d+1, re, 2*d, d))
    for k in range(2*d):
        for i in range(d):
            partialDU[:,:,k,i] = fprime(preDz[k]) * fprime(preWz[i]) * (SRW[i,k] + RSW[i,k]) * yxV[:,:,i]
    
    return partialDU


# ---- harness ----
import numpy as np

_s0 = int(open('s0.txt').read())
_s1 = int(open('s1.txt').read())
_a2 = np.array([float(t) for t in open('in2.txt').read().split()]).reshape(20, 11, 5)
_a3 = np.array([float(t) for t in open('in3.txt').read().split()]).reshape(10, 11, 5)
_a4 = np.array([float(t) for t in open('in4.txt').read().split()]).reshape(10, 20)
_a5 = np.array([float(t) for t in open('in5.txt').read().split()]).reshape(10, 20)
_a6 = np.array([float(t) for t in open('in6.txt').read().split()]).reshape(11, 5, 10)
_a7 = np.array([float(t) for t in open('in7.txt').read().split()]).reshape(11, 5, 10)
_s8 = int(open('s8.txt').read())

reps = int(open('reps.txt').read())
res = slowparts(_s0, _s1, _a2, _a3, _a4, _a5, _a6, _a7, _s8)
print(np.sum(res))
print(np.sum(res * res))
_chk = 0.0
for _i in range(reps):
    res = slowparts(_s0, _s1, _a2, _a3, _a4, _a5, _a6, _a7, _s8)
    _chk = _chk + res[0, 0, 0, 0]
print(_chk)
