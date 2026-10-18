# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
#from: http://stackoverflow.com/questions/17112550/python-and-numba-for-vectorized-functions
#setup: import numpy as np ; np.random.seed(0); N = 100000 ; a, b, c = np.random.rand(N), np.random.rand(N), np.random.rand(N)
#run: vibr_energy(a, b, c)

#pythran export vibr_energy(float64[], float64[], float64[])
import numpy
def vibr_energy(harmonic, anharmonic, i):
    return numpy.exp(-harmonic * i - anharmonic * (i ** 2))


# ---- harness ----
import numpy as np

_a0 = np.array([float(t) for t in open('in0.txt').read().split()])
_a1 = np.array([float(t) for t in open('in1.txt').read().split()])
_a2 = np.array([float(t) for t in open('in2.txt').read().split()])

reps = int(open('reps.txt').read())
res = vibr_energy(_a0, _a1, _a2)
print(np.sum(res))
print(np.sum(res * res))
_chk = 0.0
for _i in range(reps):
    res = vibr_energy(_a0, _a1, _a2)
    _chk = _chk + res[0]
print(_chk)
