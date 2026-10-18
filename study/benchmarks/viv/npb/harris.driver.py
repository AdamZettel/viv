# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
#from: parakeet testbed
#setup: import numpy as np ; np.random.seed(0); M, N = 512, 512 ; I = np.random.randn(M,N)
#run: harris(I)

#pythran export harris(float64[][])
import numpy as np



def harris(I):
  m,n = I.shape
  dx = (I[1:, :] - I[:m-1, :])[:, 1:]
  dy = (I[:, 1:] - I[:, :n-1])[1:, :]

  #
  #   At each point we build a matrix
  #   of derivative products
  #   M =
  #   | A = dx^2     C = dx * dy |
  #   | C = dy * dx  B = dy * dy |
  #
  #   and the score at that point is:
  #      det(M) - k*trace(M)^2
  #
  A = dx * dx
  B = dy * dy
  C = dx * dy
  tr = A + B
  det = A * B - C * C
  k = 0.05
  return det - k * tr * tr


# ---- harness ----
import numpy as np

_a0 = np.array([float(t) for t in open('in0.txt').read().split()]).reshape(512, 512)

reps = int(open('reps.txt').read())
res = harris(_a0)
print(np.sum(res))
print(np.sum(res * res))
_chk = 0.0
for _i in range(reps):
    res = harris(_a0)
    _chk = _chk + res[0, 0]
print(_chk)
