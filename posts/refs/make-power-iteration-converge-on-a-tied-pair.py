# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-25-make-power-iteration-converge-on-a-tied-pair.viv (a problem).

The eigenvalues and unit eigenvectors of the two test matrices, from
numpy.linalg.eigh, each vector signed so that its first entry is positive,
beside the closed forms (+-2 with (1, +-1)/sqrt 2; +-sqrt 5). Also what plain
power iteration does on the first matrix: the iterate after an even and an odd
number of steps, which never settle.

    python3 posts/refs/make-power-iteration-converge-on-a-tied-pair.py > posts/refs/make-power-iteration-converge-on-a-tied-pair.out
"""
import numpy as np

for name, A in (("[[0, 2], [2, 0]]", np.array([[0.0, 2.0], [2.0, 0.0]])),
                ("[[1, 2], [2, -1]]", np.array([[1.0, 2.0], [2.0, -1.0]]))):
    w, V = np.linalg.eigh(A)
    print(f"A = {name}")
    for k in range(2):
        v = V[:, k] * np.sign(V[0, k])
        print(f"  eigenvalue {w[k]!r}, vector [{v[0]!r}, {v[1]!r}]")
print("closed forms: 1/sqrt 2 =", repr(1 / np.sqrt(2)), " sqrt 5 =", repr(np.sqrt(5.0)))
A = np.array([[0.0, 2.0], [2.0, 0.0]])
v = np.array([1.0, 0.3])
v = v / np.linalg.norm(v)
for k in range(1, 61):
    v = A @ v
    v = v / np.linalg.norm(v)
    if k in (59, 60):
        print(f"plain power iteration from (1, 0.3), step {k}: {v.tolist()}, Rayleigh quotient {v @ A @ v:.6f}")
