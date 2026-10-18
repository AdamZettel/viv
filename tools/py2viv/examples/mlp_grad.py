# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""A two-layer NumPy MLP's loss and its gradient with respect to the first weight matrix,
by Viv's reverse mode (value_and_grad).

Under CPython, `from viv import value_and_grad` is tools/py2viv/pyshim/viv.py, which
differentiates by the complex step; grad_preview.py also compares with torch in float64.
# py2viv: rtol=1e-13 atol=1e-16
"""
import numpy as np
from viv import value_and_grad


def loss(W1, X, Y, W2):
    H = np.tanh(X @ W1)
    P = H @ W2
    d = P - Y
    return np.sum(d * d) / d.size


N = 16
D = 4
K = 8
O = 2
X = np.zeros((N, D))
Y = np.zeros((N, O))
W1 = np.zeros((D, K))
W2 = np.zeros((K, O))
for i in range(N):
    for j in range(D):
        X[i, j] = np.sin(0.3 * i + 0.7 * j)
    for j in range(O):
        Y[i, j] = np.cos(0.5 * i - 0.2 * j)
for i in range(D):
    for j in range(K):
        W1[i, j] = 0.5 * np.sin(1.3 * i + 0.4 * j + 0.1)
for i in range(K):
    for j in range(O):
        W2[i, j] = 0.5 * np.cos(0.9 * i + 1.1 * j)

l, g1 = value_and_grad(loss)(W1, X, Y, W2)
print("loss", l)
print("dloss/dW1", g1)
l2, g2 = value_and_grad(loss, argnums=3)(W1, X, Y, W2)
print("dloss/dW2", g2)
