# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ imports
import numpy as np

# @@ data: points near the line y = 1 + 2 x
X = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [1.0, 1.5], [1.0, 2.0]])
y = np.array([1.1, 1.9, 3.2, 3.9, 5.1])

# @@ the loss
def loss(w, X, y):
    r = X @ w - y
    return np.sum(r * r) / len(y)

# @@ its gradient
def loss_grad(w, X, y):
    r = X @ w - y
    return 2.0 * (X.T @ r) / len(y)

# @@ check the gradient at one point
w0 = np.array([0.5, 0.5])
H = 1e-6
fd = np.array([(loss(w0 + H * e, X, y) - loss(w0 - H * e, X, y)) / (2 * H) for e in np.eye(2)])
print("gradient   ", " ".join(f"{g:.6f}" for g in loss_grad(w0, X, y)))
print("differences", " ".join(f"{g:.6f}" for g in fd))

# @@ gradient descent
RATE = 0.1
STEPS = 500
REPORT = 100
w = w0.copy()
for step in range(STEPS):
    w = w - RATE * loss_grad(w, X, y)
    if (step + 1) % REPORT == 0:
        print(f"step {step + 1}: loss {loss(w, X, y):.8f}")
print(f"w = {w[0]:.6f}, {w[1]:.6f}")

# @@ a derivative of one variable
def f(x):
    return np.sin(x) * x * x

def df(x):
    return np.cos(x) * x * x + 2.0 * x * np.sin(x)

print(f"f'(1.3) = {df(1.3):.12f}")

# @@ checks
assert np.allclose(loss_grad(w0, X, y), fd, atol=1e-6)
