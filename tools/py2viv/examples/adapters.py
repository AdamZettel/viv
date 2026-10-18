# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The NumPy calls whose Viv builtins have another calling convention, each checked against
NumPy. frontends.md found the 2026-09-24 prototype silently wrong on the first three: np.sort
returned 0 and sorted its argument, np.fft.fft returned 0, np.linalg.solve failed with an
arity error.

Viv's FFT, LU, Cholesky and Jacobi eigensolver are not pocketfft's or LAPACK's, so these agree
to rounding, not to the bit.
# py2viv: rtol=1e-12 atol=1e-13
"""
import numpy as np

# np.sort returns a sorted copy and leaves its argument alone
a = np.array([0.5, -1.0, 2.0, 0.25])
b = np.sort(a)
print("sorted", b)
print("original untouched", a)
print("dot(a, sort(a))", np.dot(a, b))
s = sorted(a)
print("sorted() of an array", s[0], s[3])

# FFT of a real signal, and back
n = 8
x = np.zeros(n)
for k in range(n):
    x[k] = np.sin(2.0 * np.pi * k / n) + 0.5 * np.cos(4.0 * np.pi * k / n) + 0.25
X = np.fft.fft(x)
print("fft real parts", X.real)
print("fft imaginary parts", X.imag)
print("|X|", np.abs(X))
y = np.fft.ifft(X)
print("ifft(fft(x)) real", y.real)

# solve, cholesky, inv
A = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0]])
rhs = np.array([1.0, 2.0, 3.0])
sol = np.linalg.solve(A, rhs)
print("solve", sol)
print("residual", np.max(np.abs(A @ sol - rhs)))
L = np.linalg.cholesky(A)
print("cholesky", L)
Ainv = np.linalg.inv(A)
print("inv", Ainv)

# np.linalg.eigh is refused: each eigenvector's sign (and a degenerate pair's basis) is the
# solver's choice, so Viv's Jacobi and LAPACK can disagree on inputs a program cannot see.
print("eigvalsh", np.linalg.eigvalsh(A))


def test_sort_copies():
    c = np.array([3.0, 1.0, 2.0])
    d = np.sort(c)
    assert d[0] == 1.0 and d[2] == 3.0
    assert c[0] == 3.0


def test_solve_residual():
    M = np.array([[2.0, 1.0], [1.0, 3.0]])
    v = np.array([1.0, 1.0])
    u = np.linalg.solve(M, v)
    assert abs(2.0 * u[0] + u[1] - 1.0) < 1e-14
    assert abs(u[0] + 3.0 * u[1] - 1.0) < 1e-14


def test_ifft_inverts_fft():
    z = np.zeros(5)
    for k in range(5):
        z[k] = 1.0 + k * k
    back = np.fft.ifft(np.fft.fft(z))
    assert np.allclose(back.real, z, rtol=0.0, atol=1e-13)
