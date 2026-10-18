#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The phrase cases: one test per phrase, with the value
NumPy (or the closed form) gives.

    python3 tools/phrases/cases.py          # rewrites lang/viv/tests/phrases/phrases.viv
    python3 tools/phrases/cases.py --check  # exits 1 if it would change

Each case is a phrase as a Viv program writes it and the Python expression
that computes the same thing, from the same inputs. The generated program has
one `test` per case: the phrase's value against the Python value, written out
to 17 digits, within `TOL * (1 + |value|)` (the largest |value| of an array). lang/viv/tests/phrases.rs checks
that every pattern of the table (lang/viv/src/phrases.rs) has a case and that
the program passes, with the same output, in the interpreter, as C and as
wasm; tools/three-engines.sh lang/viv/tests/phrases runs it the same way.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "imports"))
from viv_imports import migrated  # noqa: E402  (tools/imports/viv_imports.py)

import numpy as np

ROOT = __file__.rsplit("/tools/", 1)[0]
OUT = ROOT + "/lang/viv/tests/phrases/phrases.viv"

# the inputs, the same on both sides
INPUTS = {
    "v": [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0],
    "u": [2.0, 7.0, 1.0, 8.0, 2.5, 8.0, 1.0, 8.0],
    "w": [3.0, 1.0, 4.0, 1.5, 5.0],
    "M": [[4.0, 1.0, 2.0], [1.5, 5.0, 3.0], [2.0, 3.5, 6.0]],
    "B": [[1.0, 0.5, 0.0], [2.0, 1.0, 3.0], [0.0, 1.0, 1.0]],
    "x": 0.7,
    "c": 2.0,
    "n": 5.0,
    "q": [0.5, -1.0, 2.0],
    "tgt": [0.0, 2.0, 1.0],
}

# defs the gradient phrases differentiate
DEFS = """def ph_quad(x, A) {
    return sum(x * (A @ x))
}
"""


def softmax(a):
    e = np.exp(a - np.max(a, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def cross_entropy(L, y):
    L = np.asarray(L)
    top = np.max(L, axis=1)
    lse = np.log(np.sum(np.exp(L - top[:, None]), axis=1)) + top
    return np.mean(lse - L[np.arange(len(L)), np.asarray(y, dtype=int)])

# tolerance, relative to 1 + |value|: the lowerings add in NumPy's order or
# compute a closed form, so the differences are rounding
TOL = 1e-12


def simpson(f, a, b, n):
    if n % 2:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for k in range(int(n) - 1):
        s += (4.0 if (k + 1) % 2 == 1 else 2.0) * f(a + (k + 1) * h)
    return s * h / 3.0


def trapezoid(f, a, b, n):
    h = (b - a) / n
    s = (f(a) + f(b)) / 2.0
    for k in range(int(n) - 1):
        s += f(a + (k + 1) * h)
    return s * h


GAUSS = math.sqrt(math.pi) / 2.0 * math.erf(1.0)

# (phrase, Python value)
CASES = [
    # aggregates
    ("take the sum of v", "np.sum(v)"),
    ("take the product of v", "np.prod(v)"),
    ("take the mean of v", "np.mean(v)"),
    ("take the average of v", "np.mean(v)"),
    ("take the median of v", "np.median(v)"),
    ("take the median of w", "np.median(w)"),
    ("take the max of v", "np.max(v)"),
    ("take the maximum of v", "np.max(v)"),
    ("take the min of v", "np.min(v)"),
    ("take the minimum of v", "np.min(v)"),
    ("take the variance of v", "np.var(v)"),
    ("take the standard deviation of v", "np.std(v)"),
    ("take the cumulative sum of v", "np.cumsum(v)"),
    ("take the cumulative product of w", "np.cumprod(w)"),
    # matrices
    ("take the transpose of M", "M.T"),
    ("take the trace of M", "np.trace(M)"),
    ("take the diagonal of M", "np.diag(M)"),
    ("take the diagonal part of M", "np.diag(np.diag(M))"),
    ("take the off-diagonal part of M", "M - np.diag(np.diag(M))"),
    ("take the sum of the off-diagonal entries of M", "np.sum(M - np.diag(np.diag(M)))"),
    ("take the off-diagonal row sums of M", "np.sum(M - np.diag(np.diag(M)), axis=1)"),
    ("take the off-diagonal column sums of M", "np.sum(M - np.diag(np.diag(M)), axis=0)"),
    ("take the upper triangle of M", "np.triu(M)"),
    ("take the lower triangle of M", "np.tril(M)"),
    ("take the strict upper triangle of M", "np.triu(M, 1)"),
    ("take the strict lower triangle of M", "np.tril(M, -1)"),
    ("take the row sums of M", "np.sum(M, axis=1)"),
    ("take the column sums of M", "np.sum(M, axis=0)"),
    ("take the sum of each row of M", "np.sum(M, axis=1)"),
    ("take the mean of each row of M", "np.mean(M, axis=1)"),
    ("take the sum of each column of M", "np.sum(M, axis=0)"),
    ("take the mean of each column of M", "np.mean(M, axis=0)"),
    ("take the max of each row of M", "np.max(M, axis=1)"),
    ("take the maximum of each column of M", "np.max(M, axis=0)"),
    ("take the min of each row of M", "np.min(M, axis=1)"),
    ("take the minimum of each column of M", "np.min(M, axis=0)"),
    ("take the inverse of M", "np.linalg.inv(M)"),
    ("sum the off-diagonal entries of M", "np.sum(M - np.diag(np.diag(M)))"),
    ("sum the off-diagonal entries of each row of M", "np.sum(M - np.diag(np.diag(M)), axis=1)"),
    ("sum the off-diagonal entries of each column of M", "np.sum(M - np.diag(np.diag(M)), axis=0)"),
    ("sum each row of M", "np.sum(M, axis=1)"),
    ("sum each column of M", "np.sum(M, axis=0)"),
    # vectors
    ("take the L2 norm of v", "np.linalg.norm(v)"),
    ("take the Frobenius norm of M", "np.linalg.norm(M)"),
    ("take the norm of v", "np.linalg.norm(v)"),
    ("take the magnitude of v", "np.linalg.norm(v)"),
    ("take the dot product of u and v", "np.dot(u, v)"),
    ("take the inner product of u and v", "np.inner(u, v)"),
    ("take the outer product of u and w", "np.outer(u, w)"),
    # sizes and elements
    ("take the size of v", "np.size(v)"),
    ("take the length of v", "len(v)"),
    ("take the number of rows of B", "B.shape[0]"),
    ("take the number of columns of B", "B.shape[1]"),
    ("take the number of elements of M", "M.size"),
    ("take the shape of M", "np.array(M.shape, dtype=float)"),
    ("take the first element of v", "v[0]"),
    ("take the last element of v", "v[-1]"),
    ("take element 2 of v", "v[2]"),
    ("take row 1 of M", "M[1, :]"),
    ("take column 2 of M", "M[:, 2]"),
    ("take v sorted", "np.sort(v)"),
    ("take the reverse of v", "v[::-1]"),
    # elementary functions
    ("take the sine of x", "math.sin(x)"),
    ("take the cosine of x", "math.cos(x)"),
    ("take the tangent of x", "math.tan(x)"),
    ("take the arcsine of x", "math.asin(x)"),
    ("take the arccosine of x", "math.acos(x)"),
    ("take the arctangent of x", "math.atan(x)"),
    ("take the exponential of x", "math.exp(x)"),
    ("take the natural log of x", "math.log(x)"),
    ("take the logarithm of c", "math.log(c)"),
    ("take the square root of c", "math.sqrt(c)"),
    ("take the absolute value of -x", "abs(-x)"),
    ("take the floor of 2.5", "math.floor(2.5)"),
    ("take the ceiling of 2.5", "math.ceil(2.5)"),
    ("take the nearest integer to 2.7", "round(2.7)"),
    ("take the sine of v", "np.sin(v)"),
    # geometry and linear algebra
    ("take the matrix product of M and B", "M @ B"),
    ("take the projection of u onto v", "np.dot(u, v) / np.dot(v, v) * v"),
    ("take the component of u along v", "np.dot(u, v) / np.dot(v, v) * v"),
    ("take the unit vector along v", "v / np.linalg.norm(v)"),
    ("take the unit vector in the direction of v", "v / np.linalg.norm(v)"),
    ("take the angle between u and v", "math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))"),
    ("take the distance from u to v", "np.linalg.norm(u - v)"),
    ("take the distance between u and v", "np.linalg.norm(u - v)"),
    ("take the solution of M @ y = w[0:3]", "np.linalg.solve(M, w[0:3])"),
    ("solve M @ y = w[0:3]", "np.linalg.solve(M, w[0:3])"),
    ("take the residual of M @ u[0:3] = w[0:3]", "M @ u[0:3] - w[0:3]"),
    ("take the residual norm of M @ u[0:3] = w[0:3]", "np.linalg.norm(M @ u[0:3] - w[0:3])"),
    ("take the largest residual of M @ u[0:3] = w[0:3]", "np.max(np.abs(M @ u[0:3] - w[0:3]))"),
    ("take the largest difference between u and v", "np.max(np.abs(u - v))"),
    # construction
    ("take a vector of zeros of length n", "np.zeros(int(n))"),
    ("take a vector of ones of length 3", "np.ones(3)"),
    ("take a matrix of zeros with 2 rows and 3 columns", "np.zeros((2, 3))"),
    ("take the identity matrix of size 3", "np.eye(3)"),
    ("take 5 evenly spaced points from 0 to 1", "np.linspace(0, 1, 5)"),
    ("take the range from 2 to 6", "np.arange(2.0, 6.0)"),
    # integrals, roots and derivatives
    ("take the integral of s^2 from 1 to 2", "7.0 / 3.0"),
    ("take the integral of exp(-t * t) from 0 to 1", "GAUSS"),
    ("take the integral of exp(-t * t) from 0 to 1 with 60 intervals", "simpson(lambda t: math.exp(-t * t), 0.0, 1.0, 60)"),
    ("take the integral of exp(-t * t) from 0 to 1 using every method", "GAUSS"),
    ("take the integral of exp(-t * t) from 0 to 1 using simpson with 60 intervals", "simpson(lambda t: math.exp(-t * t), 0.0, 1.0, 60)"),
    ("take the integral of exp(-t * t) from 0 to 1 using the trapezoid rule with 60 intervals", "trapezoid(lambda t: math.exp(-t * t), 0.0, 1.0, 60)"),
    ("take the integral of exp(-t * t) from 0 to 1 using gauss-legendre", "GAUSS"),
    ("take the integral of exp(-t * t) from 0 to 1 using adaptive simpson", "GAUSS"),
    ("take the integral of exp(-t * t) from 0 to 1 using romberg", "GAUSS"),
    ("take the root of z^2 - c starting at 1", "math.sqrt(2.0)"),
    ("take the root of z^2 = c starting at 1 using newton", "math.sqrt(2.0)"),
    ("take the solution of z^2 = c using newton starting at 1", "math.sqrt(2.0)"),
    ("take the root of z^2 - c between 0 and 2", "math.sqrt(2.0)"),
    ("take the root of z^2 - c between 0 and 2 using every method", "math.sqrt(2.0)"),
    ("take the root of z^2 - c between 0 and 2 using bisection", "math.sqrt(2.0)"),
    ("take the solution of cos(z) = z between 0 and 1 using illinois", "0.7390851332151607"),
    ("take the root of cos(z) - z between 0 and 1 using regula falsi", "0.7390851332151607"),
    ("take the root of cos(z) - z between 0 and 1 using false position", "0.7390851332151607"),
    ("take the root of cos(z) - z between 0 and 1 using safeguarded secant", "0.7390851332151607"),
    ("take the derivative of sin(z) * z at x", "math.cos(x) * x + math.sin(x)"),
    # without `take`
    ("integrate t^3 from 0 to 2", "4.0"),
    ("integrate t^3 from 0 to 2 using every method", "4.0"),
    ("integrate exp(-t * t) from 0 to 1 with 60 intervals", "simpson(lambda t: math.exp(-t * t), 0.0, 1.0, 60)"),
    ("integrate exp(-t * t) from 0 to 1 using simpson with 60 intervals", "simpson(lambda t: math.exp(-t * t), 0.0, 1.0, 60)"),
    ("integrate exp(-t * t) from 0 to 1 using the trapezoid rule with 60 intervals", "trapezoid(lambda t: math.exp(-t * t), 0.0, 1.0, 60)"),
    ("integrate exp(-t * t) from 0 to 1 using gauss-legendre", "GAUSS"),
    ("integrate exp(-t * t) from 0 to 1 using adaptive simpson", "GAUSS"),
    ("integrate exp(-t * t) from 0 to 1 using romberg", "GAUSS"),
    ("solve z^3 = c starting at 1", "2.0 ** (1.0 / 3.0)"),
    ("solve z^3 = c starting at 1 using newton", "2.0 ** (1.0 / 3.0)"),
    ("solve z^3 = c using newton starting at 1", "2.0 ** (1.0 / 3.0)"),
    ("solve z^3 = c between 0 and 2", "2.0 ** (1.0 / 3.0)"),
    ("solve z^3 = c between 0 and 2 using every method", "2.0 ** (1.0 / 3.0)"),
    ("solve z^3 = c between 0 and 2 using bisection", "2.0 ** (1.0 / 3.0)"),
    ("solve z^3 = c between 0 and 2 using illinois", "2.0 ** (1.0 / 3.0)"),
    ("solve z^3 = c between 0 and 2 using regula falsi", "2.0 ** (1.0 / 3.0)"),
    ("solve z^3 = c between 0 and 2 using false position", "2.0 ** (1.0 / 3.0)"),
    ("solve z^3 = c between 0 and 2 using safeguarded secant", "2.0 ** (1.0 / 3.0)"),
    ("differentiate exp(z) * z at x", "math.exp(x) * (1.0 + x)"),
    # sizes, probability and learning (the GPT rewrite)
    ("take the first row of M", "M[0]"),
    ("take the last row of M", "M[-1]"),
    ("take the last 3 elements of v", "v[-3:]"),
    ("take the last 20 elements of w", "w[-20:]"),
    ("take the softmax of v", "softmax(v)"),
    ("take the softmax of each row of M", "softmax(M)"),
    ("take the cross-entropy of M against tgt", "cross_entropy(M, tgt)"),
    ("take the causal mask of size n", "np.triu(np.ones((5, 5)), 1)"),
    ("take the gradient of ph_quad(q, M) with respect to q", "(M + M.T) @ q"),
    # a phrase with two values: the value, then the gradient
    ("take the value and gradient of ph_quad(q, M) with respect to q", ("q @ M @ q", "(M + M.T) @ q")),
    # the symbolic derivative, by lib/cas (the program uses it, below)
    ("take the derivative of atan(z) * z at x using the symbolic derivative", "math.atan(x) + x / (1.0 + x * x)"),
    ("differentiate erf(z) * c at x using the symbolic derivative", "c * 2.0 / math.sqrt(math.pi) * math.exp(-x * x)"),
]


def viv_num(x):
    r = repr(float(x))
    return r if ("e" in r or "." in r or "inf" in r or "nan" in r) else r + ".0"


def viv_lit(a):
    a = np.asarray(a, dtype=float)
    if a.ndim == 0:
        return viv_num(a)
    return "[" + ", ".join(viv_lit(x) for x in a) + "]"


def build():
    env = {"np": np, "math": math, "simpson": simpson, "trapezoid": trapezoid, "GAUSS": GAUSS, "softmax": softmax, "cross_entropy": cross_entropy}
    for k, val in INPUTS.items():
        env[k] = np.array(val) if isinstance(val, list) else val
    lines = [
        "# Generated by tools/phrases/cases.py: do not edit. One test per phrase",
        "# of lang/viv/src/phrases.rs, each against the value NumPy or the closed",
        "# form gives for the same inputs.",
        "#! use lib/cas/cas",
        "",
    ]
    for k, val in INPUTS.items():
        lines.append(f"{k} = {viv_lit(val)}")
    lines.append("")
    lines.extend(DEFS.splitlines())
    lines.append("")
    for phrase, py in CASES:
        if isinstance(py, tuple):
            lines.append(f'test "{phrase}" {{')
            lines.append(f"    ph_value, ph_grad = {phrase}")
            for name, expr in zip(("ph_value", "ph_grad"), py):
                a = np.asarray(eval(expr, env), dtype=float)
                tol = viv_num(TOL * (1.0 + float(np.max(np.abs(a)))))
                if a.ndim == 0:
                    lines.append(f"    assert_near({name}, {viv_num(a)}, {tol})")
                else:
                    lines.append(f"    assert_allclose({name}, {viv_lit(a)}, {tol})")
            lines.append("}")
            continue
        want = eval(py, env)
        a = np.asarray(want, dtype=float)
        lines.append(f'test "{phrase}" {{')
        tol = viv_num(TOL * (1.0 + float(np.max(np.abs(a)))))
        if a.ndim == 0:
            lines.append(f"    assert_near({phrase}, {viv_num(a)}, {tol})")
        else:
            lines.append(f"    assert_allclose({phrase}, {viv_lit(a)}, {tol})")
        lines.append("}")
    return "\n".join(lines) + "\n"


def main():
    text = migrated(build(), OUT)
    if "--check" in sys.argv:
        old = open(OUT).read() if __import__("os").path.exists(OUT) else ""
        if old != text:
            print(f"{OUT} is out of date: run python3 tools/phrases/cases.py")
            sys.exit(1)
        return
    open(OUT, "w").write(text)
    print(f"wrote {OUT}: {len(CASES)} cases")


if __name__ == "__main__":
    main()
