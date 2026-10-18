# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""CPython stand-in for Viv's `grad` and `value_and_grad`, so a NumPy program that
calls them runs under CPython as the differential reference.

    from viv import value_and_grad
    l, g = value_and_grad(loss)(W, X, Y)

The gradient is computed by the complex step, df/dx_i = Im f(x + i h e_i) / h with
h = 1e-200, which has no subtraction and so no cancellation: it is accurate to rounding
for code that is analytic in x (sums, products, tanh, exp, @, ...). Code that branches on
x, takes abs or max of it, or writes x into a real array gives wrong reference gradients;
the report says the reference gradient came from the complex step.
"""
import numpy as np

H = 1e-200


def _cstep(f, argnums, args):
    args = list(args)
    x = np.asarray(args[argnums], dtype=float)
    g = np.zeros(x.shape)
    flat = g.reshape(-1)
    for i in range(x.size):
        xc = x.astype(complex)
        xc.reshape(-1)[i] += 1j * H
        a = list(args)
        a[argnums] = xc if x.ndim else complex(xc)
        flat[i] = np.imag(f(*a)) / H
    return g if x.ndim else float(g)


def value_and_grad(f, argnums=0):
    def run(*args):
        return f(*args), _cstep(f, argnums, args)
    return run


def grad(f, argnums=0):
    def run(*args):
        return _cstep(f, argnums, args)
    return run
