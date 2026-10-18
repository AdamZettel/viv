# Optimization

Minimizing a function numerically: which direction to step in, how far to go, and how many steps it takes. Each post runs a method on a test function whose minimum is known and checks the iteration count and the final error against an independent implementation.

## Quasi-Newton methods
- bfgs-against-gradient-descent-on-rosenbrock — on Rosenbrock's valley BFGS is within 1e-9 of the minimum after 35 iterations; gradient descent with the same line search takes 22,962
