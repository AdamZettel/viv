# pde — partial differential equations, solved and checked

One program per equation. Each solves it, draws the solution, and ends in tests
against something exact: a closed-form solution, a conserved quantity, a
measured order of accuracy.

```sh
cargo build --release -p viv

target/release/viv run pde/heat_1d.viv     # the interpreter
pde/check.sh                                # all of them, through BOTH engines
pde/check.sh euler_sod kdv_soliton
```

`check.sh` runs every solver through the interpreter and through
Viv → C → native, requires every test to pass in each, and requires the two
outputs to agree line for line.

| Kind | Solvers |
|---|---|
| parabolic | `heat_1d` (Crank–Nicolson vs FTCS), `heat_2d_adi` (Peaceman–Rachford), `black_scholes` (Crank–Nicolson with a Rannacher start) |
| hyperbolic | `advection_1d` (upwind, Lax–Wendroff), `wave_1d` (leapfrog), `burgers_shock` (Godunov), `euler_sod` (finite volumes, HLL, with the exact Riemann solution) |
| elliptic | `laplace_2d` (SOR vs Gauss–Seidel), `poisson_2d` (matrix-free conjugate gradients), `drum_modes` (Helmholtz eigenproblem) |
| dispersive | `schrodinger_1d`, `kdv_soliton`, `nls_soliton` (split-step Fourier) |
| nonlinear parabolic | `burgers_viscous` (travelling wave), `fisher_kpp` (front speed, with Bramson's correction) |

## The header is the web page

<https://viv-lang.org/pde> is generated from these files by
`www/scripts/pde-build.mjs`, which runs each one and refuses to build the site
if a test fails. A solver carries its page in its `#!` header:

```viv
#! pde       The heat equation
#! equation  u_t = \kappa\, u_{xx}
#! kind      parabolic
#! method    Crank-Nicolson: implicit, second order in space and time
#! checks    the exact decay of a sine mode; the measured order of accuracy
#! | Prose, with $math$, `code` and *emphasis*. A bare `#! |` ends a paragraph.
```

The numbers and the figure on the page are that run's `show` lines and its
`figure()` / `plot()` records, so a page cannot state a result the program did
not print.

## What the language needed

Writing these added to Viv: `roll` (a periodic stencil is
`roll(u, 1) - 2.0 * u + roll(u, -1)`), `tridiag_solve` (the Thomas algorithm),
`max(a)` / `min(a)` as reductions, `pow(array, s)`, and unary minus on arrays.
Running them through both engines also found two bugs in the compiler's type
inference, now regression-tested in `lang/viv/tests/e2e.rs`:
`cos(theta)` taught an untyped parameter that it was a number, though `cos` maps
over arrays; and inside a function, a local whose type was not yet known let a
global of the same name show through.
