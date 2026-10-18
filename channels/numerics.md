# Numerics

Numerical methods run on problems whose exact answer is known: floating-point arithmetic, time integrators, interpolation, Fourier sums, quadrature, eigenvalue solvers, matrix equations, partial differential equations and root finding. Each post tests how far the computed answer is from the exact one, or why it differs.

## Floating point
- kahan-summation-horner-noise-and-randus-planes — Kahan's sum of $1/k$ is correctly rounded where the plain sum is 414 ulps off; $(x - 1)^7$ expanded is noise near $x = 1$; RANDU's triples lie on 15 planes

## Integrating in time
- euler-loses-the-planet — forward Euler gains energy until the planet escapes; leapfrog, at the same cost, keeps the energy error under one percent
- forward-euler-blows-up-past-h-equals-2-over-lambda — on a stiff equation forward Euler's growth factor $|1 - h\lambda|$ passes 1 at $h = 2/\lambda$, and backward Euler's stays below 1
- make-euler-stable-on-a-stiff-equation — problem: forward Euler reaches 129.7 on y' = -50y; write backward Euler
- make-numerov-find-the-decaying-solution — a problem, outward Numerov is off by $2 \times 10^9$ at $x = 18$; follow $e^{-x}$ to $10^{-5}$

## Approximation
- runge-phenomenon — equispaced interpolation diverges as points are added; Chebyshev points converge geometrically
- gibbs-overshoot — a Fourier partial sum overshoots a jump by 9%, at every order

## Quadrature
- gauss-legendre-quadrature-and-where-tanh-sinh-wins — five Gauss-Legendre points are exact through degree 9; on $1/\sqrt{1 - x^2}$ 128 of them miss by $1.4 \times 10^{-2}$, and tanh-sinh reaches rounding with 37

## Eigenvalues
- oscillator-levels — the harmonic oscillator's levels from a finite-difference Hamiltonian
- wigner-semicircle — the eigenvalues of a random symmetric matrix fill a semicircle
- lobpcg-finds-the-lowest-states — the six lowest states of a particle in a box, by LOBPCG, without writing down the Hamiltonian matrix
- jacobi-eigenvalue-sweeps — plane rotations diagonalize an 8 by 8 matrix; the off-diagonal norm is rounding after five sweeps, and the eigenvalues match numpy's eigh to $10^{-14}$
- make-power-iteration-converge-on-a-tied-pair — problem: eigenvalues +2 and -2 stall power iteration; a shift breaks the tie
- lanczos-loses-orthogonality — sixty Lanczos steps find both ends of the spectrum to $10^{-14}$, and without reorthogonalization find the top eigenvalue three times

## Orthogonalization
- make-gram-schmidt-orthogonal-to-rounding — problem: classical Gram-Schmidt leaves two of ten vectors at 60 degrees; make them orthogonal to 1e-15

## Partial differential equations
- make-the-heat-equation-step-stable — problem: the explicit heat step grows past r = 1/2; write the implicit one

## Matrix equations
- the-error-follows-the-condition-number — Hilbert-system errors track $\kappa\varepsilon$ up to $\kappa = 1.7 \times 10^{16}$; in least squares, Householder QR's error is about $\kappa\varepsilon$ and the normal equations' about $\kappa^2\varepsilon$
- jacobi-sor-and-cg-on-the-poisson-equation — 948 Jacobi sweeps, 60 of SOR and 25 conjugate-gradient iterations on a 15 by 15 Poisson grid, each count set by the spectral radius or the condition number
- solving-the-riccati-equation-without-integrating-it — the Schur method solves the LQR Riccati equation to rounding, where integrating it in time returns 73.783 instead of 414.214

## PDEs
- upwind-advection-smears-by-its-modified-equation — upwind keeps the mass and adds no extrema, and its modified equation's diffusion predicts the peak's fall to $1/\sqrt 3$
- crank-nicolson-is-second-order-in-time — halving the time step cuts the Crank-Nicolson error by 4 and backward Euler's by 2, until the space error takes over
- burgers-in-nonconservative-form-puts-the-shock-in-the-wrong-place — Godunov's conservative scheme moves the shock at the Rankine-Hugoniot speed 0.625; upwinding u u_x moves it at 0.513 on every grid

## Roots
- newton-basins — which root Newton's method finds, from every starting point
