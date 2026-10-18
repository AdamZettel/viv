# Dynamics and chaos

Systems that change step by step: iterated maps and interacting atoms. Each post simulates one and checks a number the theory predicts.

## Maps
- logistic-lyapunov — the logistic map at r = 4: nearby starts separate by a factor of two a step, and the exponent is ln 2 exactly
- pi-in-the-mandelbrot-set — just past the cusp at $c = 1/4 + \varepsilon$, the escape count times $\sqrt\varepsilon$ tends to $\pi$

## Mechanics
- molecular-dynamics-2d — a hundred Lennard-Jones atoms; the total energy never moves from its starting value by more than a thousandth of the kinetic energy

## Circuits
- chuas-circuit-double-scroll — two capacitors, an inductor and a piecewise-linear diode; the largest Lyapunov exponent is scipy's 0.328, and the orbit hops between the two scrolls about 41 times in 210 time units
