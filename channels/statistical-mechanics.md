# Statistical mechanics

Sample the states a many-particle system can be in with Boltzmann weights, and compare what the sampling finds with the few models solved exactly. Each post runs the simulation and tests it against the exact answer, with a tolerance measured from the sampling's own error.

## The Ising model
- metropolis-ising-against-onsager — Metropolis sampling of a 16 by 16 grid of spins finds Onsager and Yang's magnetization within three standard errors, and $T_c = 2 / \ln(1 + \sqrt 2)$

## Random walks and rates
- a-random-walk-becomes-diffusion — 4000 walkers: the mean squared displacement grows as 2Dt, and the spread turns Gaussian as the excess kurtosis falls as -6/(5n)
- kinetic-monte-carlo-against-the-master-equation — Gillespie's algorithm on a three-state chain matches the matrix exponential within four standard errors at every time
