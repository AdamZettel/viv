# Finance

Prices of money over time and of contracts on uncertain prices, computed two ways and compared: a closed form where one exists, and a simulation or a numerical solution that does not rely on it. Each post tests the two against each other, with the tolerance a simulation's own standard error allows.

Interest and doubling times, the arithmetic underneath, are in the Algebra 2 channel: the compound-interest problem and the doubling-time posts.

## Options
- black-scholes-by-monte-carlo — 131072 simulated stock prices price a call within 2 standard errors of the Black-Scholes formula, and the error falls as $1/\sqrt{N}$
