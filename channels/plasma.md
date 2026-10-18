# Plasma physics

Charged particles and the fields they make: single-particle motion, waves, instabilities and kinetic theory. Each post runs a calculation and checks its number against linear theory or an independent code.

## Particle motion
- the-boris-push-gyrates-and-drifts — the Boris step keeps the speed to rounding; the gyroperiod is 6.28449 instead of $2\pi$, and a particle started at $\mathbf E \times \mathbf B / B^2$ keeps that velocity

## Instabilities
- two-stream-instability-in-a-particle-code — a 2000-particle PIC code grows the fastest two-stream mode at 0.35366 against the cold-beam rate 0.35355
- hasegawa-wakatani-one-rung-at-a-time — drift waves from the Hasegawa-Mima frequency to the growth rate, and a zonal flow the modified model leaves alone

## Kinetic theory
- landau-damping-from-the-dispersion-relation — the least-damped Langmuir wave at $k\lambda_D = 0.5$ is $1.415662 - 0.153359i$, from the plasma dispersion function
- trapped-electron-bounce-period — in O'Neil trapping an electron caught in a wave trough bounces with period $4K(\sin^2(\theta_0/2))/\omega_B$, 6.287115 at $\theta_0 = 0.1$

## Equilibrium
- grad-shafranov-solovev — a finite-difference Grad-Shafranov solve converges to Solov'ev's closed form at second order, with the Shafranov shift read off the grid
