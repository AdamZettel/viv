# Robotics

Systems that move, and the controllers that decide how. Each post writes out the dynamics and a controller and tests what the controller does in simulation; where an exact value exists (an equilibrium, an eigenvalue, a Riccati residual) it tests that too.

Robotics research mostly benchmarks on simulators this site cannot run in a browser tab: MuJoCo locomotion (Hopper, Walker2d, HalfCheetah, Ant, Humanoid), manipulation suites (Meta-World, robosuite, ManiSkill, RLBench, Franka Kitchen, Adroit), navigation (Habitat, AI2-THOR), and the large-scale trainers around them (Isaac Lab, Brax, MJX). The posts here cover what those simulators are built on: the equations they integrate, the classical controllers that learned policies are compared against, and tests that check both.

## Balancing
- cart-pole-balanced — a PD controller with gains chosen by hand
- lqr-balances-the-cart-pole — the same pole, gains from the Riccati equation

## Swinging up
- a-swing-up-policy-trained-offline-and-flown-in-the-tab — nine trained weights swing an acrobot up from hanging, then an LQR controller balances it
