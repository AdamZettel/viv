# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference values for posts/2026-09-25-the-boris-push-gyrates-and-drifts.viv.

Two independent sources for every number the post checks:
  1. closed forms for the Boris map (rotation angle 2 atan(w dt / 2), orbit
     radius sqrt(1 + (w dt)^2 / 4) v / w, E x B drift as the map's fixed point);
  2. a separate numpy implementation of the Boris push, written from
     Birdsall & Langdon, Plasma Physics via Computer Simulation (1991), sec. 4-4,
     in vector form with np.cross, not the component form the post uses.

Run: python3 posts/refs/boris_push.py   (numpy 1.26)
"""
import numpy as np

DT = 0.05
NSTEP = 20000
E0 = 0.1
B = np.array([0.0, 0.0, 1.0])


def boris(x, v, E, dt, n):
    xs = [x.copy()]
    for _ in range(n):
        vm = v + 0.5 * dt * E
        t = 0.5 * dt * B
        s = 2.0 * t / (1.0 + t @ t)
        vp = vm + np.cross(vm + np.cross(vm, t), s)
        v = vp + 0.5 * dt * E
        x = x + dt * v
        xs.append(x.copy())
    return np.array(xs), v


theta = 2.0 * np.arctan(DT / 2.0)
print("rotation angle per step 2 atan(dt/2) =", repr(theta))
print("gyroperiod 2 pi dt / theta           =", repr(2 * np.pi * DT / theta))
print("relative period error                 =", repr(DT / theta - 1.0), " dt^2/12 =", DT**2 / 12)
print("orbit radius sqrt(1 + dt^2/4)         =", repr(np.sqrt(1 + DT**2 / 4)))

# pure gyration, numpy Boris
xs, v = boris(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.zeros(3), DT, NSTEP)
print("numpy: |v| after", NSTEP, "steps - 1       =", repr(np.linalg.norm(v) - 1.0))
c = xs[:, :2].mean(axis=0)
# circle through the first three positions
(ax, ay), (bx, by), (cx, cy) = xs[0, :2], xs[1, :2], xs[2, :2]
d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
r = np.hypot(xs[:, 0] - ux, xs[:, 1] - uy)
print("numpy: circumcenter                    =", ux, uy)
print("numpy: radius min, max                 =", repr(r.min()), repr(r.max()))

# crossed fields
E = np.array([E0, 0.0, 0.0])
vE = np.cross(E, B) / (B @ B)
print("E x B / B^2                            =", vE)
xs, v = boris(np.zeros(3), vE.copy(), E, DT, NSTEP)
print("numpy: start at v_E, |v - v_E| at end  =", repr(np.linalg.norm(v - vE)))
xs, v = boris(np.zeros(3), np.array([0.0, 1.0, 0.0]), E, DT, NSTEP)
mean_v = (xs[-1] - xs[0]) / (NSTEP * DT)
u0 = np.linalg.norm(np.array([0.0, 1.0, 0.0]) - vE)
pred = u0 * abs(np.sin(NSTEP * theta / 2)) / (NSTEP * np.sin(theta / 2))
print("numpy: mean velocity                   =", mean_v)
print("numpy: |mean v - v_E|                  =", repr(np.linalg.norm(mean_v - vE)))
print("closed form |u0| |sin(N th/2)| / (N sin(th/2)) =", repr(pred))
print("bound |u0| / (N sin(th/2))             =", repr(u0 / (NSTEP * np.sin(theta / 2))))

# period error vs dt
for dt in [0.4, 0.2, 0.1, 0.05, 0.025]:
    th = 2 * np.arctan(dt / 2)
    print(f"dt = {dt:6.3f}: T/2pi - 1 = {dt / th - 1:.6e}   dt^2/12 = {dt * dt / 12:.6e}")
