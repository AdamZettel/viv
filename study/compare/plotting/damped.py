# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ imports
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# @@ the data: a damped oscillation and its envelope
N = 400
t = np.linspace(0.0, 10.0, N)
x = np.exp(-t / 4.0) * np.cos(2.0 * t)
env = np.exp(-t / 4.0)

# @@ the peaks, where x is larger than both neighbors
peaks = [i for i in range(1, N - 1) if x[i] > x[i - 1] and x[i] > x[i + 1]]
print("peaks at t =", " ".join(f"{t[i]:.3f}" for i in peaks))

# @@ a figure with a title and axis labels
fig, ax = plt.subplots()
ax.set_title("A damped oscillation")
ax.set_xlabel("t")
ax.set_ylabel("x")

# @@ two lines and a set of points
ax.plot(t, x, label="x(t)")
ax.plot(t, env, "--", label="envelope")
ax.plot(t[peaks], x[peaks], "o", label="peaks")
ax.legend()

# @@ save it
fig.savefig("damped.png", dpi=100)
print("saved damped.png")

# @@ checks
assert len(peaks) == 3
