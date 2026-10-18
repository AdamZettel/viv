# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Reference numbers for posts/2026-09-24-a-box-hanging-from-two-ropes.viv.

The tensions are found here without the post's closed forms: the two balance
equations at the knot,

    sideways   T1 cos(th1) - T2 cos(th2) = 0
    upward     T1 sin(th1) + T2 sin(th2) = W,

are solved as a 2 by 2 linear system by numpy.linalg.solve (LAPACK), for every
pair of angles the post quotes. Angles are in degrees above the horizontal.

    python3 posts/refs/a-box-hanging-from-two-ropes.py > posts/refs/a-box-hanging-from-two-ropes.out
"""
import math

import numpy as np

G = 9.8
MASS = 10.0
W = MASS * G


def tensions(w, th1, th2):
    a, b = math.radians(th1), math.radians(th2)
    A = np.array([[math.cos(a), -math.cos(b)], [math.sin(a), math.sin(b)]])
    return np.linalg.solve(A, np.array([0.0, w]))


def show(name, value):
    print(f"{name} = {value!r}")


show("weight_N", W)
t = tensions(W, 30.0, 30.0)
show("tension_30_30_N", float(t[0]))
show("tension_30_30_right_N", float(t[1]))
show("upward_part_30_N", float(t[0] * math.sin(math.radians(30.0))))
show("sideways_part_30_N", float(t[0] * math.cos(math.radians(30.0))))
show("ratio_90_90", float(tensions(1.0, 90.0, 90.0)[0]))
show("ratio_30_30", float(tensions(1.0, 30.0, 30.0)[0]))
t = tensions(W, 60.0, 30.0)
show("tension_60_30_steep_N", float(t[0]))
show("tension_60_30_shallow_N", float(t[1]))
show("sideways_60_30_N", float(t[0] * math.cos(math.radians(60.0))))
show("upward_60_30_steep_N", float(t[0] * math.sin(math.radians(60.0))))
show("upward_60_30_shallow_N", float(t[1] * math.sin(math.radians(30.0))))
t = tensions(1.0, 60.0, 30.0)
show("ratio_60_30_steep", float(t[0]))
show("ratio_60_30_shallow", float(t[1]))
show("ratio_10_10", float(tensions(1.0, 10.0, 10.0)[0]))
show("ratio_5_5", float(tensions(1.0, 5.0, 5.0)[0]))
show("ratio_flat_0p2", float(tensions(1.0, 0.2, 0.2)[0]))
# the angle trap: 30 degrees put into W / (2 cos(phi)), the formula for the angle from the vertical
show("wrong_angle_ratio", 1.0 / (2.0 * math.cos(math.radians(30.0))))
t = tensions(50.0, 45.0, 60.0)
show("sign_45_wire_N", float(t[0]))
show("sign_60_wire_N", float(t[1]))
