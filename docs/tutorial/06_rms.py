# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
import numpy as np

def rms(x):
    return np.sqrt(np.mean(x * x))

x = np.array([3.0, 4.0])
print("rms", rms(x))

def test_rms():
    assert abs(rms(x) - np.sqrt(12.5)) < 1e-12
