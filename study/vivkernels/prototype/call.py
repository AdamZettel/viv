# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""PROTOTYPE: call the C that main's Viv compiler
wrote for axpy.viv, through the hand-written shim.c, with ctypes.
Shows: zero copy (the kernel writes the NumPy array's own buffer), the
result against NumPy, and a Viv runtime error raised as a Python exception
instead of ending the process."""
import ctypes as C, os, numpy as np

lib = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libaxpy.so"))

class F64(C.Structure):
    _fields_ = [("data", C.POINTER(C.c_double)), ("rank", C.c_int64), ("dims", C.POINTER(C.c_int64))]

lib.vk_axpy_v0.argtypes = [C.c_double, F64, F64, C.POINTER(C.c_double), C.c_char_p, C.c_size_t]
lib.vk_axpy_v0.restype = C.c_int

class VivError(RuntimeError):
    pass

def view(a, writable):
    if a.dtype != np.float64 or not a.flags.c_contiguous or (writable and not a.flags.writeable):
        if writable:
            raise TypeError("an in-place argument must be a writable C-contiguous float64 array")
        a = np.ascontiguousarray(a, dtype=np.float64)   # the documented copy rule
    dims = np.asarray(a.shape, dtype=np.int64)
    return a, dims, F64(a.ctypes.data_as(C.POINTER(C.c_double)), a.ndim, dims.ctypes.data_as(C.POINTER(C.c_int64)))

def axpy(a, x, y):
    x, xd, xv = view(x, False)
    y, yd, yv = view(y, True)
    out = C.c_double()
    msg = C.create_string_buffer(1024)
    if lib.vk_axpy_v0(a, xv, yv, C.byref(out), msg, len(msg)):
        raise VivError(msg.value.decode())
    return out.value

rng = np.random.default_rng(0)
x = rng.standard_normal(1000)
y = rng.standard_normal(1000)
y_ref = 2.5 * x + y
addr = y.ctypes.data
s = axpy(2.5, x, y)
print("same buffer after the call:", y.ctypes.data == addr)
print("max |y - (2.5 x + y0)|    :", np.max(np.abs(y - y_ref)))
print("sum matches np.sum        :", s == np.sum(y_ref), abs(s - np.sum(y_ref)))
try:
    axpy(1.0, np.zeros(3), np.zeros(5))
except VivError as e:
    print("VivError raised, process alive; message:")
    print(str(e).rstrip())
print("next call still works     :", axpy(1.0, np.ones(4), np.zeros(4)))
