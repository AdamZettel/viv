# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""PROTOTYPE pwdft.py: the Python side of
pwdft.c. Hand-written here; `viv export` would generate it, docstrings from
the intent lens. Needs ctypes and NumPy, nothing else.

    import pwdft
    r = pwdft.run(ecut=10.0)        # silicon, LDA; seconds
    r["etot"], r["bands_ev"].shape

The library is built from pwdft.c on first import when libpwdft.so is missing
or older than pwdft.c:  cc -O2 -ffp-contract=off -shared -fPIC pwdft.c -o libpwdft.so
"""
import ctypes as C
import os
import signal
import subprocess
import threading
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(_here, "pwdft.c")
_lib_path = os.path.join(_here, "libpwdft.so")


def _build():
    cc = os.environ.get("CC", "cc")
    subprocess.run([cc, "-O2", "-ffp-contract=off", "-fvisibility=hidden", "-shared", "-fPIC",
                    "-Wno-unused-function", _src, "-o", _lib_path], check=True)


if not os.path.exists(_lib_path) or (os.path.exists(_src) and os.path.getmtime(_src) > os.path.getmtime(_lib_path)):
    _build()

_LINE = C.CFUNCTYPE(C.c_int, C.c_char_p, C.c_void_p)


class _Params(C.Structure):
    _fields_ = [("ecut", C.c_double), ("nkmesh", C.c_double), ("alat", C.c_double),
                ("on_line", _LINE), ("ctx", C.c_void_p)]


class _Results(C.Structure):
    _fields_ = [("etot", C.c_double), ("efermi", C.c_double), ("gap_ev", C.c_double),
                ("iter", C.c_double), ("drho", C.c_double),
                ("tests_passed", C.c_int), ("tests_failed", C.c_int),
                ("npts", C.c_int), ("nbands", C.c_int),
                ("bands_ev", C.POINTER(C.c_double))]


_lib = C.CDLL(_lib_path)
_lib.pwdft_run.argtypes = [C.POINTER(_Params), C.POINTER(_Results), C.c_char_p, C.c_size_t]
_lib.pwdft_run.restype = C.c_int
_lib.pwdft_free.argtypes = [C.POINTER(_Results)]
_lib.pwdft_abi.restype = C.c_int


class VivError(RuntimeError):
    """A run-time error inside the program; the message is Viv's, with the source line."""


def run(ecut=10.0, nkmesh=2, alat=10.26, on_line=None, log=None):
    """Silicon in the diamond structure: Kohn-Sham LDA in plane waves, GTH
    pseudopotentials, SCF to 1e-9, then the band structure along L-Gamma-X.

    ecut    plane-wave cutoff, hartree
    nkmesh  the k-mesh is nkmesh^3, shifted
    alat    lattice constant, bohr
    on_line called with each line the program prints (progress, test results);
            default: dropped
    log     a list to append every printed line to

    Returns a dict: etot, efermi (hartree), gap_ev, iter, drho,
    tests_passed, tests_failed, bands_ev (NumPy array, path points x bands).
    Ctrl-C stops the run at its next printed line (one per SCF iteration)
    and raises KeyboardInterrupt.
    """
    state = {"interrupted": False}

    def line_cb(raw, _ctx):
        if log is not None:
            log.append(raw.decode())
        if on_line is not None:
            on_line(raw.decode())
        return 1 if state["interrupted"] else 0

    # Ctrl-C during the run: Python runs a signal handler only between
    # bytecodes, which here means inside line_cb. The handler sets a flag;
    # line_cb returns 1, and pwdft.c leaves the run at that line.
    def on_sigint(_sig, _frame):
        state["interrupted"] = True

    in_main = threading.current_thread() is threading.main_thread()
    old = signal.signal(signal.SIGINT, on_sigint) if in_main else None
    cb = _LINE(line_cb)
    p = _Params(float(ecut), float(nkmesh), float(alat), cb, None)
    r = _Results()
    msg = C.create_string_buffer(2048)
    try:
        code = _lib.pwdft_run(C.byref(p), C.byref(r), msg, len(msg))
    finally:
        if in_main:
            signal.signal(signal.SIGINT, old)
    if code == 2 or state["interrupted"]:
        raise KeyboardInterrupt("pwdft.run stopped")
    if code != 0:
        raise VivError(msg.value.decode())
    try:
        bands = np.ctypeslib.as_array(r.bands_ev, shape=(r.npts, r.nbands)).copy()
    finally:
        _lib.pwdft_free(C.byref(r))
    return {"etot": r.etot, "efermi": r.efermi, "gap_ev": r.gap_ev, "iter": int(r.iter),
            "drho": r.drho, "tests_passed": r.tests_passed, "tests_failed": r.tests_failed,
            "bands_ev": bands}
