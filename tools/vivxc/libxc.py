# SPDX-License-Identifier: MIT OR Apache-2.0
"""ctypes access to the libxc 6.1.0 that PySCF 2.5 ships (pyscf/lib/deps/lib).

Used two ways: the translator reads each functional's catalog entry (name,
family, thresholds, external parameters, mixing), and the oracle harness
evaluates exc/vrho/vsigma through libxc's own xc_lda_exc_vxc / xc_gga_exc_vxc,
so the thresholds are applied exactly as libxc applies them."""
import ctypes as C
import glob
import os

import numpy as np

XC_UNPOLARIZED, XC_POLARIZED = 1, 2
FAMILY = {1: "lda", 2: "gga", 4: "mgga", 8: "lca", 16: "oep", 32: "hyb_gga", 64: "hyb_mgga", 128: "hyb_lda"}
KIND = {0: "x", 1: "c", 2: "xc", 3: "k"}


def _find_lib():
    env = os.environ.get("VIVXC_LIBXC")
    if env:
        return env
    import pyscf
    hits = glob.glob(os.path.join(os.path.dirname(pyscf.__file__), "lib", "deps", "lib", "libxc.*dylib")) + \
        glob.glob(os.path.join(os.path.dirname(pyscf.__file__), "lib", "deps", "lib", "libxc.so*"))
    if not hits:
        raise RuntimeError("libxc not found beside pyscf; set VIVXC_LIBXC")
    return hits[0]


class _Info(C.Structure):
    _fields_ = [("number", C.c_int), ("kind", C.c_int), ("name", C.c_char_p), ("family", C.c_int),
                ("refs", C.c_void_p * 5), ("flags", C.c_int), ("dens_threshold", C.c_double)]


class _Func(C.Structure):   # the head of struct xc_func_type, up to cam_beta
    _fields_ = [("info", C.POINTER(_Info)), ("nspin", C.c_int), ("n_func_aux", C.c_int),
                ("func_aux", C.POINTER(C.c_void_p)), ("mix_coef", C.POINTER(C.c_double)),
                ("cam_omega", C.c_double), ("cam_alpha", C.c_double), ("cam_beta", C.c_double)]


class _Ref(C.Structure):
    _fields_ = [("ref", C.c_char_p), ("doi", C.c_char_p), ("bibtex", C.c_char_p), ("key", C.c_char_p)]


_lib = None


def lib():
    global _lib
    if _lib is None:
        L = C.CDLL(_find_lib())
        L.xc_version_string.restype = C.c_char_p
        L.xc_func_alloc.restype = C.c_void_p
        L.xc_func_init.argtypes = [C.c_void_p, C.c_int, C.c_int]
        L.xc_func_end.argtypes = [C.c_void_p]
        L.xc_func_free.argtypes = [C.c_void_p]
        L.xc_functional_get_name.restype = C.c_char_p
        L.xc_func_info_get_ext_params_name.restype = C.c_char_p
        L.xc_func_info_get_ext_params_name.argtypes = [C.c_void_p, C.c_int]
        L.xc_func_info_get_ext_params_default_value.restype = C.c_double
        L.xc_func_info_get_ext_params_default_value.argtypes = [C.c_void_p, C.c_int]
        L.xc_func_info_get_n_ext_params.argtypes = [C.c_void_p]
        L.xc_func_info_get_references.restype = C.POINTER(_Ref)
        L.xc_func_info_get_references.argtypes = [C.c_void_p, C.c_int]
        for f in ("xc_lda_exc_vxc", "xc_gga_exc_vxc"):
            getattr(L, f).restype = None
        _lib = L
    return _lib


def version():
    return lib().xc_version_string().decode()


def all_ids():
    L = lib()
    n = L.xc_number_of_functionals()
    ids = (C.c_int * n)()
    L.xc_available_functional_numbers(ids)
    return list(ids)


class Func:
    def __init__(self, fid, nspin=XC_UNPOLARIZED):
        L = lib()
        self.p = L.xc_func_alloc()
        if L.xc_func_init(self.p, fid, nspin) != 0:
            raise ValueError(f"libxc cannot init {fid}")
        self.f = C.cast(self.p, C.POINTER(_Func)).contents
        self.nspin = nspin

    def close(self):
        if self.p:
            lib().xc_func_end(self.p)
            lib().xc_func_free(self.p)
            self.p = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def info(self):
        L = lib()
        i = self.f.info.contents
        ip = C.cast(self.f.info, C.c_void_p)
        n = L.xc_func_info_get_n_ext_params(ip)
        ext = [(L.xc_func_info_get_ext_params_name(ip, k).decode(),
                L.xc_func_info_get_ext_params_default_value(ip, k)) for k in range(n)]
        refs = []
        for k in range(5):
            r = L.xc_func_info_get_references(ip, k)
            if not r:
                break
            refs.append(r.contents.ref.decode())
        aux = []
        for k in range(self.f.n_func_aux):
            sub = C.cast(self.f.func_aux[k], C.POINTER(_Func)).contents
            aux.append((sub.info.contents.number, self.f.mix_coef[k]))
        return dict(id=i.number, kind=KIND.get(i.kind, str(i.kind)), family=FAMILY.get(i.family, str(i.family)),
                    descr=i.name.decode(), flags=i.flags, dens_threshold=i.dens_threshold, ext=ext,
                    refs=refs, aux=aux, cam_omega=self.f.cam_omega, cam_alpha=self.f.cam_alpha,
                    cam_beta=self.f.cam_beta)

    def lda(self, rho):
        """rho: (np,) unpolarized or (np, 2) polarized -> zk (np,), vrho (np, nspin)"""
        rho = np.ascontiguousarray(rho, dtype=float)
        npt = rho.shape[0]
        zk = np.zeros(npt)
        vrho = np.zeros((npt, self.nspin))
        lib().xc_lda_exc_vxc(C.c_void_p(self.p), C.c_size_t(npt), rho.ctypes.data_as(C.c_void_p),
                             zk.ctypes.data_as(C.c_void_p), vrho.ctypes.data_as(C.c_void_p))
        return zk, vrho

    def gga(self, rho, sigma):
        rho = np.ascontiguousarray(rho, dtype=float)
        sigma = np.ascontiguousarray(sigma, dtype=float)
        npt = rho.shape[0]
        ns = 1 if self.nspin == 1 else 3
        zk = np.zeros(npt)
        vrho = np.zeros((npt, self.nspin))
        vsigma = np.zeros((npt, ns))
        lib().xc_gga_exc_vxc(C.c_void_p(self.p), C.c_size_t(npt), rho.ctypes.data_as(C.c_void_p),
                             sigma.ctypes.data_as(C.c_void_p), zk.ctypes.data_as(C.c_void_p),
                             vrho.ctypes.data_as(C.c_void_p), vsigma.ctypes.data_as(C.c_void_p))
        return zk, vrho, vsigma


def name(fid):
    return lib().xc_functional_get_name(fid).decode()
