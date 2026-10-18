# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The PySCF side of port P0: the oracle, the stage timings and the profile.

    python3 study/ports/pyscf/pyscf_side.py oracle [case ...]
    python3 study/ports/pyscf/pyscf_side.py time <case> <reps> [--chk]   (warm, in process)
    python3 study/ports/pyscf/pyscf_side.py cold <case>             (one run, imports included)
    python3 study/ports/pyscf/pyscf_side.py profile <case>

`oracle` writes oracle/<case>.json: PySCF's energy, orbital energies and
iteration count at the verification settings of port.toml, and the same at
the timing settings. Every float is written with repr(), so it round-trips.

`time` runs `scf.RHF(mol).kernel()` (or UHF) `reps` times in one process
after one warm-up run, each time on a fresh SCF object so that the integrals
are computed every time, as Viv computes them. Each stage is timed by
wrapping the PySCF method that does it (no change to PySCF):

    one_electron   get_hcore, get_ovlp (libcint int1e_*)
    two_electron   mol.intor('int2e', aosym='s8') inside the first get_jk
    guess          get_init_guess
    jk             get_veff / get_jk, minus the int2e build (libcvhf)
    diis           CDIIS.update (Python + NumPy)
    diagonalize    eig (scipy.linalg.eigh(F, S): orthogonalization and
                   diagonalization together, LAPACK)
    energy         energy_tot
    other          everything else in kernel, also timed apart: get_fock
                   without DIIS (fock), make_rdm1 (density), get_grad
                   (gradient), get_occ (occupation), dump_chk (checkpoint:
                   the HDF5 file PySCF writes every cycle by default), and
                   the loop's own lines

It prints one JSON line per run on stdout.

`profile` runs the case once under cProfile after a warm-up and splits the
time into C (libcint, libcvhf through ctypes; NumPy, SciPy, BLAS and LAPACK
builtins) and Python, by function.
"""
import json
import os
import resource
import sys
import time
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
CONF = tomllib.load(open(os.path.join(HERE, "port.toml"), "rb"))


def case_conf(name):
    for c in CONF["case"]:
        if c["name"] == name:
            return c
    raise SystemExit(f"no case {name!r} in port.toml")


def atoms_of(c):
    g = CONF["geometry"]
    key = c["geometry"]
    if key.startswith("cluster"):
        n = int(key[len("cluster"):])
        a = g["cluster_spacing"]
        out = []
        for corner in g["cluster_corners"][:n]:
            for part in g["water"].split(";"):
                sym, x, y, z = part.split()
                out.append("%s %r %r %r" % (sym, float(x) + a * corner[0], float(y) + a * corner[1], float(z) + a * corner[2]))
        return "; ".join(out)
    return g[key]


def build(c):
    from pyscf import gto, scf
    mol = gto.M(atom=atoms_of(c), basis=c["basis"], charge=c.get("charge", 0), spin=c.get("spin", 0), verbose=0)
    return mol


def new_mf(mol, c, conv_tol, conv_tol_grad=None, incore=None):
    from pyscf import scf
    mf = scf.UHF(mol) if c["method"] == "uhf" else scf.RHF(mol)
    direct = c.get("jk") == "direct" if incore is None else not incore
    if direct:
        # refuse the in-core ERI store: get_jk then takes the direct path
        # (libcvhf, Schwarz and density screening, incremental Fock build)
        mf._is_mem_enough = lambda: False
    mf.init_guess = "1e"
    mf.conv_tol = conv_tol
    if conv_tol_grad is not None:
        mf.conv_tol_grad = conv_tol_grad
    mf.diis_space = 8
    mf.diis_start_cycle = 1
    mf.verbose = 0
    mf.max_memory = 12000
    return mf


class Cycles:
    """Counts kernel cycles and J/K builds through the callback and get_jk."""
    def __init__(self):
        self.cycles = 0

    def __call__(self, envs):
        self.cycles = envs["cycle"] + 1


def run(mol, c, conv_tol, conv_tol_grad=None):
    mf = new_mf(mol, c, conv_tol, conv_tol_grad)
    cyc = Cycles()
    mf.callback = cyc
    e = mf.kernel()
    return mf, e, cyc.cycles


def num(x):
    return repr(float(x))


def oracle(names):
    import numpy
    import pyscf
    from pyscf import lib
    lib.num_threads(1)
    v = CONF["verify"]
    t = CONF["timing"]
    for name in names:
        c = case_conf(name)
        mol = build(c)
        mf, e, cycles = run(mol, c, v["pyscf_conv_tol"], v["pyscf_conv_tol_grad"])
        assert mf.converged, name
        mft, et, cyct = run(mol, c, t["pyscf_conv_tol"])
        assert mft.converged, name
        moe = numpy.asarray(mf.mo_energy)
        rec = {
            "case": name,
            "pyscf": pyscf.__version__,
            "atom": atoms_of(c),
            "basis": c["basis"],
            "method": c["method"],
            "charge": c.get("charge", 0),
            "spin": c.get("spin", 0),
            "nao": int(mol.nao_nr()),
            "nelec": [int(x) for x in mol.nelec],
            "enuc": num(mol.energy_nuc()),
            "settings": {"jk": c.get("jk", "incore"), "init_guess": "1e", "diis_space": 8, "diis_start_cycle": 1,
                         "conv_tol": v["pyscf_conv_tol"], "conv_tol_grad": v["pyscf_conv_tol_grad"]},
            "e_tot": num(e),
            "cycles": cycles,
            "s2": num(mf.spin_square()[0]) if c["method"] == "uhf" else "0.0",
            "timing_settings": {"conv_tol": t["pyscf_conv_tol"], "conv_tol_grad": "sqrt(conv_tol)"},
            "timing_e_tot": num(et),
            "timing_cycles": cyct,
        }
        if c["method"] == "uhf":
            rec["mo_energy_alpha"] = [num(x) for x in moe[0]]
            rec["mo_energy_beta"] = [num(x) for x in moe[1]]
        else:
            rec["mo_energy"] = [num(x) for x in moe]
        path = os.path.join(HERE, "oracle", name + ".json")
        with open(path, "w") as f:
            json.dump(rec, f, indent=1)
            f.write("\n")
        print("%-22s nao=%3d  E=%.12f  cycles=%d  (timing: %d cycles, E=%.12f)" % (name, mol.nao_nr(), e, cycles, cyct, et))


# ---- stage timing ---------------------------------------------------------------

class StageClock:
    """Exclusive time per stage: a stage's time excludes the stages it calls."""
    def __init__(self):
        self.acc = {}
        self.calls = {}
        self.stack = []

    def wrap(self, owner, attr, stage):
        orig = getattr(owner, attr)
        clock = self

        def wrapped(*a, **k):
            clock.stack.append([stage, 0.0])
            t0 = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                dt = time.perf_counter() - t0
                st, child = clock.stack.pop()
                clock.acc[st] = clock.acc.get(st, 0.0) + dt - child
                clock.calls[st] = clock.calls.get(st, 0) + 1
                if clock.stack:
                    clock.stack[-1][1] += dt
        setattr(owner, attr, wrapped)
        return orig

    def reset(self):
        self.acc = {}
        self.calls = {}


def install(clock, c):
    """Wrap the PySCF methods that do each stage, at class level."""
    from pyscf import gto, scf
    from pyscf.scf import diis as scf_diis, hf, uhf
    cls = uhf.UHF if c["method"] == "uhf" else hf.RHF
    orig_intor = gto.Mole.intor

    def intor(self, intor_name, *a, **k):
        stage = "two_electron" if intor_name.startswith("int2e") else "one_electron"
        clock.stack.append([stage, 0.0])
        t0 = time.perf_counter()
        try:
            return orig_intor(self, intor_name, *a, **k)
        finally:
            dt = time.perf_counter() - t0
            st, child = clock.stack.pop()
            clock.acc[st] = clock.acc.get(st, 0.0) + dt - child
            clock.calls[st] = clock.calls.get(st, 0) + 1
            if clock.stack:
                clock.stack[-1][1] += dt
    gto.Mole.intor = intor
    # one-electron: get_hcore and get_ovlp are module-level in hf and bound
    # on the class; wrap the class attributes
    clock.wrap(hf.SCF, "get_hcore", "one_electron")
    clock.wrap(hf.SCF, "get_ovlp", "one_electron")
    clock.wrap(cls, "get_init_guess", "guess")
    clock.wrap(cls, "get_veff", "jk")
    clock.wrap(cls, "get_jk", "jk")
    clock.wrap(scf_diis.CDIIS, "update", "diis")
    clock.wrap(hf.SCF, "eig", "diagonalize")
    if c["method"] == "uhf":
        clock.wrap(uhf.UHF, "eig", "diagonalize")
    clock.wrap(cls, "energy_tot", "energy")
    clock.wrap(cls, "energy_elec", "energy")
    # the rest of the kernel, kept apart so the "other" row can be read
    clock.wrap(cls, "get_fock", "fock")
    clock.wrap(cls, "make_rdm1", "density")
    clock.wrap(cls, "get_grad", "gradient")
    clock.wrap(cls, "get_occ", "occupation")
    clock.wrap(hf.SCF, "dump_chk", "checkpoint")


def cpu_now():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def time_case(name, reps, chk=False, incore=None):
    from pyscf import lib
    lib.num_threads(1)
    c = case_conf(name)
    t = CONF["timing"]
    oc = json.load(open(os.path.join(HERE, "oracle", name + ".json")))
    mol = build(c)
    clock = StageClock()
    install(clock, c)
    for rep in range(reps + 1):
        clock.reset()
        mf = new_mf(mol, c, t["pyscf_conv_tol"], incore=incore)
        if not chk:
            # the timed column: no checkpoint file (Viv writes none); the
            # `--chk` run keeps PySCF's default HDF5 checkpoint every cycle
            mf.chkfile = None
        cyc = Cycles()
        mf.callback = cyc
        c0 = cpu_now()
        t0 = time.perf_counter()
        e = mf.kernel()
        wall = time.perf_counter() - t0
        cpu = cpu_now() - c0
        ok = abs(e - float(oc["e_tot"])) < t["timing_energy_tol"] and mf.converged
        stages = dict(clock.acc)
        stages["other"] = wall - sum(stages.values())
        rec = {"case": name, "rep": rep, "chkfile": chk, "direct": c.get("jk") == "direct" if incore is None else not incore, "warmup": rep == 0, "total": wall, "cpu": cpu,
               "cpu_over_wall": cpu / wall, "stages": stages, "calls": dict(clock.calls),
               "cycles": cyc.cycles, "e_tot": num(e), "ok": bool(ok),
               "threads": lib.num_threads()}
        print(json.dumps(rec), flush=True)


def cold_case(name):
    """One fresh run: the caller times this whole process (imports included)."""
    t0 = time.perf_counter()
    from pyscf import lib
    lib.num_threads(1)
    c = case_conf(name)
    mol = build(c)
    mf = new_mf(mol, c, CONF["timing"]["pyscf_conv_tol"])
    mf.chkfile = None
    e = mf.kernel()
    print(json.dumps({"case": name, "e_tot": num(e), "in_process": time.perf_counter() - t0}))


# functions whose own time (tottime) is C reached through ctypes: libcint and
# libcvhf. cProfile does not see a ctypes call as a function of its own, so
# the C time lands in the Python function that makes the call.
CTYPES_CALLERS = {
    ("moleintor.py", "getints2c"): "libcint",
    ("moleintor.py", "getints4c"): "libcint",
    ("moleintor.py", "getints3c"): "libcint",
    ("_vhf.py", "incore"): "libcvhf",
    ("_vhf.py", "direct"): "libcvhf",
    ("_vhf.py", "direct_mapdm"): "libcvhf",
    ("_vhf.py", "direct_bindm"): "libcvhf",
    ("_vhf.py", "nr_direct_drv"): "libcvhf",
}
# f2py's LAPACK wrappers are not builtins either: scipy.linalg.eigh's own
# time is the LAPACK call (dsygvd) it makes
CTYPES_CALLERS[("_decomp.py", "eigh")] = "numpy/scipy/BLAS/LAPACK (C)"


def profile_case(name):
    import cProfile
    import pstats
    from pyscf import lib
    lib.num_threads(1)
    c = case_conf(name)
    mol = build(c)
    new_mf(mol, c, CONF["timing"]["pyscf_conv_tol"]).kernel()
    pr = cProfile.Profile()
    mf = new_mf(mol, c, CONF["timing"]["pyscf_conv_tol"])
    mf.chkfile = None
    t0 = time.perf_counter()
    pr.enable()
    mf.kernel()
    pr.disable()
    wall = time.perf_counter() - t0
    st = pstats.Stats(pr)
    groups = {"libcint": 0.0, "libcvhf": 0.0, "numpy/scipy/BLAS/LAPACK (C)": 0.0, "Python": 0.0}
    rows = []
    for (fn, line, func), (cc, nc, tt, ct, callers) in st.stats.items():
        base = os.path.basename(fn)
        if (base, func) in CTYPES_CALLERS:
            g = CTYPES_CALLERS[(base, func)]
        elif fn == "~" or fn.startswith("<built-in") or "numpy" in fn and base.endswith(".so"):
            g = "numpy/scipy/BLAS/LAPACK (C)" if ("numpy" in func or "scipy" in func or "method" in func or "built-in" in func or fn == "~") else "Python"
            # builtins that are plain Python machinery stay Python
            if any(k in func for k in ("isinstance", "getattr", "hasattr", "len", "setattr", "__build_class__", "print", "time.perf_counter", "dict", "list")):
                g = "Python"
        else:
            g = "Python"
        groups[g] += tt
        rows.append((tt, g, f"{base}:{line}({func})", nc))
    rows.sort(reverse=True)
    total_prof = sum(groups.values())
    out = {"case": name, "wall_profiled": wall, "sum_tottime": total_prof,
           "groups": groups, "top": [{"tottime": r[0], "group": r[1], "func": r[2], "calls": r[3]} for r in rows[:30]]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "oracle":
        oracle(sys.argv[2:] or [c["name"] for c in CONF["case"]])
    elif cmd == "time":
        time_case(sys.argv[2], int(sys.argv[3]), chk="--chk" in sys.argv, incore=True if "--incore" in sys.argv else None)
    elif cmd == "cold":
        cold_case(sys.argv[2])
    elif cmd == "profile":
        profile_case(sys.argv[2])
    else:
        raise SystemExit(__doc__)
