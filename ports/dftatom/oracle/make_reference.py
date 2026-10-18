# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2006-2012 Ondřej Čertík, Jiří Vackář, John Pask
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Runs upstream dftatom (the Fortran) on every comparison case and writes
ports/dftatom/oracle/upstream_reference.json.

    python3 ports/dftatom/oracle/make_reference.py DFTATOM_CHECKOUT

DFTATOM_CHECKOUT is a clone of https://github.com/certik/dftatom at
e49b3042, built twice with CMake:

    cmake -S . -B build-strict -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_Fortran_FLAGS_RELEASE="-O2 -fno-fast-math -ffp-contract=off"
    cmake -S . -B build        -DCMAKE_BUILD_TYPE=Release   # upstream's flags
    make -C build-strict && make -C build

and oracle.f90 (beside this file) compiled against each library, the strict
one twice, with the platform's libm and with Viv's (vmath_shim.c):

    cc -O2 -ffp-contract=off -c -I <viv>/lang/viv/src vmath_shim.c
    gfortran -O2 -fno-fast-math -ffp-contract=off -I build-strict/mod_files \
        oracle.f90 build-strict/src/libdftatom.a vmath_shim.o -o oracle_vmath
    gfortran -O2 -fno-fast-math -ffp-contract=off -I build-strict/mod_files \
        oracle.f90 build-strict/src/libdftatom.a -o oracle_strict
    gfortran -O3 -march=native -ffast-math -funroll-loops -I build/mod_files \
        oracle.f90 build/src/libdftatom.a -o oracle_fast

The strict build (IEEE arithmetic, no fused multiply-add) linked with Viv's
libm is the oracle the port must equal bit for bit ("upstream"); the same
build with the platform's libm ("upstream_libm") measures what an ulp of
exp does to the answers; the fast build, upstream's own release flags, gives
the timings. The cases: the exported program's cases (atom.cases.json), and
upstream's own tests/lda and tests/rlda runs for the atoms NIST lists, each on
the mesh upstream's data8.dat gives that atom (its 1e-8 Hartree meshes).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = os.path.dirname(HERE)

# upstream's tests/lda/conv_lda.f90 and tests/rlda/conv_rlda.f90
LDA_MESH = dict(r_min=1e-7, r_max=50.0, a=2.7e6)
RLDA_MESH = dict(r_min=1e-8, r_max=50.0, a=6.2e7)
DEFAULTS = dict(z=26, relativistic=False, r_min=1e-7, r_max=50.0, a=2.7e6, n=6000, c=137.0359895,
                reigen_eps=1e-10, reigen_max_iter=100, mixing_eps=5e-9, mixing_alpha=0.5,
                mixing_max_iter=200, perturb=True)
NIST_Z = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 36, 54, 79, 92]


def data8(checkout, sub):
    out = {}
    for line in open(os.path.join(checkout, "tests", sub, "data8.dat")):
        z, n = line.split()
        out[int(z)] = int(n)
    return out


def upstream_converged(checkout, sub, name):
    """Upstream's own converged energies (tests/lda/nonrel_energies.dat,
    tests/rlda/rel_energies.dat): {Z: (E_tot, [eigenvalues])}."""
    lines = [l for l in open(os.path.join(checkout, "tests", sub, name)) if not l.startswith("#")]
    out, i = {}, 0
    while i < len(lines):
        z = int(lines[i])
        etot = float(lines[i + 1])
        eig = [float(x) for x in lines[i + 2].split()[1:]]
        out[z] = (etot, eig)
        i += 3
    return out


def cases(checkout):
    cs = {}
    for name, p in json.load(open(os.path.join(PORT, "atom.cases.json"))).items():
        cs[name] = dict(DEFAULTS, **p)
    lda_n, rlda_n = data8(checkout, "lda"), data8(checkout, "rlda")
    for z in NIST_Z:
        cs["lda_%d" % z] = dict(DEFAULTS, z=z, n=lda_n[z], **LDA_MESH)
        cs["rlda_%d" % z] = dict(DEFAULTS, z=z, n=rlda_n[z], relativistic=True, **RLDA_MESH)
    return cs


def run(exe, p, reps=1):
    args = [exe, str(p["z"]), "1" if p["relativistic"] else "0", str(p["n"]), repr(p["r_min"]),
            repr(p["r_max"]), repr(p["a"]), repr(p["c"]), repr(p["reigen_eps"]), repr(p["mixing_eps"]),
            repr(p["mixing_alpha"]), str(p["mixing_max_iter"]), str(reps), "1" if p["perturb"] else "0"]
    out = subprocess.run(args, check=True, capture_output=True, text=True).stdout
    r = {"orbitals": [], "density": []}
    for line in out.splitlines():
        w = line.split()
        if w[0] == "orbital":
            n, l, so = int(w[1]), int(w[2]), int(w[3])
            kappa = 0 if not p["relativistic"] else (-l - 1 if so == 1 else l)
            r["orbitals"].append([n, l, kappa, float(w[4]), float(w[5])])
        elif w[0] == "density":
            r["density"].append([int(w[1]), float(w[2]), float(w[3])])
        elif w[0] == "iterations":
            r["iterations"] = int(w[1])
        else:
            r[w[0]] = float(w[1])
    return r


def main():
    checkout = os.path.abspath(sys.argv[1])
    exact = os.path.join(checkout, "oracle_vmath")
    strict = os.path.join(checkout, "oracle_strict")
    fast = os.path.join(checkout, "oracle_fast")
    ref = {
        "upstream": "https://github.com/certik/dftatom",
        "commit": subprocess.run(["git", "-C", checkout, "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip(),
        "oracle": "upstream: oracle_vmath, gfortran -O2 -fno-fast-math -ffp-contract=off with Viv's libm; upstream_libm: the same with the platform's libm",
        "timing": "seconds per SCF solve with oracle_fast (upstream's -O3 -march=native -ffast-math), best of 3",
        "gfortran": subprocess.run(["gfortran", "--version"], capture_output=True,
                                   text=True).stdout.splitlines()[0],
        "upstream_converged": {
            "lda": {str(z): v for z, v in upstream_converged(checkout, "lda", "nonrel_energies.dat").items()},
            "rlda": {str(z): v for z, v in upstream_converged(checkout, "rlda", "rel_energies.dat").items()},
        },
        "cases": {},
    }
    for name, p in cases(checkout).items():
        r = run(exact, p)
        r.pop("seconds")
        s = run(strict, p)
        r["seconds_fast"] = min(run(fast, p)["seconds"] for _ in range(3))
        r["seconds_strict"] = s.pop("seconds")
        ref["cases"][name] = {"params": p, "upstream": r, "upstream_libm": s}
        print("%-12s %3d iterations  E_tot %.10f  %.3f s" % (name, r["iterations"], r["etot"], r["seconds_fast"]))
    with open(os.path.join(HERE, "upstream_reference.json"), "w") as f:
        json.dump(ref, f, indent=1)


if __name__ == "__main__":
    main()
