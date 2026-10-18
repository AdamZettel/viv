# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Run study/ports/pyscf/vivscf.viv natively with parameters, until `viv export` exists.

    from vivscf_native import run
    r = run(atoms=[8, 1, 1], coords=[[0, 0, 0], [0, -0.757, 0.587], [0, 0.757, 0.587]],
            basis="6-31g", method="rhf")
    r["e_tot"], r["mo_energy"], r["dm"]

`run` has the signature the generated wrapper of vivscf.export.toml will have:
keyword-only parameters, `None` keeps
the program's default, enums by their manifest labels, and a dict of results
in the manifest's names, plus `tests` (name, passed) and `log` (the printed
lines).

How it gets the parameters into the program: `viv build --set` replaces a
top-level `NAME = <number>` only (an array, an enum value, a Bool or a string
is refused: "the value must be a number"), and it is a compile-time setting
in any case. So this adapter does what --set does, for every type: it copies
vivscf.viv with each given parameter's line rewritten (each line must be
found exactly once, as study/vivkernels/pwdft_m1/assemble.py requires),
writes the copy under target/vivscf/ (the module search starts at the
program's directory, so the copy has to stay inside the repository), and
builds it with `viv build`, whose object cache makes every build after the
first one compile the program's own unit only. A build is reused when the
rewritten text is the same.

VIV names the compiler (default: target/release/viv of this repository).
"""
import hashlib
import os
import re
import subprocess
import time

import numpy

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
PROGRAM = os.path.join(HERE, "vivscf.viv")
BUILD = os.path.join(REPO, "target", "vivscf")

# the manifest's enum labels -> the Viv variants (vivscf.export.toml)
BASIS = {"sto-3g": "Sto3g", "6-31g": "Pople631g", "6-31g*": "Pople631gs", "cc-pvdz": "CcPvdz", "cc-pvtz": "CcPvtz"}
METHOD = {"rhf": "Rhf", "uhf": "Uhf"}
JK = {"auto": "Auto", "incore": "Incore", "direct": "Direct"}

PARAMS = ["atoms", "coords", "basis", "charge", "spin", "method", "jk", "e_tol", "grad_tol",
          "max_iter", "diis_size", "diis_start", "print_arrays"]


def basis_key(name):
    """PySCF's spelling rule: lower case, no `-`, `_` or spaces."""
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def _num(x):
    x = float(x)
    if not numpy.isfinite(x):
        raise ValueError(f"not a finite number: {x!r}")
    s = repr(x)
    return s if ("." in s or "e" in s or "n" in s) else s + ".0"


def _int(x, name):
    if isinstance(x, bool) or int(x) != x:
        raise TypeError(f"{name} must be an integer, got {x!r}")
    return str(int(x))


def _label(table, value, name):
    keys = {basis_key(k): v for k, v in table.items()}
    v = keys.get(basis_key(str(value)))
    if v is None:
        raise ValueError(f"{name} {value!r}: one of {', '.join(table)}")
    return v


def literal(name, value):
    """The Viv text of parameter `name` = `value`."""
    if name == "atoms":
        z = numpy.asarray(value)
        if z.ndim != 1 or len(z) == 0:
            raise ValueError("atoms: a non-empty list of atomic numbers")
        return "[" + ", ".join(_int(a, "atoms") for a in z.tolist()) + "]"
    if name == "coords":
        c = numpy.asarray(value, dtype=float)
        if c.ndim != 2 or c.shape[1] != 3:
            raise ValueError(f"coords: shape (natm, 3), got {c.shape}")
        return "[" + ", ".join("[" + ", ".join(_num(x) for x in row) + "]" for row in c.tolist()) + "]"
    if name == "basis":
        return "BasisSet." + _label(BASIS, value, "basis")
    if name == "method":
        return "Method." + _label(METHOD, value, "method")
    if name == "jk":
        return "JkMode." + _label(JK, value, "jk")
    if name in ("charge", "spin", "max_iter", "diis_size", "diis_start"):
        return _int(value, name)
    if name == "print_arrays":
        return "true" if value else "false"
    return _num(value)


def program_text(params):
    """vivscf.viv with the given parameters' lines rewritten."""
    lines = open(PROGRAM).read().split("\n")
    for name, value in params.items():
        pat = re.compile(r"^(" + re.escape(name) + r"(?:: *[^=]+?)?) = (.*?)(\s+#:.*)?$")
        hits = [i for i, l in enumerate(lines) if pat.match(l)]
        if len(hits) != 1:
            raise RuntimeError(f"vivscf.viv: {len(hits)} top-level lines set {name!r}; the adapter needs exactly one")
        m = pat.match(lines[hits[0]])
        lines[hits[0]] = f"{m.group(1)} = {literal(name, value)}" + (m.group(3) or "")
    return "\n".join(lines)


def viv_bin():
    return os.environ.get("VIV", os.path.join(REPO, "target", "release", "viv"))


def build(params, mode=()):
    """The executable of vivscf.viv with `params`; built once per text."""
    text = program_text(params)
    tag = hashlib.sha256((text + "\0" + " ".join(mode)).encode()).hexdigest()[:16]
    os.makedirs(BUILD, exist_ok=True)
    src = os.path.join(BUILD, f"vivscf_{tag}.viv")
    exe = os.path.join(BUILD, f"vivscf_{tag}")
    if not os.path.exists(exe):
        with open(src, "w") as f:
            f.write(text)
        env = dict(os.environ, VIV_JOBS=os.environ.get("VIV_JOBS", "2"))
        r = subprocess.run([viv_bin(), "build", *mode, src, "-o", exe], cwd=REPO, capture_output=True, text=True, env=env)
        if r.returncode:
            raise RuntimeError("viv build: " + (r.stdout + r.stderr)[-2000:])
    return exe


class VivError(RuntimeError):
    pass


def parse(stdout):
    """The result dict from the program's printed lines."""
    vals, flat, tests, log = {}, {}, [], []
    for line in stdout.splitlines():
        if line.startswith("__"):
            continue
        log.append(line)
        if line.startswith("PASS ") or line.startswith("FAIL "):
            tests.append((line[5:], line.startswith("PASS")))
            continue
        key, _, rest = line.partition(" ")
        if key in ("e_tot", "e_nuc"):
            vals[key] = float(rest)
        elif key in ("iterations", "nao"):
            vals[key] = int(rest)
        elif key in ("converged", "direct"):
            vals[key] = rest.strip() == "true"
        elif key == "basis":
            vals[key] = rest.strip()
        elif key in ("mo_energy_shape", "mo_energy", "mo_coeff", "dm"):
            flat[key] = numpy.array([float(x) for x in rest.split()]) if rest.strip() else numpy.zeros(0)
    if "mo_energy" in flat:
        shp = tuple(int(x) for x in flat["mo_energy_shape"])
        nmo, nao = shp[-1], vals["nao"]
        lead = shp[:-1]
        vals["mo_energy"] = flat["mo_energy"].reshape(shp)
        vals["mo_coeff"] = flat["mo_coeff"].reshape(lead + (nao, nmo))
        vals["dm"] = flat["dm"].reshape(lead + (nao, nao))
    vals["tests"] = tests
    vals["log"] = log
    return vals


def run(*, atoms=None, coords=None, basis=None, charge=None, spin=None, method=None, jk=None,
        e_tol=None, grad_tol=None, max_iter=None, diis_size=None, diis_start=None,
        arrays=True, mode=()):
    """Hartree-Fock on the molecule of `atoms` (atomic numbers) at `coords`
    (angstrom, natm x 3) in `basis`; None keeps vivscf.viv's default. The
    result: e_tot, e_nuc, mo_energy, mo_coeff, dm, converged, iterations (and
    nao, direct, tests, log, wall). `arrays=False` skips printing the arrays
    (for timing)."""
    given = {k: v for k, v in dict(atoms=atoms, coords=coords, basis=basis, charge=charge, spin=spin,
                                    method=method, jk=jk, e_tol=e_tol, grad_tol=grad_tol, max_iter=max_iter,
                                    diis_size=diis_size, diis_start=diis_start).items() if v is not None}
    if (atoms is None) != (coords is None):
        raise ValueError("atoms and coords are given together")
    given["print_arrays"] = bool(arrays)
    exe = build(given, mode)
    t0 = time.perf_counter()
    r = subprocess.run([exe], capture_output=True, text=True)
    wall = time.perf_counter() - t0
    out = parse(r.stdout)
    # a failed test ends the process with status 1 after the results are
    # printed; anything else that stops it is an error
    if r.returncode and ("e_tot" not in out or all(ok for _, ok in out["tests"])):
        raise VivError((r.stderr or r.stdout)[-2000:])
    out["wall"] = wall
    return out
