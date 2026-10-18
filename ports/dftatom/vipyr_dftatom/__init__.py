# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2006-2012 Ondřej Čertík, Jiří Vackář, John Pask
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vipyr-dftatom: dftatom's self-consistent atom, ported to Viv.

dftatom is by Ondrej Certik, John E. Pask and Jiri Vackar
(https://github.com/certik/dftatom, MIT). If you use it, cite
O. Certik, J. E. Pask and J. Vackar, "dftatom: A robust and general
Schrodinger and Dirac solver for atomic structure calculations",
Comput. Phys. Commun. 184, 1777 (2013), doi:10.1016/j.cpc.2013.02.014.

    import vipyr as vp
    r = vp.dftatom.atom(Z=26)                       # iron, LDA
    r = vp.dftatom.atom(Z=92, relativistic=True,    # uranium, Dirac RLDA,
                        r_min=1e-8, a=6.2e7, N=18082)   # upstream's mesh
    r.etot, r.ekin, r.ecoul, r.eenuc, r.exc         # Hartree
    for o in r.orbitals: print(o.n, o.l, o.kappa, o.occupation, o.energy)
    r.mesh, r.density                               # Bohr, electrons / Bohr^3

The same module is `import vipyr_dftatom` without vipyr's entry points.
Every keyword is checked against the bounds in ports/dftatom/atom.export.toml
before anything runs; a keyword left out keeps the program's value
(iron, nonrelativistic, upstream's LDA test mesh with N = 6000).
"""
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np

__all__ = ["atom", "AtomResult", "Orbital", "CITATION"]
__version__ = "0.0.1"

CITATION = ("O. Certik, J. E. Pask and J. Vackar, dftatom: A robust and general Schrodinger and Dirac "
            "solver for atomic structure calculations, Comput. Phys. Commun. 184, 1777 (2013)")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROGRAM = os.path.join(os.path.dirname(_HERE), "atom.viv")
if not os.path.isfile(_PROGRAM):            # an installed wheel: the sources are in _root
    _PROGRAM = os.path.join(_HERE, "_root", "atom.viv")
# Python's spelling of the program's parameters: the export lowers names.
_ALIASES = {"Z": "z", "N": "n"}
_mod = None


@dataclass(frozen=True)
class Orbital:
    """One Kohn-Sham orbital. kappa is Dirac's (-l-1 for j = l+1/2, l for
    j = l-1/2) and 0 in a nonrelativistic run; energy in Hartree."""
    n: int
    l: int
    kappa: int
    occupation: float
    energy: float


@dataclass(frozen=True)
class AtomResult:
    """A self-consistent atom. Energies in Hartree, the mesh in Bohr, the
    density in electrons per cubic Bohr, the potential in Hartree."""
    Z: int
    relativistic: bool
    etot: float
    ekin: float
    ecoul: float
    eenuc: float
    exc: float
    orbitals: List[Orbital]
    mesh: np.ndarray
    density: np.ndarray
    v_tot: np.ndarray
    charge: float
    converged: bool
    iterations: int
    tests: list = field(repr=False)

    def check(self):
        """The program's own tests on this run: [(name, passed, message)]."""
        failed = [t for t in self.tests if not t[1]]
        if failed:
            raise AssertionError("; ".join("%s: %s" % (t[0], t[2]) for t in failed))
        return True


def _load():
    global _mod
    if _mod is None:
        import vipyr
        _mod = vipyr.load(_PROGRAM)
    return _mod


def atom(**kwargs) -> AtomResult:
    """One atom in LDA (or, with relativistic=True, Dirac RLDA). Keywords:
    Z, relativistic, r_min, r_max, a, N, c, reigen_eps, reigen_max_iter,
    mixing_eps, mixing_alpha, mixing_max_iter, perturb (atom.viv's `#:`
    lines say what each one is)."""
    kw = {_ALIASES.get(k, k): v for k, v in kwargs.items()}
    r = _load().run(**kw)
    z = int(kw.get("z", 26))
    orbs = [Orbital(int(n), int(l), int(k), float(f), float(e)) for n, l, k, f, e in
            zip(r["orbital_n"], r["orbital_l"], r["orbital_kappa"], r["occupation"], r["orbital_energies"])]
    return AtomResult(Z=z, relativistic=bool(kw.get("relativistic", False)), etot=r["etot"], ekin=r["ekin"],
                      ecoul=r["ecoul"], eenuc=r["eenuc"], exc=r["exc"], orbitals=orbs,
                      mesh=np.asarray(r["mesh"]), density=np.asarray(r["density"]), v_tot=np.asarray(r["v_tot"]),
                      charge=r["charge"], converged=bool(r["scf_converged"]), iterations=int(r["scf_iterations"]),
                      tests=list(r["tests"]))
