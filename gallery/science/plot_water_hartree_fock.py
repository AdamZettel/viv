# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
Water Hartree-Fock energies compared with PySCF
===============================================

Run restricted Hartree-Fock on water in five Gaussian basis sets with
`vp.scf.run`, the Viv port of PySCF's SCF, and compare every total energy and
orbital energy with PySCF 2.5.0's own numbers for the same molecule. The
energy falls toward the Hartree-Fock limit as the basis grows. At every size
the total energy agrees with PySCF to a few times 1e-13 hartree, and the
orbital energies to about 2e-9, the size the orbital-gradient tolerance
allows.
"""

# %%
# The PySCF numbers come from `study/ports/pyscf/oracle/`, written by running
# PySCF 2.5.0 itself (`oracle/make_oracle.sh`), so this page does not need
# PySCF installed. The geometry is PySCF's water example, in angstrom.

import json
import numpy as np
import vipyr as vp
from vipyr.figure import figure

bases = ["sto-3g", "6-31g", "6-31g*", "cc-pvdz", "cc-pvtz"]
oracle_file = {
    "sto-3g": "water_sto3g",
    "6-31g": "water_631g",
    "6-31g*": "water1_631gs",
    "cc-pvdz": "water_ccpvdz",
    "cc-pvtz": "water_ccpvtz",
}
pyscf = {
    b: json.load(open("study/ports/pyscf/oracle/%s.json" % oracle_file[b]))
    for b in bases
}
print(pyscf["sto-3g"]["atom"])

# %%
# ## Total energy in five basis sets
#
# Each call compiles the program once (cached), runs the SCF to an energy
# change below 1e-12 and an orbital gradient below 1e-9, and returns the
# energy in hartree.

runs = {b: vp.scf.run(basis=b) for b in bases}
print(
    "%-8s %5s %22s %22s %9s"
    % ("basis", "AOs", "vipyr E (hartree)", "PySCF E (hartree)", "|diff|")
)
for b in bases:
    e, ref = runs[b].e_tot, float(pyscf[b]["e_tot"])
    print(
        "%-8s %5d %22.12f %22.12f %9.1e"
        % (b, pyscf[b]["nao"], e, ref, abs(e - ref))
    )
print("all converged:", all(r.converged for r in runs.values()))

fig = figure(
    "Water RHF energy against basis size", "basis functions", "E (hartree)"
)
nao = [pyscf[b]["nao"] for b in bases]
fig.plot(nao, [runs[b].e_tot for b in bases], "vipyr")
fig.plot(nao, [float(pyscf[b]["e_tot"]) for b in bases], "PySCF 2.5.0")
fig.style("vipyr", "s1 circle")
fig.style("PySCF 2.5.0", "s2 dash ring")

# %%
# ## Orbital energies
#
# The energy is one number; the orbital energies are all of them. The worst
# difference from PySCF over every occupied and virtual orbital, per basis.
# vipyr stops when the orbital gradient is below 1e-9, and PySCF's reference
# run stopped below 1e-8. An orbital energy carries an error about that size.
# The total energy is stationary at the solution, so its error is of the order
# of the gradient squared, and rounding is what limits the agreement instead.

worst = []
for b in bases:
    ref = np.array([float(x) for x in pyscf[b]["mo_energy"]])
    d = np.max(np.abs(np.asarray(runs[b].mo_energy) - ref))
    worst.append(d)
    print("%-8s %3d orbitals   worst |diff| %.1e hartree" % (b, len(ref), d))

fig = figure(
    "Largest orbital-energy difference from PySCF",
    "basis functions",
    "hartree",
)
fig.plot(nao, worst, "max |vipyr - PySCF|")
fig.plot(
    nao,
    [abs(runs[b].e_tot - float(pyscf[b]["e_tot"])) for b in bases],
    "|E difference|",
)
fig.style("max |vipyr - PySCF|", "s1 circle")
fig.style("|E difference|", "s2 square")
fig.yscale("log")

# %%
# ## check() on each run
#
# `check()` runs the program's own tests on the result.

for b in bases:
    rep = runs[b].check()
    print("%-8s passed=%s  (%d tests)" % (b, rep.passed, len(runs[b].tests)))
