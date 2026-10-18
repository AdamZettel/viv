# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
Lanczos and dense diagonalization in the sd shell
=================================================

`vp.nuclear.shell_model` finds the lowest states with Lanczos. The program
solves that step with the contract `lowest_eigenpairs` (lib/eigen/lowest),
whose other methods form the Hamiltonian as a matrix,
one column per basis state, and diagonalize all of it: `householder_ql`
by Householder reduction and QL, `jacobi` by cyclic Jacobi. Lanczos and
Householder + QL are run here on the same nuclei; their energies agree to within 1e-12 MeV, and the cost
of each is read from the contract's recorded sweep.
"""

# %%
# ## Swapping the eigensolver
#
# `eigensolver="householder_ql"` picks that method; leaving it out runs the
# default, Lanczos. The two share everything outside the contract,
# including `apply_h`, so their agreement checks the eigensolvers; the
# comparison with BIGSTICK below checks the Hamiltonian.

import json
import numpy as np
import vipyr as vp
from vipyr.figure import figure

nuclei = {
    "c12": ("C-12", "ckpot", 4, 4),
    "b10": ("B-10", "ckpot", 3, 3),
    "o20": ("O-20", "usdb", 0, 4),
    "f20": ("F-20", "usdb", 1, 3),
    "ne20": ("Ne-20", "usdb", 2, 2),
}
ref = json.load(open("lib/nuclear/shell/bigstick_reference.json"))
runs = {}
for key, (name, inter, z, n) in nuclei.items():
    lz = vp.nuclear.shell_model(interaction=inter, z=z, n=n, nstates=10)
    dn = vp.nuclear.shell_model(
        interaction=inter, z=z, n=n, nstates=10, eigensolver="householder_ql"
    )
    runs[key] = (lz, dn)
    gap = np.max(np.abs(np.asarray(lz.energies) - np.asarray(dn.energies)))
    line = "%-6s dimension %4d: Lanczos %3d products, Householder + QL %4d; " % (
        name, lz.dim, lz.iterations, dn.iterations
    )
    line += "energies agree to below 1e-12 MeV: %s" % (gap < 1e-12)
    if key in ref:
        b = np.asarray(ref[key]["energies"][: len(lz.energies)])
        line += "; |E - E_BIGSTICK| < 1e-5 MeV: %s" % (
            np.max(np.abs(np.asarray(dn.energies) - b)) < 1e-5
        )
    print(line)

# %%
# ## Every method at once
#
# `eigensolver="every"` runs both and stops the program if any
# energy differs by more than `tol`, the residual bound Lanczos converges to.

r = vp.nuclear.shell_model(z=2, n=2, nstates=5, eigensolver="every")
print("every: Ne-20 lowest energies", np.round(r.energies, 6))

# %%
# ## Cost against the basis dimension
#
# `lib/nuclear/shell/eigensolver_sweep.py` timed both on nine nuclei and
# wrote `lowest_eigenpairs_sweep.json`; the times are wall times of the
# whole run.
# Lanczos takes one product H x per step and needs 50 to a few hundred
# steps here, so its cost grows about like the dimension. Householder + QL
# takes one product per basis state and then an O(dim^3) diagonalization:
# it is faster at 51 to 84 states and slower from 394 on, and at 1,935 states
# nearly all of its time is the QL step. The program refuses a matrix above
# 8,000 states (a 512 MB matrix); magnesium-24's 28,503 would need 6.5 GB.

sweep = json.load(open("lib/nuclear/shell/lowest_eigenpairs_sweep.json"))
fig = figure(
    "Shell model: time against basis dimension", "dimension", "seconds"
)
fig.xscale("log")
fig.yscale("log")
for alt in ("lanczos", "householder_ql"):
    rows = [r for r in sweep["rows"] if alt in r]
    pts = sorted((r["dim"], r[alt]["seconds"]) for r in rows)
    fig.plot([p[0] for p in pts], [p[1] for p in pts], alt)
for row in sweep["rows"]:
    d = row.get("householder_ql", {}).get("seconds")
    print(
        "%-5s dim %6d   lanczos %8.3f s   householder_ql %s"
        % (row["case"], row["dim"], row["lanczos"]["seconds"],
           "%8.3f s" % d if d else "       -")
    )
checked = [a.check().passed and b.check().passed for a, b in runs.values()]
print("check():", all(checked))
