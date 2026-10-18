# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Write the Viv input deck AND its check file for one of the usual suspects.

    python3 dft/ref/make_deck.py Al     ->  dft/al.viv, dft/al_check.viv

Both are generated: the deck from materials.py plus CP2K's pseudopotential
table, the check file by RUNNING the numpy reference at exactly the deck's
settings and pinning what it prints. So a deck and the numbers it is tested
against cannot drift apart, and nothing in either is typed by hand.
"""
import os, subprocess, sys
import numpy as np
from pwdft_ref import scf, bands, mp_grid, fft_dims
import pwdft_iter
from materials import MATERIALS, build, NO_DENSE_ENGINE, ITERATIVE_REFERENCE

HA = 27.211386245988
HERE = os.path.dirname(os.path.abspath(__file__))
DFT = os.path.dirname(HERE)

NOTES = {
    'Si': "Silicon, diamond structure.",
    'Ge': "Germanium, diamond structure. LDA closes its gap ENTIRELY -- the classic failure -- and it\n# takes a converged cutoff to see it: at silicon's 6 Ha germanium passes for a semiconductor.",
    'GaAs': "Gallium arsenide, zincblende: two species, and NO inversion symmetry, so the\n# density's Fourier coefficients are genuinely complex.",
    'C': "Diamond. A hard first-row pseudopotential: it wants a 20 Ha cutoff where silicon wants 6.",
    'Al': "Aluminum, fcc: a METAL, so occupations are Fermi-Dirac and the Fermi level is solved for.",
    'Cu': "Copper, fcc, with its 3d shell in the valence: 11 electrons, five flat d bands 2-5 eV below\n# the Fermi level, and ~1400 plane waves per k-point -- so the ITERATIVE engine only.",
}

# What each material must get right PHYSICALLY: published LDA values, with a
# tolerance that reflects the spread between papers (pseudopotentials and
# k-meshes differ), not the precision of this code. A failure here would mean
# the METHOD is wrong, where a failure against the numpy reference means a bug.
# Indices: L is point 0, Gamma is GAMMA_P, X is LAST; NV = NELEC / 2.
PHYSICS = {
    'Si': """
NV = NELEC / 2.0
GAP = 1.0e9
KGAP = 0.0
for p to NPTS {
    if EVB[p, NV] < GAP {
        GAP = EVB[p, NV]
        KGAP = p
    }
}
show GAP
test "the valence-band top at Gamma is a TRIPLET: the density has the crystal's full symmetry" {
    assert_near(EVB[GAMMA_P, NV - 1.0] - EVB[GAMMA_P, NV - 3.0], 0.0, 1.0e-4)
}
test "valence band width is about 12 eV" {
    assert_near(EVB[GAMMA_P, 0.0], -11.95, 0.15)
}
test "silicon is an INDIRECT-gap semiconductor, minimum near 0.85 of Gamma-X" {
    assert_lt(GAP, EVB[GAMMA_P, NV])
    assert_near((KGAP - GAMMA_P) / NSEG, 0.85, 0.1)
}
test "the LDA gap is about 0.5 eV -- less than half the measured 1.17 eV" {
    assert_near(GAP, 0.5, 0.1)
    assert_lt(GAP, 0.6 * 1.17)
}
test "zone-boundary valence states: X4 and L3'" {
    assert_near(EVB[LAST, NV - 1.0], -2.87, 0.12)
    assert_near(EVB[0.0, NV - 1.0], -1.2, 0.1)
}
""",
    'Ge': """
NV = NELEC / 2.0
# At Gamma the four levels around the gap are the Gamma25' TRIPLET (the valence-band top)
# and the s-like Gamma2' SINGLET. In LDA the singlet comes out on the wrong side, so find
# them by their degeneracy rather than by their order.
E1 = EVB[GAMMA_P, NV - 3.0]
E2 = EVB[GAMMA_P, NV - 2.0]
E3 = EVB[GAMMA_P, NV - 1.0]
E4 = EVB[GAMMA_P, NV]
#: G2P  the Gamma2' singlet, from the triplet. Measured: +0.89 eV, the direct gap.
G2P = E4
#: TRIP  the spread of the three-fold level: zero if the density has the crystal's symmetry
TRIP = E3 - E1
if E2 - E1 > E4 - E3 {
    G2P = E1
    TRIP = E4 - E2
}
#: LC, XC  the lowest empty state at L and at X
LC = EVB[0.0, NV]
XC = EVB[LAST, NV]
show G2P
show LC
show XC
test "the valence-band top at Gamma is a TRIPLET: the density has the crystal's full symmetry" {
    assert_near(TRIP, 0.0, 1.0e-4)
}
test "valence band width is about 12.7 eV" {
    assert_near(EVB[GAMMA_P, 0.0], -12.75, 0.3)
}
test "LDA's known failure: germanium has NO gap -- Gamma2' lands on the valence-band top (measured: 0.89 eV above it)" {
    assert_near(G2P, 0.0, 0.15)
}
test "the L valley is within 0.3 eV of the valence-band top (measured gap, at L: 0.74 eV) and below X, as in real germanium" {
    assert_gt(LC, 0.0)
    assert_lt(LC, 0.3)
    assert_lt(LC, XC)
}
""",
    'GaAs': """
NV = NELEC / 2.0
GGAP = EVB[GAMMA_P, NV]
LGAP = EVB[0.0, NV]
XGAP = EVB[LAST, NV]
show GGAP
test "the valence-band top at Gamma is a TRIPLET: the density has the crystal's full symmetry" {
    assert_near(EVB[GAMMA_P, NV - 1.0] - EVB[GAMMA_P, NV - 3.0], 0.0, 1.0e-4)
}
test "gallium arsenide is a DIRECT-gap semiconductor: Gamma below L below X" {
    assert_lt(GGAP, LGAP)
    assert_lt(LGAP, XGAP)
}
test "the LDA gap is about 0.45 eV -- under a third of the measured 1.52 eV" {
    assert_near(GGAP, 0.45, 0.2)
    assert_lt(GGAP, 1.52 / 3.0 + 0.1)
}
test "valence band width is about 12.7 eV" {
    assert_near(EVB[GAMMA_P, 0.0], -12.7, 0.3)
}
""",
    'C': """
NV = NELEC / 2.0
GAP = 1.0e9
KGAP = 0.0
for p to NPTS {
    if EVB[p, NV] < GAP {
        GAP = EVB[p, NV]
        KGAP = p
    }
}
show GAP
test "the valence-band top at Gamma is a TRIPLET: the density has the crystal's full symmetry" {
    assert_near(EVB[GAMMA_P, NV - 1.0] - EVB[GAMMA_P, NV - 3.0], 0.0, 1.0e-4)
}
test "diamond's valence band is about 21.5 eV wide" {
    assert_near(EVB[GAMMA_P, 0.0], -21.5, 0.4)
}
test "the direct gap at Gamma is about 5.55 eV" {
    assert_near(EVB[GAMMA_P, NV], 5.55, 0.2)
}
test "the gap is INDIRECT, about 4.0 eV, with its minimum along Gamma-X" {
    assert_near(GAP, 4.05, 0.3)
    assert_gt(KGAP, GAMMA_P)
    assert_lt(KGAP, LAST)
}
""",
    'Al': """
#: EF_FREE  the free-electron Fermi energy for three electrons per cell, in eV
EF_FREE = 0.5 * pow(3.0 * PI * PI * NELEC / OMEGA, 2.0 / 3.0) * HA_EV
BOTTOM = 0.0 - EVB[GAMMA_P, 0.0]
show EF_FREE
show BOTTOM
test "the band bottom is about 11.1 eV below the Fermi level" {
    assert_near(BOTTOM, 11.1, 0.3)
}
test "aluminum is nearly free-electron-like: within 10% of (hbar k_F)^2 / 2m" {
    assert_near(BOTTOM / EF_FREE, 1.0, 0.1)
}
test "the occupied levels at X" {
    assert_near(EVB[LAST, 0.0], -2.95, 0.3)
    assert_near(EVB[LAST, 1.0], -1.55, 0.3)
}
test "it is a metal: bands cross the Fermi level along the path" {
    assert_true(any(EVB[:, 1.0] > 0.0))
    assert_true(any(EVB[:, 1.0] < 0.0))
}
""",
    # Marini, Onida & Del Sole, Phys. Rev. Lett. 88, 016403 (2002), Table I, the
    # DFT-LDA column (and the measured values it is compared with, from Courths &
    # Hufner, Phys. Rep. 112, 53 (1984)). Band order at each point, from below:
    #   Gamma: Gamma1, Gamma25' (x3), Gamma12 (x2)
    #   X:     X1, X3, X2, X5 (x2), then X4' above the Fermi level
    #   L:     L1, L3 (x2), L3 (x2), L2', then L1 above the Fermi level
    'Cu': """
G1 = EVB[GAMMA_P, 0.0]
G25 = EVB[GAMMA_P, 1.0]
G12 = EVB[GAMMA_P, 4.0]
X1 = EVB[LAST, 0.0]
X5 = EVB[LAST, 4.0]
L1 = EVB[0.0, 0.0]
L3 = EVB[0.0, 4.0]
L2P = EVB[0.0, 5.0]
#: LGAP  L1 - L2', the gap above the neck. SHOWN, NOT ASSERTED: it comes out near 4.9 eV
#:       here, against 5.40 in the paper's LDA column (whose pseudopotential keeps the 3s
#:       and 3p shells in the valence; this one does not) and 4.95 measured. Every
#:       OCCUPIED level below agrees with that column to 0.2 eV or better; why this
#:       empty one differs by 0.5 has not been run down, so no tolerance is claimed.
LGAP = EVB[0.0, 6.0] - L2P
show G1
show G12
show X5
show L3
show L2P
show LGAP
test "the s band bottoms out at Gamma1, about 9.8 eV below the Fermi level" {
    assert_near(G1, -9.79, 0.3)
}
test "the d bands sit where LDA puts them: Gamma12, X5 and L3" {
    assert_near(G12, -2.27, 0.25)
    assert_near(X5, -1.40, 0.25)
    assert_near(L3, -1.63, 0.25)
}
test "d-band degeneracies: Gamma25' is a triplet; Gamma12, X5 and both L3 levels are doublets" {
    assert_near(EVB[GAMMA_P, 3.0] - EVB[GAMMA_P, 1.0], 0.0, 1.0e-4)
    assert_near(EVB[GAMMA_P, 5.0] - EVB[GAMMA_P, 4.0], 0.0, 1.0e-4)
    assert_near(EVB[LAST, 4.0] - EVB[LAST, 3.0], 0.0, 1.0e-4)
    assert_near(EVB[0.0, 2.0] - EVB[0.0, 1.0], 0.0, 1.0e-4)
    assert_near(EVB[0.0, 4.0] - EVB[0.0, 3.0], 0.0, 1.0e-4)
}
test "d-band widths: Gamma12 - Gamma25', X5 - X3, X5 - X1, L3 - L3 and L3 - L1" {
    assert_near(G12 - G25, 0.91, 0.15)
    assert_near(X5 - EVB[LAST, 1.0], 3.23, 0.15)
    assert_near(X5 - X1, 3.70, 0.25)
    assert_near(L3 - EVB[0.0, 1.0], 1.58, 0.15)
    assert_near(L3 - L1, 3.72, 0.25)
}
test "LDA's known failure: the d bands are too shallow (X5 is measured at -2.01 eV) and too wide (X5 - X1 is measured at 3.17 eV)" {
    assert_gt(X5, -2.01 + 0.3)
    assert_gt(X5 - X1, 3.17 + 0.3)
}
test "the neck of the Fermi surface: L2' is OCCUPIED, about 1.1 eV below the Fermi level" {
    assert_near(L2P, -1.12, 0.3)
}
test "it is a metal: the sixth band is empty at X and occupied at L" {
    assert_gt(EVB[LAST, 5.0], 0.5)
    assert_lt(L2P, -0.5)
}
""",
}


def deck(name):
    a, atoms, ecut, nk, smear, nb = MATERIALS[name]
    cr, _, _, _, _, names = build(name)
    low = name.lower()
    els = [n.split()[0] + ':' + n.split('-')[-1] for n in names]
    pseudo = subprocess.run([sys.executable, os.path.join(DFT, 'pseudo', 'make_pseudo.py')] + els,
                            capture_output=True, text=True, check=True).stdout
    runs = ([] if name in NO_DENSE_ENGINE else [f"#     dft/run.sh {low}         dense engine:     dft/pwdft.viv    + dft/{low}_check.viv"]) + \
           [f"#     dft/run.sh {low} it      iterative engine: dft/pwdft_it.viv + dft/{low}_it_check.viv"]
    L = [f"# {NOTES[name]}", "#"] + runs + [
         "#", "# GENERATED by dft/ref/make_deck.py from dft/ref/materials.py -- do not edit by hand.", "",
         pseudo.rstrip(), "",
         "# --- the crystal: an fcc lattice ---------------------------------------------",
         f"ALAT = {a!r}                          #: ALAT  lattice constant, bohr",
         "A = [[0.0, ALAT / 2.0, ALAT / 2.0],",
         "     [ALAT / 2.0, 0.0, ALAT / 2.0],",
         "     [ALAT / 2.0, ALAT / 2.0, 0.0]]   #: A  primitive vectors, one per ROW",
         f"NAT = {len(atoms)}.0",
         "ATSPEC = [" + ", ".join(f"{names.index(e)}.0" for e, _ in atoms) + "]" + " " * 8 + "#: ATSPEC  species index of each atom",
         "ATFRAC = [" + ",\n          ".join("[" + ", ".join(repr(float(x)) for x in f) + "]" for _, f in atoms) + "]   #: ATFRAC  positions, fractions of A",
         "", "# --- the calculation --------------------------------------------------------",
         f"ECUT = {float(ecut)!r}                           #: ECUT  plane-wave cutoff, hartree",
         f"NBANDS = {nb}.0                         #: NBANDS  bands computed at every k-point",
         f"SMEAR = {float(smear)!r}                          #: SMEAR  Fermi-Dirac width, hartree; 0 for an insulator",
         f"NKMESH = {nk}.0                         #: NKMESH  the k-mesh is NKMESH^3, shifted (dft/kmesh.viv)",
         "MIX = 0.6                             #: MIX  the damping in the Anderson mixer (dft/pw_common.viv)",
         "SCF_TOL = 1.0e-9", "MAXIT = 80.0",
         "NITER = 3.0                           #: NITER  LOBPCG steps per k-point per SCF iteration (iterative engine)",
         "NITER_BANDS = 60.0                    #: NITER_BANDS  at most this many LOBPCG steps per band-path point, from a cold start",
         "EIG_TOL = 1.0e-5                      #: EIG_TOL  residual norm |H psi - eps psi| every band that matters must reach", "",
         "# --- the band path: L -> Gamma -> X ------------------------------------------",
         "NCORNER = 3.0",
         "CORNER = [[0.5, 0.5, 0.5],", "          [0.0, 0.0, 0.0],", "          [0.5, 0.0, 0.5]]",
         f"NSEG = {NSEG}.0                           #: NSEG  points per path segment; the figure has 2 NSEG + 1", ""]
    return "\n".join(L)

NSEG = 30      # points per segment of the L - Gamma - X path: 61 in all

TITLES = {'Si': 'Silicon', 'Ge': 'Germanium', 'GaAs': 'Gallium arsenide', 'C': 'Diamond', 'Al': 'Aluminum', 'Cu': 'Copper'}


def path_points(nseg):
    corners = np.array([[.5, .5, .5], [0, 0, 0], [.5, 0, .5]])
    return [corners[c] + (corners[c + 1] - corners[c]) * t / nseg for c in range(2) for t in range(nseg)] + [corners[-1]]


def cached_reference(name, nseg=NSEG, log=None):
    """reference(), remembered. The key is the material's settings plus the text
    of both numpy implementations, so editing a test's wording reuses the numbers
    and editing anything that could change them does not."""
    import hashlib, json
    src = "".join(open(os.path.join(HERE, f)).read() for f in ("pwdft_ref.py", "pwdft_iter.py", "materials.py"))
    key = hashlib.sha256((repr(MATERIALS[name]) + repr(nseg) + src).encode()).hexdigest()
    path = os.path.join(HERE, ".cache", f"{name}.json")
    if os.path.exists(path):
        c = json.load(open(path))
        if c["key"] == key:
            return c["out"], np.array(c["ev"]), c["nv"], c["nseg"], c["which"]
    out, ev, nv, nseg, which = reference(name, nseg, log)
    slim = {"etot": float(out["etot"]), "mu": None if out["mu"] is None else float(out["mu"]), "iters": int(out["iters"])}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"key": key, "out": slim, "ev": ev.tolist(), "nv": int(nv), "nseg": nseg, "which": which}, open(path, "w"))
    return slim, ev, nv, nseg, which


def reference(name, nseg=NSEG, log=None):
    """Run the numpy reference at the deck's settings: the DENSE one where it is
    affordable, the iterative one where it is not. Returns the SCF result, the
    bands along the path in eV from the reference energy, and which was run."""
    cr, ecut, nk, smear, nb, _ = build(name)
    ks, w = mp_grid((nk, nk, nk))
    pts = path_points(nseg)
    # one grid for the mesh AND the path, as pw_common.viv sizes it
    dims = fft_dims(cr, ecut, list(ks) + pts)
    if name in ITERATIVE_REFERENCE:
        out = pwdft_iter.scf(cr, ecut, ks, w, nb, dims, smear=smear, mix=0.6, anderson=3, nit=3, verbose=log is not None)
        E = np.array([pwdft_iter.bands_at(cr, ecut, out['vr'], dims, k, nb, nit=60)[0] for k in pts])
        which = 'iterative'
    else:
        out = scf(cr, ecut, ks, w, nbands=nb, smear=smear, verbose=False, maxit=80, mix=0.6, anderson=3, dims=dims)
        E = np.array([bands(cr, ecut, out['vtot_g'], out['dims'], k, nb) for k in pts])
        which = 'dense'
    nv = cr.nelec // 2
    eref = out['mu'] if smear > 0 else E[:, nv - 1].max()
    return out, (E - eref) * HA, nv, nseg, which


def check_file(name, ref, engine):
    """The check file for one engine. `engine` is 'dense' or 'it'.

    The dense engine is the numpy reference's twin, step for step, so it is held
    to the SAME iteration count and to 1e-8 Ha. The iterative engine is a
    different algorithm (LOBPCG from random vectors; an eigensolver is free to
    return any basis of a degenerate subspace), so it is held to the converged
    ANSWER: 1e-7 Ha and 1e-5 eV, and to no iteration count."""
    out, ev, nv, nseg, which = ref
    low = name.lower()
    a, atoms, ecut, nk, smear, nb = MATERIALS[name]
    metal = smear > 0
    dense = engine == 'dense'
    etol, btol = ("1.0e-8", "1.0e-6") if dense else ("1.0e-7", "1.0e-5")
    refname = {"dense": "dft/ref/pwdft_ref.py -- an independent numpy implementation that diagonalizes H densely --",
               "iterative": "dft/ref/pwdft_iter.py -- an independent numpy implementation of the same iterative method --"}[which]
    L = [f"# {name}: what the calculation must reproduce, and the picture of it.", "#",
         f"#     dft/run.sh {low}{'' if dense else ' it'}", "#",
         f"# GENERATED by dft/ref/make_deck.py, which RAN {refname}",
         "# at exactly this deck's settings and pinned what it printed.",
         "# A disagreement here is a bug in one of the two implementations.", "",
         "GAMMA_P = NSEG", "LAST = NPTS - 1.0", ""]
    if dense:
        L += ['test "SCF converged, in the same number of iterations as the reference" {',
              "    assert_lt(DRHO, 1.0e-8)", f"    assert_near(ITER, {out['iters']}.0, 0.5)", "}"]
    else:
        L += ['test "SCF converged: density, energy, and every occupied band\'s residual" {',
              "    assert_lt(DRHO, 1.0e-8)", "    assert_lt(EIGRES, EIG_TOL)", "    assert_lt(ITER, MAXIT)", "}",
              'test "every band-path point converged" {', "    assert_lt(BANDRES, EIG_TOL)", "}"]
    L += [f'test "{"free" if metal else "total"} energy agrees with the numpy reference" {{',
          f"    assert_near(ETOT, {out['etot']:.12f}, {etol})", "}"]
    if metal:
        L += ['test "the Fermi level agrees with the numpy reference" {',
              f"    assert_near(EFERMI, {out['mu']:.12f}, {etol})", "}"]
    L += ['test "band energies at L, Gamma and X agree with the numpy reference (eV)" {']
    for label, p in (("L", 0), ("Gamma", nseg), ("X", 2 * nseg)):
        for b in range(min(nb, nv + 2)):
            L.append(f"    assert_near(EVB[{'0.0' if p == 0 else ('GAMMA_P' if p == nseg else 'LAST')}, {b}.0], {ev[p, b]:.9f}, {btol})   # {label}, band {b + 1}")
    L += ["}", ""]
    # ... and then EVERYWHERE: every checked band at every point of the path, so a
    # band missed or mis-ordered between the corners cannot hide.
    ncheck = min(nb, nv + 2)
    rows = ",\n           ".join("[" + ", ".join(f"{ev[p, b]:.7f}" for b in range(ncheck)) + "]" for p in range(ev.shape[0]))
    L += [f"#: REFPATH  the reference's lowest {ncheck} bands at all {ev.shape[0]} path points, eV",
          f"REFPATH = [{rows}]",
          f'test "every band at every point of the path agrees with the numpy reference (eV)" {{',
          f"    assert_lt(max(abs(EVB[:, 0:shape(REFPATH)[1]] - REFPATH)), {btol})", "}", "",
          "# --- the physics: published LDA values, loose on purpose -----------------------",
          PHYSICS[name].strip(), ""]
    zero = "the Fermi level" if metal else "the valence-band maximum"
    L += [f'figure("{TITLES[name]}: LDA band structure", "wavevector along the path", "energy from {zero} (eV)")',
          "for b to NBANDS {", '    plot(PATHX, EVB[:, b], "band")', "}",
          'xtick("L", PATHX[0])', 'xtick("Γ", PATHX[GAMMA_P])', 'xtick("X", PATHX[LAST])', ""]
    return "\n".join(L)


if __name__ == '__main__':
    for name in sys.argv[1:]:
        low = name.lower()
        open(os.path.join(DFT, f"{low}.viv"), 'w').write(deck(name))
        ref = cached_reference(name, log=True)
        wrote = [f"dft/{low}.viv"]
        if name not in NO_DENSE_ENGINE:
            open(os.path.join(DFT, f"{low}_check.viv"), 'w').write(check_file(name, ref, 'dense'))
            wrote.append(f"dft/{low}_check.viv")
        elif os.path.exists(os.path.join(DFT, f"{low}_check.viv")):
            os.remove(os.path.join(DFT, f"{low}_check.viv"))
        open(os.path.join(DFT, f"{low}_it_check.viv"), 'w').write(check_file(name, ref, 'it'))
        wrote.append(f"dft/{low}_it_check.viv")
        print(f"{name}: wrote {', '.join(wrote)}   ({ref[4]} reference: {ref[0]['iters']} iterations, "
              f"E = {ref[0]['etot']:.8f} Ha)", flush=True)
