# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Convert a proton-neutron interaction (.snt, KSHELL's layout) into the
isospin layout BIGSTICK and lib/nuclear/shell read (.int with a .sps "iso").

    python3 lib/nuclear/shell/data/snt_to_iso.py ckpot.snt > ckpot.int

Used once, for ckpot.int: the Cohen-Kurath (8-16)POT p-shell interaction,
S. Cohen and D. Kurath, Nucl. Phys. 73, 1 (1965). Only the numbers of the
.snt file are read; no KSHELL code is used.

The relation between the two layouts, for normalized antisymmetric isospin
matrix elements V_JT(ab, cd) and the pn element with a, c proton orbits and
b, d neutron orbits:

    V_pn(ab, cd; J) = sqrt((1 + d_ab)(1 + d_cd)) / 2 * (V_J1(ab, cd) + V_J0(ab, cd))

T = 1 comes from the proton-proton block, T = 0 from the formula above. The
exchange phase |ba; JT> = (-1)^(ja + jb - J + T) |ab; JT> brings every
element to a <= b, c <= d, (a, b) <= (c, d). Elements that reach the same
canonical slot must agree, and the script stops if they do not.
"""
import sys
from math import sqrt


def read_snt(path):
    rows = [ln.split("!")[0].split() for ln in open(path)]
    rows = [r for r in rows if r]
    nprot, nneut = int(rows[0][0]), int(rows[0][1])
    norb = nprot + nneut
    orbs = {}
    for r in rows[1:1 + norb]:
        k, n, l, j2, tz = map(int, r)
        orbs[k] = (n, l, j2, tz)
    at = 1 + norb
    nspe = int(rows[at][0])
    spe = {int(r[0]): float(r[2]) for r in rows[at + 1:at + 1 + nspe]}
    at += 1 + nspe
    nme, mass = int(rows[at][0]), int(rows[at][1])
    if mass != 0:
        raise SystemExit("mass-dependent .snt files are not handled")
    tb = [(int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4]), float(r[5]))
          for r in rows[at + 1:at + 1 + nme]]
    return nprot, orbs, spe, tb


def main(path):
    nprot, orbs, spe, tb = read_snt(path)
    iso = lambda k: k if k <= nprot else k - nprot
    j2 = {iso(k): v[2] for k, v in orbs.items()}

    def phase(a, b, J, T):
        return -1.0 if ((j2[a] + j2[b]) // 2 - J + T) % 2 else 1.0

    def canon(a, b, c, d, J, T):
        s = 1.0
        if a > b:
            s *= phase(a, b, J, T)
            a, b = b, a
        if c > d:
            s *= phase(c, d, J, T)
            c, d = d, c
        if (a, b) > (c, d):
            a, b, c, d = c, d, a, b
        return (a, b, c, d, J, T), s

    V = {}

    def put(key, val):
        k, s = canon(*key)
        if k in V and abs(V[k] - s * val) > 1e-9:
            raise SystemExit(f"inconsistent element {k}: {V[k]} vs {s * val}")
        V[k] = s * val

    def get1(a, b, c, d, J):
        k, s = canon(a, b, c, d, J, 1)
        return s * V.get(k, 0.0)

    for a, b, c, d, J, v in tb:
        if a <= nprot and b <= nprot:
            put((a, b, c, d, J, 1), v)
    for a, b, c, d, J, v in tb:
        if a <= nprot and b > nprot:
            ia, ib, ic, id_ = iso(a), iso(b), iso(c), iso(d)
            norm = sqrt((1 + (ia == ib)) * (1 + (ic == id_))) / 2.0
            v0 = v / norm - get1(ia, ib, ic, id_, J)
            if (ia == ib and phase(ia, ib, J, 0) < 0) or (ic == id_ and phase(ic, id_, J, 0) < 0):
                # (aa; J T=0) is forbidden by the Pauli principle; what is left
                # is the rounding of the five-decimal .snt values
                if abs(v0) > 1e-5:
                    raise SystemExit(f"forbidden T=0 element is {v0}")
                continue
            put((ia, ib, ic, id_, J, 0), v0)
    nis = len({iso(k) for k in orbs})
    spes = [spe[k] for k in sorted(spe) if k <= nprot]
    print("! Cohen-Kurath (8-16)POT p-shell interaction, isospin form")
    print("! S. Cohen and D. Kurath, Nucl. Phys. 73, 1 (1965)")
    print("! converted from the proton-neutron .snt by lib/nuclear/shell/data/snt_to_iso.py")
    print("! orbits: " + ", ".join(f"{k}=p{j2[k]}/2" for k in range(1, nis + 1)))
    print(f"{len(V):3d} " + " ".join(f"{e:9.5f}" for e in spes) + "   4.0000   4.0000 0.0000000")
    for k in sorted(V, key=lambda k: (k[5], k[4], k[:4])):
        a, b, c, d, J, T = k
        print(f"{a:3d}{b:3d}{c:3d}{d:3d}{J:5d}{T:3d} {V[k]:12.7f}")


if __name__ == "__main__":
    main(sys.argv[1])
