# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2006-2012 Ondřej Čertík, Jiří Vackář, John Pask
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Writes ports/dftatom/oracle/nist_reference.json: NIST's LDA and RLDA
numbers for the atoms the comparison uses.

    python3 ports/dftatom/oracle/nist.py CACHE_DIR

NIST Standard Reference Database 141, Atomic Reference Data for Electronic
Structure Calculations (S. Kotochigova, Z. H. Levine, E. L. Shirley, M. D.
Stiles and C. W. Clark, Phys. Rev. A 55, 191 (1997);
https://math.nist.gov/DFTdata/atomdata/tables/ptable.html). CACHE_DIR holds
the element pages, 26Fe.html and so on; a missing one is fetched once.

Each page lists LDA, LSD, RLDA and ScRLDA side by side. An orbital row reads
`2p  LDA  LSD(up)  RLDA  ScRLDA` and the line under it `LSD(down)` and, for
l > 0, the RLDA eigenvalue of the second fine-structure level. The first
RLDA value is j = l - 1/2, the second j = l + 1/2: the order dftatom lists
them in.
"""
import html
import json
import os
import re
import sys
import time
import urllib.request

ATOMS = [(1, "H"), (2, "He"), (3, "Li"), (4, "Be"), (5, "B"), (6, "C"), (7, "N"), (8, "O"),
         (9, "F"), (10, "Ne"), (11, "Na"), (12, "Mg"), (13, "Al"), (14, "Si"), (15, "P"),
         (16, "S"), (17, "Cl"), (18, "Ar"), (26, "Fe"), (36, "Kr"), (54, "Xe"), (79, "Au"),
         (92, "U")]


def page(cache, z, sym):
    path = os.path.join(cache, "%02d%s.html" % (z, sym))
    if not os.path.exists(path):
        url = "https://math.nist.gov/DFTdata/atomdata/tables/%02d%s.html" % (z, sym)
        req = urllib.request.Request(url, headers={"User-Agent": "viv-lang reference harvester"})
        with urllib.request.urlopen(req, timeout=30) as r:
            open(path, "wb").write(r.read())
        time.sleep(0.5)
    t = open(path, encoding="utf-8", errors="replace").read()
    t = re.sub(r"<script.*?</script>", "", t, flags=re.S | re.I)
    t = re.sub(r"<[^>]+>", "\n", t)
    return [html.unescape(l).strip() for l in t.split("\n") if l.strip()]


def parse(lines, z, sym):
    start = next(i for i, l in enumerate(lines) if re.match(r"^%d\s+%s\s" % (z, sym), l))
    stop = next((i for i, l in enumerate(lines[start + 1:], start + 1)
                 if re.match(r"^%d\s+%s\+" % (z, sym), l)), len(lines))
    block = lines[start:stop]
    out = {"lda": {"orbitals": []}, "rlda": {"orbitals": []}}
    i = 0
    while i < len(block):
        line = block[i]
        m = re.match(r"^(Etot|Ekin|Ecoul|Eenuc|Exc)\s*=\s*(.*)$", line)
        if m:
            v = [float(x) for x in m.group(2).split()]
            out["lda"][m.group(1).lower()] = v[0]
            out["rlda"][m.group(1).lower()] = v[2]
        m = re.match(r"^(\d)([spdf])\s+(-?\d.*)$", line)
        if m:
            n, l = int(m.group(1)), "spdf".index(m.group(2))
            v = [float(x) for x in m.group(3).split()]
            cont = [float(x) for x in block[i + 1].split()]
            out["lda"]["orbitals"].append([n, l, 0, v[0]])
            if l == 0:
                out["rlda"]["orbitals"].append([n, l, -1, v[2]])
            else:
                out["rlda"]["orbitals"].append([n, l, l, v[2]])
                out["rlda"]["orbitals"].append([n, l, -l - 1, cont[1]])
            i += 1
        i += 1
    return out


def main():
    cache = sys.argv[1]
    os.makedirs(cache, exist_ok=True)
    ref = {
        "source": "NIST Standard Reference Database 141, https://math.nist.gov/DFTdata/atomdata/tables/ptable.html",
        "units": "Hartree",
        "orbitals": "[n, l, kappa (0 in LDA), eigenvalue], NIST's order",
        "atoms": {},
    }
    for z, sym in ATOMS:
        ref["atoms"][str(z)] = dict(symbol=sym, **parse(page(cache, z, sym), z, sym))
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "nist_reference.json"), "w") as f:
        json.dump(ref, f, indent=1)
    print("wrote", len(ref["atoms"]), "atoms")


if __name__ == "__main__":
    main()
