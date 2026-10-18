# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Write a MATPOWER case as a PSS/E RAW file (revision 33 or 34), for the
tests of lib/power/raw.viv.

The file describes the same network as the .m file in every encoding the
reader handles, so that reading it back must give the case's admittance
matrix and power flow: transformers rotate through the winding codes CW 1,
2, 3 and the impedance codes CZ 1, 2, 3 (one with WINDV2 != 1); a bus
shunt is split between a fixed shunt and the constant-admittance part of
a load; a quarter of each load is written as constant current at the
case's voltage; a transformer's charging is written as its magnetizing
admittance (CM 1) plus a fixed shunt at the other bus.

    python3 lib/power/oracle/write_raw.py case14 33   # -> lib/power/cases/raw/case14_v33.raw
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from vipyr._power import load_case  # noqa: E402

KV_DEFAULT = 100.0
SBASE12 = 50.0


def g(x):
    return repr(float(x))


def write(name, rev):
    c, _ = load_case(name)
    base = c["baseMVA"]
    bus, gen, br = c["bus"], c["gen"], c["branch"]
    kv = {int(b[0]): (b[9] if b[9] > 0 else KV_DEFAULT) for b in bus}
    vm = {int(b[0]): b[7] for b in bus}
    out = [
        "0, %s, %d, 0, 1, 60.0 / written by lib/power/oracle/write_raw.py"
        % (g(base), rev),
        "%s as PSS/E RAW revision %d" % (name, rev),
        "MATPOWER case, BSD-3-Clause",
    ]
    if rev == 34:
        out.append("0 / END OF SYSTEM-WIDE DATA, BEGIN BUS DATA")
    for b in bus:
        out.append(
            "%d,'B%-6d', %s, %d, %d, %d, 1, %s, %s, %s, %s, 1.1, 0.9"
            % (
                b[0],
                b[0],
                g(kv[int(b[0])]),
                b[1],
                b[6],
                b[10],
                g(b[7]),
                g(b[8]),
                g(b[11]),
                g(b[12]),
            )
        )
    out.append("0 / END OF BUS DATA, BEGIN LOAD DATA")
    extra_shunt = {}
    for b in bus:
        i = int(b[0])
        pl, ql, gs, bs = b[2], b[3], b[4], b[5]
        if pl == 0 and ql == 0 and gs == 0 and bs == 0:
            continue
        # three quarters constant power, a quarter constant current at the case voltage;
        # half the bus shunt as constant admittance (YQ is negative for a capacitor)
        ip, iq = 0.25 * pl / vm[i], 0.25 * ql / vm[i]
        out.append(
            "%d,'1 ', 1, %d, %d, %s, %s, %s, %s, %s, %s, 1"
            % (
                i,
                b[6],
                b[10],
                g(pl - ip * vm[i]),
                g(ql - iq * vm[i]),
                g(ip),
                g(iq),
                g(gs / 2),
                g(-bs / 2),
            )
        )
        extra_shunt[i] = (gs - gs / 2, bs - bs / 2)
    out.append("0 / END OF LOAD DATA, BEGIN FIXED SHUNT DATA")
    is_xf = (br[:, 8] != 0) | (br[:, 9] != 0)
    for k in np.where(is_xf)[0]:
        if br[k, 4] != 0:
            j = int(br[k, 1])
            gs, bs = extra_shunt.get(j, (0.0, 0.0))
            extra_shunt[j] = (gs, bs + br[k, 4] / 2 * base * (br[k, 10] > 0))
    for i, (gs, bs) in extra_shunt.items():
        if gs != 0 or bs != 0:
            out.append("%d,'1 ', 1, %s, %s" % (i, g(gs), g(bs)))
    out.append("0 / END OF FIXED SHUNT DATA, BEGIN GENERATOR DATA")
    for r in gen:
        nreg = " 0," if rev == 34 else ""
        out.append(
            "%d,'1 ', %s, %s, %s, %s, %s, 0,%s %s, 0.0, 0.2, 0.0, 0.0, 1.0, %d, 100.0, %s, %s, 1, 1.0"
            % (
                r[0],
                g(r[1]),
                g(r[2]),
                g(r[3]),
                g(r[4]),
                g(r[5]),
                nreg,
                g(r[6]),
                r[7],
                g(r[8]),
                g(r[9]),
            )
        )
    out.append("0 / END OF GENERATOR DATA, BEGIN BRANCH DATA")
    for k in np.where(~is_xf)[0]:
        r = br[k]
        if rev == 34:
            rates = "'L%d', %s, %s, %s, 0, 0, 0, 0, 0, 0, 0, 0, 0" % (
                k,
                g(r[5]),
                g(r[6]),
                g(r[7]),
            )
            tail = "0.0, 0.0, 0.0, 0.0, %d, 1, 0.0, 1, 1.0" % r[10]
        else:
            rates = "%s, %s, %s" % (g(r[5]), g(r[6]), g(r[7]))
            tail = "0.0, 0.0, 0.0, 0.0, %d, 1, 0.0, 1, 1.0" % r[10]
        out.append(
            "%d, %d,'1 ', %s, %s, %s, %s, %s"
            % (r[0], r[1], g(r[2]), g(r[3]), g(r[4]), rates, tail)
        )
    if rev == 34:
        out.append(
            "0 / END OF BRANCH DATA, BEGIN SYSTEM SWITCHING DEVICE DATA"
        )
        out.append(
            "0 / END OF SYSTEM SWITCHING DEVICE DATA, BEGIN TRANSFORMER DATA"
        )
    else:
        out.append("0 / END OF BRANCH DATA, BEGIN TRANSFORMER DATA")
    for n, k in enumerate(np.where(is_xf)[0]):
        r = br[k]
        f, t = int(r[0]), int(r[1])
        tap = r[8] if r[8] != 0 else 1.0
        cw, cz = 1 + n % 3, 1 + (n // 3) % 3
        rr, xx, sb = r[2], r[3], base
        w2 = 1.0
        if cw == 1 and n % 2 == 1:
            w2 = (
                1.02  # WINDV2 != 1: the reader refers the impedance through it
            )
            rr, xx = rr / w2**2, xx / w2**2
        if cz >= 2:
            rr, xx, sb = rr * SBASE12 / base, xx * SBASE12 / base, SBASE12
        if cz == 3:
            rr, xx = rr * sb * 1e6, np.hypot(rr, xx)
        if cw == 1:
            w1, n1, w2v, n2 = tap * w2, 0.0, w2, 0.0
        elif cw == 2:
            w1, n1, w2v, n2 = tap * kv[f], 0.0, kv[t], 0.0
        else:
            n1 = 1.05 * kv[f]
            w1, w2v, n2 = tap / 1.05, 1.0, 0.0
        mag2 = r[4] / 2 / tap**2 * (r[10] > 0)
        rates = ("%s, %s, %s" % (g(r[5]), g(r[6]), g(r[7]))) + (
            ", 0" * 9 if rev == 34 else ""
        )
        out.append(
            "%d, %d, 0,'1 ', %d, %d, 1, 0.0, %s, 2, 'T%d', %d, 1, 1.0"
            % (f, t, cw, cz, g(mag2), k, r[10])
        )
        out.append("%s, %s, %s" % (g(rr), g(xx), g(sb)))
        out.append(
            "%s, %s, %s, %s, 0, 0, 1.1, 0.9, 1.1, 0.9, 33, 0, 0.0, 0.0, 0.0"
            % (g(w1), g(n1), g(r[9]), rates)
        )
        out.append("%s, %s" % (g(w2v), g(n2)))
    out.append("0 / END OF TRANSFORMER DATA, BEGIN AREA DATA")
    areas = sorted({int(a) for a in bus[:, 6]})
    for a in areas:
        sw = (
            int(bus[(bus[:, 6] == a) & (bus[:, 1] == 3), 0][0])
            if np.any((bus[:, 6] == a) & (bus[:, 1] == 3))
            else 0
        )
        out.append("%d, %d, 0.0, 10.0, 'A%d'" % (a, sw, a))
    rest = [
        "TWO-TERMINAL DC",
        "VSC DC LINE",
        "IMPEDANCE CORRECTION",
        "MULTI-TERMINAL DC",
        "MULTI-SECTION LINE",
    ]
    out.append("0 / END OF AREA DATA, BEGIN TWO-TERMINAL DC DATA")
    for a, b in zip(rest, rest[1:] + ["ZONE"]):
        out.append("0 / END OF %s DATA, BEGIN %s DATA" % (a, b))
    for z in sorted({int(z) for z in bus[:, 10]}):
        out.append("%d, 'Z%d'" % (z, z))
    tail = [
        "ZONE",
        "INTER-AREA TRANSFER",
        "OWNER",
        "FACTS DEVICE",
        "SWITCHED SHUNT",
        "GNE DEVICE",
        "INDUCTION MACHINE",
    ] + (["SUBSTATION"] if rev == 34 else [])
    for a, b in zip(tail, tail[1:] + [""]):
        out.append(
            "0 / END OF %s DATA%s" % (a, (", BEGIN %s DATA" % b) if b else "")
        )
    out.append("Q")
    path = (
        Path(__file__).resolve().parents[1]
        / "cases"
        / "raw"
        / ("%s_v%d.raw" % (name, rev))
    )
    path.write_text("\n".join(out) + "\n")
    return path


if __name__ == "__main__":
    print(write(sys.argv[1], int(sys.argv[2])))
