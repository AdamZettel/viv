#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Check the leak probes' tables: nothing grows with the length of a loop.

    tools/memcheck/flat.py scale.tsv [adversarial.tsv]

scale.tsv (tools/memcheck/scale.sh) has two rows per (case, place), at two
loop lengths; adversarial.tsv (adversarial.sh) two per program. A row pair
passes when the compiled program printed what the interpreter printed, the
array arena holds no more blocks at exit after the longer loop than its live
data accounts for, and the malloc zones hold at most SLACK bytes more than
the growth of that live data (the allocator's own tables and stdio's buffers
wobble by up to a page or so; a leak of one 128-byte array a pass adds
megabytes). Where the table has the live columns (adversarial.tsv: what the
program's globals still reach at exit), a program whose live data grows with
the loop (a list it appends to) is judged by its dead part. Exits 1 and names
each pair that fails.
"""
import csv
import sys

SLACK = 16384


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def pairs(table, key):
    by = {}
    for r in table:
        by.setdefault(key(r), []).append(r)
    for k, rs in by.items():
        rs.sort(key=lambda r: int(r["N"]))
        yield k, rs


def num(x):
    try:
        return int(x)
    except ValueError:
        return None


def check(path, key, same_col, same_ok):
    bad = []
    n = 0
    for k, rs in pairs(rows(path), key):
        if len(rs) < 2:
            continue
        a, b = rs[0], rs[-1]
        n += 1
        what = " ".join(k)
        if any(r[same_col] != same_ok for r in rs):
            bad.append(f"{what}: output {[r[same_col] for r in rs]}")
            continue
        if "sanitizers" in b and b["sanitizers"] != "clean":
            bad.append(f"{what}: {b['sanitizers']}")
        an, bn = num(a["arena_n"]), num(b["arena_n"])
        az, bz = num(a["zone_in_use"]), num(b["zone_in_use"])
        if an is None or bn is None or az is None or bz is None:
            bad.append(f"{what}: no memory report")
            continue
        # the live part, when the table has it
        live = "live_n" in a and num(a["live_n"]) is not None and num(b["live_n"]) is not None
        aln, bln = (num(a["live_n"]), num(b["live_n"])) if live else (0, 0)
        alb, blb = (num(a["live_bytes"]), num(b["live_bytes"])) if live else (0, 0)
        if bn - bln > an - aln:
            bad.append(f"{what}: arena {an} blocks ({an - aln} dead) at N={a['N']}, {bn} ({bn - bln} dead) at N={b['N']}")
        if (bz - az) - (blb - alb) > SLACK:
            bad.append(f"{what}: malloc zones {az} bytes at N={a['N']}, {bz} at N={b['N']} (live data {alb} -> {blb})")
    return n, bad


def main():
    fail = False
    for path in sys.argv[1:]:
        if "adversarial" in path:
            n, bad = check(path, lambda r: (r["program"],), "same_output", "yes")
        else:
            n, bad = check(path, lambda r: (r["case"], r["place"]), "interp", "ok")
        print(f"   {path}: {n} pairs, {len(bad)} growing")
        for b in bad:
            print(f"   FAIL {b}")
        fail |= bool(bad)
    sys.exit(1 if fail else 0)


main()
