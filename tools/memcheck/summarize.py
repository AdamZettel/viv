#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Summarize study/memcheck/corpus.tsv by directory.

    tools/memcheck/summarize.py study/memcheck/corpus.tsv
"""
import sys
from collections import Counter, defaultdict

rows = [line.rstrip("\n").split("\t") for line in open(sys.argv[1])][1:]
by = defaultdict(list)
for r in rows:
    by[r[0]].append(r)


def num(x):
    try:
        return float(x)
    except ValueError:
        return None


print("| directory | programs | built | ASan errors | UBSan (programs) | ASan run timed out | `leaks` > 0 | arena at exit, unreachable (max) | peak RSS (max) |")
print("|---|---|---|---|---|---|---|---|---|")
tot = Counter()
ub_sites = Counter()
for d in sorted(by):
    rs = by[d]
    built = [r for r in rs if r[2] == "yes"]
    asan = sum(1 for r in built if r[3] == "ASAN")
    ub = [r for r in built if r[3] == "UBSAN"]
    for r in ub:
        for s in r[15].split("; "):
            ub_sites[s] += 1
    to = sum(1 for r in built if r[3] == "timeout" or r[4] == "142")
    lk = sum(1 for r in built if num(r[13]) and num(r[13]) > 0)
    dead = max([num(r[10]) or 0 for r in built] or [0])
    rss = max([num(r[12]) or 0 for r in built] or [0])
    print(f"| {d} | {len(rs)} | {len(built)} | {asan} | {len(ub)} | {to} | {lk} | {dead / 1e6:.2f} MB | {rss / 1e6:.0f} MB |")
    tot.update(programs=len(rs), built=len(built), asan=asan, ub=len(ub), to=to, lk=lk)
print(f"| **all** | {tot['programs']} | {tot['built']} | {tot['asan']} | {tot['ub']} | {tot['to']} | {tot['lk']} | | |")
print()
print("UBSan findings, by C function (programs):")
for s, n in ub_sites.most_common():
    print(f"- {s} ({n})")
leaked = [r for r in rows if num(r[13]) and num(r[13]) > 0]
if leaked:
    print()
    print("Programs with blocks `leaks` found unreachable:")
    for r in leaked:
        print(f"- {r[1]}: {r[13]} blocks, {r[14]} bytes")
notbuilt = [r for r in rows if r[2] != "yes"]
if notbuilt:
    print()
    print("Not compiled by viv on their own:", ", ".join(r[1] for r in notbuilt))
