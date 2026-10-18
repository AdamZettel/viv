#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Score the Viv Hartree-Fock code against PySCF v2.5.0's own unit tests.
#
#     tools/hf_score.sh            every scf/suite program, then the benzene row
#     tools/hf_score.sh -v         also list every row
#     BZ=0 tools/hf_score.sh       skip the native benzene run (about 2 minutes)
#
# The rows are the assertions harvested into tools/pyscf_suite.tsv (and, as
# Viv constants, lib/hfref/pyscf_tests.viv). Each scf/suite/*.viv program
# replays a group of them. A program prints one line per row it replays,
#
#     sts_line: [file code, line, pass, decade of the gap]
#
# where pass is unittest's rule, round(|value - target|, places) == 0.
#
# First every program runs through tools/both-engines.sh (interpreter and
# native, tests must pass, outputs must agree byte for byte), which prints
# its usual line ending in "identical output". The rows are then read from
# a native run. The per-tier lines count passing rows against the tier's
# size in the manifest. Rows tagged `path` (the result depends on PySCF's
# exact iteration path) are not counted in a tier's score: they are listed
# on their own, so a path row that is not replayed is never an energy
# failure. Rows that no program replays are listed with the manifest's note.
#
# scf/suite/native/bz.viv (benzene cc-pVDZ, 114 AOs, direct J/K) is built
# and run natively only, timed, and reported apart; it is not gated.
#
# Exits nonzero when a program fails either engine or the engines differ,
# or when a tier is below its gate: T1 17/17, T1s 1/1, T3 >= 17, T2 >= 25,
# T2+T3 >= 20.
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
viv="${VIV:-$root/target/release/viv}"
verbose=0
[ "${1:-}" = "-v" ] && verbose=1
jobs=${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

names=$(cd "$root/scf/suite" && ls *.viv | sed 's/\.viv$//')

# both engines, several programs at once; the slow direct ones first
for n in $names; do echo "$n"; done | awk '/direct/ { print "0 " $0; next } { print "1 " $0 }' |
    sort -s -k1,1 | cut -d' ' -f2 |
    xargs -P "$jobs" -n 1 sh -c '
        "$1/tools/both-engines.sh" scf/suite "$4" > "$2/$4.both" 2>&1
        echo $? > "$2/$4.rc"
        "$3" "$1/scf/suite/$4.viv" -o "$2/$4.c" > /dev/null 2>&1 &&
            ${CC:-cc} -O2 -o "$2/$4.bin" "$2/$4.c" -lm 2> /dev/null &&
            "$2/$4.bin" > "$2/$4.rows" 2>&1 || true
        ' score "$root" "$tmp" "$viv" || true
# (in the "sh -c" script: $1 is the repo root, $2 the output directory, $3
# the viv binary, and $4 the program name that xargs appends)

bad=0
for n in $names; do
    if [ ! -f "$tmp/$n.rc" ] || [ "$(cat "$tmp/$n.rc")" != 0 ]; then bad=1; fi
    if [ -f "$tmp/$n.both" ]; then cat "$tmp/$n.both"; else printf '%-34s DID NOT RUN\n' "$n"; bad=1; fi
done

# the benzene row: native only, timed inside one process
bzline=""
if [ "${BZ:-1}" != 0 ]; then
    if "$viv" "$root/scf/suite/native/bz.viv" -o "$tmp/bz.c" > /dev/null 2>&1 &&
        ${CC:-cc} -O2 -o "$tmp/bz.bin" "$tmp/bz.c" -lm 2> /dev/null; then
        bzline=$(python3 -c '
import subprocess, sys, time
t = time.perf_counter()
p = subprocess.run([sys.argv[1]], capture_output=True, text=True)
print("%.1f" % (time.perf_counter() - t))
sys.stdout.write(p.stdout)
' "$tmp/bz.bin")
        printf '%s\n' "$bzline" > "$tmp/bz.out"
    else
        echo "bz: DOES NOT BUILD" > "$tmp/bz.out"
    fi
fi

python3 - "$root" "$tmp" "$verbose" <<'PY' || bad=1
import glob, os, re, sys

root, tmp, verbose = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
# the file codes of lib/hfref/pyscf_tests.viv (PST_FILE_*)
codes = {}
for line in open(os.path.join(root, "lib/hfref/pyscf_tests.viv")):
    m = re.match(r"^PST_FILE_([A-Z0-9_]+) = (\d+)$", line.strip())
    if m:
        codes[int(m.group(2))] = "test_%s.py" % m.group(1).lower()

rows = []
with open(os.path.join(root, "tools/pyscf_suite.tsv")) as f:
    head = f.readline().rstrip("\n").split("\t")
    for line in f:
        r = dict(zip(head, line.rstrip("\n").split("\t")))
        rows.append(r)

def parse(path):
    got = {}
    if not os.path.exists(path):
        return got
    for line in open(path):
        m = re.match(r"^sts_line: \[(\d+), (\d+), (\d), (-?\d+)\]$", line.strip())
        if m:
            loc = "%s:%s" % (codes[int(m.group(1))], m.group(2))
            got[loc] = (int(m.group(3)), int(m.group(4)))
    return got

seen = {}
where = {}
for p in sorted(glob.glob(os.path.join(tmp, "*.rows"))):
    name = os.path.basename(p)[:-5]
    for loc, v in parse(p).items():
        if loc in seen and seen[loc][0] != v[0]:
            print("row %s replayed twice with different results" % loc)
            sys.exit(1)
        seen[loc] = v
        where[loc] = name

bz = parse(os.path.join(tmp, "bz.out"))
bzloc = "test_bz.py:56"

tiers = ["T1", "T1s", "T1x", "T2", "T3", "T2+T3"]
gates = {"T1": 17, "T1s": 1, "T3": 17, "T2": 25, "T2+T3": 20}
ok = True
print("")
print("rows replayed from PySCF v2.5.0's tests (tools/pyscf_suite.tsv):")
for t in tiers:
    tr = [r for r in rows if r["tier"] == t]
    plain = [r for r in tr if r["tags"] != "path" and r["loc"] != bzloc]
    path = [r for r in tr if r["tags"] == "path"]
    npass = sum(1 for r in plain if seen.get(r["loc"], (0, 0))[0] == 1)
    ppass = sum(1 for r in path if seen.get(r["loc"], (0, 0))[0] == 1)
    notes = []
    if path:
        notes.append("path rows %d/%d replayed and passing, not counted" % (ppass, len(path)))
    if any(r["loc"] == bzloc for r in tr):
        notes.append("the benzene row is reported below, not counted")
    line = "%-6s %d/%d" % (t, npass, len(tr))
    if notes:
        line += "   (" + "; ".join(notes) + ")"
    if t in gates and npass < gates[t]:
        line += "   BELOW GATE %d" % gates[t]
        ok = False
    print(line)

missing = [r for r in rows if r["loc"] not in seen and r["loc"] != bzloc]
failed = [r for r in rows if seen.get(r["loc"], (1, 0))[0] == 0]
if failed:
    print("")
    print("rows replayed that do not pass:")
    for r in failed:
        print("  %-22s %-6s %s (gap below 1e%d)" % (r["loc"], r["tier"], where[r["loc"]], seen[r["loc"]][1] + 1))
print("")
print("rows no program replays:")
for r in missing:
    tag = " path" if r["tags"] == "path" else ""
    print("  %-22s %-6s%s  %s: %s" % (r["loc"], r["tier"], tag, r["test"], r["method"]))

print("")
bzout = os.path.join(tmp, "bz.out")
if os.path.exists(bzout):
    lines = open(bzout).read().split("\n")
    secs = lines[0]
    if bzloc in bz:
        print("benzene cc-pVDZ (test_bz.py:56, native, direct J/K): %s in %s s" %
              ("pass" if bz[bzloc][0] == 1 else "FAIL (gap below 1e%d)" % (bz[bzloc][1] + 1), secs))
    else:
        print("benzene cc-pVDZ (test_bz.py:56): no row printed; %s" % " ".join(lines[1:])[:200])
else:
    print("benzene cc-pVDZ (test_bz.py:56): skipped (BZ=0)")

if verbose:
    print("")
    for r in rows:
        v = bz.get(r["loc"]) if r["loc"] == bzloc else seen.get(r["loc"])
        st = "not run" if v is None else ("pass" if v[0] == 1 else "FAIL")
        gap = "" if v is None else ("exact" if v[1] == -99 else "gap<1e%d" % (v[1] + 1))
        print("  %-22s %-6s %-5s %-8s %-9s %s" % (r["loc"], r["tier"], r["tags"], st, gap, where.get(r["loc"], "")))
sys.exit(0 if ok else 1)
PY
exit $bad
