#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Check the fast build of every Viv program in a directory against the exact
# engines: `viv check --fast` per program.
# The exact engines must agree byte for byte (tools/three-engines.sh is the
# full check of that); the fast build's output must match theirs with every
# number within its tolerance, every integer and every piece of text equal.
#
#     tools/fast-check.sh posts                    every .viv in posts/
#     tools/fast-check.sh lib/hf scf_rhf           one program
#     WASM=1 tools/fast-check.sh pde               the direct wasm as a reference too
#     ATTRIBUTE=1 tools/fast-check.sh posts        for a program that differs, which
#                                                  freedom moves it (contract, libm, blas)
#     NO_INTERP=1 tools/fast-check.sh dft          the exact C build alone as the reference
#                                                  (for programs too long for the interpreter)
#     IDENTITY=1 tools/fast-check.sh posts         the fast build with all three freedoms
#                                                  taken away must print the exact bytes
#
# One line per program: how many numbers it printed, how many differ, the
# largest difference in ulps and as a share of its tolerance. VIV=<path> runs
# another viv; ULPS=<n> sets the budget (default 2^20); FASTCHECK_JSON=<file>
# appends each program's JSON report to a file.
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
dir=$1; shift
case "$dir" in /*) ;; *) dir="$root/$dir" ;; esac
viv="${VIV:-$root/target/release/viv}"
if [ $# -gt 0 ]; then names="$*"; else names=$(cd "$dir" && ls *.viv | sed 's/\.viv$//'); fi
summarize() {
    python3 -c '
import json, sys
t = [l for l in sys.stdin.read().splitlines() if l.startswith("{")]
if not t:
    print("NO REPORT"); sys.exit()
r = json.loads(t[-1]); c = r["compare"]
if sys.argv[1] == "identity":
    same = c["identical"] == c["numbers"] and not c["structural"] and r["references_agree"]
    print(("identical " if same else "NOT IDENTICAL ") + "%d numbers, %d differ" % (c["numbers"], len(c["diffs"])))
    sys.exit()
d = c["diffs"]
ulps = "%.0f" % c["max_ulps"] if c["max_ulps"] is not None else "-"
worst = "%.2e" % c["worst"] if c["worst"] is not None else "-"
msg = "%5d numbers, %4d differ, max %s ulps, %s of tolerance" % (c["numbers"], len(d), ulps, worst)
if c["floor_only"]:
    msg += " (%d by the output scale)" % c["floor_only"]
if r["solves"] or r["gemms"]:
    msg += "; %d dgemm, %d solves" % (r["gemms"], r["solves"])
if not r["references_agree"]:
    msg += "; EXACT ENGINES DISAGREE"
if sys.argv[1] == "full":
    for s in c["structural"][:3]:
        msg += "\n      " + s.replace("\n", "\n      ")[:400]
print(("ok      " if r["ok"] else "DIFFERS ") + msg)
' "$1"
}
bad=0
for name in $names; do
    src="$dir/$name.viv"
    if grep -qx '#! timing' "$src"; then
        printf '%-38s timing program, not compared\n' "$name"
        continue
    fi
    set -- --fast
    [ -n "$ULPS" ] && set -- "$@" --ulps "$ULPS"
    [ -n "$WASM" ] && set -- "$@" --wasm
    [ -n "$NO_INTERP" ] && set -- "$@" --no-interp
    if [ -n "$IDENTITY" ]; then
        out=$("$viv" check "$@" --without contract,libm,blas --json "$src" 2>&1) || true
        line=$(printf '%s\n' "$out" | summarize identity)
        printf '%-38s %s\n' "$name" "$line"
        case "$line" in identical*) ;; *) bad=1 ;; esac
        continue
    fi
    if out=$("$viv" check "$@" --json "$src" 2>&1); then st=0; else st=1; fi
    [ -n "$FASTCHECK_JSON" ] && printf '%s\n' "$out" | grep '^{' >> "$FASTCHECK_JSON" || true
    line=$(printf '%s\n' "$out" | summarize full)
    case "$line" in "NO REPORT") line=$(printf '%s\n' "$out" | head -4) ;; esac
    printf '%-38s %s\n' "$name" "$line"
    if [ $st != 0 ]; then
        bad=1
        if [ -n "$ATTRIBUTE" ]; then
            for only in contract libm blas; do
                case $only in contract) w=libm,blas ;; libm) w=contract,blas ;; blas) w=contract,libm ;; esac
                o=$("$viv" check "$@" --without $w --json "$src" 2>&1) || true
                printf '    %-10s alone: %s\n' "$only" "$(printf '%s\n' "$o" | summarize short)"
            done
        fi
    fi
done
exit $bad
