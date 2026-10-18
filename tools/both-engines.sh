#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Run every Viv program in a directory through BOTH engines -- the interpreter,
# and Viv -> C -> native -- and require (1) every test to pass in each and
# (2) the two outputs to agree line for line. A PROBLEM post (`#! kind problem`)
# is a program whose tests are meant to fail; for it only (2) is required, and
# the failures are shown as what the problem asks for.
#
#     tools/both-engines.sh pde                 every .viv in pde/
#     tools/both-engines.sh posts 2026-09-21-gibbs-overshoot
#
# Needs a built compiler (cargo build --release -p viv) and a C compiler.
# VIV=<path> runs a different viv binary (the test suite passes the one it built).
# A program Viv accepts but the C compiler rejects is reported as
# DOES NOT BUILD (cc), and the rest of the directory is still checked.
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
dir=$1; shift
case "$dir" in /*) ;; *) dir="$root/$dir" ;; esac
viv="${VIV:-$root/target/release/viv}"
tmp=$(mktemp -d)
# several of these run at once under scf/check.sh: two compilers each
export VIV_JOBS=${VIV_JOBS:-2}
trap 'rm -rf "$tmp"' EXIT
if [ $# -gt 0 ]; then names="$*"; else names=$(cd "$dir" && ls *.viv | sed 's/\.viv$//'); fi
bad=0
for name in $names; do
    src="$dir/$name.viv"
    # the interpreter also prints its call graph and trace; the program's own
    # output is the rest. Both engines write the "N assertions FAILED" summary
    # and any run-time error to stderr after the program's output, so stderr
    # is merged in and compared too.
    "$viv" run "$src" 2>&1 | grep -v '^__graph__\|^__trace__' > "$tmp/$name.interp" || true
    # `viv build`: the exact build, its modules' objects from the cache.
    # Viv accepted the program but the C
    # compiler did not: a compiler bug, not the program's, and the remaining
    # programs still get checked.
    if ! "$viv" build "$src" -o "$tmp/$name" > /dev/null 2> "$tmp/$name.err"; then
        if grep -q 'rejected the generated C\|could not link' "$tmp/$name.err"; then
            printf '%-34s DOES NOT BUILD (cc)\n' "$name"; grep -m 3 'error' "$tmp/$name.err" | cut -c1-170
        else
            printf '%-34s DOES NOT COMPILE\n' "$name"; head -6 "$tmp/$name.err"
        fi
        bad=1; continue
    fi
    "$tmp/$name" > "$tmp/$name.native" 2>&1 || true
    pass=$(grep -c '^PASS ' "$tmp/$name.native" || true)
    fail=$(grep -c '^FAIL \|^error' "$tmp/$name.interp" "$tmp/$name.native" | awk -F: '{s += $2} END {print s}')
    if cmp -s "$tmp/$name.interp" "$tmp/$name.native"; then same="identical output"; else same="OUTPUT DIFFERS"; bad=1; fi
    if grep -q '^#! kind problem' "$src" || grep -q '^kind: problem' "$src"; then
        # a problem: its tests fail by design, and an `error` line is still an error
        [ "$fail" != 0 ] || bad=1
        grep -q '^error' "$tmp/$name.interp" "$tmp/$name.native" && bad=1
        unsolved=$(grep -c '^FAIL ' "$tmp/$name.native" || true)
        printf '%-34s %2s tests pass, %s to solve (a problem); %s\n' "$name" "$pass" "$unsolved" "$same"
    else
        [ "$fail" = 0 ] || bad=1
        # a module whose tests all need reference data keeps them in its
        # companion file, <dir>/tests/<name>.viv, which is checked on its own
        [ "$pass" != 0 ] || [ -f "$dir/tests/$name.viv" ] || bad=1
        printf '%-34s %2s tests pass, %s failed; %s\n' "$name" "$pass" "$fail" "$same"
        if [ "$fail" != 0 ]; then grep -h '^FAIL \|^error' "$tmp/$name.interp" "$tmp/$name.native" | sort -u | cut -c1-170 | head -6; fi
    fi
    if [ "$same" != "identical output" ]; then diff "$tmp/$name.interp" "$tmp/$name.native" | cut -c1-160 | head -8; fi
done
exit $bad
