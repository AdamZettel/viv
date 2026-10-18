#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Fuzz the hand-written C runtime against the interpreter: random programs from gen.py, each run by the interpreter and as
# C built with AddressSanitizer and UndefinedBehaviorSanitizer. They must print
# the same bytes and stop (when they stop) with the same first line; a
# sanitizer report is a failure whatever the output.
#
#     tools/memcheck/fuzz/run.sh [OUT_DIR]
#
# FUZZ_SECONDS (default 60) bounds the run, FUZZ_SEED the first seed (default:
# the time, so each run tries new programs; the seed of every failure is
# printed and the program kept in OUT_DIR). VIV=<path>, CC=<compiler>.
set -u
root=$(cd "$(dirname "$0")/../../.." && pwd)
viv="${VIV:-$root/target/release/viv}"
cc="${CC:-cc}"
out=${1:-$root/target/memcheck-ci/fuzz}
mkdir -p "$out"
secs=${FUZZ_SECONDS:-60}
seed=${FUZZ_SEED:-$(date +%s)}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
end=$(( $(date +%s) + secs ))
n=0; bad=0; stopped=0
while [ "$(date +%s)" -lt "$end" ]; do
    python3 "$root/tools/memcheck/fuzz/gen.py" "$seed" > "$tmp/p.viv"
    perl -e 'alarm shift; exec @ARGV' 30 "$viv" run "$tmp/p.viv" > "$tmp/i.raw" 2>&1
    # the program's output, then the first line of an error (its location
    # lines, which the two engines draw differently, are left out)
    grep -v '^__graph__\|^__trace__' "$tmp/i.raw" | awk '/^error/ { print; exit } { print }' > "$tmp/i.out"
    why=""
    if ! "$viv" "$tmp/p.viv" -o "$tmp/p.c" > /dev/null 2> "$tmp/v.err"; then
        # a program the interpreter runs must compile
        if ! grep -q '^error' "$tmp/i.out"; then why="the interpreter ran it and the compiler refused it: $(head -1 "$tmp/v.err")"; fi
    elif ! $cc -w -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer -o "$tmp/p" "$tmp/p.c" -lm 2> "$tmp/cc.err"; then
        why="the C does not build: $(grep -m1 error "$tmp/cc.err")"
    else
        ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 perl -e 'alarm shift; exec @ARGV' 30 "$tmp/p" > "$tmp/c.raw" 2> "$tmp/c.err"
        if grep -q 'Sanitizer\|runtime error:' "$tmp/c.err"; then
            why="sanitizer: $(grep -m1 'ERROR: AddressSanitizer\|runtime error:' "$tmp/c.err" | cut -c1-200)"
        else
            { cat "$tmp/c.raw"; grep -m1 '^error' "$tmp/c.err"; } > "$tmp/c.out"
            grep -q '^error' "$tmp/i.out" && stopped=$((stopped + 1))
            if ! cmp -s "$tmp/i.out" "$tmp/c.out"; then
                why="the engines disagree: $(diff "$tmp/i.out" "$tmp/c.out" | head -4 | tr '\n' ' ' | cut -c1-240)"
            fi
        fi
    fi
    if [ -n "$why" ]; then
        bad=$((bad + 1))
        cp "$tmp/p.viv" "$out/fail_$seed.viv"
        echo "   FAIL seed $seed: $why (kept as $out/fail_$seed.viv)"
    fi
    n=$((n + 1)); seed=$((seed + 1))
done
echo "   fuzz: $n programs in ${secs}s, $stopped stopped on a run-time error in both engines alike, $bad failures"
[ $bad = 0 ]
