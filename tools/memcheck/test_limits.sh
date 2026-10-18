#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# The time limits of tools/memcheck/ci.sh, demonstrated (tools/gate/load.py):
#
#     tools/memcheck/test_limits.sh
#
# 1. A program that spins past LIMIT seconds of CPU is TIMEOUT, whatever the load.
# 2. A program that needs less CPU than LIMIT is clean even when the machine's
#    load stretches its wall time past LIMIT (here: 3 busy loops per core).
# 3. A program that blocks (it opens a FIFO nobody writes) uses no CPU: the
#    wall-clock cap, 10 times LIMIT, stops it. On a quiet machine that is
#    TIMEOUT (stuck); on a busy one (GATE_LOAD1 pretends) it is BLOCKED-LOAD,
#    which ci.sh reruns when the load falls and the gate reports as such.
# Exits nonzero if any row is not what it should be. About 40 seconds.
set -u
root=$(cd "$(dirname "$0")/../.." && pwd)
export VIV="${VIV:-$root/target/release/viv}"
d=$(mktemp -d)
pids=""
trap 'kill $pids 2> /dev/null; rm -rf "$d"' EXIT
loop() { printf 'n = %s\ns = 0.0\nfor i to n {\n    s = s + 1.0\n}\nprint(s)\n' "$1" > "$d/$2.viv"; }
loop 3000000000 spin
loop 600000000 short
printf 's = read_text("hang.fifo")\nprint(len(s))\n' > "$d/hang.viv"
mkfifo "$d/hang.fifo"
bad=0
expect() {   # expect <status> <row>
    st=$(printf '%s\n' "$2" | cut -f2)
    if [ "$st" = "$1" ]; then echo "ok    $1: $(printf '%s\n' "$2" | cut -f3)"; else echo "FAIL  wanted $1, got: $2"; bad=1; fi
}
expect TIMEOUT "$(LIMIT=2 sh "$root/tools/memcheck/ci.sh" one "$d/spin.viv")"
n=$(( $(sysctl -n hw.ncpu 2>/dev/null || nproc) * 3 ))
pids=""
i=0; while [ $i -lt $n ]; do yes > /dev/null & pids="$pids $!"; i=$((i + 1)); done
t0=$(date +%s)
row=$(LIMIT=3 sh "$root/tools/memcheck/ci.sh" one "$d/short.viv")
t1=$(date +%s)
kill $pids 2> /dev/null; wait $pids 2> /dev/null
expect clean "$row"
echo "      (it took $((t1 - t0)) s of wall time against a limit of 3 s of CPU, with $n busy loops running)"
expect TIMEOUT "$(GATE_LOAD1=0 LIMIT=1 sh "$root/tools/memcheck/ci.sh" one "$d/hang.viv")"
expect BLOCKED-LOAD "$(GATE_LOAD1=1000 LIMIT=1 sh "$root/tools/memcheck/ci.sh" one "$d/hang.viv")"
exit $bad
