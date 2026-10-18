#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Race implementations of the same stage.
#
#     tools/race.sh bench/eri_closed_form.viv bench/eri_hermite.viv
#
# The other half of the method in lib/qcfuzz. That file settles whether two
# implementations AGREE; this one settles which is faster. An implementation may
# be as clever as it likes so long as it still satisfies the contract, and the
# fastest one that does is the one that ships.
#
# Each program is compiled through Viv -> C -> native and run three times; the
# best time is reported, because the worst is the machine's business and not the
# program's. Anything a program prints is shown, so a race can also be a check:
# if two of them print different answers, the race is meaningless and that is
# visible rather than hidden in a number.
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
viv="$root/target/release/viv"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

printf '%-38s %10s  %s\n' "implementation" "best of 3" "what it printed"
for src in "$@"; do
    name=$(basename "$src" .viv)
    if ! "$viv" "$src" -o "$tmp/$name.c" >/dev/null 2>"$tmp/$name.err"; then
        printf '%-38s %10s  %s\n' "$name" "-" "DOES NOT COMPILE"
        head -3 "$tmp/$name.err"
        continue
    fi
    ${CC:-cc} -O2 -o "$tmp/$name" "$tmp/$name.c" -lm
    # One python process times all three runs with perf_counter around
    # subprocess.run, so the floor is the program's own startup. Timing with a
    # separate `python3 -c` before and after each run added two interpreter
    # startups: a program that only prints 1 measured 0.023 s that way.
    python3 - "$tmp/$name" "$name" <<'PY'
import subprocess, sys, time
exe, name = sys.argv[1], sys.argv[2]
best, out = None, ""
for _ in range(3):
    t0 = time.perf_counter()
    r = subprocess.run([exe], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    t = time.perf_counter() - t0
    best = t if best is None or t < best else best
    out = r.stdout.decode(errors="replace")
out = " ".join(l for l in out.split("\n") if not l.startswith("PASS "))
print("%-38s %9ss  %s" % (name, "%.3f" % best, out[:40]))
PY
done
