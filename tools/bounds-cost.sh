#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# What the array bounds checks cost.
#
#     tools/bounds-cost.sh bench/hf_md_eri.viv bench/matmul_loops.viv
#
# Each program is compiled to C once and built twice with cc -O2: as it
# ships (every array read and write checked) and with -DVIV_NO_BOUNDS (the
# checks compiled out, everything else the same). The two binaries run
# RUNS times each, alternating, so a slow moment of the machine lands on
# both; the report is the best time of each and the cost of the checks as a
# percentage of the unchecked time. The time is the child's CPU time (user +
# system, from getrusage), not the wall clock: other jobs on a shared machine
# stretch the wall time of a run but barely touch its CPU time. The two must print the same thing, or
# the comparison means nothing and the script says so.
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
viv="${VIV:-$root/target/release/viv}"
runs=${RUNS:-5}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

printf '%-26s %10s %10s %8s\n' "program" "checked" "unchecked" "cost"
for src in "$@"; do
    name=$(basename "$src" .viv)
    "$viv" "$src" -o "$tmp/$name.c" > /dev/null
    ${CC:-cc} -O2 -o "$tmp/$name.checked" "$tmp/$name.c" -lm
    ${CC:-cc} -O2 -DVIV_NO_BOUNDS -o "$tmp/$name.unchecked" "$tmp/$name.c" -lm
    python3 - "$tmp/$name" "$name" "$runs" <<'PY'
import resource, subprocess, sys
base, name, runs = sys.argv[1], sys.argv[2], int(sys.argv[3])
def cpu():
    u = resource.getrusage(resource.RUSAGE_CHILDREN)
    return u.ru_utime + u.ru_stime
best, outs = {}, {}
for _ in range(runs):
    for kind in ("checked", "unchecked"):
        t0 = cpu()
        r = subprocess.run([f"{base}.{kind}"], capture_output=True, text=True)
        dt = cpu() - t0
        best[kind] = min(best.get(kind, dt), dt)
        outs[kind] = r.stdout
if outs["checked"] != outs["unchecked"]:
    print(f"{name}: the two builds print different output -- no comparison")
    sys.exit(1)
cost = 100 * (best["checked"] / best["unchecked"] - 1)
print(f"{name:26s} {best['checked']:9.3f}s {best['unchecked']:9.3f}s {cost:+7.1f}%")
PY
done
