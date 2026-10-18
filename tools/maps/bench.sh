#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Time maps with string keys in every engine:
#
#     tools/maps/bench.sh [n ...]        (default: 1000000 2000000)
#
# `store`: n stores of distinct string keys (`m[f"key{j}"] = i`, the keys in a
# scrambled order) and n reads of them. `keys`: the same loops making the same
# strings, with no map, so store minus keys is the map's own time. Each time
# is the median of three whole-process runs; the wasm column includes node's
# start. Every engine must print the same line.
#
# Needs target/release/viv (or VIV=<path>), cc, node and a viv.wasm with the
# runtime exports (www/build-wasm.sh, or VIV_WASM=<path>).
set -e
root=$(cd "$(dirname "$0")/../.." && pwd)
viv="${VIV:-$root/target/release/viv}"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
secs() { perl -MTime::HiRes=time -e '$t = time; system @ARGV; printf "%.2f", time - $t' "$@"; }
median() { printf '%s\n' "$@" | sort -n | sed -n 2p; }
sizes=${*:-1000000 2000000}
bad=0
printf '%-6s %9s %12s %12s %12s\n' bench n interpreter native wasm
for n in $sizes; do
    cat > "$tmp/store.viv" <<V
N = $n
m = map()
for i to N {
    m[f"key{mod(i * 7919, N):d}"] = i
}
s = 0.0
for i to N {
    s = s + m[f"key{i:d}"]
}
print(fmt(s, "d"), fmt(len(m), "d"))
V
    cat > "$tmp/keys.viv" <<V
N = $n
s = 0.0
for i to N {
    k = f"key{mod(i * 7919, N):d}"
    s = s + len(k)
}
for i to N {
    k = f"key{i:d}"
    s = s + len(k)
}
print(fmt(s, "d"))
V
    for b in store keys; do
        src="$tmp/$b.viv"
        "$viv" "$src" -o "$tmp/$b.c" > /dev/null
        ${CC:-cc} -O2 -o "$tmp/$b" "$tmp/$b.c" -lm 2> /dev/null
        "$viv" "$src" -o "$tmp/$b.wasm" > /dev/null
        "$tmp/$b" > /dev/null
        ti=""; tn=""; tw=""
        for r in 1 2 3; do
            ti="$ti $(secs sh -c "'$viv' run -q '$src' > '$tmp/$b.i'")"
            tn="$tn $(secs sh -c "'$tmp/$b' > '$tmp/$b.n'")"
            tw="$tw $(secs sh -c "node '$root/tools/run-wasm.mjs' '$tmp/$b.wasm' > '$tmp/$b.w'")"
        done
        printf '%-6s %9s %11ss %11ss %11ss\n' "$b" "$n" "$(median $ti)" "$(median $tn)" "$(median $tw)"
        for e in n w; do
            if ! cmp -s "$tmp/$b.i" "$tmp/$b.$e"; then
                echo "  $b ($e) printed:"; head -3 "$tmp/$b.$e"; echo "  the interpreter:"; cat "$tmp/$b.i"; bad=1
            fi
        done
    done
done
exit $bad
