#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Time the string benchmarks in every engine and check
# each prints what CPython printed (tools/strings/bench/*.out):
#
#     tools/strings/bench.sh
#
# join: 10^6 one-character strings into one; split: a line of 10^6 tokens;
# floats: 10^6 numbers parsed from text. Each is linear; a quadratic path
# would take minutes here instead of a second.
#
# Needs target/release/viv (or VIV=<path>), cc, node and a viv.wasm with the
# runtime exports (www/build-wasm.sh, or VIV_WASM=<path>).
set -e
root=$(cd "$(dirname "$0")/../.." && pwd)
viv="${VIV:-$root/target/release/viv}"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
secs() { perl -MTime::HiRes=time -e '$t = time; system @ARGV; printf "%.2f", time - $t' "$@"; }
bad=0
printf '%-8s %12s %12s %12s\n' bench interpreter native wasm
for b in join split floats; do
    src="$root/tools/strings/bench/$b.viv"
    want="$root/tools/strings/bench/$b.out"
    "$viv" "$src" -o "$tmp/$b.c" > /dev/null
    ${CC:-cc} -O2 -o "$tmp/$b" "$tmp/$b.c" -lm
    "$viv" "$src" -o "$tmp/$b.wasm" > /dev/null
    # macOS checks a new executable the first time it runs; that is not the
    # program's time, so the native binary runs once before it is timed
    "$tmp/$b" > /dev/null
    ti=$(secs sh -c "'$viv' run -q '$src' > '$tmp/$b.i'")
    tn=$(secs sh -c "'$tmp/$b' > '$tmp/$b.n'")
    tw=$(secs sh -c "node '$root/tools/run-wasm.mjs' '$tmp/$b.wasm' > '$tmp/$b.w'")
    printf '%-8s %11ss %11ss %11ss\n' "$b" "$ti" "$tn" "$tw"
    for e in i n w; do
        if ! cmp -s "$want" "$tmp/$b.$e"; then
            echo "  $b ($e) printed:"; head -3 "$tmp/$b.$e"; echo "  CPython:"; cat "$want"; bad=1
        fi
    done
done
exit $bad
