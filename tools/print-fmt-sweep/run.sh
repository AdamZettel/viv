#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# `print` and `fmt` over 10^6 doubles. sweep.viv writes
# each double 26 ways -- repr, %g, %.1g to %.17g, three %e, two %f, `d`, a
# width -- in chunks of 100 000. Every chunk runs through the interpreter,
# Viv -> C -> native (whose %g, %e and %f are the C library's printf), and
# Viv -> wasm, and the three outputs must be the same bytes; then check.py
# holds every field to CPython's repr and format.
#
#     tools/print-fmt-sweep/run.sh            10 chunks: 10^6 doubles
#     CHUNKS=2 tools/print-fmt-sweep/run.sh   fewer
#
# Needs a built compiler (cargo build --release -p viv), a C compiler, node,
# python3, and www/public/viv/viv.wasm with the runtime exports
# (www/build-wasm.sh).
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
viv="${VIV:-$root/target/release/viv}"
chunks=${CHUNKS:-10}
n=100000
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
bad=0
c=0
while [ "$c" -lt "$chunks" ]; do
    start=$((c * n))
    set -- --set START=$start --set COUNT=$n
    "$viv" run --quiet "$@" "$here/sweep.viv" > "$tmp/interp"
    "$viv" "$@" "$here/sweep.viv" -o "$tmp/sweep.c" > /dev/null
    ${CC:-cc} -O2 -o "$tmp/sweep" "$tmp/sweep.c" -lm
    "$tmp/sweep" > "$tmp/native"
    "$viv" "$@" "$here/sweep.viv" -o "$tmp/sweep.wasm" > /dev/null
    node "$root/tools/run-wasm.mjs" "$tmp/sweep.wasm" > "$tmp/direct"
    lines=$(wc -l < "$tmp/native" | tr -d ' ')
    same=identical
    cmp -s "$tmp/interp" "$tmp/native" || { same="INTERP AND C DIFFER"; bad=1; }
    cmp -s "$tmp/interp" "$tmp/direct" || { same="INTERP AND WASM DIFFER"; bad=1; }
    printf 'doubles %7d-%7d: %s lines, three engines %s\n' "$start" $((start + n - 1)) "$lines" "$same"
    cp "$tmp/native" "$tmp/chunk$c"
    c=$((c + 1))
done
python3 "$here/check.py" "$tmp"/chunk* || bad=1
exit $bad
