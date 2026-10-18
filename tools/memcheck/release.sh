#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# What -DVIV_RELEASE does with an index out of range (study/memcheck/release/*.viv).
#
#     tools/memcheck/release.sh OUT.txt
#
# Each program indexes out of range on purpose. It is run by the interpreter,
# built as it ships by default (checked), built with -DVIV_RELEASE (the flag the
# served wasm is built with), and built with -DVIV_RELEASE under ASan+UBSan.
# Printed: what each build prints or reports, and its exit status.
set -u
root=$(cd "$(dirname "$0")/../.." && pwd)
viv="${VIV:-$root/target/release/viv}"
out=$1
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
: > "$out"
one() { tr '\n' ' ' < "$1" | cut -c1-150; }
for f in "$root"/study/memcheck/release/*.viv; do
    name=$(basename "$f" .viv)
    "$viv" run "$f" 2>&1 | grep -v '^__graph__\|^__trace__' | grep -v '^ *|\|^ *-->\|^ *$' > "$tmp/i"
    "$viv" "$f" -o "$tmp/p.c" > /dev/null
    cc -w -O2 -o "$tmp/chk" "$tmp/p.c" -lm
    cc -w -O2 -DVIV_RELEASE -o "$tmp/rel" "$tmp/p.c" -lm
    cc -w -O1 -g -DVIV_RELEASE -fsanitize=address,undefined -fno-omit-frame-pointer -o "$tmp/rsan" "$tmp/p.c" -lm
    "$tmp/chk" > "$tmp/c" 2>&1; cs=$?
    grep -v '^ *|\|^ *-->\|^ *$' "$tmp/c" > "$tmp/c2"
    "$tmp/rel" > "$tmp/r" 2>&1; rs=$?
    ASAN_OPTIONS=detect_leaks=0 "$tmp/rsan" > "$tmp/s" 2>&1; ss=$?
    {
        echo "== $name"
        echo "  interpreter:        $(one "$tmp/i")"
        echo "  C, checked:         exit $cs: $(one "$tmp/c2")"
        echo "  C, -DVIV_RELEASE:   exit $rs: $(one "$tmp/r")"
        echo "  release under ASan: exit $ss: $(grep -m1 'ERROR: AddressSanitizer\|runtime error' "$tmp/s" | sed 's/^==[0-9]*==//' | cut -c1-120)"
    } >> "$out"
done
