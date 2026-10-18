#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Programs written to break the loop arena (study/memcheck/adversarial/*.viv).
#
#     tools/memcheck/adversarial.sh OUT.tsv [N1 N2]
#
# Each program has NITER where its iteration count goes. At N1 and N2 it is
# run by the interpreter, built with ASan+UBSan and run, and built with the
# memory report (instrument.py) and run. A row records whether the compiled
# output matched the interpreter's, the first sanitizer report, and the arena
# blocks and zone bytes at exit: blocks that grow with N are a leak.
set -u
root=$(cd "$(dirname "$0")/../.." && pwd)
viv="${VIV:-$root/target/release/viv}"
here="$root/tools/memcheck"
out=$1; n1=${2:-1000}; n2=${3:-4000}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
printf 'program\tN\tsame_output\tsanitizers\tarena_n\tarena_bytes\tlive_n\tlive_bytes\tzone_in_use\tmax_rss\n' > "$out"
for f in "$root"/study/memcheck/adversarial/*.viv; do
    for n in $n1 $n2; do
        sed "s/NITER/$n/g" "$f" > "$tmp/p.viv"
        "$viv" run "$tmp/p.viv" 2>&1 | grep -v '^__graph__\|^__trace__' > "$tmp/i.out"
        "$viv" "$tmp/p.viv" -o "$tmp/p.c" > /dev/null
        cc -w -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -o "$tmp/a" "$tmp/p.c" -lm
        ASAN_OPTIONS=detect_leaks=0 "$tmp/a" > /dev/null 2> "$tmp/a.err"
        san=$(grep -m1 'ERROR: AddressSanitizer\|runtime error:' "$tmp/a.err" | cut -c1-100)
        python3 "$here/instrument.py" "$tmp/p.c" > "$tmp/pm.c"
        cc -w -O1 -o "$tmp/pm" "$tmp/pm.c" -lm
        VIV_MEMSTAT="$tmp/stat" "$tmp/pm" > "$tmp/c.out" 2>&1
        if cmp -s "$tmp/i.out" "$tmp/c.out"; then same=yes; else same=NO; fi
        set -- $(cat "$tmp/stat")
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(basename "$f" .viv)" "$n" "$same" "${san:-clean}" "$1" "$2" "$3" "$4" "$5" "$6" >> "$out"
    done
done
