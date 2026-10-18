#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# What `parallel for` guarantees (study/memcheck/parallel/*.viv).
#
#     tools/memcheck/parallel.sh OUT.txt
#
# Each program is run by the interpreter, then compiled and built twice with
# cc -O2: without OpenMP (the pragma is ignored and the loop runs in order)
# and with it (libomp from Homebrew, 8 threads). The OpenMP build runs three
# times. Printed: whether Viv accepted the loop, each result and the peak
# resident set. A race shows as results that differ from run to run and from
# the interpreter.
set -u
root=$(cd "$(dirname "$0")/../.." && pwd)
viv="${VIV:-$root/target/release/viv}"
omp=${LIBOMP:-/opt/homebrew/opt/libomp}
out=$1
# OpenMP: gcc and Linux clang take -fopenmp; Apple clang needs Homebrew's libomp
cc=${CC:-cc}
printf 'int main(void){return 0;}\n' > /tmp/viv_omp_probe_$$.c
if $cc -fopenmp /tmp/viv_omp_probe_$$.c -o /tmp/viv_omp_probe_$$ > /dev/null 2>&1; then
    ompflags="-fopenmp"
else
    ompflags="-Xpreprocessor -fopenmp -I$omp/include -L$omp/lib -lomp"
fi
rm -f /tmp/viv_omp_probe_$$ /tmp/viv_omp_probe_$$.c
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
: > "$out"
rss() { grep 'maximum resident' "$1" | awk '{printf "%d MB", $1 / 1048576}'; }
for f in "$root"/study/memcheck/parallel/*.viv; do
    name=$(basename "$f" .viv)
    for n in 50 200 50000 200000; do
        case $name in alloc_in_body_def) [ $n -le 200 ] || continue ;; *) [ $n -ge 50000 ] || continue ;; esac
        sed "s/NITER/$n/g" "$f" > "$tmp/p.viv"
        interp=$("$viv" run "$tmp/p.viv" 2>&1 | grep -v '^__graph__\|^__trace__' | head -1)
        if ! "$viv" "$tmp/p.viv" -o "$tmp/p.c" > /dev/null 2> "$tmp/e"; then
            echo "$name N=$n: refused by Viv: $(head -1 "$tmp/e")" >> "$out"; continue
        fi
        $cc -w -O2 -o "$tmp/ser" "$tmp/p.c" -lm
        $cc -w -O2 $ompflags -o "$tmp/par" "$tmp/p.c" -lm
        /usr/bin/time -l "$tmp/ser" > "$tmp/o" 2> "$tmp/t"
        line="$name N=$n: accepted; interpreter $interp; C serial $(head -1 "$tmp/o") ($(rss "$tmp/t")); OpenMP x8:"
        for r in 1 2 3; do
            OMP_NUM_THREADS=8 /usr/bin/time -l "$tmp/par" > "$tmp/o" 2> "$tmp/t"
            line="$line $(head -1 "$tmp/o") ($(rss "$tmp/t"))"
        done
        echo "$line" >> "$out"
    done
done
