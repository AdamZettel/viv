#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Does memory grow with the number of loop iterations?
#
#     tools/memcheck/scale.sh OUT.tsv [N1 N2]
#
# For each case of scale_cases.txt, a program evaluates the case's expression
# N times in a loop, in three places: a loop at the top level, a loop inside a
# def, and a scalar loop calling a def that evaluates it once. Each program is
# compiled to C, instrumented (instrument.py) and run at N1 and N2 iterations
# (default 2000 and 8000). The table gives the blocks the array arena holds at
# exit, the bytes the malloc zones hold at exit, and the peak resident set, at
# both sizes. A count that grows with N is a leak proportional to the work,
# not to the live data. The interpreter runs each program first; a case it
# refuses is reported and skipped. The last column says whether the compiled
# program printed what the interpreter printed.
set -u
root=$(cd "$(dirname "$0")/../.." && pwd)
viv="${VIV:-$root/target/release/viv}"
here="$root/tools/memcheck"
out=$1; n1=${2:-2000}; n2=${3:-8000}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
setup='record R {
    xs: Arr
}
def g(v) {
    return v * 2.0
}
def mkmap(k) {
    mm = map()
    mm[k] = 1.0
    return len(mm)
}
def mksmap(k) {
    sm = map()
    sm["key"] = str(k)
    return len(sm["key"])
}
a = linspace(0.0, 1.0, 64)
m = zeros(8, 8)
spd = zeros(8, 8)
for i to 8 {
    for j to 8 {
        m[i, j] = i + 0.5 * j
        spd[i, j] = 1.0 / (1.0 + i + j)
    }
    spd[i, i] = spd[i, i] + 8.0
}
b8 = linspace(1.0, 2.0, 8)
mp = map()
for i to 16 {
    mp[i] = i * 1.0
}'
printf 'case\tplace\tN\tarena_n\tarena_bytes\tzone_in_use\tmax_rss\tinterp\n' > "$out"
grep -v '^#' "$here/scale_cases.txt" | grep . | while IFS='|' read -r name expr red; do
    name=$(echo $name); expr=$(echo "$expr" | sed 's/^ *//; s/ *$//'); red=$(echo $red)
    case $red in id) wrap="$expr" ;; *) wrap="$red($expr)" ;; esac
    for place in top def call; do
        for n in $n1 $n2; do
            p="$tmp/${name}_${place}_$n.viv"
            {
                echo "$setup"
                case $place in
                top)
                    echo "s = 0.0"
                    echo "for k to $n {"
                    echo "    s = s + $wrap"
                    echo "}"
                    echo "print(s)" ;;
                def)
                    echo "def work(n) {"
                    echo "    s = 0.0"
                    echo "    for k to n {"
                    echo "        s = s + $wrap"
                    echo "    }"
                    echo "    return s"
                    echo "}"
                    echo "print(work($n))" ;;
                call)
                    echo "def once(k) {"
                    echo "    return $wrap"
                    echo "}"
                    echo "s = 0.0"
                    echo "for k to $n {"
                    echo "    s = s + once(k)"
                    echo "}"
                    echo "print(s)" ;;
                esac
            } > "$p"
            if ! "$viv" run "$p" > "$tmp/i.raw" 2>&1; then
                cp "$tmp/i.raw" "$tmp/i.out"
                printf '%s\t%s\t%s\t-\t-\t-\t-\t%s\n' "$name" "$place" "$n" "$(grep -m1 error "$tmp/i.out" | cut -c1-100)" >> "$out"
                continue
            fi
            grep -v '^__graph__\|^__trace__' "$tmp/i.raw" > "$tmp/i.out"
            if ! "$viv" "$p" -o "$tmp/p.c" > /dev/null 2> "$tmp/v.err"; then
                printf '%s\t%s\t%s\t-\t-\t-\t-\tC: %s\n' "$name" "$place" "$n" "$(head -1 "$tmp/v.err" | cut -c1-100)" >> "$out"
                continue
            fi
            python3 "$here/instrument.py" "$tmp/p.c" > "$tmp/pm.c"
            nice -n 10 cc -w -O1 -o "$tmp/pm" "$tmp/pm.c" -lm
            VIV_MEMSTAT="$tmp/stat" nice -n 10 "$tmp/pm" > "$tmp/c.out" 2>&1
            if ! cmp -s "$tmp/i.out" "$tmp/c.out"; then same="output differs"; else same=ok; fi
            set -- $(cat "$tmp/stat")
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$place" "$n" "$1" "$2" "$5" "$6" "$same" >> "$out"
        done
    done
done
