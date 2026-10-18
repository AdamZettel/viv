#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Memory checks of the C that Viv emits, over whole directories of programs.
#
#     tools/memcheck/memcheck.sh OUT.tsv posts lib pde ...
#
# For every .viv under each directory (recursively), the program is compiled
# to C once and built twice:
#
#   1. with AddressSanitizer and UndefinedBehaviorSanitizer
#      (cc -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer) and run;
#      the first AddressSanitizer report, or every distinct UBSan report
#      named by the C function it is in (ubsan_sites.py), is recorded. LeakSanitizer does not
#      run on arm64 macOS, so this build finds errors, not leaks.
#   2. plainly (cc -O1 -g), from a scratch copy of the C with a memory report
#      appended (tools/memcheck/instrument.py), and run: what the array arena
#      still holds at exit, how much of it the program's globals can reach,
#      the malloc zones' bytes in use at exit and the peak resident set. When
#      that run takes under $LEAKS_MAX seconds it is run again under
#      `leaks --atExit`, which counts blocks no pointer reaches.
#
# Each run is from the program's own directory, under nice, with a time limit
# ($LIMIT seconds, default 300). Nothing in the repository is changed.
#
# The table has one row per program:
#   dir prog compiled asan exit secs arena_n arena_bytes live_n live_bytes
#   dead_bytes zone_in_use max_rss leaks_n leaks_bytes detail
set -u
root=$(cd "$(dirname "$0")/../.." && pwd)
viv="${VIV:-$root/target/release/viv}"
here="$root/tools/memcheck"
limit=${LIMIT:-300}
leaks_max=${LEAKS_MAX:-30}
out=$1; shift
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
run_limited() { # seconds, command...
    s=$1; shift
    nice -n 10 perl -e 'alarm shift; exec @ARGV' "$s" "$@"
}
printf 'dir\tprog\tcompiled\tasan\texit\tsecs\tarena_n\tarena_bytes\tlive_n\tlive_bytes\tdead_bytes\tzone_in_use\tmax_rss\tleaks_n\tleaks_bytes\tdetail\n' > "$out"
for d in "$@"; do
    case "$d" in /*) dir="$d" ;; *) dir="$root/$d" ;; esac
    [ -d "$dir" ] || continue
    for src in $(find "$dir" -name '*.viv' | sort); do
        rel=${src#$root/}
        name=$(basename "$src" .viv)
        pdir=$(dirname "$src")
        c="$tmp/p.c"
        rm -f "$tmp"/p.c "$tmp"/asan "$tmp"/plain "$tmp"/stat "$tmp"/*.err "$tmp"/*.out
        # dft's check programs are built with their material, the k-mesh, the
        # shared code and an engine given as --use modules (dft/run.sh); the
        # modules themselves are not programs and are skipped
        uses=""
        case "$rel" in
        dft/*_it_check.viv) m=${name%_it_check}; uses="--use $pdir/$m.viv --use $pdir/kmesh.viv --use $pdir/pw_common.viv --use $pdir/pwdft_it.viv" ;;
        dft/*_check.viv) m=${name%_check}; uses="--use $pdir/$m.viv --use $pdir/kmesh.viv --use $pdir/pw_common.viv --use $pdir/pwdft.viv" ;;
        dft/kmesh.viv|dft/pw_common.viv|dft/pwdft.viv|dft/pwdft_it.viv) continue ;;
        esac
        if ! run_limited 120 "$viv" $uses "$src" -o "$c" > /dev/null 2> "$tmp/viv.err"; then
            printf '%s\t%s\tno\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t%s\n' "$d" "$rel" "$(head -1 "$tmp/viv.err" | tr '\t' ' ' | cut -c1-120)" >> "$out"
            continue
        fi
        # 1. sanitizers
        asan=clean; detail=""; code=-
        if ! nice -n 10 cc -w -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -o "$tmp/asan" "$c" -lm 2> "$tmp/cc.err"; then
            asan=cc-failed; detail=$(grep -m1 error "$tmp/cc.err" | cut -c1-120)
        else
            (cd "$pdir" && ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=print_stacktrace=1 \
                run_limited "$limit" "$tmp/asan" > "$tmp/asan.out" 2> "$tmp/asan.err")
            code=$?
            if grep -q 'ERROR: AddressSanitizer' "$tmp/asan.err"; then
                asan=ASAN; detail=$(grep -m1 'ERROR: AddressSanitizer' "$tmp/asan.err" | sed 's/.*AddressSanitizer: //' | cut -c1-80)
                loc=$(grep -m1 ' in viv\| in [a-z_]* .*\.c:' "$tmp/asan.err" | sed 's/.* in //' | cut -c1-80)
                detail="$detail @ $loc"
            elif grep -q 'runtime error:' "$tmp/asan.err"; then
                asan=UBSAN; detail=$(python3 "$here/ubsan_sites.py" "$c" "$tmp/asan.err" | cut -c1-400)
            elif [ $code = 142 ]; then
                asan=timeout
            fi
        fi
        # 2. what the run holds at exit
        python3 "$here/instrument.py" "$c" > "$tmp/pm.c"
        stat="- - - - - -"; secs=-; ln_=-; lb_=-
        if nice -n 10 cc -w -O1 -g -o "$tmp/plain" "$tmp/pm.c" -lm 2> "$tmp/cc2.err"; then
            t0=$(perl -MTime::HiRes=time -e 'printf "%.2f", time')
            (cd "$pdir" && VIV_MEMSTAT="$tmp/stat" run_limited "$limit" "$tmp/plain" > "$tmp/plain.out" 2>&1)
            pcode=$?
            t1=$(perl -MTime::HiRes=time -e 'printf "%.2f", time')
            secs=$(echo "$t1 - $t0" | bc)
            [ -s "$tmp/stat" ] && stat=$(cat "$tmp/stat")
            [ "$code" = - ] && code=$pcode
            if [ -s "$tmp/stat" ] && [ "$(echo "$secs < $leaks_max" | bc)" = 1 ]; then
                (cd "$pdir" && run_limited $((leaks_max * 4)) leaks --atExit -- "$tmp/plain" > "$tmp/leaks.out" 2>&1)
                l=$(grep -m1 'leaks for' "$tmp/leaks.out")
                if [ -n "$l" ]; then
                    ln_=$(echo "$l" | sed 's/.*: \([0-9]*\) leaks* for.*/\1/')
                    lb_=$(echo "$l" | sed 's/.* for \([0-9]*\) total.*/\1/')
                fi
            fi
        fi
        set -- $stat
        if [ "$1" = - ]; then dead=-; else dead=$(($2 - $4)); fi
        printf '%s\t%s\tyes\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$d" "$rel" "$asan" "$code" "$secs" "$1" "$2" "$3" "$4" "$dead" "$5" "$6" "$ln_" "$lb_" "$detail" >> "$out"
        set -- "$d"
    done
done
