#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# The sanitizer and leak gate.
#
#     tools/memcheck/ci.sh [--quick] [OUT_DIR]
#
# 1. Every program in posts, lib, pde, engines, scf, dft (its checks, built
#    with their modules as dft/run.sh builds them), study/compare,
#    study/idioms/syntax-probes and study/memcheck/adversarial is compiled to
#    C as it ships by default (checked: bounds, the solvers' certificates,
#    the `disjoint` pins), built with AddressSanitizer and
#    UndefinedBehaviorSanitizer (-fno-sanitize-recover, so a report stops the
#    run) and run from its own directory under a time limit. Any sanitizer
#    report fails the gate. A program's own exit status does not: a PROBLEM
#    post's tests are meant to fail. Programs that cannot finish under ASan in
#    the limit are listed in tools/memcheck/ci_slow.txt and only compiled and
#    built. On Linux, LSAN=1 turns on LeakSanitizer as well (it does not run on
#    arm64 macOS).
# 2. The leak probes: tools/memcheck/scale.sh (23 allocation patterns at the
#    top level, in a def and through a call) and adversarial.sh (programs
#    written to break the loop arena) at two loop lengths; no row may grow
#    with the length (tools/memcheck/flat.py).
# 3. The parallel probes: the three racing programs must be refused, and a
#    def whose `parallel for` allocates must run in flat memory under OpenMP
#    (when the C compiler has OpenMP).
# 4. The runtime fuzzers (tools/memcheck/fuzz/run.sh) for FUZZ_SECONDS each.
#
# --quick: one program in four of step 1 (the whole of study/memcheck; not
# the DFT checks, which take minutes under ASan), the probes at short
# lengths, and 10 seconds of the fuzzer: a few minutes, for a local check
# before a push. JOBS=<n> programs at once (default 4),
# LIMIT=<s> of CPU time per run (default 300, quick 60; CPU, not wall
# time, so a loaded machine does not stretch it; a run also stops at 10 times
# LIMIT of wall time, which catches a hang that does not spin: TIMEOUT, or
# BLOCKED-LOAD when the machine was busy, tools/gate/load.py), VIV=<path> another compiler,
# CC=<compiler>.
#
# Writes OUT_DIR/asan.tsv (one row per program: path, status, detail) and
# the probes' tables; exits 1 if anything failed.
set -u
root=$(cd "$(dirname "$0")/../.." && pwd)
here="$root/tools/memcheck"
export VIV="${VIV:-$root/target/release/viv}"
export CC="${CC:-cc}"

# one program: compile, build with the sanitizers, run; one row on stdout
if [ "${1:-}" = "one" ]; then
    src=$2
    rel=${src#$root/}
    name=$(basename "$src" .viv)
    pdir=$(dirname "$src")
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    uses=""
    case "$rel" in
    dft/*_it_check.viv) m=${name%_it_check}; uses="--use $pdir/$m.viv --use $pdir/kmesh.viv --use $pdir/pw_common.viv --use $pdir/pwdft_it.viv" ;;
    dft/*_check.viv) m=${name%_check}; uses="--use $pdir/$m.viv --use $pdir/kmesh.viv --use $pdir/pw_common.viv --use $pdir/pwdft.viv" ;;
    esac
    # (ulimit -t: CPU seconds; perl's alarm: the wall-clock cap, 10 times as long)
    if ! (ulimit -t 120; exec perl -e 'alarm shift; exec @ARGV' 1200 "$VIV" $uses "$src" -o "$tmp/p.c") > /dev/null 2> "$tmp/viv.err"; then
        printf '%s\tno-compile\t%s\n' "$rel" "$(head -1 "$tmp/viv.err" | tr '\t' ' ' | cut -c1-160)"
        exit 0
    fi
    if ! nice -n 10 $CC -w -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer -o "$tmp/p" "$tmp/p.c" -lm 2> "$tmp/cc.err"; then
        printf '%s\tCC-FAILED\t%s\n' "$rel" "$(grep -m1 error "$tmp/cc.err" | cut -c1-160)"
        exit 0
    fi
    if grep -qx "$rel" "$here/ci_slow.txt" 2>/dev/null; then
        printf '%s\tslow\tbuilt, not run (ci_slow.txt)\n' "$rel"
        exit 0
    fi
    leak=0
    if [ "${LSAN:-0}" = 1 ]; then leak=1; fi
    # (the shell's own "Alarm clock" report on a run the limit stops is not the program's)
    ( (cd "$pdir" && ulimit -t "$LIMIT" && ASAN_OPTIONS=detect_leaks=$leak:halt_on_error=1:abort_on_error=0 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
        nice -n 10 perl -e 'alarm shift; exec @ARGV' "$((LIMIT * 10))" "$tmp/p" > "$tmp/out" 2> "$tmp/err") ) 2> /dev/null
    code=$?
    if grep -q 'ERROR: AddressSanitizer\|ERROR: LeakSanitizer' "$tmp/err"; then
        printf '%s\tASAN\t%s\n' "$rel" "$(grep -m1 'ERROR: [A-Za-z]*Sanitizer' "$tmp/err" | sed 's/.*Sanitizer: //' | cut -c1-160)"
    elif grep -q 'runtime error:' "$tmp/err"; then
        printf '%s\tUBSAN\t%s\n' "$rel" "$(python3 "$here/ubsan_sites.py" "$tmp/p.c" "$tmp/err" | cut -c1-200)"
    elif [ $code = 152 ] || [ $code = 137 ]; then
        # SIGXCPU (or SIGKILL at the hard limit): the program's own work passed the limit
        printf '%s\tTIMEOUT\tover %ss of CPU under ASan; list it in ci_slow.txt if it is slow, not stuck\n' "$rel" "$LIMIT"
    elif [ $code = 142 ]; then
        # the wall-clock cap, with CPU to spare: stuck, or starved by the machine's load
        if python3 "$root/tools/gate/load.py" busy; then
            printf '%s\tBLOCKED-LOAD\tover %ss of wall time with the machine busy (%s); rerun when the load falls\n' "$rel" "$((LIMIT * 10))" "$(python3 "$root/tools/gate/load.py" show)"
        else
            printf '%s\tTIMEOUT\tover %ss of wall time without using its %ss of CPU: stuck\n' "$rel" "$((LIMIT * 10))" "$LIMIT"
        fi
    else
        printf '%s\tclean\texit %s\n' "$rel" "$code"
    fi
    exit 0
fi

quick=
if [ "${1:-}" = "--quick" ]; then quick=1; shift; fi
out=${1:-$root/target/memcheck-ci}
case "$out" in /*) ;; *) out="$(pwd)/$out" ;; esac
# Two at a time by default: one clang under ASan on a large generated module
# (lib/hf/mdg_kernels) takes several GB, and four at once on a 16 GB machine
# swapped tens of GB to disk. Programs that reach a generated kernel module
# are built one at a time (see `heavy` below).
jobs=${JOBS:-2}
if [ -n "$quick" ]; then limit=${LIMIT:-60}; else limit=${LIMIT:-300}; fi
export LIMIT=$limit
mkdir -p "$out"
[ -x "$VIV" ] || { echo "no viv at $VIV: cargo build --release -p viv" >&2; exit 2; }

fail=0
note() { echo "$*"; }

# ---- 1. the corpus under ASan and UBSan
note "== 1. the corpus under AddressSanitizer and UndefinedBehaviorSanitizer (limit ${limit}s, $jobs at once)"
list="$out/programs.txt"
(cd "$root" && {
    find posts lib pde engines scf study/compare study/idioms/syntax-probes study/memcheck/adversarial -name '*.viv' -not -path '*/wasm/*' -not -path '*/node_modules/*'
    find dft -name '*_check.viv'
} | sort) > "$list.all"
if [ -n "$quick" ]; then
    # the DFT checks take minutes under ASan: the full run has them
    awk '!/^dft\// && (NR % 4 == 1 || /^study\/memcheck\//)' "$list.all" > "$list"
else
    cp "$list.all" "$list"
fi
# the adversarial programs take their loop length as NITER
sed_dir=$(mktemp -d)
trap 'rm -rf "$sed_dir"' EXIT
while read -r p; do
    case "$p" in
    study/memcheck/adversarial/*) sed 's/NITER/200/g' "$root/$p" > "$sed_dir/$(basename "$p")"; echo "$sed_dir/$(basename "$p")" ;;
    *) echo "$root/$p" ;;
    esac
done < "$list" > "$out/paths.txt"
# programs whose expanded source reaches a generated kernel module are heavy to
# compile under ASan: they run one at a time, the rest $jobs at once
: > "$out/heavy.txt"; : > "$out/light.txt"
while read -r f; do
    if "$VIV" expand "$f" 2>/dev/null | grep -q 'def mdg_r_'; then echo "$f" >> "$out/heavy.txt"; else echo "$f" >> "$out/light.txt"; fi
done < "$out/paths.txt"
note "   $(wc -l < "$out/heavy.txt" | tr -d ' ') programs reach a generated kernel module: built one at a time"
if [ "${GATE_CACHE:-}" = 1 ]; then
    # under the gate: a program whose inputs have not changed since its row
    # was clean is not run again (tools/gate/cache.py)
    { JOBS=$jobs python3 "$root/tools/gate/cache.py" memcheck < "$out/light.txt" || fail=1
      JOBS=1 python3 "$root/tools/gate/cache.py" memcheck < "$out/heavy.txt" || fail=1; } > "$out/asan.tsv"
else
    { xargs -P "$jobs" -I{} sh "$0" one {} < "$out/light.txt"; xargs -P 1 -I{} sh "$0" one {} < "$out/heavy.txt"; } > "$out/asan.tsv"
fi
sort -o "$out/asan.tsv" "$out/asan.tsv"
# a run the machine's load stopped is not a verdict: wait for the load to fall, run it again
blocked=0
if grep -q "$(printf '\tBLOCKED-LOAD\t')" "$out/asan.tsv"; then
    nb=$(grep -c "$(printf '\tBLOCKED-LOAD\t')" "$out/asan.tsv")
    note "   $nb runs stopped while the machine was busy; waiting for the load to fall (tools/gate/load.py), then running them again"
    if python3 "$root/tools/gate/load.py" wait 20; then
        grep "$(printf '\tBLOCKED-LOAD\t')" "$out/asan.tsv" | cut -f1 | while read -r rel; do
            case "$rel" in /*) p=$rel ;; *) p="$root/$rel" ;; esac
            sh "$0" one "$p"
        done > "$out/rerun.tsv"
        grep -v "$(printf '\tBLOCKED-LOAD\t')" "$out/asan.tsv" | cat - "$out/rerun.tsv" | sort > "$out/asan.tsv.new"
        mv "$out/asan.tsv.new" "$out/asan.tsv"
    fi
    if grep -q "$(printf '\tBLOCKED-LOAD\t')" "$out/asan.tsv"; then blocked=1; fi
fi
n=$(wc -l < "$out/asan.tsv" | tr -d ' ')
bad=$(awk -F'\t' '$2 == "ASAN" || $2 == "UBSAN" || $2 == "CC-FAILED" || $2 == "TIMEOUT"' "$out/asan.tsv")
awk -F'\t' '{c[$2]++} END {for (k in c) printf "   %-11s %d\n", k, c[k]}' "$out/asan.tsv" | sort
if [ -n "$bad" ]; then
    echo "$bad" | sed 's/^/   FAIL /'
    fail=1
fi
note "   $n programs; table in $out/asan.tsv"

# ---- 2. the leak probes
note "== 2. memory against the length of a loop"
if [ -n "$quick" ]; then s1=300; s2=1200; a1=200; a2=800; else s1=2000; s2=8000; a1=1000; a2=4000; fi
sh "$here/scale.sh" "$out/scale.tsv" $s1 $s2
sh "$here/adversarial.sh" "$out/adversarial.tsv" $a1 $a2
python3 "$here/flat.py" "$out/scale.tsv" "$out/adversarial.tsv" || fail=1

# ---- 3. parallel for
note "== 3. parallel for: the races refused, a pass's allocations freed under OpenMP"
for r in race_same_cell race_neighbor race_through_def; do
    sed 's/NITER/1000/g' "$root/study/memcheck/parallel/$r.viv" > "$sed_dir/p.viv"
    if "$VIV" "$sed_dir/p.viv" -o "$sed_dir/p.c" > /dev/null 2> "$sed_dir/e"; then
        echo "   FAIL $r: accepted"; fail=1
    else
        echo "   ok   $r: $(head -1 "$sed_dir/e")"
    fi
done
omp_flags=""
for f in "-fopenmp" "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp"; do
    printf 'int main(void){return 0;}\n' > "$sed_dir/t.c"
    if $CC $f "$sed_dir/t.c" -o "$sed_dir/t" > /dev/null 2>&1; then omp_flags=$f; break; fi
done
if [ -n "$omp_flags" ]; then
    rss() { python3 -c "import resource,subprocess,sys; subprocess.run(sys.argv[1:], stdout=subprocess.DEVNULL); print(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)" "$@"; }
    for n in 20 200; do
        sed "s/NITER/$n/g" "$root/study/memcheck/parallel/alloc_in_body_def.viv" > "$sed_dir/a$n.viv"
        "$VIV" "$sed_dir/a$n.viv" -o "$sed_dir/a$n.c" > /dev/null
        $CC -w -O2 $omp_flags -o "$sed_dir/a$n" "$sed_dir/a$n.c" -lm
    done
    r20=$(OMP_NUM_THREADS=8 rss "$sed_dir/a20"); r200=$(OMP_NUM_THREADS=8 rss "$sed_dir/a200")
    # ru_maxrss: bytes on macOS, KB on Linux; 16 MB either way is far under
    # the 60 MB the leak added between 50 and 200 calls
    lim=16777216; [ "$(uname)" = Linux ] && lim=16384
    if [ $((r200 - r20)) -lt $lim ]; then echo "   ok   alloc_in_body_def under OpenMP: peak $r20 at 20 calls, $r200 at 200"; else echo "   FAIL alloc_in_body_def under OpenMP: peak $r20 at 20 calls, $r200 at 200"; fail=1; fi
else
    echo "   (no OpenMP in $CC: the OpenMP probe is skipped)"
fi

# ---- 4. the runtime fuzzers
note "== 4. the C runtime fuzzed against the interpreter"
if [ -n "$quick" ]; then secs=${FUZZ_SECONDS:-10}; else secs=${FUZZ_SECONDS:-60}; fi
FUZZ_SECONDS=$secs sh "$here/fuzz/run.sh" "$out/fuzz" || fail=1

if [ $fail = 0 ] && [ $blocked = 1 ]; then
    # EX_TEMPFAIL: the gate reports BLOCKED-LOAD and runs the check again later
    echo "memcheck ci: BLOCKED-LOAD (runs stopped by the machine's load, still busy)"
    exit 75
fi
if [ $fail = 0 ]; then echo "memcheck ci: ok"; else echo "memcheck ci: FAILED"; fi
exit $fail
