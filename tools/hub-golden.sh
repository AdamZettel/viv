#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# The golden snapshot of the knotkit hub migration : for every program in the corpus, what the compiler and
# the three engines make of it, written to a directory so that the snapshot
# from before a change can be compared with the one after.
#
#     tools/hub-golden.sh record <dir> [paths...]   write a snapshot
#     tools/hub-golden.sh compare <dir-a> <dir-b>   list every difference; exit 1 if any
#
# With no paths, the corpus is every .viv under posts, pde, lib, engines, scf,
# dft and study/idioms (not the committed wasm folders). A path may be a
# directory or one .viv file. Keep <dir> out of git: target/golden/<name> is
# ignored by .gitignore's /target.
#
# For each program P (named by its path from the repository root) the snapshot
# holds, in <dir>/P/:
#
#   hash      `viv hash P`: every def's content address
#   ast.gz    `viv P --emit ast` (gzip -n; the compare unpacks it)
#   ir.gz     `viv P --emit ir`
#   c.gz      the C that `viv P -o x.c` writes
#   c.log     what that command printed, and its exit status
#   wasm      the sha256 of the module `viv P -o x.wasm` writes, and what it printed
#   interp    `viv run P`: stdout, then stderr, then the exit status
#   native    the exact build (`viv build`: cc -O3, the modules' objects
#             from the cache) run: the same three
#   direct    the wasm module run by tools/run-wasm.mjs: the same three
#
# A timing program (a line `#! timing`: it may read the clock) has its static
# artifacts recorded and its runs left out.
#
# Each file ends with a line `exit <status>` for the command that made it. A
# program that does not compile is recorded as such (its diagnostics are the
# snapshot); it is not an error here.
#
# What is left out or rewritten, because it would differ between two runs of
# one tree:
#
# - The temporary directory the C and the wasm are written to: the lines
#   `wrote <path> (...)` have that directory replaced by `$TMP`.
# - The order of the functions in the interpreter's `__graph__` line, which
#   comes from a hash map: the list is sorted.
# - Wall-clock time: no timing is recorded. A run that takes longer than
#   RUN_TIMEOUT seconds of CPU (default 120; wall time capped at 10 times
#   that) is killed and recorded as `timeout`,
#   which is the one entry that can differ between two runs of one tree; the
#   compare lists it separately. The programs in tools/hub-golden.slow are not
#   run at all (their static artifacts are still recorded) unless FULL=1.
# - Nothing about the machine: programs are named by repository-relative
#   paths, which is also the name the C's location table uses.
#
# Switches: NO_RUN=1 records the static artifacts only (hash, ast, ir, C,
# wasm). NO_C=1 skips building and running the C. JOBS=<n> programs at once
# (default 4). VIV=<path> another viv binary; VIV_WASM=<path> another
# viv.wasm for run-wasm.mjs (default www/public/viv/viv.wasm, built by
# www/build-wasm.sh). CC=<compiler> for the C.
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
viv="${VIV:-$root/target/release/viv}"
export VIV_WASM="${VIV_WASM:-$root/www/public/viv/viv.wasm}"

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

# run <out-file> <cmd...>: stdout, stderr and the status of one command
run() {
    o=$1; shift
    set +e
    # in a subshell whose stderr is dropped: the shell's own "Alarm clock"
    # report on a killed run is not the program's output
    # The limit is RUN_TIMEOUT seconds of CPU (ulimit -t), which the machine's
    # load does not stretch, with a wall-clock cap of 10 times that for a run
    # that blocks (tools/gate/load.py)
    lim=${RUN_TIMEOUT:-120}
    (ulimit -t "$lim"; exec perl -e 'alarm shift; exec @ARGV' "$((lim * 10))" "$@" > "$o.stdout" 2> "$o.stderr") 2> /dev/null
    st=$?
    set -e
    # perl's alarm kills with SIGALRM (142), the CPU limit with SIGXCPU (152) or SIGKILL (137)
    case "$st" in 142|152|137) st=timeout ;; esac
    { cat "$o.stdout"; echo "--- stderr"; cat "$o.stderr"; echo "exit $st"; } > "$o"
    rm -f "$o.stdout" "$o.stderr"
}

# sort_graph <file>: the interpreter's `__graph__` line lists the functions
# in the order of a hash map, which changes from run to run; sorted here. (Its
# edges are already sorted, and `__trace__` is in call order.)
sort_graph() {
    perl -i -pe 's/^(__graph__ \{"nodes":\[)([^\]]*)/$1 . join(",", sort split(",", $2))/e' "$1"
}

# one <program> <dir>: the snapshot of one program
one() {
    p=$1; out="$2/$p"
    mkdir -p "$out"
    tmp=$(mktemp -d)
    b=$(basename "$p" .viv)
    cd "$root"
    run "$out/hash" "$viv" hash "$p"
    run "$out/ast" "$viv" "$p" --emit ast
    run "$out/ir" "$viv" "$p" --emit ir
    run "$out/c.log" "$viv" "$p" -o "$tmp/$b.c"
    if [ -f "$tmp/$b.c" ]; then cp "$tmp/$b.c" "$out/c"; else echo "(no C)" > "$out/c"; fi
    # the AST of a program with its integral tables inlined runs to 100 MB of text:
    # kept compressed (gzip -n: no name or time in the header, so the bytes repeat)
    gzip -n -f "$out/ast" "$out/ir" "$out/c"
    run "$out/wasm" "$viv" "$p" -o "$tmp/$b.wasm"
    if [ -f "$tmp/$b.wasm" ]; then
        echo "sha256 $(shasum -a 256 < "$tmp/$b.wasm" | cut -d' ' -f1)" >> "$out/wasm"
    fi
    # the temporary directory is the one part of these logs that changes between runs
    for f in c.log wasm; do
        sed "s|$tmp|\$TMP|g" "$out/$f" > "$out/$f.n" && mv "$out/$f.n" "$out/$f"
    done
    skip=
    if [ -z "$FULL" ] && grep -qx "$p" "$root/tools/hub-golden.slow" 2>/dev/null; then skip=1; fi
    # a timing program (`#! timing`) reads the clock: its output differs
    # from run to run, so its runs are not recorded
    if grep -qx '#! timing' "$p"; then skip=1; fi
    if [ -n "$NO_RUN" ] || [ -n "$skip" ]; then
        rm -rf "$tmp"
        return 0
    fi
    run "$out/interp" "$viv" run "$p"
    sort_graph "$out/interp"
    if [ -z "$NO_C" ] && [ -f "$tmp/$b.c" ]; then
        if "$viv" build "$p" -o "$tmp/$b" > /dev/null 2> "$tmp/cc.err"; then
            run "$out/native" "$tmp/$b"
        else
            { echo "does not build (cc)"; grep -m 3 error "$tmp/cc.err" | sed "s|$tmp|\$TMP|g"; } > "$out/native"
        fi
    fi
    if [ -f "$tmp/$b.wasm" ]; then
        run "$out/direct" node "$root/tools/run-wasm.mjs" "$tmp/$b.wasm"
    fi
    rm -rf "$tmp"
}

programs() {
    cd "$root"
    if [ $# -eq 0 ]; then set -- posts pde lib engines scf dft study/idioms; fi
    for a in "$@"; do
        a=${a%/}
        if [ -d "$a" ]; then
            find "$a" -name '*.viv' -not -path '*/wasm/*' -not -path '*/node_modules/*'
        else
            echo "$a"
        fi
    done | sort
}

cmd=$1
[ -n "$cmd" ] || usage
shift
case "$cmd" in
one)
    one "$1" "$2"
    ;;
record)
    [ -n "$1" ] || usage
    dir=$1; shift
    case "$dir" in /*) ;; *) dir="$(pwd)/$dir" ;; esac
    [ -x "$viv" ] || { echo "no viv at $viv: cargo build --release -p viv" >&2; exit 2; }
    [ -n "$NO_RUN" ] || [ -f "$VIV_WASM" ] || { echo "no viv.wasm at $VIV_WASM: www/build-wasm.sh" >&2; exit 2; }
    rm -rf "$dir"
    mkdir -p "$dir"
    programs "$@" > "$dir/.programs"
    # what the snapshot was made with; not compared
    {
        echo "made $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "tree $(git -C "$root" rev-parse HEAD)$(git -C "$root" diff --quiet HEAD || echo ' (with uncommitted changes)')"
        echo "viv $(shasum -a 256 < "$viv" | cut -d' ' -f1)"
        [ -f "$VIV_WASM" ] && echo "viv.wasm $(shasum -a 256 < "$VIV_WASM" | cut -d' ' -f1)"
        echo "NO_RUN=$NO_RUN NO_C=$NO_C FULL=$FULL RUN_TIMEOUT=${RUN_TIMEOUT:-120}"
    } > "$dir/.meta"
    n=$(wc -l < "$dir/.programs" | tr -d ' ')
    echo "recording $n programs into $dir"
    if [ "${GATE_CACHE:-}" = 1 ]; then
        # under the gate: a program whose inputs have not changed is not
        # recorded again; its files come from tools/gate/cache.py's store
        (cd "$root" && python3 tools/gate/cache.py golden "$dir" < "$dir/.programs") || exit 1
    else
        xargs -P "${JOBS:-4}" -I{} sh "$0" one {} "$dir" < "$dir/.programs"
    fi
    echo "done: $n programs"
    ;;
compare)
    a=$1; b=$2
    [ -d "$a" ] && [ -d "$b" ] || usage
    a=$(cd "$a" && pwd); b=$(cd "$b" && pwd)
    bad=0
    if ! cmp -s "$a/.programs" "$b/.programs"; then
        echo "the program lists differ:"
        diff "$a/.programs" "$b/.programs" | sed 's/^/  /'
        bad=1
    fi
    files=$( (cd "$a" && find . -type f -not -name '.*'; cd "$b" && find . -type f -not -name '.*') | sort -u)
    same=0; diffs=0; timeouts=0
    for f in $files; do
        if [ ! -f "$a/$f" ] || [ ! -f "$b/$f" ]; then
            echo "only in one snapshot: ${f#./}"; bad=1; diffs=$((diffs + 1)); continue
        fi
        if cmp -s "$a/$f" "$b/$f"; then same=$((same + 1)); continue
        fi
        if grep -qx 'exit timeout' "$a/$f" || grep -qx 'exit timeout' "$b/$f"; then
            echo "timeout in one snapshot (not a difference in the program): ${f#./}"
            timeouts=$((timeouts + 1)); continue
        fi
        echo "DIFFERS: ${f#./}"
        case "$f" in
        *.gz)
            t=$(mktemp -d)
            gunzip -c "$a/$f" > "$t/a"; gunzip -c "$b/$f" > "$t/b"
            diff "$t/a" "$t/b" | head -${DIFF_LINES:-12} | cut -c1-200 | sed 's/^/    /'
            rm -rf "$t"
            ;;
        *) diff "$a/$f" "$b/$f" | head -${DIFF_LINES:-12} | cut -c1-200 | sed 's/^/    /' ;;
        esac
        diffs=$((diffs + 1)); bad=1
    done
    echo "$same files identical, $diffs differ, $timeouts timed out in one snapshot"
    exit $bad
    ;;
*)
    usage
    ;;
esac
