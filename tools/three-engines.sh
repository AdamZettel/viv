#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Run every Viv program in a directory through THREE engines -- the
# interpreter, Viv -> C -> native, and Viv -> wasm by the direct backend, run
# in node -- and require every test to pass in each and all three outputs to
# agree byte for byte. tools/both-engines.sh checks the first two; this is it
# plus engine 3.
#
#     tools/three-engines.sh pde                 every .viv in pde/
#     tools/three-engines.sh posts 2026-09-21-gibbs-overshoot
#
# Needs a built compiler (cargo build --release -p viv), a C compiler, node,
# and a viv.wasm with the runtime exports (www/build-wasm.sh; VIV_WASM=<path>
# for another). VIV=<path> runs a different viv binary. NO_C=1 skips engine 2
# (a quick interpreter-vs-wasm check). A PROBLEM post (`#! kind problem`) is a
# program whose tests are meant to fail; for it only the agreement is required.
#
# A program that reads files names each on a line `#! file <path>`, the path
# from the repository root: the engines then run from the root, and the wasm
# run attaches each file under that name, as a page does.
#
# All three compute sin, exp, pow and the rest with Viv's own libm
# (lang/viv/src/vmath/ in the interpreter and the wasm runtime, its C twin
# vmath.c in a compiled program), so there is no allowance for rounding: any
# byte that differs fails the program.
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
dir=$1; shift
case "$dir" in /*) ;; *) dir="$root/$dir" ;; esac
viv="${VIV:-$root/target/release/viv}"
tmp=$(mktemp -d)
export VIV_JOBS=${VIV_JOBS:-2}
trap 'rm -rf "$tmp"' EXIT
if [ $# -gt 0 ]; then names="$*"; else names=$(cd "$dir" && ls *.viv | sed 's/\.viv$//'); fi
bad=0
for name in $names; do
    src="$dir/$name.viv"
    # a timing program reads the clock (clock()): its output differs from
    # run to run and is not compared
    if grep -qx '#! timing' "$src"; then
        printf '%-34s a timing program (#! timing): not compared\n' "$name"
        continue
    fi
    # files the program reads (`#! file <path>`, from the root): run there,
    # and attach each to the wasm run
    attach=""
    for f in $(sed -n 's/^#! file //p' "$src"); do attach="$attach --file=$f=$root/$f"; done
    [ -z "$attach" ] || cd "$root"
    # the interpreter also prints its call graph and trace; the program's own output is the rest
    "$viv" run "$src" 2>&1 | grep -v '^__graph__\|^__trace__\|assertions\? FAILED$\|holds\? FAILED$' > "$tmp/$name.interp" || true
    engines="interp"
    if [ -z "$NO_C" ]; then
        # `viv build`: the exact build, its modules' objects from the cache
        if ! "$viv" build "$src" -o "$tmp/$name" > /dev/null 2> "$tmp/$name.err"; then
            if grep -q 'rejected the generated C\|could not link' "$tmp/$name.err"; then
                printf '%-34s DOES NOT BUILD (cc)\n' "$name"; grep -m 3 'error' "$tmp/$name.err" | cut -c1-170
            else
                printf '%-34s DOES NOT COMPILE (C)\n' "$name"; head -6 "$tmp/$name.err"
            fi
            bad=1; continue
        fi
        "$tmp/$name" 2>&1 | grep -v 'assertions\? FAILED$\|holds\? FAILED$' > "$tmp/$name.native" || true
        engines="$engines native"
    fi
    if ! "$viv" "$src" -o "$tmp/$name.wasm" > /dev/null 2> "$tmp/$name.werr"; then
        printf '%-34s DOES NOT COMPILE (wasm)\n' "$name"; head -6 "$tmp/$name.werr"; bad=1; continue
    fi
    node "$root/tools/run-wasm.mjs" $attach "$tmp/$name.wasm" 2>&1 | grep -v 'assertions\? FAILED$\|holds\? FAILED$' > "$tmp/$name.wasm.out" || true
    mv "$tmp/$name.wasm.out" "$tmp/$name.direct"
    engines="$engines direct"
    pass=$(grep -c '^PASS ' "$tmp/$name.direct" || true)
    fail=0
    for e in $engines; do
        f=$(grep -c '^FAIL \|^error' "$tmp/$name.$e" || true)
        fail=$((fail + f))
    done
    same="identical output"
    for e in $engines; do
        [ "$e" = interp ] && continue
        cmp -s "$tmp/$name.interp" "$tmp/$name.$e" && continue
        same="OUTPUT DIFFERS ($e)"; bad=1
    done
    if grep -q '^#! kind problem' "$src" || grep -q '^kind: problem' "$src"; then
        [ "$fail" != 0 ] || bad=1
        unsolved=$(grep -c '^FAIL ' "$tmp/$name.direct" || true)
        printf '%-34s %2s tests pass, %s to solve (a problem); %s\n' "$name" "$pass" "$unsolved" "$same"
    else
        [ "$fail" = 0 ] || bad=1
        # a module whose tests all need reference data keeps them in its
        # companion file, <dir>/tests/<name>.viv, which is checked on its own
        [ "$pass" != 0 ] || [ -f "$dir/tests/$name.viv" ] || bad=1
        printf '%-34s %2s tests pass, %s failed; %s\n' "$name" "$pass" "$fail" "$same"
        if [ "$fail" != 0 ]; then for e in $engines; do cat "$tmp/$name.$e"; done | grep '^FAIL \|^error' | sort -u | cut -c1-170 | head -6; fi
    fi
    for e in $engines; do
        [ "$e" = interp ] && continue
        [ "$e" = direct ] && [ -f "$tmp/$name.iwasm" ] && cmp -s "$tmp/$name.iwasm" "$tmp/$name.direct" && continue
        cmp -s "$tmp/$name.interp" "$tmp/$name.$e" || { echo "  interp vs $e:"; diff "$tmp/$name.interp" "$tmp/$name.$e" | cut -c1-160 | head -8; }
    done
done
exit $bad
