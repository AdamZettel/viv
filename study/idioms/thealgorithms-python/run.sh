#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Run each port in the interpreter and as native C, save both outputs, compare them,
# and compare the interpreter's output with the reference's (ref/<name>.out).
#     ./run.sh              every port
#     ./run.sh sorting      one
# With VIV_WASM=<path to a built viv.wasm> it also compiles each port with the direct
# wasm backend, runs it in node (tools/run-wasm.mjs) and compares that output too.
here=$(cd "$(dirname "$0")" && pwd)
viv=${VIV:-$here/../../../../bin/viv}
cd "$here"
mkdir -p out build
names=${*:-"sorting graphs dp strings maths structures backtracking"}
for n in $names; do
    t0=$(date +%s)
    "$viv" run --quiet "./$n.viv" > "out/$n.interp" 2>&1
    t1=$(date +%s)
    if "$viv" "./$n.viv" -o "build/$n.c" > /dev/null 2> "build/$n.err" && cc -O2 -o "build/$n" "build/$n.c" -lm 2> "build/$n.ccerr"; then
        "./build/$n" > "out/$n.native" 2>&1
        if cmp -s "out/$n.interp" "out/$n.native"; then eng="engines identical"; else eng="ENGINES DIFFER"; fi
    else
        eng="NATIVE DOES NOT BUILD: $(head -1 build/$n.err build/$n.ccerr | grep -v '^==\|^$' | head -1)"
    fi
    if [ -n "$VIV_WASM" ]; then
        if "$viv" "./$n.viv" -o "build/$n.wasm" > /dev/null 2>&1; then
            node "$here/../../../tools/run-wasm.mjs" "build/$n.wasm" > "out/$n.wasm" 2>&1
            if cmp -s "out/$n.interp" "out/$n.wasm"; then eng="$eng, wasm identical"; else eng="$eng, WASM DIFFERS"; fi
        else eng="$eng, WASM DOES NOT COMPILE"; fi
    fi
    pass=$(grep -c '^PASS' "out/$n.interp"); fail=$(grep -c '^FAIL\|^error' "out/$n.interp")
    grep -v '^PASS\|^FAIL\|assertion' "out/$n.interp" > "out/$n.values"
    if [ -f "ref/$n.out" ]; then
        if cmp -s "out/$n.values" "ref/$n.out"; then refm="matches reference"; else refm="DIFFERS FROM REFERENCE"; fi
    else refm="no reference output"; fi
    printf '%-13s %2s pass %s fail  %2ss  %s; %s\n' "$n" "$pass" "$fail" "$((t1 - t0))" "$eng" "$refm"
done
