#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# How often do three libms disagree -- the platform's C libm, V8's Math, and
# Viv's own (lang/viv/src/vmath/ and its C twin vmath.c, which every engine
# calls)? This runs sin, cos, tan, exp, log, atan, asin, acos, sinh, cosh and
# tanh on a million inputs (five ranges, 200 000 each) and pow on 200 000
# pairs, through each, and counts the results that differ in any bit. The last
# column is Viv's libm in viv.wasm against the same compiled natively from
# vmath.c: the engines' agreement, required to be 0 everywhere.
#
#     tools/libm-sweep/run.sh
#
# Needs node, a C compiler, and www/public/viv/viv.wasm with the runtime
# exports (www/build-wasm.sh).
set -e
here=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"
node "$here/gen.mjs"
${CC:-cc} -O2 -o ref "$here/ref.c" -lm
./ref
${CC:-cc} -O2 -DVIV_VMATH -o vref "$here/ref.c" -lm
./vref
VIV_WASM="${VIV_WASM:-$here/../../www/public/viv/viv.wasm}" node "$here/cmp.mjs"
# the native interpreter: Viv's libm in Rust, against vmath.c; must be 0
if command -v rustc > /dev/null; then
    rustc --edition 2021 -O -o native "$here/native.rs" && ./native
fi
