#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Compile every material's DFT program to WebAssembly: Viv -> C -> wasm.
#
#     dft/build-wasm.sh            build dft/wasm/<mat>.wasm and the manifest  (needs emcc)
#     dft/build-wasm.sh --check    verify the committed wasm is built from TODAY's sources
#                                  (needs only the viv compiler; this is what CI runs)
#
# This is the COMPILED path in the browser: the same C that dft/run.sh hands to cc
# goes to emscripten instead, as a standalone module -- no JavaScript glue, only the
# five WASI calls a printf needs. The website's worker runs these; the interpreter
# (viv.wasm, which reads the .viv source) is its other engine.
#
# The .wasm files are committed because building them needs emscripten, which the
# site's host does not have. What keeps a committed binary from going stale is the
# manifest: it records the SHA-256 of the C each module was compiled from, and
# --check regenerates that C from the current sources and compares. Change a deck,
# the engine, or the compiler, and CI fails until this script is run again.
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
viv="$root/target/release/viv"
out="$here/wasm"
mats="si ge c gaas al cu"
tmp=${TMPDIR:-/tmp}/viv_dft_wasm_$$
mkdir -p "$tmp" "$out"
trap 'rm -rf "$tmp"' EXIT

gen_c() {   # $1 = material -> $tmp/$1.c
    "$viv" --use "$here/$1.viv" --use "$here/kmesh.viv" --use "$here/pw_common.viv" \
        --use "$here/pwdft_it.viv" "$here/$1_it_check.viv" -o "$tmp/$1.c" > /dev/null
    # the manifest hashes this C, so it must be the same on every machine:
    # no path of this checkout in it (the compiler names a program by its
    # file name in the error locations it writes)
    if grep -q "$root" "$tmp/$1.c"; then echo "FAIL: $1.c names a local path; its hash would depend on the checkout" >&2; exit 1; fi
}
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }

if [ "$1" = "--check" ]; then
    status=0
    for m in $mats; do
        gen_c "$m"
        want=$(sed -n "s/.*\"$m\": *{ *\"c_sha256\": *\"\([0-9a-f]*\)\".*/\1/p" "$out/manifest.json")
        got=$(sha "$tmp/$m.c")
        if [ "$want" = "$got" ] && [ -f "$out/$m.wasm" ]; then
            echo "ok    $m.wasm is built from the current sources"
        else
            echo "STALE $m.wasm -- the C it would be compiled from has changed; run dft/build-wasm.sh"
            status=1
        fi
    done
    exit $status
fi

command -v emcc > /dev/null || { echo "emcc not found: source your emsdk_env.sh first" >&2; exit 1; }
ver=$(emcc --version | head -1 | sed 's/.*) \([0-9.]*\).*/\1/')
{
    echo "{"
    echo "  \"_\": \"written by dft/build-wasm.sh; do not edit\","
    echo "  \"emcc\": \"$ver\","
    echo "  \"flags\": \"-O2 -DVIV_RELEASE -sSTANDALONE_WASM=1 -sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=1048576\","
    echo "  \"modules\": {"
} > "$tmp/manifest.json"
first=1
for m in $mats; do
    gen_c "$m"
    # -DVIV_RELEASE: a served module is a release build, so the compiled-in
    # array bounds checks (and their table of source lines) are left out;
    # both engines run the same program with the checks on in dft/check.sh.
    # STACK_SIZE: the fused elementwise reductions (lang/viv/src/fuse.rs) keep
    # about 160 KB of block buffers on the stack; emcc's default 64 KB stack
    # overflows into the heap ("memory access out of bounds")
    emcc -O2 -DVIV_RELEASE "$tmp/$m.c" -o "$out/$m.wasm" -sSTANDALONE_WASM=1 -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=33554432 -sSTACK_SIZE=1048576
    [ $first = 1 ] || echo "," >> "$tmp/manifest.json"
    first=0
    printf '    "%s": { "c_sha256": "%s", "bytes": %s }' "$m" "$(sha "$tmp/$m.c")" "$(wc -c < "$out/$m.wasm" | tr -d ' ')" >> "$tmp/manifest.json"
    echo "built $m.wasm  ($(wc -c < "$out/$m.wasm" | tr -d ' ') bytes)"
done
printf '\n  }\n}\n' >> "$tmp/manifest.json"
mv "$tmp/manifest.json" "$out/manifest.json"
# the no-local-paths promise, checked rather than trusted
if strings "$out"/*.wasm | grep -q "$HOME"; then echo "FAIL: a local path leaked into a wasm module" >&2; exit 1; fi
