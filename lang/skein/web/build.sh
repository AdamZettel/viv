#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Build the browser page: compile the skein crate to wasm (no wasm-bindgen,
# no emscripten — just the standard wasm32 Rust target) and drop it next to index.html.
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)

cd "$root"
rustup target add wasm32-unknown-unknown >/dev/null 2>&1 || true
cargo build -p skein --lib --target wasm32-unknown-unknown --release
cp target/wasm32-unknown-unknown/release/skein.wasm "$here/skein.wasm"

# Optional size shrink if wasm-opt is on PATH (e.g. from emscripten).
if command -v wasm-opt >/dev/null 2>&1; then
    wasm-opt -Oz "$here/skein.wasm" -o "$here/skein.wasm"
    echo "optimized with wasm-opt"
fi

echo "built $here/skein.wasm ($(wc -c < "$here/skein.wasm") bytes)"
echo "serve it:  (cd $here && python3 -m http.server 8000)  then open http://localhost:8000/"
