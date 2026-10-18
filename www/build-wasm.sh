#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Build the browser compiler: the `viv` crate's cdylib to wasm32-unknown-unknown.
# No emscripten and no wasm-bindgen — lib.rs exposes a hand-rolled byte-buffer
# ABI (viv_alloc / viv_run / viv_compile / viv_doc / ...) that the site calls
# directly through WebAssembly.instantiate.
#
# Produces www/public/viv/viv.wasm (and the embed's copy). The site build does
# not need Rust; CI runs this and publishes the artifact.
set -e

WWW=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$WWW/.." && pwd)
OUT="$WWW/public/viv"
mkdir -p "$OUT" "$WWW/public/embed"

echo "[1/2] cargo: viv crate -> wasm32 cdylib"
cd "$ROOT"
rustup target add wasm32-unknown-unknown >/dev/null 2>&1 || true
# rustc embeds absolute source paths in panic messages, and this wasm is served
# publicly, so remap them to neutral roots and ship no local path.
# When several prefixes match, rustc applies the LAST one given, so the list
# runs from the most general prefix to the most specific. ($HOME used to come
# last, which turned the repo root into "~/Documents/..." in the wasm.)
TARGET="${CARGO_TARGET_DIR:-$ROOT/target}"
REMAP="--remap-path-prefix=$HOME=home"
REMAP="$REMAP --remap-path-prefix=$HOME/.cargo=cargo --remap-path-prefix=$HOME/.rustup=rustup"
REMAP="$REMAP --remap-path-prefix=$(dirname "$ROOT")=src"
REMAP="$REMAP --remap-path-prefix=$TARGET=target --remap-path-prefix=$ROOT=viv"
RUSTFLAGS="$REMAP" \
  cargo build -p viv --lib --target wasm32-unknown-unknown --release
WASM="$TARGET/wasm32-unknown-unknown/release/viv.wasm"

echo "[2/2] wasm-opt: shrink (if available)"
WASMOPT=$(command -v wasm-opt || ls "$HOME"/emsdk/upstream/bin/wasm-opt 2>/dev/null | head -1)
if [ -n "$WASMOPT" ]; then
  "$WASMOPT" -Oz "$WASM" -o "$OUT/viv.wasm"
else
  echo "      (wasm-opt not found — shipping unoptimized)"
  cp "$WASM" "$OUT/viv.wasm"
fi
cp "$OUT/viv.wasm" "$WWW/public/embed/viv.wasm"

# Guard the no-local-paths promise. Any home, temp or user directory in the
# served wasm fails the build, in absolute or home-relative form. A bare "~/"
# also appears in the compiler's own help text ("~/.cache/viv"), so the
# home-relative check looks for a source file or a home folder after it.
LEAK='/Users/|/home/|/private/tmp|/private/var|/var/folders|Documents/|portfolio_projects'
LEAK="$LEAK"'|~/[^ ]*\.(rs|c|h|py|viv)\b|~/(Documents|Desktop|Downloads|Library|src|code|projects)/'
for f in "$OUT/viv.wasm" "$WWW/public/embed/viv.wasm"; do
  if strings "$f" | grep -E "$LEAK|$HOME" >&2; then
    echo "FAIL: a local path leaked into $f (lines above)" >&2
    exit 1
  fi
done
echo "done: $OUT/viv.wasm ($(wc -c < "$OUT/viv.wasm") bytes; embed copy synced)"
