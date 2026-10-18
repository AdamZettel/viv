#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# lib/gguf against gguf-py, on the fixture and on the tiny models when they
# are here; exits nonzero on any difference.
#
#     tools/gguf/check.sh
#     GGUF_MODELS=<dir> GGUF_PY=<python with gguf> VIV=<viv> tools/gguf/check.sh
#
# The models are llama2.c's tinyllamas in GGUF (huggingface.co/ggml-org/models,
# MIT): stories260K.gguf is in the repository (lib/gguf/tests);
# stories15M-q4_0.gguf and stories15M-q8_0.gguf are read from $GGUF_MODELS
# (default ~/models/gguf), and a missing one is skipped. The reference is gguf-py (pip install gguf==0.19.0)
# run by $GGUF_PY (default ~/.venvs/ggml/bin/python): tools/gguf/ref.py prints
# what it reads in the form lib/gguf prints it, and each program's output
# must equal it byte for byte, in the interpreter, as C and as wasm.
set -e
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
viv="${VIV:-$root/target/release/viv}"
py="${GGUF_PY:-$HOME/.venvs/ggml/bin/python}"
models="${GGUF_MODELS:-$HOME/models/gguf}"
if ! "$py" -c "import gguf" 2>/dev/null; then
    echo "skipped: no gguf-py at $py (GGUF_PY=...)"
    exit 0
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
bad=0
files="lib/gguf/tests/tiny.gguf lib/gguf/tests/stories260K.gguf"
for m in stories15M-q4_0 stories15M-q8_0; do
    if [ -f "$models/$m.gguf" ]; then files="$files $models/$m.gguf"; else echo "skipped: $models/$m.gguf is not here"; fi
done
# the ggml types lib/gguf reads; tools/gguf/values.viv compares each tensor
# of these types with gguf-py's dequantize, number for number
TYPES="F32,F16,BF16,Q8_0,Q4_0,Q4_1,Q4_K,Q6_K"
# tools/gguf/<prog>.viv in the three engines, with the --set arguments given;
# the wasm run attaches the files in $attach (paths without spaces)
run3() {
    prog=$1; shift
    nice "$viv" run -q "$@" "tools/gguf/$prog.viv" > "$tmp/interp.txt"
    # a build reads MODEL as it was compiled: one per file
    nice "$viv" build "$@" "tools/gguf/$prog.viv" -o "$tmp/$prog" > /dev/null
    "$tmp/$prog" > "$tmp/native.txt"
    nice "$viv" "$@" "tools/gguf/$prog.viv" -o "$tmp/$prog.wasm" > /dev/null
    node tools/run-wasm.mjs $attach "$tmp/$prog.wasm" > "$tmp/wasm.txt"
}
# one check: tools/gguf/<prog>.viv on each file, against `ref.py <prog>`
check() {
    prog=$1
    for f in $files; do
        name=$(basename "$f" .gguf)
        attach="--file=$f=$f"
        if [ "$prog" = values ]; then
            ref="$tmp/ref_$name"; rm -rf "$ref"; mkdir -p "$ref"
            "$py" tools/gguf/ref.py values "$f" "$ref" "$TYPES" > "$tmp/ref.txt"
            for r in "$ref"/*.f32; do [ -f "$r" ] && attach="$attach --file=$r=$r"; done
            run3 values --set "MODEL=\"$f\"" --set "REFDIR=\"$ref\"" --set "TYPES=\"$TYPES\""
            rm -rf "$ref"
        else
            "$py" tools/gguf/ref.py "$prog" "$f" > "$tmp/ref.txt"
            run3 "$prog" --set "MODEL=\"$f\""
        fi
        for e in interp native wasm; do
            if cmp -s "$tmp/$e.txt" "$tmp/ref.txt"; then
                printf '%-8s %-20s %-7s identical to gguf-py (%s lines)\n' "$prog" "$name" "$e" "$(wc -l < "$tmp/ref.txt" | tr -d ' ')"
            else
                printf '%-8s %-20s %-7s DIFFERS from gguf-py\n' "$prog" "$name" "$e"
                diff "$tmp/$e.txt" "$tmp/ref.txt" | head -6
                bad=1
            fi
        done
    done
}
check meta
check tensors
check values
exit $bad
