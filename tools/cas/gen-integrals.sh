#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Write lib/hf/mdg_kernels.viv, the McMurchie-Davidson kernels lib/cas
# generates (tools/cas/gen_integrals.viv), and its
# intent lens lib/hf/mdg_kernels.intent, each def stamped with its body hash
# (tools/cas/gen_intent.py).
#
#     tools/cas/gen-integrals.sh            regenerate the module
#     tools/cas/gen-integrals.sh --check    exit 1 when the committed module or
#                                           its lens is not what they print
#
# The generator runs in the interpreter (about 3 s). It stops without printing
# when its proof that the E recurrence equals the direct expansion fails.
# VIV=<path> runs a different viv binary.
set -e
root=$(cd "$(dirname "$0")/../.." && pwd)
viv="${VIV:-$root/target/release/viv}"
out="$root/lib/hf/mdg_kernels.viv"
lens="$root/lib/hf/mdg_kernels.intent"
tmp=$(mktemp -d)/mdg_kernels.viv
trap 'rm -rf "$(dirname "$tmp")"' EXIT
(cd "$root" && "$viv" run tools/cas/gen_integrals.viv) 2>&1 | grep -v '^__graph__\|^__trace__\|^__steps__\|^__story__' > "$tmp"
if ! head -1 "$tmp" | grep -q '^###$' || ! tail -1 "$tmp" | grep -q '^# generated: '; then
    echo "gen-integrals: the generator failed:" >&2
    cat "$tmp" >&2
    exit 1
fi
"$viv" lens init "$tmp" --stdout | python3 "$root/tools/cas/gen_intent.py" > "$tmp.intent"
if [ "${1:-}" = "--check" ]; then
    if cmp -s "$tmp" "$out" && cmp -s "$tmp.intent" "$lens"; then
        echo "lib/hf/mdg_kernels.viv and .intent are what tools/cas/gen_integrals.viv prints"
        exit 0
    fi
    echo "lib/hf/mdg_kernels.viv or .intent is stale: run tools/cas/gen-integrals.sh" >&2
    diff "$out" "$tmp" | head -20 >&2 || true
    diff "$lens" "$tmp.intent" | head -20 >&2 || true
    exit 1
fi
cp "$tmp" "$out"
cp "$tmp.intent" "$lens"
echo "wrote lib/hf/mdg_kernels.viv ($(wc -l < "$out" | tr -d ' ') lines) and its .intent"
