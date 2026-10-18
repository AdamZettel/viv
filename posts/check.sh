#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Every post through `viv lint` (no integer literal where a size belongs:
# lang/viv/src/lint.rs), then through BOTH engines: all tests pass in each,
# and the two print the same thing line for line. Run it before committing a
# post.
#
#     posts/check.sh
#     posts/check.sh 2026-09-21-gibbs-overshoot
#
# The lint is enforced on posts/, dft/, pde/ and all of lib/ here (scf/check.sh
# also holds lib/hf and lib/hfref to it).
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
viv="${VIV:-$root/target/release/viv}"
if [ $# -gt 0 ]; then
    files=$(for n in "$@"; do printf '%s ' "$here/$n.viv"; done)
    # shellcheck disable=SC2086
    "$viv" lint $files || exit 1
else
    echo "== viv lint (enforced)"
    (cd "$root" && "$viv" lint posts dft pde lib) || exit 1
    echo "posts, dft, pde, lib: no literal sizes"
fi
"$root/tools/both-engines.sh" "$here" "$@" || exit 1

# py2viv's translation rules: every rule's generated differential test passes (run where NumPy,
# viv, cc, node and viv.wasm are all present; elsewhere the committed ledger must be current and
# clean), the precedence audit passes, and SUBSET.md's rule tables are the generated ones.
echo "== py2viv rules"
python3 "$root/tools/py2viv/check_rules.py" || exit 1

# The prose against the run: every number a post states, linked to the test,
# printed value or reference that establishes it (www/scripts/claims-build.mjs),
# and a failure where the prose disagrees with what the program computes
# (www/scripts/claims-status.mjs). Orphans fail the whole-directory run, as in
# CI (--strict), and are only reported when one post is named. Needs
# node and the browser compiler (www/build-wasm.sh); skipped without them.
if command -v node > /dev/null && [ -f "$root/www/public/viv/viv.wasm" ]; then
    echo "== claims"
    node "$root/www/scripts/claims-build.mjs" > /dev/null || exit 1
    out=$(mktemp)
    bad=0
    if [ $# -gt 0 ]; then
        for n in "$@"; do
            node "$root/www/scripts/claims-status.mjs" --post "${n#????-??-??-}" > "$out" || bad=1
            sed -n '/^Post claims/p;/^MISMATCH/,/^$/p;/^INCONSISTENT/,/^$/p;/orphan number/p;/^claims:status/p' "$out"
        done
    else
        node "$root/www/scripts/claims-status.mjs" --strict > "$out" || bad=1
        sed -n '/^Post claims/p;/^MISMATCH/,/^$/p;/^INCONSISTENT/,/^$/p;/orphan number/p;/^claims:status/p' "$out"
    fi
    rm -f "$out"
    exit $bad
fi
echo "== claims: skipped (needs node and www/public/viv/viv.wasm)"
