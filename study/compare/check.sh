#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Every Python/Viv pair in study/compare, three ways:
#
#   1. viv lint: no integer literal where a size belongs (lang/viv/src/lint.rs);
#   2. check.py: each pair's regions line up, both programs run, and they print
#      the same text (after the drops a pair's notes state); each pair's
#      .gen.json, which the site's /compare pages are built from, is current;
#      and measure.py: the line and lookup counts in a measured pair's notes
#      are current;
#   3. tools/three-engines.sh on each topic: every Viv program's tests pass in
#      the interpreter, compiled to C, and compiled to wasm, with identical output.
#
#     study/compare/check.sh               everything
#     study/compare/check.sh --update      also rewrite the .gen.json files
#     QUICK=1 study/compare/check.sh       skip step 3
#
# PYTHON=<path> picks the Python (it needs NumPy 1.24.4 and matplotlib); the
# default is python3 on the PATH.
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
viv="${VIV:-$root/target/release/viv}"
py="${PYTHON:-python3}"
echo "== viv lint"
"$viv" lint "$here" || exit 1
echo "study/compare: no literal sizes"
echo "== the pairs"
"$py" "$here/check.py" "$@" || exit 1
echo "== the measured tables"
case " $* " in *" --update "*) "$py" "$here/measure.py" --update || exit 1 ;; esac
"$py" "$here/measure.py" --check || exit 1
[ -n "$QUICK" ] && exit 0
echo "== three engines"
# a pair that writes a file writes it in the directory it runs in
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
bad=0
for d in $(cd "$here" && ls -d */ | tr -d /); do
    (cd "$tmp" && "$root/tools/three-engines.sh" "$here/$d") || bad=1
done
exit $bad
