#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Every engine (dft-bands, hf-energy, precise) through `viv lint`, then
# through BOTH engines: all tests pass in each, and the two print the same
# thing line for line, as pde/check.sh and posts/check.sh do for theirs.
#
#     engines/check.sh               all of them
#     engines/check.sh dft-bands     one
here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
if [ $# -gt 0 ]; then
    files=$(for n in "$@"; do printf '%s ' "$here/$n.viv"; done)
else
    files=$here
fi
# shellcheck disable=SC2086
"${VIV:-$root/target/release/viv}" lint $files || exit 1
exec "$here/../tools/both-engines.sh" "$here" "$@"
