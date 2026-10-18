#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Every PDE solver through `viv lint` (no integer literal where a size
# belongs: lang/viv/src/lint.rs), then through BOTH engines: all tests pass in
# each, and the two print the same thing line for line.
#
#     pde/check.sh                all of them
#     pde/check.sh kdv_soliton    one
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
