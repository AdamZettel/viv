#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Run the tutorial's programs the way docs/tutorial.md says they run: every
# .viv file through the three engines with identical output, and the Python
# file through py2viv. VIV=<path> and VIV_WASM=<path> as for three-engines.sh.
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
viv=${VIV:-$root/target/release/viv}
sh "$root/tools/three-engines.sh" "$here"
"$viv" run -q "$here/06_rms.py" | grep -q '^PASS test_rms$'
echo "06_rms.py                           py2viv: PASS test_rms"
