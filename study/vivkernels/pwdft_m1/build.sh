#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# PROTOTYPE: the Si program as one C file
# and a shared library Python loads.
#
#     study/vivkernels/pwdft_m1/build.sh          writes pwdft.c, builds libpwdft.so
#
# The maintainer's half needs Viv (it writes the C); the user's half is the
# last line alone: cc on one file.
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
VIV=${VIV:-$root/target/release/viv}
cd "$root"     # relative paths, so no local path lands in the C
"$VIV" --use dft/si.viv --use dft/kmesh.viv --use dft/pw_common.viv --use dft/pwdft_it.viv \
    --use dft/si_it_check.viv study/vivkernels/pwdft_m1/report.viv -o "$here/program.c" > /dev/null
python3 "$here/assemble.py" "$here/program.c" "$here/pwdft.c"
cd "$here"
${CC:-cc} -O2 -ffp-contract=off -fvisibility=hidden -shared -fPIC -Wno-unused-function pwdft.c -o libpwdft.so
