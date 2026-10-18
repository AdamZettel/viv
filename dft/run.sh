#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Compile one material's deck + an engine + its check file to C, build, run.
#
#     dft/run.sh si            the dense engine      (dft/pwdft.viv,    dft/si_check.viv)
#     dft/run.sh si it         the iterative engine  (dft/pwdft_it.viv, dft/si_it_check.viv)
#
# Needs a built compiler (cargo build --release -p viv) and a C compiler.
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
mat=$1
if [ "${2:-dense}" = it ]; then engine=pwdft_it; check=${mat}_it_check; else engine=pwdft; check=${mat}_check; fi
out=${TMPDIR:-/tmp}/viv_dft_${mat}_${engine}
"$root/target/release/viv" --use "$here/$mat.viv" --use "$here/kmesh.viv" --use "$here/pw_common.viv" \
    --use "$here/$engine.viv" "$here/$check.viv" -o "$out.c" > /dev/null
${CC:-cc} -O2 -o "$out" "$out.c" -lm
shift; [ $# -gt 0 ] && shift
exec "$out" "$@"
