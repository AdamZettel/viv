#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Run every material's checks, natively, and fail if any test fails.
#
#     dft/check.sh          Si, Al, Ge, C, GaAs on BOTH engines  (~2.5 min)
#     dft/check.sh cu       copper, iterative engine only         (~5 min)
#     dft/check.sh si al    just these, both engines
#
# Each line is one compiled run: tests passed / failed, SCF iterations, seconds.
#
# First, `viv lint` (lang/viv/src/lint.rs): no integer literal where a size
# belongs, in dft/ and in the lib/ modules it uses. Sizes come from len/shape
# of the data or from a named constant.
#
# Then the lenses: dft/lens.sh check reads the .intent
# and .debug files of kmesh, pw_common, pwdft and pwdft_it. A def whose body
# changed since its entry was stamped, a public def with no `what`, a step
# with no text, or a number in the prose with no test cited fails the script
# before any material runs.
here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
[ $# -eq 0 ] && set -- si al ge c gaas
status=0
(cd "$root" && "${VIV:-$root/target/release/viv}" lint dft lib/lobpcg.viv) || exit 1
echo "viv lint dft: no literal sizes"
if ! lens=$("$here/lens.sh" check 2>&1); then
    printf '%s\n' "$lens"
    exit 1
fi
printf '%s\n' "$lens"
for mat in "$@"; do
    engines="dense it"
    [ -f "$here/${mat}_check.viv" ] || engines="it"
    for engine in $engines; do
        start=$(date +%s)
        out=$("$here/run.sh" "$mat" "$engine" 2>&1)
        secs=$(( $(date +%s) - start ))
        pass=$(printf '%s\n' "$out" | grep -c '^PASS')
        fail=$(printf '%s\n' "$out" | grep -c '^FAIL')
        iters=$(printf '%s\n' "$out" | sed -n 's/^ITER: //p')
        if [ "$fail" -ne 0 ] || [ "$pass" -eq 0 ]; then
            status=1
            printf '%-5s %-6s FAILED  (%s pass, %s fail)\n' "$mat" "$engine" "$pass" "$fail"
            printf '%s\n' "$out" | grep -E '^(FAIL|error|eig_herm)' | sed 's/^/        /'
            [ "$pass" -eq 0 ] && [ "$fail" -eq 0 ] && printf '%s\n' "$out" | tail -5 | sed 's/^/        /'
        else
            printf '%-5s %-6s %2s pass   %2s SCF iterations   %4ss\n' "$mat" "$engine" "$pass" "$iters" "$secs"
        fi
    done
done
exit $status
