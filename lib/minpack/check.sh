#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Everything that checks the MINPACK port, in order; exits nonzero on any
# failure.
#
#     lib/minpack/check.sh            all of it
#     VIV=/path/to/viv lib/minpack/check.sh
#
# 1. The lenses: `viv lens check` on every module
#    with an .intent file, the Rosenbrock story and the post: stamps,
#    coverage (every variable of every def documented: each module says
#    `documents every variable`), cited tests and holds, the reading path
#    of the story, and math KaTeX can typeset. enorm is
#    let through without steps (see below).
#    lmder and lmder1 take a function argument; `viv hash` hashes such a def
#    from its own tokens (it is compiled only as instances, whose bodies
#    depend on the caller), so `viv lens check` sees their stamps go stale
#    like any other def's.
# 2. Every program here and the post through both engines
#    (tools/both-engines.sh): all tests pass in the interpreter and in
#    Viv -> C -> native, and the two print the same bytes.
# 3. The stories of Rosenbrock and Osborne 1, in both engines: every hold
#    passes, and the two engines tell the same story.
# 4. When gfortran is installed: rebuild the Fortran oracle
#    (oracle/build.sh), regenerate mgh_ref.viv from it and require it to
#    equal the committed file; and when scipy is installed, compare the
#    oracle with scipy.optimize.leastsq (oracle/scipy_check.py).
# 5. Print the table of mgh_check.viv: one row per test case.
# 6. `viv lint lib/minpack`: no integer literal where a size belongs
#    (lang/viv/src/lint.rs); sizes come from len/shape or a named constant.
#
# enorm has 31 statements and no steps. A def with steps is recorded in the
# story at every call, and enorm is called several times per trial step and
# once per column in every QR factorization: its accumulation loop is the
# kind of inner loop the story should show as one call with a result. Its
# lines are explained in enorm.intent instead.
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
if [ -z "$VIV" ]; then
    if [ -x "$root/target/release/viv" ]; then VIV="$root/target/release/viv"; else VIV="$root/../bin/viv"; fi
fi
[ -x "$VIV" ] || { echo "no viv binary: set VIV"; exit 2; }
export VIV
status=0
fail() { echo "FAILED: $*"; status=1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$root"

echo "== lenses"
# intent coverage may not fall below lens_coverage.json
if "$VIV" lens coverage --check lib/minpack >/dev/null 2>"$tmp/cov"; then
    echo "lens coverage lib/minpack: holds its ratchet"
else
    grep -v '^note:' "$tmp/cov"
    fail "lens coverage --check lib/minpack"
fi
for m in enorm qrfac qrsolv lmpar lmder rosenbrock_figures story_rosenbrock; do
    out=$("$VIV" lens check "lib/minpack/$m.viv" 2>&1)
    kept=$(printf '%s\n' "$out" | grep -v '^lens check ' | grep -v "enorm.viv:[0-9]*: steps: \`enorm\` has")
    if [ -n "$kept" ]; then
        printf '%s\n' "$kept"
        fail "lens check lib/minpack/$m.viv"
    else
        echo "lens check lib/minpack/$m.viv: ok$( [ $m = enorm ] && echo ' (enorm without steps let through: see check.sh)')"
    fi
done

post=posts/2026-09-25-fitting-electron-mobility-with-minpack.viv
if "$VIV" lens check "$post"; then :; else fail "lens check $post"; fi

echo "== both engines"
# mgh and mgh_ref hold definitions and data only, and have no tests to run
if ! "$root/tools/both-engines.sh" lib/minpack enorm qrfac qrsolv lmpar lmder fit mgh_ls rosenbrock_figures mgh_check story_rosenbrock story_osborne1; then
    fail "tools/both-engines.sh lib/minpack"
fi
if ! "$root/tools/both-engines.sh" posts 2026-09-25-fitting-electron-mobility-with-minpack; then fail "the post"; fi

echo "== stories"
for s in story_rosenbrock story_osborne1; do
    "$VIV" story "lib/minpack/$s.viv" --engine interp > "$tmp/$s.interp" 2>&1
    "$VIV" story "lib/minpack/$s.viv" --engine c > "$tmp/$s.c" 2>&1
    last=$(tail -1 "$tmp/$s.interp")
    case "$last" in
        "holds: "*" checked, 0 failed") ;;
        *) fail "$s: $last" ;;
    esac
    cmp -s "$tmp/$s.interp" "$tmp/$s.c" || fail "$s: the interpreter and C tell different stories"
    printf '%-20s %s; %s iterations; interpreter = C\n' "$s" "$last" "$(grep -c '^    iteration' "$tmp/$s.interp")"
done

echo "== oracle"
if command -v "${FC:-gfortran}" > /dev/null 2>&1; then
    if "$here/oracle/build.sh" "$tmp/oracle"; then
        python3 "$here/oracle/make_ref.py" "$tmp/oracle/trace.txt" > "$tmp/mgh_ref.viv"
        if cmp -s "$tmp/mgh_ref.viv" "$here/mgh_ref.viv"; then
            echo "mgh_ref.viv = the oracle's output, regenerated"
        else
            fail "mgh_ref.viv differs from what the oracle writes now"
        fi
        if python3 -c 'import scipy' 2> /dev/null; then
            if python3 "$here/oracle/scipy_check.py" "$tmp/oracle/plain.txt" > "$tmp/scipy.txt"; then
                echo "scipy.optimize.leastsq against the oracle: $(grep -c . "$tmp/scipy.txt") lines, no case differs ($(grep -c rounding-sensitive "$tmp/scipy.txt") rounding-sensitive)"
            else
                cat "$tmp/scipy.txt"
                fail "scipy_check.py"
            fi
        else
            echo "scipy not installed: skipped the scipy comparison"
        fi
    else
        fail "oracle/build.sh"
    fi
else
    echo "gfortran not installed: skipped rebuilding the oracle"
fi

echo "== lint"
if "$VIV" lint lib/minpack; then echo "viv lint lib/minpack: no literal sizes"; else fail "viv lint lib/minpack"; fi

echo "== the test cases (mgh_check.viv)"
echo "nprob, factor, info, nfev, njev, worst rel. diff in x, worst in the history, trial steps, counts match, history entries that differ, worst column"
"$VIV" run --quiet lib/minpack/mgh_check.viv 2>&1 | sed -n '/^mc_results/,/\]\]/p'

[ $status = 0 ] && echo "lib/minpack/check.sh: all passed" || echo "lib/minpack/check.sh: FAILED"
exit $status
