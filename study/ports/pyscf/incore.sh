#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# The two direct cases of port P0 run in core, natively.
#
#     study/ports/pyscf/incore.sh [case ...]     default: benzene_ccpvdz water8_631gs
#
# benzene cc-pVDZ (21.1 million packed integrals) and (H2O)8 6-31G* (54.5
# million) passed the 16,000,000-element array limit every engine had, so
# port.toml runs them direct (INT_JK_DIRECT). Native code now holds arrays
# of up to 2147483647 elements. This script
# takes each case's verification program (viv/<case>.viv), drops the direct
# option so lib/hf keeps the packed integrals in core, builds it natively
# (cc -O2, checks on) and runs it: its tests hold the energy to PySCF's
# within 1e-10 Eh and the orbital energies within 1e-8 Eh, as for every
# case. The in-core programs are written to target/pyscf-incore/ (not in
# git). Exits nonzero on any failure.
set -e
root=$(cd "$(dirname "$0")/../../.." && pwd)
here="$root/study/ports/pyscf"
viv="${VIV:-$root/target/release/viv}"
out="$root/target/pyscf-incore"
mkdir -p "$out"
cd "$root"
[ $# -gt 0 ] || set -- benzene_ccpvdz water8_631gs
bad=0
for c in "$@"; do
    src="$here/viv/$c.viv"
    grep -q 'jk: INT_JK_DIRECT, ' "$src" || { echo "$c: not a direct case"; bad=1; continue; }
    # the `#! use` of the check module names it from the root, which resolves
    # from any program in the tree
    sed 's/jk: INT_JK_DIRECT, //' "$src" > "$out/${c}_incore.viv"
    "$viv" "$out/${c}_incore.viv" -o "$out/${c}_incore.c" > /dev/null
    ${CC:-cc} -O2 -o "$out/${c}_incore" "$out/${c}_incore.c" -lm
    t0=$(date +%s)
    if ! "$out/${c}_incore" > "$out/${c}_incore.out" 2> "$out/${c}_incore.err"; then
        echo "$c in core: FAILED"; cat "$out/${c}_incore.out" "$out/${c}_incore.err" | head -20; bad=1; continue
    fi
    t1=$(date +%s)
    pass=$(grep -c '^PASS ' "$out/${c}_incore.out" || true)
    fail=$(grep -c '^FAIL ' "$out/${c}_incore.out" || true)
    e=$(grep '^energy ' "$out/${c}_incore.out" | cut -d' ' -f2)
    gap=$(grep '^energy_gap ' "$out/${c}_incore.out" | cut -d' ' -f2)
    mo=$(grep '^mo_gap ' "$out/${c}_incore.out" | cut -d' ' -f2)
    printf '%-16s in core: energy %s (|E - PySCF| %s, orbital gap %s), %s tests pass, %s failed, %s s\n' "$c" "$e" "$gap" "$mo" "$pass" "$fail" "$((t1 - t0))"
    [ "$fail" = 0 ] && [ "$pass" != 0 ] || bad=1
done
exit $bad
