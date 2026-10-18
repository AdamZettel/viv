#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Port P0's correctness check: lib/hf against PySCF 2.5.0
#.
#
#     study/ports/pyscf/check.sh
#
# 1. The generated programs in viv/ are current with port.toml and oracle/
#    (`port.py gen pyscf` writes them).
# 2. The parity cases (port.toml, `parity = true`) and the check module run
#    in three engines, interpreter, native and wasm, and must pass their
#    tests and print the same bytes (tools/three-engines.sh).
# 3. Every case runs natively at the verification settings: its tests pass
#    (energy within 1e-10 Eh of PySCF's, orbital energies within 1e-8 Eh),
#    and the harness compares the printed energy with oracle/<case>.json a
#    second time. results/verify.json keeps the table.
#
# The timing programs (<case>_t.viv) carry their own test, the energy within
# 1e-8 Eh of the oracle, which `port.py time` requires before a time counts.
# Exits nonzero on any failure.
set -e
root=$(cd "$(dirname "$0")/../../.." && pwd)
here="$root/study/ports/pyscf"
cd "$root"

python3 "$here/driver.py" gen --check

parity=$(python3 -c "
import tomllib
c = tomllib.load(open('$here/port.toml', 'rb'))
print(' '.join(x['name'] for x in c['case'] if x.get('parity')))")
tools/three-engines.sh study/ports/pyscf/viv pzcheck $parity

python3 "$here/driver.py" native
