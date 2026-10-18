#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# The oracle: PySCF 2.5.0's numbers for every case of port.toml, written to
# oracle/<case>.json (energy, orbital energies, cycle counts at the
# verification and the timing settings), then the Viv programs regenerated
# from them. Run from anywhere; one thread.
set -e
here=$(cd "$(dirname "$0")/.." && pwd)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -W ignore "$here/pyscf_side.py" oracle "$@"
python3 "$here/driver.py" gen
