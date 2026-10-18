#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Prove that py2viv with rules/*.toml translates as the table-in-code translator it replaced,
# except where a listed hole was fixed (tools/py2viv/check_rules_equiv.py has the details).
# Runs under this python3 and under Python 3.12 (PY312), which TheAlgorithms' PEP 695 syntax
# needs and where the builtin sum of a float list is compensated. Arguments go to both runs
# (--only, --base, --random).
set -e
cd "$(dirname "$0")/../.."
PY312=${PY312:-/opt/homebrew/bin/python3.12}
python3 tools/py2viv/check_rules_equiv.py "$@"
if [ -x "$PY312" ]; then
    "$PY312" tools/py2viv/check_rules_equiv.py "$@"
else
    echo "no Python 3.12 at PY312=$PY312: TheAlgorithms not checked"
    exit 1
fi
