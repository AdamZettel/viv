#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Build every platform wheel of vipyr on this machine: the core (with the
# bundled `viv` and its programs precompiled) and each distribution under
# packages/ and ports/ that prebuilds programs. Used by
# .github/workflows/wheels.yml on each platform, and by hand:
#
#     tools/wheels/build_all.sh [OUT]        (default: dist/)
#
# Needs target/release/viv (or $VIPYR_BUNDLE_VIV) and a C compiler ($CC;
# MinGW-w64 gcc on Windows). $VIPYR_PREBUILD_FAST=1 adds the fast builds.
# The wheels are py3-none-<platform>: one per platform, not per Python.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-$ROOT/dist}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
PY=${PYTHON:-python3}
build() {
    (cd "$1" && PYTHONPATH="$ROOT/python" "$PY" -c "import sys, vipyr.build as b; print(b.build_wheel(sys.argv[1]))" "$OUT")
}
build "$ROOT/python"
for d in "$ROOT"/packages/*/ "$ROOT"/ports/*/; do
    case "$(basename "$d")" in _*) continue ;; esac
    [ -f "$d/pyproject.toml" ] && build "$d"
done
ls -l "$OUT"
