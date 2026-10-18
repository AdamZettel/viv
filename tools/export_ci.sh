#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Build and test every exported program and kernel collection.
#
#     tools/export_ci.sh                 every *.export.toml in the repository
#     tools/export_ci.sh dft/pwdft.export.toml kernels/k1/k1.export.toml
#
# For each manifest <dir>/<name>.export.toml:
#   1. `viv export` writes target/export_ci/<name>/: <name>.c, <name>.h,
#      <name>.py, test_<name>.py and, when <dir>/<name>.exact_hashes.json is
#      committed, a copy of it as exact_hashes.json;
#   2. cc builds lib<name>.so (.dylib on macOS, the name the wrapper loads)
#      from <name>.c with the exact-mode flags, and nothing else on the line
#      but the output and -lm;
#   3. the stored hashes are required, and the copy in the export must be
#      byte-equal to them;
#   4. test_<name>.py runs: the program's own tests through Python, and the
#      library's results hashed against exact_hashes.json (the interpreter's bits);
#   5. if a hand-written test_<name>.py sits beside the manifest (K1's
#      reference tests), it runs too, against the exported module
#      (<NAME>_BACKEND=export, <NAME>_DIST=the export's directory).
#
# A viv binary without the verb makes the script say so and exit 0, as does
# a kernel manifest the binary cannot read; EXPORT_CI_STRICT=1 makes either
# a failure.
#
# Environment: VIV (default target/release/viv), CC (default cc), PYTHON
# (default python3), EXPORT_CI_OUT (default target/export_ci).
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
VIV=${VIV:-$ROOT/target/release/viv}
CC=${CC:-cc}
PYTHON=${PYTHON:-python3}
OUT=${EXPORT_CI_OUT:-$ROOT/target/export_ci}
STRICT=${EXPORT_CI_STRICT:-0}
EXACT_FLAGS="-O2 -ffp-contract=off -shared -fPIC"

skip_all() {
    echo "export_ci: SKIP: $1"
    [ "$STRICT" = 1 ] && exit 1
    exit 0
}

[ -x "$VIV" ] || skip_all "no viv binary at $VIV (cargo build --release -p viv, or set VIV)"
# A binary without the verb prints the general usage for `viv export --help`.
help=$("$VIV" export --help 2>&1)
case "$help" in
    *"viv export"*) ;;
    *) skip_all "this viv has no \`viv export\` verb" ;;
esac

if [ $# -gt 0 ]; then
    manifests=("$@")
else
    manifests=()
    while IFS= read -r f; do manifests+=("$f"); done < <(
        cd "$ROOT" && find . -name '*.export.toml' \
            -not -path './target/*' -not -path '*/node_modules/*' -not -path './www/dist/*' \
            | sed 's|^\./||' | sort)
fi
[ ${#manifests[@]} -gt 0 ] || skip_all "no *.export.toml in the repository"

# a manifest's `name` and `kind`, read with Python's TOML reader
field() {
    "$PYTHON" -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb")).get(sys.argv[2], sys.argv[3]))' "$1" "$2" "$3"
}

libext=so
case "$(uname -s)" in
    Darwin) platform=macos; libext=dylib ;;
    Linux) platform=linux ;;
    *) platform=$(uname -s) ;;
esac
echo "export_ci: $(${CC} --version 2>/dev/null | head -1) on $platform $(uname -m)"
echo "export_ci: flags $EXACT_FLAGS"

fail=0
summary=()
for rel in "${manifests[@]}"; do
    m=$ROOT/${rel#"$ROOT"/}
    name=$(field "$m" name "")
    kind=$(field "$m" kind program)
    dir=$OUT/$name
    stored=$(dirname "$m")/$name.exact_hashes.json
    rm -rf "$dir"
    mkdir -p "$dir"
    echo
    echo "== $rel ($kind $name)"

    if ! (cd "$ROOT" && "$VIV" export "$m" -o "$dir") > "$dir/export.log" 2>&1; then
        sed 's/^/   /' "$dir/export.log" | head -20
        if [ "$kind" = kernels ] && [ "$STRICT" != 1 ]; then
            summary+=("SKIP  $name  this viv export does not read kernel manifests")
            continue
        fi
        summary+=("FAIL  $name  viv export")
        fail=1
        continue
    fi

    lib=$dir/lib$name.$libext
    # shellcheck disable=SC2086
    if ! "$CC" $EXACT_FLAGS "$dir/$name.c" -o "$lib" -lm > "$dir/cc.log" 2>&1; then
        sed 's/^/   /' "$dir/cc.log" | head -20
        summary+=("FAIL  $name  cc")
        fail=1
        continue
    fi

    if [ ! -f "$stored" ]; then
        echo "   no stored hashes at ${stored#"$ROOT"/}"
        echo "   they are the interpreter's results, recorded once and committed"
        summary+=("FAIL  $name  no stored hashes")
        fail=1
        continue
    fi
    if ! cmp -s "$stored" "$dir/exact_hashes.json"; then
        echo "   the export's exact_hashes.json differs from ${stored#"$ROOT"/}"
        summary+=("FAIL  $name  hashes not carried into the export")
        fail=1
        continue
    fi

    upper=$(printf '%s' "$name" | tr '[:lower:]-' '[:upper:]_')
    status=ok
    if ! (cd "$dir" && "$PYTHON" "test_$name.py") > "$dir/test.log" 2>&1; then
        tail -20 "$dir/test.log" | sed 's/^/   /'
        status="test_$name.py failed"
    fi
    beside=$(dirname "$m")/test_$name.py
    if [ "$status" = ok ] && [ -f "$beside" ]; then
        if ! env "${upper}_BACKEND=export" "${upper}_DIST=$dir" "$PYTHON" "$beside" > "$dir/reference.log" 2>&1; then
            tail -20 "$dir/reference.log" | sed 's/^/   /'
            status="${beside#"$ROOT"/} failed"
        fi
    fi
    if [ "$status" = ok ]; then
        summary+=("ok    $name  built, tests pass, results match the stored hashes")
    else
        summary+=("FAIL  $name  $status")
        fail=1
    fi
done

echo
echo "export_ci summary ($platform):"
for line in "${summary[@]}"; do
    echo "  $line"
done
exit $fail
