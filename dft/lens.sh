#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# The lenses of the DFT modules: check, stamp, view,
# explain, story, init.
#
#     dft/lens.sh check                     every module below; exits 1 on any problem
#     dft/lens.sh check pwdft               one module
#     dft/lens.sh stamp pwdft build_vnl     the entry still holds: record the def's body hash
#     dft/lens.sh view pwdft_it [lens]      intent (the default), all, code or debug
#     dft/lens.sh explain pwdft             the intent walked along the call graph
#     dft/lens.sh story pwdft_it [args]     one silicon run told in the intent's words
#     dft/lens.sh init kmesh                print a skeleton with the current hashes
#
# The modules: pw_common, kmesh, pwdft (dense engine), pwdft_it (iterative).
#
# Why a wrapper. The lens tools hash a def by building the program it is in, and
# these modules do not build alone: they read globals that an input deck sets
# (the crystal, the cutoff, the pseudopotential), and dft/run.sh hands them to
# the compiler as --use preludes. So each module is checked as part of the
# program it belongs to, with silicon's deck. The wrapper makes a copy of the
# module whose first few comment lines are replaced by import lines for the
# files that come before it (the deck, then the modules it reads), and appends
# import lines for the files that come after it, ending with silicon's check
# file, so that intent text can cite the tests in it. Replacing comment lines
# keeps every line number, so each finding points at the right line of the
# real file; paths in the output are rewritten to the real file. The copy, and
# copies of the module's .intent and .debug files, live in a temporary
# directory under dft/ (the lens tools look up imports from the
# repository root) and are removed afterwards.
#
# Two long defs have no steps and are let through by name: anderson_step in
# pw_common and setup_k in pwdft_it. A step block adds lines, and every line of
# pw_common.viv, kmesh.viv and pwdft_it.viv is part of the C that the served
# wasm modules are compiled from (the bounds-check table quotes source lines by
# number), so adding one changes that C and dft/build-wasm.sh --check reports
# the modules stale. These two get steps when the wasm modules are next rebuilt.
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
viv="$root/target/release/viv"
ALL="kmesh pw_common pwdft pwdft_it"
WAIVED="pw_common:anderson_step pwdft_it:setup_k"

before() {
    case "$1" in
        kmesh) echo "dft/si" ;;
        pw_common) echo "dft/si dft/kmesh" ;;
        pwdft | pwdft_it) echo "dft/si dft/kmesh dft/pw_common" ;;
        *) echo "unknown module $1 (one of: $ALL)" >&2; exit 2 ;;
    esac
}
after() {
    case "$1" in
        kmesh) echo "dft/pw_common dft/pwdft_it dft/si_it_check" ;;
        pw_common) echo "dft/pwdft_it dft/si_it_check" ;;
        pwdft) echo "dft/si_check" ;;
        pwdft_it) echo "dft/si_it_check" ;;
    esac
}

tmp=$(mktemp -d "$here/.lens.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
rel=${tmp#"$root"/}

# $1 = module -> $tmp/$1.viv (+ .intent, .debug)
prepare() {
    m=$1
    src="$here/$m.viv"
    pre=$(before "$m")
    n=$(echo "$pre" | wc -w | tr -d ' ')
    # the imports `viv imports` writes for the files before and after the
    # module: from a stub holding `#! use` lines and the module's code
    body_lines=$(($(wc -l < "$src") - n))
    {
        for u in $pre; do echo "#! use $u"; done
        tail -n "+$((n + 1))" "$src"
    } > "$tmp/.pre_$m.viv"
    {
        cat "$tmp/.pre_$m.viv"
        for u in $(after "$m"); do echo "#! use $u"; done
    } > "$tmp/.full_$m.viv"
    (cd "$root" && "$viv" imports --write "$tmp/.pre_$m.viv" "$tmp/.full_$m.viv" 2>/dev/null) || { echo "dft/$m.viv: viv imports could not write the wrapper's imports" >&2; exit 2; }
    h=$(($(wc -l < "$tmp/.pre_$m.viv") - body_lines))
    # the lines replaced must be plain comments, or the copy would not be the module
    bad=$(head -n "$h" "$src" | grep -v '^#' | head -1 || true)
    lens=$(head -n "$h" "$src" | grep '^#[|:@!]' | head -1 || true)
    if [ -n "$bad$lens" ]; then
        echo "dft/$m.viv: the first $h lines must be plain # comments (the wrapper puts its import lines there)" >&2
        exit 2
    fi
    {
        head -n "$h" "$tmp/.pre_$m.viv"
        tail -n "+$((h + 1))" "$src"
        tail -n "+$((h + body_lines + 1))" "$tmp/.full_$m.viv"
    } > "$tmp/$m.viv"
    rm -f "$tmp/.pre_$m.viv" "$tmp/.full_$m.viv"
    [ -f "$here/$m.intent" ] && cp "$here/$m.intent" "$tmp/$m.intent"
    [ -f "$here/$m.debug" ] && cp "$here/$m.debug" "$tmp/$m.debug"
    return 0
}

# the tools' output, with the copy's paths turned back into the real ones
real() {
    sed -e "s|$tmp/|dft/|g" -e "s|$rel/|dft/|g" -e "s|dft/\.lens\.[A-Za-z0-9]*/|dft/|g" \
        -e "s|viv lens stamp dft/\([a-z_]*\)\.viv|dft/lens.sh stamp \1|g"
}

cmd=${1:-check}
[ $# -gt 0 ] && shift
case "$cmd" in
    check)
        mods=${*:-$ALL}
        status=0
        for m in $mods; do
            prepare "$m"
            out=$(cd "$root" && "$viv" lens check "$rel/$m.viv" 2>&1 | real) || true
            # drop the waived findings, then the summary line, and count what is left
            kept=$(printf '%s\n' "$out" | grep -v '^lens check ' | while IFS= read -r line; do
                skip=0
                for w in $WAIVED; do
                    if [ "${w%%:*}" = "$m" ] && printf '%s' "$line" | grep -q ": steps: \`${w#*:}\` has "; then skip=1; fi
                done
                [ $skip = 1 ] || printf '%s\n' "$line"
            done)
            waived=$(printf '%s\n' "$out" | grep -c ': steps: ' || true)
            left=$(printf '%s' "$kept" | grep -c '' || true)
            if [ "$left" -eq 0 ]; then
                printf 'lens check dft/%s.viv: ok' "$m"
                [ "$waived" -gt 0 ] && printf ' (%s long def%s without steps let through: see dft/lens.sh)' "$waived" "$([ "$waived" = 1 ] || echo s)"
                printf '\n'
            else
                printf '%s\n' "$kept"
                echo "lens check dft/$m.viv: $left line(s) of problems"
                status=1
            fi
        done
        exit $status
        ;;
    stamp)
        m=$1
        shift
        prepare "$m"
        cp "$tmp/$m.viv" "$tmp/$m.viv.orig"
        for d in "$@"; do
            (cd "$root" && "$viv" lens stamp "$rel/$m.viv" "$d") | real
        done
        # the stamp belongs in the .intent file; a stamp written into the code
        # (an inline `#@` line) would land in the copy and be lost, so refuse it
        if ! cmp -s "$tmp/$m.viv" "$tmp/$m.viv.orig"; then
            echo "dft/$m.viv: the stamp went into the code (an inline #| entry); move that entry to dft/$m.intent" >&2
            exit 1
        fi
        cp "$tmp/$m.intent" "$here/$m.intent"
        ;;
    view)
        m=$1
        prepare "$m"
        (cd "$root" && "$viv" view "$rel/$m.viv" --lens "${2:-intent}") | real
        ;;
    explain)
        m=$1
        shift
        prepare "$m"
        (cd "$root" && "$viv" explain "$rel/$m.viv" "$@") | real
        ;;
    story)
        m=$1
        shift
        prepare "$m"
        (cd "$root" && "$viv" story "$rel/$m.viv" "$@") | real
        ;;
    init)
        m=$1
        prepare "$m"
        (cd "$root" && "$viv" lens init "$rel/$m.viv" --stdout) | real
        ;;
    *)
        sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
