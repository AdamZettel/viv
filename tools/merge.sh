#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Merge a branch into main only after the premerge gate passes on the merged tree.
#
#     tools/merge.sh <branch> [-m "message"] [-- --ignore-lock|--keep-going|--no-cache]
#
# 1. In the merge worktree (../viv-wt/.merge next to the main repository, kept
#    between merges so its target/ stays built), check out main detached and
#    merge <branch> with --no-ff.
# 2. A conflict in www/src/data/ (files the site's staging regenerates) is
#    settled by taking main's copy; the staging in www-check regenerates
#    them from the merged sources (the branch touched www/, so its premerge
#    run includes www-check), and the merge commit is amended to carry
#    what it wrote. Any other conflict stops the merge.
# 3. tools/gate.sh premerge runs on the merged tree (the
#    language checks, the library set, the smoke set, and the post-merge
#    suites whose paths the branch touches), with the gate's cache: a
#    check or program whose inputs are unchanged since it passed (on any
#    worktree) is reused, and a sample of those reruns (tools/gate/cache.py).
#    `-- --no-cache` runs everything.
#    A retry after a failed gate may run fewer checks: when main has not
#    moved and only Rust integration tests changed since the failed attempt,
#    its passes stand and only the failed checks, the checks that read Rust
#    tests and the whole-tree scans run (tools/gate/reuse.py says which and
#    why). `-- --no-cache` turns that off too.
# 4. Only when it passes, and main has not moved meanwhile, main is
#    fast-forwarded to the merge commit (in the worktree that has main checked
#    out, so its files follow).
#
# On any failure main is left as it was, the gate's report is printed, and
# the logs are kept in target/gate/merges/<branch>-<time>/ of the main
# repository. Exits nonzero on failure.
set -eu
branch=""; msg=""; gate_opts=""
while [ $# -gt 0 ]; do
    case "$1" in
        -m) msg=$2; shift 2 ;;
        --) shift; gate_opts="$*"
            for o in "$@"; do case "$o" in --ignore-lock|--keep-going|--no-cache) ;;
                *) echo "merge.sh: the gate option $o is not allowed: a merge runs the whole premerge tier" >&2; exit 2 ;; esac; done
            break ;;
        -*) echo "merge.sh: unknown option $1" >&2; exit 2 ;;
        *) [ -z "$branch" ] || { echo "merge.sh: one branch at a time" >&2; exit 2; }; branch=$1; shift ;;
    esac
done
[ -n "$branch" ] || { sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

here=$(cd "$(dirname "$0")/.." && pwd)
common=$(cd "$here" && cd "$(git rev-parse --git-common-dir)/.." && pwd)
git -C "$common" rev-parse --verify -q "refs/heads/$branch" > /dev/null || { echo "merge.sh: no branch $branch" >&2; exit 2; }
base=$(git -C "$common" rev-parse main)
tip=$(git -C "$common" rev-parse "$branch")
if git -C "$common" merge-base --is-ancestor "$tip" "$base"; then
    echo "merge.sh: $branch ($tip) is already in main"; exit 0
fi
[ -n "$msg" ] || msg="Merge $branch"
stamp=$(date +%Y%m%d-%H%M%S)
keep="$common/target/gate/merges/$branch-$stamp"
wt="$(dirname "$common")/viv-wt/.merge"
lock="$common/target/gate/merge.lock"
mkdir -p "$common/target/gate"
mkdir "$lock" 2> /dev/null || { echo "merge.sh: another merge is running ($lock; remove it if not)" >&2; exit 1; }
trap 'rmdir "$lock" 2> /dev/null || true' EXIT

fail() {
    echo
    echo "MERGE REFUSED: $1"
    echo "main is unchanged at $(git -C "$common" rev-parse --short main)."
    if [ -d "$wt/target/gate/premerge" ]; then
        mkdir -p "$keep" && cp -R "$wt/target/gate/premerge/." "$keep/"
        echo "gate logs: $keep"
    fi
    echo "the merged tree is left in $wt for a look"
    exit 1
}

echo "== merging $branch ($(git -C "$common" rev-parse --short "$tip")) into main ($(git -C "$common" rev-parse --short "$base")) in $wt"
if [ -d "$wt" ] && git -C "$wt" rev-parse --git-dir > /dev/null 2>&1; then
    git -C "$wt" merge --abort > /dev/null 2>&1 || true
    git -C "$wt" checkout -q --detach -f "$base"
    git -C "$wt" reset -q --hard "$base"
    git -C "$wt" clean -q -fd     # untracked files go; ignored ones (target/) stay
else
    git -C "$common" worktree add -q --detach "$wt" "$base"
fi
rm -rf "$wt/target/gate/premerge"

mergebase=$(git -C "$common" merge-base "$base" "$tip")
no_markers() {   # a conflict marker left in what is about to be committed stops the merge
    if ! git -C "$wt" diff --cached --check > "$wt/.check.out" 2>&1; then
        if grep -q 'conflict marker' "$wt/.check.out"; then
            grep 'conflict marker' "$wt/.check.out" | sed 's/^/     /'
            rm -f "$wt/.check.out"
            git -C "$wt" merge --abort 2> /dev/null || true
            fail "conflict markers in the merge (listed above)"
        fi
    fi
    rm -f "$wt/.check.out"
}
ledger_fix() {   # the bug ledger: a union of lines, a branch's colliding ids renumbered (tools/gate/ledger.py)
    if [ -f "$wt/tools/gate/ledger.py" ]; then
        (cd "$wt" && python3 tools/gate/ledger.py merge "$mergebase" "$tip") > "$wt/.ledger.out" || fail "tools/gate/ledger.py could not merge bugs/ledger.jsonl"
        if [ -s "$wt/.ledger.out" ]; then
            echo "   bug ledger:"; sed 's/^/     /' "$wt/.ledger.out"
            cat "$wt/.ledger.out" >> "$common/target/gate/renumbered.log"
        fi
        rm -f "$wt/.ledger.out"
    fi
}
if ! git -C "$wt" merge --no-ff --no-edit -m "$msg" "$branch" > "$wt/.merge.out" 2>&1; then
    conflicts=$(git -C "$wt" diff --name-only --diff-filter=U)
    if printf '%s\n' "$conflicts" | grep -qx 'bugs/ledger.jsonl'; then
        ledger_fix
        git -C "$wt" add -- bugs/ledger.jsonl
        # the files whose ids it rewrote: modified, and not among the conflicts
        # (git diff lists a conflicted file as modified too, and adding it would
        # stage its conflict markers as the resolution)
        git -C "$wt" diff --name-only --diff-filter=M | while read -r f; do
            printf '%s\n' "$conflicts" | grep -qxF -- "$f" || git -C "$wt" add -- "$f"
        done
        conflicts=$(git -C "$wt" diff --name-only --diff-filter=U)
    fi
    other=$(printf '%s\n' "$conflicts" | grep -v '^www/src/data/' | grep -v '^$' || true)
    if [ -n "$other" ]; then
        cat "$wt/.merge.out"
        git -C "$wt" merge --abort
        fail "conflicts outside www/src/data:
$other"
    fi
    echo "   conflicts only in regenerated data; taking main's copy:"
    printf '%s\n' "$conflicts" | sed 's/^/     /'
    printf '%s\n' "$conflicts" | while read -r f; do
        [ -n "$f" ] || continue   # the ledger was the only conflict, and ledger_fix settled it
        git -C "$wt" checkout --ours -- "$f" && git -C "$wt" add -- "$f"
    done
    no_markers
    git -C "$wt" commit -q --no-edit -m "$msg"
else
    # no conflict, but git may have interleaved two appends with one id: the same fix
    ledger_fix
    if [ -n "$(git -C "$wt" status --porcelain --untracked-files=no)" ]; then
        git -C "$wt" add -u && git -C "$wt" commit -q --amend --no-edit
    fi
fi
rm -f "$wt/.merge.out"
merged=$(git -C "$wt" rev-parse HEAD)
case " $gate_opts " in
    *" --no-cache "*) how="without the cache (--no-cache): every check runs" ;;
    *) how="with the cache: unchanged checks and programs are reused, a sample rerun (-- --no-cache runs everything)" ;;
esac
only=""
case " $gate_opts " in
    *" --no-cache "*) ;;
    *) if [ -f "$wt/tools/gate/reuse.py" ]; then
           only=$(cd "$wt" && python3 tools/gate/reuse.py "$branch" "$merged" "$base" 2> "$wt/.reuse.out") || only=""
           echo "   retry: $(cat "$wt/.reuse.out")"
           rm -f "$wt/.reuse.out"
       fi ;;
esac
echo "   merge commit $(git -C "$wt" rev-parse --short HEAD); running tools/gate.sh premerge on it, $how"
[ -z "$only" ] || echo "   only these checks: $only"
echo

# shellcheck disable=SC2086
[ -x "$wt/tools/gate.sh" ] || fail "the merged tree has no tools/gate.sh"
if [ -n "$only" ]; then
    GATE_SOURCE=merge "$wt/tools/gate.sh" premerge $gate_opts --only "$only" || fail "the premerge gate failed on the merged tree (report above)"
else
    GATE_SOURCE=merge "$wt/tools/gate.sh" premerge $gate_opts || fail "the premerge gate failed on the merged tree (report above)"
fi

# what the staging regenerated belongs in the merge commit
if [ -n "$(git -C "$wt" status --porcelain -- www/src/data)" ]; then
    git -C "$wt" add -- www/src/data
    git -C "$wt" commit -q --amend --no-edit
    echo "   restaged the regenerated www/src/data into the merge commit"
    new=$(git -C "$wt" rev-parse HEAD)
    g="$common/target/gate/golden"
    [ -d "$g/$merged" ] && mv "$g/$merged" "$g/$new"   # hub-golden does not read www/
    merged=$new
fi
left=$(git -C "$wt" status --porcelain --untracked-files=no)
[ -z "$left" ] || { echo "   the gate changed tracked files (not committed):"; echo "$left" | sed 's/^/     /'; }

now=$(git -C "$common" rev-parse main)
[ "$now" = "$base" ] || fail "main moved during the gate ($base -> $now); run tools/merge.sh $branch again"

mainwt=$(git -C "$common" worktree list --porcelain | awk '/^worktree /{w=substr($0,10)} /^branch refs\/heads\/main$/{print w}')
if [ -n "$mainwt" ]; then
    git -C "$mainwt" merge -q --ff-only "$merged" || fail "fast-forward refused in $mainwt (uncommitted changes in the way?)"
else
    git -C "$common" update-ref refs/heads/main "$merged" "$base"
fi
git -C "$wt" checkout -q --detach "$merged"
echo
echo "MERGED: main is now $(git -C "$common" rev-parse --short main) ($msg)"
python3 - "$wt/target/gate/premerge/summary.json" <<'PY' || true
import json, sys
s = json.load(open(sys.argv[1]))
c = s.get("cache") or {}
if not c.get("on"):
    print("the gate ran every check (cache off)")
else:
    whole = [r["name"] for r in s["checks"] if r.get("cached")]
    items = [r["cache"] for r in s["checks"] if r.get("cache")]
    print(f"the gate used the cache: {len(whole)} checks reused whole"
          + (f" ({', '.join(whole)})" if whole else "")
          + f"; programs {sum(k.get('cached', 0) for k in items)} reused, {sum(k.get('run', 0) for k in items)} run, "
          f"{sum(k.get('sampled', 0) for k in items)} reruns sampled; {s['seconds'] / 60:.0f} minutes")
PY
