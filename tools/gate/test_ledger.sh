#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# tools/gate/ledger.py on a scratch repository: main and a branch both add
# bug B003, the branch's doc and test cite its B003, and main's doc cites
# main's B003. After `ledger.py merge`, the ledger holds both bugs (the
# branch's as B004), the branch's citations say B004, main's still say B003,
# and the ids are unique. Run both ways: the appends conflicting, and not.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
bug() { printf '{"id": "%s", "title": "%s", "found": "2026-10-06", "found_by": "t", "engines": [], "disagreement": false, "status": "open", "fixed": null, "commit": null, "test": null, "root_cause": null}\n' "$1" "$2"; }
bad=0
for mode in conflict clean; do
    r="$d/$mode"; mkdir -p "$r/bugs" "$r/docs"; cd "$r"
    git init -q -b main . && git config user.email t@t && git config user.name t
    { bug B001 one; bug B002 two; } > bugs/ledger.jsonl
    echo "nothing yet" > docs/main.md
    git add -A && git commit -qm base
    git checkout -qb br
    if [ $mode = conflict ]; then bug B003 "branch bug" >> bugs/ledger.jsonl
    else { bug B003 "branch bug"; cat bugs/ledger.jsonl; } > x && mv x bugs/ledger.jsonl; fi
    echo "see B003 for the branch's bug" > docs/branch.md
    echo "fn b003() {} // B003" > test.rs
    git add -A && git commit -qm branch
    git checkout -q main
    bug B003 "main bug" >> bugs/ledger.jsonl
    echo "see B003 for main's bug" > docs/main.md
    git add -A && git commit -qm main
    base=$(git merge-base main br)
    git merge -q --no-ff --no-edit br > /dev/null 2>&1 || true
    python3 "$here/ledger.py" merge "$base" br
    ids=$(python3 -c "import json;print(' '.join(json.loads(l)['id'] for l in open('bugs/ledger.jsonl') if l.strip()))")
    ok=1
    [ "$ids" = "B001 B002 B003 B004" ] || ok=0
    grep -q '"id": "B004", "title": "branch bug"' bugs/ledger.jsonl || ok=0
    grep -q 'B004' docs/branch.md && grep -q 'B004' test.rs || ok=0
    grep -q "B003 for main's bug" docs/main.md || ok=0
    if [ $ok = 1 ]; then echo "ok    $mode: ids $ids; the branch's citations renumbered, main's kept"
    else echo "FAIL  $mode: ids $ids"; cat bugs/ledger.jsonl docs/*.md test.rs; bad=1; fi
done
exit $bad
