#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Merging bugs/ledger.jsonl, whose ids two branches may have picked alike.

    python3 tools/gate/ledger.py merge <base> <branch>    in the merge worktree, mid-merge or after it

Branches number their bugs themselves (B087, B088, ...), so two branches that
found bugs at the same time pick the same ids. tools/merge.sh runs this on
every merge, conflict or not:

1. The merged ledger is the union of lines: main's ledger in its order, then
   each line the branch added that main does not have. (Git's own merge of
   two appends is either a conflict or, worse, a silent interleave.)
2. A branch entry whose id main already uses for a different bug gets the
   next free id. The old id is rewritten to the new one in the lines the
   branch added to the files it changed (docs, FRICTION.json, tests,
   RESULTS): only those lines, so main's own citations of the id stay.
3. Each renumbering is printed (`renumbered B089 -> B096 (title...)`, and
   the files rewritten), for merge.sh's report.

The bugs check (`tools/gate.sh bugs --check`) fails on any id used twice.

Why not ids that cannot collide (a date and a hash)? The ids are cited in
prose, tests and commit messages, where B089 is what people read and say;
collisions are rare and now fixed mechanically at the one place branches
meet. Exits nonzero only when the ledger cannot be read.
"""
import json
import os
import re
import subprocess
import sys

LEDGER = "bugs/ledger.jsonl"


def git(*args, check=False):
    p = subprocess.run(["git", *args], capture_output=True, text=True)
    if check and p.returncode:
        sys.exit(f"ledger.py: git {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout if p.returncode == 0 else None


def lines_of(rev):
    t = git("show", f"{rev}:{LEDGER}")
    return [l for l in (t or "").splitlines() if l.strip()]


def bid(line):
    try:
        return json.loads(line).get("id")
    except ValueError:
        return None


def next_free(used):
    n = max([int(i[1:]) for i in used if i and re.fullmatch(r"B\d+", i)] or [0]) + 1
    return f"B{n:03d}"


def merge(base, branch):
    # mid-merge, HEAD is main; after the merge commit, HEAD^1 is
    main_lines = lines_of("HEAD") if git("rev-parse", "-q", "--verify", "MERGE_HEAD") else lines_of("HEAD^1")
    base_lines = set(lines_of(base))
    branch_lines = lines_of(branch)
    have = set(main_lines)
    used = {bid(l) for l in main_lines}
    main_by_id = {bid(l): l for l in main_lines}
    out, renames = list(main_lines), []
    for l in branch_lines:
        if l in have or l in base_lines:
            continue   # main has it, or it is the base's (main may have changed or dropped it)
        i = bid(l)
        if i is None:
            sys.exit(f"ledger.py: a line of {branch}'s ledger is not JSON: {l[:80]}")
        if i in used and main_by_id.get(i) != l:
            new = next_free(used | {bid(x) for x in out})
            e = json.loads(l)
            e["id"] = new
            l = json.dumps(e, ensure_ascii=False)
            renames.append((i, new, e.get("title", "")))
        out.append(l)
        have.add(l)
        used.add(bid(l))
    with open(LEDGER, "w") as f:
        f.write("\n".join(out) + "\n")
    # rewrite the old ids in what the branch added, file by file
    changed = (git("diff", "--name-only", f"{base}...{branch}") or "").split()
    for old, new, title in renames:
        print(f"renumbered {old} -> {new}: main already has {old}; {title[:90]}")
        pat = re.compile(r"\b" + re.escape(old) + r"\b")
        for f in changed:
            if f == LEDGER or not os.path.isfile(f):
                continue
            added = set()
            for d in (git("diff", "-U0", f"{base}...{branch}", "--", f) or "").splitlines():
                if d.startswith("+") and not d.startswith("+++") and pat.search(d):
                    added.add(d[1:])
            if not added:
                continue
            try:
                text = open(f).read()
            except UnicodeDecodeError:
                continue
            lines = text.split("\n")
            n = 0
            for k, line in enumerate(lines):
                if line in added:
                    lines[k] = pat.sub(new, line)
                    n += 1
            if n:
                open(f, "w").write("\n".join(lines))
                print(f"    {f}: {n} line{'s' if n > 1 else ''} rewritten")
    return renames


def main(argv):
    if len(argv) != 3 or argv[0] != "merge":
        print(__doc__.split("\n\n")[1])
        return 2
    merge(argv[1], argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
