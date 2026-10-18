#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Which checks a merge retry must run, when its last attempt's gate failed.

    reuse.py <branch> <merged> <base>

tools/merge.sh asks this before it runs the gate. The last attempt is the
newest target/gate/merges/<branch>-*/summary.json. Its passes are reused
only when all of these hold:

- it ran the whole premerge tier (no --only) and failed;
- its merge commit's first parent is <base>: main has not moved since;
- every path that differs between its tree and <merged> is a Rust
  integration test (`<crate>/tests/**.rs`), which only cargo compiles.

Then the retry runs the checks that failed, the checks that compile or read
Rust tests (cargo_tests.py and the bug ledger's check), the cheap checks
that scan the whole tree (licenses and the ratchets), and www-check when the
attempt ran it: its staging regenerates www/src/data, which merge.sh commits
into the merge. Anything else: the whole tier, as before.

Prints the comma-separated check names for `gate.sh premerge --only`, or
nothing for the whole tier; the reason goes to stderr."""
import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# a Rust integration test: compiled by cargo test and nothing else
RUST_TEST = re.compile(r"(^|/)tests/(.*/)?[^/]+\.rs$")
# the checks that compile or read Rust tests: every cargo_tests.py check, and
# `bugs`, which looks up each fixed bug's test function in its tests/*.rs
READS_RUST_TESTS = {"cargo-test", "cargo-test-app", "fast-check", "bugs"}
# the checks that read every tracked file: cheap, so always rerun
WHOLE_TREE = {"licenses", "lens-coverage", "arch-check", "features-ratchet"}
# the check whose staging writes what merge.sh commits into the merge
REGENERATES = {"www-check"}


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, check=True).stdout


def common_dir():
    return os.path.dirname(os.path.abspath(os.path.join(ROOT, git("rev-parse", "--git-common-dir").strip())))


def plan(branch, merged, base):
    """(the checks to run, or None for the whole tier; why)."""
    runs = sorted(glob.glob(os.path.join(common_dir(), "target", "gate", "merges", f"{branch}-*", "summary.json")))
    if not runs:
        return None, "no earlier attempt of this branch"
    path = runs[-1]
    s = json.load(open(path))
    if s.get("tier") != "premerge" or s.get("only"):
        return None, f"the last attempt ({path}) did not run the whole premerge tier"
    if s.get("verdict") != "FAIL":
        return None, f"the last attempt ({path}) did not fail"
    prev = s.get("sha")
    try:
        parent = git("rev-parse", f"{prev}^1").strip()
    except subprocess.CalledProcessError:
        return None, f"the last attempt's merge commit {prev} is gone"
    if parent != git("rev-parse", base).strip():
        return None, f"main has moved since the last attempt ({parent[:8]} -> {base[:8]})"
    changed = [p for p in git("diff", "--name-only", prev, merged).splitlines() if p]
    if not changed:
        return None, "the tree is the same as the last attempt's: rerun everything to see it fail again"
    other = [p for p in changed if not RUST_TEST.search(p)]
    if other:
        return None, f"{len(other)} changed paths are not Rust tests (first: {other[0]})"
    names = {c["name"] for c in s["checks"]}
    failed = {c["name"] for c in s["checks"] if c.get("status") != "PASS"}
    run = failed | ((READS_RUST_TESTS | WHOLE_TREE | REGENERATES) & names)
    kept = sorted(names - run)
    why = (f"the last attempt ({os.path.basename(os.path.dirname(path))}, {prev[:8]}) failed {', '.join(sorted(failed))}; "
           f"since then only {len(changed)} Rust test file(s) changed ({', '.join(changed)}), so its "
           f"{len(kept)} other passes stand and {len(run)} checks run again")
    return sorted(run), why


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__.split("\n\n")[1].strip())
    run, why = plan(*sys.argv[1:])
    print(why, file=sys.stderr)
    if run:
        print(",".join(run))


if __name__ == "__main__":
    main()
