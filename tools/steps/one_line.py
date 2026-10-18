#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Write each step of one short statement in the one-line form.

    python3 tools/steps/one_line.py FILE_OR_DIR...           list what would change
    python3 tools/steps/one_line.py --fix FILE_OR_DIR...     rewrite in place

A step is rewritten when its body is one line holding one statement, the
statement does not open a block, and the result fits in MAX_WIDTH columns:

    step attn_residual {          step attn_residual: X1 = X + Y
        X1 = X + Y          ->
    }

The one-line form is the braced form to every tool (the lexer writes it as
that), and this script checks it: after a
rewrite, `viv hash` and the syntax tree (`--emit ast`, digits taken out, since
the spans move) must be what they were, or the file is left as it was and
reported. A step whose header or closing brace carries a comment is left
alone, so no comment moves.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIV = os.environ.get("VIV", os.path.join(ROOT, "target", "release", "viv"))
# the widest line the rewrite may produce, indentation included
MAX_WIDTH = 100

HEADER = re.compile(r'^(?P<lead>\s*)step (?P<id>[A-Za-z_]\w*)(?P<text> "(?:[^"\\]|\\.)*")? \{\s*$')
OPENS_BLOCK = re.compile(r'\{\s*(#.*)?$')


def rewrite(text):
    """The text with every candidate step in one line, and how many there were."""
    lines = text.split("\n")
    out = []
    n = 0
    i = 0
    while i < len(lines):
        m = HEADER.match(lines[i])
        if m and i + 2 < len(lines):
            body, close = lines[i + 1], lines[i + 2]
            lead = m.group("lead")
            stmt = body.strip()
            ok = (
                close == lead + "}"
                and body.startswith(lead + " ")
                and stmt
                and not stmt.startswith("#")
                and not stmt.startswith("step ")
                and not OPENS_BLOCK.search(stmt)
            )
            one = "%sstep %s%s: %s" % (lead, m.group("id"), m.group("text") or "", stmt)
            if ok and len(one) <= MAX_WIDTH:
                out.append(one)
                n += 1
                i += 3
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out), n


def fingerprint(path):
    """`viv hash` and the digit-free syntax tree of a file, or None when viv refuses it."""
    got = []
    for args in (["hash", path], [path, "--emit", "ast"]):
        r = subprocess.run([VIV] + args, capture_output=True, text=True, cwd=ROOT)
        if r.returncode != 0:
            return None
        got.append(re.sub(r"\d+", "", r.stdout.replace(path, "")))
    return got


def files(paths):
    for p in paths:
        if os.path.isdir(p):
            for d, _, names in sorted(os.walk(p)):
                for name in sorted(names):
                    if name.endswith(".viv"):
                        yield os.path.join(d, name)
        else:
            yield p


def main(argv):
    fix = "--fix" in argv
    paths = [a for a in argv if not a.startswith("--")]
    if not paths:
        print(__doc__.strip())
        return 2
    total = changed = refused = 0
    for f in files(paths):
        with open(f) as fh:
            old = fh.read()
        new, n = rewrite(old)
        if not n:
            continue
        total += n
        if not fix:
            print("%4d  %s" % (n, f))
            continue
        # in place, so the file keeps its name and its .debug and .intent
        # files; put back when anything changed
        before = fingerprint(f)
        with open(f, "w") as fh:
            fh.write(new)
        after = fingerprint(f)
        if before is None or after != before:
            with open(f, "w") as fh:
                fh.write(old)
            print("SKIP  %s: the hash or the syntax tree would change" % f)
            refused += 1
            continue
        changed += 1
        print("%4d  %s" % (n, f))
    print("%d steps%s" % (total, " in %d files rewritten, %d files skipped" % (changed, refused) if fix else " would be rewritten"))
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
