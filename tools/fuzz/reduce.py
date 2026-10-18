#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Shrink a failing Viv program to a small one that fails the same way.

    python3 tools/fuzz/reduce.py fail_123.viv [-o fail_123.min.viv] [--engines native,wasm] [--san]
                                 [--budget 1500]

The failure is what tools/fuzz/run.py's check() says of the program (its kind,
and the engine and message with numbers and names taken out); a candidate is
kept when check() says the same of it. Every candidate runs in the interpreter
first, so a cut that makes the program invalid (an unknown name, an index out
of range, an Int overflow) is refused by the front end or the interpreter and
dropped: the reduced program is still valid and still has one meaning.

Passes, repeated until none makes progress (C-Reduce style):
  1. delete statements and whole blocks: halves, quarters, ... down to one
     at a time (delta debugging over the statement list);
  2. unwrap a block: keep an if's or a loop's body, drop its head;
  3. simplify expressions: replace a parenthesized group or a call by one of
     its operands or arguments, or by a small literal.
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run import check, run, VIV  # noqa: E402
import tempfile  # noqa: E402


def front_end_accepts(lines):
    with tempfile.TemporaryDirectory(prefix="vivsmith-") as d:
        p = os.path.join(d, "p.viv")
        with open(p, "w") as f:
            f.write("\n".join(lines) + "\n")
        # `viv run`, not `viv check`: check passes a program with an unknown name
        rc, out = run([VIV, "run", p], cwd=d)
        return rc == 0 and "error" not in out


def signature(kind, detail, lines=None):
    """What must stay the same: the kind, the engine, the message's words, and
    for a refusal or an interpreter error the text of the line it names (else
    a cut elsewhere could make a different line fail with the same words)."""
    d = detail or ""
    at = ""
    m = re.search(r"\(line (\d+)\)", d)
    if m and lines is not None and kind in ("refused", "interp") and int(m.group(1)) <= len(lines):
        at = lines[int(m.group(1)) - 1].strip()
    d = re.sub(r"\(line \d+\)", "", d)
    d = re.sub(r"`[^`]*`", "`_`", d)
    if kind == "differs":
        d = d.split(":")[0]           # the engine; the bytes may change
    if "panicked at" in d:
        return kind, d[d.index("panicked at"):].split()[2], at    # the panic's source location
    d = re.sub(r"-?\d+(\.\d+)?(e[-+]?\d+)?", "N", d)
    return kind, d.strip(), at


class Reducer:
    def __init__(self, kind, detail, engines, san, budget, lines=None, checker=check):
        self.want = signature(kind, detail, lines)
        self.check = checker
        self.engines, self.san = engines, san
        self.budget = budget
        self.tests = 0
        self.cache = {}

    def interesting(self, lines):
        src = "\n".join(lines) + "\n"
        if src in self.cache:
            return self.cache[src]
        if self.tests >= self.budget:
            return False
        self.tests += 1
        k, d = self.check(src, self.engines, self.san)
        ok = k is not None and signature(k, d, lines) == self.want
        if ok and k == "refused":
            # the refusal is the named line's alone: without it the front end
            # accepts the rest, so no cut elsewhere made the program invalid
            m = re.search(r"\(line (\d+)\)", d)
            rest = [l for j, l in enumerate(lines) if j != int(m.group(1)) - 1]
            ok = front_end_accepts(rest)
        self.cache[src] = ok
        return ok

    # ------------------------------------------------------------ statements
    @staticmethod
    def spans(lines):
        """(start, end) of every statement: a line, or a block to its closing brace."""
        out = []
        for i, l in enumerate(lines):
            s = l.strip()
            if not s or s.startswith("}") or s.startswith("#"):
                continue
            if s.endswith("{"):
                depth, j = 0, i
                while j < len(lines):
                    t = lines[j].strip()
                    depth += t.count("{") - t.count("}")
                    if depth == 0 and j > i:
                        break
                    if depth == 0 and j == i:
                        break
                    j += 1
                # a chain `} else {` keeps depth at 1; the block ends where it closes
                out.append((i, min(j, len(lines) - 1)))
            else:
                out.append((i, i))
        return out

    def delete_pass(self, lines):
        progress = False
        sp = self.spans(lines)
        n = max(1, len(sp) // 2)
        while n >= 1:
            i = 0
            sp = self.spans(lines)
            while i < len(sp):
                chunk = sp[i:i + n]
                drop = set()
                for a, b in chunk:
                    drop.update(range(a, b + 1))
                cand = [l for k, l in enumerate(lines) if k not in drop]
                if cand != lines and self.interesting(cand):
                    lines = cand
                    progress = True
                    sp = self.spans(lines)
                else:
                    i += n
            n //= 2
        return lines, progress

    def unwrap_pass(self, lines):
        progress = False
        i = 0
        while i < len(lines):
            s = lines[i].strip()
            if s.endswith("{") and not s.startswith(("def ", "record ", "}", "test ")):
                ends = [b for a, b in self.spans(lines) if a == i]
                if ends:
                    b = ends[0]
                    body = lines[i + 1:b]
                    cand = lines[:i] + body + lines[b + 1:]
                    if self.interesting(cand):
                        lines = cand
                        progress = True
                        continue
            i += 1
        return lines, progress

    # ------------------------------------------------------------ expressions
    @staticmethod
    def groups(line):
        """(start, end, pieces) for each parenthesized group or call in the line."""
        out = []
        stack = []
        for k, ch in enumerate(line):
            if ch in "([":
                stack.append(k)
            elif ch in ")]" and stack:
                a = stack.pop()
                if line[a] == "[" and a > 0 and (line[a - 1].isalnum() or line[a - 1] in "_])"):
                    continue          # an index, not a list
                start = a
                m = re.search(r"[A-Za-z_][A-Za-z_0-9.]*$", line[:a])
                if m and line[a] == "(":
                    start = m.start()
                inner = line[a + 1:k]
                pieces = Reducer.split_top(inner)
                out.append((start, k + 1, pieces))
        return out

    @staticmethod
    def split_top(s):
        """s's operands at depth 0: by commas, else by binary operators and if/else."""
        parts, depth, cur = [], 0, ""
        for ch in s:
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(cur)
                cur = ""
            else:
                cur += ch
        parts.append(cur)
        if len(parts) == 1:
            parts, depth, cur, k = [], 0, "", 0
            ops = sorted([" + ", " - ", " * ", " // ", " / ", " % ", " ^ ", " if ", " else ", " and ", " or ",
                          " <= ", " >= ", " < ", " > ", " == ", " != "], key=len, reverse=True)
            while k < len(s):
                ch = s[k]
                if ch in "([":
                    depth += 1
                elif ch in ")]":
                    depth -= 1
                op = next((o for o in ops if depth == 0 and s.startswith(o, k)), None)
                if op:
                    parts.append(cur)
                    cur = ""
                    k += len(op)
                    continue
                cur += ch
                k += 1
            parts.append(cur)
        parts = [p.split("=", 1)[1] if re.match(r"^\s*\w+=", p) else p for p in parts]
        return [p.strip() for p in parts if p.strip()]

    def expr_pass(self, lines):
        progress = False
        for i in range(len(lines)):
            changed = True
            while changed:
                changed = False
                for a, b, pieces in sorted(self.groups(lines[i]), key=lambda g: g[0] - g[1]):
                    old = lines[i][a:b]
                    cands = sorted(set(pieces), key=len) + ["0", "1", "0.0", "1.0"]
                    for c in cands:
                        if c == old or len(c) >= len(old):
                            continue
                        new = lines[i][:a] + c + lines[i][b:]
                        cand = lines[:i] + [new] + lines[i + 1:]
                        if self.interesting(cand):
                            lines = cand
                            progress = changed = True
                            break
                    if changed:
                        break
        return lines, progress

    def reduce(self, lines):
        if not self.interesting(lines):
            raise SystemExit("reduce.py: the program does not fail as described")
        progress = True
        while progress and self.tests < self.budget:
            progress = False
            # a refusal keeps every expression as written: simplifying one can make
            # the program invalid for a reason of its own (a literal where an Int
            # was), and the refusal would then be right
            passes = (self.delete_pass, self.unwrap_pass)
            if self.want[0] != "refused":
                passes += (self.expr_pass,)
            for p in passes:
                lines, pr = p(lines)
                progress |= pr
        return lines


def reduce_file(path, out, kind, detail, engines=("native", "wasm"), san=False, budget=1500):
    src = open(path).read().splitlines()
    r = Reducer(kind, detail, engines, san, budget, src)
    lines = r.reduce(src)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  reduced {os.path.basename(path)}: {len(src)} -> {len(lines)} lines in {r.tests} tests -> {out}", flush=True)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("-o", "--out")
    ap.add_argument("--engines", default="native,wasm")
    ap.add_argument("--san", action="store_true")
    ap.add_argument("--budget", type=int, default=1500)
    a = ap.parse_args()
    try:
        os.nice(10)
    except OSError:
        pass
    engines = tuple(e for e in a.engines.split(",") if e)
    k, d = check(open(a.file).read(), engines, a.san)
    if k is None:
        sys.exit("reduce.py: the program passes")
    print(f"  failure: {k}: {d}")
    out = a.out or a.file[:-4] + ".min.viv"
    lines = reduce_file(a.file, out, k, d, engines, a.san, a.budget)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
