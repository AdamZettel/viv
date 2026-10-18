#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Rewrite `print` calls for the separator Viv's `print` now puts between its
arguments, so every program prints exactly what it printed before.

Until 2026-09-30 `print(a, b, ...)` wrote its arguments with nothing between
them, and two values in a row were refused, so programs wrote the spaces by
hand: `print(a, " ", b)`, `print("x = ", x)`. Now `print` puts one space
between arguments, as Python's does, and `print(..., sep="")` puts none. The
parser joins each separator to the text beside it (lang/viv/src/parser.rs,
`print_call`): to the string literal before it when there is one, else to the
string literal after it, else it is an argument of its own. This script
inverts that rule on each call:

    print(a, " ", b)          ->  print(a, b)
    print("x = ", x)          ->  print("x =", x)
    print(e, " eV")           ->  print(e, "eV")
    print("[", x, "]")        ->  print("[", x, "]", sep="")

It takes out one space from the text on each side of every junction; when a
junction has no such space, the call keeps its arguments and gets `sep=""`
(dropping `""` arguments, which only kept two values apart). Every rewrite is
checked by running the parser's rule on the new arguments: it must give back
the old arguments exactly, text for text, so the engines receive the same call.
A call the check does not pass is left alone and reported.

    tools/migrate_print_sep.py [--check] FILE...     rewrite (or only report) files

A `.viv` file is Viv throughout (fenced `###` blocks and comments are skipped).
In a `.rs` file the Viv is inside string literals; in `.md` files inside
fenced code blocks marked `viv`; in `.mjs`, `.js`, `.jsx` and `.astro` inside
template literals; in `.toml` inside strings; in `.sh` files the whole text is
scanned. A file whose Viv defines its own `print` is left alone. Run it twice:
the second run changes nothing (a call that has a keyword argument is already
in the new form).
"""

import re
import sys

IDENT = re.compile(r"[A-Za-z0-9_]")
KW = re.compile(r"\s*[A-Za-z_][A-Za-z0-9_]*\s*=(?!=)")


def skip_string(t, i):
    """`t[i]` is `"` (or `f"`'s quote): the index after the closing quote."""
    fstr = i > 0 and t[i - 1] == "f" and (i < 2 or not IDENT.match(t[i - 2]))
    i += 1
    depth = 0
    while i < len(t):
        c = t[i]
        if c == "\\":
            i += 2
            continue
        if fstr and c == "{":
            if t.startswith("{{", i) and depth == 0:
                i += 2
                continue
            depth += 1
        elif fstr and c == "}" and depth > 0:
            depth -= 1
        elif c == '"':
            if depth == 0:
                return i + 1
            # a string inside an f-string field
            i = skip_string(t, i)
            continue
        elif c == "\n" and depth == 0:
            return i  # unterminated: stop at the line
        i += 1
    return i


def code_mask(t):
    """For each index of `t`, whether it is code (not a comment and not in a
    `###` fenced block); strings are code here, and `scan` steps over them."""
    mask = [True] * len(t)
    pos = 0
    fenced = False
    for line in t.split("\n"):
        end = pos + len(line)
        if line.rstrip() == "###":
            fenced = not fenced
            for k in range(pos, end):
                mask[k] = False
        elif fenced:
            for k in range(pos, end):
                mask[k] = False
        pos = end + 1
    return mask


def defines_print(t):
    return re.search(r"(^|\n)\s*def\s+print\s*\(", t) is not None


class Arg:
    def __init__(self, start, end, text):
        self.start, self.end = start, end  # the argument's text, trimmed
        self.text = text
        m = re.fullmatch(r'(f?)"(.*)"', text, re.S)
        self.lit = False
        if m and skip_string(text, len(m.group(1))) == len(text):
            content = m.group(2)
            if m.group(1) == "" or "{" not in content.replace("{{", ""):
                self.lit = True
                self.prefix = m.group(1)
                self.content = content
                self.cstart = start + len(m.group(1)) + 1
                self.cend = end - 1


def split_args(t, open_i):
    """The arguments of the call whose `(` is at `open_i`, and the index of
    its `)`; None when the call does not close."""
    i = open_i + 1
    depth = 0
    starts = [i]
    ends = []
    while i < len(t):
        c = t[i]
        if c == '"':
            i = skip_string(t, i)
            continue
        if c == "#":
            while i < len(t) and t[i] != "\n":
                i += 1
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                if c != ")":
                    return None
                ends.append(i)
                break
            depth -= 1
        elif c == "," and depth == 0:
            ends.append(i)
            starts.append(i + 1)
        i += 1
    else:
        return None
    args = []
    for s, e in zip(starts, ends):
        raw = t[s:e]
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw) - len(raw.rstrip())
        a, b = s + lead, e - trail
        if a >= b:
            if len(starts) == 1:
                return [], ends[-1]
            return None  # an empty argument: not a call we understand
        args.append(Arg(a, b, t[a:b]))
    return args, ends[-1]


def simulate(items):
    """The parser's rule (parser.rs, `print_call`) with the separator " ":
    `items` are (is_literal, content); the call's arguments after it."""
    out = []
    for k, (lit, c) in enumerate(items):
        if k > 0:
            if out and out[-1][0]:
                out[-1] = (True, out[-1][1] + " ")
            elif lit:
                c = " " + c
            else:
                out.append((True, " "))
        out.append((lit, c))
    return out


def rewrite(args):
    """The new arguments for one call: a list of (arg, new_content or None,
    keep) and whether `sep=""` is added; None when no rewrite passes."""
    old = [(a.lit, a.content if a.lit else a.text) for a in args]
    plan = []  # [arg, content, kept]
    ok = True
    k = 0
    while k < len(args):
        a = args[k]
        entry = [a, a.content if a.lit else None, True]
        k += 1
        if not plan:
            plan.append(entry)
            continue
        kept = [p for p in plan if p[2]]
        last = kept[-1]
        nxt = args[k] if k < len(args) else None
        if last[0].lit and last[1].endswith(" "):
            last[1] = last[1][:-1]
            before = kept[-2] if len(kept) >= 2 else None
            if last[1] == "" and not a.lit and not (before and before[0].lit):
                last[2] = False
        elif not last[0].lit and a.lit and entry[1] == " " and nxt is not None and not nxt.lit:
            # value, " ", value: the separator alone
            entry[2] = False
            plan.append(entry)
            plan.append([nxt, None, True])
            k += 1
            continue
        elif not last[0].lit and a.lit and entry[1].startswith(" "):
            entry[1] = entry[1][1:]
        else:
            ok = False
            break
        plan.append(entry)
    if ok:
        new = [(p[0].lit, p[1] if p[0].lit else p[0].text) for p in plan if p[2]]
        if simulate(new) == old:
            return plan, False
    # no separator: the arguments as they were, less the "" that only kept
    # two values apart
    plan = [[a, a.content if a.lit else None, not (a.lit and a.content == "")] for a in args]
    if not any(p[2] for p in plan):
        plan[0][2] = True
    return plan, True


def migrate_viv(t):
    """Edits (start, end, replacement) for every `print` call in Viv text `t`,
    and a list of calls left alone with the reason."""
    if defines_print(t):
        return [], ["defines its own print: left alone"]
    mask = code_mask(t)
    edits, notes = [], []
    i = 0
    while i < len(t):
        c = t[i]
        if not mask[i]:
            i += 1
            continue
        if c == '"':
            i = skip_string(t, i)
            continue
        if c == "#":
            while i < len(t) and t[i] != "\n":
                i += 1
            continue
        if t.startswith("print", i) and (i == 0 or not (IDENT.match(t[i - 1]) or t[i - 1] == ".")):
            j = i + 5
            if j < len(t) and IDENT.match(t[j]):
                i = j
                continue
            k = j
            while k < len(t) and t[k] in " \t":
                k += 1
            if k < len(t) and t[k] == "(":
                got = split_args(t, k)
                if got is None:
                    notes.append("line %d: a print call this script cannot read" % (t.count("\n", 0, i) + 1))
                    i = k + 1
                    continue
                args, close = got
                if len(args) >= 2 and not any(KW.match(a.text) for a in args):
                    edits.extend(call_edits(args, close))
                i = k + 1
                continue
            i = j
            continue
        i += 1
    return edits, notes


def call_edits(args, close):
    plan, sep = rewrite(args)
    edits = []
    kept_before = None
    for k, (a, content, kept) in enumerate(plan):
        if not kept:
            # drop the argument with the comma before it
            prev_end = plan[k - 1][0].end if k > 0 else a.start
            if k == 0:
                nxt = plan[1][0].start
                edits.append((a.start, nxt, ""))
            else:
                edits.append((prev_end, a.end, ""))
            continue
        if a.lit and content != a.content:
            edits.append((a.cstart, a.cend, content))
        kept_before = a
    if sep:
        edits.append((kept_before.end if kept_before else args[-1].end, kept_before.end if kept_before else args[-1].end, ', sep=""'))
    return edits


def apply(t, edits):
    for s, e, r in sorted(edits, key=lambda x: (x[0], x[1]), reverse=True):
        t = t[:s] + r + t[e:]
    return t


# ---- the Viv inside other files --------------------------------------------------------------

def rust_strings(t):
    """(start, end, raw) of each Rust string literal's contents: `raw` says
    whether escapes are text."""
    out = []
    i = 0
    n = len(t)
    while i < n:
        c = t[i]
        if t.startswith("//", i):
            while i < n and t[i] != "\n":
                i += 1
            continue
        if c == "'" :
            # a char literal ('"' or '\\'), or a lifetime
            m = re.match(r"'(\\.|[^\\'])'", t[i:])
            if m:
                i += m.end()
                continue
            i += 1
            continue
        m = re.match(r'(b?)r(#*)"', t[i:])
        if m and (i == 0 or not IDENT.match(t[i - 1])):
            hashes = m.group(2)
            s = i + m.end()
            e = t.index('"' + hashes, s)
            out.append((s, e, True))
            i = e + 1 + len(hashes)
            continue
        if c == '"':
            s = i + 1
            j = s
            while t[j] != '"':
                j += 2 if t[j] == "\\" else 1
            out.append((s, j, False))
            i = j + 1
            continue
        i += 1
    return out


def decode_rust(src):
    """The text of a normal Rust string literal, and for each of its
    characters (and one past the end) the offset in `src`."""
    text, pos = [], []
    i = 0
    simple = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\", '"': '"', "'": "'"}
    while i < len(src):
        c = src[i]
        if c == "\\":
            d = src[i + 1]
            if d in simple:
                text.append(simple[d])
                pos.append(i)
                i += 2
            elif d == "\n":
                i += 2
                while i < len(src) and src[i] in " \t\n\r":
                    i += 1
            elif d == "u":
                m = re.match(r"\\u\{([0-9a-fA-F]+)\}", src[i:])
                text.append(chr(int(m.group(1), 16)))
                pos.append(i)
                i += m.end()
            elif d == "x":
                text.append(chr(int(src[i + 2:i + 4], 16)))
                pos.append(i)
                i += 4
            else:
                raise ValueError("unknown escape \\%s" % d)
        else:
            text.append(c)
            pos.append(i)
            i += 1
    pos.append(len(src))
    return "".join(text), pos


def encode_rust(s):
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def migrate_embedded(t, spans, decode, encode):
    """Edits to `t` for the Viv inside each (start, end, raw) span."""
    edits, notes = [], []
    for s, e, raw in spans:
        body = t[s:e]
        if "print" not in body:
            continue
        if raw:
            text, pos = body, list(range(len(body) + 1))
            enc = lambda x: x
        else:
            try:
                text, pos = decode(body)
            except (ValueError, AttributeError, IndexError):
                notes.append("a string at offset %d this script cannot decode" % s)
                continue
            enc = encode
        ed, nt = migrate_viv(text)
        notes.extend(nt)
        for a, b, r in ed:
            edits.append((s + pos[a], s + pos[b], enc(r)))
    return edits, notes


def md_blocks(t):
    out = []
    for m in re.finditer(r"```viv[^\n]*\n(.*?)```", t, re.S):
        out.append((m.start(1), m.end(1), True))
    return out


def template_literals(t):
    out = []
    i = 0
    while True:
        i = t.find("`", i)
        if i < 0:
            return out
        j = i + 1
        while j < len(t) and t[j] != "`":
            j += 2 if t[j] == "\\" else 1
        out.append((i + 1, j, True))
        i = j + 1


def toml_strings(t):
    out = []
    for m in re.finditer(r"'''(.*?)'''|\"\"\"(.*?)\"\"\"", t, re.S):
        g = 1 if m.group(1) is not None else 2
        out.append((m.start(g), m.end(g), g == 1))
    return out


def migrate_file(path, t):
    if path.endswith(".viv") or path.endswith(".sh"):
        return migrate_viv(t)
    if path.endswith(".rs"):
        return migrate_embedded(t, rust_strings(t), decode_rust, encode_rust)
    if path.endswith(".md"):
        return migrate_embedded(t, md_blocks(t), None, None)
    if path.endswith((".mjs", ".js", ".jsx", ".astro")):
        return migrate_embedded(t, template_literals(t), None, None)
    if path.endswith(".toml"):
        return migrate_embedded(t, toml_strings(t), None, None)
    return [], ["not a kind of file this script reads"]


def main(argv):
    check = "--check" in argv
    files = [a for a in argv if a != "--check"]
    changed = 0
    for path in files:
        with open(path, encoding="utf-8") as f:
            t = f.read()
        edits, notes = migrate_file(path, t)
        for n in notes:
            print("%s: %s" % (path, n), file=sys.stderr)
        if not edits:
            continue
        new = apply(t, edits)
        if new != t:
            changed += 1
            print("%s: %d edit(s)" % (path, len(edits)))
            if not check:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new)
    print("%d file(s) %s" % (changed, "would change" if check else "changed"))
    return 1 if check and changed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
