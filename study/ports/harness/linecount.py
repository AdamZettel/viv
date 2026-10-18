# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Code-line counts for a port: the lines of the original and of the Viv
that hold code.

  Python: a line counts if it holds a token other than a comment, newline or
          indent and is not inside a docstring (the first string statement
          of a module, class or def).
  Viv:    non-blank, not a `#` comment line (this drops `#!` directives and
          `#|` notes), not inside a `###` fenced block, not inside a
          `test "..." { }` block. Lines that are only `}` (or `} else {`) are
          returned apart as closers, so the brace cost can be shown.

Standard library only."""
import ast
import io
import re
import tokenize


def py_docstring_lines(text):
    out = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
                out.update(range(body[0].lineno, body[0].end_lineno + 1))
    return out


def py_code_lines(text):
    """Set of 1-based line numbers that hold Python code."""
    doc = py_docstring_lines(text)
    lines = set()
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER}
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type in skip:
            continue
        for ln in range(tok.start[0], tok.end[0] + 1):
            lines.add(ln)
    return {l for l in lines if l not in doc}


def viv_code_lines(text):
    """(code lines, closer lines) of a Viv file, as sets of 1-based line numbers."""
    lines, closers = set(), set()
    fenced = False
    depth_test = 0
    in_test = False
    for i, raw in enumerate(text.split("\n"), 1):
        s = raw.strip()
        if s == "###":
            fenced = not fenced
            continue
        if fenced:
            continue
        if not s or s.startswith("#"):
            continue
        code = re.sub(r'"(\\.|[^"\\])*"', '""', s)
        code = code.split("#", 1)[0]
        if not in_test and re.match(r'^test\s+"', s):
            in_test = True
            depth_test = 0
        if in_test:
            depth_test += code.count("{") - code.count("}")
            if depth_test <= 0:
                in_test = False
            continue
        lines.add(i)
        if re.fullmatch(r"\}(\s*else\s*\{)?", code.strip()) or re.fullmatch(r"\}\s*else\s+if .*\{", code.strip()):
            closers.add(i)
    return lines, closers
