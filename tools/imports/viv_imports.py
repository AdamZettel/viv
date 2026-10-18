# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Imports for generated Viv programs.

A generator that writes a Viv program knows which modules the program needs
but not which names it reads from each, and a program must import every name
it reads from a module by name (`from m import a, b`). So a generator writes
the modules as `#! use <path>` lines, as it always did, and `migrated` turns
them into the import lines `viv imports` writes for any file: the same tool,
so a generated file and a migrated one cannot differ.

    from viv_imports import migrated
    text = migrated(text, "lib/algo/tests/sort.viv")

`path` is where the program will live (modules beside it are found from
there); the text is written to a scratch file in that directory, rewritten by
`viv imports --write`, and read back. It needs a built compiler
(`cargo build --release -p viv`; VIV=<path> for another).
"""
import os
import subprocess
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIV = os.environ.get("VIV", os.path.join(ROOT, "target", "release", "viv"))


def migrated(text, path):
    """`text` with its `#! use` lines rewritten as imports, read as if at `path`."""
    if not any(l.startswith("#! use ") for l in text.splitlines()):
        return text
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".imports-", suffix=".viv", dir=d)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        p = subprocess.run([VIV, "imports", "--write", tmp], capture_output=True, text=True, cwd=ROOT)
        if p.returncode != 0:
            raise RuntimeError(f"viv imports on the program for {path}: {p.stderr.strip()}")
        with open(tmp) as f:
            return f.read()
    finally:
        os.unlink(tmp)
