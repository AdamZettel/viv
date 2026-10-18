# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# REUSE-IgnoreStart (this file writes the tags; it does not use them)
"""Add the SPDX header to hand-written source files that lack one.

    python3 tools/licenses/add_headers.py            # every tracked file check.py asks for
    python3 tools/licenses/add_headers.py FILE...    # these files

The license and copyright come from the REUSE.toml annotation that covers
the file (Viv's own, or a port's upstream). The header goes at the top, after
a `#!` line, a coding line or Astro's front-matter fence; in a .viv file it
goes at the end, so the line numbers of the program (in diagnostics, in
expected outputs, in the docs) stay what they were.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check  # noqa: E402

STYLE = {
    "#": {".py", ".viv", ".vivh", ".sh", ".toml", ".yml", ".yaml", ".jl", ".pl"},
    "//": {".rs", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".cpp", ".astro"},
    "/*": {".c", ".h", ".css"},
    "--": {".sql"},
    "!": {".f90"},
    "C": {".f"},
}


def comment(lines, suffix):
    for mark, exts in STYLE.items():
        if suffix in exts:
            if mark == "/*":
                return ["/* %s */" % l for l in lines]
            return ["%s %s" % (mark, l) for l in lines]
    return None


def header_lines(path, anns):
    hit = check.toml_info(path, anns)
    lic = hit["license"] if hit else ["MIT OR Apache-2.0"]
    cop = hit["copyright"] if hit else ["2026 The Viv authors"]
    return ["SPDX-License-Identifier: %s" % " AND ".join(lic)] + ["SPDX-FileCopyrightText: %s" % c for c in cop]


def add(path, anns):
    suffix = Path(path).suffix
    full = check.ROOT / path
    text = full.read_text(encoding="utf-8")
    if check.header_info(text)[0]:
        return False
    head = header_lines(path, anns)
    if "SunPro" in head[0] and "Sun Microsystems" not in text:
        # musl's own code in vmath: MIT only
        head = ["SPDX-License-Identifier: MIT"] + [l for l in head[1:] if "Sun Microsystems" not in l]
    block = comment(head, suffix)
    if block is None:
        return False
    lines = text.split("\n")
    if suffix in (".viv", ".vivh"):
        body = text.rstrip("\n")
        full.write_text(body + "\n\n" + "\n".join(block) + "\n", encoding="utf-8")
        return True
    at = 0
    if suffix == ".astro":
        if lines and lines[0].strip() == "---":
            at = 1
        else:
            block = ["---"] + block + ["---"]
    else:
        while at < len(lines) and at < 2 and (lines[at].startswith("#!") or "coding" in lines[at][:30] and lines[at].startswith("#")):
            at += 1
    full.write_text("\n".join(lines[:at] + block + lines[at:]), encoding="utf-8")
    return True


def main(argv):
    anns = check.annotations()
    if argv:
        paths = argv
    else:
        paths = []
        for path in check.tracked():
            if check.IGNORED.match(path) or Path(path).suffix not in check.COMMENTED:
                continue
            hit = check.toml_info(path, anns)
            if hit and hit["precedence"] == "override":
                continue
            text = check.read_text(check.ROOT / path)
            if path in check.HEADER_IN_TOML or text is None or check.GENERATED.search("\n".join(text.splitlines()[:15])):
                continue
            paths.append(path)
    n = sum(add(p, anns) for p in paths)
    print("add_headers: %d files" % n)


if __name__ == "__main__":
    main(sys.argv[1:])
# REUSE-IgnoreEnd
