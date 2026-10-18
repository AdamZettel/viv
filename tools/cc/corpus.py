# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Check lib/cc against clang: the C corpus in tests/cc, compiled both ways.

    python3 tools/cc/corpus.py            # write lib/cc/tests/corpus.viv
    python3 tools/cc/corpus.py --check    # fail if it is out of date

Each tests/cc/*.c is built for the host by clang -O0 with tests/cc/prelude.h
(which defines print_int with printf) and run; its standard output and exit
status are the reference. The generated Viv program compiles the same
source with lib/cc, runs it on lib/rv32's simulator, and asserts the same
output and the same exit status (the low 8 bits, which is what a host
process reports).
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "tests" / "cc"
OUT = ROOT / "lib" / "cc" / "tests" / "corpus.viv"
CLANG = [
    "clang",
    "-O0",
    "-std=c99",
    "-Wall",
    "-Werror",
    "-Wno-constant-logical-operand",  # chibicc's tests write 0 || (2 - 2)
]
EXIT_MOD = 256  # a host process reports the low 8 bits of main's value


def reference(path):
    """stdout and exit status of the program built natively by clang."""
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "a.out")
        subprocess.run(
            CLANG
            + ["-include", str(CORPUS / "prelude.h"), "-o", exe, str(path)],
            check=True,
        )
        run = subprocess.run([exe], capture_output=True, text=True, timeout=60)
    return run.stdout, run.returncode


def viv_string(text):
    return (
        '"'
        + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        + '"'
    )


def program():
    parts = [
        "# The C corpus of tests/cc compiled by lib/cc and run on lib/rv32's",
        "# simulator, against clang -O0 on the host: written by",
        "# tools/cc/corpus.py, which ran each program natively for the expected",
        "# output and exit status. Generated: do not edit.",
        "",
        "from cc.driver import run_c",
        "",
        "EXIT_MOD: Int = 256       # a host process reports the low 8 bits of main's value",
        "",
    ]
    for path in sorted(CORPUS.glob("*.c")):
        name = path.stem
        out, code = reference(path)
        src = path.read_text()
        var = "SRC_" + re.sub(r"\W", "_", name).upper()
        parts += [
            f'{var} = """{src}"""',
            "",
            f'test "{path.name}: the output and exit status clang -O0 gives" {{',
            f"    m = run_c({var})",
            "    assert_true(m.halted)",
            f"    assert_true(m.output == {viv_string(out)})",
            f"    assert_true(m.exit_code % EXIT_MOD == {code})",
            "}",
            "",
        ]
    return "\n".join(parts)


def _body(path):
    """The committed program without the SPDX lines the license tool appends."""
    text = path.read_text()
    return re.sub(r"\n# SPDX-[^\n]*", "", text).rstrip("\n") + "\n"


def main():
    want = program().rstrip("\n") + "\n"
    if "--check" in sys.argv:
        if not OUT.exists() or _body(OUT) != want:
            print("corpus: lib/cc/tests/corpus.viv is out of date")
            return 1
        print(f"corpus: {len(list(CORPUS.glob('*.c')))} programs current")
        return 0
    OUT.write_text(want)
    print(f"corpus: wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
