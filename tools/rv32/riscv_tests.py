# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Turn the official riscv-tests rv32ui and rv32um suites into Viv tests.

    python3 tools/rv32/riscv_tests.py RISCV_TESTS_DIR [--check]

RISCV_TESTS_DIR is a checkout of github.com/riscv-software-src/riscv-tests
(BSD-3-Clause, The Regents of the University of California); the commit
used is PINNED below. Each isa/rv32ui/*.S and isa/rv32um/*.S file goes
through the host C preprocessor (`clang -E`, only as a preprocessor: no
RISC-V toolchain is involved) with the environment in
tools/rv32/env/riscv_test.h. The test macros leave constant expressions
such as ((0xffffffff80000000) & ((1 << (32 - 1) << 1) - 1)) in the text;
this script folds them to integers, since the assembler in lib/rv32/asm.viv
reads only literals. Everything else, instructions, labels and data, is
left for lib/rv32/asm.viv to assemble and lib/rv32/sim.viv to run.

Each test becomes lib/rv32/tests/riscv/<suite>_<name>.viv, which asserts
that the program exits with 0 (a failing case exits with its number).
--check regenerates into memory and fails if a committed file differs.
"""

import re
import subprocess
import sys
from pathlib import Path

PINNED = "7b68f7b91f827f8d94cc1c57ff123752c5c8f704"
ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / "tools" / "rv32" / "env"
OUT = ROOT / "lib" / "rv32" / "tests" / "riscv"
SUITES = ["rv32ui", "rv32um"]
# fence_i writes instructions into memory through `sh rs, label, tmp`, a
# store-to-symbol pseudo-instruction the assembler does not read;
# ma_data tests misaligned accesses that the reference environment
# handles with a trap handler, which this environment does not have.
SKIP = {
    "rv32ui": {
        "fence_i": "store-to-symbol pseudo-instruction",
        "ma_data": "needs a misaligned-access trap handler",
    }
}

INNER = re.compile(r"\(([^()]*)\)")
HEX = re.compile(r"0x[0-9a-fA-F]+")
CONSTANT = re.compile(r"[-+*&|^~<> 0-9]*[0-9][-+*&|^~<> 0-9]*")


def fold(line):
    """Replace innermost parenthesized constant expressions by their values
    until none is left; `8(x2)` and other register operands stay."""
    start = 0
    while True:
        m = INNER.search(line, start)
        if not m:
            return line
        expr = m.group(1)
        if CONSTANT.fullmatch(HEX.sub("0", expr)):
            line = (
                line[: m.start()]
                + str(eval(expr, {"__builtins__": {}}))
                + line[m.end() :]
            )
            start = 0
        else:
            start = m.end()


def preprocess(path):
    out = subprocess.run(
        [
            "clang",
            "-E",
            "-P",
            "-x",
            "assembler-with-cpp",
            "-w",
            f"-I{ENV}",
            f"-I{path.parents[1] / 'macros' / 'scalar'}",
            "-D__riscv_xlen=32",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    lines = []
    for raw in out.split("\n"):
        for stmt in raw.split("#")[0].split(";"):
            s = stmt.strip()
            if s:
                lines.append(fold(s))
    return lines


def program(suite, name, lines):
    body = "\n".join(lines).replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    return f'''# riscv-tests {suite}/{name}.S, preprocessed by tools/rv32/riscv_tests.py
# (riscv-tests commit {PINNED[:12]}; BSD-3-Clause, The Regents of the
# University of California). Generated: do not edit.

from rv32.system import run_asm, STEP_LIMIT

SOURCE = """
{body}
"""

result = run_asm(SOURCE, STEP_LIMIT)
show "{suite}-{name}", result.exit_code, result.steps

test "{suite}-{name} passes every case" {{
    assert_true(result.halted)
    assert_true(result.exit_code == 0)
}}
'''


def main():
    src = Path(sys.argv[1])
    check = "--check" in sys.argv
    want = {}
    for suite in SUITES:
        for path in sorted((src / "isa" / suite).glob("*.S")):
            name = path.stem
            if name in SKIP.get(suite, {}):
                continue
            want[f"{suite}_{name}.viv"] = program(
                suite, name, preprocess(path)
            )
    stale = [
        n
        for n, text in want.items()
        if not (OUT / n).exists() or _body(OUT / n) != text
    ]
    if check:
        if stale:
            print("riscv_tests: out of date:", ", ".join(stale))
            return 1
        print(f"riscv_tests: {len(want)} programs current")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for n in stale:
        (OUT / n).write_text(want[n])
    print(
        f"riscv_tests: wrote {len(stale)} of {len(want)} programs into {OUT}"
    )
    return 0


def _body(path):
    """A committed program without the SPDX lines the license tool appends."""
    text = path.read_text()
    return re.sub(r"\n# SPDX-[^\n]*", "", text).rstrip("\n") + "\n"


if __name__ == "__main__":
    sys.exit(main())
