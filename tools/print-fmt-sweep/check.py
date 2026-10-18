# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Hold every field of sweep.viv's lines to CPython.

    python3 tools/print-fmt-sweep/check.py <output of sweep.viv> ...

Each line is one double written 26 ways (sweep.viv's `print`). The first
field is `fmt(x, 17)`, C's %.17g, which always reads back as x; the rest are
compared with what CPython writes for the same double:

    fmt(x)             repr(x)
    x (bare)           format(x + 0.0, '.6g')    (a negative zero is 0, as `show`)
    fmt(x, p)          format(x, '.<p>g'), p = 1..16 (and 17: the first field)
    fmt(x, ".0e")      format(x, '.0e'), and '.3e', '.16e'
    fmt(x, ".2f")      format(x, '.2f'), and '.6f'
    fmt(x, "d")        format(x, '.0f')
    fmt(x, "12.4g")    format(x, '12.4g')

Prints a count per field and the first mismatches; exits 1 on any.
"""

import math
import sys

SPECS = (
    ["repr", "bare"]
    + [f".{p}g" for p in range(1, 17)]
    + [".0e", ".3e", ".16e", ".2f", ".6f", ".0f", "12.4g"]
)


def want(x, spec):
    if spec == "repr":
        return repr(x)
    if spec == "bare":
        return format(x + 0.0, ".6g")
    return format(x, spec)


def main(paths):
    lines = 0
    bad = {s: 0 for s in [".17g"] + SPECS}
    shown = 0
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                fields = line.split(" ", len(SPECS))
                if len(fields) != len(SPECS) + 1:
                    print(f"{path}: a line with {len(fields)} fields: {line[:120]}")
                    return 1
                lines += 1
                x = float(fields[0])
                checks = [(".17g", fields[0])] + list(zip(SPECS, fields[1:]))
                for spec, got in checks:
                    w = want(x, spec)
                    if got != w:
                        bad[spec] += 1
                        if shown < 20:
                            shown += 1
                            print(f"x = {x!r} ({x.hex() if math.isfinite(x) else x}): {spec}: viv {got!r}, python {w!r}")
    total = sum(bad.values())
    print(f"{lines} doubles, {len(bad)} formats each, {lines * len(bad)} fields; {total} differ from CPython {sys.version.split()[0]}")
    for spec, n in bad.items():
        if n:
            print(f"  {spec}: {n}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
