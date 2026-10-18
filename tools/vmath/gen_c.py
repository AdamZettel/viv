#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Write lang/viv/src/vmath.c, the C twin of lang/viv/src/vmath/ (Rust).

Both are fdlibm as musl 1.1.19 carries it; the Rust side is the `libm`
crate 0.2.8, a line-by-line port of these files. This script takes musl's
C files, makes every function `static` and prefixes it with `viv_` (so a
program's C file carries its own copy and never reaches the platform libm),
gives each file's private constants a prefix of their own (the files reuse
names such as `pio2_hi` with different values), and appends vmath_tail.c
(Viv's erf, gamma and complex functions, and the #defines that route the
generated C's `sin(...)` to `viv_sin`).

    tools/vmath/gen_c.py <musl-1.1.19 source dir>

musl: https://musl.libc.org/releases/musl-1.1.19.tar.gz (MIT).
`cargo test -p viv --test vmath` checks the two twins bit for bit.
"""
import os
import re
import sys

MUSL = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'lang', 'viv', 'src', 'vmath.c')

# dependency order: a function is defined before its first caller
FILES = ['scalbn', '__rem_pio2_large', '__rem_pio2', '__sin', '__cos', '__tan',
         'sin', 'cos', 'tan', 'exp', 'expm1', '__expo2', 'log', 'log10', 'pow',
         'atan', 'atan2', 'asin', 'acos', 'sinh', 'cosh', 'tanh', 'hypot']
# musl name -> Viv's; musl's internal __x become viv__x
FUNCS = {f: 'viv_' + f.lstrip('_') if not f.startswith('__') else 'viv__' + f[2:] for f in FILES}


def strip_comments(s):
    return re.sub(r'/\*.*?\*/', lambda m: '\n' * m.group(0).count('\n'), s, flags=re.S)


def file_scope_names(src):
    """Identifiers a file declares `static` at file scope: its constants,
    tables and helper functions."""
    s = strip_comments(src)
    s = re.sub(r'^#.*$', '', s, flags=re.M)
    # collapse every brace group to {} (bodies and initializers alike)
    while True:
        t = re.sub(r'\{[^{}]*\}', '<>', s)
        if t == s:
            break
        s = t
    s = re.sub(r'\)\s*<>', ');', s)  # a function definition ends a statement
    names = set()
    for stmt in s.split(';'):
        stmt = stmt.strip()
        if not stmt.startswith('static'):
            continue
        head = stmt.split('=')[0]
        if '(' in head:  # a helper function
            names.add(re.search(r'(\w+)\s*\(', head).group(1))
            continue
        body = re.sub(r'^static\s+(const\s+)?(unsigned\s+)?\w+\s*', '', stmt)
        for d in split_top(body):
            m = re.match(r'\s*\*?\s*(\w+)', d)
            if m:
                names.add(m.group(1))
    return names


def split_top(s):
    """Split on the commas outside any brackets."""
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch in '{(<':
            depth += 1
        elif ch in '})>':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur)
            cur = ''
        else:
            cur += ch
    out.append(cur)
    return out


def sub_code(pattern, repl, src):
    """re.sub on the code only: comments keep musl's words."""
    parts = re.split(r'(/\*.*?\*/)', src, flags=re.S)
    return ''.join(p if p.startswith('/*') else re.sub(pattern, repl, p) for p in parts)


def convert(name):
    src = open(os.path.join(MUSL, 'src', 'math', name + '.c')).read()
    src = re.sub(r'^#include .*\n', '', src, flags=re.M)
    tag = FUNCS[name]
    for n in sorted(file_scope_names(src)):
        src = sub_code(r'\b%s\b' % re.escape(n), '%s_%s' % (tag, n), src)
    defines = re.findall(r'^#define\s+(\w+)', src, flags=re.M)
    for f, v in FUNCS.items():
        src = sub_code(r'(?<![\w.])%s(?=\s*\()' % re.escape(f), v, src)
    # every function definition becomes static: `double sin(double x)` at
    # the start of a line, outside any body
    src = re.sub(r'^(double|int|float)(\s+viv_\w+\()', r'static \1\2', src, flags=re.M)
    src = src.replace('double_t', 'double')
    undef = ''.join('#undef %s\n' % d for d in sorted(set(defines)))
    return '/* ---- musl src/math/%s.c ---- */\n%s\n%s' % (name, src.strip('\n'), undef)


head = open(os.path.join(HERE, 'vmath_head.c')).read()
tail = open(os.path.join(HERE, 'vmath_tail.c')).read()
body = '\n'.join(convert(f) for f in FILES)
open(OUT, 'w').write(head + '\n' + body + '\n' + tail)
print('wrote', os.path.relpath(OUT))
