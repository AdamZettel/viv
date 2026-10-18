#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Count what a pair's two programs spend, region by region, and keep the counts
in the pair's notes.

    python3 study/compare/measure.py            print the tables
    python3 study/compare/measure.py --update   write them into the notes
    python3 study/compare/measure.py --check    fail if the notes are stale

A pair takes part when MEASURE below names it. The tables go between the
lines `%% measured` and `%% end` of the notes (compare-build leaves `%%` lines
out of the page). What is counted:

- code lines: lines that are not blank and not comments (`#` lines, and in
  Python docstring lines), per `# @@` region; a Viv `#! use` line is code,
  as a Python `import` line is;
- the parameters of the defs the forward and backward passes run through,
  `self` not counted (MEASURE[pair]['passes'] names them per language), and
  how many of them are output parameters, arrays the def writes its results
  into;
- parameter lookups: every place a program finds a weight matrix or a size
  by index or by name (MEASURE[pair]['lookups'], a regex per language).

- characters of code: the characters of those lines, leading and trailing
  spaces not counted, per region.

A pair with a third program, <name>.phrased.viv (the Viv written with
phrases), gets more columns: its code lines,
lookups and characters per region.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARK = re.compile(r'^\s*# @@ (.+?)\s*$')

VIV_PASSES = r'^(embed|self_attention|block|gpt_forward|softmax_rows|attention|row_rms|rmsnorm|relu2|cross_entropy|sequence_loss)$'
# a weight fetched or stored by index in a flat parameter vector, or a size
# read from the config record (`c.width`, `m.cfg.heads`)
VIV_LOOKUPS_TORCH = r'\b(param_get|param_set|param_add|param_add_row|param_offset|param_rows|param_cols)\(|\b(m\.cfg|c)\.\w'
# and, as the NumPy column counts `params["q"]`, a weight read from its record
# by name (`h.Wq`, `b.mlp.up`, `m.tok`)
VIV_LOOKUPS = VIV_LOOKUPS_TORCH + r'|\.(Wq|Wk|Wv|Wo|up|down|tok|pos|unembed)\b'

MEASURE = {
    'ml/tiny_gpt': {
        'passes': {'py': r'^(forward|backward|loss|softmax|rmsnorm|rmsnorm_backward)$', 'viv': VIV_PASSES},
        # a weight by name, or a size from the config record
        'lookups': {'py': r'params\["|\bp\["|cfg\.\w', 'viv': VIV_LOOKUPS},
    },
    'ml/tiny_gpt_torch': {
        'passes': {'py': r'^(forward|rmsnorm)$', 'viv': VIV_PASSES},
        # torch's `self.q(x)` is not a lookup, and neither is Viv's `h.Wq`
        'lookups': {'py': r'cfg\.\w', 'viv': VIV_LOOKUPS_TORCH},
    },
}


def code_lines(src, lang):
    """[(region, line)] for every line that is code, the region by its marker title."""
    out = []
    region = None
    in_doc = False
    for line in src.split('\n'):
        m = MARK.match(line)
        if m:
            region = m.group(1)
            continue
        s = line.strip()
        if lang == 'py':
            if in_doc:
                in_doc = '"""' not in s
                continue
            if s.startswith('"""'):
                in_doc = s.count('"""') == 1
                continue
        # a Viv `#!` directive line is code; an import line is code anyway
        if not s or (s.startswith('#') and not (lang == 'viv' and s.startswith('#! '))):
            continue
        out.append((region, line))
    return out


def defs(src, lang):
    """[(name, parameters, output parameters)] of every def, `self` not counted. An
    output parameter is one the def writes into: `p[...] = ...` (or `+=` and the
    like), or `p` handed to setslab, param_set, param_add or param_add_row."""
    lines = src.split('\n')
    found = []
    for i, line in enumerate(lines):
        m = re.match(r'^(\s*)def (\w+)\(([^)]*)\)', line)
        if not m:
            continue
        indent = len(m.group(1))
        j = i + 1
        while j < len(lines):
            l = lines[j]
            if lang == 'viv' and l.startswith(' ' * indent + '}') and len(l) - len(l.lstrip()) == indent:
                break
            if lang == 'py' and l.strip() and len(l) - len(l.lstrip()) <= indent:
                break
            j += 1
        body = '\n'.join(lines[i + 1:j])
        params = [p.split(':')[0].split('=')[0].strip() for p in m.group(3).split(',') if p.strip()]
        params = [p for p in params if p != 'self']
        written = [p for p in params
                   if re.search(rf'^\s*{p}\[[^\n]*\]\s*[-+*/]?=[^=]', body, re.M)
                   or re.search(rf'\b(setslab|param_set|param_add|param_add_row)\({p}\b', body)]
        found.append((m.group(2), len(params), len(written)))
    return found


def measure(topic, name):
    base = os.path.join(HERE, topic, name)
    spec = MEASURE[f'{topic}/{name}']
    src = {'py': open(base + '.py').read(), 'viv': open(base + '.viv').read()}
    titles = [MARK.match(l).group(1) for l in src['py'].split('\n') if MARK.match(l)]
    rows = {t: {'py': 0, 'viv': 0, 'phr': 0, 'py_look': 0, 'viv_look': 0, 'phr_look': 0, 'py_chars': 0, 'viv_chars': 0, 'phr_chars': 0} for t in titles}
    langs = [('py', 'py'), ('viv', 'viv')]
    if os.path.exists(base + '.phrased.viv'):
        src['phr'] = open(base + '.phrased.viv').read()
        langs.append(('phr', 'viv'))
    for key, lang in langs:
        rx = re.compile(spec['lookups'][lang])
        for region, line in code_lines(src[key], lang):
            rows[region][key] += 1
            rows[region][key + '_look'] += len(rx.findall(line))
            rows[region][key + '_chars'] += len(line.strip())
    passes = {}
    for lang in ('py', 'viv'):
        rx = re.compile(spec['passes'][lang])
        ds = [d for d in defs(src[lang], lang) if rx.match(d[0])]
        passes[lang] = ds
    return titles, rows, passes, 'phr' in src


def ratio(a, b):
    return '' if a == 0 else f'{b / a:.1f}'


def tables(topic, name):
    titles, rows, passes, phrased = measure(topic, name)
    extra_head = ' Viv in phrases lines | Viv in phrases lookups |' if phrased else ''
    extra_rule = '---|---|' if phrased else ''
    out = [f'| region | Python lines | Viv lines | Viv / Python | Python lookups | Viv lookups |{extra_head}', f'|---|---|---|---|---|---|{extra_rule}']
    tot = {k: 0 for k in rows[titles[0]]} if titles else {}
    for t in titles:
        r = rows[t]
        extra = f" {r['phr']} | {r['phr_look']} |" if phrased else ''
        out.append(f"| {t} | {r['py']} | {r['viv']} | {ratio(r['py'], r['viv'])} | {r['py_look']} | {r['viv_look']} |{extra}")
        for k in tot:
            tot[k] += r[k]
    extra = f" **{tot['phr']}** | **{tot['phr_look']}** |" if phrased else ''
    out.append(f"| **all** | **{tot['py']}** | **{tot['viv']}** | **{ratio(tot['py'], tot['viv'])}** | **{tot['py_look']}** | **{tot['viv_look']}** |{extra}")
    out.append('')
    # characters of code, region by region
    head = '| characters of code | Python | Viv | Viv / Python |' + (' Viv in phrases |' if phrased else '')
    out.append(head)
    out.append('|---|---|---|---|' + ('---|' if phrased else ''))
    for t in titles:
        r = rows[t]
        extra = f" {r['phr_chars']} |" if phrased else ''
        out.append(f"| {t} | {r['py_chars']} | {r['viv_chars']} | {ratio(r['py_chars'], r['viv_chars'])} |{extra}")
    extra = f" **{tot['phr_chars']}** |" if phrased else ''
    out.append(f"| **all** | **{tot['py_chars']}** | **{tot['viv_chars']}** | **{ratio(tot['py_chars'], tot['viv_chars'])}** |{extra}")
    out.append('')
    out.append('| the forward and backward defs | Python | Viv |')
    out.append('|---|---|---|')
    p, v = passes['py'], passes['viv']
    out.append(f'| defs | {len(p)} | {len(v)} |')
    out.append(f'| parameters, all of them together | {sum(d[1] for d in p)} | {sum(d[1] for d in v)} |')
    out.append(f'| output parameters (arrays the def itself writes results into) | {sum(d[2] for d in p)} | {sum(d[2] for d in v)} |')
    most = lambda ds: max(ds, key=lambda d: d[1]) if ds else ('', 0, 0)
    mp, mv = most(p), most(v)
    out.append(f'| most in one def | {mp[1]} (`{mp[0]}`) | {mv[1]} (`{mv[0]}`) |')
    return '\n'.join(out)


def pairs():
    for key in MEASURE:
        topic, name = key.split('/')
        yield topic, name, os.path.join(HERE, topic, name + '.md')


def with_tables(md, table):
    lines = md.split('\n')
    try:
        a = lines.index('%% measured')
        b = lines.index('%% end', a)
    except ValueError:
        raise SystemExit('the notes have no `%% measured` ... `%% end` block')
    return '\n'.join(lines[:a + 1] + table.split('\n') + lines[b:])


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    bad = 0
    for topic, name, md_path in pairs():
        table = tables(topic, name)
        md = open(md_path).read()
        new = with_tables(md, table)
        if mode == '--update':
            open(md_path, 'w').write(new)
        elif mode == '--check':
            if new != md:
                print(f'{topic}/{name}: the measured tables in the notes are stale (run study/compare/measure.py --update)')
                bad = 1
            else:
                print(f'{topic}/{name}: measured tables current')
        else:
            print(f'== {topic}/{name}\n{table}\n')
    sys.exit(bad)


if __name__ == '__main__':
    main()
