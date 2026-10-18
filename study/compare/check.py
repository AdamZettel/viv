#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Run every Python/Viv pair in study/compare and compare what they print.

    python3 study/compare/check.py              check every pair
    python3 study/compare/check.py strings      the pairs of one topic
    python3 study/compare/check.py --update     also rewrite each pair's .gen.json

A pair is three files in a topic directory: <name>.py, <name>.viv and <name>.md.

- Both programs are split into regions by marker comments, `# @@ <title>` on a
  line of its own. The two files must have the same titles in the same order,
  so the viewer (viv-lang.org/compare) can line them up region by region.
- <name>.md starts with front matter between `---` lines:

      title: Word frequency           the page's title
      order: 3                        its place in the topic
      output: identical               or `differs`, with a `reason:` line
      viv_drop: ^figure:|^plot:       lines left out of Viv's output (a regex)
      py_drop: ^saved                 lines left out of Python's output
      keep_pass: yes                  compare Viv's PASS lines too

  and the rest is the notes: the syntax differences a reader should notice.

Viv's `test` blocks print one `PASS <name>` line each; the Python programs
check the same facts with bare `assert`s, which print nothing. So Viv's PASS
lines are left out before comparing, except in a pair that says `keep_pass`.
After the drops the two outputs must be identical, for every pair: a pair that
says `output: differs` states in `reason` what the drops take out.

Python is run with this interpreter (NumPy 1.24.4 for the random pair), in a
fresh temporary directory, so a pair that writes a file writes it there. Viv
is run in the interpreter (`viv run`); tools/three-engines.sh checks that the
C and wasm engines print the same bytes (check.sh runs both).

<name>.gen.json holds what each program printed and what py2viv made of the
Python (its translation, or its refusal). The site's pages are built from it,
so the site needs neither Python nor a compiled viv; this script fails when
the file is out of date, and `--update` rewrites it.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
VIV = os.environ.get('VIV', os.path.join(ROOT, 'target/release/viv'))
PY2VIV = os.path.join(ROOT, 'tools/py2viv/py2viv.py')
MARK = re.compile(r'^\s*# @@ (.+?)\s*$')


def front_matter(text):
    """The `key: value` lines between the first two `---` lines, and the rest."""
    lines = text.split('\n')
    if not lines or lines[0].strip() != '---':
        raise ValueError('the notes must start with a --- line')
    meta = {}
    for i in range(1, len(lines)):
        if lines[i].strip() == '---':
            return meta, '\n'.join(lines[i + 1:]).strip() + '\n'
        key, _, value = lines[i].partition(':')
        meta[key.strip()] = value.strip()
    raise ValueError('the front matter has no closing --- line')


def regions(src):
    """The marker titles of a program, in order."""
    return [m.group(1) for m in (MARK.match(line) for line in src.split('\n')) if m]


def run(cmd, cwd, env=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env, timeout=600)
    return p.returncode, p.stdout, p.stderr


def drop(text, pattern, keep_pass=True):
    lines = text.split('\n')
    if not keep_pass:
        lines = [l for l in lines if not l.startswith('PASS ')]
    if pattern:
        rx = re.compile(pattern)
        lines = [l for l in lines if not rx.search(l)]
    return '\n'.join(lines)


def py2viv(py_path, py_out):
    """py2viv's translation of the Python, or its refusal, with local paths
    removed; a translation is run in the interpreter, and its output kept."""
    code, out, err = run([sys.executable, PY2VIV, py_path], cwd=os.path.dirname(py_path))
    text = out if code == 0 else err
    text = text.replace(os.path.dirname(py_path) + '/', '').replace(ROOT + '/', '')
    result = {'ok': code == 0, 'text': text}
    if code == 0:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'translated.viv')
            with open(path, 'w') as f:
                f.write(out)
            vc, vout, verr = run([VIV, 'run', '--quiet', path], cwd=tmp)
        result['output'] = vout if vc == 0 else vout + verr.replace(tmp + '/', '')
        result['same_output'] = vc == 0 and vout == py_out
    return result


def figure_svg(viv_out):
    """The figure records of a Viv run drawn by `viv plot`, or None."""
    if not re.search(r'^figure:', viv_out, re.M):
        return None
    p = subprocess.run([VIV, 'plot', '-o', '/dev/stdout'], input=viv_out, capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


def check_pair(topic, name, update):
    base = os.path.join(HERE, topic, name)
    py_src = open(base + '.py').read()
    viv_src = open(base + '.viv').read()
    meta, notes = front_matter(open(base + '.md').read())
    problems = []
    rp, rv = regions(py_src), regions(viv_src)
    if not rp:
        problems.append('no `# @@` regions')
    if rp != rv:
        problems.append(f'regions differ:\n    python: {rp}\n    viv:    {rv}')
    for key in ('title', 'order', 'output'):
        if key not in meta:
            problems.append(f'the notes have no `{key}:`')
    if meta.get('output') not in ('identical', 'differs'):
        problems.append('`output:` is `identical` or `differs`')
    if meta.get('output') == 'differs' and not meta.get('reason'):
        problems.append('`output: differs` needs a `reason:`')
    if meta.get('output') == 'identical' and (meta.get('viv_drop') or meta.get('py_drop')):
        problems.append('an `identical` pair drops no lines')

    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, MPLBACKEND='Agg', PYTHONHASHSEED='0')
        pc, pout, perr = run([sys.executable, base + '.py'], cwd=tmp, env=env)
        vc, vout, verr = run([VIV, 'run', '--quiet', base + '.viv'], cwd=tmp)
        # a picture the Python saved is kept beside the pair (not compared:
        # matplotlib's bytes change between versions)
        pngs = sorted(f for f in os.listdir(tmp) if f.endswith('.png'))
        if update and pngs:
            shutil.copyfile(os.path.join(tmp, pngs[0]), base + '.py.png')
    if pc != 0:
        problems.append(f'python exited {pc}:\n{perr[-800:]}')
    if vc != 0:
        problems.append(f'viv exited {vc}:\n{verr[-800:]}')
    keep = meta.get('keep_pass') == 'yes'
    if not keep and not re.search(r'^PASS ', vout, re.M):
        problems.append('the Viv program has no passing test block')
    a = drop(pout, meta.get('py_drop'))
    b = drop(vout, meta.get('viv_drop'), keep_pass=keep)
    if a != b:
        import difflib
        d = difflib.unified_diff(a.split('\n'), b.split('\n'), 'python', 'viv', lineterm='', n=1)
        problems.append('outputs differ:\n' + '\n'.join(list(d)[:30]))

    gen = {
        'python': pout,
        'viv': vout,
        'py2viv': py2viv(base + '.py', pout),
    }
    svg = figure_svg(vout)
    if svg is not None:
        gen['figure_svg'] = os.path.basename(base) + '.viv.svg'
        if update:
            with open(base + '.viv.svg', 'w') as f:
                f.write(svg)
        elif not os.path.exists(base + '.viv.svg') or open(base + '.viv.svg').read() != svg:
            problems.append(f'{name}.viv.svg is out of date: run study/compare/check.sh --update')
    if os.path.exists(base + '.py.png'):
        gen['figure_png'] = os.path.basename(base) + '.py.png'
    gen_path = base + '.gen.json'
    text = json.dumps(gen, indent=1, ensure_ascii=False) + '\n'
    if update:
        with open(gen_path, 'w') as f:
            f.write(text)
    elif not os.path.exists(gen_path) or open(gen_path).read() != text:
        problems.append(f'{name}.gen.json is out of date: run study/compare/check.sh --update')
    return problems, meta


def main():
    args = sys.argv[1:]
    update = '--update' in args
    only = [a for a in args if not a.startswith('--')]
    topics = [t['dir'] for t in json.load(open(os.path.join(HERE, 'topics.json')))]
    for d in sorted(os.listdir(HERE)):
        if os.path.isdir(os.path.join(HERE, d)) and not d.startswith(('.', '_')) and d not in topics:
            print(f'{d}/ is not in topics.json')
            sys.exit(1)
    bad = 0
    count = 0
    for topic in topics:
        if only and topic not in only:
            continue
        names = sorted(f[:-3] for f in os.listdir(os.path.join(HERE, topic)) if f.endswith('.py'))
        for name in names:
            count += 1
            problems, meta = check_pair(topic, name, update)
            label = f'{topic}/{name}'
            if problems:
                bad += 1
                print(f'{label:34s} FAIL')
                for p in problems:
                    print('  ' + p.replace('\n', '\n  '))
            else:
                how = 'identical' if meta['output'] == 'identical' else 'identical after the stated drops'
                print(f'{label:34s} ok: {how}')
    print(f'{count - bad} of {count} pairs pass')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
