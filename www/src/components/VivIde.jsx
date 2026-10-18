// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
import { useEffect, useRef, useState, useCallback } from 'react';
import Editor, { loader } from '@monaco-editor/react';
import { parseOwnOutput, Chart, CallGraph, withModules } from './VivCell.jsx';
import VivTerm from './VivTerm.jsx';
import { vivMonarch, proseRules } from '../lib/vivMonarch.js';
import { attachFiles } from '../lib/programFiles.js';
import { runProgram } from '../lib/runProgram.js';
import Notebook from './Notebook.jsx';
import { writeDraft } from '../lib/atlasLink.js';
import { vivExports } from '../lib/vivWasm.js';

// Self-host Monaco from the bundled npm build (public/monaco/vs, copied from
// node_modules/monaco-editor/min/vs) instead of the jsdelivr CDN — the CDN
// load can hang behind strict networks / CSP, leaving the editor stuck on
// "Loading…". Same-origin path works offline and in prod (public/ ships as-is).
loader.config({ paths: { vs: '/monaco/vs' } });

// VivIde -- the workbench: a full-viewport editor for Viv.
//   ┌ toolbar: rail toggle, file picker ─────────── Notebook, Run ┐
//   │ rail: outline + files │ editor │ Output · Call graph · C · Terminal │
//   └─────────────────────────────────────────────────────────────┘
// The wasm module (viv_run / viv_compile) is the whole backend. Output is one
// stream: printed lines, narrate labels and figures in the order the program
// made them. Compiler diagnostics appear as live Monaco squiggles. Under 900px
// the editor comes first, the output under it, Run pinned to the bottom.

const EXAMPLES = [
  {
    name: 'Damped oscillation (plot + narrate)',
    src: `# A sine ringing inside a decaying envelope: sampled,
# plotted, and its first peak tested against the exact
# one. Press Run.
#| The decay that caps the oscillation.
#|| Pure exponential with rate 0.15.
#~ signal, demo
def envelope(t) {
    return exp(-0.15 * t)
}

#| The measured thing: a sine ringing inside the envelope.
#|| envelope(t) times sin(3t).
#||| Factoring envelope out keeps one definition of the decay rate.
#~ signal, demo
#: t  time in seconds
def signal(t) {
    return envelope(t) * sin(3.0 * t)
}

narrate "sampling the signal"
n = 200
t = linspace(0.0, 16.0, n)
y = zeros(n)
env = zeros(n)
for i to n {
    y[i] = signal(t[i])
    env[i] = envelope(t[i])
}

narrate "plotting"
figure("A damped oscillation", "time (s)", "amplitude")
plot(t, y, "signal")
plot(t, env, "envelope")

# The #| lines above a def say what it does (#|| how,
# #||| why, #~ tags, #: a variable); the outline and
# the call graph show them.
#| Largest element of an array.
#~ util
#: v  the samples to scan
#: p  best value seen so far
def peak_of(v) {
    p = v[0]
    for i to len(v) {
        #| strict > keeps the FIRST peak on ties
        if v[i] > p { p = v[i] }
    }
    return p
}
peak = peak_of(y)
show peak

# The first peak is where tan(3t) = 3/0.15, so t = atan(20)/3.
# Some sample lies within half a spacing h of it, where the
# signal is lower by at most |y''| (h/2)^2 / 2, and |y''| is
# about 3^2 there: so the sampled peak is within 9 h^2 / 8.
t_peak = atan(3.0 / 0.15) / 3.0
exact = signal(t_peak)
h = t[1] - t[0]
show exact
test "the sampled peak is within 9 h^2 / 8 of the exact one" {
    assert_near(peak, exact, 9.0 * h * h / 8.0)
}
`,
  },
  {
    name: 'Heatmap + 3D curve (plot2d / plot3d)',
    src: `# The three figure builtins in one program:
#   plot(x, y, "label")      a line chart
#   plot2d(Z, "label")       a matrix as a heatmap
#   plot3d(x, y, z, "label") a curve in 3-space - drag the figure to rotate it

#| A standing-wave interference field on a square grid.
#~ demo
#: Z  the field sin(3x) cos(3y) e^{-r^2/4}, sampled 40 x 40
def make_field(Z, n) {
    for i to n {
        for j to n {
            x = -3.0 + 6.0 * i / (n - 1.0)
            y = -3.0 + 6.0 * j / (n - 1.0)
            Z[i, j] = sin(3.0 * x) * cos(3.0 * y) * exp(0.0 - (x * x + y * y) / 4.0)
        }
    }
}

narrate "sampling the interference field"
n = 40
Z = zeros(n, n)
make_field(Z, n)
figure("An interference field", "x", "y")
plot2d(Z, "sin(3x) cos(3y) exp(-r^2/4)")

narrate "winding the helix"
m = 120
t = linspace(0.0, 12.566370614359172, m)
hx = zeros(m)
hy = zeros(m)
hz = zeros(m)
for i to m {
    r = 1.0 - 0.05 * t[i]
    hx[i] = r * cos(t[i])
    hy[i] = r * sin(t[i])
    hz[i] = 0.12 * t[i]
}
figure("A tightening helix", "x", "y")
plot3d(hx, hy, hz, "a tightening helix")

# and the classic, for comparison
figure("Height along the helix", "t", "z")
plot(t, hz, "height along the winding")
`,
  },
  {
    name: "Newton's method (the error at each step)",
    src: `# Root of f(x) = x^2 - 2. Each function has one job; the call graph
# shows newton_step leaning on f and df.
#| The function whose root we want: zero at sqrt(2).
#~ newton
def f(x)  { return x * x - 2.0 }

#| Its derivative, by hand.
#~ newton
def df(x) { return 2.0 * x }

#| One Newton update: slide down the tangent line to its zero.
#|| x - f(x)/f'(x): the whole method is this one line.
#||| Quadratic convergence - the error roughly squares every step.
#~ newton, core
def newton_step(x) {
    return x - f(x) / df(x)
}

narrate "iterating"
steps = 8
its = linspace(0.0, 7.0, steps)
err = zeros(steps)
x = 1.0
for k to steps {
    x = newton_step(x)
    err[k] = abs(x * x - 2.0)
}
figure("Newton's method on x^2 - 2", "step", "|x^2 - 2|")
plot(its, err, "|x^2 - 2| per step")
show x

test "converged to sqrt(2)" {
    assert_near(x, sqrt(2.0), 1.0e-12)
}
`,
  },
  {
    name: 'Fibonacci, recursive and iterative (call graph)',
    src: `# The same number two ways. Run, then open the Call graph tab:
# fib_rec's self-loop carries hundreds of calls; fib_loop makes none.
#| Fibonacci by textbook recursion.
#|| Two self-calls per level: the call tree doubles until the base case.
#||| Exponential work for a linear problem - the call graph shows the repeated calls.
#~ fibonacci, recursion
def fib_rec(n) {
    if n < 1.5 { return n }
    return fib_rec(n - 1.0) + fib_rec(n - 2.0)
}

#| The same numbers in linear time.
#|| Two running values, one pass - no calls at all.
#~ fibonacci
def fib_loop(n) {
    a = 0.0
    b = 1.0
    for i to n {
        t = a + b
        a = b
        b = t
    }
    return a
}

narrate "recursive"
by_recursion = fib_rec(15.0)
show by_recursion
narrate "iterative"
by_loop = fib_loop(15.0)
show by_loop
`,
  },
  {
    name: 'Logistic map: order vs chaos',
    src: `# One orbit function, two parameter values: r = 3.2 settles into a
# cycle, r = 3.9 never does.
#| One application of the logistic map.
#|| rx(1-x): growth times crowding.
#~ chaos
def step(r, x) {
    return r * x * (1.0 - x)
}

#| Iterate the map and record the trajectory.
#|| Writes into the caller's array - array parameters alias in Viv.
#~ chaos
def orbit(r, x0, out) {
    x = x0
    for i to len(out) {
        x = step(r, x)
        out[i] = x
    }
    return x
}

narrate "iterating both orbits"
n = 120
idx = linspace(0.0, 119.0, n)
periodic = zeros(n)
chaotic = zeros(n)
last1 = orbit(3.2, 0.2, periodic)
last2 = orbit(3.9, 0.2, chaotic)
figure("The logistic map: a cycle and chaos", "step", "x")
plot(idx, periodic, "r = 3.2")
plot(idx, chaotic, "r = 3.9")
`,
  },
  {
    name: 'Power method vs eig_sym',
    src: `# The dominant eigenvalue two ways: thirty matvecs, then the exact
# Jacobi answer. dot and normalize compose — see the call graph.
#| The dot product - Viv has no builtin, so it is three lines.
#~ linalg, util
def dot(a, b) {
    s = 0.0
    for i to len(a) {
        s += a[i] * b[i]
    }
    return s
}

#| Scale a vector to unit length.
#~ linalg
def normalize(v) {
    return v / sqrt(dot(v, v))
}

#| The Rayleigh quotient: the eigenvalue estimate a vector implies.
#|| v . Av for unit v - exact when v is an eigenvector.
#~ linalg, core
def rayleigh(A, v) {
    return dot(v, A @ v)
}

A = [[2.0, 1.0], [1.0, 3.0]]

narrate "power iteration"
v = [1.0, 0.0]
for i to 30 {
    v = normalize(A @ v)
}
estimate = rayleigh(A, v)
show estimate

narrate "exact, for comparison"
vals = zeros(2)
vecs = zeros(2, 2)
eig_sym(A, vals, vecs)
exact = vals[1]
show exact
`,
  },
  {
    name: 'Symbolic differentiation (expression tapes)',
    src: `# Symbolic computation with no pointers and no strings: an expression is
# rows in flat arrays, and node k's children are INDICES into those arrays.
# The call graph shows the structure: diff and eval_at call the
# constructors, which all call push.

# ops: 0 const, 1 var, 2 add, 4 mul, 6 exp
op = zeros(128)
ca = zeros(128)
cb = zeros(128)
cc = zeros(128)
nn = zeros(1)

#| Append one node to the expression tape; returns its index.
#|| Four parallel arrays hold op, two child indices, and a constant.
#||| Flat arrays instead of pointers: the whole CAS is index arithmetic.
#~ tape, core
def push(o, a, b, c) {
    i = nn[0]
    op[i] = o
    ca[i] = a
    cb[i] = b
    cc[i] = c
    nn[0] = i + 1
    return i
}
def cst(v) { return push(0, 0, 0, v) }
def vx() { return push(1, 0, 0, 0) }
def add(a, b) { return push(2, a, b, 0) }
def mul(a, b) { return push(4, a, b, 0) }
def expn(a) { return push(6, a, 0, 0) }

ev = zeros(128)
#| Evaluate the expression at x.
#|| One forward pass: children precede parents on the tape, so a single loop suffices.
#~ tape
def eval_at(root, x) {
    for k to nn[0] {
        o = op[k]
        if o == 0 { ev[k] = cc[k] }
        if o == 1 { ev[k] = x }
        if o == 2 { ev[k] = ev[ca[k]] + ev[cb[k]] }
        if o == 4 { ev[k] = ev[ca[k]] * ev[cb[k]] }
        if o == 6 { ev[k] = exp(ev[ca[k]]) }
    }
    return ev[root]
}

dd = zeros(128)
#| Differentiate symbolically: append derivative nodes for every node.
#|| Product and chain rules build new tape entries; dd[k] maps node to derivative.
#||| Run it twice for exact second derivatives; each run adds nodes to the tape.
#~ tape, core
def diff(root) {
    n = nn[0]
    for k to n {
        o = op[k]
        a = ca[k]
        b = cb[k]
        if o == 0 { dd[k] = cst(0) }
        if o == 1 { dd[k] = cst(1) }
        if o == 2 { dd[k] = add(dd[a], dd[b]) }
        if o == 4 { dd[k] = add(mul(dd[a], b), mul(a, dd[b])) }
        if o == 6 { dd[k] = mul(dd[a], k) }
    }
    return dd[root]
}

narrate "building g(x) = x^2 + exp(2x)"
x0 = vx()
g = add(mul(x0, x0), expn(mul(cst(2), x0)))

narrate "differentiating symbolically"
g1 = diff(g)

narrate "sampling both on a grid"
m = 60
grid = linspace(0.0 - 1.5, 1.45, m)
gs = zeros(m)
g1s = zeros(m)
for i to m {
    gs[i] = eval_at(g, grid[i])
    g1s[i] = eval_at(g1, grid[i])
}
figure("g and its symbolic derivative", "x", "value")
plot(grid, gs, "g")
plot(grid, g1s, "g'")

test "g'(0) = 2" {
    assert_near(eval_at(g1, 0.0), 2.0, 1.0e-12)
}
`,
  },
];

// ── Viv language for Monaco ────────────────────────────────────────────────
const VIV_BUILTINS = ['zeros', 'len', 'sqrt', 'exp', 'log', 'sin', 'cos', 'tan', 'abs', 'floor', 'ceil', 'pow',
  'min', 'max', 'eig_sym', 'transpose', 'simpson', 'trapezoid', 'deriv', 'assert_near', 'linspace', 'plot', 'plot2d', 'plot3d', 'complex', 're', 'im', 'conj', 'fill', 'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh', 'erf', 'gamma', 'round', 'mod'];

function defineSiteThemes(monaco) {
  // Token colors matched to the site's own code cells (the kh-* palette in
  // VivCell); the surfaces come from the page's theme tokens when they can be read.
  const dark = typeof document !== 'undefined' && document.documentElement.getAttribute('data-theme') === 'dark';
  const surf = siteSurfaces();
  monaco.editor.defineTheme('noc-dark', {
    base: 'vs-dark',
    inherit: true,
    rules: [
      { token: 'comment', foreground: '8b95a7', fontStyle: 'italic' },
      { token: 'comment.doc', foreground: 'a5b4fc', fontStyle: 'italic' },
      { token: 'string', foreground: 'd0a35c' },
      { token: 'keyword', foreground: 'e06c8c', fontStyle: 'bold' },
      { token: 'type.identifier', foreground: '6aa9e8' },
      { token: 'identifier', foreground: 'e2e8f0' },
      { token: 'number', foreground: 'a48fe8' },
      { token: 'number.float', foreground: 'a48fe8' },
      { token: 'operator', foreground: '7fb98e' },
      { token: 'delimiter', foreground: '94a3b8' },
      ...proseRules('e2e8f0', 'a8b3c4'),
    ],
    colors: {
      'editor.background': '#1e293b',
      'editor.foreground': '#e2e8f0',
      'editor.lineHighlightBackground': '#26334a',
      'editor.selectionBackground': '#3b4a6b',
      'editor.inactiveSelectionBackground': '#2c3a57',
      'editorLineNumber.foreground': '#475569',
      'editorLineNumber.activeForeground': '#94a3b8',
      'editorCursor.foreground': '#e2e8f0',
      'editorIndentGuide.background': '#334155',
      'editorWhitespace.foreground': '#334155',
      'editorWidget.background': '#111c33',
      'editorHoverWidget.background': '#111c33',
      'editorHoverWidget.border': '#334155',
      'editorSuggestWidget.background': '#111c33',
      'editorSuggestWidget.border': '#334155',
      'scrollbarSlider.background': '#33415580',
      'scrollbarSlider.hoverBackground': '#47556980',
      ...(dark ? surf : {}),
    },
  });
  monaco.editor.defineTheme('noc-light', {
    base: 'vs',
    inherit: true,
    rules: [
      { token: 'comment', foreground: '7a8494', fontStyle: 'italic' },
      { token: 'comment.doc', foreground: '4338ca', fontStyle: 'italic' },
      { token: 'string', foreground: 'b0813d' },
      { token: 'keyword', foreground: 'c04d6e', fontStyle: 'bold' },
      { token: 'type.identifier', foreground: '2f7fd1' },
      { token: 'identifier', foreground: '1f2937' },
      { token: 'number', foreground: '7a5fd0' },
      { token: 'number.float', foreground: '7a5fd0' },
      { token: 'operator', foreground: '5c8a68' },
      { token: 'delimiter', foreground: '64748b' },
      ...proseRules('1f2937', '4b5563'),
    ],
    colors: {
      'editor.background': '#ffffff',
      'editor.foreground': '#1f2937',
      'editor.lineHighlightBackground': '#f1f5f9',
      'editor.selectionBackground': '#c7d2fe',
      'editorLineNumber.foreground': '#94a3b8',
      'editorLineNumber.activeForeground': '#475569',
      'editorCursor.foreground': '#1f2937',
      'editorIndentGuide.background': '#e5e7eb',
      'editorWidget.background': '#f1f5f9',
      'editorHoverWidget.background': '#f8fafc',
      'editorHoverWidget.border': '#e5e7eb',
      ...(dark ? {} : surf),
    },
  });
}

// The editor's surfaces take the page's own colors (the theme tokens), so the
// editor sits on the page in both themes instead of on a fixed slate.
function tokenHex(name) {
  if (typeof document === 'undefined') return null;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return /^#[0-9a-f]{6}$/i.test(v) ? v : null;
}
function siteSurfaces() {
  const bg = tokenHex('--bg'), soft = tokenHex('--bg-soft'), card = tokenHex('--bg-card'), border = tokenHex('--border');
  const muted = tokenHex('--text-muted'), text = tokenHex('--text');
  if (!bg || !soft || !card || !border) return {};
  return {
    // line numbers in the page's muted text color: the slate grays were under 3:1 on these surfaces
    ...(muted ? { 'editorLineNumber.foreground': muted } : {}),
    ...(text ? { 'editorLineNumber.activeForeground': text } : {}),
    'editor.background': bg,
    'editor.lineHighlightBackground': soft,
    'editorGutter.background': bg,
    'editorWidget.background': card,
    'editorHoverWidget.background': card,
    'editorHoverWidget.border': border,
    'editorSuggestWidget.background': card,
    'editorSuggestWidget.border': border,
    'editorIndentGuide.background': border,
  };
}

function registerViv(monaco) {
  defineSiteThemes(monaco);
  if (monaco.languages.getLanguages().some((l) => l.id === 'viv')) return;
  monaco.languages.register({ id: 'viv' });
  monaco.languages.setMonarchTokensProvider('viv', vivMonarch(VIV_BUILTINS));
  monaco.languages.setLanguageConfiguration('viv', {
    comments: { lineComment: '#' },
    brackets: [['{', '}'], ['[', ']'], ['(', ')']],
    autoClosingPairs: [
      { open: '{', close: '}' }, { open: '[', close: ']' }, { open: '(', close: ')' },
    ],
  });
}

// ── compiler diagnostics → Monaco markers ───────────────────────────────────
function diagnosticsToMarkers(text, monaco) {
  const markers = [];
  const lines = text.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^error:\s*(.*)$/);
    if (!m) continue;
    let message = m[1];
    let line = 1, col = 1, len = 1;
    for (let j = i + 1; j < Math.min(i + 6, lines.length); j++) {
      const loc = lines[j].match(/-->\s*.*?:(\d+):(\d+)/);
      if (loc) { line = +loc[1]; col = +loc[2]; }
      const caret = lines[j].match(/(\^+)\s*(.*)$/);
      if (caret) {
        len = caret[1].length;
        if (caret[2]) message += ` — ${caret[2]}`;
        break;
      }
    }
    markers.push({
      severity: monaco.MarkerSeverity.Error,
      message,
      startLineNumber: line, startColumn: col,
      endLineNumber: line, endColumn: col + len,
    });
  }
  return markers;
}

// ── run output → one stream ─────────────────────────────────────────────────
// VivCell's parser gives the output in the order it was made: printed text,
// narrate labels and charts. One IDE extra on top: for the older physics decks
// that predate plot(), a shown array with 2+ points (`a: [v, v, ...]`) is also
// drawn, right after the line that printed it.
function parseRun(text) {
  const { flow, graph, trace } = parseOwnOutput(text);
  const items = [];
  for (const b of flow) {
    if (b.type !== 'text') {
      items.push(b);
      continue;
    }
    let buf = [];
    for (const line of b.s.split('\n')) {
      buf.push(line);
      const arr = line.match(/^([a-zA-Z_]\w*): \[([^\[\]]*)\]\s*$/);
      if (!arr) continue;
      const values = arr[2].split(',').map(Number).filter((v) => Number.isFinite(v));
      if (values.length < 2) continue;
      items.push({ type: 'text', s: buf.join('\n') });
      buf = [];
      items.push({ type: 'chart', chart: { key: 'shown-' + arr[1], x: { values: values.map((_, k) => k) }, series: [{ name: arr[1], values }] } });
    }
    if (buf.length) items.push({ type: 'text', s: buf.join('\n') });
  }
  return { items, graph, trace };
}

// A text item split into runs of plain lines and error lines: an `error:` line
// and the location and caret lines under it, or a failed test.
function errorRuns(s) {
  const runs = [];
  let inErr = false;
  for (const line of s.split('\n')) {
    if (/^error\b/.test(line)) inErr = true;
    else if (inErr && !/^\s*(-->|\||\d+\s*\||\^|=)/.test(line) && line.trim() !== '') inErr = false;
    const err = inErr || /^FAIL\b/.test(line);
    const last = runs[runs.length - 1];
    if (last && last.err === err) last.s += '\n' + line;
    else runs.push({ err, s: line });
  }
  return runs;
}

// Where a def starts in source (its `def` line), for jumping the editor there.
function defLine(text, name) {
  const re = new RegExp('^\\s*def\\s+' + name + '\\s*\\(');
  const lines = text.split('\n');
  for (let i = 0; i < lines.length; i++) if (re.test(lines[i])) return i + 1;
  return 0;
}

// file extension → Monaco language, for files opened from the tree
const EXT_LANG = {
  viv: 'viv', py: 'python', c: 'c', h: 'c', cpp: 'cpp', cc: 'cpp', hpp: 'cpp',
  md: 'markdown', json: 'json', toml: 'ini', sh: 'shell', ts: 'typescript',
  js: 'javascript', mjs: 'javascript', txt: 'plaintext', yml: 'yaml', yaml: 'yaml',
};
const langOfFile = (name) => EXT_LANG[name.split('.').pop()?.toLowerCase()] || 'plaintext';

const GCC = 'gcc -O2 -o program program.c -lm && ./program';
const readPref = (k) => { try { return localStorage.getItem(k); } catch { return null; } };
const writePref = (k, v) => { try { localStorage.setItem(k, v); } catch { /* private window */ } };

// ── the file picker: the small examples and every project file, one menu ────
function FilePicker({ label, entries, onPick }) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState('');
  const [hi, setHi] = useState(0);
  const boxRef = useRef(null);
  const inputRef = useRef(null);
  const listRef = useRef(null);
  const query = q.trim().toLowerCase();
  const shown = query
    ? entries.filter((e) => e.label.toLowerCase().includes(query) || (e.sub || '').toLowerCase().includes(query) || e.group.toLowerCase().includes(query))
    : entries;
  useEffect(() => {
    if (!open) return;
    const away = (e) => { if (boxRef.current && !boxRef.current.contains(e.target)) setOpen(false); };
    document.addEventListener('pointerdown', away);
    inputRef.current?.focus();
    return () => document.removeEventListener('pointerdown', away);
  }, [open]);
  useEffect(() => { setHi(0); }, [q]);
  useEffect(() => {
    listRef.current?.querySelector('.ki-pick-item.hi')?.scrollIntoView({ block: 'nearest' });
  }, [hi]);
  const pick = (e) => {
    setOpen(false);
    setQ('');
    onPick(e);
  };
  const onKey = (e) => {
    if (e.key === 'Escape') setOpen(false);
    else if (e.key === 'ArrowDown') { e.preventDefault(); setHi((h) => Math.min(shown.length - 1, h + 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setHi((h) => Math.max(0, h - 1)); }
    else if (e.key === 'Enter' && shown[hi]) { e.preventDefault(); pick(shown[hi]); }
  };
  let lastGroup = null;
  return (
    <div className="ki-pick" ref={boxRef}>
      <button type="button" className="ki-pick-btn" aria-haspopup="listbox" aria-expanded={open}
        onClick={() => setOpen((v) => !v)} title="Open an example or a file">
        <span className="ki-pick-name">{label}</span>
        <svg className="ki-pick-caret" width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
          <path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {open && (
        <div className="ki-pick-menu" role="dialog" aria-label="Open a file">
          <input ref={inputRef} className="ki-pick-q" placeholder="Search examples and files" value={q}
            onChange={(e) => setQ(e.target.value)} onKeyDown={onKey} spellCheck="false" />
          <div className="ki-pick-list" role="listbox" ref={listRef}>
            {shown.length === 0 && <div className="ki-pick-none">Nothing matches</div>}
            {shown.map((e, k) => {
              const head = e.group !== lastGroup ? <div className="ki-pick-group">{e.group}</div> : null;
              lastGroup = e.group;
              return (
                <div key={e.id}>
                  {head}
                  <button type="button" role="option" aria-selected={k === hi}
                    className={`ki-pick-item${k === hi ? ' hi' : ''}`}
                    onMouseEnter={() => setHi(k)} onClick={() => pick(e)}>
                    <span className="ki-pick-label">{e.label}</span>
                    {e.sub && <span className="ki-pick-sub">{e.sub}</span>}
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export default function VivIde({ featured = {}, projects = [] }) {
  const exportsRef = useRef(null);
  const editorRef = useRef(null);
  const monacoRef = useRef(null);
  const debounceRef = useRef(0);
  const outRef = useRef(null);
  const [status, setStatus] = useState('loading');
  const [errMsg, setErrMsg] = useState('');
  const [tab, setTab] = useState('out');
  // the editor's share of the editor + output width, persisted
  const [hSplit, setHSplit] = useState(() => {
    const v = parseFloat(readPref('viv-ide-hsplit'));
    return v >= 0.25 && v <= 0.8 ? v : 0.58;
  });
  const [wide, setWide] = useState(true);
  const [railOpen, setRailOpen] = useState(() => readPref('viv-ide-rail') !== '0');
  const [drawer, setDrawer] = useState(false); // the rail, on a phone
  useEffect(() => {
    const mq = window.matchMedia('(min-width: 901px)');
    const upd = () => setWide(mq.matches);
    upd();
    mq.addEventListener('change', upd);
    return () => mq.removeEventListener('change', upd);
  }, []);
  // iOS opens its keyboard only from a tap on a control; Monaco's own button
  // for that floats over the code, so the toolbar carries it instead (and the
  // floating one is hidden, workbench styles)
  const [ios, setIos] = useState(false);
  useEffect(() => {
    setIos(/iP(hone|ad|od)/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1));
  }, []);
  const isMac = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
  const runKey = isMac ? '⌘↵' : 'Ctrl+↵';
  const dragRef = useRef(null);
  const startDrag = useCallback((e) => {
    e.preventDefault();
    const main = e.currentTarget.parentElement;
    dragRef.current = { rect: main.getBoundingClientRect() };
    const move = (ev) => {
      const d = dragRef.current;
      if (!d) return;
      const railW = main.querySelector('.ki-rail')?.getBoundingClientRect().width || 0;
      const x = (ev.clientX - d.rect.left - railW) / (d.rect.width - railW);
      const f = Math.min(0.8, Math.max(0.25, x));
      setHSplit(f);
      writePref('viv-ide-hsplit', String(f));
    };
    const up = () => {
      dragRef.current = null;
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      document.body.style.cursor = '';
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    document.body.style.cursor = 'col-resize';
  }, []);
  const [termMounted, setTermMounted] = useState(false);
  // the terminal sees every project file plus the editor buffer as `main.viv`
  const termFilesRef = useRef(null);
  if (termFilesRef.current === null) {
    termFilesRef.current = {
      files: projects.flatMap((p) => p.files.map((f) => ({ name: f, url: `/programs/${p.slug}/${f}` }))),
      live: { 'main.viv': () => editorRef.current?.getValue() ?? '' },
    };
  }
  const [out, setOut] = useState('');
  const [items, setItems] = useState(null); // the output stream, null before the first run
  const [graph, setGraph] = useState(null);
  const [trace, setTrace] = useState([]);
  const [deps, setDeps] = useState(null); // static: {functions:[{name,params,calls}], main_calls}
  const [docInfo, setDocInfo] = useState(null); // viv_doc: {code, functions:[{name,line,end,what,how,why,tags,vars}]}
  const [selFn, setSelFn] = useState(null);
  const [fnFilter, setFnFilter] = useState('');
  const [cSrc, setCSrc] = useState('');
  const [stats, setStats] = useState('');
  const [editorTheme, setEditorTheme] = useState('noc-dark');
  // what the editor holds: a small example (path null) or a project file
  const [current, setCurrent] = useState({ label: EXAMPLES[0].name, path: null });
  // the text the buffer was opened with: the picker's name gains ", edited" once it differs
  const openedRef = useRef(EXAMPLES[0].src);
  const [edited, setEdited] = useState(false);
  // the open file's staged path without `.viv`, so its imports of modules beside it resolve
  const currentPathRef = useRef(null);
  currentPathRef.current = current.path;
  const basePath = () => (currentPathRef.current ? currentPathRef.current.replace(/\.viv$/, '') : null);
  const [openGroups, setOpenGroups] = useState({});
  // The Notebook mode: the editor's text read as a notebook, cells at its
  // `###` prose fences, each cell run in a worker. Edits made there come back
  // to the editor when it closes.
  const [nb, setNb] = useState(null); // { src, key } while the notebook view is open
  const nbText = useRef(null);
  const toggleNotebook = useCallback(() => {
    const ed = editorRef.current;
    if (!ed) return;
    if (nb) {
      if (nbText.current !== null && nbText.current !== ed.getValue()) ed.setValue(nbText.current);
      nbText.current = null;
      setNb(null);
    } else {
      nbText.current = null;
      setNb({ src: ed.getValue(), key: Date.now() });
    }
  }, [nb]);

  useEffect(() => {
    const update = () => {
      const t = document.documentElement.getAttribute('data-theme') || 'light';
      if (monacoRef.current) defineSiteThemes(monacoRef.current);
      setEditorTheme(t === 'dark' ? 'noc-dark' : 'noc-light');
      monacoRef.current?.editor.setTheme(t === 'dark' ? 'noc-dark' : 'noc-light');
    };
    update();
    const obs = new MutationObserver(update);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const ex = await vivExports();
        if (cancelled) return;
        exportsRef.current = ex;
        setStatus('ready');
      } catch (e) {
        if (cancelled) return;
        setErrMsg(String(e));
        setStatus('error');
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const call = useCallback((fn, text) => {
    const ex = exportsRef.current;
    const enc = new TextEncoder().encode(text);
    const ptr = ex.viv_alloc(enc.length);
    new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
    const packed = fn(ptr, enc.length);
    return new TextDecoder().decode(
      new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
  }, []);

  const lint = useCallback(async () => {
    const ed = editorRef.current, mo = monacoRef.current, ex = exportsRef.current;
    if (!ed || !mo || !ex) return;
    // Only Viv buffers get compiled; a Python or C file opened from the tree
    // should not produce nonsense markers.
    if (ed.getModel()?.getLanguageId() !== 'viv') {
      mo.editor.setModelMarkers(ed.getModel(), 'viv', []);
      setStats('');
      setCSrc('');
      setDeps(null);
      setDocInfo(null);
      return;
    }
    const written = ed.getValue();
    let src = written;
    try { src = await withModules(written, basePath()); } catch (e) {
      mo.editor.setModelMarkers(ed.getModel(), 'viv', []);
      setStats(`error: ${e.message}`);
      return;
    }
    if (ed.getValue() !== written) return; // it changed while the module was fetched
    const t0 = performance.now();
    const res = call(ex.viv_compile, src);
    const ms = performance.now() - t0;
    if (res.includes('#include')) {
      mo.editor.setModelMarkers(ed.getModel(), 'viv', []);
      setCSrc(res);
      setStats(`${src.split('\n').length} lines of Viv${src !== written ? ' with its modules' : ''}, ${res.split('\n').length} lines of C, compiled in ${ms.toFixed(1)} ms`);
      // static dependency graph (language-level, no execution needed)
      try {
        const dj = call(ex.viv_deps, src);
        setDeps(dj.startsWith('{') ? JSON.parse(dj) : null);
      } catch { setDeps(null); }
      try {
        const doc = call(ex.viv_doc, src);
        setDocInfo(doc.startsWith('{') ? JSON.parse(doc) : null);
      } catch { setDocInfo(null); }
    } else {
      mo.editor.setModelMarkers(ed.getModel(), 'viv', diagnosticsToMarkers(res, mo));
      setStats('');
      setCSrc('');
      // the outline still lists the defs while the program does not compile
      const names = [...written.matchAll(/^\s*def\s+([A-Za-z_]\w*)\s*\(/gm)].map((m) => m[1]);
      setDeps({ functions: names.map((name) => ({ name, params: 0, calls: [] })), main_calls: [], partial: true });
      setDocInfo(null);
    }
  }, [call]);

  const scheduleLint = useCallback(() => {
    window.clearTimeout(debounceRef.current);
    debounceRef.current = window.setTimeout(lint, 350);
  }, [lint]);

  useEffect(() => { if (status === 'ready') scheduleLint(); }, [status, scheduleLint]);

  const showText = useCallback((text) => {
    setTab('out');
    setOut(text);
    setItems([{ type: 'text', s: text }]);
  }, []);

  const run = useCallback(async () => {
    const ed = editorRef.current;
    if (status !== 'ready' || !ed) return;
    if (ed.getModel()?.getLanguageId() !== 'viv') {
      showText('This file is not Viv, so it does not run here.');
      return;
    }
    // an imported module is spliced in before the program runs, exactly as it is
    // for a post's card and for the compiler: a file opened from the feed runs
    // here as it runs there.
    let source;
    try {
      source = await withModules(ed.getValue(), basePath());
      await attachFiles(exportsRef.current, source);
    } catch (e) {
      showText(`error: ${e.message}`);
      return;
    }
    let text = await runProgram(exportsRef.current, source);
    // a run the compiler refuses: its error as the compiler gives it, with the
    // line, the column and the caret (the interpreter's says only what)
    const at = text.search(/(^|\n)error: /);
    if (at >= 0 && !text.includes('-->')) {
      const c = call(exportsRef.current.viv_compile, source);
      if (c.startsWith('error') && c.includes('-->')) text = text.slice(0, at + (text[at] === '\n' ? 1 : 0)) + c;
    }
    const { items: it, graph: g, trace: tr } = parseRun(text);
    setTab('out');
    setOut(text);
    setItems(it);
    setGraph(g);
    setTrace(tr);
    // on a phone the output is below the editor: bring it into view
    if (!window.matchMedia('(min-width: 901px)').matches) {
      requestAnimationFrame(() => outRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }));
    }
  }, [call, status, showText]);

  const runRef = useRef(run);
  useEffect(() => { runRef.current = run; }, [run]);

  // ⌘↵ from anywhere on the page, not only inside the editor
  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== 'Enter' || !(e.metaKey || e.ctrlKey) || nb) return;
      if (e.target?.closest?.('.ki-editor, .vt')) return; // the editor has its own binding
      e.preventDefault();
      runRef.current();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [nb]);

  const showC = useCallback(() => { lint(); setTab('c'); }, [lint]);

  const downloadC = useCallback(async () => {
    const ed = editorRef.current;
    if (status !== 'ready' || !ed) return;
    const c = call(exportsRef.current.viv_compile, await withModules(ed.getValue(), basePath()).catch(() => ed.getValue()));
    if (!c.includes('#include')) { showText(c); return; }
    const blob = new Blob([c], { type: 'text/x-c' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'program.c';
    a.click();
    URL.revokeObjectURL(a.href);
  }, [call, status, showText]);

  function onMount(editor, monaco) {
    editorRef.current = editor;
    monacoRef.current = monaco;
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => runRef.current());
    if (status === 'ready') scheduleLint();
    // deep link: /workbench?open=<slug>/<relPath> opens a hosted file on arrival
    // (how ladders, posts and tutorial pages hand a program to the workbench)
    const want = new URLSearchParams(window.location.search).get('open');
    if (want) {
      const slash = want.indexOf('/');
      if (slash > 0) openFile(want.slice(0, slash), want.slice(slash + 1));
    }
  }

  function setBuffer(text, lang, cur) {
    const ed = editorRef.current, mo = monacoRef.current;
    if (!ed || !mo) return;
    openedRef.current = text;
    ed.setValue(text);
    mo.editor.setModelLanguage(ed.getModel(), lang);
    setCurrent(cur);
    setEdited(false);
    currentPathRef.current = cur.path;
    setOut('');
    setItems(null);
    setGraph(null);
    setTrace([]);
    setSelFn(null);
    setTab('out');
    if (cur.path) {
      const slug = cur.path.slice(0, cur.path.indexOf('/'));
      setOpenGroups((g) => ({ ...g, [slug]: true }));
    }
    scheduleLint();
  }

  // Open a hosted file (contents come from the static /programs tree, so
  // nothing is inlined into the page).
  async function openFile(slug, rel) {
    try {
      const res = await fetch(`/programs/${slug}/${rel}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setBuffer(await res.text(), langOfFile(rel), { label: `${slug}/${rel}`, path: `${slug}/${rel}` });
    } catch (e) {
      showText(`error: could not open ${slug}/${rel} (${e.message})`);
    }
    setDrawer(false);
  }

  // the atlas of the editor's text (/atlas?draft, lib/atlas.js): the text is
  // left in this browser for the new page, named by its file so the modules
  // beside it resolve
  const openAtlas = () => {
    const ed = editorRef.current;
    if (!ed) return;
    const name = current.path ? current.path.replace(/\.viv$/, '') : 'scratch';
    if (!writeDraft(name, ed.getValue())) {
      showText('error: this browser does not keep data for the site, so the atlas page cannot get the program');
      return;
    }
    window.open('/atlas?draft', '_blank', 'noopener');
  };

  const pickerEntries = [
    ...EXAMPLES.map((e) => ({ id: 'ex:' + e.name, group: 'Small examples', label: e.name, example: e })),
    ...projects.flatMap((p) => {
      const files = p.slug === 'examples'
        ? [...p.files.filter((f) => featured[f]), ...p.files.filter((f) => !featured[f])]
        : p.files;
      return files.map((f) => ({
        id: `${p.slug}/${f}`, group: p.title, label: `${p.slug}/${f}`, sub: p.slug === 'examples' ? featured[f] : undefined, slug: p.slug, file: f,
      }));
    }),
  ];
  const onPick = (e) => {
    if (e.example) setBuffer(e.example.src, 'viv', { label: e.example.name, path: null });
    else openFile(e.slug, e.file);
  };

  const toggleRail = () => {
    if (!wide) { setDrawer((v) => !v); return; }
    setRailOpen((v) => {
      writePref('viv-ide-rail', v ? '0' : '1');
      return !v;
    });
  };

  const revealDef = (name) => {
    const ed = editorRef.current;
    if (!ed) return;
    const line = defLine(ed.getValue(), name);
    if (!line) return;
    ed.revealLineInCenter(line);
    ed.setPosition({ lineNumber: line, column: 1 });
    if (wide) ed.focus();
  };
  const selectDef = (name) => {
    setSelFn(name);
    revealDef(name);
  };

  const opts = {
    fontSize: 13,
    minimap: { enabled: false },
    scrollBeyondLastLine: false,
    tabSize: 4,
    insertSpaces: true,
    automaticLayout: true,
    padding: { top: 10 },
    renderLineHighlight: 'line',
    fixedOverflowWidgets: true,
    // a long line wraps instead of disappearing off the right edge; the
    // continuation is indented so the structure still reads
    wordWrap: 'on',
    wrappingIndent: 'indent',
    smoothScrolling: true,
    scrollbar: { verticalScrollbarSize: 10, horizontalScrollbarSize: 10 },
  };

  const fns = deps?.functions || [];
  const docOf = (name) => docInfo?.functions?.find((x) => x.name === name);
  const q = fnFilter.trim().toLowerCase();
  const shownFns = !q ? fns : fns.filter((f) => {
    const d = docOf(f.name);
    return f.name.toLowerCase().includes(q)
      || (d && ((d.what || '').toLowerCase().includes(q) || (d.tags || []).some((t) => t.toLowerCase().includes(q))));
  });

  const defCard = (name) => {
    const d = docOf(name);
    const me = fns.find((f) => f.name === name);
    const callers = fns.filter((f) => f.calls.includes(name)).map((f) => f.name);
    const fromMain = deps?.main_calls?.includes(name);
    return (
      <div className="ki-def">
        {d?.what && <p className="ki-def-what">{d.what}</p>}
        {d?.how && <p className="ki-def-more">{d.how}</p>}
        {d?.why && <p className="ki-def-more">{d.why}</p>}
        {d?.vars?.length > 0 && (
          <dl className="ki-def-vars">
            {d.vars.map((v) => (
              <div key={v.name}><dt>{v.name}</dt><dd>{v.desc}</dd></div>
            ))}
          </dl>
        )}
        {!deps?.partial && <div className="ki-def-row">
          <span className="ki-def-lab">Calls</span>
          {me && me.calls.length
            ? me.calls.map((c) => <button key={c} type="button" className="ki-chip" onClick={() => selectDef(c)}>{c}</button>)
            : <span className="ki-def-none">nothing</span>}
        </div>}
        {!deps?.partial && <div className="ki-def-row">
          <span className="ki-def-lab">Called by</span>
          {callers.map((c) => <button key={c} type="button" className="ki-chip" onClick={() => selectDef(c)}>{c}</button>)}
          {fromMain && <span className="ki-chip ki-chip-main">top level</span>}
          {!callers.length && !fromMain && <span className="ki-def-none">nothing</span>}
        </div>}
        {deps?.partial && <p className="ki-def-more">Calls show once the program compiles</p>}
        {d?.tags?.length > 0 && (
          <div className="ki-def-row">
            {d.tags.map((t) => <span key={t} className="ki-tag">{t}</span>)}
          </div>
        )}
      </div>
    );
  };

  const rail = (
    <aside className={`ki-rail${!wide ? ' ki-drawer' : ''}`} aria-label="Outline and files">
      {!wide && (
        <div className="ki-drawer-bar">
          <span>Outline and files</span>
          <button type="button" className="ki-btn" onClick={() => setDrawer(false)}>Close</button>
        </div>
      )}
      <div className="ki-rail-head">Outline</div>
      {fns.length > 0 ? (
        <>
          <input className="ki-filter" placeholder="Filter" value={fnFilter} aria-label="Filter the outline"
            onChange={(e) => setFnFilter(e.target.value)} spellCheck="false" />
          {shownFns.map((f) => (
            <div key={f.name}>
              <button type="button" className={`ki-row ki-fn${selFn === f.name ? ' on' : ''}`}
                title={docOf(f.name)?.what || undefined}
                onClick={() => (selFn === f.name ? setSelFn(null) : selectDef(f.name))}>
                {f.name}
              </button>
              {selFn === f.name && defCard(f.name)}
            </div>
          ))}
        </>
      ) : <div className="ki-rail-none">No defs</div>}
      <div className="ki-rail-head">Files</div>
      {projects.map((p) => (
        <div key={p.slug} className="ki-group">
          <button type="button" className="ki-row ki-group-head" aria-expanded={!!openGroups[p.slug]} title={p.title}
            onClick={() => setOpenGroups((g) => ({ ...g, [p.slug]: !g[p.slug] }))}>
            <span>{p.slug}</span><span className="ki-count">{p.files.length}</span>
          </button>
          {openGroups[p.slug] && p.files.map((f) => (
            <button key={f} type="button"
              className={`ki-row ki-file${current.path === `${p.slug}/${f}` ? ' on' : ''}`}
              onClick={() => openFile(p.slug, f)}>{f}</button>
          ))}
        </div>
      ))}
    </aside>
  );

  const showRail = wide ? railOpen : drawer;
  const outputView = (
    <div className="ki-out" ref={outRef}>
      {status === 'error' && <pre className="ki-out-text ki-err">{`error: the compiler did not load (${errMsg})`}</pre>}
      {items === null
        ? status !== 'error' && <div className="ki-out-empty">{wide ? `Press Run (${runKey})` : 'Press Run'}</div>
        : items.length === 0
          ? <div className="ki-out-empty">The program printed nothing</div>
          : items.map((b, i) => {
            if (b.type === 'narrate') return <div key={i} className="ki-out-label">{b.s}</div>;
            if (b.type === 'chart') return <div key={i} className="ki-figcard"><Chart chart={b.chart} /></div>;
            return errorRuns(b.s).map((r, k) => (
              <pre key={`${i}-${k}`} className={`ki-out-text${r.err ? ' ki-err' : ''}`}>{r.s}</pre>
            ));
          })}
    </div>
  );

  return (
    <div className={`ki${wide ? '' : ' ki-narrow'}`}>
      <div className="ki-bar">
        <button type="button" className="ki-btn ki-railbtn" aria-pressed={showRail}
          title={wide ? 'Show or hide the outline and files' : 'Outline and files'}
          onClick={toggleRail}>
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
            <rect x="1.5" y="2.5" width="13" height="11" rx="1.5" fill="none" stroke="currentColor" strokeWidth="1.3" />
            <path d="M6 2.5v11" stroke="currentColor" strokeWidth="1.3" />
          </svg>
        </button>
        <FilePicker label={edited ? `${current.label}, edited` : current.label} entries={pickerEntries} onPick={onPick} />
        <span className="ki-spacer" />
        {ios && !nb && (
          <button type="button" className="ki-btn ki-kbbtn" onClick={() => editorRef.current?.focus()}
            aria-label="Show the keyboard" title="Show the keyboard">
            <svg width="18" height="14" viewBox="0 0 18 14" aria-hidden="true">
              <rect x="0.75" y="0.75" width="16.5" height="12.5" rx="2" fill="none" stroke="currentColor" strokeWidth="1.3" />
              <path d="M4 5h1M7 5h1M10 5h1M13 5h1M4 8h1M13 8h1M7 8h4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
            </svg>
          </button>
        )}
        <button type="button" className={`ki-btn ki-nbtoggle${nb ? ' on' : ''}`} onClick={toggleNotebook} aria-pressed={!!nb}
          title="Read this file as a notebook, prose between the cells and each cell's output under it">Notebook</button>
        {!nb && (
          <button type="button" className="ki-btn ki-atlasbtn" onClick={openAtlas}
            title="The atlas of the program in the editor: its defs and the calls between them, on a page of its own">Atlas</button>
        )}
        {!nb && (
          <button type="button" className="ki-go" onClick={run} disabled={status !== 'ready'}>
            Run <span className="ki-kbd">{runKey}</span>
          </button>
        )}
      </div>

      {nb && (
        <div className="ki-nbpane">
          <Notebook key={nb.key} src={nb.src} name={current.path || 'scratch.viv'} onSource={(t) => { nbText.current = t; }} />
        </div>
      )}

      <div
        className={`ki-main${showRail && wide ? ' with-rail' : ''}${nb ? ' ki-hidden' : ''}`}
        style={!wide ? undefined : {
          gridTemplateColumns: showRail
            ? `236px minmax(0, ${hSplit}fr) 6px minmax(0, ${1 - hSplit}fr)`
            : `minmax(0, ${hSplit}fr) 6px minmax(0, ${1 - hSplit}fr)`,
        }}>
        {showRail && rail}
        {!wide && drawer && <div className="ki-scrim" onClick={() => setDrawer(false)} />}
        <div className="ki-editor">
          <Editor
            height="100%"
            defaultLanguage="viv"
            defaultValue={EXAMPLES[0].src}
            theme={editorTheme}
            options={opts}
            beforeMount={registerViv}
            onMount={onMount}
            onChange={(v) => { setEdited(v !== openedRef.current); scheduleLint(); }}
          />
        </div>

        {wide && <div className="ki-hgrip" onPointerDown={startDrag} role="separator" aria-orientation="vertical" aria-label="Resize the editor" />}

        <div className="ki-right">
          <div className="ki-tabs" role="tablist">
            <button type="button" role="tab" aria-selected={tab === 'out'} className={`ki-tab${tab === 'out' ? ' on' : ''}`} onClick={() => setTab('out')}>Output</button>
            <button type="button" role="tab" aria-selected={tab === 'graph'} className={`ki-tab${tab === 'graph' ? ' on' : ''}`} onClick={() => setTab('graph')}>Call graph</button>
            <button type="button" role="tab" aria-selected={tab === 'c'} className={`ki-tab${tab === 'c' ? ' on' : ''}`} onClick={showC}>C</button>
            <button type="button" role="tab" aria-selected={tab === 'term'} className={`ki-tab${tab === 'term' ? ' on' : ''}`} onClick={() => { setTermMounted(true); setTab('term'); }}>Terminal</button>
          </div>
          {tab === 'out' && outputView}
          {tab === 'graph' && (
            <div className="ki-graphpane">
              {graph
                ? <CallGraph key={out} graph={graph} trace={trace} defaultOpen={true} />
                : <div className="ki-out-empty">Run the program to draw its call graph</div>}
            </div>
          )}
          {tab === 'c' && (
            <div className="ki-cpane">
              <div className="ki-cbar">
                <span className="ki-cstats">{stats || 'The C appears once the program compiles'}</span>
                <code className="ki-gcc">{GCC}</code>
                <button type="button" className="ki-btn ki-dl" onClick={downloadC} disabled={status !== 'ready' || !cSrc}>Download .c</button>
              </div>
              <div className="ki-csrc">
                <Editor height="100%" language="c" value={cSrc}
                  theme={editorTheme} beforeMount={registerViv} options={{ ...opts, readOnly: true }} />
              </div>
            </div>
          )}
          {termMounted && (
            <div className="ki-termpane" hidden={tab !== 'term'}>
              <VivTerm fill files={termFilesRef.current.files} liveFiles={termFilesRef.current.live} />
            </div>
          )}
        </div>
      </div>

      <style>{css}</style>
    </div>
  );
}

const css = `
.ki { display: flex; flex-direction: column; height: 100%; min-height: 480px; background: var(--bg); }
.ki-bar { display: flex; align-items: center; gap: 0.5rem; padding: 0.45rem 0.9rem;
  background: var(--bg-soft); border-bottom: 1px solid var(--border); min-width: 0; }
.ki-spacer { flex: 1; }
.ki-btn { font: 600 12.5px var(--font-sans); padding: 0.4rem 0.8rem; border: 1px solid var(--border);
  background: var(--bg-card); color: var(--text); border-radius: 7px; cursor: pointer; white-space: nowrap; }
.ki-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
/* the keyboard button lives in the toolbar on iOS (ki-kbbtn); Monaco's floating one covered the code */
.monaco-editor .iPadShowKeyboard { display: none !important; }
.ki-kbbtn { display: inline-flex; align-items: center; padding: 0.4rem 0.55rem; }
.ki-btn:disabled, .ki-go:disabled { opacity: 0.45; cursor: default; }
.ki-railbtn { padding: 0.36rem 0.5rem; line-height: 0; color: var(--text-muted); }
.ki-railbtn[aria-pressed="true"] { color: var(--accent); }
.ki-go { font: 600 13.5px var(--font-sans); padding: 0.42rem 1.1rem; border: 1px solid var(--accent-fill);
  background: var(--accent-fill); color: var(--on-accent); border-radius: 7px; cursor: pointer; white-space: nowrap; }
.ki-go:hover:not(:disabled) { filter: brightness(1.08); }
.ki-kbd { font: 500 11px var(--font-mono); opacity: 0.8; margin-left: 0.3rem; }
.ki-nbtoggle.on { background: var(--accent-wash); color: var(--accent); border-color: var(--accent); }

/* the file picker */
.ki-pick { position: relative; min-width: 0; }
.ki-pick-btn { display: inline-flex; align-items: center; gap: 0.45rem; max-width: min(34rem, 100%); min-width: 0;
  font: 600 13px var(--font-mono); color: var(--heading); background: transparent; border: 1px solid transparent;
  border-radius: 7px; padding: 0.38rem 0.6rem; cursor: pointer; }
.ki-pick-btn:hover, .ki-pick-btn[aria-expanded="true"] { border-color: var(--border); background: var(--bg-card); }
.ki-pick-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }
.ki-pick-caret { flex: none; color: var(--text-muted); }
.ki-pick-menu { position: absolute; z-index: 40; top: calc(100% + 4px); left: 0; width: min(30rem, calc(100vw - 32px));
  background: var(--bg-card); border: 1px solid var(--border-strong); border-radius: 9px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.14); display: flex; flex-direction: column; max-height: min(70vh, 34rem); }
.ki-pick-q { margin: 0.5rem; padding: 0.45rem 0.6rem; font: 13px var(--font-sans); color: var(--text);
  background: var(--bg); border: 1px solid var(--border); border-radius: 6px; }
.ki-pick-q:focus { outline: 2px solid var(--accent-soft); border-color: var(--accent); }
.ki-pick-list { overflow-y: auto; padding: 0 0.3rem 0.4rem; }
.ki-pick-group { font: 600 11.5px var(--font-sans); color: var(--text-faint); padding: 0.55rem 0.55rem 0.2rem; }
.ki-pick-item { display: flex; flex-direction: column; align-items: flex-start; width: 100%; text-align: left; gap: 0.05rem;
  border: none; background: transparent; padding: 0.3rem 0.55rem; border-radius: 6px; cursor: pointer; color: var(--text); }
.ki-pick-item.hi { background: var(--accent-wash); }
.ki-pick-label { font: 12.5px var(--font-mono); overflow-wrap: anywhere; }
.ki-pick-sub { font: 12px var(--font-sans); color: var(--text-muted); }
.ki-pick-none { font: 13px var(--font-sans); color: var(--text-faint); padding: 0.5rem 0.6rem; }

/* the work area */
.ki-main { flex: 1; display: grid; grid-template-columns: minmax(0, 1.4fr) 6px minmax(0, 1fr); min-height: 0; }
.ki-main.ki-hidden { display: none; }
.ki-nbpane { flex: 1; min-height: 0; overflow-y: auto; padding: 1rem var(--gutter) 3rem; }
.ki-hgrip { cursor: col-resize; background: var(--border); opacity: 0.6; transition: opacity 0.15s, background 0.15s; }
.ki-hgrip:hover { opacity: 1; background: var(--accent-fill); }
.ki-editor { min-width: 0; min-height: 0; }

/* the rail: outline, then files */
.ki-rail { min-height: 0; overflow-y: auto; border-right: 1px solid var(--border); background: var(--bg-soft); padding-bottom: 1rem; }
.ki-rail-head { font: 600 12px var(--font-sans); color: var(--text-muted); padding: 0.7rem 0.85rem 0.3rem; }
.ki-rail-none { font: 12.5px var(--font-sans); color: var(--text-faint); padding: 0.1rem 0.85rem 0.3rem; }
.ki-filter { display: block; width: calc(100% - 1.7rem); margin: 0 0.85rem 0.3rem; font: 12px var(--font-mono);
  padding: 0.28rem 0.5rem; border: 1px solid var(--border); border-radius: 6px; background: var(--bg-card); color: var(--text); }
.ki-row { display: block; width: 100%; text-align: left; border: none; background: transparent; cursor: pointer;
  font: 12px var(--font-mono); color: var(--text-muted); padding: 0.22rem 0.85rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ki-row:hover { color: var(--accent); background: var(--accent-wash); }
.ki-row.on { color: var(--accent); background: var(--accent-wash); font-weight: 600; }
.ki-fn { color: var(--text); }
.ki-group-head { display: flex; justify-content: space-between; font-weight: 600; color: var(--heading); }
.ki-group-head[aria-expanded="true"] { color: var(--accent); }
.ki-count { font-weight: 400; color: var(--text-faint); }
.ki-file { padding-left: 1.5rem; }
.ki-def { margin: 0.1rem 0.6rem 0.5rem; padding: 0.45rem 0.6rem; border: 1px solid var(--border); border-radius: 7px;
  background: var(--bg-card); font: 12.5px/1.45 var(--font-sans); color: var(--text); display: flex; flex-direction: column; gap: 0.35rem; }
.ki-def p { margin: 0; }
.ki-def-more { color: var(--text-muted); }
.ki-def-vars { margin: 0; display: flex; flex-direction: column; gap: 0.1rem; }
.ki-def-vars div { display: flex; gap: 0.45rem; }
.ki-def-vars dt { font: 600 11.5px var(--font-mono); color: var(--accent); }
.ki-def-vars dd { margin: 0; color: var(--text-muted); }
.ki-def-row { display: flex; flex-wrap: wrap; align-items: center; gap: 0.25rem; }
.ki-def-lab { font: 600 11.5px var(--font-sans); color: var(--text-faint); margin-right: 0.15rem; }
.ki-def-none { font: 11.5px var(--font-mono); color: var(--text-faint); }
.ki-chip { font: 600 11px var(--font-mono); padding: 0.08rem 0.45rem; border-radius: 999px; border: 1px solid var(--border);
  background: var(--bg); color: var(--accent); cursor: pointer; }
.ki-chip:hover { border-color: var(--accent); }
.ki-chip-main { color: var(--text-muted); cursor: default; }
.ki-tag { font: 600 11px var(--font-mono); background: var(--accent-wash); color: var(--accent); padding: 0.06rem 0.45rem; border-radius: 999px; }

/* the right column: tabs over one view */
.ki-right { display: flex; flex-direction: column; min-height: 0; min-width: 0; background: var(--bg); }
.ki-tabs { display: flex; gap: 0.1rem; padding: 0.3rem 0.5rem 0; background: var(--bg-soft);
  border-bottom: 1px solid var(--border); flex: none; overflow-x: auto; scrollbar-width: none; }
.ki-tab { font: 600 12px var(--font-sans); padding: 0.36rem 0.75rem; border: 1px solid transparent; white-space: nowrap;
  border-bottom: none; background: transparent; color: var(--text-muted); border-radius: 7px 7px 0 0; cursor: pointer; margin-bottom: -1px; }
.ki-tab:hover { color: var(--accent); }
.ki-tab.on { background: var(--bg); border-color: var(--border); color: var(--accent); }

/* the output stream */
.ki-out { flex: 1; min-height: 0; overflow: auto; padding: 0.6rem 0.9rem 1.2rem; display: flex; flex-direction: column; gap: 0.35rem; }
.ki-out-empty { font: 13px var(--font-sans); color: var(--text-faint); padding: 0.3rem 0; }
.ki-out-text { margin: 0; color: var(--text); font: 12.5px/1.55 var(--font-mono); white-space: pre-wrap; overflow-wrap: anywhere; }
.ki-out-text.ki-err { color: var(--danger); }
.ki-out-label { font: 600 12px var(--font-sans); color: var(--text-muted); margin-top: 0.45rem; }
.ki-out-label:first-child { margin-top: 0; }
.ki-figcard { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-card); padding: 4px; max-width: 46rem; }

.ki-graphpane { flex: 1; min-height: 0; overflow: auto; padding: 0.4rem 0.6rem; }
.ki-graphpane .ki-out-empty { padding: 0.3rem 0.3rem; }
.ki-cpane { flex: 1; min-height: 0; display: flex; flex-direction: column; }
.ki-cbar { display: flex; align-items: center; flex-wrap: wrap; gap: 0.4rem 0.7rem; padding: 0.45rem 0.9rem;
  border-bottom: 1px solid var(--border); background: var(--bg-soft); }
.ki-cstats { font: 12px var(--font-sans); color: var(--text-muted); }
.ki-gcc { font: 11.5px var(--font-mono); color: var(--text); background: var(--bg-code); border: 1px solid var(--border);
  padding: 0.12rem 0.45rem; border-radius: 5px; overflow-wrap: anywhere; user-select: all; }
.ki-dl { margin-left: auto; padding: 0.28rem 0.7rem; font-size: 12px; }
.ki-csrc { flex: 1; min-height: 0; }
.ki-termpane { flex: 1; min-height: 0; display: flex; flex-direction: column; }
.ki-termpane[hidden] { display: none; }

/* phone: editor first, output under it, Run pinned to the bottom */
.ki-narrow { height: auto; min-height: 0; padding-bottom: 4.5rem; }
.ki-narrow .ki-bar { padding: 0.4rem var(--gutter); }
.ki-narrow .ki-kbd { display: none; }
.ki-narrow .ki-main { display: flex; flex-direction: column; }
.ki-narrow .ki-editor { height: 58svh; border-bottom: 1px solid var(--border); }
.ki-narrow .ki-right { min-height: 40svh; }
.ki-narrow .ki-out { overflow: visible; padding: 0.6rem var(--gutter) 1rem; }
.ki-narrow .ki-graphpane, .ki-narrow .ki-cpane, .ki-narrow .ki-termpane { flex: none; height: 70svh; }
.ki-narrow .ki-tabs { padding: 0.3rem var(--gutter) 0; }
.ki-narrow .ki-tab { padding: 0.36rem 0.6rem; }
.ki-narrow .ki-go { position: fixed; z-index: 30; left: var(--gutter); right: var(--gutter); bottom: 12px;
  padding: 0.7rem 1rem; font-size: 15px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2); }
.ki-narrow .ki-nbtoggle { padding: 0.36rem 0.6rem; }
.ki-narrow .ki-pick-btn { max-width: 100%; }
.ki-drawer { position: fixed; z-index: 50; top: 0; bottom: 0; left: 0; width: min(20rem, 86vw); box-shadow: 4px 0 24px rgba(0, 0, 0, 0.2); }
.ki-drawer-bar { display: flex; align-items: center; justify-content: space-between; padding: 0.6rem 0.85rem;
  border-bottom: 1px solid var(--border); font: 600 13px var(--font-sans); color: var(--heading); }
.ki-scrim { position: fixed; inset: 0; z-index: 45; background: rgba(0, 0, 0, 0.3); }
`;
