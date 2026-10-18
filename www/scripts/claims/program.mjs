// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The program half of the claims database: what a post's code and its run
// establish. Tests and their assertions, the values the run prints and
// `show`s, the constants the program is given, the figures it draws, and the
// numbers its reference script printed. Pure functions over text.
import { testBlocksOf } from '../../src/lib/postCheck.js';

// ---- numbers as printed ---------------------------------------------------------

/** "0.956686" -> { value, ulp }: the printed digits say how precise the value is. */
export function printed(raw, g = true) {
  const m = String(raw).trim().match(/^([-+]?)(\d*)(?:\.(\d*))?(?:[eE]([-+]?\d+))?$/);
  if (!m || (m[2] === '' && !m[3])) return null;
  const e = Number(m[4] || 0);
  const value = Number(raw);
  if (!Number.isFinite(value)) return null;
  // %g prints 6 significant digits and drops trailing zeros, so "1" may be
  // 1.00000: the ulp of a printed value is the larger of its last digit and
  // the sixth significant digit
  const dec = (m[3] || '').length;
  const lastDigit = 10 ** (e - dec);
  // an integer printed as one ("2", "256") is exact; a %g value's trailing zeros were dropped
  const sixth = value === 0 ? 1e-300 : 10 ** (Math.floor(Math.log10(Math.abs(value))) - 5);
  // a mantissa with no decimal point but an exponent ("1e-11", "5e-07") is not
  // fixed-precision output (%e always prints a point): it is repr or %g, good
  // to at least six significant digits
  if (!g && m[3] === undefined && m[4]) return { value, ulp: Number(Math.min(lastDigit, sixth).toPrecision(3)) };
  if (!g) return { value, ulp: m[3] === undefined && !m[4] ? 0 : Number(lastDigit.toPrecision(3)) };
  return { value, ulp: Number(Math.min(lastDigit, Math.max(sixth, lastDigit / 1e10)).toPrecision(3)) };
}

const NUM_RE = /(?<![\w.^_])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])/g;

// the run's lines that are figure records, animation frames, or the
// interpreter's call graph: drawn, not printed
const FIGURE_LINE = /^(figure|plot|plot2d|plot3d|scatter|quiver|bars|fill_between|rangebars|stage|axes|curve|dots|label|hline|vline|xtick|ytick|xscale|yscale|aspect|yright|grid|share|style|origin|colormap|clim|levels|circle|rect|segment|poly|draw|fade_in|fade_out|morph|move_to|say|wait|frame|together|apart):/;

/**
 * What a run prints, as values:
 *   shown    `name: value` and `name: [v0, v1, ...]` lines (what `show` prints)
 *   printed  every number on any other printed line, with the line
 *   tests    PASS / FAIL lines
 */
export function parseRun(out) {
  const shown = [];
  const printedVals = [];
  const pass = [];
  const fail = [];
  for (const line of String(out || '').split('\n')) {
    if (!line.trim() || line.startsWith('__') || FIGURE_LINE.test(line)) continue;
    if (line.startsWith('PASS ')) { pass.push(line.slice(5)); continue; }
    if (line.startsWith('FAIL ')) { fail.push(line.slice(5)); continue; }
    const sm = line.match(/^([A-Za-z_]\w*): (.*)$/);
    if (sm) {
      const v = sm[2].trim();
      if (/^\[.*\]$/.test(v)) {
        const items = v.slice(1, -1).split(',').map((x) => x.trim());
        const vals = items.map((x) => printed(x));
        if (vals.every(Boolean)) {
          vals.forEach((p, i) => shown.push({ name: sm[1], index: i, ...p }));
          continue;
        }
      } else {
        const p = printed(v);
        if (p) { shown.push({ name: sm[1], index: null, ...p }); continue; }
      }
    }
    for (const m of line.matchAll(NUM_RE)) {
      const p = printed(m[0]);
      if (p) printedVals.push({ line: line.trim().slice(0, 160), ...p });
    }
  }
  return { shown, printed: printedVals, pass, fail };
}

/** The figures a run draws: each with its title, axes, and the series in it. */
export function parseFigures(out) {
  const figs = [];
  let cur = null;
  let group = '';
  for (const line of String(out || '').split('\n')) {
    const m = line.match(/^([a-z_0-9]+): (.*)$/);
    if (!m || !FIGURE_LINE.test(line)) continue;
    const [, kind, rest] = m;
    const f = rest.split(' ; ');
    if (kind === 'grid') { group = f[0]; continue; }
    if (kind === 'figure' || kind === 'stage') {
      cur = { kind: kind === 'stage' ? 'animation' : 'figure', title: f[0] || '', x: f[1] || '', y: f[2] || '', group, series: [] };
      figs.push(cur);
      continue;
    }
    if (!cur) continue;
    if (['plot', 'scatter', 'plot2d', 'plot3d', 'fill_between', 'rangebars', 'bars', 'curve', 'dots', 'quiver'].includes(kind)) {
      const arr = (rest.match(/\[[^\]]*\]/) || [''])[0];
      const n = arr ? arr.split(',').length : 0;
      cur.series.push({ type: kind, label: f[0] || '', points: kind === 'plot2d' ? (f[1] || '').replace(/\s+/, ' x ') : n });
    } else if (kind === 'vline' || kind === 'hline') {
      cur.series.push({ type: kind, label: f[0] || '', at: Number(f[1]) });
    }
  }
  return figs;
}

// ---- constant expressions ------------------------------------------------------

const FN = { sqrt: Math.sqrt, log: Math.log, exp: Math.exp, sin: Math.sin, cos: Math.cos, tan: Math.tan, abs: Math.abs, pow: Math.pow, min: Math.min, max: Math.max, sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh, atan: Math.atan, atan2: Math.atan2, floor: Math.floor, ceil: Math.ceil, log10: Math.log10, log2: Math.log2 };
/**
 * Evaluate a Viv expression made of numbers, known constants, + - * /, and the
 * math builtins. Returns a number, or null when it names anything else.
 */
export function evalConst(expr, env = {}) {
  const src = String(expr).trim();
  let i = 0;
  const peek = () => { while (src[i] === ' ') i++; return src[i]; };
  const fail = () => { throw new Error('not constant'); };
  function primary() {
    const c = peek();
    if (c === '(') { i++; const v = sum(); if (peek() !== ')') fail(); i++; return v; }
    if (c === '-') { i++; return -primary(); }
    if (c === '+') { i++; return primary(); }
    const num = src.slice(i).match(/^(\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?/);
    if (num) { i += num[0].length; return Number(num[0]); }
    const id = src.slice(i).match(/^[A-Za-z_]\w*/);
    if (!id) fail();
    i += id[0].length;
    if (peek() === '(') {
      i++;
      const args = [];
      if (peek() !== ')') { args.push(sum()); while (peek() === ',') { i++; args.push(sum()); } }
      if (peek() !== ')') fail();
      i++;
      if (!FN[id[0]]) fail();
      return FN[id[0]](...args);
    }
    if (id[0] === 'pi' || id[0] === 'PI') return env[id[0]] ?? Math.PI;
    if (id[0] === 'true') return 1;
    if (id[0] === 'false') return 0;
    if (typeof env[id[0]] === 'number') return env[id[0]];
    return fail();
  }
  function prod() {
    let v = primary();
    for (;;) {
      const c = peek();
      if (c === '*') { i++; v *= primary(); } else if (c === '/') { i++; v /= primary(); } else return v;
    }
  }
  function sum() {
    let v = prod();
    for (;;) {
      const c = peek();
      if (c === '+') { i++; v += prod(); } else if (c === '-') { i++; v -= prod(); } else return v;
    }
  }
  try {
    const v = sum();
    if (peek() !== undefined) return null;
    return Number.isFinite(v) ? v : null;
  } catch { return null; }
}

// ---- the program's text ----------------------------------------------------------

/** The code with strings blanked and comments dropped, line by line. */
const stripCode = (line) => line.replace(/"(?:[^"\\]|\\.)*"/g, '""').replace(/#.*$/, '');

const LIT = /(?<![\w.])(\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])/g;
/**
 * The numeric literals of a line of code, and the quotient of two literals
 * divided one by the other (`1.0 / 24.0` gives the program 1/24 as well as 24).
 */
export function literalsIn(line) {
  const out = [];
  for (const m of String(line).matchAll(LIT)) {
    const v = Number(m[0]);
    if (Number.isFinite(v)) out.push(v);
  }
  for (const m of String(line).matchAll(/(?<![\w.)\]])(\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*\/\s*(\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(?![\w.(\[])/g)) {
    const [a, b] = m[0].split('/').map((x) => Number(x.trim()));
    if (Number.isFinite(a) && b) out.push(a / b);
  }
  return out;
}

/** Split a call's argument list at its top-level commas. */
function splitArgs(s) {
  const out = [];
  let depth = 0;
  let cur = '';
  for (const c of s) {
    if (c === '(' || c === '[') depth++;
    if (c === ')' || c === ']') depth--;
    if (c === ',' && depth === 0) { out.push(cur.trim()); cur = ''; } else cur += c;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

/** The assertions inside a test block's text. */
export function assertsOf(text) {
  const out = [];
  const re = /\bassert(_[a-z]+)?\s*\(/g;
  let m;
  while ((m = re.exec(text))) {
    let depth = 1;
    let j = re.lastIndex;
    for (; j < text.length && depth; j++) {
      if (text[j] === '(') depth++;
      else if (text[j] === ')') depth--;
    }
    const args = splitArgs(text.slice(re.lastIndex, j - 1));
    out.push({ fn: `assert${m[1] || ''}`, args });
    re.lastIndex = j;
  }
  return out;
}

const identifiers = (expr) => [...new Set([...String(expr).matchAll(/(?<![\w.])([A-Za-z_]\w*)(?!\s*\()/g)].map((m) => m[1]).filter((x) => !/^(pi|PI|true|false|and|or|not|len)$/.test(x)))];

/**
 * A program's constants: every top-level `NAME = <constant expression>`,
 * evaluated in file order so later ones can use earlier ones, and every
 * numeric literal outside tests and figure calls. These are what the program
 * is GIVEN: a number in the prose that matches one is restating an input.
 */
export function constantsOf(code) {
  const env = {};
  const consts = [];
  const literals = [];
  let depth = 0;
  let inTest = 0;
  for (const raw of String(code).split('\n')) {
    const line = stripCode(raw);
    const top = depth === 0;
    if (/^\s*test\s+""/.test(line)) inTest = depth + 1;
    if (top) {
      const a = line.match(/^([A-Za-z_]\w*)\s*=\s*(.+?)\s*$/);
      if (a && !/^\s*\[/.test(a[2])) {
        const v = evalConst(a[2], env);
        if (v !== null) { env[a[1]] = v; consts.push({ name: a[1], value: v, expr: a[2] }); }
      } else if (a && /^\[[^\]]*\]$/.test(a[2])) {
        const items = splitArgs(a[2].slice(1, -1)).map((x) => evalConst(x, env));
        items.forEach((v, k) => { if (v !== null) consts.push({ name: a[1], index: k, value: v, expr: a[2] }); });
      }
    }
    if (!inTest && !/^\s*(figure|plot|scatter|plot2d|label|style|vline|hline|xtick|ytick|colormap|grid|rangebars|fill_between|axes|stage|say|frame)\s*\(/.test(line)) {
      for (const v of literalsIn(line)) literals.push({ value: v, line: raw.trim().slice(0, 120) });
    }
    for (const c of line) { if (c === '{') depth++; else if (c === '}') depth--; }
    if (inTest && depth < inTest) inTest = 0;
  }
  return { env, consts, literals };
}

/**
 * The tests of a program: name, each assertion's kind, the value it expects
 * and the tolerance or bound it allows (evaluated when they are constants),
 * and the names it asserts on.
 */
export function testsOf(src, env) {
  return testBlocksOf(src).map((t) => {
    // a test's own constants (`expected = 0.44303797`) and the literals it
    // gives the code it calls (`root(0.3)`), outside its assertions
    const local = { ...env };
    const given = [];
    for (const line of t.text.split('\n').slice(1).map(stripCode)) {
      if (/\bassert(_[a-z]+)?\s*\(/.test(line)) continue;
      const a = line.match(/^([A-Za-z_]\w*)\s*=\s*(.+?)\s*$/);
      if (a) { const v = evalConst(a[2], local); if (v !== null) local[a[1]] = v; else delete local[a[1]]; }
      given.push(...literalsIn(line));
    }
    const asserts = assertsOf(t.text).map((a) => {
      const r = { fn: a.fn, actual: a.args[0] || '' };
      r.actualValue = evalConst(r.actual, local);
      // every literal in the assertion's arguments: `assert_lt(100.0 * weight, t)` states 100
      r.literals = literalsIn(a.args.join(' , '));
      if (a.fn === 'assert_near') {
        r.expected = a.args[1] || '';
        r.tolerance = a.args[2] || '';
        r.expectedValue = evalConst(r.expected, local);
        r.toleranceValue = evalConst(r.tolerance, local);
      } else if (a.fn === 'assert_lt' || a.fn === 'assert_gt') {
        r.bound = a.args[1] || '';
        r.boundValue = evalConst(r.bound, local);
      }
      // `on`: what the assertion measures, when it is one variable (or one element of one)
      const bare = r.actual.match(/^([A-Za-z_]\w*)(?:\[[^\]]*\])?$/);
      r.on = bare ? bare[1] : null;
      r.names = [...identifiers(r.actual), ...(r.expectedValue === null && r.expected ? identifiers(r.expected) : []), ...(r.boundValue === null && r.bound ? identifiers(r.bound) : [])];
      return r;
    });
    return { name: t.name, asserts, literals: given };
  });
}

/** Numbers in a reference script's output, with their line. */
export function parseRefOut(text) {
  const out = [];
  String(text || '').split('\n').forEach((line, n) => {
    for (const m of line.matchAll(NUM_RE)) {
      const p = printed(m[0], false);
      if (p) out.push({ line: n + 1, text: line.trim().slice(0, 160), ...p });
    }
  });
  return out;
}
