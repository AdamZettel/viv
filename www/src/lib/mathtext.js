// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Mathtext for the site's charts: the `$...$` parts of a figure label set as
// math in SVG. The same code as lang/viv/src/mathtext.rs (`viv plot`), line
// for line, so a chart drawn on the page and one drawn by `viv plot` set a
// label alike; www/scripts/mathtexttest.mjs holds both to
// lang/viv/tests/figures/mathtext_labels.tsv. See mathtext.rs for the subset.
//
//   mathLabel(s, size) -> { inner, shapes, w, h, d, math }   (throws MathError)
//   shapesSvg(label, x, y, anchor, rotate, fill) -> SVG string
//   mathHtml(s) -> HTML with the math set by KaTeX, for labels outside SVG
import { FONTS } from './mathtextMetrics.js';

const F = { Main: 0, Bold: 1, Italic: 2, Math: 3, MathBold: 4, Size1: 5 };
const C = { Ord: 'Ord', Op: 'Op', Bin: 'Bin', Rel: 'Rel', Open: 'Open', Close: 'Close', Punct: 'Punct', Inner: 'Inner' };

const MAIN_STACK = "KaTeX_Main, 'Latin Modern Roman', 'Times New Roman', serif";
const MATH_STACK = "KaTeX_Math, 'Latin Modern Math', 'Times New Roman', serif";
const SIZE1_STACK = "KaTeX_Size1, KaTeX_Main, 'Times New Roman', serif";

export const MATH_SCALE = 1.15;
const SCALES = [1.0, 0.7, 0.5];
const X_HEIGHT = 0.431;
const NUM1 = 0.677;
const NUM2 = 0.394;
const DENOM1 = 0.686;
const DENOM2 = 0.345;
const SUP1 = 0.413;
const SUB1 = 0.15;
const SUB2 = 0.247;
const SUP_DROP = 0.386;
const SUB_DROP = 0.05;
const AXIS = 0.25;
const RULE = 0.04;
const SCRIPT_SPACE = 0.05;
const NULL_DELIM = 0.12;
const MU = 1.0 / 18.0;

export class MathError extends Error {
  constructor(col, msg) {
    super(`${msg} at column ${col}`);
    this.col = col;
    this.msg = msg;
  }
}

const GREEK_LOWER = {
  alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ϵ', varepsilon: 'ε', zeta: 'ζ', eta: 'η', theta: 'θ',
  vartheta: 'ϑ', iota: 'ι', kappa: 'κ', lambda: 'λ', mu: 'μ', nu: 'ν', xi: 'ξ', pi: 'π', varpi: 'ϖ', rho: 'ρ',
  varrho: 'ϱ', sigma: 'σ', varsigma: 'ς', tau: 'τ', upsilon: 'υ', phi: 'ϕ', varphi: 'φ', chi: 'χ', psi: 'ψ', omega: 'ω',
};
const TABLE = [
  ['Gamma', 'Γ', 'Ord'], ['Delta', 'Δ', 'Ord'], ['Theta', 'Θ', 'Ord'], ['Lambda', 'Λ', 'Ord'], ['Xi', 'Ξ', 'Ord'],
  ['Pi', 'Π', 'Ord'], ['Sigma', 'Σ', 'Ord'], ['Upsilon', 'Υ', 'Ord'], ['Phi', 'Φ', 'Ord'], ['Psi', 'Ψ', 'Ord'],
  ['Omega', 'Ω', 'Ord'],
  ['partial', '∂', 'Ord'], ['nabla', '∇', 'Ord'], ['infty', '∞', 'Ord'], ['hbar', 'ℏ', 'Ord'], ['ell', 'ℓ', 'Ord'],
  ['prime', '′', 'Ord'], ['top', '⊤', 'Ord'], ['bot', '⊥', 'Ord'], ['emptyset', '∅', 'Ord'], ['forall', '∀', 'Ord'],
  ['exists', '∃', 'Ord'], ['neg', '¬', 'Ord'], ['ldots', '…', 'Inner'], ['cdots', '⋯', 'Inner'], ['dots', '…', 'Inner'],
  ['degree', '°', 'Ord'], ['|', '∥', 'Ord'], ['Vert', '∥', 'Ord'], ['vert', '∣', 'Ord'], ['star', '⋆', 'Bin'],
  ['pm', '±', 'Bin'], ['mp', '∓', 'Bin'], ['times', '×', 'Bin'], ['cdot', '⋅', 'Bin'], ['div', '÷', 'Bin'],
  ['ast', '∗', 'Bin'], ['circ', '∘', 'Bin'], ['bullet', '∙', 'Bin'], ['oplus', '⊕', 'Bin'], ['otimes', '⊗', 'Bin'],
  ['cup', '∪', 'Bin'], ['cap', '∩', 'Bin'], ['wedge', '∧', 'Bin'], ['vee', '∨', 'Bin'], ['setminus', '∖', 'Bin'],
  ['le', '≤', 'Rel'], ['leq', '≤', 'Rel'], ['ge', '≥', 'Rel'], ['geq', '≥', 'Rel'], ['ne', '≠', 'Rel'], ['neq', '≠', 'Rel'],
  ['approx', '≈', 'Rel'], ['sim', '∼', 'Rel'], ['simeq', '≃', 'Rel'], ['propto', '∝', 'Rel'], ['equiv', '≡', 'Rel'],
  ['ll', '≪', 'Rel'], ['gg', '≫', 'Rel'], ['to', '→', 'Rel'], ['rightarrow', '→', 'Rel'], ['leftarrow', '←', 'Rel'],
  ['gets', '←', 'Rel'], ['Rightarrow', '⇒', 'Rel'], ['Leftarrow', '⇐', 'Rel'], ['leftrightarrow', '↔', 'Rel'],
  ['Leftrightarrow', '⇔', 'Rel'], ['mapsto', '↦', 'Rel'], ['in', '∈', 'Rel'], ['notin', '∉', 'Rel'],
  ['subset', '⊂', 'Rel'], ['subseteq', '⊆', 'Rel'], ['supset', '⊃', 'Rel'], ['supseteq', '⊇', 'Rel'],
  ['perp', '⊥', 'Rel'], ['mid', '∣', 'Rel'], ['parallel', '∥', 'Rel'], ['cong', '≅', 'Rel'],
  ['langle', '⟨', 'Open'], ['rangle', '⟩', 'Close'], ['lvert', '∣', 'Open'], ['rvert', '∣', 'Close'],
  ['lVert', '∥', 'Open'], ['rVert', '∥', 'Close'], ['lfloor', '⌊', 'Open'], ['rfloor', '⌋', 'Close'],
  ['lceil', '⌈', 'Open'], ['rceil', '⌉', 'Close'], ['lbrace', '{', 'Open'], ['rbrace', '}', 'Close'],
  ['{', '{', 'Open'], ['}', '}', 'Close'],
  ['$', '$', 'Ord'], ['%', '%', 'Ord'], ['&', '&', 'Ord'], ['_', '_', 'Ord'], ['#', '#', 'Ord'],
];
const BIG = { sum: '∑', prod: '∏', coprod: '∐', int: '∫', oint: '∮' };

function symbol(name) {
  if (Object.hasOwn(GREEK_LOWER, name)) return [GREEK_LOWER[name], 'Ord', F.Math];
  const t = TABLE.find((r) => r[0] === name);
  if (t) return [t[1], t[2], F.Main];
  if (Object.hasOwn(BIG, name)) return [BIG[name], 'Op', F.Size1];
  return null;
}

const OPERATORS = [
  'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh', 'coth',
  'exp', 'log', 'ln', 'lg', 'max', 'min', 'sup', 'inf', 'lim', 'det', 'dim', 'ker', 'arg', 'deg', 'gcd', 'Pr',
];

const ACCENTS = { hat: 'ˆ', bar: 'ˉ', vec: '→', dot: '˙', ddot: '¨', tilde: '˜' };

const isAlpha = (c) => /^[A-Za-z]$/.test(c);
const isDigit = (c) => /^[0-9]$/.test(c);
const isSpace = (c) => /^\s$/u.test(c);

function charNode(ch, style) {
  let cls = 'Ord';
  if (ch === '+') cls = 'Bin';
  else if (ch === '-') { ch = '−'; cls = 'Bin'; }
  else if (ch === '*') { ch = '∗'; cls = 'Bin'; }
  else if ('=<>:'.includes(ch)) cls = 'Rel';
  else if (',;'.includes(ch)) cls = 'Punct';
  else if ('(['.includes(ch)) cls = 'Open';
  else if (')]!?'.includes(ch)) cls = 'Close';
  const letter = isAlpha(ch);
  let font;
  if (style === F.Math) font = letter ? F.Math : F.Main;
  else if (style === F.Main) font = F.Main;
  else if (style === F.Bold) font = F.Bold;
  else if (style === F.Italic) font = letter || isDigit(ch) ? F.Italic : F.Main;
  else font = F.Main;
  return { t: 'glyph', ch, font, cls };
}

function nodeCls(n) {
  switch (n.t) {
    case 'glyph': case 'word': return n.cls;
    case 'scripts': return nodeCls(n.base);
    case 'frac': case 'delim': return 'Inner';
    case 'space': return null;
    default: return 'Ord';
  }
}

class Parser {
  constructor(s, base) { this.s = s; this.i = 0; this.base = base; }
  err(at, msg) { return new MathError(this.base + at + 1, msg); }
  skipSpace() { while (this.i < this.s.length && isSpace(this.s[this.i])) this.i++; }
  command() {
    const start = this.i;
    this.i++;
    if (this.i >= this.s.length) throw this.err(start, 'a backslash at the end of the math');
    if (isAlpha(this.s[this.i])) {
      const a = this.i;
      while (this.i < this.s.length && isAlpha(this.s[this.i])) this.i++;
      return this.s.slice(a, this.i).join('');
    }
    this.i++;
    return this.s[this.i - 1];
  }
  peekCommand(name) {
    const n = Array.from(name);
    const end = this.i + 1 + n.length;
    return this.s[this.i] === '\\' && end <= this.s.length
      && this.s.slice(this.i + 1, end).join('') === name
      && !(end < this.s.length && isAlpha(this.s[end]));
  }
  list(style, inGroup) {
    const out = [];
    for (;;) {
      this.skipSpace();
      if (this.i >= this.s.length) return out;
      const c = this.s[this.i];
      if (c === '}') {
        if (inGroup) return out;
        throw this.err(this.i, 'a } with no {');
      }
      if (c === '\\' && this.peekCommand('right')) return out;
      if (c === '^' || c === '_' || c === "'") {
        const last = out[out.length - 1];
        const base = last && last.t !== 'space' ? out.pop() : { t: 'group', v: [] };
        out.push(this.scripts(base, style));
        continue;
      }
      out.push(this.atom(style));
    }
  }
  scripts(base0, style) {
    let base = base0, sup = null, sub = null;
    if (base0.t === 'scripts') { base = base0.base; sup = base0.sup; sub = base0.sub; }
    for (;;) {
      this.skipSpace();
      if (this.i >= this.s.length) break;
      const at = this.i;
      const c = this.s[at];
      if (c === "'") {
        this.i++;
        const p = { t: 'glyph', ch: '′', font: F.Main, cls: 'Ord' };
        if (sup) sup.unshift(p); else sup = [p];
      }
      else if (c === '^' || c === '_') {
        this.i++;
        this.skipSpace();
        if (this.i >= this.s.length || this.s[this.i] === '}') throw this.err(at, `${c} needs something to raise or lower`);
        const arg = this.argument(style);
        if (c === '^') {
          if (sup && !(sup.length === 1 && sup[0].t === 'glyph' && sup[0].ch === '′')) throw this.err(at, 'a second superscript (use braces: x^{ab})');
          if (sup) sup.push(...arg); else sup = arg;
        }
        else {
          if (sub) throw this.err(at, 'a second subscript (use braces: x_{ab})');
          sub = arg;
        }
      }
      else break;
    }
    if (base.t === 'scripts') base = { t: 'group', v: [base] };
    return { t: 'scripts', base, sup, sub };
  }
  argument(style) {
    this.skipSpace();
    if (this.i >= this.s.length) throw this.err(Math.max(this.i - 1, 0), 'a missing argument');
    if (this.s[this.i] === '{') {
      const open = this.i;
      this.i++;
      const v = this.list(style, true);
      if (this.i >= this.s.length) throw this.err(open, 'a { with no }');
      this.i++;
      return v;
    }
    return [this.atom(style)];
  }
  rawArgument(what) {
    this.skipSpace();
    if (this.i >= this.s.length || this.s[this.i] !== '{') throw this.err(Math.min(this.i, Math.max(this.s.length - 1, 0)), `${what} needs a braced argument`);
    const open = this.i;
    this.i++;
    let t = '';
    while (this.i < this.s.length && this.s[this.i] !== '}') {
      const c = this.s[this.i];
      if (c === '\\') {
        const e = this.s[this.i + 1];
        if (e !== undefined && '${}%&_#'.includes(e)) { t += e; this.i += 2; continue; }
        throw this.err(this.i, `only \\$ \\{ \\} \\% \\& \\_ \\# may be used inside ${what}`);
      }
      if (c === '{') throw this.err(this.i, `a { inside ${what}`);
      t += c;
      this.i++;
    }
    if (this.i >= this.s.length) throw this.err(open, 'a { with no }');
    this.i++;
    return t;
  }
  delimiter(which) {
    this.skipSpace();
    const at = this.i;
    if (at >= this.s.length) throw this.err(Math.max(at - 1, 0), `\\${which} needs a delimiter`);
    const c = this.s[at];
    if (c === '\\') {
      const name = this.command();
      const m = { langle: '⟨', rangle: '⟩', '{': '{', lbrace: '{', '}': '}', rbrace: '}', '|': '∥', Vert: '∥', lVert: '∥', rVert: '∥', vert: '|', lvert: '|', rvert: '|', lfloor: '⌊', rfloor: '⌋', lceil: '⌈', rceil: '⌉' };
      if (Object.hasOwn(m, name)) return m[name];
      throw this.err(at, `\\${name} is not a delimiter for \\${which}`);
    }
    this.i++;
    if ('()[]|/'.includes(c)) return c;
    if (c === '.') return null;
    throw this.err(at, `${c} is not a delimiter for \\${which}`);
  }
  atom(style) {
    const at = this.i;
    const c = this.s[at];
    if (c === '{') return { t: 'group', v: this.argument(style) };
    if (c === '$') throw this.err(at, 'a $ inside math');
    if ('&#%~'.includes(c)) throw this.err(at, `${c} is not supported in a label (write \\${c} for the character)`);
    if (c !== '\\') { this.i++; return charNode(c, style); }
    const name = this.command();
    const sym = symbol(name);
    if (sym) {
      let [ch, cls, font] = sym;
      if (font === F.Math && style === F.Bold) font = F.MathBold;
      else if (font === F.Main && style === F.Bold) font = F.Bold;
      return { t: 'glyph', ch, font, cls };
    }
    if (OPERATORS.includes(name)) return { t: 'word', text: name, font: F.Main, cls: 'Op' };
    if (Object.hasOwn(ACCENTS, name)) return { t: 'accent', ch: ACCENTS[name], body: this.argument(style) };
    const space = (mu) => ({ t: 'space', em: mu * MU });
    switch (name) {
      case ',': return space(3.0);
      case ':': case '>': return space(4.0);
      case ';': return space(5.0);
      case '!': return space(-3.0);
      case ' ': return space(6.0);
      case 'quad': return space(18.0);
      case 'qquad': return space(36.0);
      case 'frac': case 'dfrac': case 'tfrac': {
        const num = this.argument(style);
        this.skipSpace();
        if (this.i >= this.s.length) throw this.err(at, `\\${name} needs two arguments`);
        const den = this.argument(style);
        return { t: 'frac', num, den, display: name === 'dfrac' };
      }
      case 'sqrt':
        this.skipSpace();
        if (this.i < this.s.length && this.s[this.i] === '[') throw this.err(this.i, '\\sqrt[n] (a root index) is not supported');
        return { t: 'sqrt', v: this.argument(style) };
      case 'overline': return { t: 'overline', v: this.argument(style) };
      case 'mathrm': return { t: 'group', v: this.argument(F.Main) };
      case 'mathbf': return { t: 'group', v: this.argument(F.Bold) };
      case 'mathit': return { t: 'group', v: this.argument(F.Italic) };
      case 'text': case 'textrm': case 'mbox': return { t: 'word', text: this.rawArgument(`\\${name}`), font: F.Main, cls: 'Ord' };
      case 'operatorname': return { t: 'word', text: this.rawArgument('\\operatorname'), font: F.Main, cls: 'Op' };
      case 'left': {
        const left = this.delimiter('left');
        const body = this.list(style, false);
        if (this.i >= this.s.length || !this.peekCommand('right')) throw this.err(at, '\\left with no \\right');
        this.i += 6;
        const right = this.delimiter('right');
        return { t: 'delim', left, body, right };
      }
      case 'right': throw this.err(at, '\\right with no \\left');
      default: throw this.err(at, `unknown command \\${name}`);
    }
  }
}

// ---- layout -------------------------------------------------------------------

const FALLBACK = [0.0, 0.7, 0.0, 0.0, 0.6];

function metric(font, ch) {
  if (ch === '≠') ch = '=';
  const m = FONTS[font].get(ch.codePointAt(0));
  if (m) return m;
  if (font !== F.Main) {
    const r = metric(F.Main, ch);
    if (r !== FALLBACK) return r;
  }
  return FALLBACK;
}

function spacing(a, b, script) {
  const k = `${a},${b}`;
  let mu = 0, keep = true;
  if (['Ord,Op', 'Op,Ord', 'Op,Op', 'Close,Op', 'Inner,Op'].includes(k)) { mu = 3; keep = true; }
  else if (['Ord,Bin', 'Bin,Ord', 'Bin,Op', 'Bin,Open', 'Bin,Inner', 'Close,Bin', 'Inner,Bin'].includes(k)) { mu = 4; keep = false; }
  else if (['Ord,Rel', 'Op,Rel', 'Rel,Ord', 'Rel,Op', 'Rel,Open', 'Rel,Inner', 'Close,Rel', 'Inner,Rel'].includes(k)) { mu = 5; keep = false; }
  else if (['Ord,Inner', 'Op,Inner', 'Close,Inner', 'Inner,Ord', 'Inner,Open', 'Inner,Punct', 'Inner,Inner'].includes(k)) { mu = 3; keep = false; }
  else if (['Punct,Rel', 'Punct,Ord', 'Punct,Op', 'Punct,Open', 'Punct,Close', 'Punct,Punct', 'Punct,Inner'].includes(k)) { mu = 3; keep = false; }
  return script && !keep ? 0 : mu;
}

function classes(nodes) {
  const out = nodes.map(nodeCls);
  let prev = null;
  for (let k = 0; k < out.length; k++) {
    const c = out[k];
    if (c === null) continue;
    if (c === 'Bin') {
      const before = prev === null ? null : out[prev];
      if ([null, 'Bin', 'Op', 'Rel', 'Open', 'Punct'].includes(before)) out[k] = 'Ord';
    }
    if (['Rel', 'Close', 'Punct'].includes(c) && prev !== null && out[prev] === 'Bin') out[prev] = 'Ord';
    prev = k;
  }
  if (prev !== null && out[prev] === 'Bin') out[prev] = 'Ord';
  return out;
}

const hbox = () => ({ items: [], w: 0, h: 0, d: 0 });
function put(b, c, x, y) {
  for (const it of c.items) b.items.push({ ...it, x: it.x + x, y: it.y + y });
}

class Lay {
  constructor(base) { this.base = base; }
  size(level) { return this.base * SCALES[Math.min(level, 2)]; }
  glyph(ch, font, size) {
    const m = metric(font, ch);
    const w = m[4] * size;
    return { items: [{ kind: 'glyph', text: ch, font, size, w, x: 0, y: 0 }], w, h: m[1] * size, d: m[0] * size };
  }
  list(nodes, level) {
    const size = this.size(level);
    const cls = classes(nodes);
    const b = hbox();
    let prev = null;
    nodes.forEach((n, k) => {
      const c = cls[k];
      if (prev !== null && c !== null) b.w += spacing(prev, c, level > 0) * MU * size;
      const nb = this.node(n, level);
      const x = b.w;
      b.h = Math.max(b.h, nb.h);
      b.d = Math.max(b.d, nb.d);
      b.w += nb.w;
      put(b, nb, x, 0);
      if (c !== null) prev = c;
    });
    return b;
  }
  node(n, level) {
    const size = this.size(level);
    switch (n.t) {
      case 'glyph': return this.glyph(n.ch, n.font, size);
      case 'word': {
        let w = 0, h = 0, d = 0;
        for (const ch of n.text) {
          const m = metric(n.font, ch);
          w += m[4] * size;
          h = Math.max(h, m[1] * size);
          d = Math.max(d, m[0] * size);
        }
        return { items: [{ kind: 'glyph', text: n.text, font: n.font, size, w, x: 0, y: 0 }], w, h, d };
      }
      case 'group': return this.list(n.v, level);
      case 'space': return { ...hbox(), w: n.em * size };
      case 'scripts': return this.scripts(n.base, n.sup, n.sub, level);
      case 'frac': return this.frac(n.num, n.den, n.display, level);
      case 'sqrt': return this.sqrt(n.v, level);
      case 'overline': {
        const bb = this.list(n.v, level);
        const th = RULE * size;
        const top = bb.h + 3.0 * th;
        const b = { w: bb.w, h: top + th, d: bb.d, items: [] };
        b.items.push({ kind: 'rule', w: bb.w, h: th, x: 0, y: -(top + th) });
        put(b, bb, 0, 0);
        return b;
      }
      case 'accent': return this.accent(n.ch, n.body, level);
      case 'delim': return this.delimited(n.left, n.body, n.right, level);
      default: return hbox();
    }
  }
  scripts(base, sup, sub, level) {
    const size = this.size(level);
    const ssize = this.size(level + 1);
    const bb = this.node(base, level);
    let isChar = false, italic = 0;
    if (base.t === 'glyph' && base.font !== F.Size1) {
      isChar = true;
      italic = base.font === F.Math || base.font === F.MathBold ? metric(base.font, base.ch)[2] * size : 0;
    }
    else if (base.t === 'glyph') italic = metric(base.font, base.ch)[2] * size;
    let u = isChar ? 0 : bb.h - SUP_DROP * ssize;
    let v = isChar ? 0 : bb.d + SUB_DROP * ssize;
    const sp = sup ? this.list(sup, level + 1) : null;
    const sb = sub ? this.list(sub, level + 1) : null;
    const xh = X_HEIGHT * size;
    if (sp && !sb) u = Math.max(u, SUP1 * size, sp.d + 0.25 * xh);
    else if (!sp && sb) v = Math.max(v, SUB1 * size, sb.h - 0.8 * xh);
    else if (sp && sb) {
      u = Math.max(u, SUP1 * size, sp.d + 0.25 * xh);
      v = Math.max(v, SUB2 * size);
      const th = RULE * size;
      if ((u - sp.d) - (sb.h - v) < 4.0 * th) {
        v = 4.0 * th - (u - sp.d) + sb.h;
        const psi = 0.8 * xh - (u - sp.d);
        if (psi > 0) { u += psi; v -= psi; }
      }
    }
    const b = { w: bb.w, h: bb.h, d: bb.d, items: [] };
    const bw = bb.w;
    put(b, bb, 0, 0);
    let extra = 0;
    if (sp) {
      b.h = Math.max(b.h, u + sp.h);
      extra = Math.max(extra, italic + sp.w);
      put(b, sp, bw + italic, -u);
    }
    if (sb) {
      b.d = Math.max(b.d, v + sb.d);
      extra = Math.max(extra, sb.w);
      put(b, sb, bw, v);
    }
    b.w = bw + extra + SCRIPT_SPACE * size;
    return b;
  }
  frac(num, den, display, level) {
    const size = this.size(level);
    const inner = display ? level : level + 1;
    const n = this.list(num, inner);
    const dn = this.list(den, inner);
    const th = RULE * size, a = AXIS * size;
    let u = display ? NUM1 * size : NUM2 * size;
    let v = display ? DENOM1 * size : DENOM2 * size;
    const phi = display ? 3.0 * th : th;
    let clr = (u - n.d) - (a + th / 2.0);
    if (clr < phi) u += phi - clr;
    clr = (a - th / 2.0) - (dn.h - v);
    if (clr < phi) v += phi - clr;
    const pad = NULL_DELIM * size;
    const w = Math.max(n.w, dn.w);
    const b = { w: w + 2.0 * pad, h: u + n.h, d: v + dn.d, items: [] };
    put(b, n, pad + (w - n.w) / 2.0, -u);
    put(b, dn, pad + (w - dn.w) / 2.0, v);
    b.items.push({ kind: 'rule', w, h: th, x: pad, y: -(a + th / 2.0) });
    return b;
  }
  sqrt(body, level) {
    const size = this.size(level);
    const bb = this.list(body, level);
    const th = RULE * size;
    const gap = 0.1 * size;
    const top = Math.max(bb.h, X_HEIGHT * size) + gap + th;
    const bottom = bb.d + 0.08 * size;
    const sw = 0.55 * size;
    const t = top + bottom;
    const y = (f) => -top + f * t;
    const end = sw + bb.w + 0.05 * size;
    const pts = [[0.0, y(0.62)], [0.16 * sw, y(0.55)], [0.45 * sw, bottom], [sw, -top + th / 2.0], [end, -top + th / 2.0]];
    const b = { w: end + 0.05 * size, h: top, d: bottom, items: [] };
    b.items.push({ kind: 'surd', pts, stroke: th * 1.25, x: 0, y: 0 });
    put(b, bb, sw, 0);
    return b;
  }
  accent(ch, body, level) {
    const size = this.size(level);
    const bb = this.list(body, level);
    const skew = body.length === 1 && body[0].t === 'glyph' ? metric(body[0].font, body[0].ch)[3] * size : 0;
    const b = { w: bb.w, h: bb.h, d: bb.d, items: [] };
    const bh = bb.h;
    put(b, bb, 0, 0);
    if (ch === '→') {
      const asz = 0.7 * size;
      const m = metric(F.Main, ch);
      const lift = bh + 0.08 * size + m[0] * asz;
      const g = this.glyph(ch, F.Main, asz);
      put(b, g, (b.w - g.w) / 2.0 + skew, -lift);
      b.h = Math.max(b.h, lift + m[1] * asz);
    }
    else {
      const m = metric(F.Main, ch);
      const raise = Math.max(bh - X_HEIGHT * size, 0);
      const g = this.glyph(ch, F.Main, size);
      put(b, g, (b.w - g.w) / 2.0 + skew, -raise);
      b.h = Math.max(b.h, raise + m[1] * size);
    }
    return b;
  }
  delimited(left, body, right, level) {
    const size = this.size(level);
    const bb = this.list(body, level);
    const a = AXIS * size;
    const need = 2.0 * Math.max(bb.h - a, bb.d + a);
    const b = hbox();
    const delim = (c) => {
      const m = metric(F.Main, c);
      const natural = (m[0] + m[1]) * size;
      const scale = natural > 0 ? Math.max(need / natural, 1.0) : 1.0;
      const ds = size * scale;
      const shift = (m[1] - m[0]) / 2.0 * ds - a;
      const g = this.glyph(c, F.Main, ds);
      const x = b.w;
      b.w += g.w;
      b.h = Math.max(b.h, m[1] * ds - shift);
      b.d = Math.max(b.d, m[0] * ds + shift);
      put(b, g, x, shift);
    };
    if (left !== null) delim(left); else b.w += NULL_DELIM * size;
    const x = b.w;
    b.w += bb.w;
    b.h = Math.max(b.h, bb.h);
    b.d = Math.max(b.d, bb.d);
    put(b, bb, x, 0);
    if (right !== null) delim(right); else b.w += NULL_DELIM * size;
    return b;
  }
}

// ---- the label ------------------------------------------------------------------

export function sansWidth(text, size) {
  let w = 0;
  for (const c of text) {
    if ("ilj.,:;'|! ".includes(c)) w += 0.27;
    else if ('ftrI()[]-/'.includes(c)) w += 0.34;
    else if (c === 'm' || c === 'w') w += 0.82;
    else if (c === 'M' || c === 'W') w += 0.88;
    else if (isDigit(c)) w += 0.57;
    else if (/^[A-Z]$/.test(c)) w += 0.66;
    else if (/^[a-z]$/.test(c)) w += 0.53;
    else w += 0.6;
  }
  return w * size;
}

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

/** A number as the SVG writes it: two decimals, and never -0.00. */
export function c(v) {
  const s = v.toFixed(2);
  return s === '-0.00' ? '0.00' : s;
}

function dollars(s) {
  const out = [];
  for (let k = 0; k < s.length; k++) {
    if (s[k] === '\\' && k + 1 < s.length) { k++; continue; }
    if (s[k] === '$') out.push(k);
  }
  return out;
}

/** Whether a label has math in it (an even, nonzero number of unescaped dollars). */
export function hasMath(s) {
  const n = dollars(Array.from(s || '')).length;
  return n > 0 && n % 2 === 0;
}

const unescapePlain = (s) => s.replace(/\\\$/g, '$');

export function plain(s, size) {
  return { inner: esc(s), shapes: [], w: sansWidth(s, size), h: 0.72 * size, d: 0.2 * size, math: false };
}

/** Lay out a label at a text size in px (throws MathError). */
export function mathLabel(s, size) {
  const cs = Array.from(s);
  const ds = dollars(cs);
  if (ds.length === 0 || ds.length % 2 === 1) {
    const t = s.includes('\\$') ? unescapePlain(s) : s;
    return { inner: esc(t), shapes: [], w: sansWidth(t, size), h: 0.72 * size, d: 0.2 * size, math: false };
  }
  const lay = new Lay(size * MATH_SCALE);
  const all = { items: [], w: 0, h: 0.72 * size, d: 0.2 * size };
  let at = 0;
  for (let k = 0; k <= ds.length; k += 2) {
    const end = k < ds.length ? ds[k] : cs.length;
    const next = k < ds.length ? ds[k + 1] : cs.length;
    const pl = cs.slice(at, end).join('');
    if (pl.length) {
      const t = unescapePlain(pl);
      const w = sansWidth(t, size);
      all.items.push({ kind: 'plain', text: t, w, x: all.w, y: 0 });
      all.w += w;
    }
    if (k === ds.length) break;
    const p = new Parser(cs.slice(end + 1, next), end + 1);
    const nodes = p.list(F.Math, false);
    if (p.i < p.s.length) throw p.err(p.i, '\\right with no \\left');
    const b = lay.list(nodes, 0);
    const x = all.w;
    all.w += b.w;
    all.h = Math.max(all.h, b.h);
    all.d = Math.max(all.d, b.d);
    put(all, b, x, 0);
    at = next + 1;
  }
  return { inner: tspans(all.items), shapes: all.items.filter((i) => i.kind === 'rule' || i.kind === 'surd'), w: all.w, h: all.h, d: all.d, math: true };
}

function fontAttrs(f) {
  const [family, style, weight] = [
    [MAIN_STACK, 'normal', 'normal'], [MAIN_STACK, 'normal', 'bold'], [MAIN_STACK, 'italic', 'normal'],
    [MATH_STACK, 'italic', 'normal'], [MATH_STACK, 'italic', 'bold'], [SIZE1_STACK, 'normal', 'normal'],
  ][f];
  return ` font-family="${family}" font-style="${style}" font-weight="${weight}"`;
}

function tspans(items) {
  let out = '';
  let px = 0, py = 0;
  let open = null; // { key, f, moves, text }
  const flush = () => {
    if (!open) return;
    const attrs = open.f ? `${open.moves}${fontAttrs(open.f[0])} font-size="${open.f[1]}"` : open.moves;
    out += `<tspan${attrs}>${esc(open.text)}</tspan>`;
    open = null;
  };
  for (const it of items) {
    let f;
    if (it.kind === 'glyph') f = [it.font, c(it.size)];
    else if (it.kind === 'plain') f = null;
    else continue;
    const dx = c(it.x - px);
    const dy = c(it.y - py);
    const moved = dx !== '0.00' || dy !== '0.00';
    const key = f ? f.join('|') : '';
    if (!moved && open && open.key === key) open.text += it.text;
    else {
      flush();
      let moves = '';
      if (dx !== '0.00') moves += ` dx="${dx}"`;
      if (dy !== '0.00') moves += ` dy="${dy}"`;
      open = { key, f, moves, text: it.text };
    }
    px += parseFloat(dx) + it.w;
    py += parseFloat(dy);
  }
  flush();
  return out;
}

/** The fraction bars and radicals of a label whose text is at (x, y), as SVG. */
export function shapesSvg(l, x, y, anchor, rotate, fill) {
  if (!l.shapes.length) return '';
  const x0 = anchor === 'middle' ? -l.w / 2 : anchor === 'end' ? -l.w : 0;
  const turn = rotate ? ` rotate(${c(rotate)})` : '';
  let s = `<g transform="translate(${c(x)} ${c(y)})${turn}">`;
  // a CSS color (var(--text-muted)) is a style; an SVG color an attribute
  const paint = (prop) => (fill.startsWith('var(') ? ` style="${prop}:${fill}"` : ` ${prop}="${fill}"`);
  for (const it of l.shapes) {
    if (it.kind === 'rule') s += `<rect x="${c(x0 + it.x)}" y="${c(it.y)}" width="${c(it.w)}" height="${c(it.h)}"${paint('fill')}/>`;
    else {
      const d = it.pts.map(([px, py], k) => `${k ? 'L' : 'M'}${c(x0 + it.x + px)} ${c(it.y + py)}`).join('');
      s += `<path d="${d}" fill="none"${paint('stroke')} stroke-width="${c(it.stroke)}" stroke-linecap="round" stroke-linejoin="round"/>`;
    }
  }
  return s + '</g>';
}

/** A label for a chart: laid out, or as plain text when its math has an error. */
export function chartLabel(s, size) {
  try {
    return mathLabel(s || '', size);
  }
  catch (e) {
    if (e instanceof MathError) return { ...plain(s, size), error: e.message };
    throw e;
  }
}
