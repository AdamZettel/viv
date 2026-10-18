// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The prose half of the claims database: a post's prose cut into sentences,
// each sentence given a stable id and a kind, and every number stated in it
// parsed with its precision, its units and the symbol it is about.
//
// Pure functions, no I/O, so scripts/claimstest.mjs can hold them to examples.
//
// A sentence's id is sha1 of its normalized text (whitespace collapsed,
// lowercased), cut to 12 hex characters, the rule of the earlier book
// pipeline (build_claims.mjs). Editing a sentence
// changes its id, so a verdict filed against the old text no longer applies
// and the sentence drops back to unchecked. Unlike the book, the math stays in
// the normalized text: in a post the numbers live in the math, and changing
// one must reset the verdict.
import { createHash } from 'node:crypto';

export const sentenceId = (text) => createHash('sha1').update(normalize(text)).digest('hex').slice(0, 12);
export const normalize = (text) => String(text).replace(/\s+/g, ' ').trim().toLowerCase();

// ---- spans: math, code, and the text between -------------------------------

const DOLLAR = '\u0001';
/** Cut markdown into [{ kind: 'text'|'math'|'display'|'code', s }]; `\$` is a dollar sign. */
export function spans(src) {
  const t = String(src).replace(/\\\$/g, DOLLAR);
  const out = [];
  const re = /\$\$([\s\S]+?)\$\$|\$([^$]+?)\$|`([^`]+)`/g;
  let at = 0;
  let m;
  while ((m = re.exec(t))) {
    if (m.index > at) out.push({ kind: 'text', s: t.slice(at, m.index) });
    if (m[1] !== undefined) out.push({ kind: 'display', s: m[1] });
    else if (m[2] !== undefined) out.push({ kind: 'math', s: m[2] });
    else out.push({ kind: 'code', s: m[3] });
    at = re.lastIndex;
  }
  if (at < t.length) out.push({ kind: 'text', s: t.slice(at) });
  return out.map((p) => ({ ...p, s: p.s.split(DOLLAR).join('$') }));
}

// ---- sentences ---------------------------------------------------------------

// words after which a period does not end a sentence
const ABBR = /(?:^|[\s(])(?:e\.g|i\.e|cf|etc|vs|Fig|Figs|Eq|Eqs|Sec|Ch|Ref|Refs|No|Vol|Phys|Rev|Rep|Lett|Mod|Comput|Chem|Math|Nucl|Astrophys|Proc|Ann|Trans|Soc|Acad|Sci|Natl|Commun|Numer|Appl|Stat|Comp|Proc|Mech|et al|approx|resp|min|max|Dr|Mr|Ms|Prof|St|Jr|Sr|[A-Z])\.$/;

/**
 * Cut a paragraph of markdown into sentences. Math and code spans are never
 * cut. A sentence ends at . ! or ? followed by space and something that can
 * start a sentence, unless the word before the period is an abbreviation or
 * an initial ("C. N. Yang", "Phys. Rev.").
 * Returns [{ text, start }] with `start` the offset in the paragraph.
 */
export function splitSentences(para) {
  const t = String(para);
  const out = [];
  let from = 0;
  let i = 0;
  const push = (end) => {
    const raw = t.slice(from, end);
    const lead = raw.length - raw.trimStart().length;
    const s = raw.trim();
    if (s) out.push({ text: s, start: from + lead });
    from = end;
  };
  while (i < t.length) {
    const c = t[i];
    if (c === '\\' && t[i + 1] === '$') { i += 2; continue; }
    if (c === '$') {
      const dbl = t[i + 1] === '$';
      const close = t.indexOf(dbl ? '$$' : '$', i + (dbl ? 2 : 1));
      if (close < 0) { i = t.length; break; }
      i = close + (dbl ? 2 : 1);
      continue;
    }
    if (c === '`') {
      const close = t.indexOf('`', i + 1);
      i = close < 0 ? t.length : close + 1;
      continue;
    }
    if (c === '.' || c === '!' || c === '?') {
      let j = i + 1;
      while (j < t.length && /["')\]*_]/.test(t[j])) j++;
      if (j < t.length && /\s/.test(t[j])) {
        let k = j;
        while (k < t.length && /\s/.test(t[k])) k++;
        const next = t[k];
        const before = t.slice(Math.max(from, i - 24), i + 1);
        if (next && /[A-Z0-9$`(*"'\[\\]/.test(next) && !(c === '.' && ABBR.test(before))) {
          push(j);
          i = k;
          continue;
        }
      }
    }
    i++;
  }
  push(t.length);
  return out;
}

/**
 * Markdown prose (a header's body, a prose cell, a run of `#|` lines) into
 * paragraphs: { text, line, section } with `line` the 1-based line of the
 * paragraph's first line relative to `firstLine`, and a `lineAt(offset)` to
 * place a sentence inside it. Headings set the section and are not claims;
 * a `[figure]` mark is skipped; list items are paragraphs of their own;
 * display math stays inside its paragraph.
 */
export function paragraphs(body, firstLine = 1) {
  const lines = String(body).split('\n');
  const out = [];
  let section = '';
  let cur = null;
  let inDisplay = false;
  let inFence = false;
  const flush = () => { if (cur) out.push(cur); cur = null; };
  lines.forEach((raw, n) => {
    const line = raw.trim();
    const ln = firstLine + n;
    if (/^```/.test(line)) { flush(); inFence = !inFence; return; }
    if (inFence) return;
    if (!inDisplay && !line) { flush(); return; }
    if (!inDisplay && /^#{1,6}\s/.test(line)) { flush(); section = line.replace(/^#+\s*/, ''); return; }
    if (!inDisplay && /^\[figure\]$/i.test(line)) { flush(); return; }
    const item = !inDisplay && /^(?:[-*+]|\d+\.)\s+/.test(line);
    if (item) flush();
    const text = item ? line.replace(/^(?:[-*+]|\d+\.)\s+/, '') : line;
    if (!cur) cur = { text: '', section, line: ln, offsets: [] };
    if (cur.text) cur.text += ' ';
    cur.offsets.push([cur.text.length, ln]);
    cur.text += text;
    // a $$ block may span lines, and blank lines inside it do not end the paragraph
    const dd = (text.match(/\$\$/g) || []).length;
    if (dd % 2 === 1) inDisplay = !inDisplay;
  });
  flush();
  for (const p of out) {
    const offs = p.offsets;
    p.lineAt = (off) => { let l = offs[0][1]; for (const [o, ln] of offs) { if (o <= off) l = ln; else break; } return l; };
    delete p.offsets;
  }
  return out;
}

// ---- kinds of sentence ------------------------------------------------------

const PROGRAM_NOUN = /\b(the|this|each|every|that|a|its|our|next|above|below|second|first|last)\s+(program|cell|cells|code|test|tests|run|script|reference|function|loop|plot|figure|post|solver|search|sweep|scan|check|checks|step|steps|implementation|module|notebook|rung|ladder|version|card|page|grid|panel)s?\b/i;
const PROGRAM_VERB = /\b(runs?|computes?|draws?|checks?|compares?|prints?|shows?|takes?|repeats?|scans?|finds?|stops?|starts?|uses?|calls?|returns?|builds?|writes?|reads?|plots?|keeps?|measures?|records?|asserts?|integrates?|solves?|sets?|holds?|tests?|steps?|iterates?|refines?|samples?|counts?|makes?|adds?|multiplies|divides?|stores?|passes|pass|fits?|follows?|marks?)\b/i;
const IMPERATIVE = /^(Make|Copy|Return|Compute|Fill|Build|Solve|Find|Draw|Plot|Apply|Give|Store|Write|Keep|Stop|Step|Scale|Sum|Multiply|Divide|Subtract|Integrate|Evaluate|Advance|Update|Normalize|Sort|Count|Check|Test|Visit|Take|Repeat|Chop|Put|Start|Run|Compare|Press|Watch|Look|Try|Change|Set|Edit|Note|Read|See|Pick|Choose|Imagine|Suppose|Consider|Replace|Halve|Double|Add|Drop|Move|Open|Click)\b/;
const DEFINITION = /\b(is|are) (called|named|known as|defined as|the name for)\b|\bwe (call|write|denote|define)\b|\b(means|stands for|is short for|is shorthand for)\b|^(?:an?|the)\s+\*[^*]+\*\s+(is|are)\b|\bby \*[^*]+\*\b|\b(is|are|called|gives|give) (?:an?|the)?\s*\*[^*]+\*|\*[^*]+\*,? (is|are|means)\b/i;
const RHETORICAL = /^(Before (running|looking|reading|you run|pressing)|Guess|Predict|Say you|Suppose|Imagine)\b/;

/**
 * What a sentence claims. Given the numbers already parsed from it.
 *   rhetorical  a question, a prompt to guess, a transition: nothing to check
 *   numeric     states a number (a value or a count) that can be linked
 *   definition  says what a word means
 *   procedural  says what the program or the reader does
 *   factual     the rest: a statement about the world, checkable by reference
 */
export function classify(text, numbers = []) {
  const plain = String(text).replace(/\$[^$]*\$/g, ' X ').replace(/`[^`]*`/g, ' X ');
  const words = plain.split(/\s+/).filter((w) => /[a-z]/i.test(w));
  if (/\?\s*["')*]*$/.test(text) || RHETORICAL.test(text.trim()) || words.length < 4) return 'rhetorical';
  // a displayed formula on its own: it defines a quantity
  if (/^\$\$[\s\S]*\$\$[.,;]?$/.test(String(text).trim())) return 'definition';
  if (DEFINITION.test(text)) return 'definition';
  if (numbers.some((n) => n.role === 'value' || n.role === 'count')) return 'numeric';
  if ((PROGRAM_NOUN.test(plain) && PROGRAM_VERB.test(plain)) || IMPERATIVE.test(text.trim())) return 'procedural';
  return 'factual';
}

// ---- numbers -----------------------------------------------------------------

const SUP = { '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', '⁻': '-', '⁺': '+' };
// superscripts become a TeX exponent: 10⁻¹⁵ -> 10^{-15}
const deSup = (s) => s.replace(/[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+/g, (run) => `^{${[...run].map((c) => SUP[c]).join('')}}`);
const MINUS = /[−–]/g;

/**
 * A number's literal as written -> { value, ulp } where `ulp` is the place of
 * its last stated digit: "0.577" -> 0.001, "3.1e-14" -> 1e-15, "1200" -> 1,
 * "10^{-13}" -> 0 (an exact power of ten). A value v matches the literal when
 * it rounds to it: |v - value| <= ulp / 2.
 */
export function literal(mant, exp = 0) {
  const m = String(mant).replace(MINUS, '-').replace(/,/g, '').replace(/\\,/g, '');
  const e = Number(exp) || 0;
  const [, sign, int, frac, e2] = m.match(/^([-+]?)(\d*)(?:\.(\d*))?(?:[eE]([-+]?\d+))?$/) || [];
  if (int === undefined && frac === undefined) return null;
  const ee = e + (Number(e2) || 0);
  const decimals = (frac || '').length;
  // "57,000" and "1200" say nothing about the digits their trailing zeros hold
  // (only for large ones written that way: "20" is twenty, not "about 20")
  const tz = frac === undefined && !e2 && int && /[1-9]/.test(int) && (/,/.test(String(mant)) || int.length >= 5) ? int.match(/0*$/)[0].length : 0;
  const value = Number(`${sign}${int || '0'}.${frac || '0'}`) * 10 ** ee;
  const ulp = 10 ** (ee - decimals + tz);
  return { value: Number(value.toPrecision(15)), ulp: Number(ulp.toPrecision(3)), digits: sigDigits(int, frac) };
}
function sigDigits(int = '', frac = '') {
  const all = (int + frac).replace(/^0+/, '');
  return Math.max(1, all.length);
}

// units a number may carry, spelled as they are in the posts
const UNIT_WORDS = ['%', 'percent', 'per cent', 'eV', 'keV', 'MeV', 'GeV', 'meV', 'Hz', 'kHz', 'MHz', 'GHz', 'THz', 'ms', 'µs', 'μs', 'ns', 'ps', 'fs', 's', 'sec', 'seconds', 'second', 'minutes', 'minute', 'hours', 'hour', 'days', 'day', 'years', 'year', 'months', 'K', 'kelvin', 'mK', 'm', 'cm', 'mm', 'km', 'fm', 'nm', 'Å', 'bohr', 'hartree', 'hartrees', 'Ha', 'Eh', 'Ry', 'kg', 'g', 'suns', 'sun', 'solar masses', 'M_sun', 'km/s', 'm/s', 'digits', 'digit', 'bits', 'bit', 'bytes', 'KB', 'MB', 'GB', 'kB', 'times', 'x', '×', 'degrees', '°', 'rad', 'radians', 'T', 'tesla', 'G', 'gauss', 'V', 'volts', 'A', 'amps', 'W', 'kW', 'MW', 'J', 'erg', 'ohm', 'Ω', 'F', 'H', 'dB', 'cm^2/Vs', 'fm^-3', 'g/cm^3', 'MeV/fm^3', 'dollars'];
const UNIT_RE = new RegExp(`^\\s*(${UNIT_WORDS.map((u) => u.replace(/[.*+?^${}()|[\]\\/]/g, '\\$&')).sort((a, b) => b.length - a.length).join('|')})(?![A-Za-z0-9])`);
const LABEL_BEFORE = /\b(rung|step|cell|figure|fig\.|table|chapter|section|part|lecture|exercise|problem|course|version|v|row|column|day|page|line|lines|item|case|test|stage|phase|level|order|rank|appendix|eq\.|equation|number|no\.|#)\s*$/i;
const APPROX_BEFORE = /\b(about|roughly|approximately|nearly|almost|around|some|close to|of order|order|near)\s*$|[~≈]\s*$/i;
const UPPER_BEFORE = /\b(under|below|less than|smaller than|at most|within|to within|up to|no more than|no larger than|below|beneath|fewer than|not exceed)\s*$|<\s*$|≤\s*$/i;
const LOWER_BEFORE = /\b(over|above|more than|greater than|larger than|at least|exceeds?|beyond)\s*$|>\s*$|≥\s*$/i;

/** A text span's numbers, with role and qualifier. */
function textNumbers(s, offset) {
  const out = [];
  let t = deSup(s);
  const orig = t;
  // powers of ten first: "3.1 x 10^-14", "10^{-15}"; then they are blanked so
  // their digits are not read again as numbers
  const pw = /(?<![\w.^_])(?:([-−]?\d+(?:\.\d+)?)\s*[×x]\s*)?10\s*\^\s*\{?\s*([-+−]?\d+)\s*\}?/g;
  let q;
  while ((q = pw.exec(orig))) {
    const before = orig.slice(0, q.index);
    const e = q[2].replace(MINUS, '-');
    const lit = q[1] ? literal(q[1], e) : { value: Number((10 ** Number(e)).toPrecision(15)), ulp: 0, digits: 1 };
    const qualifier = APPROX_BEFORE.test(before) ? 'approx' : UPPER_BEFORE.test(before) || boundTo(before, lit) ? 'upper' : LOWER_BEFORE.test(before) ? 'lower' : '';
    out.push({ raw: q[0], ...lit, unit: '', percent: false, role: 'value', qualifier, symbol: '', where: 'text', ...(q[1] ? {} : { pow10: true }), at: offset + q.index });
  }
  t = t.replace(pw, (x) => '#'.repeat(x.length)).replace(/\^\s*\{[^}]*\}|\^\s*\([^)]*\)|\^-?\d+(?:\.\d+)?|_\{[^}]*\}/g, (x) => '#'.repeat(x.length));
  // mantissa x 10^e written out: "3.1 × 10^-14", "3.1 x 10-14" from superscripts
  const re = /(?<![\w.^_]|\b[A-Z][A-Za-z]*-)([-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+−]?\d+)?)(?:\s*[×x]\s*10\^?\{?([-+−]?\d+)\}?)?(\s*\/\s*(\d{1,3})(?![\d.]))?(?![\w.]\d|[A-Za-z]|\d|-\d+[A-Za-z])/g;
  // dates and version numbers are not claims: blank them first
  const blank = (x) => x.replace(/[0-9]/g, '#');
  const t0 = t;
  // and a citation's volume, page and year: "Rev. Mod. Phys. 81, 109 (2009)", "Eq. (3.12)", arXiv ids
  t = t.replace(/((?:Phys\.|Rev\.|Rep\.|J\.|Nucl\.|Astrophys\.|ApJ|Comput\.|Math\.|Proc\.|Ann\.|Nature|Science|SIAM|Commun\.|Lett\.)[A-Za-z. ]*\s)(\d+,\s*\d+(?:\s*\(\d{4}\)|,\s*\d{4})?)/g, (x, a, b) => a + blank(b))
    .replace(/\b(Eqs?\.|Sec\.|Fig\.|Table|Ch\.)\s*\(?[\d.]+\)?(?:\s*[-–]\s*\(?[\d.]+\)?)?/g, blank).replace(/arXiv:\s*[\w./-]+/g, blank);
  t = t.replace(/\b\d{4}-\d{2}-\d{2}\b/g, blank).replace(/\b(v|version |PySCF |Python |numpy |scipy |Viv |astro |node )(\d+(?:\.\d+){1,3})\b/gi, (x, a, b) => a + blank(b)).replace(/\b\d+\.\d+\.\d+\b/g, blank);
  let m;
  while ((m = re.exec(t))) {
    const raw = t0.slice(m.index, m.index + m[0].length);
    const before = t.slice(0, m.index);
    const after = t.slice(m.index + raw.length);
    // 10^-13 written without the mantissa
    let lit;
    if (m[4]) {
      // a fraction a/b
      const a = Number(m[1]);
      const b = Number(m[4]);
      if (!b || /\d{4}-\d{2}/.test(raw)) continue;
      lit = { value: a / b, ulp: 0, digits: 3 };
    } else if (/^10$/.test(m[1]) && /^\^/.test(after)) continue;
    else lit = literal(m[1], m[2] ? m[2].replace(MINUS, '-') : 0);
    if (!lit) continue;
    // a date, a version, a hyphenated range of years: not a claim
    if (/^\d{4}-\d{2}-\d{2}/.test(t.slice(m.index))) continue;
    const unit = (after.match(UNIT_RE) || [])[1] || '';
    const percent = unit === '%' || unit === 'percent' || unit === 'per cent';
    let role = 'value';
    const int = Number.isInteger(lit.value) && !/\./.test(m[1]);
    if (int && lit.value >= 1500 && lit.value <= 2099 && !unit && !m[2]) role = 'year';
    else if (LABEL_BEFORE.test(before)) role = 'label';
    // a bit pattern: "count from 000 to 111", "the low four bits are 0010" --
    // only 0s and 1s, beside a zero-padded pattern of the same width
    else if (/^[01]{2,}$/.test(m[1]) && new RegExp(`(?<![\\w.])0[01]{${m[1].length - 1}}(?![\\w.])`).test(t0)) role = 'label';
    // a small integer that is an operand in a formula written as text ("1 +
    // exp(...)", "theta/2", "count + 1", "2 pi", "1 < omega < 2") is part of
    // the formula, as it is in a math span; "= 1", "(3, 2)" and "5-point" still count
    else if (int && Math.abs(lit.value) <= 12 && !unit && (/(?:[+*/^<>]|(?<![A-Za-z])-)\s*$/.test(before) || /^\s*(?:[+*/^<>]|-(?![A-Za-z])|pi\b|π)/.test(after))) role = 'expression';
    else if (int && Math.abs(lit.value) <= 12 && !unit) role = 'count';
    const qualifier = APPROX_BEFORE.test(before) ? 'approx' : UPPER_BEFORE.test(before) ? 'upper' : LOWER_BEFORE.test(before) ? 'lower' : boundTo(before, lit) ? 'upper' : '';
    out.push({ raw: raw.trim(), ...lit, unit, percent, role, qualifier, symbol: '', where: 'text', at: offset + m.index });
  }
  out.sort((a, b) => a.at - b.at);
  return out;
}

// "agree to 1e-13", "to 10^{-15}": `to` before a small number is a bound, not a range
function boundTo(before, lit) {
  if (!/\bto\s*(?:within\s*)?$/i.test(before)) return false;
  if (/\d\s*(?:\$\s*)?to\s*$/i.test(before) || /\bfrom\b[^.]{0,40}$/i.test(before)) return false;
  return Math.abs(lit.value) > 0 && Math.abs(lit.value) <= 1e-2;
}

// relations that separate a math span into the parts that can be claims
const REL = /(\\approx|\\simeq|\\sim|\\lesssim|\\gtrsim|\\leq?|\\geq?|\\ll|\\gg|\\pm|\\mp|\\to|\\rightarrow|\\quad|\\qquad|<|>|=|≈|,|;)/;
const REL_QUAL = { '\\approx': 'approx', '\\simeq': 'approx', '\\sim': 'approx', '≈': 'approx', '\\lesssim': 'upper', '<': 'upper', '\\le': 'upper', '\\leq': 'upper', '\\ll': 'upper', '>': 'lower', '\\ge': 'lower', '\\geq': 'lower', '\\gg': 'lower', '\\gtrsim': 'lower', '=': 'exact' };

/**
 * A math span's claims. The span is cut at relations (=, \approx, <, \pm,
 * commas); a piece that is only a number (with a sign, a unit in \text or
 * \mathrm, a percent sign, a power of ten) is a claim, with the piece before
 * the relation as its symbol. A number inside a formula -- 2 in 2/\ln(1 +
 * \sqrt 2), a subscript, an exponent -- is part of the formula, not a claim.
 */
function mathNumbers(tex, offset) {
  const out = [];
  let s = String(tex)
    .replace(/\\(left|right|big|Big|bigg|Bigg)\b/g, '')
    .replace(/\\[,;:! ]/g, ' ')
    .replace(/\{,\}/g, '')             // 1{,}000
    .replace(/\\%/g, '%')
    .replace(/\\(text|mathrm|textrm|mbox|operatorname)\s*\{([^}]*)\}/g, (_, c, x) => (/^\s*(and|or|for|at|if|with|where|to|from)\s*$/.test(x) ? ' , ' : ` ${x} `));
  // numbers become tokens ⟦k⟧, so a piece can be tested for being one
  const toks = [];
  const tok = (lit, raw) => { toks.push({ ...lit, raw }); return ` ⟦${toks.length - 1}⟧ `; };
  s = s.replace(/(\d+(?:\.\d+)?)\s*(?:\\times|\\cdot|×)\s*10\s*\^\s*\{?\s*([-+−]?\d+)\s*\}?/g, (raw, a, e) => tok(literal(a, e.replace(MINUS, '-')), raw));
  s = s.replace(/(?<![\w.^_\\])10\s*\^\s*\{?\s*([-+−]?\d+)\s*\}?/g, (raw, e) => tok({ value: Number((10 ** Number(e.replace(MINUS, '-'))).toPrecision(15)), ulp: 0, digits: 1, pow10: true }, raw));
  s = s.replace(/\\[td]?frac\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}|\\[td]?frac\s*(\d)(\d)/g, (raw, a, b, c, d) => tok({ value: Number(a || c) / Number(b || d), ulp: 0, digits: 3, expr: true }, raw));
  s = s.replace(/(?<![\w.^_\\}])(\d+)\s*\^\s*\{?\s*([-−]?\d+)\s*\}?/g, (raw, a, e) => tok({ value: Number((Number(a) ** Number(e.replace(MINUS, '-'))).toPrecision(15)), ulp: 0, digits: 3, expr: true }, raw));
  s = s.replace(/(?<![\w.^_\\⟦])(\d+)\s*\/\s*(\d+)(?![\w.])/g, (raw, a, b) => (Number(b) ? tok({ value: Number(a) / Number(b), ulp: 0, digits: 3, expr: true }, raw) : raw));
  s = s.replace(/(?<![\w.^_\\⟦])(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![\w.⟧])/g, (raw, a) => tok(literal(a), raw));
  // pieces between relations
  const parts = s.split(REL);
  for (let i = 0; i < parts.length; i += 2) {
    const piece = parts[i];
    const ids = [...piece.matchAll(/⟦(\d+)⟧/g)].map((x) => Number(x[1]));
    if (ids.length !== 1) continue;
    const rest = piece.replace(/⟦\d+⟧/, '').replace(/[\s{}]/g, '');
    const neg = /^[-−]/.test(rest);
    const tail = rest.replace(/^[-−+]/, '');
    const unitM = tail.match(new RegExp(`^(%|${UNIT_WORDS.filter((u) => u.length > 1 || /[a-zA-Z]/.test(u)).map((u) => u.replace(/[.*+?^${}()|[\]\\/]/g, '\\$&')).join('|')})$`));
    if (tail && !unitM) continue;
    const t = toks[ids[0]];
    const relBefore = parts[i - 1] || '';
    const relAfter = parts[i + 1] || '';
    let qualifier = REL_QUAL[relBefore] || '';
    if (qualifier === 'exact') qualifier = '';
    const symbol = i >= 2 && /^(=|\\approx|\\simeq|\\sim|<|>|\\le|\\leq|\\ge|\\geq|≈|\\lesssim)$/.test(relBefore) ? parts[i - 2].replace(/⟦(\d+)⟧/g, (_, k) => toks[k].raw).trim() : '';
    // an expression set equal to a number ("100/7 = 14.3", "2^{10} = 1024")
    // is the arithmetic, not a claim: the number after it is the claim, and
    // the linker evaluates the expression to check it
    const lone = (k) => k < parts.length && [...parts[k].matchAll(/⟦(\d+)⟧/g)].length === 1 && !parts[k].replace(/⟦\d+⟧/, '').replace(/[\s{}]/g, '').replace(/^[-−+]/, '');
    const isExpr = t.expr && (i === 0 || !/^(=|\\approx|\\simeq|\\sim|≈|<|>|\\le|\\leq|\\ge|\\geq)$/.test(relBefore)) && /^(=|\\approx|\\simeq|≈)$/.test(relAfter) && lone(i + 2);
    const role = isExpr ? 'expression' : relBefore === '\\pm' || relBefore === '\\mp' ? 'uncertainty' : Number.isInteger(t.value) && Math.abs(t.value) <= 12 && !unitM && t.ulp !== 0 ? 'count' : 'value';
    const unit = unitM ? unitM[1] : '';
    out.push({ raw: `${neg ? '-' : ''}${t.raw.trim()}`, value: neg ? -t.value : t.value, ulp: t.ulp, digits: t.digits, unit, percent: unit === '%', role, qualifier, symbol: symbol.length <= 40 ? symbol : '', where: 'math', ...(t.pow10 ? { pow10: true } : {}), at: offset, next: relAfter });
  }
  return out;
}

/**
 * Every number stated in a sentence of markdown, in order:
 *   { raw, value, ulp, digits, unit, percent, role, qualifier, symbol, where, context }
 * role: value | count | uncertainty | year | label | expression (the left side of
 *       "100/7 = 14.3")  (formula constants are not returned)
 * qualifier: '' | approx | upper | lower
 * Numbers in `code` spans are not claims; the identifiers there are symbol hints
 * and come back in `names`.
 */
export function numbersIn(sentence) {
  const out = [];
  const names = [];
  let at = 0;
  let prevText = '';
  for (const p of spans(sentence)) {
    if (p.kind === 'text') {
      out.push(...textNumbers(p.s, at));
      prevText = p.s;
    } else if (p.kind === 'math' || p.kind === 'display') {
      const nums = mathNumbers(p.s, at);
      // text before the math qualifies its first number: "to within $10^{-13}$"
      if (nums.length && !nums[0].qualifier && !nums[0].symbol) {
        const lit = nums[0];
        if (APPROX_BEFORE.test(prevText)) lit.qualifier = 'approx';
        else if (UPPER_BEFORE.test(prevText) || boundTo(prevText, lit)) lit.qualifier = 'upper';
        else if (LOWER_BEFORE.test(prevText)) lit.qualifier = 'lower';
      }
      // a unit or percent right after the math: "$0.94$%"
      out.push(...nums);
      prevText = '';
    } else {
      for (const id of p.s.matchAll(/[A-Za-z_][A-Za-z0-9_]*/g)) names.push(id[0]);
      prevText = '';
    }
    at += p.s.length + (p.kind === 'text' ? 0 : p.kind === 'display' ? 4 : 2);
  }
  const plain = String(sentence);
  for (const n of out) {
    n.context = plain.slice(Math.max(0, n.at - 50), n.at + 60).replace(/\s+/g, ' ').trim();
    n.before = plain.slice(Math.max(0, n.at - 60), n.at).replace(/\s+/g, ' ').trimStart();
    n.after = plain.slice(n.at, n.at + 24).replace(/\s+/g, ' ').trimEnd();
    delete n.at;
    delete n.next;
  }
  return { numbers: out, names: [...new Set(names)] };
}

// ---- citations -----------------------------------------------------------------

const KNOWN = [
  ['NIST', 'dataset'], ['CODATA', 'dataset'], ['IAU', 'dataset'], ['NRL Plasma Formulary', 'book'], ['NRL', 'book'],
  ['DLMF', 'book'], ['Abramowitz and Stegun', 'book'], ['Numerical Recipes', 'book'], ['Golub and Van Loan', 'book'],
  ['Trefethen', 'book'], ['Higham', 'book'], ['Griffiths', 'book'], ['Jackson', 'book'], ['Szabo and Ostlund', 'book'],
  ['PySCF', 'software'], ['scipy', 'software'], ['SciPy', 'software'], ['numpy', 'software'], ['NumPy', 'software'], ['mpmath', 'software'],
  ['LAPACK', 'software'], ['MINPACK', 'software'], ['libxc', 'software'], ['sympy', 'software'], ['SymPy', 'software'],
];
/**
 * Papers, books and datasets a piece of text names: "Takahasi and Mori (1974)",
 * "(Landau, 1946)", "Narain et al. 2006", "Rev. Mod. Phys. 81, 109 (2009)",
 * "CODATA 2018", NIST. Returns [{ text, kind: paper|book|dataset|software, year }].
 */
export function citationsIn(text) {
  const t = String(text).replace(/\$[^$]*\$/g, ' ');
  const out = [];
  const add = (s, kind, year) => { if (!out.some((c) => c.text === s)) out.push({ text: s, kind, year: year ? Number(year) : null }); };
  const NAME = "[A-Z][a-zA-Z'\\u00C0-\\u017F-]+";
  const re1 = new RegExp(`(${NAME}(?:(?:,| and| &) ${NAME})*(?: et al\\.?)?),? \\(?((?:1[6-9]|20)\\d\\d)[a-z]?\\)?`, 'g');
  let m;
  while ((m = re1.exec(t))) {
    const who = m[1];
    if (/^(In|By|At|From|Since|Until|Before|After|The|A|An|Of|On|For|To|And|Or|Its|This|That|Every|Each|Figure|Table|Step|Rung|Cell|Units|Around|About|During|Only|Up|Over|Under)$/.test(who)) continue;
    add(`${who} (${m[2]})`, 'paper', m[2]);
  }
  const re2 = new RegExp(`\\((${NAME}(?: et al\\.)?), ((?:1[6-9]|20)\\d\\d)\\)`, 'g');
  while ((m = re2.exec(t))) add(`${m[1]} (${m[2]})`, 'paper', m[2]);
  const re3 = /((?:Phys\.|Rev\.|J\.|Nucl\.|Astrophys\.|ApJ|Comput\.|Math\.|Proc\.|Ann\.|Nature|Science|SIAM)[A-Za-z. ]*\s\d+,\s*\d+(?:\s*\((\d{4})\)|,\s*(\d{4})))/g;
  while ((m = re3.exec(t))) add(m[1].trim(), 'paper', m[2] || m[3]);
  const re4 = /arXiv:\s*\S+|doi:\s*\S+|https?:\/\/doi\.org\/\S+/gi;
  while ((m = re4.exec(t))) add(m[0], 'paper', null);
  for (const [name, kind] of KNOWN) {
    const r = new RegExp(`(?<![A-Za-z])${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![A-Za-z])(?:'s)?(?: (\\d{4}))?`);
    const k = t.match(r);
    if (k) add(k[1] ? `${name} ${k[1]}` : name, kind, k[1] || null);
  }
  return out;
}

// ---- arithmetic the prose writes out ------------------------------------------

const TEX_FN = { ln: Math.log, sqrt: Math.sqrt, exp: Math.exp, sin: Math.sin, cos: Math.cos, tan: Math.tan, log_2: Math.log2, log_10: Math.log10 };
/**
 * The value of a formula the prose writes out in TeX, when it has one: numbers,
 * + - × · / ^, \frac, \tfrac12, \sqrt, \ln, \exp, \pi, e^{...}, parentheses and
 * braces, and implicit products ("2\pi"). Anything else -- a letter standing
 * for a quantity, an integral, a sum -- and there is no value: null. So
 * "\ln 2 / \ln 1.07" is 10.2447 and "\pi^2/6" is 1.64493, while "x^2" is null.
 */
export function texEval(tex) {
  const src = String(tex)
    .replace(/\\(left|right|big|Big|bigg|Bigg|displaystyle)\b/g, '')
    .replace(/\\[,;:! ]/g, ' ')
    .replace(/\{,\}/g, '')
    .replace(/\\log_\{?(2|10)\}?/g, '\\log_$1')
    .trim();
  if (!src || /[a-zA-Z]/.test(src.replace(/\\[a-zA-Z_0-9]+/g, '').replace(/\be\^/g, ''))) return null;
  let i = 0;
  const ws = () => { while (/\s/.test(src[i] || '')) i++; };
  const fail = () => { throw new Error('no value'); };
  const word = () => { const m = src.slice(i).match(/^\\([a-zA-Z]+(?:_(?:2|10))?)/); return m ? m[1] : null; };
  function group() {
    ws();
    if (src[i] === '{') { i++; const v = sum(); ws(); if (src[i] !== '}') fail(); i++; return v; }
    // an undelimited argument: one digit after \frac or \tfrac, else a primary
    return primary(true);
  }
  function primary(single = false) {
    ws();
    const c = src[i];
    if (c === '(' || c === '[') { i++; const v = sum(); ws(); if (src[i] !== ')' && src[i] !== ']') fail(); i++; return v; }
    if (c === '{') { i++; const v = sum(); ws(); if (src[i] !== '}') fail(); i++; return v; }
    if (c === '-' || c === '−') { i++; return -power(); }
    if (c === '+') { i++; return power(); }
    const num = src.slice(i).match(single ? /^\d/ : /^(\d+\.?\d*|\.\d+)/);
    if (num) { i += num[0].length; return Number(num[0]); }
    if (c === 'e' && src[i + 1] === '^') { i++; return Math.E; }
    const w = word();
    if (!w) fail();
    i += w.length + 1;
    if (w === 'pi') return Math.PI;
    if (w === 'frac' || w === 'tfrac' || w === 'dfrac') { const a = group(); const b = group(); return a / b; }
    if (TEX_FN[w]) { ws(); return TEX_FN[w](src[i] === '{' || src[i] === '(' ? primary() : power()); }
    return fail();
  }
  function power() {
    const b = primary();
    ws();
    if (src[i] === '^') { i++; ws(); const e = src[i] === '{' ? group() : primary(true); return b ** e; }
    return b;
  }
  function product() {
    let v = power();
    for (;;) {
      ws();
      const c = src[i];
      if (c === '*' || c === '×' || c === '·') { i++; v *= power(); continue; }
      if (c === '/') { i++; v /= power(); continue; }
      const w = word();
      if (w === 'times' || w === 'cdot') { i += w.length + 1; v *= power(); continue; }
      // an implicit product: "2\pi", "3(1 + x)"
      if (c === '(' || c === '{' || (w && (w === 'pi' || w === 'sqrt' || w === 'ln' || w === 'frac' || w === 'tfrac' || w === 'exp'))) { v *= power(); continue; }
      return v;
    }
  }
  function sum() {
    let v = product();
    for (;;) {
      ws();
      const c = src[i];
      if (c === '+') { i++; v += product(); } else if (c === '-' || c === '−') { i++; v -= product(); } else return v;
    }
  }
  try {
    const v = sum();
    ws();
    if (i < src.length) return null;
    return Number.isFinite(v) ? v : null;
  } catch { return null; }
}
