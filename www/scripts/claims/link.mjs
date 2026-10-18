// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The linker: every number a post's prose states, tied to what establishes it.
//
//   checked    a passing test expects it (its expected value, tolerance or
//              bound), or it is a value the run prints and a passing test
//              asserts on, or it is the value of the arithmetic the sentence
//              writes out beside it ("$\ln 2 \approx 0.693$", "$100/7 = 14.3$"),
//              which the linker evaluates
//   computed   the run prints it (`show` or print), but no test asserts it
//   reference  the post's reference script printed it (posts/refs/*.out)
//   input      the program is given it: a constant or a literal in the code
//   cited      nothing in the program has it, and the sentence cites a source
//   orphan     nothing establishes it
//   mismatch   the prose disagrees with what the program computes
//
// A number matches a value when the value rounds to it at the precision the
// prose states: "0.577" matches 0.5767095 (|v - 0.577| <= 0.0005), and a value
// printed to six digits carries its own half-digit of slack. A percentage is
// tried as a fraction too ("6.33%" matches 0.0632976).
//
// A mismatch needs evidence that the number is ABOUT a particular computed
// quantity, since a number that matches nothing may simply be about something
// the program does not compute:
//   - a `name` in backticks after the number ("|m| = 0.14 (`m_T`)") ties the
//     numbers before it, back to the previous name, to that variable;
//   - a test whose name shares the sentence's words ties the sentence's
//     numbers to that test's expected values and to the variables it asserts;
//   - a bound ("to within 1e-15", "under 1%") is a claim about the values it
//     bounds: it is a mismatch when no associated value satisfies it.
// A number near a computed value (within 1%) with none of that evidence is an
// orphan with a `suspect` note: worth a look, not a failure.
import { spans, texEval } from './text.mjs';

export const STATUSES = ['checked', 'computed', 'reference', 'input', 'cited', 'orphan', 'mismatch'];
const RANK = { checked: 0, computed: 1, reference: 2, input: 3 };

const STOP = new Set('the a an and or of to in at on by for with from as is are be was were it its this that these those which what when where than then there here each every one two three four five six seven eight nine ten not no but so if into over under about after before between both same other more most less least very only also just like per its their them they we our you your has have had does do did can could will would should may might must'.split(' '));
const stem = (w) => w.replace(/ies$/, 'y').replace(/(?<=[a-z]{3})(es|s)$/, '');
/** The words and numbers of a text, for matching a sentence to a test's name. */
export function tokens(text) {
  const t = String(text).toLowerCase().replace(/[`$\\{}()]/g, ' ');
  const out = new Set();
  for (const w of t.split(/[^a-z0-9.+_-]+/)) {
    const x = w.replace(/^[.+-]+|[.+-]+$/g, '');
    if (!x) continue;
    if (/^[-+]?\d/.test(x)) { const v = Number(x); if (Number.isFinite(v) && !(Number.isInteger(v) && Math.abs(v) <= 12)) out.add(`#${v}`); continue; }
    if (x.length < 3 || STOP.has(x)) continue;
    out.add(stem(x));
  }
  return out;
}

/** Does the prose number n (value, ulp, percent) match a candidate value? */
const printedPct = (c) => c.src === 'shown' || c.src === 'printed' || c.src === 'recorded' || c.src === 'ref' || c.src === undefined;
/** A percentage is its fraction; it is also its own number when the run or a
 *  reference prints percentages, but not when a literal in the code is 0.6. */
const variantsOf = (n, c) => (n.percent ? [[n.value / 100, n.ulp / 100], ...(printedPct(c) ? [[n.value, n.ulp]] : [])] : [[n.value, n.ulp]]);

/**
 * Does the prose number n (value, ulp, percent) match candidate c? A value
 * printed less precisely than the prose states cannot establish it (a
 * reference's "0.5" says nothing about the prose's 0.46875), unless `loose`:
 * then two statements only have to agree at the lesser precision, which is
 * what consistency between posts asks.
 */
export function matches(n, c, loose = false) {
  for (const [x, u] of variantsOf(n, c)) {
    if (!loose && (c.ulp || 0) > Math.max(u * 1.0001, Math.abs(x) * 1e-5)) continue;
    const tol = (u / 2 + (c.ulp || 0) / 2) * (1 + 1e-9) + 1e-12 * Math.max(Math.abs(x), Math.abs(c.value));
    if (Math.abs(Math.abs(x) - Math.abs(c.value)) <= tol) return true;
  }
  return false;
}
const relDiff = (n, c) => Math.min(...variantsOf(n, c).map(([v]) => Math.abs(Math.abs(v) - Math.abs(c.value)) / Math.max(Math.abs(v), Math.abs(c.value), 1e-300)));

function candStatus(c) {
  if (c.src === 'test') return c.pass ? 'checked' : 'computed';
  if (c.src === 'shown') return c.asserted ? 'checked' : 'computed';
  if (c.src === 'printed' || c.src === 'recorded') return 'computed';
  if (c.src === 'ref') return 'reference';
  return 'input';
}
const describe = (c) => {
  const v = Number(c.value.toPrecision(8));
  if (c.src === 'test') return `test "${c.test}": ${c.role} ${v}`;
  if (c.src === 'shown' || c.src === 'recorded') return `${c.src === 'recorded' ? `recorded ${c.scale} run: ` : ''}${c.name}${c.index !== null && c.index !== undefined ? `[${c.index}]` : ''} = ${v}`;
  if (c.src === 'printed') return `printed "${c.line}"`;
  if (c.src === 'ref') return `${c.file}:${c.line} "${c.text}"`;
  if (c.src === 'arithmetic') return `$${c.expr}$ = ${v}`;
  return c.name ? `${c.name}${c.index !== undefined ? `[${c.index}]` : ''} = ${v} in the code` : `literal ${v} in "${c.line}"`;
};
const linkOf = (c) => ({ src: c.src, value: c.value, what: describe(c), ...(c.test ? { test: c.test } : {}), ...(c.name ? { name: c.name } : {}), ...(c.file ? { file: c.file, line: c.line } : {}) });

/**
 * The candidates a post offers: every value that could establish a number.
 *   tests    [{ name, pass, asserts: [...] }]  from program.testsOf + the run
 *   shown    [{ name, index, value, ulp }]
 *   printed  [{ line, value, ulp }]
 *   refs     [{ file, line, text, value, ulp }]
 *   consts   [{ name, index?, value }] and literals [{ value, line }]
 *   recorded [{ scale, name, index, value, ulp }]
 */
export function candidates({ tests, shown, printed, refs, consts, literals, recorded = [] }) {
  const asserted = new Set();
  for (const t of tests) if (t.pass) for (const a of t.asserts) for (const n of a.names) asserted.add(n);
  const out = [];
  // the literals a test gives the code it calls, outside its assertions
  // (in a failing test, a problem's, they are only what the program is given)
  for (const t of tests) for (const v of t.literals || []) out.push(t.pass ? { src: 'test', test: t.name, pass: true, ulp: 0, names: [], role: 'literal', value: v } : { src: 'input', ulp: 0, value: v, line: `test "${t.name}"` });
  for (const t of tests) for (const a of t.asserts) {
    const base = { src: 'test', test: t.name, pass: t.pass, ulp: 0, names: a.names };
    if (a.expectedValue != null) out.push({ ...base, role: 'expected', value: a.expectedValue });
    if (a.toleranceValue != null && a.toleranceValue !== 0) out.push({ ...base, role: 'tolerance', value: a.toleranceValue });
    if (a.boundValue != null) out.push({ ...base, role: a.fn === 'assert_lt' ? 'upper bound' : 'lower bound', value: a.boundValue });
    // `assert_lt(0.999, x)` says x > 0.999
    if (a.actualValue != null && (a.fn === 'assert_lt' || a.fn === 'assert_gt')) out.push({ ...base, role: a.fn === 'assert_lt' ? 'lower bound' : 'upper bound', value: a.actualValue });
    // a tolerance relative to the expected value: `0.15 * 56.21` around 56.21 is 15 percent
    if (a.toleranceValue && a.expectedValue) out.push({ ...base, role: 'relative tolerance', value: Math.abs(a.toleranceValue / a.expectedValue) });
    for (const v of a.literals || []) out.push({ ...base, role: 'literal', value: v });
  }
  for (const s of shown) out.push({ src: 'shown', ...s, asserted: asserted.has(s.name) });
  for (const r of recorded) out.push({ src: 'recorded', ...r });
  for (const p of printed) out.push({ src: 'printed', ...p });
  for (const r of refs) out.push({ src: 'ref', ...r });
  for (const c of consts) out.push({ src: 'input', ulp: 0, ...c });
  for (const l of literals) out.push({ src: 'input', ulp: 0, ...l });
  return out;
}

/** The test whose name shares the most words with the sentence, if it shares enough. */
export function associatedTest(sentence, tests) {
  const s = tokens(sentence);
  if (!s.size) return null;
  let best = null;
  let second = 0;
  for (const t of tests) {
    const tt = tokens(t.name);
    let k = 0;
    for (const w of s) if (tt.has(w)) k++;
    const score = k / Math.max(1, Math.min(s.size, tt.size));
    if (!best || score > best.score || (score === best.score && k > best.k)) { second = best ? best.score : 0; best = { test: t, score, k }; } else if (score > second) second = score;
  }
  if (!best || best.k < 2 || best.score < 0.34 || best.score === second) return null;
  return best.test;
}

/**
 * Which numbers of a sentence each backticked name labels: "a (`x`), b and c
 * (`y`, `z`)" ties a to x and b, c to y and z. Returns an array parallel to
 * `numbers` of name lists.
 */
export function namesByNumber(sentence, numbers) {
  const res = numbers.map(() => []);
  // walk the spans in order: numbers accumulate until a code span names them
  let pending = [];
  let k = 0;
  const pieces = spans(sentence);
  // count how many of the sentence's numbers each non-code span holds
  let seen = 0;
  for (let i = 0; i < pieces.length; i++) {
    const p = pieces[i];
    if (p.kind === 'code') {
      const names = [...p.s.matchAll(/[A-Za-z_]\w*/g)].map((m) => m[0]);
      // consecutive code spans ("(`m_mc`, `sigma`)") name the same numbers
      let j = i + 1;
      while (pieces[j] && pieces[j].kind === 'text' && /^[\s,and]*$/.test(pieces[j].s) && pieces[j + 1]?.kind === 'code') {
        names.push(...[...pieces[j + 1].s.matchAll(/[A-Za-z_]\w*/g)].map((m) => m[0]));
        j += 2;
      }
      for (const idx of pending) res[idx].push(...names);
      pending = [];
      i = j - 1;
      continue;
    }
    // numbers are returned in span order by numbersIn, so this span's share is
    // the ones whose raw text it contains, in order
    const text = p.s;
    let from = 0;
    while (seen < numbers.length) {
      const raw = numbers[seen].raw.replace(/^[-−]/, '');
      const at = text.indexOf(raw, from);
      if (at < 0) break;
      pending.push(seen);
      from = at + raw.length;
      seen++;
    }
    k++;
  }
  return res;
}

const symKey = (s) => String(s).replace(/\\[a-zA-Z]+/g, (m) => m.slice(1)).replace(/[{}\s_\\^|]/g, '').toLowerCase();

/**
 * Link every number of every sentence of one post. Mutates each number,
 * adding { status, link, alternatives, note, suspect }.
 */
export function linkPost({ sentences, cands, tests, citations }) {
  const strong = cands.filter((c) => c.src === 'test' || c.src === 'shown' || c.src === 'recorded' || c.src === 'printed');
  const byName = new Map();
  for (const c of cands) if (c.name && (c.src === 'shown' || c.src === 'recorded' || c.src === 'input')) (byName.get(c.name) || byName.set(c.name, []).get(c.name)).push(c);
  const bySym = new Map();
  for (const name of byName.keys()) bySym.set(symKey(name), name);
  for (const s of sentences) {
    const nums = s.numbers;
    if (!nums.length) continue;
    const names = namesByNumber(s.text, nums);
    const test = s.origin === 'test' ? tests.find((t) => t.name === s.text) || associatedTest(s.text, tests) : associatedTest(s.text, tests);
    const cited = citations.length > 0;
    nums.forEach((n, i) => {
      if (n.role === 'year' || n.role === 'label' || n.role === 'expression') { n.status = 'skip'; return; }
      // arithmetic the sentence writes out: "$\ln 2 \approx 0.693$" is
      // established by evaluating ln 2, and contradicted when it does not round
      const ev = n.where === 'math' && n.symbol ? texEval(n.symbol) : null;
      if (ev !== null && n.role !== 'uncertainty') {
        const c = { src: 'arithmetic', value: ev, ulp: 0, expr: n.symbol };
        const V = Math.abs(n.percent ? n.value / 100 : n.value);
        const ok = n.qualifier === 'upper' ? Math.abs(ev) <= V * (1 + 1e-12)
          : n.qualifier === 'lower' ? Math.abs(ev) >= V * (1 - 1e-12)
            : matches(n, c) || (n.qualifier === 'approx' && n.pow10 && V > 0 && Math.abs(Math.log10(Math.abs(ev) / V)) < 0.5);
        if (ok) { n.status = 'checked'; n.link = linkOf(c); return; }
        if (n.qualifier !== 'approx' || relDiff(n, c) > 0.1) return mismatch(n, c, 'the arithmetic the sentence writes out');
      }
      // the values this number is ABOUT, when the sentence says
      const assoc = [];
      const why = [];
      for (const nm of names[i]) for (const c of byName.get(nm) || []) { assoc.push(c); }
      if (names[i].length) why.push(`named \`${names[i].join('`, `')}\``);
      if (!names[i].length && n.symbol && bySym.has(symKey(n.symbol))) {
        const nm = bySym.get(symKey(n.symbol));
        assoc.push(...byName.get(nm));
        why.push(`$${n.symbol}$ is \`${nm}\``);
      }
      const testAssoc = [];
      if (test) {
        for (const c of cands) if (c.src === 'test' && c.test === test.name) testAssoc.push(c);
        for (const a of test.asserts) for (const nm of a.names) for (const c of byName.get(nm) || []) if (c.src !== 'input') testAssoc.push(c);
      }
      // a count ("4 iterations", "= 0", "the 2 in 2 pi") is an exact integer: a
      // tolerance of 1e-6 or an expected 0.99919 does not establish it
      const exactCount = (n.role === 'count' && !n.qualifier) || (n.value === 0 && !n.qualifier.startsWith('approx') && !/[.eE]/.test(n.raw));
      // A 1 in some unrelated test does not check the prose's "q = m =
      // 1": for a count, a test's number checks it only in the test the
      // sentence is about, or as the value a test expects and names ("3
      // roots"); elsewhere it is a number the code is given
      const named = (c) => c.role === 'expected' && new RegExp(`(?<![\\w.])${Math.abs(n.value)}(?![\\w.])`).test(c.test);
      const asInput = (c) => (exactCount && c.src === 'test' && !(test && c.test === test.name) && !named(c) ? { src: 'input', ulp: 0, value: c.value, line: `test "${c.test}"` } : c);
      const hits = cands.filter((c) => matches(n, c) && (!exactCount || Math.abs(Math.abs(c.value) - Math.abs(n.value)) <= 1e-9 * Math.abs(n.value))).map(asInput);
      // a bound: established by a test's tolerance or bound equal to it, or by
      // the values it bounds
      const bound = n.qualifier === 'upper' || n.qualifier === 'lower';
      if (bound) {
        // the values it bounds: variables named beside it, and what the
        // associated test asserts the same kind of bound on
        const onNames = new Set();
        const B0 = Math.abs(n.percent ? n.value / 100 : n.value);
        // an assertion bounds the same quantity only if its bound is of the same size
        if (test) for (const a of test.asserts) if (a.on && a.boundValue && B0 && Math.abs(a.boundValue) / B0 < 100 && B0 / Math.abs(a.boundValue) < 100 && ((n.qualifier === 'upper' && a.fn === 'assert_lt') || (n.qualifier === 'lower' && a.fn === 'assert_gt'))) onNames.add(a.on);
        let bounded = [...assoc, ...[...onNames].flatMap((nm) => byName.get(nm) || [])].filter((c) => c.src === 'shown' || c.src === 'recorded');
        // a smallness claim with nothing named ("T + R = 1 to 1e-15"): the
        // errors the program measures and bounds at a similar size are what it
        // is about. If none of them is that small, the prose claims more than
        // the program shows.
        const B = Math.abs(n.percent ? n.value / 100 : n.value);
        if (!bounded.length && n.qualifier === 'upper' && B > 0 && B <= 1e-6) {
          const errs = new Set();
          for (const t of tests) if (t.pass) for (const a of t.asserts) if (a.on && a.fn === 'assert_lt' && a.boundValue > 0 && a.boundValue / B < 1e3 && B / a.boundValue < 1e3) errs.add(a.on);
          bounded = [...errs].flatMap((nm) => byName.get(nm) || []).filter((c) => c.src === 'shown' || c.src === 'recorded');
        }
        const eqBound = hits.find((c) => c.src === 'test' && c.role !== 'expected' && test && c.test === test.name);
        if (eqBound) { settle(n, [eqBound, ...hits]); return; }
        if (bounded.length && n.role === 'value') {
          const ok = bounded.filter((c) => (n.qualifier === 'upper' ? Math.abs(c.value) <= Math.abs(n.percent ? n.value / 100 : n.value) : Math.abs(c.value) >= Math.abs(n.percent ? n.value / 100 : n.value)));
          if (ok.length) {
            n.status = ok.some((c) => c.asserted) ? 'checked' : 'computed';
            n.link = { ...linkOf(ok[0]), bound: n.qualifier };
            n.note = `${n.qualifier === 'upper' ? 'at most' : 'at least'} ${n.raw}${n.percent ? '%' : ''}: ${describe(ok[0])} satisfies it`;
            return;
          }
          const worst = bounded.reduce((a, c) => (Math.abs(c.value) > Math.abs(a.value) === (n.qualifier === 'upper') ? c : a));
          const ratio = Math.abs(worst.value) / Math.abs(n.percent ? n.value / 100 : n.value);
          if (ratio < 1e3 && ratio > 1e-3) {
            n.status = 'mismatch';
            n.link = linkOf(worst);
            n.note = `the prose bounds it by ${n.raw}${n.percent ? '%' : ''} (${n.qualifier}), but ${describe(worst)}${why.length ? `; ${why.join(', ')}` : ''}${test ? `; test "${test.name}"` : ''}`;
            return;
          }
        }
      }
      if (hits.length) { settle(n, hits, new Set([...assoc, ...testAssoc])); return; }
      if (n.role === 'count' || n.role === 'uncertainty' && !assoc.length) { orphan(n); return; }
      // no value matches: is there evidence the number is about one that disagrees?
      const near = (list, lim) => {
        let best = null;
        for (const c of list) { const d = relDiff(n, c); if (!best || d < best.d) best = { c, d }; }
        return best && best.d <= lim ? best : null;
      };
      const a1 = near(assoc.filter((c) => c.src !== 'input'), 0.5);
      if (a1) return mismatch(n, a1.c, why.join(', '));
      const a2 = bound ? null : near(testAssoc, 0.05);
      if (a2) return mismatch(n, a2.c, `test "${test.name}" shares its words`);
      if (cited) {
        n.status = 'cited';
        n.link = { src: 'citation', what: citations.map((c) => c.text).join('; ') };
        return;
      }
      orphan(n);
      if (n.digits >= 3) {
        const s3 = near(strong, 0.01);
        if (s3) n.suspect = `within ${(100 * s3.d).toPrecision(2)}% of ${describe(s3.c)}, but does not round to it`;
      }
    });
  }
}

/**
 * Choose what establishes a number among the values it matches: a value the
 * sentence names or whose test it echoes first, then one that equals the
 * number as written (within a tenth of its last digit) before one it merely
 * rounds to, and among those the strongest status. So "0.2" is the program's
 * 0.2, not a 0.1534 that happens to round to it at one digit.
 */
function settle(n, hits, preferred = new Set()) {
  const x = n.percent ? null : n.value;
  const tight = (c) => {
    const vs = n.percent ? [n.value / 100, n.value] : [x];
    const u = n.percent ? [n.ulp / 100, n.ulp] : [n.ulp];
    return vs.some((v, i) => Math.abs(Math.abs(v) - Math.abs(c.value)) <= Math.max(u[i] * 0.05, (c.ulp || 0) / 2, 1e-12 * Math.abs(v)));
  };
  const key = (c) => (preferred.has(c) ? 0 : 2) + (tight(c) ? 0 : 1);
  const ranked = [...hits].sort((a, b) => key(a) - key(b) || RANK[candStatus(a)] - RANK[candStatus(b)]);
  n.status = candStatus(ranked[0]);
  n.link = linkOf(ranked[0]);
  n.alternatives = hits.length - 1;
}
function orphan(n) {
  n.status = 'orphan';
}
function mismatch(n, c, why) {
  n.status = 'mismatch';
  n.link = linkOf(c);
  n.note = `the prose says ${n.raw}${n.percent ? '%' : ''}, but ${describe(c)}${why ? ` (${why})` : ''}`;
}
