// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The site's mathtext (src/lib/mathtext.js) against `viv plot`'s
// (lang/viv/src/mathtext.rs): every label of
// lang/viv/tests/figures/mathtext_labels.tsv must give the same tspans, shapes
// and width. Numbers may differ in the last printed digit (JavaScript's
// toFixed and Rust's formatting round a tie differently).
//
//   node scripts/mathtexttest.mjs
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { mathLabel, shapesSvg, chartLabel, hasMath, MathError } from '../src/lib/mathtext.js';

const here = dirname(fileURLToPath(import.meta.url));
const tsv = readFileSync(join(here, '../../lang/viv/tests/figures/mathtext_labels.tsv'), 'utf8');
let fails = 0;
let checks = 0;
const ok = (cond, what) => { checks++; if (!cond) { fails++; console.log(`FAIL ${what}`); } };

const NUM = /-?\d+\.\d+/g;
function same(a, b) {
  const sa = a.split(NUM), sb = b.split(NUM);
  if (sa.length !== sb.length || sa.some((p, k) => p !== sb[k])) return false;
  const na = a.match(NUM) || [], nb = b.match(NUM) || [];
  return na.every((v, k) => Math.abs(parseFloat(v) - parseFloat(nb[k])) <= 0.0101);
}

for (const line of tsv.split('\n').filter(Boolean)) {
  const [label, inner, shapes, w] = line.split('\t');
  const m = mathLabel(label, 12);
  ok(same(m.inner, inner), `${label}: tspans\n  js:   ${m.inner}\n  rust: ${inner}`);
  ok(same(shapesSvg(m, 0, 0, 'start', 0, '#000'), shapes), `${label}: shapes`);
  ok(Math.abs(m.w - parseFloat(w)) <= 0.0101, `${label}: width ${m.w} vs ${w}`);
}

// the errors are the Rust ones: message and column
const err = (s) => { try { mathLabel(s, 12); return null; } catch (e) { return e instanceof MathError ? e.message : String(e); } };
ok(err('energy $\\foo x$') === 'unknown command \\foo at column 9', 'unknown command, located');
ok(err('$x^$') !== null && err('$\\frac{a}$') !== null && err('$\\left( x$') !== null && err('$x^2^3$') !== null, 'malformed math is an error');
ok(chartLabel('$\\foo$', 12).error && chartLabel('$\\foo$', 12).inner === '$\\foo$', 'a chart draws a bad label as its text');
ok(!hasMath('price ($)') && mathLabel('price ($)', 12).inner === 'price ($)', 'one dollar is text');
ok(mathLabel('cost \\$5', 12).inner === 'cost $5', '\\$ is a dollar');

console.log(`${checks - fails}/${checks} mathtext checks pass`);
process.exit(fails ? 1 : 0);
