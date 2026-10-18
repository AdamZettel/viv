// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// claims -- query the posts' claims database (src/data/claims.json, built by
// scripts/claims-build.mjs) with its verdicts (posts/claim_verdicts.json).
//
//   node scripts/claims.mjs query [filters] [--json] [--limit N]
//       --post <slug or part>     --channel <slug>     --text "<words or /regex/>"
//       --kind numeric|factual|definition|procedural|rhetorical   (the sentence's kind)
//       --status checked|computed|reference|input|cited|orphan|mismatch   (a number's)
//       --mismatch                 same as --status mismatch
//       --verdict verified|plausible|disputed|not-checkable|unchecked     (a sentence's)
//       --origin summary|header|prose|doc|test
//       --role value|count|uncertainty      --suspect   (orphans near a computed value)
//     Rows are numbers when a number filter (--status, --mismatch, --role,
//     --suspect) is given, sentences otherwise; --table numbers|sentences|tests|
//     figures|citations|values forces one.
//   node scripts/claims.mjs stats [--by post|channel|origin] [--json]
//   node scripts/claims.mjs consistency [--json]      quantities stated across posts
//   node scripts/claims.mjs post <slug>                one post's record, summarized
//
// Examples:
//   node scripts/claims.mjs query --post ising --kind numeric --status orphan
//   node scripts/claims.mjs query --channel plasma --text "growth rate"
//   node scripts/claims.mjs query --mismatch
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { load, stats, consistency } from './claims/db.mjs';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);

const argv = process.argv.slice(2);
const cmd = argv[0] && !argv[0].startsWith('--') ? argv.shift() : 'query';
const flag = (k) => argv.includes(`--${k}`);
const opt = (k) => (argv.includes(`--${k}`) ? argv[argv.indexOf(`--${k}`) + 1] : null);
const JSON_OUT = flag('json');
const LIMIT = Number(opt('limit') || (JSON_OUT ? Infinity : 60));

const db = load(ROOT);

const cut = (s, n) => { const t = String(s ?? ''); return t.length > n ? `${t.slice(0, n - 1)}…` : t; };
function table(rows, cols) {
  if (!rows.length) { console.log('(no rows)'); return; }
  const w = cols.map(([h, f, max]) => Math.min(max || 40, Math.max(h.length, ...rows.map((r) => String(f(r) ?? '').length))));
  console.log(cols.map(([h], i) => h.padEnd(w[i])).join('  '));
  console.log(w.map((x) => '-'.repeat(x)).join('  '));
  for (const r of rows) console.log(cols.map(([, f], i) => cut(f(r), w[i]).padEnd(w[i])).join('  ').trimEnd());
}
const textMatch = (t) => {
  const q = opt('text');
  if (!q) return true;
  if (/^\/.*\/[a-z]*$/.test(q)) { const [, b, fl] = q.match(/^\/(.*)\/([a-z]*)$/); return new RegExp(b, fl || 'i').test(t); }
  return String(t).toLowerCase().includes(q.toLowerCase());
};
const postMatch = (slug) => {
  const q = opt('post');
  if (q && !slug.includes(q)) return false;
  const ch = opt('channel');
  if (ch && db.postOf(slug)?.channel !== ch) return false;
  return true;
};

if (cmd === 'query') {
  const numberFilter = opt('status') || flag('mismatch') || opt('role') || flag('suspect');
  const tbl = opt('table') || (numberFilter ? 'numbers' : 'sentences');
  let rows;
  if (tbl === 'numbers') {
    const status = flag('mismatch') ? 'mismatch' : opt('status');
    rows = db.numbers.filter((n) => n.status !== 'skip' && postMatch(n.post)).filter((n) => {
      const s = db.sentenceOf(n);
      if (status && n.status !== status) return false;
      if (opt('role') && n.role !== opt('role')) return false;
      if (flag('suspect') && !n.suspect) return false;
      if (opt('kind') && s?.kind !== opt('kind')) return false;
      if (opt('origin') && n.origin !== opt('origin')) return false;
      return textMatch(s?.text || n.context);
    });
    if (JSON_OUT) console.log(JSON.stringify(rows.slice(0, LIMIT), null, 1));
    else table(rows.slice(0, LIMIT), [['post', (n) => n.post, 28], ['line', (n) => n.line, 5], ['number', (n) => `${n.raw}${n.percent ? '%' : ''}`, 14], ['status', (n) => `${n.status}${n.verdict ? ` (${n.verdict})` : ''}`, 20], ['established by / note', (n) => n.note || n.link?.what || n.suspect || '', 70], ['context', (n) => n.context, 60]]);
  } else if (tbl === 'sentences') {
    rows = db.sentences.filter((s) => postMatch(s.post)).filter((s) => {
      if (opt('kind') && s.kind !== opt('kind')) return false;
      if (opt('verdict') && s.verdict !== opt('verdict')) return false;
      if (opt('origin') && s.origin !== opt('origin')) return false;
      return textMatch(s.text);
    });
    if (JSON_OUT) console.log(JSON.stringify(rows.slice(0, LIMIT), null, 1));
    else table(rows.slice(0, LIMIT), [['id', (s) => s.id, 12], ['post', (s) => s.post, 28], ['line', (s) => s.line, 5], ['kind', (s) => s.kind, 10], ['verdict', (s) => s.verdict, 13], ['sentence', (s) => s.text, 110]]);
  } else {
    rows = (db[tbl] || []).filter((r) => postMatch(r.post)).filter((r) => textMatch(JSON.stringify(r)));
    if (JSON_OUT) console.log(JSON.stringify(rows.slice(0, LIMIT), null, 1));
    else if (tbl === 'tests') table(rows.slice(0, LIMIT), [['post', (t) => t.post, 28], ['interp', (t) => t.interp, 7], ['native', (t) => t.native ?? '-', 7], ['assertions', (t) => t.asserts.map((a) => a.fn.replace('assert_', '')).join(','), 24], ['test', (t) => t.name, 90]]);
    else if (tbl === 'figures') table(rows.slice(0, LIMIT), [['post', (f) => f.post, 28], ['cell', (f) => f.cell, 4], ['title', (f) => f.title, 50], ['draws', (f) => f.series.map((x) => `${x.type} "${x.label}"${x.points ? ` (${x.points})` : ''}`).join('; '), 90]]);
    else if (tbl === 'citations') table(rows.slice(0, LIMIT), [['post', (c) => c.post, 28], ['kind', (c) => c.kind, 8], ['year', (c) => c.year ?? '', 4], ['citation', (c) => c.text, 90]]);
    else table(rows.slice(0, LIMIT), [['post', (v) => v.post, 28], ['row', (v) => JSON.stringify({ ...v, post: undefined }), 140]]);
  }
  if (!JSON_OUT && rows.length > LIMIT) console.log(`... ${rows.length - LIMIT} more (--limit N, or --json for all)`);
  if (!JSON_OUT) console.log(`${rows.length} row${rows.length === 1 ? '' : 's'}`);
} else if (cmd === 'stats') {
  const by = opt('by');
  const rows = stats(db, by).filter((r) => postMatch(by === 'post' ? r.group : db.posts.find((p) => p.channel === r.group)?.slug || (opt('post') || opt('channel') ? '~' : '')));
  if (JSON_OUT) console.log(JSON.stringify(rows, null, 1));
  else table(rows, [[by || 'all', (r) => r.group, 44], ['numbers', (r) => r.numbers], ['checked', (r) => r.checked], ['computed', (r) => r.computed], ['reference', (r) => r.reference], ['input', (r) => r.input], ['cited', (r) => r.cited], ['orphan', (r) => r.orphan], ['mismatch', (r) => r.mismatch], ['sentences', (r) => r.sentences], ['verified', (r) => r.v_verified || 0], ['plausible', (r) => r.v_plausible || 0], ['disputed', (r) => r.v_disputed || 0], ['unchecked', (r) => r.v_unchecked || 0]]);
} else if (cmd === 'consistency') {
  const c = consistency(db);
  if (JSON_OUT) { console.log(JSON.stringify(c, null, 1)); process.exit(0); }
  console.log('Registered quantities (posts/claim_quantities.json):');
  table(c.quantities, [['quantity', (q) => q.key, 26], ['accepted value', (q) => q.value, 20], ['statements', (q) => q.statements.length], ['posts', (q) => q.posts.length], ['disagree', (q) => q.statements.filter((s) => !s.ok).length], ['as stated', (q) => [...new Set(q.statements.map((s) => `${s.raw}${s.percent ? '%' : ''}`))].join(', '), 60]]);
  console.log(`\n${c.inconsistent.length} statement(s) disagree with a registered value:`);
  for (const x of c.inconsistent) console.log(`  ${x.key}: ${x.post}:${x.line} says ${x.raw}${x.percent ? '%' : ''}, accepted ${x.accepted}  "${cut(x.context, 90)}"`);
  console.log(`\n${c.suspects.length} unregistered near-miss pair(s) across posts (same words, values within 0.5%, disagreeing at stated precision):`);
  for (const s of c.suspects) console.log(`  ${s.a.post}:${s.a.line} ${s.a.raw}  vs  ${s.b.post}:${s.b.line} ${s.b.raw}   [${s.shared.join(' ')}]`);
} else if (cmd === 'post') {
  const slug = argv[0];
  const p = db.posts.find((x) => x.slug === slug) || db.posts.find((x) => x.slug.includes(slug));
  if (!p) { console.error(`no post matches ${slug}`); process.exit(2); }
  console.log(`${p.title}\n${p.slug}  ${p.date}  ${p.kind}  channel ${p.channel || '-'}  lane ${p.lane}  ${p.author}`);
  if (p.from) console.log(`${p.edge} of ${p.from}`);
  console.log(`tests: ${p.tests.total} (${p.tests.interp} pass in the interpreter${p.tests.native !== null ? `, ${p.tests.native} natively` : ''})`);
  console.log(`refs: ${p.refs.map((r) => r.py).join(', ') || '-'}`);
  console.log(`numbers: ${Object.entries(p.counts).filter(([k]) => k !== 'sentences').map(([k, v]) => `${v} ${k}`).join(', ')}`);
  const sents = db.sentences.filter((s) => s.post === p.slug);
  const vh = {};
  for (const s of sents) vh[s.verdict] = (vh[s.verdict] || 0) + 1;
  console.log(`sentences: ${sents.length}; verdicts ${Object.entries(vh).map(([k, v]) => `${v} ${k}`).join(', ')}`);
  console.log(`figures: ${db.figures.filter((f) => f.post === p.slug).map((f) => f.title).join(' | ')}`);
  console.log(`citations: ${db.citations.filter((c) => c.post === p.slug).map((c) => c.text).join('; ') || '-'}`);
} else {
  console.error(`unknown command ${cmd}; see the top of scripts/claims.mjs`);
  process.exit(2);
}
