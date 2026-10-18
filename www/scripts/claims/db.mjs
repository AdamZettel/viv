// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The claims database with its verdicts merged in, and the checks across posts.
// Shared by scripts/claims.mjs (queries), scripts/claims-status.mjs (CI) and
// the site's /claims page.
//
// Verdicts live in posts/claim_verdicts.json, the plasma book's schema:
//
//   verdicts  { <sentence id>: { status, refs: [...], note } }
//             status: verified | plausible | disputed | not-checkable
//             A sentence with no entry is `unchecked`.
//   numbers   { <number id>: { status, note } }
//             status: accepted (a mismatch or orphan a reader has checked and
//             found correct, say why in the note) | disputed
//
// Both are keyed by hashes of the sentence text, so editing a sentence changes
// its id and its verdict no longer applies: it is STALE, and the sentence is
// unchecked again. That is the point.
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { tokens, matches } from './link.mjs';

export const VERDICT_STATUSES = ['verified', 'plausible', 'disputed', 'not-checkable'];
export const NUMBER_VERDICTS = ['accepted', 'disputed'];

export function paths(ROOT) {
  return {
    db: join(ROOT, 'www/src/data/claims.json'),
    verdicts: join(ROOT, 'posts/claim_verdicts.json'),
    quantities: join(ROOT, 'posts/claim_quantities.json'),
  };
}

export function loadVerdicts(file) {
  if (!existsSync(file)) return { verdicts: {}, numbers: {} };
  const v = JSON.parse(readFileSync(file, 'utf8'));
  return { verdicts: v.verdicts || {}, numbers: v.numbers || {} };
}

/** The database, with each sentence's verdict and each number's accepted/disputed mark. */
export function load(ROOT) {
  const p = paths(ROOT);
  if (!existsSync(p.db)) throw new Error('www/src/data/claims.json is missing: run `npm run claims:build`');
  const db = JSON.parse(readFileSync(p.db, 'utf8'));
  const v = loadVerdicts(p.verdicts);
  const sentenceIds = new Set(db.sentences.map((s) => s.id));
  const numberIds = new Set(db.numbers.map((n) => n.id));
  for (const s of db.sentences) {
    const vd = v.verdicts[s.id];
    s.verdict = vd ? vd.status : 'unchecked';
    if (vd) s.verdictNote = vd.note || '', s.verdictRefs = vd.refs || [];
  }
  for (const n of db.numbers) {
    const vd = v.numbers[n.id];
    if (vd) { n.verdict = vd.status; n.verdictNote = vd.note || ''; }
  }
  db.stale = {
    verdicts: Object.keys(v.verdicts).filter((id) => !sentenceIds.has(id)),
    numbers: Object.keys(v.numbers).filter((id) => !numberIds.has(id)),
  };
  db.rawVerdicts = v;
  db.quantities = existsSync(p.quantities) ? JSON.parse(readFileSync(p.quantities, 'utf8')).quantities : [];
  db.sentenceOf = (n) => db.sentences.find((s) => s.id === n.sentence && s.post === n.post);
  db.postOf = (slug) => db.posts.find((x) => x.slug === slug);
  return db;
}

/** A number counts against CI unless a verdict accepts it. */
export const failing = (n) => n.status === 'mismatch' && n.verdict !== 'accepted' || n.verdict === 'disputed';

/**
 * Cross-post consistency.
 *   quantities   each registered quantity (posts/claim_quantities.json): every
 *                statement of it, and whether it rounds to the accepted value
 *   inconsistent statements that do not (these fail CI)
 *   suspects     pairs of numbers in different posts, each with four or more
 *                digits, within 0.5% of each other but disagreeing at their
 *                stated precision, in sentences that share at least two words:
 *                probably one quantity stated two ways (reported, not failed)
 */
export function consistency(db) {
  const nums = db.numbers.filter((n) => n.status !== 'skip' && n.role === 'value' && n.qualifier !== 'upper' && n.qualifier !== 'lower');
  const quantities = [];
  const inconsistent = [];
  for (const q of db.quantities) {
    const re = new RegExp(q.match, 'i');
    const statements = [];
    for (const n of nums) {
      const v = Math.abs(n.value);
      const vv = n.percent ? v / 100 : v;
      const inRange = (x) => x >= Math.min(...q.range.map(Math.abs)) && x <= Math.max(...q.range.map(Math.abs));
      if (!inRange(v) && !inRange(vv)) continue;
      if (!re.test(`${n.before || ''} ${n.symbol || ''} ${n.unit || ''} ${n.after || ''}`)) continue;
      const ok = matches(n, { src: 'registry', value: q.value, ulp: q.ulp || 0 }, true);
      const st = { post: n.post, id: n.id, raw: n.raw, value: n.value, percent: n.percent, line: n.line, ok, context: n.context };
      statements.push(st);
      if (!ok && n.verdict !== 'accepted') inconsistent.push({ ...st, key: q.key, name: q.name, accepted: q.value });
    }
    quantities.push({ ...q, statements, posts: [...new Set(statements.map((s) => s.post))] });
  }
  // unregistered: near-equal numbers across posts that disagree at their precision
  const big = nums.filter((n) => n.digits >= 4 && n.value !== 0);
  const tok = new Map();
  for (const n of big) tok.set(n, tokens(n.context));
  const suspects = [];
  const sorted = [...big].sort((a, b) => Math.abs(a.value) - Math.abs(b.value));
  for (let i = 0; i < sorted.length; i++) {
    const a = sorted[i];
    for (let j = i + 1; j < sorted.length; j++) {
      const b = sorted[j];
      const d = (Math.abs(b.value) - Math.abs(a.value)) / Math.abs(b.value);
      if (d > 0.005) break;
      if (a.post === b.post || a.percent !== b.percent) continue;
      if (matches(a, { value: b.value, ulp: b.ulp }, true)) continue;
      const ta = tok.get(a);
      const shared = [...tok.get(b)].filter((w) => ta.has(w) && !w.startsWith('#'));
      if (shared.length < 2) continue;
      suspects.push({ a: { post: a.post, id: a.id, raw: a.raw, line: a.line, context: a.context }, b: { post: b.post, id: b.id, raw: b.raw, line: b.line, context: b.context }, shared });
    }
  }
  return { quantities, inconsistent, suspects };
}

/** Histogram of number statuses and sentence verdicts, overall and grouped. */
export function stats(db, by = null) {
  const nums = db.numbers.filter((n) => n.status !== 'skip');
  const groupOf = (n) => (by === 'post' ? n.post : by === 'channel' ? db.postOf(n.post)?.channel || '(none)' : by === 'origin' ? n.origin : 'all');
  const rows = new Map();
  for (const n of nums) {
    const g = groupOf(n);
    const r = rows.get(g) || rows.set(g, { group: g, numbers: 0, sentences: 0, checked: 0, computed: 0, reference: 0, input: 0, cited: 0, orphan: 0, mismatch: 0, accepted: 0 }).get(g);
    r.numbers++;
    r[n.status]++;
    if (n.verdict === 'accepted') r.accepted++;
  }
  for (const s of db.sentences) {
    const g = by === 'post' ? s.post : by === 'channel' ? db.postOf(s.post)?.channel || '(none)' : by === 'origin' ? s.origin : 'all';
    const r = rows.get(g) || rows.set(g, { group: g, numbers: 0, sentences: 0, checked: 0, computed: 0, reference: 0, input: 0, cited: 0, orphan: 0, mismatch: 0, accepted: 0 }).get(g);
    r.sentences++;
    r[`v_${s.verdict}`] = (r[`v_${s.verdict}`] || 0) + 1;
    r[`k_${s.kind}`] = (r[`k_${s.kind}`] || 0) + 1;
  }
  return [...rows.values()].sort((a, b) => a.group.localeCompare(b.group));
}
