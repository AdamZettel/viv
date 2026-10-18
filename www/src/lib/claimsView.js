// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The claims database as the site shows it, read at build time: the per-post
// Claims panel (components/ClaimsPanel.astro) and the /claims page.
// src/data/claims.json is written by scripts/claims-build.mjs; the verdicts
// and registered quantities come from posts/ through scripts/claims/db.mjs.
import { join } from 'node:path';
import { load, consistency, failing } from '../../scripts/claims/db.mjs';

let cached = null;
export function claimsDb() {
  if (!cached) {
    const db = load(join(process.cwd(), '..'));
    db.consistency = consistency(db);
    cached = db;
  }
  return cached;
}

export const STATUS_LABELS = {
  checked: 'checked',
  computed: 'computed',
  reference: 'reference',
  input: 'input',
  cited: 'cited',
  orphan: 'orphan',
  mismatch: 'mismatch',
};
export const STATUS_NOTES = {
  checked: 'a passing test expects it, or the run prints it and a test asserts on it',
  computed: 'the run prints it; no test asserts it',
  reference: 'the post\'s reference script printed it',
  input: 'the program is given it: a constant or literal in the code',
  cited: 'nothing in the program has it; the sentence cites a source',
  orphan: 'nothing establishes it',
  mismatch: 'the prose disagrees with what the program computes',
};
export const STATUS_ORDER = ['checked', 'computed', 'reference', 'input', 'cited', 'orphan', 'mismatch'];

/** One post's claims, summarized for its panel. */
export function postClaims(slug) {
  const db = claimsDb();
  const p = db.postOf(slug);
  if (!p) return null;
  const numbers = db.numbers.filter((n) => n.post === slug && n.status !== 'skip');
  const sentences = db.sentences.filter((s) => s.post === slug);
  const counts = Object.fromEntries(STATUS_ORDER.map((s) => [s, numbers.filter((n) => n.status === s).length]));
  const checkable = sentences.filter((s) => s.kind === 'factual' || s.kind === 'definition');
  const verdicts = {};
  for (const s of sentences) if (s.verdict !== 'unchecked') verdicts[s.verdict] = (verdicts[s.verdict] || 0) + 1;
  const textOf = new Map(sentences.map((s) => [s.id, s.text]));
  const row = (n) => ({ id: n.id, raw: `${n.raw}${n.percent ? '%' : ''}`, status: n.status, line: n.line, what: n.note || n.link?.what || n.suspect || '', accepted: n.verdict === 'accepted' ? n.verdictNote : '', failing: failing(n), sentence: textOf.get(n.sentence) || n.context });
  return {
    slug,
    total: numbers.length,
    counts,
    failing: numbers.filter(failing).length,
    attention: numbers.filter((n) => n.status !== 'checked').map(row),
    // every number in line order, with the test that checks it or what else ties it
    all: [...numbers].sort((a, b) => a.line - b.line).map((n) => ({ ...row(n), test: n.status === 'checked' && n.link?.src === 'test' ? n.link.test : '', checkedBy: n.status === 'checked' ? n.link?.what || '' : '' })),
    sentences: sentences.length,
    checkable: checkable.length,
    checkedSentences: checkable.filter((s) => s.verdict !== 'unchecked').length,
    verdicts,
    disputed: sentences.filter((s) => s.verdict === 'disputed').map((s) => ({ id: s.id, text: s.text, note: s.verdictNote })),
  };
}

/** A one-line summary for the badge on the post's record line. */
export function claimsBadge(slug) {
  const c = postClaims(slug);
  if (!c || !c.total) return null;
  return { checked: c.counts.checked, total: c.total, orphan: c.counts.orphan, failing: c.failing };
}
