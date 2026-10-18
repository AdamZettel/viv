// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// claims-status -- CI for the posts' claims (npm run claims:status).
//
// Merges src/data/claims.json (scripts/claims-build.mjs) with the verdicts
// (posts/claim_verdicts.json) and the registered quantities
// (posts/claim_quantities.json), prints a histogram, and FAILS (exit 1) on
//   - a number whose prose disagrees with what the program computes
//     (`mismatch`), unless a verdict accepts it
//   - a sentence or number with a `disputed` verdict
//   - a registered quantity stated with a value that does not round to its
//     accepted one (the same constant stated differently in two posts)
//   - a claims.json older than the posts (rebuild it: npm run claims:build)
// Orphans (numbers nothing establishes) are reported and fail only with
// --strict, which CI uses (npm run claims:status, posts/check.sh). Stale
// verdicts (their sentence was edited or removed) are listed.
//
//   node scripts/claims-status.mjs               the check
//   node scripts/claims-status.mjs --strict      orphans fail too
//   node scripts/claims-status.mjs --warn        report, never fail
//   node scripts/claims-status.mjs --post <slug> one post
//   node scripts/claims-status.mjs --unchecked   list the sentences awaiting a verdict
//   node scripts/claims-status.mjs --orphans     list the orphans
import { readFileSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { load, consistency, failing, VERDICT_STATUSES } from './claims/db.mjs';
import { STATUSES } from './claims/link.mjs';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const argv = process.argv.slice(2);
const flag = (k) => argv.includes(`--${k}`);
const opt = (k) => (argv.includes(`--${k}`) ? argv[argv.indexOf(`--${k}`) + 1] : null);
const STRICT = flag('strict');
const WARN = flag('warn');
const only = opt('post');

const db = load(ROOT);
const inPost = (slug) => !only || slug.includes(only);
const cut = (s, n) => { const t = String(s ?? '').replace(/\s+/g, ' '); return t.length > n ? `${t.slice(0, n - 1)}…` : t; };

// is the database current? every post file, hashed, against the record
const outdated = [];
const files = readdirSync(join(ROOT, 'posts')).filter((f) => /^\d{4}-\d{2}-\d{2}-[a-z0-9-]+\.viv$/.test(f));
for (const f of files) {
  const rec = db.posts.find((p) => p.file === f);
  const h = createHash('sha256').update(readFileSync(join(ROOT, 'posts', f), 'utf8')).digest('hex').slice(0, 16);
  if (!rec) outdated.push(`${f}: not in claims.json`);
  else if (rec.source !== h) outdated.push(`${f}: changed since claims.json was built`);
}
for (const p of db.posts) if (!files.includes(p.file)) outdated.push(`${p.file}: in claims.json but no longer in posts/`);

const nums = db.numbers.filter((n) => n.status !== 'skip' && inPost(n.post));
const sents = db.sentences.filter((s) => inPost(s.post));
const hist = Object.fromEntries(STATUSES.map((s) => [s, 0]));
for (const n of nums) hist[n.status]++;
const vh = Object.fromEntries([...VERDICT_STATUSES, 'unchecked'].map((s) => [s, 0]));
for (const s of sents) vh[s.verdict]++;
const kinds = {};
for (const s of sents) kinds[s.kind] = (kinds[s.kind] || 0) + 1;

const mism = nums.filter((n) => n.status === 'mismatch');
const failingNums = nums.filter(failing);
const disputedSents = sents.filter((s) => s.verdict === 'disputed');
const cons = consistency(db);
const inconsistent = cons.inconsistent.filter((x) => inPost(x.post));
const orphans = nums.filter((n) => n.status === 'orphan' && n.verdict !== 'accepted');

console.log(`\nPost claims CI${only ? ` (${only})` : ''}: ${new Set(nums.map((n) => n.post)).size} posts, ${sents.length} sentences, ${nums.length} numbers`);
console.log('─'.repeat(64));
console.log('Numbers, by what establishes them:');
for (const s of STATUSES) {
  const mark = s === 'mismatch' && failingNums.some((n) => n.status === 'mismatch') ? '  <-- FAIL' : s === 'orphan' && orphans.length ? (STRICT ? '  <-- FAIL (--strict)' : '  (warning)') : '';
  console.log(`  ${String(hist[s]).padStart(5)}  ${s}${mark}`);
}
const accepted = nums.filter((n) => n.verdict === 'accepted');
if (accepted.length) console.log(`         (${accepted.length} of them accepted by a verdict: ${[...new Set(accepted.map((n) => n.status))].join(', ')})`);
console.log('Sentences, by kind:  ' + Object.entries(kinds).map(([k, v]) => `${v} ${k}`).join(', '));
console.log('Sentences, by verdict:');
for (const s of [...VERDICT_STATUSES, 'unchecked']) console.log(`  ${String(vh[s]).padStart(5)}  ${s}${s === 'disputed' && vh[s] ? '  <-- FAIL' : ''}`);
const checkable = sents.filter((s) => s.kind === 'factual' || s.kind === 'definition');
const checkedF = checkable.filter((s) => s.verdict !== 'unchecked').length;
console.log(`  factual and definition sentences with a verdict: ${checkedF}/${checkable.length}`);
console.log(`Registered quantities: ${cons.quantities.filter((q) => q.statements.length).length} stated, ${inconsistent.length} inconsistent statement(s)${inconsistent.length ? '  <-- FAIL' : ''}`);
console.log(`Stale verdicts: ${db.stale.verdicts.length} sentence, ${db.stale.numbers.length} number`);
console.log('─'.repeat(64));

if (mism.length) {
  console.log('\nMISMATCH: the prose disagrees with what the program computes');
  for (const n of mism) {
    const acc = n.verdict === 'accepted';
    console.log(`  ${acc ? '[accepted] ' : ''}${n.post}:${n.line}  ${n.id}  "${n.raw}${n.percent ? '%' : ''}"`);
    console.log(`     ${cut(db.sentenceOf(n)?.text, 150)}`);
    console.log(`     -> ${n.note}${acc ? `\n     accepted: ${n.verdictNote}` : ''}`);
  }
}
const disputedNums = nums.filter((n) => n.verdict === 'disputed');
if (disputedNums.length || disputedSents.length) {
  console.log('\nDISPUTED (must be fixed):');
  for (const n of disputedNums) console.log(`  number ${n.post}:${n.line} "${n.raw}" -> ${n.verdictNote}`);
  for (const s of disputedSents) console.log(`  ${s.id} ${s.post}:${s.line} "${cut(s.text, 110)}"\n     -> ${s.verdictNote}`);
}
if (inconsistent.length) {
  console.log('\nINCONSISTENT with a registered quantity (posts/claim_quantities.json):');
  for (const x of inconsistent) console.log(`  ${x.key}: ${x.post}:${x.line} says ${x.raw}${x.percent ? '%' : ''}; accepted ${x.accepted}\n     "${cut(x.context, 110)}"`);
}
if (cons.suspects.length && !only) {
  console.log('\nNear-miss pairs across posts (reported, not failed):');
  for (const s of cons.suspects) console.log(`  ${s.a.post}:${s.a.line} ${s.a.raw}  vs  ${s.b.post}:${s.b.line} ${s.b.raw}`);
}
if (db.stale.verdicts.length || db.stale.numbers.length) {
  console.log('\nSTALE verdicts (the sentence changed or is gone; re-check it and file a new verdict):');
  for (const id of db.stale.verdicts) console.log(`  ${id}  ${cut(db.rawVerdicts.verdicts[id].note, 90)}`);
  for (const id of db.stale.numbers) console.log(`  ${id}  ${cut(db.rawVerdicts.numbers[id].note, 90)}`);
}
if (outdated.length) {
  console.log('\nOUT OF DATE: src/data/claims.json does not match the posts; run `npm run claims:build`');
  for (const o of outdated.slice(0, 20)) console.log(`  ${o}`);
}
if (flag('orphans') || STRICT) {
  console.log(`\nOrphans (${orphans.length}): numbers nothing in the program, its reference or a citation establishes`);
  for (const n of orphans.slice(0, flag('orphans') ? Infinity : 40)) console.log(`  ${n.post}:${n.line}  "${n.raw}${n.percent ? '%' : ''}"  ${cut(n.context, 90)}${n.suspect ? `\n     suspect: ${n.suspect}` : ''}`);
} else if (orphans.length) {
  console.log(`\n${orphans.length} orphan number(s): stated in prose, established by nothing (warning; --orphans lists them, --strict fails on them)`);
}
if (flag('unchecked')) {
  const un = checkable.filter((s) => s.verdict === 'unchecked');
  console.log(`\nUnchecked factual and definition sentences (${un.length}):`);
  for (const s of un.slice(0, 80)) console.log(`  ${s.id} ${s.post}:${s.line} [${s.kind}] ${cut(s.text, 100)}`);
  if (un.length > 80) console.log(`  ... and ${un.length - 80} more (scripts/claims-batches.mjs writes them all out)`);
}

const fail = failingNums.length + disputedSents.length + inconsistent.length + outdated.length + (STRICT ? orphans.length : 0);
if (fail && !WARN) {
  console.log(`\nclaims:status FAILED: ${failingNums.length} mismatch or disputed number(s), ${disputedSents.length} disputed sentence(s), ${inconsistent.length} inconsistent quantity statement(s)${outdated.length ? `, claims.json out of date` : ''}${STRICT ? `, ${orphans.length} orphan(s)` : ''}`);
  process.exit(1);
}
console.log(`\nclaims:status ok${fail ? ' (--warn: failures reported, not enforced)' : ''}`);
