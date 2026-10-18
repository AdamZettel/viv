// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// claims-merge -- fold checking agents' verdicts into posts/claim_verdicts.json.
//
//     node scripts/claims-merge.mjs <verdicts_K.json | dir> ... [--force] [--dry-run]
//
// Each input is { "<id>": { status, refs, note } } (or { verdicts: {...},
// numbers: {...} }). A sentence id (12 hex digits) takes a sentence verdict:
// verified | plausible | disputed | not-checkable. A number id
// (<sentence id>.<k>) takes a number verdict: accepted | disputed.
//
// Refused, with the reason: an id that is not in src/data/claims.json (the
// sentence changed since the batch was cut: re-batch it), a status outside the
// enum, a verified or plausible verdict with no reference, a disputed one with
// no note. An existing verdict is kept unless --force, and a conflict is
// reported. The file is written with its keys sorted, so a merge diffs cleanly.
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { load, paths, VERDICT_STATUSES, NUMBER_VERDICTS } from './claims/db.mjs';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const argv = process.argv.slice(2);
const FORCE = argv.includes('--force');
const DRY = argv.includes('--dry-run');
const inputs = argv.filter((a) => !a.startsWith('--')).flatMap((a) => (statSync(a).isDirectory() ? readdirSync(a).filter((f) => /^verdicts_.*\.json$/.test(f)).sort().map((f) => join(a, f)) : [a]));
if (!inputs.length) { console.error('usage: node scripts/claims-merge.mjs <verdicts_K.json | dir> ... [--force] [--dry-run]'); process.exit(2); }

const db = load(ROOT);
const sentenceIds = new Set(db.sentences.map((s) => s.id));
const numberIds = new Set(db.numbers.map((n) => n.id));
const file = paths(ROOT).verdicts;
const cur = existsSync(file) ? JSON.parse(readFileSync(file, 'utf8')) : { _comment: '', verdicts: {}, numbers: {} };
cur.verdicts ||= {};
cur.numbers ||= {};

const counts = { added: 0, replaced: 0, kept: 0, refused: 0 };
const byStatus = {};
for (const inp of inputs) {
  const raw = JSON.parse(readFileSync(inp, 'utf8'));
  const entries = raw.verdicts || raw.numbers ? [...Object.entries(raw.verdicts || {}), ...Object.entries(raw.numbers || {})] : Object.entries(raw);
  for (const [id, v] of entries) {
    const isNum = /^[0-9a-f]{12}\.\d+$/.test(id);
    const refuse = (why) => { counts.refused++; console.log(`  refused ${id} (${inp.split('/').pop()}): ${why}`); };
    if (!isNum && !/^[0-9a-f]{12}$/.test(id)) { refuse('not a sentence or number id'); continue; }
    if (isNum ? !numberIds.has(id) : !sentenceIds.has(id)) { refuse('not in claims.json: the sentence changed since the batch was cut; re-batch it'); continue; }
    const allowed = isNum ? NUMBER_VERDICTS : VERDICT_STATUSES;
    if (!allowed.includes(v?.status)) { refuse(`status "${v?.status}" is not one of ${allowed.join(', ')}`); continue; }
    const refs = Array.isArray(v.refs) ? v.refs.filter((r) => typeof r === 'string' && r.trim()) : [];
    if (!isNum && (v.status === 'verified' || v.status === 'plausible') && !refs.length) { refuse(`a ${v.status} verdict needs at least one reference`); continue; }
    if (v.status === 'disputed' && !(v.note || '').trim()) { refuse('a disputed verdict needs a note saying what is wrong'); continue; }
    const table = isNum ? cur.numbers : cur.verdicts;
    const entry = isNum ? { status: v.status, note: (v.note || '').trim() } : { status: v.status, refs, note: (v.note || '').trim() };
    if (table[id] && !FORCE) {
      if (JSON.stringify(table[id]) !== JSON.stringify(entry)) console.log(`  kept ${id}: it already has "${table[id].status}" (this input says "${v.status}"; --force replaces)`);
      counts.kept++;
      continue;
    }
    counts[table[id] ? 'replaced' : 'added']++;
    table[id] = entry;
    byStatus[v.status] = (byStatus[v.status] || 0) + 1;
  }
}
const sortObj = (o) => Object.fromEntries(Object.keys(o).sort().map((k) => [k, o[k]]));
const out = { _comment: cur._comment, verdicts: sortObj(cur.verdicts), numbers: sortObj(cur.numbers) };
if (!DRY) writeFileSync(file, `${JSON.stringify(out, null, 1)}\n`);
console.log(`claims-merge: ${counts.added} added, ${counts.replaced} replaced, ${counts.kept} kept, ${counts.refused} refused${Object.keys(byStatus).length ? ` (${Object.entries(byStatus).map(([k, v]) => `${v} ${k}`).join(', ')})` : ''}${DRY ? ' [dry run: nothing written]' : ` -> posts/claim_verdicts.json`}`);
process.exit(counts.refused ? 1 : 0);
