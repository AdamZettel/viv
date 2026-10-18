// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// claim-status -- CI for the claims (npm run claim:status).
//
// Reads claims/ and claims/results.json (scripts/claims/claimfiles.mjs) and
// FAILS (exit 1) on a claim whose recorded run failed or whose Viv program and
// vipyr twin disagree (`mismatch`), and, with --strict, on one whose sources
// changed since the recorded run (`stale`) or that never ran (`unrun`): rerun
// it with `python3 tools/claims/run.py --only <slug> --write`.
// Fails too on a post that cites a claim (`claims:` in its header, or
// `[claim: slug]` in its prose, src/lib/claimRefs.js) that does not exist, is
// not checked, or is placed without being declared; and when
// src/data/claims.json's claims section is out of date (npm run claim:build).
// With --strict, also on a claim whose title:, line: or claim: states a number
// its recorded run does not establish (tools/claims/run.py --text): the words
// are not hashed, so they are checked here without a rerun.
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildClaims, CLAIM_STATUSES, citingPosts } from './claims/claimfiles.mjs';
import { claimRefErrors } from '../src/lib/claimRefs.js';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const STRICT = process.argv.includes('--strict');
const cs = buildClaims(ROOT);
const db = JSON.parse(readFileSync(join(WWW, 'src/data/claims.json'), 'utf8'));
const hist = Object.fromEntries(CLAIM_STATUSES.map((s) => [s, cs.filter((f) => f.status === s)]));
console.log(`\nClaims CI: ${cs.length} claims, ${cs.filter((f) => f.vivOnly).length} Viv only`);
for (const s of CLAIM_STATUSES) console.log(`  ${String(hist[s].length).padStart(4)}  ${s}`);
const bad = [...hist.mismatch, ...(STRICT ? [...hist.stale, ...hist.unrun] : [])];
for (const f of bad) console.log(`  FAIL ${f.status.padEnd(8)} ${f.file}`);
const bySlug = Object.fromEntries(cs.map((c) => [c.slug, c]));
let refs = 0;
const cites = citingPosts(ROOT);
for (const p of cites) {
  for (const e of claimRefErrors(p.declared, p.placed, bySlug)) { console.log(`  FAIL posts/${p.file}: ${e}`); refs++; }
}
console.log(`  ${String(cites.length).padStart(4)}  posts cite claims, ${refs} bad reference${refs === 1 ? '' : 's'}`);
const recorded = new Map((db.claims || []).map((f) => [f.slug, `${f.sha}:${f.status}`]));
const behind = cs.filter((f) => recorded.get(f.slug) !== `${f.sha}:${f.status}`);
if (behind.length) console.log(`  FAIL src/data/claims.json's claims are out of date for ${behind.length} (npm run claim:build)`);
let text = 0;
if (STRICT) {
  const t = spawnSync('python3', [join(ROOT, 'tools/claims/run.py'), '--text'], { cwd: ROOT, encoding: 'utf8' });
  for (const l of (t.stdout || '').split('\n').filter((x) => x.startsWith('FAIL'))) console.log(`  ${l}`);
  text = t.status === 0 ? 0 : 1;
  if (text && !/FAIL/.test(t.stdout || '')) console.log(`  FAIL tools/claims/run.py --text: ${(t.stderr || '').trim().slice(-300)}`);
}
process.exit(bad.length || behind.length || refs || text ? 1 : 0);
