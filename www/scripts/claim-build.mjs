// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// claim-build -- the `claims` section of src/data/claims.json, from claims/
// and the runs tools/claims/run.py recorded in claims/results.json
// (scripts/claims/claimfiles.mjs), with the posts that cite each claim. It
// rewrites that section only; the posts' sentences and numbers are
// scripts/claims-build.mjs's.
//
//     node scripts/claim-build.mjs
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildClaims, CLAIM_STATUSES, citingPosts } from './claims/claimfiles.mjs';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const OUT = join(WWW, 'src/data/claims.json');
const codemap = existsSync(join(WWW, 'src/data/codemap.json')) ? JSON.parse(readFileSync(join(WWW, 'src/data/codemap.json'), 'utf8')) : null;
const claims = buildClaims(ROOT, codemap);
const cites = citingPosts(ROOT);
for (const c of claims) c.citedBy = cites.filter((p) => p.declared.includes(c.slug)).map((p) => p.slug);
const db = JSON.parse(readFileSync(OUT, 'utf8'));
delete db.findings;
delete db.findingStatuses;
db.claims = claims;
db.claimStatuses = CLAIM_STATUSES;
writeFileSync(OUT, `${JSON.stringify(db, null, 0).replace(/\},\{"/g, '},\n{"')}\n`);
const hist = Object.fromEntries(CLAIM_STATUSES.map((s) => [s, claims.filter((f) => f.status === s).length]));
console.log(`claim-build: ${claims.length} claims (${claims.filter((f) => f.vivOnly).length} Viv only): ${Object.entries(hist).map(([k, v]) => `${v} ${k}`).join(', ')}`);
