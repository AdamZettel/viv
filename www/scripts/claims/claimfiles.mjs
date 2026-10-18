// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Claims: statements about a topic, specific and bounded, each demonstrated
// by a Viv program and (where vipyr has the API) its vipyr twin. The files are
// claims/<topic>/<slug>.viv and .py; tools/claims/run.py runs them and
// records claims/results.json. This module joins the two for the site (each
// claim is a post of kind `claim` in the claims channel, scripts/feed-build.mjs),
// the claims database (claims.json's `claims` section, scripts/claim-build.mjs)
// and scripts/claim-status.mjs.
//
// A claim's status:
//   checked   both programs pass, agree, and ran on the sources as they are
//   mismatch  a program fails, or the two disagree on a value both show
//   stale     the sources changed since the recorded run
//   unrun     never run
import { readFileSync, readdirSync, existsSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { createHash } from 'node:crypto';
import { declaredClaims, placedClaims } from '../../src/lib/claimRefs.js';

export const CLAIM_STATUSES = ['checked', 'mismatch', 'stale', 'unrun'];
export const LANES = ['ai', 'draft', 'reviewed', 'published'];
// a claim's topic (its directory under claims/, and its header's `topic:`
// line; older headers say `track:`) and the title the site shows for it, the
// same titles as TOPICS in src/lib/site.ts, in page order
export const TOPIC_TITLES = { math: 'Math', stats: 'Statistics', science: 'Science', power: 'Power systems', finance: 'Finance', control: 'Dynamics and control', hardware: 'Hardware', ml: 'Machine learning', data: 'Data' };
const FIELD = /^([a-z][a-z_]*):[ \t]*(.*)$/;

/** The `key: value` fields of a file's first `###` block. */
export function headerFields(src) {
  const lines = src.split('\n');
  const start = lines.findIndex((l) => l.trim() === '###');
  if (start < 0) return {};
  const out = {};
  let key = null;
  for (const l of lines.slice(start + 1)) {
    if (l.trim() === '###') break;
    const m = l.match(FIELD);
    if (m) { key = m[1]; out[key] = m[2].trim(); } else if (key && /^[ \t]/.test(l)) out[key] += ` ${l.trim()}`; else key = null;
  }
  return out;
}

/** The program without its header block and SPDX lines: what the page shows. */
export function bodyOf(src) {
  const lines = src.split('\n');
  const a = lines.findIndex((l) => l.trim() === '###');
  const b = a < 0 ? -1 : lines.findIndex((l, i) => i > a && l.trim() === '###');
  const rest = b < 0 ? lines : [...lines.slice(0, a), ...lines.slice(b + 1)];
  return rest.filter((l) => !/SPDX-/.test(l)).join('\n').replace(/^(#! timing\n)?\s*\n/, '$1').replace(/\n{3,}/g, '\n\n').trim();
}

function walk(dir) {
  const out = [];
  if (!existsSync(dir)) return out;
  for (const f of readdirSync(dir)) {
    const p = join(dir, f);
    if (statSync(p).isDirectory()) out.push(...walk(p));
    else if (f.endsWith('.viv') && !f.startsWith('_')) out.push(p);
  }
  return out.sort();
}

/** The source without its `###` header block. */
export function programOf(src) {
  const lines = src.split('\n');
  const marks = lines.map((l, i) => (l.trim() === '###' ? i : -1)).filter((i) => i >= 0);
  return marks.length < 2 ? src : [...lines.slice(0, marks[0]), ...lines.slice(marks[1] + 1)].join('\n');
}

/** A claim's hash: its programs and `agree:`, not the header's words (the same
 *  as tools/claims/run.py's digest), so a title or scope edit needs no rerun. */
export function claimDigest(src, twin, h = headerFields(src)) {
  return createHash('sha256').update(`${programOf(src)}\nagree: ${h.agree || ''}\n${twin ?? ''}`).digest('hex').slice(0, 16);
}

const list = (s) => (s ? s.split(',').map((x) => x.trim()).filter(Boolean) : []);

/** Every claim with its sources, recorded run and links. `codemap` (optional) to link code maps. */
export function buildClaims(ROOT, codemap = null) {
  const results = existsSync(join(ROOT, 'claims/results.json')) ? JSON.parse(readFileSync(join(ROOT, 'claims/results.json'), 'utf8')).claims : {};
  const out = [];
  for (const p of walk(join(ROOT, 'claims'))) {
    const src = readFileSync(p, 'utf8');
    const h = headerFields(src);
    if (!h.claim) continue;
    const twinPath = p.replace(/\.viv$/, '.py');
    const twin = existsSync(twinPath) ? readFileSync(twinPath, 'utf8') : null;
    const slug = p.split('/').pop().replace(/\.viv$/, '').replace(/_/g, '-');
    const sha = claimDigest(src, twin, h);
    const r = results[slug];
    const status = !r ? 'unrun' : r.sha !== sha ? 'stale' : r.status === 'pass' ? 'checked' : 'mismatch';
    // the vipyr functions the twin calls, as vp.<family>.<name>
    const api = [...new Set([...list(h.api), ...[...(twin || '').matchAll(/\bvp\.([a-z]+)\.([a-z_]+)\(/g)].map((m) => `${m[1]}.${m[2]}`)])];
    const links = [];
    for (const g of list(h.gallery)) links.push({ kind: 'gallery', href: `/vipyr/gallery/${g}`, label: g });
    for (const a of api) {
      const [fam, name] = a.split('.');
      links.push({ kind: 'api', href: `/vipyr/${fam}/${name}`, label: `vp.${a}` });
      if (!codemap || codemap.functions?.[a]) links.push({ kind: 'map', href: `/vipyr/map/${a}`, label: `code map of vp.${a}` });
    }
    for (const rel of list(h.related)) links.push({ kind: rel.startsWith('/') ? 'page' : 'source', href: rel.startsWith('/') ? rel : null, label: rel });
    const side = (k) => (r && r[k] ? { status: r[k].status, seconds: r[k].seconds, tests: r[k].tests, values: r[k].values, output: r[k].output, figure: r[k].figure || '', error: r[k].error || null } : null);
    out.push({
      slug, file: relative(ROOT, p), twin: twin ? relative(ROOT, twinPath) : null,
      claim: h.claim, title: h.title || '', line: h.line || '', why: h.why || '', topic: h.topic || h.track || 'other', topicTitle: TOPIC_TITLES[h.topic || h.track] || h.topic || h.track || 'Other', section: h.section || '', reference: h.reference || '', scope: h.scope || '',
      lane: LANES.includes(h.lane) ? h.lane : 'ai', by: h.by || null, cost: h.cost || 'fast',
      vivOnly: !twin, status, ran: r?.ran || null, sha,
      viv: { source: bodyOf(src), run: side('viv') },
      vipyr: twin ? { source: bodyOf(twin.replace(/^"""[\s\S]*?"""\n/m, (m) => m.replace(/###[\s\S]*?###\n?/, ''))), run: side('vipyr') } : null,
      agreement: r?.agreement || [], links,
    });
  }
  return out;
}

/** The posts that cite claims: { slug, file, declared, placed } for each post
 *  whose header has a `claims:` line or whose text has a `[claim: slug]`. */
export function citingPosts(ROOT) {
  const dir = join(ROOT, 'posts');
  const out = [];
  if (!existsSync(dir)) return out;
  for (const f of readdirSync(dir).filter((x) => /^\d{4}-\d{2}-\d{2}-[a-z0-9-]+\.viv$/.test(x)).sort()) {
    const src = readFileSync(join(dir, f), 'utf8');
    const declared = declaredClaims(headerFields(src).claims);
    const placed = placedClaims(src);
    if (declared.length || placed.length) out.push({ slug: f.replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.viv$/, ''), file: f, declared, placed });
  }
  return out;
}
