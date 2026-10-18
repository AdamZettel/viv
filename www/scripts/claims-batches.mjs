// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// claims-batches -- cut the unchecked sentences into batches for fact-checking.
//
//     node scripts/claims-batches.mjs [--size 30] [--out <dir>] [--post <slug>] [--channel <slug>]
//                                     [--kinds factual,definition,numeric] [--all]
//
// The workflow of an earlier fact-checking pipeline (which cleared 1256
// claims): pull every sentence that needs a verdict and has none,
// write batch_1.json ... batch_K.json, hand each to one checking agent, and
// fold what they return back in with scripts/claims-merge.mjs.
//
// A sentence needs a verdict when it is factual or a definition, or when it is
// numeric and one of its numbers is an orphan, a citation or a mismatch (the
// linker could not tie it to the program, so a reader has to). Procedural and
// rhetorical sentences, and test names (their test is their check), are left
// out. --all includes sentences that already have a verdict.
//
// Each batch keeps a post's sentences together and carries, for each one, the
// sentence before and after it, the post's title and reference scripts, and
// what the linker found for its numbers. The agent writes
// verdicts_<K>.json next to the batch: { "<id>": { status, refs, note } }.
import { writeFileSync, mkdirSync, readdirSync, rmSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { load } from './claims/db.mjs';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const argv = process.argv.slice(2);
const opt = (k, d = null) => (argv.includes(`--${k}`) ? argv[argv.indexOf(`--${k}`) + 1] : d);
const SIZE = Number(opt('size', 30));
const OUT = resolve(opt('out', join(WWW, '.claims-cache/batches')));
const KINDS = new Set(opt('kinds', 'factual,definition,numeric').split(','));

export const INSTRUCTIONS = `You are fact-checking sentences from posts on viv-lang.org: short technical
articles, each a Viv program with prose around it. For each sentence, decide
whether what it says about the world (physics, mathematics, numerical methods,
history, software) is true, and return a verdict:

  verified       true, and you can name a reference that says so (a textbook
                 with section, a paper, a standard, documentation)
  plausible      very likely true but you could not pin it to one
                 authoritative reference, or it is approximate in a way a
                 reference would qualify
  disputed       false or misleading as written; say exactly what is wrong in
                 the note and what the sentence should say. A disputed verdict
                 fails CI until the post is fixed.
  not-checkable  nothing to check against the world: it describes this program
                 or this post, is a transition, or is a fragment of a formula

Numbers the program computes are already checked by the linker; the batch shows
each number's status. Judge those only if the sentence's claim goes beyond
what the run shows. Use the context sentences to understand the claim, but
judge only the sentence given. Every verified or plausible verdict must carry
at least one reference in refs. Keep notes to one or two sentences.

Write verdicts_<K>.json, where K is the batch's number, as
  { "<id>": { "status": "...", "refs": ["..."], "note": "..." }, ... }
with an entry for every id in the batch.`;

function main() {
  const db = load(ROOT);
  const numbersBy = new Map();
  for (const n of db.numbers) (numbersBy.get(`${n.post}|${n.sentence}`) || numbersBy.set(`${n.post}|${n.sentence}`, []).get(`${n.post}|${n.sentence}`)).push(n);
  const needs = (s) => {
    if (s.origin === 'test') return false;
    if (!argv.includes('--all') && s.verdict !== 'unchecked') return false;
    if (opt('post') && !s.post.includes(opt('post'))) return false;
    if (opt('channel') && db.postOf(s.post)?.channel !== opt('channel')) return false;
    if (s.kind === 'factual' || s.kind === 'definition') return KINDS.has(s.kind);
    if (s.kind === 'numeric' && KINDS.has('numeric')) {
      const ns = numbersBy.get(`${s.post}|${s.id}`) || [];
      return ns.some((n) => n.status === 'orphan' || n.status === 'cited' || n.status === 'skip' && n.role === 'year' || n.status === 'mismatch' && n.verdict !== 'accepted');
    }
    return false;
  };
  // one entry per sentence id (the same sentence in two posts is one claim)
  const seen = new Set();
  const items = [];
  const byPost = new Map();
  for (const s of db.sentences) (byPost.get(s.post) || byPost.set(s.post, []).get(s.post)).push(s);
  for (const [slug, list] of byPost) {
    const prose = list.filter((s) => s.origin !== 'test');
    prose.forEach((s, i) => {
      if (!needs(s) || seen.has(s.id)) return;
      seen.add(s.id);
      const p = db.postOf(slug);
      items.push({
        id: s.id, post: slug, title: p.title, channel: p.channel, file: `posts/${p.file}`, line: s.line, kind: s.kind, section: s.section || undefined,
        sentence: s.text,
        before: prose[i - 1]?.text, after: prose[i + 1]?.text,
        numbers: (numbersBy.get(`${slug}|${s.id}`) || []).filter((n) => n.status !== 'skip').map((n) => `${n.raw}${n.percent ? '%' : ''}: ${n.status}${n.link ? ` (${n.link.what})` : ''}`),
        refs: p.refs.map((r) => r.py),
        citations: s.citations,
      });
    });
  }
  mkdirSync(OUT, { recursive: true });
  // a new cut replaces the old batches (the verdicts already written stay)
  for (const f of readdirSync(OUT)) if (/^batch_\d+\.json$/.test(f)) rmSync(join(OUT, f));
  const batches = [];
  let cur = [];
  for (const it of items) {
    // keep a post's sentences together when the batch has room for a few more
    if (cur.length >= SIZE || (cur.length >= SIZE * 0.8 && cur[cur.length - 1].post !== it.post)) { batches.push(cur); cur = []; }
    cur.push(it);
  }
  if (cur.length) batches.push(cur);
  batches.forEach((b, k) => writeFileSync(join(OUT, `batch_${k + 1}.json`), JSON.stringify({ batch: k + 1, of: batches.length, instructions: INSTRUCTIONS.replace(/<K>/g, String(k + 1)), write: join(OUT, `verdicts_${k + 1}.json`), sentences: b }, null, 1)));
  const kinds = {};
  for (const it of items) kinds[it.kind] = (kinds[it.kind] || 0) + 1;
  console.log(`claims-batches: ${items.length} sentences need a verdict (${Object.entries(kinds).map(([k, v]) => `${v} ${k}`).join(', ')}), from ${new Set(items.map((i) => i.post)).size} posts`);
  console.log(`claims-batches: ${batches.length} batch(es) of up to ${SIZE} in ${OUT}`);
  console.log('claims-batches: give each batch_K.json to one checking agent; it writes verdicts_K.json; then');
  console.log(`  node scripts/claims-merge.mjs ${OUT}`);
}

if (import.meta.url === `file://${process.argv[1]}`) main();
