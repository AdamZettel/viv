// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// claims-build -- every post as a queryable record whose claims can be tested.
//
//     node scripts/claims-build.mjs                  all posts -> src/data/claims.json
//     node scripts/claims-build.mjs --native         and run each post's tests natively too
//     node scripts/claims-build.mjs --file <path.viv> [--as <post file name>]
//                                                    one file (any version of a post), to stdout
//
// The posts' counterpart of an earlier factual CI for a book (its
// build_claims.mjs and claims_status.mjs, kept outside this repository).
// For each post it records:
//
//   posts      slug, header fields, channel, kind, lane, lineage edge, date, who wrote it
//   cells      the notebook's cells: header, prose, code, with their lines
//   sentences  every prose sentence (header summary and body, prose cells, `#|`
//              docs) and every test name, each with a stable id (sha1 of the
//              normalized text, 12 hex digits) and a kind: numeric, factual,
//              definition, procedural, rhetorical
//   numbers    every number stated in a sentence: value, precision, unit, the
//              symbol it is about, and its STATUS (scripts/claims/link.mjs):
//              checked, computed, reference, input, cited, orphan, mismatch
//   tests      name, each assertion (kind, expected value, tolerance or bound),
//              pass or fail in the interpreter and, with --native, in C
//   values     what the run `show`s and prints
//   refs       posts/refs/<slug>.py and the numbers in its .out
//   figures    what each figure draws
//   citations  papers, books and datasets the prose or header names
//
// The run is the one feed-build recorded (public/programs/posts/*.cells.json)
// when it is current, and otherwise the post is run here through viv.wasm.
//
// Why JSON and not SQLite: the whole database is a few MB, the site reads it at
// build time with an import, the query tool filters it in memory in
// milliseconds, and a JSON file diffs in git -- a commit that turns a number
// into an orphan shows as that line. SQLite would add a native dependency to
// the site build and an opaque blob to review.
import { readFileSync, writeFileSync, readdirSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { splitNotebook } from '../src/lib/notebook.js';
import { headerOf, fencedHeader, parseFenced, scalesOf } from '../src/lib/postCheck.js';
import { sentenceId, paragraphs, splitSentences, numbersIn, classify, citationsIn } from './claims/text.mjs';
import { parseRun, parseFigures, constantsOf, testsOf, parseRefOut } from './claims/program.mjs';
import { candidates, linkPost, STATUSES } from './claims/link.mjs';
import { runner } from './claims/run.mjs';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const SRC = join(ROOT, 'posts');
const REFS = join(SRC, 'refs');
const OUT = join(WWW, 'src/data/claims.json');

/** The sentences of a post, from its header, prose cells, `#|` docs and test names. */
export function sentencesOf(src, cells) {
  const out = [];
  const add = (text, origin, cell, section, line) => {
    const nums = numbersIn(text);
    out.push({ id: sentenceId(text), cell, origin, section, line, text, kind: classify(text, nums.numbers), numbers: nums.numbers, names: nums.names });
  };
  const lines = src.split('\n');
  cells.forEach((c, ci) => {
    if (c.kind === 'header') {
      const fence = fencedHeader(src);
      if (!fence) return;
      // fields are `key: value` lines, and `scale <name>:` / `recorded <name>:`
      // lines (postCheck.parseFenced stops at those, so a field after one of
      // them is read there as prose; here it is still a field)
      const FIELD = /^([a-z][a-z0-9_]*)(?: [a-z][a-z0-9_-]*)?:[ \t]*(.*)$/i;
      let k = fence.start + 1;
      while (k < fence.end && FIELD.test(lines[k])) {
        const [, key, value] = lines[k].match(FIELD);
        if (key.toLowerCase() === 'summary') for (const s of splitSentences(value.trim())) add(s.text, 'summary', ci, '', k + 1);
        k++;
      }
      while (k < fence.end && !lines[k].trim()) k++;
      const first = k + 1;
      const body = lines.slice(k, fence.end).join('\n');
      for (const p of paragraphs(body, first)) for (const s of splitSentences(p.text)) add(s.text, 'header', ci, p.section, p.lineAt(s.start));
    } else if (c.kind === 'prose') {
      for (const p of paragraphs(c.body, c.start + 1)) for (const s of splitSentences(p.text)) add(s.text, 'prose', ci, p.section, p.lineAt(s.start));
    } else if (c.kind === 'code') {
      // `#|` doc lines (and the older `#! |`), a bare one ending a paragraph
      const cl = c.text.split('\n');
      let para = null;
      const flush = () => {
        if (para && para.text.trim()) for (const s of splitSentences(para.text)) {
          add(s.text, 'doc', ci, '', para.line);
          // a `#|` doc says what a function does: procedural unless it states a number or defines a term
          const last = out[out.length - 1];
          if (last.kind === 'factual') last.kind = 'procedural';
        }
        para = null;
      };
      cl.forEach((l, k) => {
        const m = l.match(/^\s*#(?:\||! \|)\s?(.*)$/);
        if (!m) { flush(); return; }
        if (!m[1].trim()) { flush(); return; }
        if (!para) para = { text: '', line: c.start + k };
        para.text += (para.text ? ' ' : '') + m[1].trim();
      });
      flush();
    }
  });
  return out;
}

/** Who wrote a post: its `agent:` line, and whether any commit touching it is the owner's. */
function provenance(file, agent) {
  let authors = [];
  try { authors = execFileSync('git', ['log', '--format=%an', '--', `posts/${file}`], { cwd: ROOT, encoding: 'utf8' }).split('\n').filter(Boolean); } catch { /* not in git */ }
  // the agents commit as `adamzettel`; the owner's own commits are `Adam Zettel`
  const owner = authors.some((a) => a !== 'adamzettel');
  if (owner) return 'owner (some commits)';
  return agent ? `agent: ${agent}` : 'agent (committed by an agent session; no agent line)';
}

/** One post, all tables. `run` is { out, cells }, `nat` the native result or null. */
export function buildPost({ file, src, run, nat = null, refsDir = REFS, recordedOuts = [] }) {
  const m = file.match(/^(\d{4}-\d{2}-\d{2})-([a-z0-9-]+)\.viv$/);
  const [, date, slug] = m || [null, '', basename(file, '.viv')];
  const h = headerOf(src);
  const cells = splitNotebook(src);
  const sentences = sentencesOf(src, cells);
  // test names are claims too: the prose a reader sees on every card
  const code = cells.filter((c) => c.kind === 'code').map((c) => c.text).join('\n');
  const { env, consts, literals } = constantsOf(code);
  const r = parseRun(run.out);
  const tests = testsOf(src, env).map((t) => ({
    ...t,
    pass: r.pass.includes(t.name),
    interp: r.pass.includes(t.name) ? 'pass' : r.fail.some((f) => f.split(' -- ')[0] === t.name || f.startsWith(t.name)) ? 'fail' : 'missing',
    native: nat ? (nat.pass.includes(t.name) ? 'pass' : nat.fail.some((f) => f.startsWith(t.name)) ? 'fail' : 'missing') : null,
  }));
  for (const t of tests) {
    const nums = numbersIn(t.name);
    const lineNo = src.split('\n').findIndex((l) => l.includes(`test "${t.name}"`)) + 1;
    sentences.push({ id: sentenceId(t.name), cell: null, origin: 'test', section: '', line: lineNo, text: t.name, kind: nums.numbers.length ? 'numeric' : 'procedural', numbers: nums.numbers, names: nums.names });
  }
  // the reference scripts the post names, and its own posts/refs/<slug>.py
  const refFiles = new Set();
  if (existsSync(join(refsDir, `${slug}.py`))) refFiles.add(`${slug}.py`);
  for (const mm of src.matchAll(/posts\/refs\/([\w.-]+?\.py)/g)) if (existsSync(join(refsDir, mm[1]))) refFiles.add(mm[1]);
  const refs = [];
  const refValues = [];
  for (const py of refFiles) {
    const out = py.replace(/\.py$/, '.out');
    const has = existsSync(join(refsDir, out));
    refs.push({ py: `posts/refs/${py}`, out: has ? `posts/refs/${out}` : null });
    if (has) for (const v of parseRefOut(readFileSync(join(refsDir, out), 'utf8'))) refValues.push({ file: `refs/${out}`, ...v });
  }
  const recorded = [];
  for (const ro of recordedOuts) for (const s of parseRun(ro.text).shown) recorded.push({ scale: ro.scale, ...s });
  // citations: the whole prose and the header's `reference:` field
  const citations = [];
  for (const s of sentences) for (const c of citationsIn(s.text)) { s.citations = [...(s.citations || []), c.text]; if (!citations.some((x) => x.text === c.text)) citations.push({ ...c, sentence: s.id }); }
  if (h.reference) for (const c of citationsIn(h.reference)) if (!citations.some((x) => x.text === c.text)) citations.push({ ...c, sentence: null, field: 'reference' });
  // a named scale's settings are inputs too: `scale large: N=2000`
  for (const sc of scalesOf(src)) for (const [k, v] of Object.entries(sc.sets)) if (Number.isFinite(Number(v))) consts.push({ name: `${k} (scale ${sc.name})`, value: Number(v) });
  const cands = candidates({ tests, shown: r.shown, printed: r.printed, refs: refValues, consts, literals, recorded });
  // a sentence's own citations make its unmatched numbers `cited`
  for (const s of sentences) {
    linkPost({ sentences: [s], cands, tests, citations: (s.citations || []).map((t) => citations.find((c) => c.text === t)).filter((c) => c && c.kind !== 'software') });
  }
  // figures, per cell
  const figures = [];
  for (const [ci, out] of Object.entries(run.cells || { 0: run.out })) for (const f of parseFigures(out)) figures.push({ cell: Number(ci), ...f });
  const values = [];
  for (const [ci, out] of Object.entries(run.cells || { 0: run.out })) {
    const pr = parseRun(out);
    for (const s of pr.shown) values.push({ cell: Number(ci), kind: 'shown', name: s.name, index: s.index, value: s.value });
    for (const p of pr.printed) values.push({ cell: Number(ci), kind: 'printed', line: p.line, value: p.value });
  }

  const numbers = [];
  for (const s of sentences) s.numbers.forEach((n, k) => numbers.push({ id: `${s.id}.${k}`, post: slug, sentence: s.id, line: s.line, origin: s.origin, ...n, ...(n.role === 'count' ? { trivial: true } : {}) }));
  const counts = { sentences: sentences.length, numbers: numbers.filter((n) => n.status !== 'skip').length };
  for (const st of STATUSES) counts[st] = numbers.filter((n) => n.status === st).length;
  return {
    post: {
      slug, file, date: h.date || date, title: h.title, kind: h.kind, lane: h.lane, channel: h.channel || null, tags: h.tags,
      agent: h.agent || null, author: null, from: h.from || null, edge: h.from ? h.edge || 'fork' : null, solves: h.solves || null,
      reference: h.reference || null, equation: h.equation || null, scales: scalesOf(src).map((s) => s.name),
      refs, counts, source: createHash('sha256').update(src).digest('hex').slice(0, 16),
      tests: { total: tests.length, interp: tests.filter((t) => t.interp === 'pass').length, native: nat ? tests.filter((t) => t.native === 'pass').length : null },
    },
    cells: cells.map((c, i) => ({ post: slug, index: i, kind: c.kind, start: c.start, end: c.end, ...(c.kind === 'code' ? { empty: !!c.empty } : {}) })),
    sentences: sentences.map((s) => ({ id: s.id, post: slug, cell: s.cell, origin: s.origin, section: s.section, line: s.line, kind: s.kind, text: s.text, numbers: s.numbers.length, ...(s.citations ? { citations: s.citations } : {}) })),
    numbers,
    tests: tests.map((t) => ({ post: slug, name: t.name, interp: t.interp, native: t.native, asserts: t.asserts.map(({ names, ...a }) => ({ ...a, on: names })) })),
    values: values.map((v) => ({ post: slug, ...v })),
    refValues: refValues.map((v) => ({ post: slug, ...v })),
    figures: figures.map((f) => ({ post: slug, ...f })),
    citations: citations.map((c) => ({ post: slug, ...c })),
  };
}

async function main() {
  const args = process.argv.slice(2);
  const run = runner(WWW);
  const fileArg = args.includes('--file') ? args[args.indexOf('--file') + 1] : null;
  if (fileArg) {
    const src = readFileSync(fileArg, 'utf8');
    const as = args.includes('--as') ? args[args.indexOf('--as') + 1] : basename(fileArg);
    const out = await run.interp(as, src);
    const rec = buildPost({ file: as, src, run: out });
    console.log(JSON.stringify(rec, null, 1));
    return;
  }
  const NATIVE = args.includes('--native');
  const files = readdirSync(SRC).filter((f) => /^\d{4}-\d{2}-\d{2}-[a-z0-9-]+\.viv$/.test(f)).sort();
  const db = { schema: 1, about: 'Every post as a record whose claims can be tested: see www/scripts/claims-build.mjs.', statuses: STATUSES, posts: [], cells: [], sentences: [], numbers: [], tests: [], values: [], refValues: [], figures: [], citations: [] };
  // the previous build's native results, so a build without --native keeps them
  const prev = existsSync(OUT) ? JSON.parse(readFileSync(OUT, 'utf8')) : null;
  for (const file of files) {
    const src = readFileSync(join(SRC, file), 'utf8');
    const out = await run.interp(file, src);
    let nat = NATIVE ? run.native(file, src) : null;
    const recordedOuts = scalesOf(src).map((s) => ({ scale: s.name, path: join(SRC, `${file.replace(/\.viv$/, '')}.${s.name}.out`) })).filter((x) => existsSync(x.path)).map((x) => ({ scale: x.scale, text: readFileSync(x.path, 'utf8') }));
    const rec = buildPost({ file, src, run: out, nat, recordedOuts });
    if (!NATIVE && prev && prev.posts.find((p) => p.file === file)?.source === rec.post.source) {
      // keep the native results of the last --native build while the post is unchanged
      for (const t of rec.tests) {
        const old = prev.tests.find((x) => x.post === t.post && x.name === t.name);
        if (old && old.native) t.native = old.native;
      }
      rec.post.tests.native = rec.tests.some((t) => t.native) ? rec.tests.filter((t) => t.native === 'pass').length : null;
    }
    rec.post.author = provenance(file, rec.post.agent);
    for (const k of Object.keys(rec)) {
      if (k === 'post') db.posts.push(rec.post);
      else db[k].push(...rec[k]);
    }
    const c = rec.post.counts;
    process.stdout.write(`claims-build: ${rec.post.slug.slice(0, 44).padEnd(44)} ${String(c.sentences).padStart(3)} sentences ${String(c.numbers).padStart(3)} numbers  ${c.checked} checked ${c.computed} computed ${c.orphan} orphan${c.mismatch ? `  ${c.mismatch} MISMATCH` : ''}\n`);
  }
  mkdirSync(dirname(OUT), { recursive: true });
  writeFileSync(OUT, `${JSON.stringify(db, null, 0).replace(/\},\{"/g, '},\n{"')}\n`);
  const nums = db.numbers.filter((n) => n.status !== 'skip');
  const hist = Object.fromEntries(STATUSES.map((s) => [s, nums.filter((n) => n.status === s).length]));
  console.log(`claims-build: ${db.posts.length} posts, ${db.sentences.length} sentences, ${nums.length} numbers -> src/data/claims.json`);
  console.log(`claims-build: ${Object.entries(hist).map(([k, v]) => `${v} ${k}`).join(', ')}`);
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
