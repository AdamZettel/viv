// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// feed-build -- the feed, generated from the posts.
//
// A post is one Viv program, posts/<YYYY-MM-DD>-<slug>.viv, and posting is
// committing the file. Its `#!` header is the card:
//
//     #! post     <title>
//     #! date     <YYYY-MM-DD, the same as the filename's>
//     #! tags     <comma separated>
//     #! summary  <one to three sentences; $math$, `code`, *emphasis*>
//     #! agent    <name>        optional: the post was drafted with an AI agent
//     #! kind     demo | article | ladder | problem | solution   (demo if absent)
//     #! solves   <slug>        a solution: the problem it solves
//     #! channel  <slug>        where it goes: a channel from channels/<slug>.md
//     #! equation  <LaTeX>       the one formula the post is about, shown on the card
//     #! from     <slug>        the post this one was derived from
//     #! edge     fork | fix | extend | rewrite | adversarial-review | validate
//                               what it did to it: a claim, checked by running both (src/lib/lineage.js)
//     #! | notes, shown on the post's own page
//
// This script RUNS every post through viv.wasm, the interpreter a reader's tab
// runs, and refuses to build the site if a post
//   - fails a test, or has none          (a demo with no check is a picture;
//                                         a PROBLEM is the one kind whose tests fail)
//   - draws no figure and stages no animation, or stages a broken one
//                                        (demos and articles; the other kinds may draw)
//   - is a ladder that breaks a ladder's rules (viv_ladder reports them), or a
//     solution that drops or edits one of its problem's tests
//   - takes longer than BUDGET_SECONDS of CPU   (a reader will press Run)
//   - emits more than FIGURE_BYTES of figure records (the feed page carries them)
// What a card shows -- the figure, the numbers, the passing tests -- is that
// run's output. `posts/check.sh` is the other half: both engines, same output.
//
// CLAIMS are posts too: each claim in claims/<topic>/<slug>.viv (statement,
// Viv program, vipyr twin; tools/claims/run.py records their runs) is a post
// of kind `claim` in the claims channel, carried by its recorded run rather
// than run here (scripts/claims/claimfiles.mjs). An article cites one with
// `claims: <slug>, ...` in its header and `[claim: <slug>]` in its prose
// (src/lib/claimRefs.js), and the build refuses a citation of a claim that is
// missing or not checked.
//
// Emits www/src/data/feed.json and stages the sources under /programs/posts
// (the claims' under /programs/claims).
import { readFileSync, writeFileSync, readdirSync, mkdirSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { prose, equation } from './prose.mjs';
import { isAnimRecord, buildTimeline } from '../src/lib/animParse.js';
import { judge, judgeRecorded, withoutHeader, scalesOf, recordedNotes, meta, BUDGET_SECONDS } from '../src/lib/postCheck.js';
import { expandUses } from '../src/lib/uses.js';
import { ladderRenderer } from './ladder-render.mjs';
import { parseChannel, syllabusSlugs } from '../src/lib/channels.js';
import { checkEdge, challengeBlocks, outcomes, defsOf, normalizeSource } from '../src/lib/lineage.js';
import { splitNotebook, isNotebook, runUnits } from '../src/lib/notebook.js';
import { renderMarkdown, texToPlain, texToAscii, MATH_SPLIT, isIndentedBlock, indentedBlockHtml } from '../src/lib/richText.js';
import katex from 'katex';
import { buildClaims, headerFields, TOPIC_TITLES } from './claims/claimfiles.mjs';
import { declaredClaims, placedClaims, claimRefErrors, claimHtml, withClaimRefs, CLAIM_REF } from '../src/lib/claimRefs.js';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const SRC = join(ROOT, 'posts');
const CHANNELS = join(ROOT, 'channels');
const STAGE = join(WWW, 'public/programs/posts');

// modules: the repository's dft/, pde/, lib/, lang/viv/examples (as examples/), and posts by slug
const DIRS = { dft: 'dft', pde: 'pde', lib: 'lib', examples: 'lang/viv/examples' };
// staged beside them, so the workbench opens every program /gallery lists:
// the root examples/ shares examples/ with lang/viv/examples (no names collide).
// claims/ is staged whole: a claim's program may import a module beside it
// (claims/math/qrlib.viv), which is no claim of its own.
const MORE_DIRS = [
  ['catalog', 'catalog'], ['ports', 'ports'], ['stat', 'stat'], ['study', 'study'], ['gallery', 'gallery'], ['examples', 'examples'],
  ['claims', 'claims'],
];
const postFiles = Object.fromEntries((existsSync(SRC) ? readdirSync(SRC) : []).filter((f) => f.endsWith('.viv')).map((f) => [f.replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.viv$/, ''), f]));
const resolve = async (name) => {
  if (name.startsWith('post:')) {
    const f = postFiles[name.slice(5)];
    if (!f) throw new Error(`module \`${name}\`: no post by that slug in posts/`);
    return readFileSync(join(SRC, f), 'utf8');
  }
  const [head, ...rest] = name.split('/');
  const dirs = [DIRS[head], ...MORE_DIRS.filter(([h]) => h === head).map(([, d]) => d)].filter(Boolean);
  const file = rest.length ? dirs.map((d) => join(ROOT, d, `${rest.join('/')}.viv`)).find(existsSync) || null : null;
  if (!file || !existsSync(file)) throw new Error(`module \`${name}\`: no module by that name`);
  return readFileSync(file, 'utf8');
};
// stage the modules the browser may fetch, and an index of posts by slug
// Subdirectories too: `import hf.hf` names lib/hf/hf.viv, and that module
// uses lib/hfref/*. Staging only the top level left every post built on lib/hf
// unable to run in the browser (it fetched /programs/lib/hf/hf.viv and got a 404).
function stageTree(from, to) {
  mkdirSync(to, { recursive: true });
  for (const f of readdirSync(from, { withFileTypes: true })) {
    if (f.isDirectory()) stageTree(join(from, f.name), join(to, f.name));
    else if (f.name.endsWith('.viv')) writeFileSync(join(to, f.name), readFileSync(join(from, f.name)));
  }
}
for (const [head, dir] of [...Object.entries(DIRS), ...MORE_DIRS]) {
  const from = join(ROOT, dir);
  if (!existsSync(from)) continue;
  stageTree(from, join(WWW, 'public/programs', head));
}
// the data files staged programs read by their repository path
// (read_case("lib/power/cases/case118.m")): the workbench attaches the ones a
// program names before it runs it (src/lib/programFiles.js), since a browser
// has no file system. Listed in /programs/data-index.json.
const DATA_DIRS = ['lib/power/cases', 'lib/stats/data'];
const DATA_EXT = /\.(m|raw|dat)$/;
function stageData(rel) {
  const from = join(ROOT, rel);
  if (!existsSync(from)) return;
  for (const f of readdirSync(from, { withFileTypes: true })) {
    if (f.isDirectory()) stageData(`${rel}/${f.name}`);
    else if (DATA_EXT.test(f.name)) {
      mkdirSync(join(WWW, 'public/programs', rel), { recursive: true });
      writeFileSync(join(WWW, 'public/programs', rel, f.name), readFileSync(join(from, f.name)));
    }
  }
}
for (const d of DATA_DIRS) stageData(d);
mkdirSync(STAGE, { recursive: true });
writeFileSync(join(STAGE, 'index.json'), JSON.stringify(postFiles));

const { instance } = await WebAssembly.instantiate(readFileSync(join(WWW, 'public/viv/viv.wasm')), {});
const ex = instance.exports;
function call(fn, text) {
  const enc = new TextEncoder().encode(text);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const packed = fn(ptr, enc.length);
  return new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
}
const renderLadder = ladderRenderer(ex);
// one code cell after the cells above it, through the interpreter's incremental
// session: what the notebook worker does in the reader's tab (public/nb-worker.js)
function callCell(prefix, cell) {
  const enc = new TextEncoder();
  const pb = enc.encode(prefix);
  const cb = enc.encode(cell);
  const ptr = ex.viv_alloc(pb.length + cb.length);
  const mem = new Uint8Array(ex.memory.buffer, ptr, pb.length + cb.length);
  mem.set(pb, 0);
  mem.set(cb, pb.length);
  const packed = ex.viv_run_cell(ptr, pb.length + cb.length, pb.length);
  return new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
}
// a version's address (src/lib/lineage.js says why the whole file)
const versionHash = (src) => createHash('sha256').update(normalizeSource(src)).digest('hex').slice(0, 12);
// A summary as plain text (the feed's <summary>, a page's description): each
// $formula$ as plain-text math (10^-14, sqrt(2)), emphasis and code marks dropped,
// a [words](address) link as its words.
const plainProse = (src) => String(src || '').replace(/\\\$/g, '\u0000').split(MATH_SPLIT)
  .map((part) => (/^\$[\s\S]*\$$/.test(part) ? texToAscii(part.replace(/^\$+|\$+$/g, '')) : part.replace(/^`|`$/g, '').replace(/\*\*?([^*]+)\*\*?/g, '$1').replace(/\[([^\]]+)\]\([^)\s]+\)/g, '$1')))
  .join('').replace(/\u0000/g, '$').replace(/ -- /g, ' — ').replace(/\s+/g, ' ').trim();
const plain = (html) => html.replace(/<annotation[^>]*>[\s\S]*?<\/annotation>/g, '').replace(/<[^>]+>/g, '').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>');

const errors = [];
const posts = [];
// what the lineage checks need of each post, beside its card: the whole source,
// its run, and the compiler's per-def hashes
const runs = new Map();
mkdirSync(STAGE, { recursive: true });
// channels: channels/<slug>.md, each a syllabus over the posts that name it
const channels = [];
for (const file of (existsSync(CHANNELS) ? readdirSync(CHANNELS) : []).filter((f) => f.endsWith('.md')).sort()) {
  const slug = file.replace(/\.md$/, '');
  const c = parseChannel(readFileSync(join(CHANNELS, file), 'utf8'));
  for (const e of c.errors) errors.push(`channels/${file}: ${e}`);
  channels.push({ slug, file, ...c });
}
writeFileSync(join(STAGE, 'channels.json'), JSON.stringify(channels.map((c) => ({ slug: c.slug, title: c.title, summary: c.summary }))));
// the claims, with their recorded runs, by slug
// with the code maps, so a claim links a map only where /vipyr/map/<fn> exists
const codemapPath = join(WWW, 'src/data/codemap.json');
const claimList = buildClaims(ROOT, existsSync(codemapPath) ? JSON.parse(readFileSync(codemapPath, 'utf8')) : null);
// a claim's sentence is typeset ($math$); its card and badges take the rendered form
// (the short title, then the key numbers; the full statement is on its page)
const claimCardHtml = (c) => (c.title ? `<span class="cr-title">${prose(c.title)}</span>${c.line ? ` <span class="cr-line">${prose(c.line)}</span>` : ''}` : prose(c.claim));
const claimsBySlug = Object.fromEntries(claimList.map((c) => [c.slug, { ...c, claim: c.title || c.claim, claimHtml: claimCardHtml(c) }]));
const files = existsSync(SRC) ? readdirSync(SRC).filter((f) => f.endsWith('.viv')).sort() : [];
for (const file of files) {
  const m = file.match(/^(\d{4}-\d{2}-\d{2})-([a-z0-9-]+)\.viv$/);
  if (!m) { errors.push(`${file}: a post's file name must be <YYYY-MM-DD>-<slug>.viv, the slug in lowercase words joined by hyphens`); continue; }
  const [, fileDate, slug] = m;
  const src = readFileSync(join(SRC, file), 'utf8');
  writeFileSync(join(STAGE, file), src);
  let expanded = src;
  try { expanded = await expandUses(src, resolve); } catch (e) { errors.push(`${slug}: ${e.message}`); continue; }
  // the budget is the run's CPU time, which a loaded machine does not stretch (the run
  // is this process's one thread); its wall time has a cap of 10 times the budget, for a
  // run that blocks without working (tools/gate/load.py)
  const t0 = performance.now();
  const c0 = process.cpuUsage();
  const out = call(ex.viv_run, expanded);
  const cpu = process.cpuUsage(c0);
  const seconds = (cpu.user + cpu.system) / 1e6;
  const wall = (performance.now() - t0) / 1000;
  if (wall > 10 * BUDGET_SECONDS) errors.push(`${slug}: the program ran for ${wall.toFixed(1)} s of wall time (${seconds.toFixed(1)} s of CPU), over 10 times the ${BUDGET_SECONDS} s budget: stuck`);
  // what the rules need beside the run: a ladder's rung-by-rung report, a solution's problem
  const extra = {};
  let ladder = null;
  if (meta(src, 'kind') === 'ladder') {
    const r = renderLadder(expanded, { stageDir: join(STAGE, slug), openBase: `posts/${slug}` });
    extra.ladder = r.report || { error: r.errors[0] };
    ladder = r.view;
  }
  if (meta(src, 'solves')) {
    const pf = postFiles[meta(src, 'solves')];
    extra.problem = pf ? readFileSync(join(SRC, pf), 'utf8') : null;
  }
  // the same rules the composer shows and the server applies (src/lib/postCheck.js)
  extra.channels = channels.map((c) => c.slug);
  const verdict = judge(src, out, seconds, extra);
  for (const e of verdict.errors) errors.push(`${slug}: ${e}`);
  if (verdict.header.date && verdict.header.date !== fileDate) errors.push(`${slug}: the header says \`date: ${verdict.header.date}\` but the file name says ${fileDate}`);
  if (posts.some((p) => p.slug === slug)) errors.push(`${slug}: two posts share this slug`);
  const anim = out.split('\n').filter(isAnimRecord);
  if (anim.length) for (const e of buildTimeline(anim).errors) errors.push(`${slug}: animation: ${e}`);
  const { tests, failed, figure, shown, header } = verdict;

  // The post's address: a hash of the compiler's content addresses of every
  // definition and of the top level. Comments and spacing do not move it; any
  // change to what the program IS does. (The compiler has no whole-program
  // address of its own: a call is hashed by the callee's name.)
  const addresses = [...call(ex.viv_hash, expanded).matchAll(/^([0-9a-f]{8})\s+body ([0-9a-f]{8})\s+(.+)$/gm)].map((h) => `${h[2]} ${h[3]}`).sort();
  const address = createHash('sha256').update(addresses.join('\n')).digest('hex').slice(0, 10);
  const hashText = call(ex.viv_hash, expanded);
  runs.set(slug, { src, out, defs: defsOf(hashText) });

  let summaryHtml = '';
  let notes = [];
  let equationHtml = '';
  try {
    summaryHtml = prose(header.summary);
    // a paragraph that is only `[claim: slug]` is the claim's card; inside a sentence, a badge
    notes = header.notes.map((n) => {
      const only = n.trim().match(/^\[claim:\s*([a-z0-9-]+)\s*\]$/);
      if (isIndentedBlock(n)) return indentedBlockHtml(n);
      return only ? claimHtml(claimsBySlug[only[1]], only[1], true) : withClaimRefs(prose(n), claimsBySlug);
    });
    if (header.equation) equationHtml = equation(header.equation);
  } catch (e) {
    errors.push(`${slug}: ${e.message}`);
  }
  const cites = declaredClaims(meta(src, 'claims'));
  for (const e of claimRefErrors(cites, placedClaims(src), claimsBySlug)) errors.push(`${slug}: ${e}`);
  const body = withoutHeader(src);
  // A post with prose cells (`###` blocks after its header) opens as a notebook
  // on its page. Its prose is rendered there by the page; here a formula KaTeX
  // cannot parse fails the build, as it does in the header.
  const cells = splitNotebook(src);
  const notebook = isNotebook(cells) ? { prose: cells.filter((c) => c.kind === 'prose').length, code: cells.filter((c) => c.kind === 'code' && !c.empty).length } : null;
  if (notebook) {
    const strict = { renderToString: (tex, o) => { try { return katex.renderToString(tex, { ...o, throwOnError: true }); } catch (e) { errors.push(`${slug}: a prose cell's math does not parse: ${e.message.split('\n')[0]}`); return ''; } } };
    for (const c of cells) if (c.kind === 'prose') renderMarkdown(c.body, strict);
  }
  // What each code cell prints, for the page's Article view, which shows a
  // cell's figures and results under it before anyone presses Run. A file of
  // one code cell prints what the whole run printed; a notebook's cells are run
  // in order, each continuing the session, as the page runs them.
  const cellOut = {};
  const units = runUnits(cells);
  if (units.length === 1) cellOut[units[0].index] = out;
  else {
    const seen = new Set();
    const texts = [];
    for (const u of units) texts.push(await expandUses(u.text, resolve, [], seen));
    texts.forEach((t, j) => { cellOut[units[j].index] = callCell(texts.slice(0, j).join('\n'), t); });
  }
  writeFileSync(join(STAGE, `${file.replace(/\.viv$/, '')}.cells.json`), JSON.stringify(cellOut));
  const doc = JSON.parse(call(ex.viv_doc, expanded));
  // recorded runs beside the post: <file>.<scale>.out, the output of `viv run --set ...`
  // the author did elsewhere. Judged for what can be judged: tests pass, a figure.
  const recorded = [];
  const rnotes = recordedNotes(src);
  for (const sc of scalesOf(src)) {
    const side = join(SRC, `${file.replace(/\.viv$/, '')}.${sc.name}.out`);
    if (!existsSync(side)) continue;
    const jr = judgeRecorded(readFileSync(side, 'utf8'));
    for (const e of jr.errors) errors.push(`${slug}: recorded run "${sc.name}": ${e}`);
    recorded.push({ scale: sc.name, sets: sc.sets, note: rnotes[sc.name] || '', figure: jr.figure, tests: jr.tests, shown: jr.shown, at: `${fileDate}T12:00:00.000Z` });
  }
  posts.push({
    scales: verdict.scales.map((sc) => ({ name: sc.name, sets: sc.sets })),
    recorded,
    functions: doc.functions.map((f) => ({ name: f.name, what: f.what || '' })),
    slug, file, date: fileDate, address,
    version: versionHash(src),
    edge: header.from ? header.edge || 'fork' : null,
    title: header.title, titleHtml: prose(header.title || ''), titleText: texToPlain(header.title || ''),
    kind: header.kind,
    lane: header.lane,
    forkedFrom: header.from || null,
    solves: header.solves || null,
    cites,
    channel: header.channel || null,
    channelTitle: null, // filled in below, once the channels are parsed
    ladder: ladder && { top: ladder.top, rungs: ladder.rungs },
    failed: header.kind === 'problem' ? failed : [],
    tags: header.tags,
    agent: header.agent || null,
    summaryHtml, summaryText: header.summary ? plainProse(header.summary) : plain(summaryHtml), notes,
    equation: header.equation || null, equationHtml,
    source: body, lines: body.replace(/\n$/, '').split('\n').length, notebook,
    tests, shown, figure,
    seconds: Math.round(seconds * 100) / 100,
  });
  console.log(`feed-build: ${slug.padEnd(28)} ${String(tests.length).padStart(2)} tests pass  ${seconds.toFixed(2)} s CPU  ${(figure.length / 1024).toFixed(0).padStart(3)} KB figure  ${address}`);
}

if (errors.length) {
  console.error(`feed-build: ${errors.length} problem${errors.length > 1 ? 's' : ''}:\n  ${errors.join('\n  ')}`);
  process.exit(1);
}
// newest first; posts of one day in filename order
posts.sort((a, b) => (a.date === b.date ? a.file.localeCompare(b.file) : b.date.localeCompare(a.date)));
const author = JSON.parse(readFileSync(join(SRC, 'author.json'), 'utf8'));
// lineage among the committed posts: a fork's parent, and each post's forks
for (const p of posts) {
  const parent = p.forkedFrom ? posts.find((q) => q.slug === p.forkedFrom) : null;
  if (p.forkedFrom && !parent) errors.push(`${p.slug}: \`from: ${p.forkedFrom}\` names no post in posts/`);
  p.parent = parent ? { slug: parent.slug, title: parent.title, author: { handle: author.handle, name: author.name } } : null;
  p.forks = posts.filter((q) => q.forkedFrom === p.slug).length;
  // a solution's problem, and a problem's solutions
  const problem = p.solves ? posts.find((q) => q.slug === p.solves) : null;
  if (p.solves && !problem) errors.push(`${p.slug}: \`solves: ${p.solves}\` names no post in posts/`);
  p.problem = problem ? { slug: problem.slug, title: problem.title, author: { handle: author.handle, name: author.name } } : null;
  p.solutions = posts.filter((q) => q.solves === p.slug).map((q) => ({ slug: q.slug, title: q.title, date: q.date, author: { handle: author.handle, name: author.name }, agent: q.agent }));
}
// ---- lineage: each committed post is one version; `from:` + `edge:` are the edges ----
// Every claimed edge is checked by running programs (src/lib/lineage.js), and a
// false claim fails the build the way a failing test does.
const runViv = async (src) => call(ex.viv_run, await expandUses(src, resolve));
const bySlug = new Map(posts.map((p) => [p.slug, p]));
const edges = [];
for (const p of posts) {
  if (!p.forkedFrom || !bySlug.has(p.forkedFrom)) continue;
  const parent = bySlug.get(p.forkedFrom);
  const verdict = await checkEdge(p.edge, runs.get(parent.slug), runs.get(p.slug), runViv);
  for (const r of verdict.reasons) errors.push(`${p.slug}: \`edge: ${p.edge}\` on ${parent.slug} is refused: ${r}`);
  edges.push({ from: parent.version, to: p.version, kind: p.edge, facts: verdict.facts, at: p.date, fromSlug: parent.slug, toSlug: p.slug });
}
// families: the posts joined by edges. A post with none is a family of one, and
// its page still shows the history: one version, its checks, who owns it.
const familyOf = new Map(posts.map((p) => [p.slug, p.slug]));
const findFamily = (s) => (familyOf.get(s) === s ? s : findFamily(familyOf.get(s)));
for (const e of edges) familyOf.set(findFamily(e.toSlug), findFamily(e.fromSlug));
const lineage = {};
for (const p of posts) {
  const fam = findFamily(p.slug);
  p.family = fam;
  const r = runs.get(p.slug);
  const results = { pass: p.tests, fail: p.failed.map((f) => f.split(' -- ')[0]) };
  (lineage[fam] ||= { versions: [], edges: [], challenges: [] }).versions.push({
    hash: p.version, slug: p.slug, title: p.title, kind: p.kind, lane: p.lane, agent: p.agent, author: { handle: author.handle, name: author.name },
    at: p.date, current: true, tests: results.pass, failed: results.fail, defs: r.defs, reviews: [],
    reference: p.edge === 'validate' ? meta(r.src, 'reference') : '',
  });
}
for (const e of edges) lineage[findFamily(e.toSlug)].edges.push({ from: e.from, to: e.to, kind: e.kind, facts: e.facts, at: e.at });
// an adversarial review's tests, run against every other version of its family:
// what is still open where, and which version closed it
for (const [fam, L] of Object.entries(lineage)) {
  for (const e of L.edges.filter((x) => x.kind === 'adversarial-review')) {
    const parent = posts.find((p) => p.version === e.from);
    const review = posts.find((p) => p.version === e.to);
    const blocks = challengeBlocks(runs.get(parent.slug).src, runs.get(review.slug).src);
    const found = new Set(e.facts.findings || []);
    const per = {};
    for (const v of L.versions.filter((x) => x.kind !== 'problem')) per[v.hash] = await outcomes(runs.get(v.slug).src, blocks.filter((b) => found.has(b.name)), runViv);
    for (const b of blocks.filter((x) => found.has(x.name))) {
      L.challenges.push({ edge: { from: e.from, to: e.to }, test: b.name, results: Object.fromEntries(Object.entries(per).map(([h, res]) => [h, res[b.name].result])) });
    }
  }
  if (L.versions.length > 1) console.log(`feed-build: lineage of ${fam}: ${L.versions.length} versions, edges ${L.edges.map((e) => e.kind).join(', ')}, ${L.challenges.length} adversarial tests`);
}
// ---- claims: one post each, in the claims channel ----
const CLAIM_STAGE = join(WWW, 'public/programs/claims');
mkdirSync(CLAIM_STAGE, { recursive: true });
const firstLines = (text, n) => {
  const lines = String(text || '').split('\n');
  return lines.slice(0, n).join('\n') + (lines.length > n ? '\n…' : '');
};
const tally = (run) => (run ? { pass: run.tests.filter((t) => t.pass).length, fail: run.tests.filter((t) => !t.pass).length } : null);
const PREVIEW_LINES = 8;
// the first sentence of a scope or reference, for the card; the page shows all of it
const firstSentence = (s) => { const m = String(s || '').match(/^(.+?\.)(\s|$)/); return m ? m[1] : String(s || ''); };
// The numbers a claim's page tabulates: each value a test checks or both programs
// show, with the tests whose bodies name it. A long array is cut to its first values.
const TABLE_CHARS = 36;
const shortValue = (v) => {
  if (v == null) return '';
  const t = String(v);
  // Viv's show prints 6 significant digits; a twin's full float is cut to the same
  if (/^-?\d+\.\d{7,}(e[-+]?\d+)?$/i.test(t)) return String(Number(Number(t).toPrecision(6)));
  if (!t.startsWith('[') || t.length <= TABLE_CHARS) return t;
  const parts = t.slice(1, -1).split(',').map((x) => x.trim());
  return `[${parts.slice(0, 3).join(', ')}, … ] (${parts.length} values)`;
};
const testBodies = (src) => [...src.matchAll(/^test\s+"([^"]+)"\s*\{([\s\S]*?)^\}/gm)].map((m) => ({ name: m[1], body: m[2] }));
function claimTable(c) {
  const vv = c.viv.run ? c.viv.run.values : {};
  const pv = c.vipyr && c.vipyr.run ? c.vipyr.run.values : {};
  const tests = testBodies(c.viv.source);
  const rows = [];
  for (const name of Object.keys(vv)) {
    const by = tests.filter((t) => new RegExp(`\\b${name.replace(/[.[\]]/g, '\\$&')}\\b`).test(t.body)).map((t) => t.name);
    if (!by.length && !(name in pv)) continue;
    rows.push({ name, viv: shortValue(vv[name]), vipyr: name in pv ? shortValue(pv[name]) : '', tests: by });
  }
  return rows;
}
for (const c of claimList) {
  if (posts.some((p) => p.slug === c.slug)) { errors.push(`claims/${c.file}: a post in posts/ has the slug ${c.slug} too`); continue; }
  // the programs as the card opens them, and the .viv files the workbench opens
  writeFileSync(join(CLAIM_STAGE, `${c.slug}.json`), JSON.stringify({
    viv: { source: c.viv.source, output: c.viv.run ? c.viv.run.output : '', seconds: c.viv.run ? c.viv.run.seconds : null },
    vipyr: c.vipyr ? { source: c.vipyr.source, output: c.vipyr.run ? c.vipyr.run.output : '', seconds: c.vipyr.run ? c.vipyr.run.seconds : null } : null,
  }));
  const rel = c.file.replace(/^claims\//, '');
  mkdirSync(dirname(join(CLAIM_STAGE, rel)), { recursive: true });
  writeFileSync(join(CLAIM_STAGE, rel), readFileSync(join(ROOT, c.file)));
  const h = headerFields(readFileSync(join(ROOT, c.file), 'utf8'));
  const citedBy = posts.filter((p) => (p.cites || []).includes(c.slug)).map((p) => ({ slug: p.slug, title: p.title }));
  posts.push({
    slug: c.slug, file: c.file, date: c.ran || '1970-01-01', address: c.sha.slice(0, 10), version: c.sha,
    kind: 'claim', lane: c.lane, title: c.title || c.claim, titleHtml: prose(c.title || c.claim), titleText: texToPlain(c.title || c.claim), channel: 'claims', channelTitle: null,
    tags: [...new Set([c.topic, c.section, ...(h.tags ? h.tags.split(',').map((t) => t.trim()) : [])])].filter(Boolean),
    agent: c.by === 'claude' ? 'claude' : null,
    summaryHtml: c.line ? prose(c.line) : '', summaryText: plainProse(c.line || c.claim), notes: [], equation: null, equationHtml: '',
    tests: c.viv.run ? c.viv.run.tests.filter((t) => t.pass).map((t) => t.name) : [], failed: [], shown: [], figure: (c.viv.run && c.viv.run.figure) || '', recorded: [], scales: [], functions: [],
    forkedFrom: null, edge: null, parent: null, forks: 0, solves: null, problem: null, solutions: [], ladder: null, notebook: null, lines: 0, seconds: c.viv.run ? c.viv.run.seconds : 0,
    claim: {
      statement: c.claim, statementHtml: prose(c.claim), line: c.line, lineHtml: prose(c.line), whyHtml: c.why ? prose(c.why) : '', table: claimTable(c),
      status: c.status, topic: c.topic, topicTitle: c.topicTitle, section: c.section, scope: c.scope, reference: c.reference, cost: c.cost, ran: c.ran,
      scopeHtml: prose(c.scope), referenceHtml: prose(c.reference), scopeFirstHtml: prose(firstSentence(c.scope)), referenceFirstHtml: prose(firstSentence(c.reference)),
      vivOnly: c.vivOnly, file: c.file, twin: c.twin, agreement: c.agreement, links: c.links, citedBy,
      preview: firstLines(c.viv.source, PREVIEW_LINES), lines: c.viv.source.replace(/\n$/, '').split('\n').length,
      pyPreview: c.vipyr ? firstLines(c.vipyr.source, PREVIEW_LINES) : '', pyLines: c.vipyr ? c.vipyr.source.replace(/\n$/, '').split('\n').length : 0,
      vivTests: tally(c.viv.run), pyTests: c.vipyr ? tally(c.vipyr.run) : null,
    },
  });
}
posts.sort((a, b) => (a.date === b.date ? a.file.localeCompare(b.file) : b.date.localeCompare(a.date)));
// the claims channel's syllabus: channels/claims.md's sections, then one section per topic for the rest
const claimChannel = channels.find((ch) => ch.slug === 'claims');
if (claimChannel) {
  const listedClaims = new Set(syllabusSlugs(claimChannel));
  for (const c of claimList.filter((x) => !listedClaims.has(x.slug))) {
    const title = c.topicTitle;
    let sec = claimChannel.sections.find((x) => x.title === title);
    if (!sec) { sec = { title, items: [], topic: c.topic }; claimChannel.sections.push(sec); }
    sec.items.push({ slug: c.slug, note: '' });
  }
  for (const sec of claimChannel.sections) sec.topic = sec.topic || (claimList.find((x) => x.slug === sec.items[0]?.slug) || {}).topic || '';
  const ORDER = Object.keys(TOPIC_TITLES);
  const rank = (sec) => (ORDER.includes(sec.topic) ? ORDER.indexOf(sec.topic) : ORDER.length);
  claimChannel.sections.sort((a, b) => rank(a) - rank(b));
}
// a syllabus names posts of its own channel, each once; the rest of the channel's posts follow it
for (const c of channels) {
  const listed = syllabusSlugs(c);
  for (const slug of listed) {
    const p = posts.find((q) => q.slug === slug);
    if (!p) errors.push(`channels/${c.file}: lists ${slug}, which is not a post in posts/`);
    else if (p.channel !== c.slug) errors.push(`channels/${c.file}: lists ${slug}, whose \`channel:\` is ${p.channel ? c.slug === p.channel ? c.slug : p.channel : 'not set'}; a syllabus lists the channel's own posts`);
  }
  for (const slug of listed) if (listed.indexOf(slug) !== listed.lastIndexOf(slug)) errors.push(`channels/${c.file}: lists ${slug} twice`);
  for (const p of posts) if (p.channel === c.slug) p.channelTitle = c.title;
  c.posts = posts.filter((p) => p.channel === c.slug).length;
  c.more = posts.filter((p) => p.channel === c.slug && !listed.includes(p.slug)).map((p) => p.slug);
  c.introHtml = c.intro.map(prose);
  c.summaryHtml = prose(c.summary);
  for (const s of c.sections) for (const i of s.items) i.noteHtml = prose(i.note);
}
if (errors.length) {
  console.error(`feed-build: ${errors.length} problem${errors.length > 1 ? 's' : ''}:\n  ${errors.join('\n  ')}`);
  process.exit(1);
}
mkdirSync(join(WWW, 'src/data'), { recursive: true });
writeFileSync(join(WWW, 'src/data/feed.json'), JSON.stringify({ author, posts, channels }, null, 1));
writeFileSync(join(WWW, 'src/data/lineage.json'), JSON.stringify(lineage));
const toSolve = posts.reduce((n, p) => n + p.failed.length, 0);
console.log(`feed-build: ${posts.length} posts, ${posts.reduce((n, p) => n + p.tests.length, 0)} tests, every one passing${toSolve ? `; ${toSolve} problem test${toSolve > 1 ? 's' : ''} left to solve` : ''}`);
