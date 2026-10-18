// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Lineage: every version of a post, and what each person did to it.
//
// ONE copy of the rules, used by two hosts, the way postCheck.js is:
//   - the site build (scripts/feed-build.mjs), for the posts committed in posts/
//   - the feed's server, for posts published from the site
// and by the page (components/History.jsx), which draws what either one produced.
// Plain JavaScript, no dependencies: the host supplies `run(src) -> output`,
// which expands imports and runs the program in viv.wasm.
//
// A VERSION is one exact source file. A post names the version it grew from
// with `from: <slug>` and says what kind of work it was with `edge: <kind>`.
// Each kind is a claim, and every claim but `fork` is checked by running
// programs, never by taking the author's word:
//
//   fork                started from that version          the parent exists
//   fix                 a test that failed now passes      a test of the child, run on the
//                                                          parent's program, fails there
//   extend              new tests, new capability          the parent's tests, word for word,
//                                                          pass here, and a test is added
//   rewrite             same behavior, new words           both programs print the same thing,
//                                                          line for line
//   adversarial-review  findings posted as failing tests   a problem whose added tests fail
//                                                          when run on the reviewed program
//   validate            reproduced against a reference     `reference:` names it; the parent's
//                                                          tests pass here and a test is added
//
// Two more edges are recorded by the server, not claimed in a header: a REVIEW
// (another person's verdict, attached to the exact version they read) and an EDIT (an
// author republishing their own post, checked only by the publish rules).
import { headerOf, fencedHeader, EDGE_NAMES } from './postCheck.js';

export const EDGE_KINDS = {
  fork: { label: 'fork', says: 'started from this version', checked: 'the original exists' },
  fix: { label: 'fix', says: 'a test that failed now passes', checked: 'a test of this version fails on the original and passes here' },
  extend: { label: 'extend', says: 'new tests for new capability', checked: 'every test of the original, word for word, still passes, and at least one test is added' },
  rewrite: { label: 'rewrite', says: 'same behavior, new words or structure', checked: 'both programs were run and printed the same thing, line for line' },
  'adversarial-review': { label: 'adversarial review', says: 'someone tried to break it and added tests that fail on it', checked: 'the added tests fail when run on the reviewed version' },
  validate: { label: 'validate', says: 'its results were compared with an outside reference and match it', checked: 'the reference is named, the original\'s tests still pass, and a test against the reference is added' },
};
/** What an author may claim in a header; review and edit are recorded, not claimed. */
export const CLAIMS = EDGE_NAMES;
export const RECORDED = {
  review: { label: 'review', says: 'someone other than the author ran this exact version and reported whether its output matches what the post claims' },
  edit: { label: 'edit', says: 'the author republished their own post; checked by the publish rules only' },
};
export const LANE_RANK = { ai: 0, draft: 1, reviewed: 2, published: 3 };

/** The version's address: the first 12 hex digits of a SHA-256 of the whole
 *  source file, header and notes included, after CRLF -> LF. The whole file,
 *  not the compiler's program address, because a review attests to what the
 *  reviewer read, and a changed claim in the notes is a different thing to have
 *  read even when the program is the same. The hash itself is the host's (node:crypto). */
export const normalizeSource = (src) => String(src || '').replace(/\r\n?/g, '\n');

// ---- reading a run ----------------------------------------------------------

/** The output a person would compare: everything the program printed, minus the
 *  interpreter's instrumentation lines (`__graph__`, `__trace__`). Those name the
 *  functions called and list them in hash-map order, so two programs that print
 *  the same thing can differ there. */
export const programOutput = (out) => String(out || '').split('\n').filter((l) => !/^__[a-z]+__ /.test(l));

/** { pass: Set, fail: Map(name -> detail), diagnostic } from a run's output. */
export function results(out) {
  const lines = String(out || '').split('\n');
  const pass = new Set(lines.filter((l) => l.startsWith('PASS ')).map((l) => l.slice(5)));
  const fail = new Map(lines.filter((l) => l.startsWith('FAIL ')).map((l) => { const [name, ...rest] = l.slice(5).split(' -- '); return [name, rest.join(' -- ')]; }));
  const diagnostic = lines.find((l) => /^error\b/.test(l)) || null;
  return { pass, fail, diagnostic };
}

/** The `test "..." { ... }` blocks of a program, with their line spans and raw text,
 *  and a normalized text (lines trimmed, blanks dropped) to compare two copies by
 *  what they say. The fenced header is skipped: prose there is not program. */
export function testBlocks(src) {
  const lines = normalizeSource(src).split('\n');
  const fence = fencedHeader(src);
  const out = [];
  for (let i = fence ? fence.end + 1 : 0; i < lines.length; i++) {
    const m = lines[i].match(/^\s*test\s+"((?:[^"\\]|\\.)*)"\s*\{/);
    if (!m) continue;
    let depth = 0;
    let j = i;
    for (; j < lines.length; j++) {
      const code = lines[j].replace(/"(?:[^"\\]|\\.)*"/g, '""').replace(/#.*$/, '');
      for (const ch of code) { if (ch === '{') depth++; else if (ch === '}') depth--; }
      if (depth <= 0) break;
    }
    const raw = lines.slice(i, j + 1);
    out.push({ name: m[1], start: i, end: j, raw: raw.join('\n'), text: raw.map((l) => l.trim()).filter(Boolean).join('\n') });
    i = j;
  }
  return out;
}

/** `src` with its own test blocks taken out and `blocks` put at the end: the
 *  program of one version, examined by the tests of another. */
export function withTests(src, blocks) {
  const lines = normalizeSource(src).split('\n');
  const drop = new Set();
  for (const b of testBlocks(src)) for (let k = b.start; k <= b.end; k++) drop.add(k);
  const kept = lines.filter((_, k) => !drop.has(k)).join('\n').replace(/\n+$/, '');
  return `${kept}\n\n${blocks.map((b) => b.raw).join('\n')}\n`;
}

async function safeRun(run, src) {
  try { return await run(src); } catch (e) {
    // a run that never happened (the server was busy or shutting down) says
    // nothing about the program: the caller must retry, not record an error
    if (e && e.transient) throw e;
    return `error: ${e.message}`;
  }
}

/** Run `blocks` against the program of `src`: { name: { result: pass|fail|error, detail } }.
 *  A diagnostic (a name the program lacks, a type error) stops the whole run, so
 *  when one appears each test is run alone to say which of them caused it. */
export async function outcomes(src, blocks, run) {
  const res = {};
  if (!blocks.length) return res;
  const r = results(await safeRun(run, withTests(src, blocks)));
  if (!r.diagnostic) {
    for (const b of blocks) {
      if (r.pass.has(b.name)) res[b.name] = { result: 'pass', detail: '' };
      else if (r.fail.has(b.name)) res[b.name] = { result: 'fail', detail: r.fail.get(b.name) };
      else res[b.name] = { result: 'error', detail: 'the test did not run' };
    }
    return res;
  }
  if (blocks.length === 1) return { [blocks[0].name]: { result: 'error', detail: r.diagnostic } };
  for (const b of blocks) Object.assign(res, await outcomes(src, [b], run));
  return res;
}

// ---- the checks ---------------------------------------------------------------

const q = (name) => `"${name}"`;

/** The parent's tests as the child carries them: missing, changed, or not passing. */
function keptTests(parentBlocks, childBlocks, childRes, why) {
  const reasons = [];
  for (const t of parentBlocks) {
    const hit = childBlocks.find((c) => c.name === t.name);
    if (!hit) reasons.push(`the parent's test ${q(t.name)} is missing: ${why}`);
    else if (hit.text !== t.text) reasons.push(`the parent's test ${q(t.name)} was changed: ${why}`);
    else if (!childRes.pass.has(t.name)) reasons.push(`the parent's test ${q(t.name)} does not pass on this version${childRes.fail.has(t.name) ? `: ${childRes.fail.get(t.name)}` : ''}`);
  }
  return reasons;
}

/**
 * Check the claim a child version makes about its parent.
 *   kind    one of CLAIMS
 *   parent  { src, out? }        out: its run's output, when the host has it
 *   child   { src, out? }
 *   run     async (src) => output
 * Returns { ok, reasons: [why the claim is refused], facts: {what the check found} }.
 */
export async function checkEdge(kind, parent, child, run) {
  if (!EDGE_KINDS[kind]) return { ok: false, reasons: [`\`edge: ${kind}\` is not a kind of edge; the kinds are ${CLAIMS.join(', ')}`], facts: {} };
  if (kind === 'fork') return { ok: true, reasons: [], facts: {} };
  const ch = headerOf(child.src);
  if (headerOf(parent.src).kind === 'ladder' || ch.kind === 'ladder') return { ok: false, reasons: ["a ladder can only claim `edge: fork`: the other edges compare test blocks, and a ladder's tests are inside its rungs"], facts: {} };
  const parentOut = parent.out ?? await safeRun(run, parent.src);
  const childOut = child.out ?? await safeRun(run, child.src);
  const pr = results(parentOut);
  const cr = results(childOut);
  const pb = testBlocks(parent.src);
  const cb = testBlocks(child.src);
  const added = cb.filter((c) => !pb.some((p) => p.name === c.name));
  const reasons = [];
  const facts = {};

  if (kind === 'extend' || kind === 'validate') {
    reasons.push(...keptTests(pb, cb, cr, kind === 'extend' ? 'an extension keeps every test of its parent, word for word' : 'a validation keeps every test of what it validates, word for word'));
    if (!added.length) reasons.push(kind === 'extend' ? 'no test is added: an extension adds a test for what it adds' : 'no test is added: a validation adds the test that compares with the reference');
    for (const a of added) if (!cr.pass.has(a.name)) reasons.push(`the added test ${q(a.name)} does not pass`);
    if (kind === 'validate') {
      if (!ch.reference) reasons.push('a validation names its outside reference: `reference: <where the numbers come from>`');
      facts.reference = ch.reference;
    }
    facts.kept = pb.length;
    facts.added = added.map((a) => a.name);
  }

  if (kind === 'rewrite') {
    const po = programOutput(parentOut);
    const co = programOutput(childOut);
    if (normalizeSource(parent.src) === normalizeSource(child.src)) reasons.push('the source is the parent\'s, unchanged: there is nothing rewritten');
    if (pr.diagnostic) reasons.push(`the parent does not run (${pr.diagnostic}), so there is no behavior to keep`);
    const n = Math.max(po.length, co.length);
    for (let i = 0; i < n; i++) {
      if (po[i] === co[i]) continue;
      if (po[i] === undefined) reasons.push(`the output differs at line ${i + 1}: this version prints ${q(co[i])} and the parent had stopped`);
      else if (co[i] === undefined) reasons.push(`the output differs at line ${i + 1}: the parent prints ${q(po[i])} and this version had stopped`);
      else reasons.push(`the output differs at line ${i + 1}: the parent prints ${q(po[i])}, this version ${q(co[i])}`);
      break; // the first difference is the reason; the rest follow from it
    }
    facts.lines = co.length;
    facts.tests = [...pr.pass].length;
  }

  if (kind === 'fix') {
    const onParent = await outcomes(parent.src, cb, run);
    const fixed = cb.filter((c) => onParent[c.name]?.result === 'fail' && cr.pass.has(c.name)).map((c) => c.name);
    const errored = cb.filter((c) => onParent[c.name]?.result === 'error').map((c) => c.name);
    if (!fixed.length) {
      reasons.push('no test of this version fails on the parent\'s program: a fix makes a failing test pass');
      for (const name of errored) reasons.push(`the test ${q(name)} could not run on the parent (${onParent[name].detail}); a fix needs a test that runs there and fails`);
    }
    facts.fixed = fixed;
    facts.onParent = Object.fromEntries(Object.entries(onParent).map(([k, v]) => [k, v.result]));
  }

  if (kind === 'adversarial-review') {
    if (ch.kind !== 'problem') reasons.push('an adversarial review is posted as a problem (`kind: problem`): its tests fail by design, and fixing them is solving it');
    if (!added.length) reasons.push('no test is added: an adversarial review must add tests that fail on the reviewed version');
    const onParent = await outcomes(parent.src, added, run);
    const findings = added.filter((a) => onParent[a.name].result === 'fail').map((a) => a.name);
    if (added.length && !findings.length) {
      reasons.push('none of the added tests fails on the reviewed version, and an adversarial review needs at least one that does');
      for (const a of added) if (onParent[a.name].result === 'error') reasons.push(`the test ${q(a.name)} could not run on the reviewed version (${onParent[a.name].detail})`);
    }
    facts.findings = findings;
    facts.onParent = Object.fromEntries(Object.entries(onParent).map(([k, v]) => [k, v.result]));
  }

  return { ok: reasons.length === 0, reasons, facts };
}

/** The tests an adversarial review added: what it challenges every version with. */
export function challengeBlocks(parentSrc, childSrc) {
  const pb = testBlocks(parentSrc);
  return testBlocks(childSrc).filter((c) => !pb.some((p) => p.name === c.name));
}

/** The defs in `viv_hash` output: [{ name, body }]. `body` is the hash of what the
 *  def computes (names and docstrings of the def itself do not move it). */
export const defsOf = (hashText) => [...String(hashText || '').matchAll(/^[0-9a-f]{8}\s+body ([0-9a-f]{8})\s+def ([A-Za-z_]\w*)/gm)].map((m) => ({ name: m[2], body: m[1] }));

// ---- the view ---------------------------------------------------------------

/**
 * What the history view draws, from what either host recorded:
 *   raw = {
 *     versions: [{ hash, slug, title, kind, lane, agent, author, at, current, tests: [names],
 *                  failed: [names], defs: [{name, body}], reviews: [{verdict, reviewer, at, inferred}] }],
 *     edges: [{ from, to, kind, facts, at }],                 from/to are version hashes
 *     challenges: [{ edge: {from, to}, test, results: { [hash]: 'pass'|'fail'|'error' } }],
 *   }
 * Returns the versions in tree order with, for each: its parent edge, depth,
 * open and closed challenges, validations, and ownership; and `best`.
 */
export function lineageView(raw, focus = null) {
  const versions = raw.versions.map((v) => ({ ...v }));
  const byHash = new Map(versions.map((v) => [v.hash, v]));
  const edges = raw.edges.filter((e) => byHash.has(e.from) && byHash.has(e.to));
  for (const v of versions) {
    v.parentEdge = edges.find((e) => e.to === v.hash) || null;
    v.childEdges = edges.filter((e) => e.from === v.hash);
    v.validations = edges.filter((e) => e.kind === 'validate' && e.from === v.hash).map((e) => ({ by: e.to, reference: e.facts?.reference || '' }));
    v.open = [];
    v.closed = [];
    v.inapplicable = [];
  }
  // challenges: each adversarial test, against each version it was run on
  for (const c of raw.challenges || []) {
    for (const [hash, result] of Object.entries(c.results || {})) {
      const v = byHash.get(hash);
      if (!v) continue;
      const item = { test: c.test, review: c.edge.to };
      if (result === 'fail') v.open.push(item);
      else if (result === 'pass') v.closed.push(item);
      else v.inapplicable.push(item);
    }
  }
  // ownership, by the proxy until intent lenses land: a def belongs to whoever
  // wrote the FIRST version in its ancestry that has it with this exact body hash;
  // a version with an `agent:` line is the model's, one without is a person's
  const ancestors = (v) => { const out = []; for (let e = v.parentEdge; e; e = byHash.get(e.from).parentEdge) { out.push(byHash.get(e.from)); if (out.length > 64) break; } return out; };
  for (const v of versions) {
    const chain = [v, ...ancestors(v)];
    let person = 0;
    let model = 0;
    for (const d of v.defs || []) {
      let origin = v;
      for (const a of chain) if ((a.defs || []).some((x) => x.body === d.body)) origin = a;
      if (origin.agent) model++; else person++;
    }
    v.ownership = { person, model, total: person + model };
  }
  // the current best: never a problem (its tests fail by design); then the fewest
  // open adversarial failures, the highest lane, the most validations, the most
  // reviews that reproduce, the most tests passing, and the newest
  const reproducing = (v) => (v.reviews || []).filter((r) => r.verdict === 'reproduces' || r.verdict === 'notes').length;
  const ranked = versions.filter((v) => v.kind !== 'problem').map((v, i) => ({ v, i })).sort((a, b) => (
    a.v.open.length - b.v.open.length
    || (LANE_RANK[b.v.lane] ?? 0) - (LANE_RANK[a.v.lane] ?? 0)
    || b.v.validations.length - a.v.validations.length
    || reproducing(b.v) - reproducing(a.v)
    || (b.v.tests || []).length - (a.v.tests || []).length
    || String(b.v.at || '').localeCompare(String(a.v.at || ''))
    || b.i - a.i));
  const best = ranked.length ? ranked[0].v.hash : null;
  // tree order from the roots; the main line is the path from the root to the
  // best version (and to the focused one), and every other branch is collapsed
  const onMain = new Set();
  for (const h of [best, focus]) {
    let v = h && byHash.get(h);
    while (v) { onMain.add(v.hash); v = v.parentEdge ? byHash.get(v.parentEdge.from) : null; }
  }
  const order = [];
  const walk = (v, depth, branch) => {
    v.depth = depth;
    v.branch = branch; // the hash of the first version off the main line, or null
    order.push(v);
    const kids = v.childEdges.map((e) => byHash.get(e.to)).sort((a, b) => Number(onMain.has(b.hash)) - Number(onMain.has(a.hash)) || String(a.at || '').localeCompare(String(b.at || '')));
    for (const k of kids) walk(k, depth + 1, branch || (onMain.has(k.hash) ? null : k.hash));
  };
  for (const root of versions.filter((v) => !v.parentEdge)) walk(root, 0, onMain.has(root.hash) ? null : root.hash);
  const kinds = [...new Set(edges.map((e) => e.kind))];
  return { versions: order, edges, best, kinds };
}
