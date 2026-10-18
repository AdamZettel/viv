// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// What the tools say back, shared by the feed's server (which reads its
// database) and the Netlify Function (fntools.mjs, which reads what the
// static build wrote). Where the facts come from differs; the text a model
// reads and the structuredContent beside it are built here, once.
import { summarize, summaryText } from './summary.mjs';
import { KINDS } from './tooldefs.mjs';

export const SOURCE_LIMIT = 200 * 1024;

export const fail = (text) => ({ text, data: { ok: false, error: text }, isError: true });

/** Why a `source` argument cannot be run, or null. */
export function checkSource(source) {
  if (typeof source !== 'string' || !source.trim()) return 'source must be the program text, a non-empty string';
  if (Buffer.byteLength(source) > SOURCE_LIMIT) return `the program is over ${SOURCE_LIMIT / 1024} KB`;
  return null;
}

/** run_viv's result from a run ({ out, seconds }). */
export function runResult(run) {
  const s = summarize(run.out);
  const text = summaryText(s, { seconds: run.seconds });
  const passed = s.tests.filter((t) => t.passed).length;
  return {
    text,
    data: { ok: !s.error && s.tests.length > 0 && passed === s.tests.length, error: s.error, tests: s.tests, printed: s.shown, printedTruncated: s.shownTruncated, figures: s.figures, animation: s.animation, seconds: run.seconds, engine: 'interpreter' },
    isError: !!s.error,
  };
}

/** check_post's result from a verdict `v` (app.mjs verify), the header `h` and the lineage claim, if any. */
export function checkPostResult(v, h, claim) {
  const ok = v.errors.length === 0 && (!claim || claim.ok);
  const lines = [ok ? 'The site would accept this post.' : 'The site would refuse this post:'];
  for (const e of v.errors) lines.push(`- ${e}`);
  if (claim && !claim.ok) for (const r of claim.reasons) lines.push(`- edge: ${claim.kind} on ${claim.parent}: ${r}`);
  if (v.tests && v.tests.length) lines.push(`Tests passed: ${v.tests.join('; ')}`);
  if (claim && claim.ok) lines.push(`The claim \`edge: ${claim.kind}\` on ${claim.parent} holds.`);
  const lane = h.lane || 'draft';
  lines.push(`It would start in the ${lane === 'ai' ? 'AI' : lane} lane${h.agent ? ` (it names agent: ${h.agent})` : ''}.`);
  if (v.address) lines.push(`Program address: ${v.address}`);
  return {
    text: lines.join('\n'),
    data: { ok, errors: v.errors, tests: v.tests || [], failed: v.failed || [], shown: v.shown || [], seconds: v.seconds, address: v.address || null, lane, claim, header: { title: h.title, date: h.date, kind: h.kind, tags: h.tags, summary: h.summary, agent: h.agent || null } },
    isError: false,
  };
}

/** search's result: the arguments checked, then `find(query, { kind, limit })` asked. */
export function searchResult({ query, kind = null, limit = 10 }, find) {
  if (typeof query !== 'string' || !query.trim()) return fail('query must be a non-empty string');
  if (kind && !KINDS.includes(kind)) return fail(`kind must be one of ${KINDS.join(', ')}`);
  const hits = find(query, { kind, limit });
  if (!hits.length) return { text: `Nothing matched "${query}".`, data: { hits: [] }, isError: false };
  const text = hits.map((h, i) => {
    const where = h.kind === 'post' ? `post ${h.slug} (${h.postKind}, ${h.lane} lane, by ${h.author})` : h.use ? `${h.kind} ${h.key} (load with \`import ${h.use.replace(/^lib\//, '').replace(/\//g, '.')}\`)` : `${h.kind} ${h.key}`;
    return `${i + 1}. ${h.title} -- ${where}${h.summary ? `\n   ${h.summary}` : ''}${h.url ? `\n   ${h.url}` : ''}`;
  }).join('\n');
  return { text, data: { hits }, isError: false };
}

export const SLUG_ARG = /^[a-z0-9-]{1,100}$/;

/** get_post's result from the post's fields. `source` is left out of a solution; `solutions` is a problem's count. */
export function postResult(post, { source }) {
  const hidden = post.kind === 'solution';
  if (!hidden && source != null) post.source = source;
  const text = [
    `${post.title} (${post.kind}, ${post.lane} lane, by ${post.author}${post.agent ? `, agent ${post.agent}` : ''})`,
    post.summary,
    post.from ? `Derived from ${post.from} (edge: ${post.edge}).` : '',
    `Tests that passed on the server: ${post.tests.join('; ') || 'none'}${post.failed.length ? `; failing (a problem's open tests): ${post.failed.join('; ')}` : ''}`,
    post.lineage.versions.length > 1 ? `Family: ${post.lineage.versions.length} versions, ${post.lineage.edges.length} edges.` : '',
    post.kind === 'problem' ? `Solutions posted: ${post.solutions} (not shown here).` : '',
    hidden ? 'This is a solution; its source is not returned here.' : source != null ? `Source:\n${source}` : '',
  ].filter(Boolean).join('\n\n');
  return { text, data: post, isError: false };
}
