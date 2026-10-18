// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The search tool without SQLite, for the Netlify Function.
//
// The feed's server searches an FTS5 table (search.mjs). A function cannot
// count on node:sqlite, so the site build writes the same documents to a JSON
// file (build-function.mjs) and this file searches them in JavaScript with the
// same contract: the same words (queryWords), each a prefix; every word
// required, and when nothing has all of them, any word; the same column
// weights; hits best first, with a snippet from the body.
//
// What differs from FTS5, measured on the same corpus by fn-test.mjs:
//   - scoring is BM25 written out here (k1 1.2, b 0.75, FTS5's formula with the
//     weights applied to each column's hit count), so ties and near-ties can
//     come out in another order;
//   - the snippet is a 12-token window around the first hit in the body, close
//     to FTS5's snippet() but not character for character the same;
//   - a query word with an underscore is split at it and each part must match,
//     where FTS5 matches the parts as a phrase (next to each other).
// Posts are the ones the static build published (feed.json); the server reads
// them from its database.

/** The weights given each column: a hit in a title counts most. */
export const WEIGHTS = [10.0, 5.0, 1.0, 3.0];
const COLUMNS = ['title', 'summary', 'body', 'tags'];
const K1 = 1.2;
const B = 0.75;
const SNIPPET_TOKENS = 12;

/** The words of a query as the index matches them: lowercased, at most 12, a
 *  plural's final s dropped. The FTS5 index (search.mjs) uses these too. */
export function queryWords(text) {
  const words = String(text || '').toLowerCase().match(/[\p{L}\p{N}_]+/gu) || [];
  return words.slice(0, 12).map((w) => (w.length > 3 && w.endsWith('s') && !w.endsWith('ss') ? w.slice(0, -1) : w));
}

// FTS5's unicode61 tokenizer: runs of letters and digits, case and diacritics folded
const fold = (s) => s.normalize('NFD').replace(/\p{M}+/gu, '').toLowerCase();
const TOKEN = /[\p{L}\p{N}]+/gu;
function tokens(text) {
  const out = [];
  for (const m of String(text || '').matchAll(TOKEN)) out.push({ t: fold(m[0]), at: m.index, end: m.index + m[0].length });
  return out;
}

/**
 * An index over `docs` ({ kind, key, title, summary, body, tags, url, path,
 * use, lane, author, postKind }). Returns { search(query, { kind, limit }),
 * counts() }, the same as createSearch in search.mjs.
 */
export function createJsonSearch(docs) {
  const rows = docs.map((d) => {
    const cols = COLUMNS.map((c) => tokens(d[c]));
    return { d, cols, length: cols.reduce((n, c) => n + c.length, 0) };
  });
  const avgLength = rows.reduce((n, r) => n + r.length, 0) / Math.max(1, rows.length);

  // how often a query term (a prefix) occurs in each column of a row
  const hitsOf = (row, term) => row.cols.map((col) => col.reduce((n, x) => n + (x.t.startsWith(term) ? 1 : 0), 0));

  function snippet(row, terms) {
    const body = row.cols[2];
    const text = row.d.body || '';
    const first = body.findIndex((x) => terms.some((t) => x.t.startsWith(t)));
    if (first < 0) return null;
    const start = Math.max(0, Math.min(first - 2, body.length - SNIPPET_TOKENS));
    const end = Math.min(body.length, start + SNIPPET_TOKENS);
    let out = start > 0 ? '…' : '';
    for (let i = start; i < end; i++) {
      const x = body[i];
      if (i > start) out += text.slice(body[i - 1].end, x.at);
      const word = text.slice(x.at, x.end);
      out += terms.some((t) => x.t.startsWith(t)) ? `[${word}]` : word;
    }
    if (end < body.length) out += '…';
    return out;
  }

  function run(words, op, kind, n) {
    // a word is one or more terms (an underscore splits it); a row has the word when it has every term
    const groups = words.map((w) => w.split('_').filter(Boolean).map(fold)).filter((g) => g.length);
    if (!groups.length) return [];
    const scored = [];
    const pool = kind ? rows.filter((r) => r.d.kind === kind) : rows;
    const perTerm = new Map();
    const docFreq = (term) => {
      if (!perTerm.has(term)) perTerm.set(term, rows.filter((r) => r.cols.some((c) => c.some((x) => x.t.startsWith(term)))).length);
      return perTerm.get(term);
    };
    for (const row of pool) {
      let score = 0;
      let matched = 0;
      for (const g of groups) {
        const hits = g.map((t) => hitsOf(row, t));
        if (hits.some((h) => h.every((x) => x === 0))) continue;
        matched++;
        g.forEach((term, i) => {
          const f = hits[i].reduce((s, x, c) => s + WEIGHTS[c] * x, 0);
          const nt = docFreq(term);
          const idf = Math.max(1e-6, Math.log((rows.length - nt + 0.5) / (nt + 0.5)));
          score += idf * (f * (K1 + 1)) / (f + K1 * (1 - B + B * (row.length / avgLength)));
        });
      }
      if (op === 'AND' ? matched === groups.length : matched > 0) scored.push({ row, score });
    }
    scored.sort((a, b) => b.score - a.score);
    const terms = groups.flat();
    return scored.slice(0, n).map(({ row }) => ({ row, snip: snippet(row, terms) }));
  }

  /** Hits, best first: { kind, key, title, summary, snippet, url, use?, lane?, author? }. */
  function search(query, { kind = null, limit = 10 } = {}) {
    const n = Math.min(25, Math.max(1, Number(limit) || 10));
    const words = queryWords(query);
    let found = run(words, 'AND', kind, n);
    if (!found.length) found = run(words, 'OR', kind, n);
    return found.map(({ row: { d }, snip }) => {
      const hit = { kind: d.kind, key: d.key, title: d.title, summary: String(d.summary || '').slice(0, 300) };
      if (d.url) hit.url = d.url;
      if (d.path) hit.path = d.path;
      if (snip && snip.includes('[')) hit.snippet = snip;
      if (d.use) hit.use = d.use;
      if (d.kind === 'post') Object.assign(hit, { slug: d.key, lane: d.lane, author: d.author, postKind: d.postKind });
      return hit;
    });
  }

  const counts = () => rows.reduce((o, r) => ({ ...o, [r.d.kind]: (o[r.d.kind] || 0) + 1 }), {});
  return { search, counts };
}
