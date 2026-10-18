// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Full-text search over what a model can use: published posts, the example
// programs, the lib/ modules, the builtins and the documented examples.
//
// The index is an FTS5 table in its own in-memory SQLite database, so the feed's
// database gets no new tables and no migration. Everything but the posts is
// read from the repository once; the posts are re-read whenever the published
// set changes (checked on each search by a count and the latest update time,
// which is one indexed query).
//
// Solution posts are left out: a problem's answers are hidden on the site from
// anyone who has not solved it, and a model context is not the place to hand
// them over.
import { DatabaseSync } from 'node:sqlite';
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { headerOf } from '../../www/src/lib/postCheck.js';
import { queryWords, WEIGHTS } from './jsearch.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const SITE = 'https://viv-lang.org';
export { KINDS } from './tooldefs.mjs';

/** The repository's documents: example programs, lib modules, builtins, doc examples. */
export function repoDocs() {
  const docs = [];
  const data = (f) => JSON.parse(readFileSync(join(ROOT, 'www', 'src', 'data', f), 'utf8'));
  for (const p of data('programs.json').programs) {
    const name = p.file.replace(/\.viv$/, '');
    docs.push({ kind: 'program', key: `examples/${name}`, title: p.title, summary: p.blurb || '', body: '', tags: p.group || '', url: `${SITE}/programs/${p.file}`, use: `examples/${name}` });
  }
  const libDir = join(ROOT, 'lib');
  const walk = (dir, prefix) => {
    for (const f of readdirSync(dir, { withFileTypes: true })) {
      if (f.isDirectory()) walk(join(dir, f.name), `${prefix}${f.name}/`);
      else if (f.name.endsWith('.viv')) {
        const src = readFileSync(join(dir, f.name), 'utf8');
        const h = headerOf(src);
        const name = `lib/${prefix}${f.name.replace(/\.viv$/, '')}`;
        // a module without a fenced header still has its leading comment lines
        const lead = h.title ? '' : src.split('\n').filter((l) => /^#( |$)/.test(l)).slice(0, 8).map((l) => l.slice(2)).join(' ');
        docs.push({ kind: 'module', key: name, title: h.title || name, summary: h.summary || lead, body: h.notes.join('\n'), tags: h.tags.join(' '), url: null, path: `${name}.viv`, use: name });
      }
    }
  };
  if (existsSync(libDir)) walk(libDir, '');
  for (const g of data('builtins.json').groups) {
    for (const b of g.items) {
      docs.push({ kind: 'builtin', key: b.sig, title: b.sig, summary: b.blurb || '', body: g.note || '', tags: g.title, url: `${SITE}/reference#builtins` });
    }
  }
  const ex = data('vivDocExamples.json');
  const topicTitle = Object.fromEntries(ex.topics.map((t) => [t.slug, t.title]));
  for (const e of ex.examples) {
    docs.push({ kind: 'example', key: `${e.topic}/${e.title}`, title: e.title, summary: e.note || '', body: e.code, tags: topicTitle[e.topic] || e.topic, url: `${SITE}/docs/${e.topic}` });
  }
  return docs;
}

/**
 * FTS5 query text from what a person or a model typed: words, each a prefix,
 * all required. No stemmer: Porter's turns "Runge" into "rung", which is a
 * different thing on this site. A plural's final s is dropped instead, so the
 * prefix "eigenvalue" finds both forms.
 */
export function ftsQuery(text, op = 'AND') {
  return queryWords(text).map((w) => `"${w}"*`).join(` ${op} `);
}

export function createSearch(db) {
  const idx = new DatabaseSync(':memory:');
  idx.exec(`CREATE VIRTUAL TABLE docs USING fts5(title, summary, body, tags, kind UNINDEXED, key UNINDEXED, meta UNINDEXED, tokenize = 'unicode61')`);
  const insert = idx.prepare('INSERT INTO docs (title, summary, body, tags, kind, key, meta) VALUES (?, ?, ?, ?, ?, ?, ?)');
  const add = (d) => insert.run(d.title, d.summary, d.body, d.tags, d.kind, d.key, JSON.stringify({ url: d.url, path: d.path || null, use: d.use || null, lane: d.lane || null, author: d.author || null, postKind: d.postKind || null }));
  for (const d of repoDocs()) add(d);

  let postsSeen = null;
  function refreshPosts() {
    if (!db) return;
    const sig = db.prepare("SELECT COUNT(*) AS n, MAX(updated_at) AS t FROM posts WHERE status = 'published'").get();
    const key = `${sig.n}|${sig.t}`;
    if (key === postsSeen) return;
    idx.exec("DELETE FROM docs WHERE kind = 'post'");
    const rows = db.prepare("SELECT p.slug, p.title, p.summary, p.notes, p.tags, p.kind, p.lane, u.handle FROM posts p JOIN users u ON u.id = p.author_id WHERE p.status = 'published' AND p.kind != 'solution'").all();
    for (const r of rows) {
      add({ kind: 'post', key: r.slug, title: r.title, summary: r.summary, body: JSON.parse(r.notes || '[]').join('\n'), tags: JSON.parse(r.tags || '[]').join(' '), url: `${SITE}/p/${r.slug}`, lane: r.lane, author: r.handle, postKind: r.kind });
    }
    postsSeen = key;
  }

  /** Hits, best first: { kind, key, title, summary, snippet, url, use?, lane?, author? }. */
  function search(query, { kind = null, limit = 10 } = {}) {
    refreshPosts();
    const n = Math.min(25, Math.max(1, Number(limit) || 10));
    const run = (q) => {
      if (!q) return [];
      let sql = `SELECT kind, key, title, summary, meta, snippet(docs, 2, '[', ']', '…', 12) AS snip, bm25(docs, ${WEIGHTS.join(', ')}) AS score FROM docs WHERE docs MATCH ?`;
      const args = [q];
      if (kind) { sql += ' AND kind = ?'; args.push(kind); }
      sql += ' ORDER BY score LIMIT ?';
      args.push(n);
      return idx.prepare(sql).all(...args);
    };
    // every word first; if nothing has all of them, any of them
    let rows = run(ftsQuery(query, 'AND'));
    if (!rows.length) rows = run(ftsQuery(query, 'OR'));
    return rows.map((r) => {
      const meta = JSON.parse(r.meta);
      const hit = { kind: r.kind, key: r.key, title: r.title, summary: r.summary.slice(0, 300) };
      if (meta.url) hit.url = meta.url;
      if (meta.path) hit.path = meta.path;
      if (r.snip && r.snip.includes('[')) hit.snippet = r.snip;
      if (meta.use) hit.use = meta.use;
      if (r.kind === 'post') Object.assign(hit, { slug: r.key, lane: meta.lane, author: meta.author, postKind: meta.postKind });
      return hit;
    });
  }

  const counts = () => { refreshPosts(); return Object.fromEntries(idx.prepare('SELECT kind, COUNT(*) AS n FROM docs GROUP BY kind').all().map((r) => [r.kind, r.n])); };
  return { search, counts };
}
