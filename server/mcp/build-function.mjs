// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Build the MCP endpoint's Netlify Function, after the site build:
//
//     node ../server/mcp/build-function.mjs            (from www/, after npm run build)
//     node server/mcp/build-function.mjs --out <dir>   (anywhere else)
//
// Writes www/netlify/functions/mcp/ (gitignored), a directory that needs
// nothing outside itself:
//   mcp.mjs          the function's entry, a default export that calls impl.mjs
//   impl.mjs         netlify.mjs and everything it imports, bundled by esbuild
//   run-worker.mjs   the worker one run gets (fn-run-worker.mjs: server/run-worker.mjs
//                    with a limit on output), bundled the same way
//   viv.wasm         www/public/viv/viv.wasm
//   reference/       the prompt pack the site build wrote (www/public)
//   modules/         lib/, dft/, pde/, examples/ and posts/ .viv files, for imports
//   data/posts.json  the published posts from feed.json, with their lineage
//   data/corpus.json the search documents (search.mjs's repoDocs and the posts)
//
// It refuses to finish when the bundle pulls in node:sqlite (the function must
// not need it) or when the site build's outputs are missing.
import { readFileSync, writeFileSync, mkdirSync, rmSync, copyFileSync, readdirSync, existsSync, statSync } from 'node:fs';
import { join, dirname, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { headerOf } from '../../www/src/lib/postCheck.js';
import { repoDocs } from './search.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..', '..');
const WWW = join(ROOT, 'www');
const SITE = 'https://viv-lang.org';
const outArg = process.argv.indexOf('--out');
const OUT = outArg > 0 ? process.argv[outArg + 1] : join(WWW, 'netlify', 'functions', 'mcp');

const need = (p) => {
  if (!existsSync(p)) {
    console.error(`build-function: ${relative(ROOT, p)} is missing; build the site first (npm run build in www/)`);
    process.exit(1);
  }
  return p;
};

const esbuild = createRequire(join(WWW, 'package.json'))('esbuild');

rmSync(OUT, { recursive: true, force: true });
mkdirSync(join(OUT, 'data'), { recursive: true });

// ---- the code -------------------------------------------------------------------
const common = { bundle: true, platform: 'node', format: 'esm', target: 'node22', metafile: true, logLevel: 'warning', legalComments: 'none' };
const fn = await esbuild.build({ ...common, entryPoints: [join(HERE, 'netlify.mjs')], outfile: join(OUT, 'impl.mjs') });
// the entry Netlify reads: a literal default export, the form its v2 detection looks for
// Netlify reads a function's rate limit from its entry file's \`config\` (it
// cannot be set in netlify.toml): MCP_RATE_LIMIT requests per
// MCP_RATE_WINDOW_S seconds per client address, then 429. A tool call is a
// few requests (initialize, tools/list, the call), so this leaves room for
// interactive use and bounds a script that calls run_viv in a loop.
const MCP_RATE_LIMIT = 20;
const MCP_RATE_WINDOW_S = 60;
writeFileSync(join(OUT, 'mcp.mjs'), `// built by server/mcp/build-function.mjs from server/mcp/netlify.mjs; do not edit\nimport handler from './impl.mjs';\n\nexport default async (request, context) => handler(request, context);\n\nexport const config = {\n  rateLimit: { windowLimit: ${MCP_RATE_LIMIT}, windowSize: ${MCP_RATE_WINDOW_S}, aggregateBy: ['ip', 'domain'] },\n};\n`);
await esbuild.build({ ...common, entryPoints: [join(HERE, 'fn-run-worker.mjs')], outfile: join(OUT, 'run-worker.mjs') });
const inputs = Object.keys(fn.metafile.inputs);
const imports = Object.values(fn.metafile.inputs).flatMap((i) => i.imports.map((x) => x.path));
if (imports.includes('node:sqlite') || inputs.some((p) => /server\/(app|db|tools|search)\.mjs$|server\/mcp\/(tools|search)\.mjs$/.test(p))) {
  console.error('build-function: the function bundle reaches node:sqlite or the database code; fntools.mjs must not import it');
  process.exit(1);
}

// ---- what the tools read ---------------------------------------------------------
copyFileSync(need(join(WWW, 'public', 'viv', 'viv.wasm')), join(OUT, 'viv.wasm'));
mkdirSync(join(OUT, 'reference'), { recursive: true });
for (const f of ['viv-prompt-pack.md', 'llms-full.txt', 'llms.txt']) copyFileSync(need(join(WWW, 'public', f)), join(OUT, 'reference', f));

let moduleCount = 0;
const copyViv = (from, to) => {
  if (!existsSync(from)) return;
  for (const f of readdirSync(from, { withFileTypes: true })) {
    if (f.isDirectory()) copyViv(join(from, f.name), join(to, f.name));
    else if (f.name.endsWith('.viv')) {
      mkdirSync(to, { recursive: true });
      copyFileSync(join(from, f.name), join(to, f.name));
      moduleCount++;
    }
  }
};
for (const [name, dir] of Object.entries({ lib: 'lib', dft: 'dft', pde: 'pde', examples: 'lang/viv/examples', posts: 'posts' })) copyViv(join(ROOT, dir), join(OUT, 'modules', name));

// the posts, as get_post returns them, with the lineage summary the server's get_post gives
const feed = JSON.parse(readFileSync(need(join(WWW, 'src', 'data', 'feed.json')), 'utf8'));
const lineage = JSON.parse(readFileSync(need(join(WWW, 'src', 'data', 'lineage.json')), 'utf8'));
const handle = feed.author.handle;
const posts = {};
for (const p of feed.posts) {
  // the whole file, header included, as the server stores a post (feed.json's `source` is the program after the header)
  const file = p.kind === 'claim' ? join(ROOT, p.file) : join(ROOT, 'posts', p.file);
  const source = p.file && existsSync(file) ? readFileSync(file, 'utf8') : null;
  if (p.kind !== 'claim' && source == null) {
    console.error(`build-function: feed.json lists ${p.slug}, but posts/${p.file} is not there`);
    process.exit(1);
  }
  const h = source ? headerOf(source) : { notes: [], edge: null, summary: '' };
  const fam = lineage[p.family];
  let lin = { versions: [], edges: [] };
  if (fam) {
    const versions = fam.versions.filter((v) => v.kind !== 'solution' || v.slug === p.slug);
    const keep = new Set(versions.map((v) => v.hash));
    lin = {
      current: p.version || null,
      versions: versions.map((v) => ({ hash: v.hash, slug: v.slug, title: v.title, kind: v.kind, author: (v.author && v.author.handle) || handle, at: v.at, current: !!v.current })),
      edges: fam.edges.filter((e) => keep.has(e.from) && keep.has(e.to)).map((e) => ({ from: e.from, to: e.to, kind: e.kind, checked: true })),
    };
  }
  const post = {
    slug: p.slug, title: p.titleText || p.title, date: p.date, kind: p.kind, tags: p.tags || [], summary: h.summary || p.summaryText || '',
    notes: h.notes || [], agent: p.agent || null, lane: p.lane, author: handle,
    from: p.forkedFrom || null, edge: p.forkedFrom ? p.edge || h.edge || 'fork' : null, solves: p.solves || null, channel: p.channel || null,
    tests: p.tests || [], failed: p.failed || [], shown: p.shown || [], seconds: p.seconds, address: p.address,
    url: `${SITE}/p/${p.slug}`,
  };
  if (p.kind === 'problem') post.solutions = feed.posts.filter((q) => q.solves === p.slug).length;
  if (p.kind === 'claim' && p.claim) post.claim = { statement: p.claim.statement, status: p.claim.status, topic: p.claim.topic, file: p.claim.file };
  post.lineage = lin;
  post.source = source;
  posts[p.slug] = post;
}
writeFileSync(join(OUT, 'data', 'posts.json'), JSON.stringify({ posts, channels: feed.channels.map((c) => c.slug) }));

// the search documents: the repository's, then the posts, solutions left out (search.mjs says why)
const docs = repoDocs();
for (const p of Object.values(posts)) {
  if (p.kind === 'solution') continue;
  docs.push({ kind: 'post', key: p.slug, title: p.title, summary: p.summary, body: p.notes.join('\n'), tags: p.tags.join(' '), url: p.url, lane: p.lane, author: p.author, postKind: p.kind });
}
writeFileSync(join(OUT, 'data', 'corpus.json'), JSON.stringify({ docs }));

const size = (dir) => readdirSync(dir, { withFileTypes: true }).reduce((n, f) => n + (f.isDirectory() ? size(join(dir, f.name)) : statSync(join(dir, f.name)).size), 0);
console.log(`build-function: ${relative(process.cwd(), OUT) || '.'}: ${(size(OUT) / 1048576).toFixed(1)} MB, ${Object.keys(posts).length} posts, ${docs.length} search documents, ${moduleCount} modules`);
