// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Internal-link check for the site's sources (no build needed).
//
//     node study/release/linkcheck.mjs
//
// Reads every href="/..." and href={`/...`} literal in www/src and checks that
// a page in www/src/pages, a file in www/public or a redirect in netlify.toml
// serves it. Dynamic segments (`${...}`) match any page whose static prefix
// exists. Exits 1 when a link has nowhere to go.
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, dirname, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const SRC = join(ROOT, 'www', 'src');
const PAGES = join(SRC, 'pages');
const PUBLIC = join(ROOT, 'www', 'public');

const walk = (d) => readdirSync(d).flatMap((n) => {
  const p = join(d, n);
  return statSync(p).isDirectory() ? walk(p) : [p];
});

// Route patterns from the pages directory: [x] and [...x] match a segment or more.
const routes = walk(PAGES).map((f) => {
  let r = '/' + relative(PAGES, f).replace(/\.(astro|js|ts|md|mdx)$/, '').replace(/(^|\/)index$/, '');
  r = r.replace(/\/$/, '') || '/';
  const re = '^' + r.split('/').map((s) => s.startsWith('[...') ? '.*' : s.startsWith('[') ? '[^/]+' : s.replace(/[.]/g, '\\.')).join('/') + '/?$';
  return new RegExp(re);
});
const redirects = [...readFileSync(join(ROOT, 'netlify.toml'), 'utf8').matchAll(/^\s*from = "([^"]+)"/gm)]
  .map((m) => new RegExp('^' + m[1].replace(/\*/g, '.*') + '/?$'));

// Written at build time, not committed: the staged programs (www/scripts/stage.mjs)
// and the sitemap (@astrojs/sitemap).
const GENERATED = [/^\/programs\//, /^\/sitemap-index\.xml$/];

const served = (path) => {
  const p = path.split(/[?#]/)[0] || '/';
  if (GENERATED.some((r) => r.test(p))) return true;
  if (routes.some((r) => r.test(p)) || redirects.some((r) => r.test(p))) return true;
  return existsSync(join(PUBLIC, p)) && p !== '/';
};

let bad = 0, seen = 0;
for (const f of walk(SRC).filter((f) => /\.(astro|jsx|tsx|js|ts|mjs)$/.test(f))) {
  const text = readFileSync(f, 'utf8');
  const re = /href=(?:"(\/[^"]*)"|\{`(\/[^`]*)`\})/g;
  let m;
  while ((m = re.exec(text))) {
    let link = m[1] ?? m[2];
    if (link.startsWith('//')) continue;
    // a template literal: keep the static prefix; a dynamic tail matches anything
    const dyn = link.indexOf('${');
    if (dyn >= 0) link = link.slice(0, dyn) + 'x';
    seen++;
    if (!served(link)) {
      bad++;
      const line = text.slice(0, m.index).split('\n').length;
      console.log(`${relative(ROOT, f)}:${line}  ${m[1] ?? m[2]}`);
    }
  }
}
console.log(`linkcheck: ${seen} internal links, ${bad} with no page`);
process.exit(bad ? 1 : 0);
