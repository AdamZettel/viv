// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The build's check on netlify.toml's security headers, against what the build
// actually wrote to dist/:
//
//   1. every inline <script> in every page has its sha256 in the
//      Content-Security-Policy's script-src. The policy has no 'unsafe-inline',
//      so a script missing from it would not run in production -- the build
//      fails here instead, naming the page;
//   2. every page, and a path with no page (what the 404 is served at), is
//      covered by a header rule that refuses framing (frame-ancestors or
//      X-Frame-Options). The embed is no exception: other sites load its
//      script and wasm, never a page of it in a frame.
//
//     node scripts/csp.mjs            check (npm run build runs this)
//     node scripts/csp.mjs --write    put the current hashes into netlify.toml
//
// The header rules are read with the same matcher serve-dist.mjs serves them
// with (headerRules / headersFor), so what is checked is what the site test ran.
import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs';
import { join, dirname, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { headerRules, headersFor, TOML_PATH } from './netlify-headers.mjs';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const DIST = join(WWW, 'dist');
const write = process.argv.includes('--write');

function htmlFiles(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) out.push(...htmlFiles(p));
    else if (name.endsWith('.html')) out.push(p);
  }
  return out;
}
/** The URL path a file is served at: /x/index.html -> /x/. */
const urlOf = (file) => '/' + relative(DIST, file).split(sep).join('/').replace(/(^|\/)index\.html$/, '$1');

// ---- 1. inline scripts ------------------------------------------------------------
const hashes = new Map(); // hash -> first page it is on
for (const file of htmlFiles(DIST)) {
  const html = readFileSync(file, 'utf8');
  for (const m of html.matchAll(/<script(\s[^>]*)?>([\s\S]*?)<\/script>/gi)) {
    if (/\ssrc\s*=/i.test(m[1] || '')) continue;
    // a data block (JSON, a template) is not run, so CSP does not apply to it
    const type = ((m[1] || '').match(/\stype\s*=\s*["']?([^"'\s>]+)/i) || [])[1];
    if (type && !/^(module|text\/javascript|application\/javascript)$/i.test(type)) continue;
    const h = `'sha256-${createHash('sha256').update(m[2], 'utf8').digest('base64')}'`;
    if (!hashes.has(h)) hashes.set(h, urlOf(file));
  }
}

let toml = readFileSync(TOML_PATH, 'utf8');
const HASH_LINE = /^[ \t]*'sha256-[^']*' \\\r?\n/gm;
if (write) {
  const lines = [...hashes.keys()].sort().map((h) => `        ${h} \\\n`).join('');
  const first = toml.search(HASH_LINE);
  if (first < 0) { console.error('csp: netlify.toml has no hash lines to replace'); process.exit(2); }
  toml = toml.replace(HASH_LINE, '');
  toml = toml.slice(0, first) + lines + toml.slice(first);
  writeFileSync(TOML_PATH, toml);
  console.log(`csp: wrote ${hashes.size} script hashes into netlify.toml`);
}

const problems = [];
const policyOf = (path) => headersFor(headerRules(), path)['Content-Security-Policy'] || '';
const scriptSrc = (policy) => ((policy.match(/(?:^|;)\s*script-src\s+([^;]*)/) || [])[1] || '').split(/\s+/);
const allowed = new Set(scriptSrc(policyOf('/')));
if (allowed.has("'unsafe-inline'")) problems.push("script-src allows 'unsafe-inline'");
for (const [h, page] of hashes) if (!allowed.has(h)) problems.push(`${page}: an inline script whose hash ${h} is not in script-src (run: node scripts/csp.mjs --write)`);
const stale = [...allowed].filter((h) => h.startsWith("'sha256-") && !hashes.has(h));

// ---- 2. framing -------------------------------------------------------------------
let framed = 0;
// a page is served at its pretty address (/posts/) and at its file's own
// (/posts/index.html, /index.html); a rule for "/" matches only the first
const addressesOf = (file) => [...new Set([urlOf(file), '/' + relative(DIST, file).split(sep).join('/')])];
const refuses = (path) => {
  const h = headersFor(headerRules(), path);
  return /^(DENY|SAMEORIGIN)$/i.test(h['X-Frame-Options'] || '') || /frame-ancestors\s+'none'/.test(h['Content-Security-Policy'] || '');
};
for (const file of htmlFiles(DIST)) for (const path of addressesOf(file)) {
  if (!refuses(path)) problems.push(`${path}: no header rule refuses framing it; netlify.toml's "/*" rule should (frame-ancestors 'none')`);
  else framed++;
}
// Netlify matches header rules against the requested path, so the 404 page is
// served with the headers of whatever path was asked for
const NO_PAGE = '/no-such-page/';
if (!refuses(NO_PAGE)) problems.push(`${NO_PAGE}: a path with no page (the 404) can be framed`);

console.log(`csp: ${hashes.size} inline scripts, all hashed in script-src; ${framed} page addresses and the 404 refuse framing${stale.length ? `; ${stale.length} hash(es) no page uses (node scripts/csp.mjs --write drops them)` : ''}`);
if (problems.length) {
  for (const p of problems) console.error(`csp: ${p}`);
  process.exit(1);
}
