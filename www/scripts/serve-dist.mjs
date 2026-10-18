// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Serve the built site the way the deploy will, so it can be rehearsed before
// it is deployed: dist/ over HTTP, the redirect rules and the response headers
// (the Content-Security-Policy among them) read from netlify.toml rather than
// restated here, and /api proxied to a running server/.
//
//     node scripts/serve-dist.mjs               port 4325, api on 8787
//     PORT=8080 VIV_API=http://localhost:8788 node scripts/serve-dist.mjs
//     DIST=/some/copy/of/dist node scripts/serve-dist.mjs   (a snapshot, so a rebuild cannot change it mid-crawl)
//
// `astro preview` cannot do this: it has no proxy and does not know the
// redirects, so a post or a channel published on the site 404s under it and
// nothing checks that netlify.toml's rules are the ones the site needs.
import { createServer, request as httpRequest } from 'node:http';
import { readFileSync, existsSync, statSync } from 'node:fs';
import { dirname, join, extname, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { headerRules, headersFor } from './netlify-headers.mjs';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const DIST = process.env.DIST || join(WWW, 'dist');
const API = new URL(process.env.VIV_API || 'http://localhost:8787');
const PORT = Number(process.env.PORT || 4325);

if (!existsSync(DIST)) {
  console.error('serve-dist: no dist/ -- run `npm run build` first');
  process.exit(2);
}

/** netlify.toml's `[[redirects]]`, in order: { from, to, status, force }. */
function redirects() {
  const toml = readFileSync(join(WWW, '..', 'netlify.toml'), 'utf8');
  const out = [];
  for (const block of toml.split('[[redirects]]').slice(1)) {
    const head = block.split(/\n\s*\[/)[0];
    const field = (k) => (head.match(new RegExp(`^\\s*${k}\\s*=\\s*"([^"]*)"`, 'm')) || [])[1];
    const status = Number((head.match(/^\s*status\s*=\s*(\d+)/m) || [])[1] || 301);
    const from = field('from');
    const to = field('to');
    const force = /^\s*force\s*=\s*true/m.test(head);
    if (from && to && !/^\s*#/.test(head.trim())) out.push({ from, to, status, force });
  }
  return out;
}
const RULES = redirects();
const HEADER_RULES = headerRules();
/** netlify.toml's headers for a path. Cache-Control is left out: rehearsing
 *  wants every reload fresh (no-store), not the deploy's caching. */
const siteHeaders = (pathname) => {
  const h = headersFor(HEADER_RULES, pathname);
  delete h['Cache-Control'];
  return h;
};

const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8', '.wasm': 'application/wasm',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.webp': 'image/webp', '.ico': 'image/x-icon',
  '.woff2': 'font/woff2', '.woff': 'font/woff', '.ttf': 'font/ttf', '.xml': 'application/xml; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8', '.viv': 'text/plain; charset=utf-8', '.map': 'application/json; charset=utf-8',
};

const fileFor = (pathname) => {
  const p = join(DIST, normalize(decodeURIComponent(pathname)).replace(/^(\.\.[/\\])+/, ''));
  if (existsSync(p) && statSync(p).isFile()) return p;
  const index = join(p, 'index.html');
  if (existsSync(index) && statSync(index).isFile()) return index;
  return null;
};

/** Apply the first matching rule, Netlify's `/x/*` -> `/y?s=:splat` shape. */
function rewrite(pathname) {
  for (const r of RULES) {
    if (r.from.endsWith('/*')) {
      const base = r.from.slice(0, -1);
      if (pathname.startsWith(base)) {
        const splat = pathname.slice(base.length);
        // a built page wins over the rule, as Netlify's static-first order does,
        // unless the rule has `force = true`
        if (!r.force && fileFor(pathname)) return null;
        return { to: r.to.replace(':splat', encodeURIComponent(splat)), status: r.status };
      }
    } else if (pathname === r.from || pathname === `${r.from}/`) {
      if (!r.force && fileFor(pathname)) return null;
      return { to: r.to, status: r.status };
    }
  }
  return null;
}

const server = createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  if (url.pathname.startsWith('/api/')) {
    const proxy = httpRequest(
      { hostname: API.hostname, port: API.port, path: req.url, method: req.method, headers: { ...req.headers, host: API.host } },
      (up) => { res.writeHead(up.statusCode || 502, up.headers); up.pipe(res); },
    );
    proxy.on('error', (e) => { res.writeHead(502, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ error: `no server behind /api: ${e.message}` })); });
    req.pipe(proxy);
    return;
  }
  const rule = rewrite(url.pathname);
  if (rule && rule.status !== 200) { res.writeHead(rule.status, { ...siteHeaders(url.pathname), Location: rule.to }); res.end(); return; }
  const target = rule ? new URL(rule.to, `http://localhost:${PORT}`) : url;
  const file = fileFor(target.pathname);
  if (!file) {
    const notFound = join(DIST, '404.html');
    res.writeHead(404, { ...siteHeaders(url.pathname), 'Content-Type': 'text/html; charset=utf-8' });
    res.end(existsSync(notFound) ? readFileSync(notFound) : 'not found');
    return;
  }
  // the headers of the path asked for (Netlify matches the request, not the rewrite target)
  res.writeHead(200, { 'Content-Type': TYPES[extname(file)] || 'application/octet-stream', ...siteHeaders(url.pathname), 'Cache-Control': 'no-store' });
  res.end(readFileSync(file));
});

server.listen(PORT, () => {
  console.log(`serve-dist: dist/ on http://localhost:${PORT}, /api -> ${API.origin}`);
  console.log(`            ${RULES.length} redirect rules from netlify.toml: ${RULES.map((r) => `${r.from} -> ${r.to}`).join(', ')}`);
  console.log(`            ${HEADER_RULES.length} header rules from netlify.toml (Content-Security-Policy ${headersFor(HEADER_RULES, '/')['Content-Security-Policy'] ? 'on' : 'OFF'})`);
});
