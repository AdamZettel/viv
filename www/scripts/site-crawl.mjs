// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Crawl the built site for dead links and broken pages.
//
//     node scripts/site-crawl.mjs --base http://127.0.0.1:4325 [--dist dist] [--out crawl.json]
//     node scripts/site-crawl.mjs --serve --base http://127.0.0.1:4377 --dist /copy/of/dist
//     node scripts/site-crawl.mjs --static-only ...      (no browser)
//     node scripts/site-crawl.mjs --limit 50 ...         (the first 50 pages, for a quick look)
//     node scripts/site-crawl.mjs --pages /posts/,/claims/ ...   (these pages, and the links they lead to)
//
// `--serve` starts scripts/serve-dist.mjs on the base URL's port with DIST set
// to --dist, so the redirects and headers are netlify.toml's. Point --dist at
// a copy of dist/ when other builds may run during the crawl.
//
// Three passes.
//   Static: every href, src and srcset in dist's HTML (script bodies left out)
//   that names a site path must be a file in dist, a directory with an
//   index.html, or a path a netlify.toml rule serves; a link to /page#id must
//   name an id that page's HTML has. Every distinct external link gets a HEAD
//   (a GET when HEAD is refused), with a timeout.
//   Browser: every page in dist (the sitemap, plus every index.html), and every
//   distinct /workbench?open=, /atlas?open= and /explore/ link found in the
//   HTML or in a rendered page. Each is loaded, given network idle and two
//   more seconds, and the crawl records responses of 400 and up (with the page
//   that asked), console errors, uncaught page errors, in-page anchors with no
//   target, errors a page prints for the reader ("could not open", "no
//   module"), and post headers shown as highlighted code.
//   Imports: every program the workbench is asked to open has its imports
//   expanded by the compiler with a resolver that answers as the browser's
//   does, so a program whose Run would fail on an import is found without
//   running it.
// The JSON has every finding and the findings grouped into classes (one root
// cause each); the summary lists each class with its count and three examples.
import { readFileSync, writeFileSync, existsSync, statSync, readdirSync } from 'node:fs';
import { join, dirname, resolve, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));

// ---- arguments ------------------------------------------------------------------

const argv = process.argv.slice(2);
const flag = (name) => argv.includes(`--${name}`);
const opt = (name, dflt) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : dflt;
};
const BASE = new URL(opt('base', 'http://127.0.0.1:4325')).origin;
const DIST = resolve(opt('dist', join(WWW, 'dist')));
const OUT = resolve(opt('out', 'crawl.json'));
const CONCURRENCY = Number(opt('concurrency', 4));
const LIMIT = Number(opt('limit', 0));
const SETTLE_MS = Number(opt('settle', 2000));
const IDLE_MS = Number(opt('idle-timeout', 15000));
const LOAD_MS = Number(opt('load-timeout', 30000));
const EXT_TIMEOUT_MS = Number(opt('external-timeout', 10000));
const STATIC_ONLY = flag('static-only');
const NO_EXTERNAL = flag('no-external');
const SERVE = flag('serve');
const SITE_HOSTS = new Set(['viv-lang.org', 'www.viv-lang.org', new URL(BASE).host]);
const EXAMPLES = 3;

if (!existsSync(join(DIST, 'index.html'))) {
  console.error(`site-crawl: no index.html in ${DIST}; build the site first`);
  process.exit(2);
}

// ---- dist -----------------------------------------------------------------------

function walk(dir, out = []) {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) walk(p, out);
    else if (e.name.endsWith('.html')) out.push(p);
  }
  return out;
}
const HTML_FILES = walk(DIST);

/** The site path a dist HTML file is served at. */
const pathOfFile = (file) => {
  const rel = relative(DIST, file).split(sep).join('/');
  if (rel === 'index.html') return '/';
  if (rel.endsWith('/index.html')) return `/${rel.slice(0, -'index.html'.length)}`;
  return `/${rel}`;
};

const isFile = (p) => existsSync(p) && statSync(p).isFile();

/** The dist file a site path is served from (a directory's index.html too), or null. */
function fileFor(pathname) {
  let dec;
  try { dec = decodeURIComponent(pathname); } catch { dec = pathname; }
  const p = join(DIST, dec.replace(/^\/+/, ''));
  if (!p.startsWith(DIST)) return null;
  if (isFile(p)) return p;
  if (isFile(join(p, 'index.html'))) return join(p, 'index.html');
  if (isFile(`${p}.html`)) return `${p}.html`;
  return null;
}

/** netlify.toml's [[redirects]] (uncommented), in order. */
function redirectRules() {
  const toml = join(WWW, '..', 'netlify.toml');
  if (!existsSync(toml)) return [];
  const out = [];
  for (const block of readFileSync(toml, 'utf8').split(/^\[\[redirects\]\]/m).slice(1)) {
    const head = block.split(/\n\s*\[/)[0];
    const field = (k) => (head.match(new RegExp(`^\\s*${k}\\s*=\\s*"([^"]*)"`, 'm')) || [])[1];
    const from = field('from');
    const to = field('to');
    if (!from || !to) continue;
    out.push({
      from, to,
      status: Number((head.match(/^\s*status\s*=\s*(\d+)/m) || [])[1] || 301),
      force: /^\s*force\s*=\s*true/m.test(head),
    });
  }
  return out;
}
const RULES = redirectRules();

/** The rule that serves a path, Netlify's way (a file wins unless `force`), with its target. */
function ruleFor(pathname) {
  for (const r of RULES) {
    const src = `^${r.from.replace(/[.+?^${}()|[\]\\]/g, '\\$&').replace(/\/\*$/, '/(?<splat>.*)').replace(/:(\w+)/g, '(?<$1>[^/]+)')}/?$`;
    const m = pathname.match(new RegExp(src));
    if (!m) continue;
    if (!r.force && fileFor(pathname)) return null;
    let to = r.to;
    for (const [k, v] of Object.entries(m.groups || {})) to = to.replace(`:${k}`, v || '');
    return { ...r, target: to };
  }
  return null;
}

/** Whether a site path is served: a file, or a rule whose target is (or is off-site). */
function served(pathname) {
  if (fileFor(pathname)) return { ok: true };
  const r = ruleFor(pathname);
  if (!r) return { ok: false };
  if (/^https?:/.test(r.target) || r.target.startsWith('/.netlify/')) return { ok: true, via: r.from };
  const t = new URL(r.target, 'http://x').pathname;
  return { ok: !!fileFor(t) || !!ruleFor(t), via: r.from };
}

// ---- the static pass ------------------------------------------------------------

/** An attribute value that is code, not a link: template text, expressions. */
const looksLikeTemplate = (v) => /\$\{|\{\{|\}\}|[{}`<>\n]|'\s*\+|"\s*\+|\+\s*'|^\s*$/.test(v);
const decodeEntities = (s) => s.replace(/&amp;/g, '&').replace(/&#x27;|&#39;/g, "'").replace(/&quot;/g, '"').replace(/&lt;/g, '<').replace(/&gt;/g, '>');

const idCache = new Map();
/** The ids and anchor names a dist HTML file has. */
function idsOf(file) {
  if (!idCache.has(file)) {
    const html = readFileSync(file, 'utf8');
    const ids = new Set();
    for (const m of html.matchAll(/\s(?:id|name)\s*=\s*(?:"([^"]*)"|'([^']*)')/g)) ids.add(decodeEntities(m[1] ?? m[2]));
    idCache.set(file, ids);
  }
  return idCache.get(file);
}

// A hash a page's script reads as state, not a place: `#view=code`, `#tag=x`,
// and the lanes /posts opens on (src/pages/posts.astro).
// A /vipyr/map/<fn>/ page reads its hash as the def to select
// (components/CodeMap.jsx), so `#run_c` names a def of that page's map file.
const HASH_ROUTES = { '/posts/': /^(published|reviewed|draft|ai)$/ };
const mapDefCache = new Map();
function mapDefIds(page) {
  const m = page.match(/^\/vipyr\/map\/([^/]+)\/$/);
  if (!m) return null;
  if (!mapDefCache.has(page)) {
    let ids = new Set();
    try {
      const index = JSON.parse(readFileSync(join(WWW, 'src', 'data', 'codemap.json'), 'utf8'));
      const key = index.functions[decodeURIComponent(m[1])]?.map;
      if (key) ids = new Set(JSON.parse(readFileSync(join(DIST, 'codemap', `${key}.json`), 'utf8')).defs.map((d) => d.id));
    } catch { /* no map: no def routes */ }
    mapDefCache.set(page, ids);
  }
  return mapDefCache.get(page);
}
const isHashRoute = (page, id) => id === '' || id === 'top' || id.includes('=') || (HASH_ROUTES[page] ? HASH_ROUTES[page].test(id) : false) || !!mapDefIds(page)?.has(id);
const anchorCandidates = new Map(); // normalized site url -> Map(id -> first page linking it)

const findings = [];
const add = (f) => findings.push(f);
const crawlTargets = new Map(); // site url (path + search) -> first page that linked it
const externals = new Map(); // external url -> [pages]

function staticPass() {
  let links = 0;
  for (const file of HTML_FILES) {
    const page = pathOfFile(file);
    const pageUrl = new URL(page, BASE);
    // script and style bodies hold code, whose strings are not links
    const html = readFileSync(file, 'utf8')
      .replace(/(<script\b[^>]*>)[\s\S]*?<\/script>/gi, '$1</script>')
      .replace(/(<style\b[^>]*>)[\s\S]*?<\/style>/gi, '$1</style>');
    const values = [];
    // a preconnect or dns-prefetch names an origin, not a page
    const tags = html.replace(/<link\b[^>]*rel=["']?(?:preconnect|dns-prefetch)[^>]*>/gi, '');
    for (const m of tags.matchAll(/\s(href|src|srcset)\s*=\s*(?:"([^"]*)"|'([^']*)')/gi)) {
      const attr = m[1].toLowerCase();
      const raw = decodeEntities(m[2] ?? m[3]);
      if (attr === 'srcset') for (const part of raw.split(',')) values.push(part.trim().split(/\s+/)[0]);
      else values.push(raw);
    }
    for (const v of values) {
      if (!v || looksLikeTemplate(v) || /^(mailto|tel|javascript|data|blob|about):/i.test(v)) continue;
      let u;
      try { u = new URL(v, pageUrl); } catch { add({ kind: 'bad-url', url: v, page }); continue; }
      links += 1;
      if (u.protocol !== 'http:' && u.protocol !== 'https:') continue;
      if (!SITE_HOSTS.has(u.host)) {
        const key = u.href.replace(/#.*$/, '');
        if (!externals.has(key)) externals.set(key, []);
        if (externals.get(key).length < EXAMPLES) externals.get(key).push(page);
        continue;
      }
      // a site link (an absolute viv-lang.org one too)
      const path = u.pathname;
      const s = served(path);
      if (!s.ok) {
        add({ kind: 'static-missing', url: path + u.search, raw: v, page });
        continue;
      }
      if (/^\/(workbench|atlas)\/?$/.test(path) && u.searchParams.has('open')) remember(`${path.replace(/\/?$/, '/')}${u.search}`, page);
      else if (/^\/explore\/./.test(path) && !/\.\w+$/.test(path)) remember(path, page);
      if (u.hash.length > 1 && !s.via) {
        const target = fileFor(path);
        if (target && target.endsWith('.html')) {
          let id = u.hash.slice(1);
          try { id = decodeURIComponent(id); } catch { /* as written */ }
          if (!idsOf(target).has(id) && !isHashRoute(pathOfFile(target), id)) {
            // a candidate: the browser pass looks for the id after the page renders
            const key = normalize(path + u.search);
            if (!anchorCandidates.has(key)) anchorCandidates.set(key, new Map());
            const m = anchorCandidates.get(key);
            // a heading id slugged from an entity (`&#39;` -> -39-) beside a link slugged from the text
            const entity = [...idsOf(target)].some((t) => t.replace(/-(?:39|34|38|60|62|quot|amp|lt|gt)-/g, '-') === id);
            if (!m.has(id)) m.set(id, { page, entity });
          }
        }
      }
    }
  }
  return links;
}

/**
 * The workbench's file lists (the `projects` prop of its island): each entry
 * opens /programs/<slug>/<file>, so each must be a file in dist. These are
 * buttons, not links, so neither the link check nor the browser pass sees them.
 */
function workbenchFilesPass() {
  const file = join(DIST, 'workbench', 'index.html');
  if (!isFile(file)) return 0;
  const html = decodeEntities(readFileSync(file, 'utf8'));
  let n = 0;
  for (const m of html.matchAll(/"slug":\[0,"([^"]+)"\],"title":\[0,"[^"]*"\],"files":\[1,\[((?:\[0,"[^"]*"\],?)*)\]\]/g)) {
    for (const f of m[2].matchAll(/\[0,"([^"]*)"\]/g)) {
      n += 1;
      if (!progExists(`${m[1]}/${f[1]}`)) add({ kind: 'workbench-file', url: `/programs/${m[1]}/${f[1]}`, group: m[1], page: '/workbench/' });
      else if (f[1].endsWith('.viv')) remember(`/workbench/?open=${m[1]}/${f[1]}`, '/workbench/');
    }
  }
  return n;
}

/**
 * Every program the workbench is asked to open (?open= links, and its own
 * lists), with its imports expanded the way the page does it: the compiler
 * (src/lib/uses.js) and a resolver that answers as the browser's does
 * (browserResolver: /programs/<name>.viv, posts by slug, and a module named
 * beside the program next to the program's own staged path). A program whose
 * imports do not resolve compiles to an error when the reader presses Run.
 */
async function importsPass() {
  const { expandUses, hasUses } = await import('../src/lib/uses.js');
  const text = (rel) => (progExists(rel) ? readFileSync(join(DIST, 'programs', rel), 'utf8') : null);
  const postIndex = (() => { try { return JSON.parse(text('posts/index.json') || '{}'); } catch { return {}; } })();
  const resolverFor = (rel) => async (name) => {
    if (name.startsWith('post:')) {
      const f = postIndex[name.slice(5)];
      const t = f ? text(`posts/${f}`) : null;
      if (t === null) throw new Error(`no post \`${name.slice(5)}\``);
      return t;
    }
    let path = name;
    if (name.startsWith('./')) {
      const dir = rel.includes('/') ? rel.slice(0, rel.lastIndexOf('/')) : '';
      path = dir ? `${dir}/${name.slice(2)}` : name.slice(2);
    }
    const t = text(`${path}.viv`);
    if (t === null) throw new Error(`no module \`${name}\``);
    return t;
  };
  const programs = new Map(); // program path under /programs -> a page that opens it
  for (const [url, page] of crawlTargets) {
    const u = new URL(url, BASE);
    const open = u.searchParams.get('open');
    if (/^\/workbench\/?$/.test(u.pathname) && open && !programs.has(open)) programs.set(open, page);
  }
  let n = 0;
  for (const [rel, page] of programs) {
    const src = text(rel);
    if (src === null) {
      add({ kind: 'open-missing', url: `/workbench/?open=${rel}`, page });
      continue;
    }
    if (!hasUses(src)) continue;
    n += 1;
    try {
      await expandUses(src, resolverFor(rel));
    } catch (e) {
      add({ kind: 'import-error', url: `/workbench/?open=${rel}`, program: rel, text: String(e.message).split('\n')[0].slice(0, 300), page });
    }
  }
  return n;
}

function remember(url, page) {
  if (!crawlTargets.has(url)) crawlTargets.set(url, page);
}

async function externalStatus(url) {
  const tryIt = async (method) => {
    const r = await fetch(url, { method, redirect: 'manual', signal: AbortSignal.timeout(EXT_TIMEOUT_MS), headers: { 'user-agent': 'viv-site-crawl (link check)' } });
    r.body?.cancel().catch(() => {});
    return r.status;
  };
  let status = await tryIt('HEAD');
  if (status === 405 || status === 403 || status === 501 || status === 400) status = await tryIt('GET');
  // rate limited: wait once and ask again
  if (status === 429) {
    await new Promise((ok) => { setTimeout(ok, 5000); });
    status = await tryIt('GET');
  }
  return status;
}

async function externalPass() {
  let list = [...externals.keys()];
  // links into a GitHub repository: ask for the repository once; when it is
  // not there (private, or not created yet) every link into it is one finding
  // each, without a request for each
  const repoOf = (url) => (url.match(/^https:\/\/github\.com\/([^/]+\/[^/#?]+)/) || [])[1];
  const repos = [...new Set(list.map(repoOf).filter(Boolean))];
  const missingRepos = new Set();
  for (const r of repos) {
    try {
      const st = await externalStatus(`https://github.com/${r}`);
      if (st === 404) missingRepos.add(r);
    } catch { /* checked link by link below */ }
  }
  list = list.filter((url) => {
    const r = repoOf(url);
    if (r && missingRepos.has(r)) {
      add({ kind: 'external', url, status: 404, repoMissing: r, pages: externals.get(url), page: externals.get(url)[0] });
      return false;
    }
    return true;
  });
  let i = 0;
  const one = async (url) => {
    try {
      const status = await externalStatus(url);
      if (status < 200 || status >= 400) add({ kind: 'external', url, status, pages: externals.get(url), page: externals.get(url)[0] });
    } catch (e) {
      add({ kind: 'external', url, status: 0, error: String(e.cause?.code || e.name || e.message), pages: externals.get(url), page: externals.get(url)[0] });
    }
  };
  // one host at a time each: GitHub answers 429 to a burst
  const byHost = new Map();
  for (const url of list) {
    const h = new URL(url).host;
    if (!byHost.has(h)) byHost.set(h, []);
    byHost.get(h).push(url);
  }
  const hosts = [...byHost.values()];
  await Promise.all(Array.from({ length: 8 }, async () => {
    while (i < hosts.length) {
      for (const url of hosts[i++]) await one(url);
    }
  }));
  return list.length + [...externals.keys()].length - list.length;
}

// ---- the browser pass -----------------------------------------------------------

function pagesToVisit() {
  const set = new Set();
  const sitemap = join(DIST, 'sitemap-0.xml');
  if (existsSync(sitemap)) {
    for (const m of readFileSync(sitemap, 'utf8').matchAll(/<loc>([^<]+)<\/loc>/g)) set.add(new URL(m[1]).pathname);
  }
  for (const f of HTML_FILES) {
    const p = pathOfFile(f);
    if (p === '/404.html') continue;
    set.add(p);
  }
  return [...set].sort();
}

/** A site path the way the deploy serves it: a directory path ends in a slash (Netlify's /x -> /x/). */
function normalize(url) {
  const u = new URL(url, BASE);
  if (!u.pathname.endsWith('/') && !/\.\w+$/.test(u.pathname) && fileFor(`${u.pathname}/`)) u.pathname += '/';
  return u.pathname + u.search;
}

function loadPlaywright() {
  const tries = [process.env.PLAYWRIGHT_CORE, join(WWW, 'node_modules', 'playwright-core'), join(WWW, '..', '..', 'viv', 'www', 'node_modules', 'playwright-core')].filter(Boolean);
  const req = createRequire(import.meta.url);
  for (const t of tries) {
    try { return req(t); } catch { /* next */ }
  }
  return req('playwright-core');
}

function chromePath() {
  if (process.env.CHROME) return process.env.CHROME;
  const root = join(process.env.HOME || '', 'Library', 'Caches', 'ms-playwright');
  if (!existsSync(root)) return undefined;
  for (const d of readdirSync(root).filter((n) => n.startsWith('chromium-')).sort().reverse()) {
    for (const mac of readdirSync(join(root, d)).filter((n) => n.startsWith('chrome-mac'))) {
      const apps = readdirSync(join(root, d, mac)).filter((n) => n.endsWith('.app'));
      for (const a of apps) {
        const dir = join(root, d, mac, a, 'Contents', 'MacOS');
        const bin = readdirSync(dir)[0];
        if (bin) return join(dir, bin);
      }
    }
  }
  return undefined;
}

// what a page runs in the browser after it settles
function inPage(wanted) {
  const out = { anchors: [], missingIds: [], links: [], printed: [], headerAsCode: [] };
  for (const id of wanted) if (!document.getElementById(id) && !document.getElementsByName(id).length) out.missingIds.push(id);
  const here = location.pathname + location.search;
  for (const a of document.querySelectorAll('a[href]')) {
    const href = a.getAttribute('href');
    let u;
    try { u = new URL(href, location.href); } catch { continue; }
    if (u.origin === location.origin) out.links.push(u.pathname + u.search);
    if (u.origin === location.origin && u.pathname + u.search === here && u.hash.length > 1) {
      let id = u.hash.slice(1);
      try { id = decodeURIComponent(id); } catch { /* as written */ }
      if (id !== 'top' && !id.includes('=') && !document.getElementById(id) && !document.getElementsByName(id).length) out.anchors.push(href);
    }
  }
  // errors the page shows its reader
  const text = document.body ? document.body.innerText : '';
  for (const re of [/error: could not open [^\n]*/g, /no module `[^`\n]*`[^\n]*/g, /no post `[^`\n]*`[^\n]*/g, /could not load [^\n]{0,120}/gi, /(?:^|\n)error: [^\n]{0,200}/g]) {
    for (const m of text.matchAll(re)) out.printed.push(m[0].slice(0, 200));
  }
  // a post's header (the fenced `###` block) shown as code
  const looksLikeHeader = (t) => /(^|\n)\s*post:\s*\S/.test(t) && /(^|\n)\s*(kind|summary|channel|date):\s*\S/.test(t);
  const codey = document.querySelectorAll('pre, code, .monaco-editor .view-lines, .cm-content, textarea');
  for (const el of codey) {
    const t = el.innerText || el.value || '';
    if (!looksLikeHeader(t)) continue;
    // the innermost such element, once
    if ([...el.querySelectorAll('pre, code')].some((c) => looksLikeHeader(c.innerText || ''))) continue;
    // Monaco's hidden input area holds the buffer's text; its view lines are checked below
    if (el.matches('.monaco-editor textarea')) continue;
    // a Monaco editor: what its grammar makes of the fenced block, not what it shows
    if (el.matches('.monaco-editor .view-lines') && window.monaco) {
      const model = window.monaco.editor.getModels().find((m) => looksLikeHeader(m.getValue()));
      if (model) {
        const lines = model.getValue().split('\n');
        const toks = window.monaco.editor.tokenize(model.getValue(), model.getLanguageId());
        let fence = false;
        let asCode = false;
        lines.forEach((l, i) => {
          if (/^###\s*$/.test(l)) { fence = !fence; return; }
          if (fence && toks[i].some((t) => !/^(prose|comment)/.test(t.type))) asCode = true;
        });
        if (!asCode) continue;
      }
    }
    const highlighted = el.matches('.monaco-editor .view-lines') || el.querySelector('span[class]') !== null;
    let where = el;
    const trail = [];
    for (let i = 0; i < 4 && where && where !== document.body; i += 1, where = where.parentElement) {
      trail.push(where.tagName.toLowerCase() + (where.className && typeof where.className === 'string' ? `.${where.className.trim().split(/\s+/).slice(0, 2).join('.')}` : ''));
    }
    out.headerAsCode.push({ highlighted, where: trail.reverse().join(' > '), text: t.slice(0, 160) });
  }
  out.links = [...new Set(out.links)];
  return out;
}

async function browserPass(initial) {
  const { chromium } = loadPlaywright();
  const browser = await chromium.launch({ executablePath: chromePath(), headless: true });
  const queue = [...initial];
  const queued = new Set(queue);
  let visited = 0;
  const started = Date.now();
  const enqueue = (url, from) => {
    const n = normalize(url);
    if (queued.has(n)) return;
    queued.add(n);
    queue.push(n);
    if (!crawlTargets.has(n)) crawlTargets.set(n, from);
  };
  const worker = async () => {
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    while (queue.length) {
      if (LIMIT && visited >= LIMIT) break;
      if (serverGone) break;
      const url = queue.shift();
      visited += 1;
      const page = await context.newPage();
      page.on('response', (r) => {
        if (r.status() >= 400) add({ kind: 'http', status: r.status(), url: r.url(), page: url, type: r.request().resourceType() });
      });
      page.on('requestfailed', (r) => {
        const why = r.failure()?.errorText || '';
        if (!/ERR_ABORTED/.test(why)) add({ kind: 'request-failed', url: r.url(), error: why, page: url });
      });
      page.on('console', (m) => {
        if (m.type() !== 'error') return;
        const text = m.text();
        // the response listener already has these, with the URL
        if (/^Failed to load resource: the server responded with a status of/.test(text)) return;
        add({ kind: 'console', text: text.slice(0, 400), at: m.location()?.url || '', page: url });
      });
      page.on('pageerror', (e) => add({ kind: 'pageerror', text: String(e.message).slice(0, 400), stack: String(e.stack || '').split('\n').slice(0, 3).join(' | '), page: url }));
      try {
        const resp = await page.goto(BASE + url, { waitUntil: 'load', timeout: LOAD_MS });
        if (resp && resp.status() >= 400) add({ kind: 'page-status', status: resp.status(), url, page: crawlTargets.get(url) || '(dist)' });
        await page.waitForLoadState('networkidle', { timeout: IDLE_MS }).catch(() => add({ kind: 'no-idle', url, page: url }));
        await page.waitForTimeout(SETTLE_MS);
        const wanted = anchorCandidates.get(url) || new Map();
        const r = await page.evaluate(inPage, [...wanted.keys()]);
        // the path the page ended on: /feed/ is a redirect to /posts/ (netlify.toml
        // 301s it; the dist page is a meta refresh), and its links are /posts/'s
        const pagePath = normalize(new URL(page.url()).pathname);
        for (const a of r.anchors) {
          const id = decodeURIComponent(a.replace(/^[^#]*#/, ''));
          if (!isHashRoute(pagePath, id) && !wanted.has(id)) add({ kind: 'anchor', url: `${url}#${id}`, page: url, same: true });
        }
        for (const id of r.missingIds) add({ kind: 'anchor', url: `${url}#${id}`, page: wanted.get(id).page, same: wanted.get(id).page === url, entity: wanted.get(id).entity });
        for (const p of r.printed) add({ kind: 'printed', text: p, page: url });
        for (const h of r.headerAsCode) add({ kind: 'header-as-code', ...h, page: url });
        for (const l of r.links) {
          const u = new URL(l, BASE);
          if ((/^\/(workbench|atlas)\/?$/.test(u.pathname) && u.searchParams.has('open')) || (/^\/explore\/./.test(u.pathname) && !/\.\w+$/.test(u.pathname))) enqueue(l, url);
        }
      } catch (e) {
        add({ kind: 'load-failed', url, error: String(e.message).split('\n')[0], page: url });
      }
      await page.close().catch(() => {});
      if (visited % 50 === 0) {
        const rate = (Date.now() - started) / visited / 1000;
        console.error(`site-crawl: ${visited} visited, ${queue.length} queued (${rate.toFixed(1)} s each, ${findings.length} findings)`);
      }
    }
    await context.close();
  };
  await Promise.all(Array.from({ length: CONCURRENCY }, worker));
  await browser.close();
  return visited;
}

// ---- classes --------------------------------------------------------------------

const progExists = (p) => isFile(join(DIST, 'programs', p));

/** A finding's class: one key per root cause, with a sentence for the summary. */
function classify(f) {
  const u = (() => { try { return new URL(f.url, BASE); } catch { return null; } })();
  const path = u ? u.pathname : String(f.url || '');
  if ((f.kind === 'http' || f.kind === 'request-failed') && u && u.origin === BASE) {
    const m = path.match(/^\/programs\/(.+)\.viv$/);
    if (m) {
      const v = probeVariant(m[1]);
      if (v) return ['import-resolver-probe', 'the import resolver fetches candidate paths for a module and 404s on every candidate that is not the module', v];
      return ['program-missing', `a program or module under /programs that is not in dist (HTTP ${f.status || f.error})`];
    }
    if (path.startsWith('/api/')) return ['api-absent', `/api requests with no server behind /api (HTTP ${f.status || f.error}); the deploy has no /api yet either`];
    return [`http-${f.status || 'failed'}-${(path.split('/')[1] || '/')}`, `HTTP ${f.status || f.error} for site resources under /${path.split('/')[1] || ''}`];
  }
  if (f.kind === 'http' || f.kind === 'request-failed') return [`offsite-${f.status || 'failed'}-${u ? u.host : '?'}`, `off-site resource ${u ? u.host : ''} answered ${f.status || f.error}`];
  if (f.kind === 'static-missing' || f.kind === 'page-status') {
    if (/\.viv[a-z]/i.test(path)) return ['concatenated-program-paths', 'a link whose path is two program paths run together (…a.vivclaims/…b.viv)'];
    if (/^\/(workbench|atlas)\//.test(path) || /^\/explore\//.test(path)) return [`dead-${path.split('/')[1]}-link`, `dead link into /${path.split('/')[1]}`];
    const seg = path.split('/').slice(0, 2).join('/') || '/';
    return [`dead-link-${seg}`, `dead site link under ${seg}`];
  }
  if (f.kind === 'import-error') {
    const mod = (f.text.match(/`([^`]+)`: no module/) || [])[1];
    const dir = f.program.includes('/') ? f.program.slice(0, f.program.lastIndexOf('/')) : '';
    if (mod) {
      const beside = `${dir ? `${dir}/` : ''}${mod.replace(/\./g, '/')}.viv`;
      if (progExists(beside)) return ['import-sibling-refused', 'a program imports a module beside it; the module is staged, but the resolver does not find it there, so Run fails with "no module by that name"'];
      if (isFile(join(WWW, '..', beside))) return ['import-sibling-not-staged', 'a program imports a module beside it that the build did not copy into /programs, so Run fails with "no module by that name"'];
    }
    return [`import-error:${f.text.replace(/`[^`]*`/g, '`…`').replace(/\d+/g, 'N').slice(0, 70)}`, `the program's imports do not expand: ${f.text.replace(/`[^`]*`/g, '`…`').slice(0, 100)}`];
  }
  if (f.kind === 'open-missing') return ['workbench-open-missing', 'a /workbench?open= link to a program that is not under /programs'];
  if (f.kind === 'workbench-file') {
    const rel = path.replace(/^\/programs\/[^/]+\//, '');
    const elsewhere = progExists(rel) ? ` (the file is at /programs/${rel})` : '';
    return [`workbench-list-${f.group}`, `the workbench's ${f.group} list has files that open as /programs/${f.group}/<file>, which is not there${elsewhere ? '; they are staged at /programs/<file>' : ''}`];
  }
  if (f.kind === 'static-anchor' && f.entity) return ['anchor-entity-slug', 'a heading id made from an HTML entity (`&#39;` became -39-) while the link was made from the text: the table of contents misses'];
  if (f.kind === 'static-anchor') return [f.same ? 'static-anchor-same-page' : 'static-anchor-other-page', f.same ? 'a #id link to an id the page\'s HTML does not have (static check; a client-rendered id would not show here)' : 'a /page#id link to an id the target page\'s HTML does not have (static check)'];
  if (f.kind === 'anchor') {
    if (f.entity) return ['anchor-entity-slug', 'a heading id made from an HTML entity (`&#39;` became -39-) while the link was made from the text: the table of contents misses'];
    return [f.same ? 'anchor-same-page' : `anchor-to-${pageFamily(String(f.url))}`, f.same ? 'an in-page #id link with no element of that id after the page rendered' : `a link to ${pageFamily(String(f.url))}/…#id where the rendered page has no such id`];
  }
  if (f.kind === 'header-as-code') return [`header-as-code-${pageFamily(f.page)}`, `a post's header and prose shown as ${f.highlighted ? 'highlighted ' : ''}code on ${pageFamily(f.page)} pages`];
  if (f.kind === 'printed') {
    if (/could not open/.test(f.text)) return ['printed-could-not-open', 'the workbench says it could not open the file it was asked to open'];
    return [`printed-${f.text.split(/[\s`]/)[0]}-${f.text.split(/[\s`]/)[1] || ''}`, `the page shows an error: ${f.text.replace(/`[^`]*`/g, '`…`').slice(0, 80)}`];
  }
  if (f.kind === 'console' || f.kind === 'pageerror') {
    const norm = f.text.replace(/https?:\/\/\S+/g, '<url>').replace(/`[^`]*`/g, '`…`').replace(/'[^']*'/g, "'…'").replace(/"[^"]*"/g, '"…"').replace(/\d+(\.\d+)?/g, 'N').slice(0, 90);
    return [`${f.kind}:${norm}`, `${f.kind === 'console' ? 'console error' : 'uncaught error'}: ${norm}`];
  }
  if (f.kind === 'external' && f.repoMissing) return [`external-repo-missing-${f.repoMissing}`, `links into github.com/${f.repoMissing}, which answers 404 (private, or not created yet)`];
  if (f.kind === 'external') return [`external-${f.status || f.error}`, `external link answered ${f.status || f.error}`];
  return [f.kind, f.kind];
}

/** Which wrong candidate a 404'd /programs/<rel>.viv is, when a right one exists nearby; else null. */
function probeVariant(rel) {
  const segs = rel.split('/');
  const parent = segs.slice(0, -1).join('/');
  if (!rel.startsWith('lib/') && progExists(`lib/${rel}.viv`)) return 'a/b.viv for the module lib/a/b.viv (no lib/ prefix)';
  if (parent && progExists(`${parent}.viv`)) return '`from a.b import NAME` fetched as the submodule a/b/NAME.viv';
  if (parent && !rel.startsWith('lib/') && progExists(`lib/${parent}.viv`)) return '`from a.b import NAME` fetched as a/b/NAME.viv, without lib/ too';
  // `./a/b` beside the program: lib/ode/ + ode/core -> lib/ode/ode/core
  const isDir = (q) => existsSync(join(DIST, 'programs', q)) && statSync(join(DIST, 'programs', q)).isDirectory();
  if (isDir(rel) || (!rel.startsWith('lib/') && isDir(`lib/${rel}`))) return '`import a` fetched as a.viv where a/ is a directory of modules';
  for (let i = 1; i < segs.length; i += 1) {
    const restParts = segs.slice(i);
    const rest = restParts.join('/');
    if (progExists(`lib/${rest}.viv`) || progExists(`${rest}.viv`)) return 'a/b.viv looked up beside the program (lib/ode/ + ode/core = lib/ode/ode/core.viv)';
    const restParent = restParts.slice(0, -1).join('/');
    if (restParent && (progExists(`lib/${restParent}.viv`) || progExists(`${restParent}.viv`))) return '`from a.b import NAME` looked up beside the program as a/b/NAME.viv';
  }
  return null;
}

function pageFamily(page) {
  const p = new URL(page, BASE).pathname;
  return `/${p.split('/')[1] || ''}`;
}

const exampleOf = (f) => {
  const where = f.page || (f.pages || []).join(', ');
  const what = f.url ? `${f.status ? `${f.status} ` : ''}${f.url}` : f.text || '';
  return `${String(what).replace(/\s*\n\s*/g, ' / ').slice(0, 160)}  <-  ${where}`;
};

function classes() {
  const by = new Map();
  for (const f of findings) {
    if (f.kind === 'no-idle') continue;
    const [key, what, variant] = classify(f);
    f.class = key;
    if (variant) f.variant = variant;
    if (!by.has(key)) by.set(key, { class: key, what, count: 0, distinct: new Set(), pages: new Set(), examples: [], variants: {} });
    const c = by.get(key);
    c.count += 1;
    if (variant) c.variants[variant] = (c.variants[variant] || 0) + 1;
    const id = f.kind === 'header-as-code' ? f.page : f.url || f.text;
    if (!c.distinct.has(id)) {
      c.distinct.add(id);
      if (c.examples.length < EXAMPLES) c.examples.push(exampleOf(f));
    }
    if (f.page) c.pages.add(f.page);
  }
  return [...by.values()]
    .map((c) => ({ class: c.class, what: c.what, count: c.count, distinct: c.distinct.size, pages: c.pages.size, examples: c.examples, ...(Object.keys(c.variants).length ? { variants: c.variants } : {}) }))
    .sort((a, b) => b.count - a.count);
}

// ---- main -----------------------------------------------------------------------

let server = null;
// set when the spawned serve-dist exits before the crawl is done: every page
// after that would fail to load, and those failures are the crawler's, not
// the site's, so the crawl stops and reports no findings
let serverGone = null;
let stopping = false;
async function startServer() {
  const port = new URL(BASE).port || '80';
  server = spawn(process.execPath, [join(WWW, 'scripts', 'serve-dist.mjs')], {
    env: { ...process.env, PORT: port, DIST, VIV_API: process.env.VIV_API || 'http://127.0.0.1:9' },
    stdio: ['ignore', 'pipe', 'inherit'],
  });
  await new Promise((ok, fail) => {
    server.stdout.on('data', (d) => { if (/serve-dist:/.test(String(d))) ok(); });
    server.on('exit', (code, signal) => {
      const why = `serve-dist exited (${signal || code})`;
      if (!stopping) serverGone = why;
      fail(new Error(why));
    });
  });
}

async function main() {
  const t0 = Date.now();
  if (SERVE) await startServer();
  const links = staticPass();
  const wbFiles = workbenchFilesPass();
  console.error(`site-crawl: ${wbFiles} files in the workbench's lists`);
  console.error(`site-crawl: static pass, ${HTML_FILES.length} HTML files, ${links} links, ${externals.size} distinct external`);
  const ext = NO_EXTERNAL ? 0 : await externalPass();
  let visited = 0;
  if (STATIC_ONLY) {
    for (const [url, ids] of anchorCandidates) for (const [id, w] of ids) add({ kind: 'static-anchor', url: `${url}#${id}`, page: w.page, same: w.page === url, entity: w.entity });
  } else {
    const only = opt('pages', '');
    const initial = only ? only.split(',').map(normalize) : [...new Set([...pagesToVisit(), ...crawlTargets.keys()].map(normalize))];
    console.error(`site-crawl: browser pass, ${initial.length} pages to start, ${CONCURRENCY} at a time`);
    visited = await browserPass(initial);
  }
  if (serverGone) throw new Error(`site-crawl: ${serverGone} after ${visited} pages; the crawl is void, no report written`);
  // after the browser pass, which adds the ?open= links pages render
  const expanded = await importsPass();
  console.error(`site-crawl: imports expanded for ${expanded} programs the workbench opens`);
  const cls = classes();
  const report = {
    base: BASE, dist: DIST, when: new Date().toISOString(), seconds: Math.round((Date.now() - t0) / 1000),
    htmlFiles: HTML_FILES.length, links, externalChecked: ext, visited,
    classes: cls, findings,
  };
  writeFileSync(OUT, `${JSON.stringify(report, null, 1)}\n`);
  const lines = [`site-crawl: ${visited} pages visited, ${HTML_FILES.length} HTML files read, ${ext} external links checked, ${findings.length} findings in ${cls.length} classes (${report.seconds} s)`, ''];
  for (const c of cls) {
    lines.push(`[${c.count}] ${c.class}  (${c.distinct} distinct, on ${c.pages} pages)`);
    lines.push(`    ${c.what}`);
    for (const [v, n] of Object.entries(c.variants || {})) lines.push(`    ${n}x ${v}`);
    for (const e of c.examples) lines.push(`    - ${e}`);
  }
  lines.push('', `site-crawl: JSON at ${OUT}`);
  console.log(lines.join('\n'));
  stopping = true;
  if (server) server.kill();
}

main().catch((e) => {
  console.error(e);
  stopping = true;
  if (server) server.kill();
  process.exit(1);
});
