// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The site test: every page the build produces, loaded in a real browser.
//
//     npm run build && npm run sitetest
//     node scripts/sitetest.mjs --only /pde       just the pages under a prefix
//     node scripts/sitetest.mjs --no-run          skip pressing Run buttons
//
// A page fails if it does any of these:
//   - answers anything but 200, or is linked from another page but does not exist;
//   - throws an uncaught exception, or logs a console error;
//   - fails to load one of its own requests (a script, the wasm, a JSON file),
//     or gets a 4xx/5xx for one;
//   - renders nothing: less than 40 characters of visible text;
//   - is a post's page (/p/<slug>/) without its history: no version drawn;
//   - has a Run button that, when pressed, throws, logs an error, or shows an
//     error message where the output goes (a FAIL line counts, except in a
//     notebook of a problem post, whose tests fail until it is solved).
//
// The pages come from dist/ (every index.html the build wrote), and the test
// also follows every same-origin link it finds, so a link to a page that was
// never built is caught too. The site is served by scripts/serve-dist.mjs, the
// same server that rehearses the deploy, so netlify.toml's redirects apply.
// /api is proxied to a running server/ when there is one; without one, requests
// to /api are expected to fail and are reported separately, not as failures,
// because every page is meant to keep working on what the build put there.
//
// Writes sitetest-report.json beside this script's package, and exits 1 when
// any page fails.
import { readdirSync, statSync, writeFileSync } from 'node:fs';
import { join, dirname, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { chromium } from 'playwright-core';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const DIST = join(WWW, 'dist');
const args = process.argv.slice(2);
const only = args.includes('--only') ? args[args.indexOf('--only') + 1] : '';
const pressRun = !args.includes('--no-run');
const PORT = Number(process.env.SITETEST_PORT || 4391);
const BASE = `http://127.0.0.1:${PORT}`;
const CONCURRENCY = Number(process.env.SITETEST_JOBS || 6);
const PAGE_TIMEOUT = 30000;
const RUN_TIMEOUT = 20000;

// ---- the pages the build wrote -------------------------------------------
function builtPages(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) {
      if (name === '_astro' || name === 'viv' || name === 'monaco') continue;
      out.push(...builtPages(p));
    } else if (name === 'index.html') {
      const rel = relative(DIST, dirname(p)).split(sep).join('/');
      out.push('/' + (rel ? rel + '/' : ''));
    } else if (name.endsWith('.html') && name !== '404.html') {
      out.push('/' + relative(DIST, p).split(sep).join('/'));
    }
  }
  return out;
}

// ---- serving -------------------------------------------------------------
function startServer() {
  const child = spawn(process.execPath, [join(WWW, 'scripts', 'serve-dist.mjs')], {
    env: { ...process.env, PORT: String(PORT) },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error('serve-dist did not start')), 15000);
    const poll = async () => {
      try {
        const r = await fetch(BASE + '/');
        if (r.status) {
          clearTimeout(t);
          resolve(child);
          return;
        }
      } catch {}
      setTimeout(poll, 200);
    };
    poll();
  });
}

async function apiAvailable() {
  try {
    const r = await fetch(BASE + '/api/health');
    return r.ok;
  } catch {
    return false;
  }
}

// ---- one page ------------------------------------------------------------
const isApi = (url) => new URL(url).pathname.startsWith('/api');
const sameOrigin = (url) => {
  try {
    return new URL(url).origin === BASE;
  } catch {
    return false;
  }
};
// A line a Run button can leave where its output goes that means it broke. A
// test's own PASS line may well contain the word "error" ("the error drops by
// four"), so only lines that START like a failure count.
const RUN_ERROR = /^\s*(error\b|FAIL\b|internal error|thread .* panicked)|DOES NOT COMPILE|entered unreachable code|failed to (load|fetch|compile) /i;

async function checkPage(browser, path, api) {
  const problems = [];
  const apiNotes = [];
  const stale = new Set();
  const links = new Set();
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => problems.push(`uncaught: ${String(e.message || e).split('\n')[0]}`));
  page.on('console', (m) => {
    if (m.type() !== 'error') return;
    const text = m.text();
    // The browser logs every failed request this way, without saying which;
    // the request itself is recorded, with its path, by the handlers below.
    if (/^Failed to load resource:/.test(text)) return;
    problems.push(`console: ${text.split('\n')[0].slice(0, 300)}`);
  });
  page.on('requestfailed', (r) => {
    const url = r.url();
    if (!sameOrigin(url)) return;
    if (isApi(url)) {
      if (!api) apiNotes.push(url);
      else problems.push(`request failed: ${new URL(url).pathname} (${r.failure()?.errorText})`);
      return;
    }
    // navigating away aborts in-flight requests; that is not the page's fault
    if (r.failure()?.errorText === 'net::ERR_ABORTED') return;
    problems.push(`request failed: ${new URL(url).pathname} (${r.failure()?.errorText})`);
  });
  page.on('response', (r) => {
    const url = r.url();
    if (!sameOrigin(url) || r.status() < 400) return;
    if (isApi(url) && !api) {
      apiNotes.push(url);
      return;
    }
    // A running server that has never been told about a post or channel the
    // build knows answers 404 for it. That is the server's database being out
    // of date (node cli.mjs import ../posts), not the page being broken -- the
    // page keeps what the build gave it. A slug with a slash in it is different:
    // that is the page asking for something that cannot exist.
    const apiPath = new URL(url).pathname;
    if (isApi(url) && r.status() === 404 && /^\/api\/(posts|channels)\/[^/%]+(\/forks|\/lineage)?$/.test(apiPath)) {
      stale.add(apiPath);
      return;
    }
    if (url === BASE + path) return; // the document itself is checked below
    problems.push(`${r.status()} for ${new URL(url).pathname}`);
  });

  let status = 0;
  try {
    const res = await page.goto(BASE + path, { waitUntil: 'load', timeout: PAGE_TIMEOUT });
    status = res ? res.status() : 0;
    if (status !== 200) problems.push(`page answered ${status}`);
    // let client-side islands hydrate and fetch what they need
    await page.waitForLoadState('networkidle', { timeout: 8000 }).catch(() => {});
    const text = await page.evaluate(() => document.body ? document.body.innerText.trim().length : 0);
    if (status === 200 && text < 40) problems.push(`renders almost nothing (${text} characters of text)`);
    // LaTeX a reader can see as source means the math renderer did not run on it
    // (KaTeX failed to load, or the text never went through it). Code blocks and
    // editors are excluded: TeX inside a program is meant to be seen.
    const rawTex = await page.evaluate(() => {
      const skip = (el) => el.closest('pre, code, textarea, .monaco-editor, .cm-editor, .cellhost, .kcell, details:not([open])');
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
      for (let n = walker.nextNode(); n; n = walker.nextNode()) {
        const el = n.parentElement;
        if (!el || skip(el) || !el.offsetParent) continue;
        const m = n.textContent.match(/\\(frac|left|right|int|sum|mathbf|mathrm|text|sqrt|theta|alpha|lambda|partial)\b[^\n]{0,40}/);
        if (m) return m[0];
      }
      return '';
    });
    if (rawTex) problems.push(`shows LaTeX source instead of math: "${rawTex}"`);
    // a post's page carries its history (components/History.jsx): at least one
    // version, with its checks, whether the build or a server supplied it
    if (status === 200 && /^\/p\/[^/]+\/$/.test(path)) {
      const versions = await page.$$eval('#history .lh-node', (n) => n.length).catch(() => 0);
      if (!versions) problems.push('the history section is missing or shows no version');
    }
    for (const href of await page.$$eval('a[href]', (as) => as.map((a) => a.href))) {
      if (!sameOrigin(href)) continue;
      const u = new URL(href);
      if (u.pathname.startsWith('/api') || /\.(xml|json|viv|wasm|js|css|png|svg|pdf|zip|txt)$/.test(u.pathname)) continue;
      links.add(u.pathname);
    }
    if (pressRun && status === 200) await pressFirstRun(page, problems);
  } catch (e) {
    problems.push(`load: ${String(e.message || e).split('\n')[0]}`);
  }
  await ctx.close();
  return { path, status, problems, apiNotes: [...new Set(apiNotes)].length, stale: [...stale], links: [...links] };
}

// Press the first visible button labeled Run, if the page has one, and wait for
// something to change where output goes. What counts as failure is an exception,
// a console error, or error text appearing in the page after the press.
async function pressFirstRun(page, problems) {
  // The first visible Run button that is not an example meant to fail (the
  // docs mark those with an "expected error" line in the same section).
  const buttons = page.getByRole('button', { name: /^\s*(▶\s*)?run\b/i });
  let run = null;
  for (let i = 0; i < Math.min(await buttons.count(), 20); i++) {
    const b = buttons.nth(i);
    if (!(await b.isVisible().catch(() => false))) continue;
    // (and a notebook of a file the interpreter cannot run, which says so and turns its Run off)
    const meantToFail = await b.evaluate((el) => !!el.closest('.doc-ex')?.querySelector('.expect-err') || !!el.closest('[data-nb-expect-error]'));
    if (!meantToFail) {
      run = b;
      break;
    }
  }
  if (!run) return;
  // a problem post read as a notebook: its tests are meant to fail until it is solved
  const failsExpected = await run.evaluate((el) => !!el.closest('[data-nb-tests-fail]'));
  const before = await page.evaluate(() => document.body.innerText);
  const errsBefore = problems.length;
  await run.click({ timeout: 5000 }).catch((e) => problems.push(`Run: could not press (${String(e.message).split('\n')[0]})`));
  const deadline = Date.now() + RUN_TIMEOUT;
  let after = before;
  while (Date.now() < deadline) {
    await page.waitForTimeout(500);
    after = await page.evaluate(() => document.body.innerText);
    const busy = await page.evaluate(() => /\b(running|compiling|loading)\b…?/i.test(document.body.innerText.slice(0, 20000)));
    if (after !== before && !busy) break;
  }
  const added = after.split('\n').filter((l) => !before.includes(l)).join('\n');
  const bad = added.split('\n').find((l) => RUN_ERROR.test(l) && !(failsExpected && /^\s*FAIL\b/.test(l)));
  if (bad) problems.push(`Run: output shows "${bad.trim().slice(0, 200)}"`);
  if (problems.length > errsBefore) problems[problems.length - 1] = problems[problems.length - 1].replace(/^(?!Run:)/, 'after Run: ');
}

// ---- pages that moved -----------------------------------------------------
// Old addresses that must answer a 301 to the new one (netlify.toml's rules,
// applied by serve-dist). The browser carries a fragment across the redirect,
// so /feed#published opens /posts#published.
const MOVED = [
  ['/feed', '/posts'],
  ['/feed/', '/posts'],
];

async function checkMoved() {
  const out = [];
  for (const [from, to] of MOVED) {
    const r = await fetch(BASE + from, { redirect: 'manual' });
    const where = r.headers.get('location');
    out.push({ path: from, status: r.status, problems: r.status === 301 && where === to ? [] : [`expected a 301 to ${to}, got ${r.status}${where ? ` to ${where}` : ''}`], links: [], apiNotes: 0, stale: [] });
  }
  return out;
}

// ---- the crawl -----------------------------------------------------------
async function main() {
  const server = await startServer();
  const api = await apiAvailable();
  const moved = await checkMoved();
  for (const r of moved) console.log(`${r.problems.length ? 'FAIL' : 'ok  '} ${r.path} redirects to ${MOVED.find(([f]) => f === r.path)[1]}${r.problems.length ? '\n       ' + r.problems.join('\n       ') : ''}`);
  const browser = await chromium.launch();
  const seeds = builtPages(DIST).filter((p) => p.startsWith(only)).sort();
  const seen = new Set(seeds);
  const queue = [...seeds];
  const linkedFrom = new Map();
  const results = [];
  let active = 0;

  console.log(`sitetest: ${seeds.length} built pages under "${only || '/'}", api ${api ? 'running' : 'not running (failed /api requests are reported separately, not as page failures)'}, Run ${pressRun ? 'pressed' : 'skipped'}`);
  await new Promise((resolve) => {
    const next = () => {
      if (!queue.length && !active) return resolve();
      while (active < CONCURRENCY && queue.length) {
        const path = queue.shift();
        active++;
        checkPage(browser, path, api).then((r) => {
          results.push(r);
          const mark = r.problems.length ? 'FAIL' : 'ok  ';
          console.log(`${mark} ${r.path}${r.problems.length ? '\n       ' + r.problems.join('\n       ') : ''}`);
          for (const l of r.links) {
            const norm = l.endsWith('/') || /\.[a-z]+$/.test(l) ? l : l + '/';
            if (!linkedFrom.has(norm)) linkedFrom.set(norm, r.path);
            if (!seen.has(norm) && norm.startsWith(only)) {
              seen.add(norm);
              queue.push(norm);
            }
          }
          active--;
          next();
        });
      }
    };
    next();
  });
  await browser.close();
  server.kill();

  results.push(...moved);
  results.sort((a, b) => a.path.localeCompare(b.path));
  for (const r of results) {
    if (r.status === 404 && linkedFrom.has(r.path)) r.problems.push(`linked from ${linkedFrom.get(r.path)}`);
  }
  const failed = results.filter((r) => r.problems.length);
  const apiPages = results.filter((r) => r.apiNotes).length;
  const staleAll = [...new Set(results.flatMap((r) => r.stale))].sort();
  if (staleAll.length) {
    console.log(`\nthe running server answered 404 for ${staleAll.length} post or channel URL(s) the build has; its database is out of date (to update it: cd server && node cli.mjs import ../posts):`);
    for (const s of staleAll) console.log(`  ${s}`);
  }
  writeFileSync(join(WWW, 'sitetest-report.json'), JSON.stringify({ base: BASE, api, results }, null, 1));
  console.log(`\nsitetest: ${results.length} pages, ${results.length - failed.length} ok, ${failed.length} failing` +
    (api ? '' : `; ${apiPages} pages made /api requests that failed because no server is running`));
  process.exit(failed.length ? 1 : 0);
}

main().catch((e) => {
  console.error('sitetest:', e);
  process.exit(2);
});
