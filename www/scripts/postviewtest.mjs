// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// A post page's four views in a real browser (components/PostViews.jsx).
//
//     npm run build && npm run postviewtest
//     POSTVIEW_SHOTS=/some/dir npm run postviewtest     also save screenshots
//
// For a notebook post, a post with no prose cells, a ladder and a problem:
//   1. every view renders, from the switch and from ?view=;
//   2. Article shows no code until a fold is opened, lists every test by name,
//      and shows each figure the build's run drew;
//   3. Notebook shows one editable cell per code cell;
//   4. Code shows the program and its tests, and none of the prose;
//   5. Source is the file, byte for byte;
//   6. the switch works from the keyboard, the choice is remembered, and
//      #view= opens a view too;
//   7. Run all in Article runs every cell and the outputs land in the article;
//   8. light and dark, 1280 and 390 px: no horizontal scroll, no page errors;
//   9. the Posts list's cards keep code behind Show code, and it opens in the card.
import { readFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { chromium } from 'playwright-core';
import { splitNotebook } from '../src/lib/notebook.js';
import { testNames } from '../src/lib/postViews.js';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const PORT = Number(process.env.POSTVIEW_PORT || 4583);
const BASE = `http://127.0.0.1:${PORT}`;
const SHOTS = process.env.POSTVIEW_SHOTS || '';
if (SHOTS) mkdirSync(SHOTS, { recursive: true });
let fails = 0;
let checks = 0;
const ok = (cond, what) => {
  checks++;
  console.log(`${cond ? 'ok  ' : 'FAIL'} ${what}`);
  if (!cond) fails++;
};

const POSTS = [
  { what: 'notebook', file: '2026-09-25-fitting-electron-mobility-with-minpack.viv', explain: true },
  { what: 'plain', file: '2026-09-21-gibbs-overshoot.viv' },
  { what: 'ladder', file: '2026-09-22-the-derivative-one-rung-at-a-time.viv' },
  { what: 'problem', file: '2026-09-22-solve-the-quadratic.viv' },
];
const slugOf = (f) => f.replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.viv$/, '');

const server = spawn(process.execPath, [join(WWW, 'scripts', 'serve-dist.mjs')], { env: { ...process.env, PORT: String(PORT), VIV_API: 'http://127.0.0.1:9' }, stdio: 'ignore' });
for (let i = 0; i < 75; i++) {
  try { if ((await fetch(BASE + '/')).status) break; } catch { /* not up yet */ }
  await new Promise((r) => setTimeout(r, 200));
}
const browser = await chromium.launch();

async function open(path, { width = 1280, theme = 'light', ctx = null } = {}) {
  const c = ctx || await browser.newContext({ viewport: { width, height: 900 } });
  if (!ctx) await c.addInitScript((t) => { try { localStorage.setItem('viv-theme', t); } catch { /* private */ } }, theme);
  const page = await c.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e.message || e)));
  await page.goto(BASE + path, { waitUntil: 'networkidle' });
  return { page, ctx: c, errors };
}
const viewOf = (page) => page.$eval('.pv', (el) => el.dataset.view);
const waitView = (page, v) => page.waitForFunction((x) => document.querySelector('.pv')?.dataset.view === x, v, { timeout: 10000 });
const noHScroll = (page) => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth);

try {
  for (const P of POSTS) {
    const slug = slugOf(P.file);
    const src = readFileSync(join(ROOT, 'posts', P.file), 'utf8');
    const cells = splitNotebook(src);
    const codeCells = cells.filter((c) => c.kind === 'code' && !c.empty);
    const tests = testNames(src);
    const { page, ctx, errors } = await open(`/p/${slug}/`);

    // ---- Article (the default) ----
    ok(await viewOf(page) === 'article', `${P.what} ${slug}: opens on Article`);
    const art = await page.evaluate(() => ({
      pre: document.querySelectorAll('.pv-panel pre, .pv-panel textarea').length,
      folds: document.querySelectorAll('.pv-panel .pv-fold').length,
      tests: [...document.querySelectorAll('.pv-panel .pv-t')].map((li) => li.textContent),
      katex: document.querySelectorAll('.pv-panel .katex').length,
      charts: document.querySelectorAll('.pv-panel .pv-fig svg, .pv-panel .pv-fig canvas').length,
    }));
    ok(art.pre === 0, `${P.what}: Article shows no raw code before a fold is opened (${art.pre} pre/textarea)`);
    ok(art.folds >= codeCells.length, `${P.what}: every code cell is folded to a line (${art.folds} folds, ${codeCells.length} cells)`);
    ok(tests.every((t) => art.tests.some((x) => x.includes(t))), `${P.what}: Article lists all ${tests.length} tests by name (found ${art.tests.length})`);
    ok(art.katex > 0, `${P.what}: the prose's math is typeset`);
    if (P.what !== 'problem') ok(art.charts > 0, `${P.what}: the build's figures are in the article (${art.charts})`);
    if (P.what === 'problem') {
      ok(await page.locator('.pv-task h2', { hasText: 'Your task' }).count() === 1, 'problem: Article leads with Your task');
      const failing = await page.locator('.pv-task .pv-t-fail').count();
      ok(failing === tests.length, `problem: its ${tests.length} failing tests are marked in the task (${failing})`);
    }
    if (P.what === 'ladder') {
      const rungs = (src.match(/^rung\s+"/gm) || []).length;
      ok(await page.locator('.pv-rung').count() === rungs, `ladder: Article keeps its ${rungs} rungs`);
    }
    // a fold opens in place, to the cell's own code
    const btn = page.locator('.pv-fold-btn').last();
    await btn.click();
    const shown = await page.locator('.pv-fold.open pre').first().textContent();
    ok(shown && src.includes(shown.trim().split('\n')[0]) && (await btn.getAttribute('aria-expanded')) === 'true', `${P.what}: a fold opens in place to code from the file`);
    ok(await page.locator('.li-block a.li-btn[href*="/walk"]').count() === 1, `${P.what}: Look inside has an Atlas button to the program's map`);
    if (P.explain) ok(await page.locator('.li-block a.li-btn[href^="/explore/"]:not([href*="/walk"])').count() === 1, `${P.what}: Look inside has an Explore button to the explorer`);
    else ok(await page.locator('.li-block .li-norun').count() === 1, `${P.what}: Look inside says there is no recorded run yet`);

    // ---- Notebook ----
    await page.getByRole('tab', { name: 'Notebook' }).click();
    await waitView(page, 'notebook');
    ok(await page.locator('.pv-panel .nb-code textarea').count() === codeCells.length, `${P.what}: Notebook has ${codeCells.length} editable code cell(s)`);
    ok(page.url().includes('view=notebook'), `${P.what}: the address says ?view=notebook`);

    // ---- Code ----
    await page.getByRole('tab', { name: 'Code' }).click();
    await waitView(page, 'code');
    const code = await page.evaluate(() => ({
      text: document.querySelector('.pv-file')?.innerText || '',
      paras: document.querySelectorAll('.pv-panel .pv-file p, .pv-panel .nb-prose').length,
    }));
    ok(code.paras === 0, `${P.what}: Code has no prose paragraphs`);
    const proseLines = cells.filter((c) => c.kind === 'prose' || c.kind === 'header').flatMap((c) => c.body.split('\n').slice(2)).filter((l) => l.trim().length > 30);
    ok(proseLines.every((l) => !code.text.includes(l.trim())), `${P.what}: none of the prose is shown in Code (${proseLines.length} lines checked)`);
    ok(tests.every((t) => code.text.includes(`test "${t}"`)), `${P.what}: Code shows every test as code`);
    const defs = [...src.matchAll(/^\s*def\s+(\w+)/gm)].map((m) => m[1]);
    ok(defs.every((d) => code.text.includes(`def ${d}`)), `${P.what}: Code shows every def (${defs.length})`);

    // ---- Source ----
    await page.getByRole('tab', { name: 'Source' }).click();
    await waitView(page, 'source');
    const text = await page.$eval('pre.pv-source', (el) => el.textContent);
    ok(text === src, `${P.what}: Source is the file byte for byte (${text.length} of ${src.length} chars)`);
    ok(await page.locator(`.pv-sourceview a[download][href="/programs/posts/${P.file}"]`).count() === 1, `${P.what}: Source has a download link`);

    // ---- keyboard, memory, links ----
    await page.getByRole('tab', { name: 'Source' }).focus();
    await page.keyboard.press('Home');
    await waitView(page, 'article');
    ok(await page.evaluate(() => document.activeElement?.textContent) === 'Article', `${P.what}: Home on the switch moves to Article, focus follows`);
    await page.keyboard.press('ArrowRight');
    await waitView(page, 'notebook');
    ok(true, `${P.what}: ArrowRight moves to Notebook`);
    await page.keyboard.press('ArrowRight');
    await waitView(page, 'code');
    await page.goto(`${BASE}/p/${slug}/`, { waitUntil: 'networkidle' });
    await waitView(page, 'code').catch(() => {});
    ok(await viewOf(page) === 'code', `${P.what}: the chosen view is remembered in this browser`);
    await page.goto(`${BASE}/p/${slug}/#view=source`, { waitUntil: 'networkidle' });
    await waitView(page, 'source').catch(() => {});
    ok(await viewOf(page) === 'source', `${P.what}: #view=source opens Source`);
    for (const v of ['article', 'notebook', 'code', 'source']) {
      await page.goto(`${BASE}/p/${slug}/?view=${v}`, { waitUntil: 'networkidle' });
      await waitView(page, v).catch(() => {});
      ok(await viewOf(page) === v, `${P.what}: ?view=${v} opens ${v}`);
    }
    ok(errors.length === 0, `${P.what}: no page errors${errors.length ? ` (${errors[0]})` : ''}`);
    await ctx.close();
  }

  // ---- Run all in Article ----
  {
    const P = POSTS[0];
    const src = readFileSync(join(ROOT, 'posts', P.file), 'utf8');
    const { page, ctx, errors } = await open(`/p/${slugOf(P.file)}/?view=article`);
    await page.getByRole('button', { name: 'Run all' }).click();
    await page.waitForFunction(() => /✓ \d+ cells|failed|Stopped/.test(document.querySelector('.nb-bar-status')?.textContent || ''), null, { timeout: 90000 });
    const r = await page.evaluate(() => ({
      status: document.querySelector('.nb-bar-status').textContent,
      prov: document.querySelector('.pv-provenance')?.textContent || '',
      pass: document.querySelectorAll('.pv-panel .pv-t-pass').length,
      charts: document.querySelectorAll('.pv-panel .pv-fig svg, .pv-panel .pv-fig canvas').length,
      pre: document.querySelectorAll('.pv-panel pre, .pv-panel textarea').length,
    }));
    const n = testNames(src).length;
    ok(/^✓/.test(r.status), `Run all in Article: every cell ran (${r.status})`);
    ok(/your run/.test(r.prov), 'Run all in Article: the page says the outputs are from this run');
    ok(r.pass === n, `Run all in Article: ${n} tests pass, marked in the article (${r.pass})`);
    ok(r.charts > 0 && r.pre === 0, `Run all in Article: figures redrawn in place, code still folded (${r.charts} figures)`);
    ok(errors.length === 0, 'Run all in Article: no page errors');
    await ctx.close();
  }

  // ---- themes and phone width ----
  for (const P of POSTS) {
    for (const theme of ['light', 'dark']) {
      for (const width of [1280, 390]) {
        const { page, ctx, errors } = await open(`/p/${slugOf(P.file)}/`, { width, theme });
        const bad = [];
        for (const v of ['article', 'notebook', 'code', 'source']) {
          await page.getByRole('tab', { name: v[0].toUpperCase() + v.slice(1) }).click();
          await waitView(page, v);
          if (v === 'article') for (const b of await page.locator('.pv-fold-btn').all()) await b.click();
          if (!(await noHScroll(page))) bad.push(v);
          if (SHOTS && P.what !== 'plain') await page.screenshot({ path: join(SHOTS, `${P.what}-${v}-${theme}-${width}.png`), fullPage: width === 390 ? false : v !== 'source' });
        }
        const bg = await page.evaluate(() => getComputedStyle(document.querySelector('.pv-switch')).backgroundColor);
        ok(bad.length === 0, `${P.what}, ${theme}, ${width} px: no horizontal scroll in any view${bad.length ? ` (${bad.join(', ')})` : ''}`);
        ok(errors.length === 0, `${P.what}, ${theme}, ${width} px: no page errors${errors.length ? ` (${errors[0]})` : ''}`);
        ok(!!bg, `${P.what}, ${theme}: the switch takes the theme's colors (${bg})`);
        await ctx.close();
      }
    }
  }

  // ---- the Posts list's cards ----
  {
    const { page, ctx, errors } = await open('/posts', { width: 390 });
    // the cards in the lane on show; the rest of the list is stubs until it scrolls near
    const cards = page.locator('.feed-item:not([hidden]) article.fp');
    const n = await cards.count();
    ok(n > 0 && await page.locator('article.fp .fp-code').count() === 0, `/posts: no card shows its code until asked (${n} cards)`);
    ok(await page.locator('.feed-item:not([hidden]) article.fp .fp-tools a.fp-read').count() === n, '/posts: every card has a Read link');
    const first = cards.first();
    await first.getByRole('button', { name: 'Show code' }).click();
    await page.waitForSelector('article.fp .fp-code');
    ok(await first.locator('.fp-code textarea').count() === 1, '/posts: Show code opens the program inside the card');
    ok(await first.getByRole('button', { name: 'Hide code' }).count() === 1, '/posts: and the toggle then reads Hide code');
    ok(await noHScroll(page), '/posts, 390 px: no horizontal scroll with the code open');
    if (SHOTS) await first.screenshot({ path: join(SHOTS, 'card-code-open-390.png') });
    ok(errors.length === 0, '/posts: no page errors');
    await ctx.close();
  }
} finally {
  await browser.close();
  server.kill();
}
console.log(`postviewtest: ${checks - fails} of ${checks} checks pass`);
process.exit(fails ? 1 : 0);
