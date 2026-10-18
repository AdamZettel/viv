// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The notebook view in a real browser.
//
//     npm run build && npm run nbtest
//
// Against the built site (served the way scripts/sitetest.mjs serves it):
//   1. Every post with prose cells: open /p/<slug>, press Run all, and check
//      that the text under each code cell is exactly what that cell prints
//      when the interpreter runs the cells in order (worked out here, in
//      node, from the same file); that each cell after the first ran alone,
//      continuing the session; and that a cell that draws has its figure
//      under it and no other cell does.
//   2. Editing a cell marks it and the cells after it stale, and not the ones
//      above; Shift+Enter runs a cell and moves the caret to the next.
//   3. Long output is capped, with a button that shows the rest.
//   4. A /nb page for a file with no prose is one cell that runs; a file the
//      interpreter cannot run says so and offers no Run.
//   5. The workbench's Notebook toggle shows the open file as cells.
//   6. No horizontal scroll at 390 px, in light and in dark.
import { readFileSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { chromium } from 'playwright-core';
import { splitNotebook, isNotebook, runUnits } from '../src/lib/notebook.js';
import { expandUses } from '../src/lib/uses.js';
import { build } from 'esbuild';

// parseOwnOutput is what the page uses to split a run into text and figures;
// it lives in a .jsx file, so it is bundled for node first
const bundled = await build({ entryPoints: [join(dirname(fileURLToPath(import.meta.url)), '../src/components/VivCell.jsx')], bundle: true, format: 'esm', platform: 'node', write: false, jsx: 'automatic', loader: { '.js': 'jsx', '.css': 'empty' }, logLevel: 'silent' });
const { parseOwnOutput } = await import(`data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString('base64')}`);

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const PORT = Number(process.env.NBTEST_PORT || 4392);
const BASE = `http://127.0.0.1:${PORT}`;
let fails = 0;
let checks = 0;
const ok = (cond, what) => {
  checks++;
  console.log(`${cond ? 'ok  ' : 'FAIL'} ${what}`);
  if (!cond) fails++;
};

// ---- what each cell should print, from the interpreter in node -------------
const { instance } = await WebAssembly.instantiate(readFileSync(join(WWW, 'public/viv/viv.wasm')), {});
const ex = instance.exports;
function runCell(prefix, cell) {
  const enc = new TextEncoder();
  const pb = enc.encode(prefix);
  const cb = enc.encode(cell);
  const ptr = ex.viv_alloc(pb.length + cb.length);
  const mem = new Uint8Array(ex.memory.buffer, ptr, pb.length + cb.length);
  mem.set(pb, 0);
  mem.set(cb, pb.length);
  const packed = ex.viv_run_cell(ptr, pb.length + cb.length, pb.length);
  return new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
}
const resolve = async (name) => {
  const [head, ...rest] = name.split('/');
  const dir = { dft: 'dft', pde: 'pde', lib: 'lib', examples: 'lang/viv/examples' }[head];
  return readFileSync(join(ROOT, dir, `${rest.join('/')}.viv`), 'utf8');
};
/** Per code cell: the text lines it prints and how many figures it draws. */
async function expected(src) {
  const units = runUnits(splitNotebook(src));
  const seen = new Set();
  const texts = [];
  for (const u of units) texts.push(await expandUses(u.text, resolve, [], seen));
  return texts.map((t, j) => {
    const { blocks, charts } = parseOwnOutput(runCell(texts.slice(0, j).join('\n'), t));
    const lines = blocks.flatMap((b) => (b.type === 'narrate' ? [`▸ ${b.s}`] : b.s.split('\n')));
    return { lines, figures: charts.length };
  });
}

// ---- serving -----------------------------------------------------------------
const server = spawn(process.execPath, [join(WWW, 'scripts', 'serve-dist.mjs')], { env: { ...process.env, PORT: String(PORT) }, stdio: 'ignore' });
for (let i = 0; i < 75; i++) {
  try { if ((await fetch(BASE + '/')).status) break; } catch { /* not up yet */ }
  await new Promise((r) => setTimeout(r, 200));
}
const browser = await chromium.launch();

async function open(path, { width = 1280, theme = 'dark' } = {}) {
  const ctx = await browser.newContext({ viewport: { width, height: 900 } });
  await ctx.addInitScript((t) => { try { localStorage.setItem('viv-theme', t); } catch { /* private window */ } }, theme);
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e.message || e)));
  await page.goto(BASE + path, { waitUntil: 'networkidle' });
  return { page, ctx, errors };
}
const runAll = async (page) => {
  await page.getByRole('button', { name: 'Run all' }).click();
  await page.waitForFunction(() => /✓ \d+ cells|failed|Stopped/.test(document.querySelector('.nb-bar-status')?.textContent || ''), null, { timeout: 60000 });
};
// the output lines under each code cell, in page order
const cellOutputs = (page) => page.$$eval('.nb-code', (secs) => secs.map((s) => ({
  status: s.dataset.status,
  continued: s.dataset.continued === 'yes',
  lines: [...s.querySelectorAll('.nb-out .nb-line')].map((l) => l.textContent.replace(/ /g, '')),
  figures: s.querySelectorAll('.nb-fig').length,
  svgs: s.querySelectorAll('.nb-fig svg, .nb-fig canvas').length,
})));

try {
  // ---- 1. every notebook post ------------------------------------------------
  const posts = readdirSync(join(ROOT, 'posts')).filter((f) => f.endsWith('.viv'));
  const notebooks = posts.filter((f) => isNotebook(splitNotebook(readFileSync(join(ROOT, 'posts', f), 'utf8'))));
  ok(notebooks.length > 0, `${notebooks.length} post(s) with prose cells`);
  for (const file of notebooks) {
    const slug = file.replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.viv$/, '');
    const src = readFileSync(join(ROOT, 'posts', file), 'utf8');
    const want = await expected(src);
    const { page, ctx, errors } = await open(`/p/${slug}/?view=notebook`);
    const prose = await page.$$eval('.nb-prose', (ps) => ps.length);
    const cellsOf = splitNotebook(src);
    const headProse = cellsOf.some((c) => c.kind === 'header' && c.prose) ? 1 : 0;
    ok(prose === cellsOf.filter((c) => c.kind === 'prose').length + headProse, `${slug}: every prose cell is rendered, and the header's prose if it has any (found ${prose})`);
    ok(await page.$$eval('.nb-prose .katex', (k) => k.length) > 0, `${slug}: the prose cells' math went through KaTeX`);
    await runAll(page);
    const got = await cellOutputs(page);
    ok(got.length === want.length, `${slug}: ${want.length} code cells on the page (found ${got.length})`);
    want.forEach((w, j) => {
      const g = got[j] || { lines: [], figures: 0, svgs: 0 };
      const same = JSON.stringify(g.lines) === JSON.stringify(w.lines);
      ok(same, `${slug}: cell ${j + 1} shows exactly its own ${w.lines.length} output line(s)`);
      if (!same) console.log(`       want ${JSON.stringify(w.lines).slice(0, 300)}\n       got  ${JSON.stringify(g.lines).slice(0, 300)}`);
      ok(g.figures === w.figures && g.svgs >= w.figures, `${slug}: cell ${j + 1} has ${w.figures} figure(s) under it, drawn (found ${g.figures}, ${g.svgs} svg or canvas)`);
      if (j > 0) ok(g.continued, `${slug}: cell ${j + 1} ran alone, continuing the session`);
    });
    const passes = got.flatMap((g) => g.lines).filter((l) => l.startsWith('PASS')).length;
    const declared = (src.match(/^\s*test\s+"/gm) || []).length;
    ok(passes === declared, `${slug}: all ${declared} tests passed, each under the cell that holds it`);
    ok(errors.length === 0, `${slug}: no uncaught errors${errors.length ? ` (${errors[0]})` : ''}`);
    await ctx.close();
  }

  // ---- 2 and 3: edits, stale marks, Shift+Enter, the output cap --------------
  if (notebooks.length) {
    const slug = notebooks[0].replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.viv$/, '');
    const { page, ctx } = await open(`/p/${slug}/?view=notebook`);
    await runAll(page);
    const tas = page.locator('.nb-code textarea');
    await tas.nth(1).click();
    await page.keyboard.press('End');
    await page.keyboard.type(' ');
    const st = await page.$$eval('.nb-code', (s) => s.map((x) => x.dataset.status));
    ok(st[0] === 'done' && st.slice(1).every((x) => x === 'stale'), `editing cell 2 marks cells 2 and on stale, not cell 1 (${st.join(' ')})`);
    await page.keyboard.press('Shift+Enter');
    await page.waitForFunction(() => document.querySelectorAll('.nb-code')[1].dataset.status === 'done', null, { timeout: 20000 });
    const focused = await page.evaluate(() => [...document.querySelectorAll('.nb-code textarea')].indexOf(document.activeElement));
    ok(focused === 2, `Shift+Enter ran cell 2 and moved to cell 3 (focus on cell ${focused + 1})`);
    const st2 = await page.$$eval('.nb-code', (s) => s.map((x) => x.dataset.status));
    ok(st2[1] === 'done' && st2[2] === 'stale', `after the rerun cell 2 is current and cell 3 is still stale (${st2.join(' ')})`);

    // a cell that prints 120 lines shows 40, and all 120 on request
    const code1 = await tas.nth(0).inputValue();
    const own = await page.locator('.nb-code').first().locator('.nb-line').count();
    await tas.nth(0).fill(`${code1}\nfor long_i to 120 {\n    print(long_i)\n}`);
    await page.getByRole('button', { name: 'Run cell 1', exact: true }).click();
    await page.waitForFunction(() => document.querySelectorAll('.nb-code')[0].dataset.status === 'done', null, { timeout: 20000 });
    const first = page.locator('.nb-code').first();
    const n1 = await first.locator('.nb-line').count();
    const more = first.getByRole('button', { name: /^Show all \d+ lines$/ });
    ok(n1 === 40 && (await more.count()) === 1, `${own + 120} output lines are capped at 40 with a Show all button (showing ${n1})`);
    await more.click();
    const n2 = await first.locator('.nb-line').count();
    ok(n2 === own + 120, `Show all shows every line (${n2} of ${own + 120})`);
    await ctx.close();
  }

  // ---- 4. /nb pages --------------------------------------------------------
  {
    const { page, ctx, errors } = await open('/nb/pde/heat_1d/');
    ok(await page.locator('.nb-code').count() === 1, '/nb/pde/heat_1d: a file with no prose is one code cell');
    await page.getByRole('button', { name: 'Run cell 1', exact: true }).click();
    await page.waitForFunction(() => document.querySelector('.nb-code').dataset.status === 'done', null, { timeout: 30000 });
    const g = (await cellOutputs(page))[0];
    ok(g.lines.some((l) => l.startsWith('PASS')) && g.svgs > 0, `/nb/pde/heat_1d: it runs, with its tests and its figure under it (${g.lines.length} lines, ${g.svgs} svg)`);
    ok(errors.length === 0, '/nb/pde/heat_1d: no uncaught errors');
    await ctx.close();
  }
  {
    const { page, ctx } = await open('/nb/examples/autodiff_forward/');
    ok(await page.locator('[data-nb-expect-error] .nb-expect').count() === 1, '/nb/examples/autodiff_forward: says the interpreter cannot run records');
    ok(await page.getByRole('button', { name: 'Run cell 1', exact: true }).isDisabled(), '/nb/examples/autodiff_forward: and its Run is off');
    await ctx.close();
  }
  {
    const { page, ctx } = await open('/nb/');
    const links = await page.$$eval('a[href^="/nb/"]', (as) => as.length);
    ok(links > 100, `/nb lists the files it can open (${links} links)`);
    await ctx.close();
  }

  // ---- 5. the workbench toggle ---------------------------------------------
  if (notebooks.length) {
    const { page, ctx, errors } = await open(`/workbench?open=posts/${notebooks[0]}`);
    await page.waitForFunction(() => document.querySelector('.ki-file')?.textContent.includes('posts/'), null, { timeout: 20000 });
    await page.getByRole('button', { name: 'Notebook', exact: true }).click();
    const want = runUnits(splitNotebook(readFileSync(join(ROOT, 'posts', notebooks[0]), 'utf8'))).length;
    ok(await page.locator('.ki-nbpane .nb-code').count() === want, `the workbench's Notebook toggle shows ${want} code cells`);
    ok(await page.locator('.ki-nbpane .nb-prose').count() > 0, 'and renders the prose between them');
    await runAll(page);
    ok((await cellOutputs(page)).every((c) => c.status === 'done'), 'and runs them');
    ok(errors.length === 0, `the workbench: no uncaught errors${errors.length ? ` (${errors[0]})` : ''}`);
    await ctx.close();
  }

  // ---- 6. phone width, both themes -------------------------------------------
  if (notebooks.length) {
    const slug = notebooks[0].replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.viv$/, '');
    for (const theme of ['light', 'dark']) {
      const { page, ctx } = await open(`/p/${slug}/?view=notebook`, { width: 390, theme });
      await runAll(page);
      const [sw, iw] = await page.evaluate(() => [document.documentElement.scrollWidth, window.innerWidth]);
      ok(sw <= iw, `390 px, ${theme}: no horizontal scroll (page ${sw} px wide)`);
      await ctx.close();
    }
  }
} finally {
  await browser.close();
  server.kill();
}
console.log(`nbtest: ${checks - fails} of ${checks} checks pass`);
process.exit(fails ? 1 : 0);
