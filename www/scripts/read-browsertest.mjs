// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The explorer's reading view (/explore/<slug>, the Read tab) in a real browser.
//
//     npm run build && npm run readbrowsertest
//     READ_SHOTS=/some/dir READ_PORT=4541 npm run readbrowsertest
//
// Against the built site, served by scripts/serve-dist.mjs:
//   1. MINPACK on Rosenbrock, dark and light, 1280 px. Read is the default
//      view, in its Core path. The path's stops are on the page in the
//      path's order, each figure it names is drawn (an SVG from the run's
//      figure records), every piece of math typesets, every hold an
//      equation cites held, every {value} of a run line was filled, and a
//      predict prompt keeps the run's answer until asked. Test citations are
//      marks in the margin, not quotes in the prose. The pass selector
//      changes the values beside the code, and every value says which pass
//      it is from. A skipped part opens to its code.
//   2. All the code: every file in order, every variable of lmder has a
//      doc, and scrolling fills in the docs of the rows on the screen
//      (each row with notes that has come into view is marked seen, and its
//      variables are documented beside it); the row at the reading line is
//      marked current and the bar names its def and step.
//   3. Reduced motion: every note is there at once, nothing animates.
//   4. 390 px, dark and light, both modes: one column, each row's notes
//      under its code, no horizontal scroll.
//   5. The notebook shows the intent's equations; the timeline still works.
//   6. Thurber, water RHF and silicon, which have no path, open in All the
//      code and render.
//   7. No page errors anywhere, and no horizontal scroll at 1280 px.
import { mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { chromium } from 'playwright-core';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const PORT = Number(process.env.READ_PORT || 4541);
const BASE = `http://127.0.0.1:${PORT}`;
const SHOTS = process.env.READ_SHOTS || join(WWW, 'read-shots');
mkdirSync(SHOTS, { recursive: true });
let fails = 0;
let checks = 0;
const ok = (cond, what) => {
  checks++;
  console.log(`${cond ? 'ok  ' : 'FAIL'} ${what}`);
  if (!cond) fails++;
};

const server = spawn(process.execPath, [join(WWW, 'scripts', 'serve-dist.mjs')], { env: { ...process.env, PORT: String(PORT), VIV_API: 'http://127.0.0.1:9' }, stdio: 'ignore' });
for (let i = 0; i < 75; i++) {
  try {
    if ((await fetch(BASE + '/')).status) break;
  } catch { /* not up yet */ }
  await new Promise((r) => setTimeout(r, 200));
}
const bundle = await (await fetch(`${BASE}/explore/minpack-rosenbrock.json`)).json();

const browser = await chromium.launch();
const errors = [];
async function open(path, { theme = 'dark', width = 1280, reduced = false, math = true } = {}) {
  const ctx = await browser.newContext({ viewport: { width, height: 900 }, colorScheme: theme, reducedMotion: reduced ? 'reduce' : 'no-preference' });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => errors.push(`${path} (${theme}, ${width}): ${e}`));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(`${path} (${theme}, ${width}): console: ${m.text()}`);
  });
  await page.goto(BASE + path);
  await page.waitForSelector('.rv-row', { timeout: 60000 });
  await page.waitForSelector('.ex-recorded:has-text("Recorded in your browser")', { timeout: 240000 });
  if (math) await page.waitForFunction(() => document.querySelectorAll('.katex').length > 0, null, { timeout: 30000 });
  return { ctx, page };
}
const noHScroll = (page) => page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth);
const settle = async (page) => {
  await page.waitForTimeout(250);
  await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
};
/** Scroll so an element's top sits at `frac` of the viewport. */
async function scrollTo(page, selector, frac = 0.3) {
  await page.evaluate(([sel, f]) => {
    const el = document.querySelector(sel);
    window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - window.innerHeight * f, behavior: 'instant' });
  }, [selector, frac]);
  await settle(page);
}
async function toMode(page, name) {
  await page.click(`.rv-modes button:has-text("${name}")`);
  await page.waitForSelector(name === 'All the code' ? '.rv-file' : '.rv-stop');
  await settle(page);
}
/** A doc's value and its pass label: the first doc with that key in the page (or under `within`). */
const docValue = (page, key, within = '') => page.evaluate(([k, w]) => {
  const d = document.querySelector(`${w} .rv-doc[data-key="${k}"]`);
  const v = d && d.querySelector('.rv-val');
  return v ? { num: v.querySelector('.rv-num').textContent, pass: v.dataset.pass } : null;
}, [key, within]);

try {
  // ---- 1. Rosenbrock, the core path, both themes ----------------------------
  for (const theme of ['dark', 'light']) {
    const { ctx, page } = await open('/explore/minpack-rosenbrock', { theme });
    ok((await page.textContent('button[role=tab][aria-selected=true]')) === 'Read', `${theme}: Read is the default view`);
    ok((await page.textContent('.rv-modes button[aria-checked=true]')).startsWith('Core path'), `${theme}: the Read view opens on its core path`);
    const want = bundle.paths[0].stops.map((s) => s.target);
    const got = await page.$$eval('.rv-stop', (s) => s.map((e) => e.dataset.stop));
    ok(JSON.stringify(got) === JSON.stringify(want), `${theme}: the core path's ${got.length} stops are on the page in the path's order`);
    ok(got.length >= 10 && got.length <= 15, `${theme}: the core path is ${got.length} stops long`);
    const figs = await page.$$eval('.rv-fig', (fs) => fs.map((f) => ({ title: f.dataset.figure, svg: !!f.querySelector('svg'), stop: f.closest('.rv-stop').dataset.stop })));
    const wantFigs = bundle.paths[0].stops.flatMap((s) => s.details.filter((d) => d.label === 'figure').map((d) => d.text));
    ok(wantFigs.length === 2 && figs.length === wantFigs.length && figs.every((f) => f.svg), `${theme}: the path's ${wantFigs.length} figures are drawn beside their stops (${figs.map((f) => `${f.stop}: ${f.svg ? 'drawn' : 'missing'}`).join(', ')})`);
    ok((await page.$$('.katex-error')).length === 0, `${theme}: KaTeX typeset every piece of math with no error (${await page.$$eval('.katex', (k) => k.length)} formulas)`);
    const marks = await page.$$eval('.rv-check', (cs) => cs.map((c) => c.className));
    ok(marks.some((c) => c.includes(' ok')) && !marks.some((c) => / (bad|none|wait)$/.test(c)), `${theme}: every hold an equation cites held in this run (${marks.filter((c) => c.includes(' ok')).length} held, ${marks.length} marks in all)`);
    const quoted = await page.evaluate(() => [...document.querySelectorAll('.rv-p, .rv-notes')].some((e) => /checked by the test|test “/.test(e.textContent)));
    ok(!quoted && marks.length > 5, `${theme}: tests are cited by marks in the margin, not quoted in the prose`);
    const tip = await page.getAttribute('.rv-check[data-kind="test"]', 'data-tip');
    ok(/^checked by the test “.+”$/.test(tip || ''), `${theme}: a test mark names its test on hover or focus (${(tip || '').slice(0, 60)}…)`);
    ok((await page.$$('.rv-missing')).length === 0 && (await page.$$('.rv-runval')).length > 8, `${theme}: every value in a run line came from the recording (${(await page.$$('.rv-runval')).length} values)`);
    // predict, then reveal
    const predicts = await page.$$('.rv-predict');
    ok(predicts.length >= 5, `${theme}: ${predicts.length} predict prompts on the core path`);
    const firstPredict = page.locator('.rv-predict:has(.rv-reveal)').first();
    ok((await firstPredict.locator('.rv-answer').count()) === 0, `${theme}: a predict prompt hides the run's answer`);
    await firstPredict.locator('.rv-reveal').click();
    ok((await firstPredict.locator('.rv-answer').count()) > 0, `${theme}: the answer shows on click`);
    // the pass selector
    await scrollTo(page, '.rv-stop[data-stop="lmder/actual"]');
    const a1 = await docValue(page, 'fnorm1', '.rv-stop[data-stop="lmder/actual"]');
    await page.screenshot({ path: join(SHOTS, `core-${theme}-actual-pass1.png`) });
    await page.selectOption('.rv-trial select', '2');
    await settle(page);
    const a2 = await docValue(page, 'fnorm1', '.rv-stop[data-stop="lmder/actual"]');
    await page.selectOption('.rv-passsel select[aria-label="Pass"]', '5');
    await settle(page);
    const a5 = await docValue(page, 'fnorm1', '.rv-stop[data-stop="lmder/actual"]');
    ok(a1 && a1.num === '48.4' && a1.pass === 'pass 1 · trial 1', `${theme}: by default the values are the first pass's rejected trial (fnorm1 = ${a1 && a1.num}, ${a1 && a1.pass})`);
    ok(a2 && a2.num === '2.097' && a2.pass === 'pass 1 · trial 2', `${theme}: trial 2 of the selector changes them (fnorm1 = ${a2 && a2.num}, ${a2 && a2.pass})`);
    ok(a5 && a5.num === '1.331' && a5.pass === 'pass 5', `${theme}: pass 5 changes them again (fnorm1 = ${a5 && a5.num}, ${a5 && a5.pass})`);
    const labels = await page.$$eval('.rv-val', (vs) => vs.map((v) => v.dataset.pass));
    ok(labels.length > 20 && labels.every((l) => /^(pass \d+( · trial \d+)?|before the first pass|after the last pass)$/.test(l)), `${theme}: every value beside the code says which pass it is from (${labels.length} values)`);
    await page.selectOption('.rv-passsel select[aria-label="Pass"]', '1');
    await settle(page);
    // screenshots at several stops
    for (const stop of ['lmder', 'lmder/lm_parameter', 'lmder/radius', 'lmpar/lower']) {
      await scrollTo(page, `.rv-stop[data-stop="${stop}"]`, 0.12);
      await page.waitForTimeout(500);
      await page.screenshot({ path: join(SHOTS, `core-${theme}-${stop.replace(/\//g, '-')}.png`) });
    }
    // a skipped part opens to its code
    const skip = page.locator('.rv-skip').first();
    const sum = await skip.locator('summary').textContent();
    await skip.locator('summary').click();
    await settle(page);
    ok(/^Skipped: /.test(sum) && (await skip.locator('.rv-row').count()) > 0, `${theme}: a skipped part opens to its code ("${sum.slice(0, 70)}")`);
    ok(await noHScroll(page), `${theme}, 1280 px, core path: no horizontal scroll`);

    // ---- 2. all the code ------------------------------------------------
    await toMode(page, 'All the code');
    const files = await page.$$eval('.rv-file', (fs) => fs.map((f) => f.dataset.file));
    ok(files[0] === 'lib/minpack/story_rosenbrock' && files.includes('lib/minpack/lmder') && files.includes('lib/minpack/enorm'), `${theme}: All the code reads the files in order: ${files.map((f) => f.split('/').pop()).join(', ')}`);
    const lmderVars = bundle.intent.lmder.vars;
    const shownVars = await page.$$eval('.rv-def[data-def="lmder"] .rv-doc', (ds) => [...new Set(ds.map((d) => d.dataset.var))]);
    const missing = lmderVars.filter((v) => !shownVars.includes(v));
    ok(lmderVars.length > 40 && missing.length === 0, `${theme}: every one of lmder's ${lmderVars.length} variables has a doc beside the code${missing.length ? ` (missing: ${missing.join(', ')})` : ''}`);
    const reuse = await page.$$eval('.rv-def[data-def="lmder"] .rv-doc.reuse', (ds) => ds.map((d) => d.dataset.key));
    ok(['wa1@qr', 'wa1@lm_parameter', 'wa1@trial', 'wa4@evaluate', 'temp@shrink'].every((k) => reuse.includes(k)), `${theme}: a reused work array is documented where its meaning changes (${reuse.length} reuse sites in lmder)`);
    ok((await page.$$('.rd-math, .rv-math')).length === 0, `${theme}: no "lines as math" section`);
    // scroll through lmder: the rows on the screen fill in with their docs
    await scrollTo(page, '.rv-def[data-def="lmder"]', 0.1);
    let filled = true;
    let current = true;
    let docsBeside = 0;
    const where = new Set();
    for (let k = 0; k < 7; k++) {
      await page.mouse.wheel(0, 700);
      await page.waitForTimeout(700);
      const st = await page.evaluate(() => {
        const h = window.innerHeight;
        const rows = [...document.querySelectorAll('.rv-def[data-def="lmder"] .rv-row.has')].filter((r) => {
          const b = r.getBoundingClientRect();
          return b.top >= 0 && b.top < h * 0.8;
        });
        const cur = document.querySelector('.rv-row.cur');
        return {
          rows: rows.length,
          unseen: rows.filter((r) => !r.classList.contains('seen')).length,
          docs: rows.reduce((s, r) => s + r.querySelectorAll('.rv-doc').length, 0),
          opaque: rows.every((r) => [...r.querySelectorAll('.rv-notes > *')].every((n) => Number(getComputedStyle(n).opacity) > 0.9)),
          cur: cur ? { def: cur.dataset.def, step: cur.dataset.step, top: cur.getBoundingClientRect().top / h, bottom: cur.getBoundingClientRect().bottom / h } : null,
          crumb: document.querySelector('.rv-crumb')?.textContent || '',
        };
      });
      if (st.rows && (st.unseen || !st.opaque)) filled = false;
      if (!st.cur || st.cur.top > 0.36 || st.cur.bottom < 0.34 || !st.crumb.includes(st.cur.def)) current = false;
      docsBeside += st.docs;
      if (st.cur) where.add(`${st.cur.def}/${st.cur.step}`);
      if (k === 1 || k === 4) await page.screenshot({ path: join(SHOTS, `all-${theme}-lmder-${k}.png`) });
    }
    ok(filled && docsBeside > 20, `${theme}: scrolling All the code fills in the docs of the rows on the screen (${docsBeside} docs beside rows in view)`);
    ok(current && where.size >= 3, `${theme}: the row at the reading line is current, and the bar names it (${[...where].join(', ')})`);
    ok(await noHScroll(page), `${theme}, 1280 px, all the code: no horizontal scroll`);

    // ---- 5. the notebook and the timeline ------------------------------
    await page.click('button[role=tab]:text("Notebook")');
    await page.waitForSelector('.ex-cell');
    const nbEq = await page.$$eval('.ex-cell .ex-eq .katex-display', (e) => e.length);
    ok(nbEq > 5, `${theme}: the notebook shows the intent with its equations (${nbEq})`);
    await page.click('button[role=tab]:text("Timeline")');
    await page.waitForSelector('.ex-grid');
    ok((await page.$$('.katex-error')).length === 0, `${theme}: the timeline renders with no KaTeX error`);
    await ctx.close();
  }

  // ---- 3. reduced motion ------------------------------------------------
  {
    const { ctx, page } = await open('/explore/minpack-rosenbrock#all', { reduced: true });
    const unseen = await page.$$eval('.rv-row', (rs) => rs.filter((r) => !r.classList.contains('seen')).length);
    const anims = await page.evaluate(() => document.getAnimations().filter((a) => a.playState === 'running').length);
    ok(unseen === 0 && anims === 0, `reduced motion: every note is there at once (${unseen} rows waiting) and nothing animates (${anims})`);
    await ctx.close();
  }

  // ---- 4. phone width -----------------------------------------------------
  for (const theme of ['dark', 'light']) {
    const { ctx, page } = await open('/explore/minpack-rosenbrock', { theme, width: 390 });
    const under = await page.$$eval('.rv-row.has', (rs) => rs.filter((r) => {
      const c = r.querySelector('.rv-src').getBoundingClientRect();
      const n = r.querySelector('.rv-notes').getBoundingClientRect();
      return n.top >= c.bottom - 1;
    }).length);
    const rows = await page.$$eval('.rv-row.has', (rs) => rs.length);
    ok(rows > 20 && under === rows, `390 px, ${theme}: one column, each row's notes under its code (${under} of ${rows})`);
    for (const stop of ['lmder/trial', 'lmder/radius', 'lmpar/lower']) {
      await scrollTo(page, `.rv-stop[data-stop="${stop}"]`, 0.15);
      await page.waitForTimeout(500);
      await page.screenshot({ path: join(SHOTS, `core-390-${theme}-${stop.replace(/\//g, '-')}.png`) });
    }
    ok(await noHScroll(page), `390 px, ${theme}, core path: no horizontal scroll`);
    // a display wider than the column is set smaller, and past that scrolls with a fade: never cut off silently
    const cut = await page.$$eval('.rv-eqbox', (bs) => bs.filter((b) => b.scrollWidth > b.clientWidth + 1 && !b.classList.contains('wide')).length);
    const fitted = await page.$$eval('.rv-eqbox > .rv-tex', (es) => es.filter((e) => e.style.fontSize).length);
    ok(cut === 0 && fitted > 0, `390 px, ${theme}: wide equations are set smaller or marked as scrolling (${fitted} set smaller, ${cut} cut off)`);
    await toMode(page, 'All the code');
    await scrollTo(page, '.rv-def[data-def="lmder"]', 0.1);
    await page.mouse.wheel(0, 1600);
    await page.waitForTimeout(600);
    await page.screenshot({ path: join(SHOTS, `all-390-${theme}-lmder.png`) });
    ok(await noHScroll(page), `390 px, ${theme}, all the code: no horizontal scroll`);
    ok((await page.$$('.katex-error')).length === 0, `390 px, ${theme}: every formula typeset`);
    await ctx.close();
  }

  // ---- 6. the other programs ------------------------------------------
  for (const slug of ['minpack-thurber', 'water-rhf', 'silicon-dense']) {
    // their first screens need not carry math
    const { ctx, page } = await open(`/explore/${slug}`, { math: false });
    const mode = await page.textContent('.rv-modes button[aria-checked=true]');
    const n = await page.$$eval('.rv-row', (r) => r.length);
    ok(mode.startsWith('All the code') && n > 10 && (await page.$$('.katex-error')).length === 0, `${slug}: no path, so it opens in All the code (${n} rows, every formula typeset)`);
    ok((await page.$$('.rv-missing')).length === 0, `${slug}: no value left unfilled`);
    await page.mouse.wheel(0, 1400);
    await page.waitForTimeout(600);
    await page.screenshot({ path: join(SHOTS, `${slug}-read.png`) });
    ok(await noHScroll(page), `${slug}: no horizontal scroll`);
    await ctx.close();
  }
} catch (e) {
  ok(false, `the test ran to the end: ${e.message}`);
}

ok(errors.length === 0, `no page errors${errors.length ? `:\n  ${errors.slice(0, 12).join('\n  ')}` : ''}`);
await browser.close();
server.kill();
console.log(`\nreadbrowsertest: ${checks - fails} of ${checks} checks passed; screenshots in ${SHOTS}`);
process.exit(fails ? 1 : 0);
