// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The walk through a run (/explore/walk/<slug>) in a real browser.
//
//     npm run build && npm run zoomwalkbrowsertest
//     ZW_BASE=http://127.0.0.1:4321 npm run zoomwalkbrowsertest   (against a running dev server)
//     ZW_SHOTS=/some/dir ...                                       (keep screenshots there)
//
// On MINPACK on Rosenbrock, in dark and in light:
//   1. the run is recorded in the tab; the graph is flat (no module regions)
//      and each box shows a name and a signature; the code pane is beside it;
//   2. stepping into the first call lights that call's edge and the callee's
//      box, and the flight between two defs zooms out mid-way (van Wijk-Nuij);
//   3. after every step the line that runs is marked in the code pane, inside
//      the def block of the box that is lit, and scrolled into view;
//   4. the breadcrumb is the call stack: it grows on a call, shrinks on step
//      out, and its last entry is the lit box; picking an earlier entry marks
//      the waiting call line and flies the graph to that def;
//   5. clicking a def box shows its code; clicking a call in the code flies
//      the graph to the callee;
//   6. Space steps one frame, Shift+Space steps back, nothing plays on its own; dragging stops following.
// With reduced motion a move is a cut, for the camera and the code pane alike.
// At 390 px wide, in both themes: the code pane is below the graph at reading
// size, the line is in view, and no horizontal page scroll. No page errors.
import { mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { chromium } from 'playwright-core';
import { interpolateZoom } from '../src/lib/zoomwalk.js';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const PORT = Number(process.env.ZW_PORT || 4399);
const BASE = process.env.ZW_BASE || `http://127.0.0.1:${PORT}`;
const SHOTS = process.env.ZW_SHOTS || join(WWW, 'explore-shots');
mkdirSync(SHOTS, { recursive: true });
let fails = 0;
let checks = 0;
const ok = (cond, what) => {
  checks++;
  console.log(`${cond ? 'ok  ' : 'FAIL'} ${what}`);
  if (!cond) fails++;
};

const server = process.env.ZW_BASE ? null : spawn(process.execPath, [join(WWW, 'scripts', 'serve-dist.mjs')], { env: { ...process.env, PORT: String(PORT), VIV_API: 'http://127.0.0.1:9' }, stdio: 'ignore' });
for (let i = 0; i < 75; i++) {
  try {
    if ((await fetch(BASE + '/')).status) break;
  } catch { /* not up yet */ }
  await new Promise((r) => setTimeout(r, 200));
}

const browser = await chromium.launch();
const errors = [];
async function open(path, { theme = 'dark', width = 1400, reducedMotion = 'no-preference' } = {}) {
  const ctx = await browser.newContext({ viewport: { width, height: 1000 }, colorScheme: theme, reducedMotion });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => errors.push(`${path} (${theme}, ${width}): ${e}`));
  await page.goto(BASE + path);
  await page.waitForSelector('.zw-def', { timeout: 60000 });
  await page.waitForTimeout(300);
  return { page, ctx };
}
// the camera's scale, from the canvas group's transform
const camera = (page) =>
  page.evaluate(() => {
    const t = document.querySelector('.zw-canvas > g').getAttribute('transform') || '';
    const [tx, ty] = (t.match(/translate\(([^)]+)\)/)?.[1] || '0 0').split(/[ ,]+/).map(Number);
    const k = Number(t.match(/scale\(([^)]+)\)/)?.[1]);
    const r = document.querySelector('.zw-canvas').getBoundingClientRect();
    return [(r.width / 2 - tx) / k, (r.height / 2 - ty) / k, r.width / k, r.width];
  });
const scale = (page) => page.evaluate(() => Number((document.querySelector('.zw-canvas > g').getAttribute('transform') || '').match(/scale\(([^)]+)\)/)?.[1]));
const state = (page) =>
  page.evaluate(() => {
    const body = document.querySelector('.zw-codebody');
    const cur = document.querySelector('.zw-ln.cur');
    const br = body.getBoundingClientRect();
    const lr = cur ? cur.getBoundingClientRect() : null;
    return {
      pos: Number(document.querySelector('.zw-scrub input').value),
      stack: [...document.querySelectorAll('.zw-stack li')].map((l) => l.textContent.trim()),
      here: document.querySelector('.zw-def.here')?.getAttribute('data-def') || null,
      hereName: document.querySelector('.zw-def.here .zw-dname')?.firstChild?.textContent || null,
      focus: document.querySelector('.zw-def.focus')?.getAttribute('data-def') || null,
      edge: !!document.querySelector('.zw-edge.on'),
      line: cur ? Number(cur.getAttribute('data-line')) : null,
      lineBlock: cur ? cur.closest('.zw-defblk')?.getAttribute('data-def') : null,
      lineVisible: !!lr && lr.top >= br.top - 1 && lr.bottom <= br.bottom + 1,
      hereBlock: document.querySelector('.zw-defblk.here')?.getAttribute('data-def') || null,
      head: document.querySelector('.zw-head')?.textContent || '',
      follow: document.querySelector('.zw-zoom button[aria-pressed]')?.getAttribute('aria-pressed'),
    };
  });
const press = async (page, key, n = 1) => {
  for (let k = 0; k < n; k++) await page.keyboard.press(key);
};

const PATH = '/explore/walk/minpack-rosenbrock';
const inStack = (s) => s.stack[s.stack.length - 1] === (s.here === '<main>' ? 'top level' : s.hereName);
for (const theme of ['dark', 'light']) {
  const { page, ctx } = await open(PATH, { theme });
  let s = await state(page);
  const flat = await page.evaluate(() => ({
    regions: document.querySelectorAll('.zw-region').length,
    boxes: document.querySelectorAll('.zw-def').length,
    sigs: document.querySelectorAll('.zw-def .zw-sig').length,
    lines: document.querySelectorAll('.zw-canvas .zw-line').length,
    code: document.querySelectorAll('.zw-codebody .zw-ln').length,
    side: document.querySelector('.zw-code').getBoundingClientRect().left > document.querySelector('.zw-stage').getBoundingClientRect().right - 1,
  }));
  ok(s.pos === 0 && flat.regions === 0 && flat.lines === 0 && flat.sigs >= flat.boxes - 1 && flat.code > 50 && flat.side,
    `${theme}: a flat graph of ${flat.boxes} boxes with signatures, no regions or source lines in it, and the code pane beside it (${flat.code} lines)`);
  await page.screenshot({ path: join(SHOTS, `zw-${theme}-overview.png`) });

  // into the first call
  let guard = 0;
  while ((await state(page)).stack.length < 2 && guard++ < 60) {
    await press(page, 'ArrowRight');
    await page.waitForTimeout(40);
  }
  s = await state(page);
  ok(s.edge && s.here && s.stack.length >= 2 && inStack(s), `${theme}: at a call the edge and the callee's box are lit, and the breadcrumb ends there (${s.stack.join(' > ')})`);
  // a long hop, from lmder to qrfac: step back one and fly into the call again, watching the scale
  guard = 0;
  while (!(await state(page)).stack.join('>').endsWith('lmder>qrfac') && guard++ < 120) await press(page, 'ArrowRight');
  await press(page, 'ArrowLeft');
  await page.waitForTimeout(1800);
  const kFrom = await scale(page);
  const from = await camera(page);
  await press(page, 'ArrowRight');
  const ks = [];
  for (let t = 0; t < 30; t++) {
    ks.push(await scale(page));
    await page.waitForTimeout(40);
  }
  await page.waitForTimeout(1600);
  const kTo = await scale(page);
  const to = await camera(page);
  const path = interpolateZoom(from.slice(0, 3), to.slice(0, 3));
  let wMax = 0;
  for (let t = 0; t <= 1; t += 0.01) wMax = Math.max(wMax, path(t)[2]);
  const kMin = from[3] / wMax;
  ok(Math.abs(Math.min(...ks) - kMin) <= 0.1 * kMin && kMin < Math.min(kFrom, kTo), `${theme}: the flight lmder -> qrfac follows the van Wijk-Nuij path, out and back in (scale ${kFrom.toFixed(3)} -> min ${Math.min(...ks).toFixed(3)}, the path's ${kMin.toFixed(3)} -> ${kTo.toFixed(3)})`);
  // the box is readable where the camera stops: its name at least 12 px tall
  const namePx = await page.evaluate(() => document.querySelector('.zw-def.here .zw-dname').getBoundingClientRect().height);
  ok(namePx >= 12, `${theme}: the lit box's name reads at ${namePx.toFixed(1)} px`);
  await page.screenshot({ path: join(SHOTS, `zw-${theme}-call.png`) });

  // every step: the line is marked in the lit def's block, and in view
  let marked = 0;
  let bad = [];
  let depthOk = true;
  for (let k = 0; k < 40; k++) {
    const before = await state(page);
    await press(page, 'ArrowRight');
    await page.waitForTimeout(450);
    const t = await state(page);
    if (t.stack.length > before.stack.length) depthOk &&= t.stack.length === before.stack.length + 1 && t.stack.slice(0, -1).join('>') === before.stack.join('>');
    depthOk &&= inStack(t);
    if (t.line === null) continue;
    marked++;
    if (!t.lineVisible || t.lineBlock !== t.here) bad.push(`${t.pos}:${t.line}${t.lineVisible ? '' : ' hidden'}${t.lineBlock !== t.here ? ` in ${t.lineBlock} not ${t.here}` : ''}`);
  }
  ok(marked > 10 && bad.length === 0, `${theme}: after each of 40 steps the running line is marked in the lit def and in view (${marked} lines${bad.length ? `; bad ${bad.slice(0, 4).join(', ')}` : ''})`);
  ok(depthOk, `${theme}: the breadcrumb grows by the callee on each call and always ends at the lit box`);
  await page.screenshot({ path: join(SHOTS, `zw-${theme}-line.png`) });

  // step out and step over
  guard = 0;
  while ((await state(page)).stack.length < 3 && guard++ < 100) await press(page, 'ArrowRight');
  s = await state(page);
  const depth = s.stack.length;
  await press(page, 'ArrowUp');
  await page.waitForTimeout(500);
  const out = await state(page);
  ok(out.stack.length === depth - 1 && out.stack.join('>') === s.stack.slice(0, -1).join('>') && inStack(out), `${theme}: step out returns to the caller, and the breadcrumb drops the callee (${s.stack.join(' > ')} -> ${out.stack.join(' > ')})`);
  let deeper = false;
  for (let k = 0; k < 12; k++) {
    const d = (await state(page)).stack.length;
    await press(page, 'ArrowDown');
    deeper ||= (await state(page)).stack.length > d;
  }
  ok(!deeper, `${theme}: step over never goes into a call`);

  // pick an earlier entry of the breadcrumb
  guard = 0;
  while ((await state(page)).stack.length < 3 && guard++ < 100) await press(page, 'ArrowRight');
  s = await state(page);
  await page.click(`.zw-stack li:nth-child(${s.stack.length - 1}) button`);
  await page.waitForTimeout(900);
  const picked = await page.evaluate(() => {
    const ln = document.querySelector('.zw-ln.picked');
    const br = document.querySelector('.zw-codebody').getBoundingClientRect();
    const r = ln && ln.getBoundingClientRect();
    return { text: ln?.textContent || null, block: ln?.closest('.zw-defblk')?.getAttribute('data-def'), visible: !!r && r.top >= br.top - 1 && r.bottom <= br.bottom + 1, focus: document.querySelector('.zw-def.focus')?.getAttribute('data-def') };
  });
  const after = await state(page);
  ok(picked.text && picked.visible && picked.block === picked.focus && after.pos === s.pos, `${theme}: picking ${s.stack[s.stack.length - 2]} in the breadcrumb marks its waiting call, in view, and flies there without moving the run (${(picked.text || '').trim().slice(0, 50)})`);

  // click a def box in the graph
  await press(page, 'ArrowRight');
  await overviewClick(page);
  const target = await page.evaluate(() => [...document.querySelectorAll('.zw-def:not(.here)')].map((g) => g.getAttribute('data-def')).find((d) => d !== '<main>'));
  const gb = await page.locator(`.zw-def[data-def="${target}"] .zw-box`).boundingBox();
  await page.mouse.click(gb.x + gb.width / 2, gb.y + gb.height / 2);
  await page.waitForTimeout(900);
  s = await state(page);
  const headIn = await page.evaluate((d) => {
    const blk = document.querySelector(`.zw-defblk[data-def="${d}"]`);
    const br = document.querySelector('.zw-codebody').getBoundingClientRect();
    const r = blk && blk.getBoundingClientRect();
    return !!r && r.top >= br.top - 2 && r.top <= br.bottom;
  }, target);
  ok(s.focus === target && s.hereBlock === target && headIn, `${theme}: clicking ${target} in the graph shows its code`);

  // click a call in the code
  const callee = await page.evaluate(() => {
    const br = document.querySelector('.zw-codebody').getBoundingClientRect();
    const b = [...document.querySelectorAll('.zw-codebody .zw-callee')].find((x) => {
      const r = x.getBoundingClientRect();
      return r.top >= br.top && r.bottom <= br.bottom && r.left >= br.left && r.right <= br.right;
    });
    if (!b) return null;
    b.setAttribute('data-test-click', '1');
    return b.getAttribute('data-callee');
  });
  if (callee) {
    await page.click('[data-test-click]');
    await page.waitForTimeout(900);
    s = await state(page);
    ok(s.focus === callee, `${theme}: clicking the call of ${callee} in the code flies the graph to it`);
  } else ok(false, `${theme}: a call in view in the code to click`);

  // manual stepping only: Space steps into, Shift+Space steps back, nothing moves on its own
  await press(page, 'Home');
  await press(page, 'ArrowRight', 4);
  const before = (await state(page)).pos;
  await press(page, ' ');
  const afterSpace = (await state(page)).pos;
  await page.keyboard.press('Shift+Space');
  await page.waitForTimeout(300);
  const afterBack = (await state(page)).pos;
  await page.waitForTimeout(1500);
  ok(afterSpace === before + 1 && afterBack === before && (await state(page)).pos === before && (await page.locator('.zw-play').count()) === 0,
     `${theme}: Space steps one frame (${before} -> ${afterSpace}), Shift+Space steps back, and the walk never moves on its own`);

  // free pan
  const box = await page.locator('.zw-canvas').boundingBox();
  await page.mouse.move(box.x + 30, box.y + box.height - 30);
  await page.mouse.down();
  await page.mouse.move(box.x + 150, box.y + box.height - 70, { steps: 6 });
  await page.mouse.up();
  ok((await state(page)).follow === 'false', `${theme}: dragging the graph stops following`);
  await ctx.close();
}

async function overviewClick(page) {
  await page.click('.zw-zoom button:has-text("Whole graph")');
  await page.waitForTimeout(1600);
}

// reduced motion: cuts, for the camera and the code pane
{
  const { page, ctx } = await open(PATH, { reducedMotion: 'reduce' });
  let guard = 0;
  while ((await state(page)).stack.length < 3 && guard++ < 100) await press(page, 'ArrowRight');
  await press(page, 'ArrowRight', 3);
  await page.waitForTimeout(60);
  const a = await scale(page);
  const sa = await state(page);
  await page.waitForTimeout(1500);
  const b = await scale(page);
  ok(Math.abs(a - b) < 1e-9, `reduced motion: the camera cuts, no flight (${a.toFixed(4)} then ${b.toFixed(4)})`);
  ok(sa.line === null || sa.lineVisible, 'reduced motion: the code pane is at the line at once, no smooth scroll');
  await ctx.close();
}

// phone width
for (const theme of ['dark', 'light']) {
  const { page, ctx } = await open(PATH, { theme, width: 390 });
  let g = 0;
  while (g++ < 120) {
    const t = await state(page);
    if (t.line !== null && t.stack.length >= 3) break;
    await press(page, 'ArrowRight');
  }
  await page.waitForTimeout(1200);
  const st = await state(page);
  const geo = await page.evaluate(() => ({
    below: document.querySelector('.zw-code').getBoundingClientRect().top >= document.querySelector('.zw-stage').getBoundingClientRect().bottom - 1,
    fs: parseFloat(getComputedStyle(document.querySelector('.zw-codebody')).fontSize),
    sw: document.documentElement.scrollWidth,
  }));
  ok(geo.below && geo.fs >= 12 && st.lineVisible && st.lineBlock === st.here, `${theme}, 390 px: the code pane is below the graph at ${geo.fs} px, with the running line in view`);
  ok(geo.sw <= 390, `${theme}, 390 px: no horizontal scroll (${geo.sw})`);
  await page.screenshot({ path: join(SHOTS, `zw-${theme}-390.png`), fullPage: false });
  await ctx.close();
}

ok(errors.length === 0, `no page errors${errors.length ? `: ${errors.join('; ')}` : ''}`);
await browser.close();
if (server) server.kill();
console.log(`${checks - fails}/${checks} checks passed; screenshots in ${SHOTS}`);
process.exit(fails ? 1 : 0);
