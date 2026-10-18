// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The multi-paths of a program in the
// explorer's picture, in a real browser.
//
//     npm run build && npm run multipathbrowsertest
//     MP_SHOTS=/some/dir MP_PORT=4519 npm run multipathbrowsertest
//
// On the water SCF, in light and dark, at 1400 by 1000:
//   - the first view draws each multi-path whose host is drawn as one closed
//     box, `multi-path ×n`, and the panel lists the program's multi-paths with
//     how many stages have two agreeing, independent implementations;
//   - picking the ERI stage from that list opens the path to its host and
//     brings its box into view; the panel shows its members by
//     implementation, the contract and the tests;
//   - a click on the closed box opens it: a fork and a join (diamonds), a
//     path per member, the default's solid and the member that ran as its own
//     def box on its path; a click on the fork closes it again;
//   - no page errors.
// Screenshots go to MP_SHOTS (default www/graph-shots).
import { mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { chromium } from 'playwright-core';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const PORT = Number(process.env.MP_PORT || 4519);
const BASE = `http://127.0.0.1:${PORT}`;
const SHOTS = process.env.MP_SHOTS || join(WWW, 'graph-shots');
mkdirSync(SHOTS, { recursive: true });
let fails = 0;
const ok = (cond, what) => {
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

const draws = (page) => page.evaluate(() => Number(document.querySelector('.pg').dataset.draws || 0));
async function redrawn(page, before) {
  await page.waitForFunction((b) => {
    const s = document.querySelector('.pg').dataset;
    return !s.busy && Number(s.draws) > Number(b);
  }, before, { timeout: 30000 });
  await page.waitForTimeout(400);
}
/** The screen center of the mark with this data-id, when it is inside the picture's pane. */
const centerOf = (page, id) => page.evaluate((id) => {
  const pane = document.querySelector('.pg-canvas').getBoundingClientRect();
  const el = [...document.querySelectorAll('.pg-canvas .dg-box[data-id]')].find((x) => x.getAttribute('data-id') === id);
  if (!el) return null;
  const r = el.getBoundingClientRect();
  const x = r.left + r.width / 2;
  const y = r.top + r.height / 2;
  return { x, y, inside: x > pane.left && x < pane.right && y > pane.top && y < pane.bottom };
}, id);
const ids = (page, prefix) => page.$$eval('.pg-canvas [data-id]', (els, p) => [...new Set(els.map((e) => e.getAttribute('data-id')).filter((i) => i.startsWith(p)))], prefix);

const browser = await chromium.launch();
const errors = [];
try {
  for (const theme of ['light', 'dark']) {
    const ctx = await browser.newContext({ viewport: { width: 1400, height: 1000 }, colorScheme: theme });
    const page = await ctx.newPage();
    page.on('pageerror', (e) => errors.push(`${theme}: ${e}`));
    await page.goto(`${BASE}/explore/water-rhf`);
    await page.waitForSelector('.pg-canvas svg', { timeout: 60000 });
    await page.waitForFunction(() => !document.querySelector('.rv-wait'), null, { timeout: 120000 });
    await page.evaluate(() => document.querySelector('.pg-split').scrollIntoView({ block: 'end', behavior: 'instant' }));
    await page.waitForTimeout(500);

    // the first view: closed multi-paths, and the list in the panel
    const closed = (await ids(page, 'mp:')).filter((i) => !i.includes('/') && !i.includes('#'));
    ok(closed.length >= 3, `${theme}: the first view draws ${closed.length} closed multi-paths (${closed.join(', ')})`);
    const summary = await page.$eval('.pg-mps summary', (s) => s.textContent).catch(() => '');
    ok(/Multi-paths: \d+ of \d+ stages/.test(summary), `${theme}: the panel lists the multi-paths: "${summary.trim()}"`);
    await page.screenshot({ path: join(SHOTS, `multipath-${theme}-1-first.png`) });

    // pick the ERI stage from the list
    let before = await draws(page);
    await page.click('.pg-mp-list button:has-text("eri")');
    await redrawn(page, before).catch(() => {});
    await page.waitForTimeout(600);
    const eri = await centerOf(page, 'mp:eri@ao_eri');
    ok(eri && eri.inside, `${theme}: picking eri draws its closed box and brings it into view`);
    const panel = await page.$eval('.pg-panel', (p) => p.textContent);
    ok(/independent implementation/.test(panel) && /Contract/.test(panel) && /Checked by/.test(panel) && /mderi_screened/.test(panel), `${theme}: the panel shows the stage's members, contract and tests`);
    await page.screenshot({ path: join(SHOTS, `multipath-${theme}-2-eri-closed.png`) });

    // open it
    before = await draws(page);
    if (eri) await page.mouse.click(eri.x, eri.y);
    await redrawn(page, before).catch(() => {});
    const marks = await ids(page, 'mp:eri@ao_eri');
    const diamonds = await page.$$eval('.pg-canvas path.dg-box[data-id^="mp:eri@ao_eri/"]', (ps) => ps.map((p) => p.getAttribute('data-id')));
    ok(diamonds.includes('mp:eri@ao_eri/fork') && diamonds.includes('mp:eri@ao_eri/join'), `${theme}: opened, the stage is a fork and a join (${diamonds.join(', ')})`);
    const paths = marks.filter((m) => /\/(to|from):/.test(m) && !m.includes('#'));
    ok(paths.length === 8, `${theme}: a path in and out for each of the 4 members (${paths.length} arrows)`);
    const defBox = await page.$$eval('.pg-canvas rect[data-id="mderi_screened"]', (r) => r.length);
    ok(defBox === 1 && marks.includes('mp:eri@ao_eri/m:hgp_eri') && marks.includes('mp:eri@ao_eri/m:rys_eri'), `${theme}: the default that ran is its own def box on its path; hgp_eri and rys_eri are cross-check boxes`);
    const style = await page.$eval('.pg-canvas [data-id="mp:eri@ao_eri/to:mderi_screened"] path:not(.dg-hit)', (p) => p.getAttribute('stroke-width')).catch(() => '');
    ok(Number(style) > 1.5, `${theme}: the default's path is drawn heavier (stroke ${style})`);
    // the whole motif in view: the host, the fork, the members and the join,
    // moved to the pane's middle by a drag, and zoomed out until it fits
    const motif = () => page.evaluate(() => {
      const pane = document.querySelector('.pg-canvas').getBoundingClientRect();
      const els = [...document.querySelectorAll('.pg-canvas .dg-box[data-id]')].filter((e) => /^(mp:eri@ao_eri\/|mderi_screened$)/.test(e.getAttribute('data-id')));
      const rs = els.map((e) => e.getBoundingClientRect());
      const r = { l: Math.min(...rs.map((x) => x.left)), r: Math.max(...rs.map((x) => x.right)), t: Math.min(...rs.map((x) => x.top)), b: Math.max(...rs.map((x) => x.bottom)) };
      return { cx: (r.l + r.r) / 2, cy: (r.t + r.b) / 2, w: r.r - r.l, h: r.b - r.t, pcx: (pane.left + pane.right) / 2, pcy: (pane.top + pane.bottom) / 2, pw: pane.width, ph: pane.height };
    });
    for (let i = 0; i < 6; i++) {
      const m = await motif();
      if (m.w < m.pw * 0.9 && m.h < m.ph * 0.9) break;
      await page.mouse.move(m.pcx, m.pcy);
      await page.mouse.wheel(0, 100);
      await page.waitForTimeout(250);
    }
    // drags that start and end inside the pane, each moving the picture at
    // most half the pane, until the motif's middle is the pane's
    for (let i = 0; i < 8; i++) {
      const m = await motif();
      const clamp = (v, lim) => Math.max(-lim, Math.min(lim, v));
      const dx = clamp(m.pcx - m.cx, m.pw / 2 - 30);
      const dy = clamp(m.pcy - m.cy, m.ph / 2 - 30);
      if (Math.abs(dx) < 8 && Math.abs(dy) < 8) break;
      await page.mouse.move(m.pcx - dx / 2, m.pcy - dy / 2);
      await page.mouse.down();
      await page.mouse.move(m.pcx + dx / 2, m.pcy + dy / 2, { steps: 8 });
      await page.mouse.up();
      await page.waitForTimeout(300);
    }
    const inView = await page.evaluate(() => {
      const pane = document.querySelector('.pg-canvas').getBoundingClientRect();
      return ['mp:eri@ao_eri/fork', 'mp:eri@ao_eri/join', 'mderi_screened'].every((id) => {
        const el = [...document.querySelectorAll('.pg-canvas .dg-box[data-id]')].find((x) => x.getAttribute('data-id') === id);
        if (!el) return false;
        const r = el.getBoundingClientRect();
        return r.left >= pane.left && r.right <= pane.right && r.top >= pane.top && r.bottom <= pane.bottom;
      });
    });
    ok(inView, `${theme}: the fork, the default's box and the join fit in the pane together`);
    await page.mouse.move(5, 5);
    await page.waitForTimeout(200);
    await page.screenshot({ path: join(SHOTS, `multipath-${theme}-3-eri-open.png`) });
    // a closer look: zoom in at the fork
    const f2 = await centerOf(page, 'mp:eri@ao_eri/fork');
    if (f2 && f2.inside) {
      for (let i = 0; i < 3; i++) {
        await page.mouse.move(f2.x, f2.y);
        await page.mouse.wheel(0, -100);
        await page.waitForTimeout(150);
      }
      const f3 = await centerOf(page, 'mp:eri@ao_eri/fork');
      ok(!!f3, `${theme}: the fork stays drawn after zooming`);
      await page.mouse.move(5, 5);
      await page.waitForTimeout(200);
      await page.screenshot({ path: join(SHOTS, `multipath-${theme}-4-eri-open-zoom.png`) });
    }

    // close it with a click on the fork
    const f4 = await centerOf(page, 'mp:eri@ao_eri/fork');
    before = await draws(page);
    if (f4 && f4.inside) await page.mouse.click(f4.x, f4.y);
    await redrawn(page, before).catch(() => {});
    const after = await ids(page, 'mp:eri@ao_eri');
    ok(after.includes('mp:eri@ao_eri') && !after.includes('mp:eri@ao_eri/fork'), `${theme}: a click on the fork closes it into one box again`);
    await ctx.close();
  }
} finally {
  await browser.close();
  server.kill();
}
ok(errors.length === 0, `no page errors${errors.length ? `: ${errors.slice(0, 3).join(' | ')}` : ''}`);
console.log(`${fails ? 'FAIL' : 'ok'}: multipath browser test, screenshots in ${SHOTS}`);
process.exit(fails ? 1 : 0);
