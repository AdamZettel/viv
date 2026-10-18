// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The whole program as a graph (ProgramGraph.jsx, GraphPanel.jsx) on the
// explorer's Read view, in a real browser.
//
//     npm run build && npm run graphbrowsertest
//     GRAPH_SHOTS=/some/dir npm run graphbrowsertest     (keep screenshots there)
//     GRAPH_SLUGS=water-rhf GRAPH_PORT=4462 npm run graphbrowsertest
//
// For each program, in light and dark, at 1400 by 1000:
//   - the picture draws one box per box record of the bundle's drawing, and
//     the picture and its panel are both in the viewport;
//   - the pointer on a def's box lights its calls and dims the rest, with a
//     tooltip; on an arrow, the arrow and its ends, with its variables or count;
//   - a click on a def's box shows its code in the panel (its first line is
//     the def's), a click on a step's box its def's code with the step's lines
//     marked and scrolled into the panel, and the picture does not move off
//     the screen;
//   - a def's `+n` mark opens it: the picture is redrawn by lib/graphdraw in
//     the tab's viv.wasm with n more boxes, the opened box stays where it was
//     on the screen, and the time it took is recorded; its `−` closes it;
//   - the search finds a def that is not drawn, opens the path to it and
//     brings its box into view; + and − zoom, Fit shows the whole picture, the
//     wheel zooms and a drag moves the picture; the arrow keys move the pick
//     from box to box and Enter opens a def;
//   - scrolling the full reading below outlines the def at the reading line;
//   - the panel's code is code only (no `###`, `#|` or `#:` line, for
//     `<main>` and for a def with `#|` lines above it), 13 px or more with
//     unwrapped lines in a block that scrolls sideways; "show comments"
//     brings back the full text and unchecking it takes it away again; the
//     divider between the picture and the panel drags, and a double-click
//     puts it back;
//   - the walkthrough (Run): it plays by itself and pauses, space and the
//     arrows drive it, Next takes it through every stop to the end with each
//     stop narrated, its box picked and in view and its lines marked; a
//     later pass opens step by step, a busy def opens in the panel and
//     "Back to the walk" returns; the last stop shows the printed output and
//     the holds; Exit leaves it; the Timeline tab still works; on a phone
//     the walk's buttons stay on the screen with the narration under the
//     picture.
// At 390 by 844 (a phone): nothing wider than the screen, the picture above
// the panel, and a tap on a box shows its code under it.
// Screenshots of each state go to GRAPH_SHOTS (default www/graph-shots).
import { mkdirSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { chromium } from 'playwright-core';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const PORT = Number(process.env.GRAPH_PORT || 4417);
const BASE = `http://127.0.0.1:${PORT}`;
const SHOTS = process.env.GRAPH_SHOTS || join(WWW, 'graph-shots');
mkdirSync(SHOTS, { recursive: true });
const SLUGS = (process.env.GRAPH_SLUGS || 'water-rhf,minpack-rosenbrock,minpack-thurber,silicon-dense,tiny-gpt,skyrme-o16,casida-water').split(',');
/** The step of each program's walk to photograph: one where the algorithm happens. */
const STEP_SHOT = { 'tiny-gpt': /attention \/ weights/, 'skyrme-o16': /diagonalize/, 'casida-water': /tdhf/ };
/** The programs whose walk must reach a core step, passes 1 and 4, open a pass and peek at a busy def. */
const WALK_FULL = ['water-rhf', 'minpack-rosenbrock', 'tiny-gpt', 'skyrme-o16', 'casida-water'];
/** A def to open in the panel and photograph: the heart of each program. */
const KEY_DEF = { 'water-rhf': 'scf_iterate', 'skyrme-o16': 'skyrme_fields', 'casida-water': 'casida_tdhf' };
/** The owner's target for opening or closing a box, for the water SCF. */
const REDRAW_TARGET_MS = 300;
let fails = 0;
let checks = 0;
const timings = [];
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

/** The center of the first mark matching `sel` (and `pred`) inside the picture's pane, or null. */
async function inPane(page, sel, pred = '() => true') {
  return page.evaluate(([sel, pred]) => {
    const pane = document.querySelector('.pg-canvas').getBoundingClientRect();
    const f = eval(pred); // eslint-disable-line no-eval
    for (const el of document.querySelectorAll(`.pg-canvas ${sel}`)) {
      if (!f(el)) continue;
      const r = el.getBoundingClientRect();
      const x = r.left + r.width / 2;
      const y = r.top + Math.min(r.height / 2, 9);
      if (r.width > 4 && x > pane.left + 4 && x < pane.right - 4 && y > pane.top + 4 && y < pane.bottom - 4) return { x, y, id: el.getAttribute('data-id') };
    }
    return null;
  }, [sel, pred]);
}
/** Bring box `id` into the pane's view through the search box (it stays open as it is). */
async function bringIntoView(page, id) {
  await page.click('#pg-search');
  await page.fill('#pg-search', id);
  await page.waitForTimeout(150);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(700);
}
const state = (page) => page.evaluate(() => ({ ...document.querySelector('.pg').dataset }));
const paneBox = (page) => page.evaluate(() => {
  const r = document.querySelector('.pg-canvas').getBoundingClientRect();
  const p = document.querySelector('.pg-panel').getBoundingClientRect();
  return { top: r.top, bottom: r.bottom, left: r.left, right: r.right, w: r.width, h: r.height, vh: innerHeight, vw: innerWidth, pTop: p.top, pBottom: p.bottom, pLeft: p.left, pW: p.width };
});
const rectOfId = (page, id) => page.evaluate((id) => {
  const r = [...document.querySelectorAll('.pg-canvas rect[data-id]')].find((x) => x.getAttribute('data-id') === id);
  if (!r) return null;
  const b = r.getBoundingClientRect();
  return { x: b.left, y: b.top, w: b.width, h: b.height };
}, id);
/** Wait until the picture has been drawn again (the count of drawings goes up). */
async function redrawn(page, before) {
  await page.waitForFunction((b) => {
    const s = document.querySelector('.pg').dataset;
    return !s.busy && Number(s.draws) > Number(b);
  }, before, { timeout: 20000 });
  await page.waitForTimeout(150);
}

const browser = await chromium.launch();
const errors = [];
try {
  for (const slug of SLUGS) {
    const bundle = JSON.parse(readFileSync(join(WWW, 'dist', 'explore', `${slug}.json`), 'utf8'));
    const want = (bundle.drawing.match(/^box: /gm) || []).length;
    const nodes = new Map(bundle.nodes.map((n) => [n.id, n]));
    for (const theme of ['light', 'dark']) {
      const ctx = await browser.newContext({ viewport: { width: 1400, height: 1000 }, colorScheme: theme });
      const page = await ctx.newPage();
      page.on('pageerror', (e) => errors.push(`${slug} (${theme}): ${e}`));
      await page.goto(`${BASE}/explore/${slug}`);
      await page.waitForSelector('.pg-canvas svg', { timeout: 60000 });
      // the recording done (the page above the picture is then as tall as it stays),
      // the picture and the panel with their bottom at the viewport's bottom
      await page.waitForFunction(() => !document.querySelector('.rv-wait'), null, { timeout: 120000 });
      await page.evaluate(() => document.querySelector('.pg-split').scrollIntoView({ block: 'end', behavior: 'instant' }));
      await page.waitForTimeout(500);
      const rects = await page.$$eval('.pg-canvas svg rect[data-id]', (rs) => rs.length);
      ok(rects === want, `${slug} (${theme}): the picture draws the ${want} boxes of its drawing (${rects} in the SVG)`);
      const pb = await paneBox(page);
      const head = await page.evaluate(() => document.querySelector('header')?.getBoundingClientRect().bottom || 0);
      ok(pb.top >= head - 1 && pb.bottom <= pb.vh + 1 && pb.w > 600 && pb.h > 400 && pb.pLeft > pb.left, `${slug} (${theme}): the picture (${Math.round(pb.w)} x ${Math.round(pb.h)}) and the panel beside it fit in the viewport under the site's header`);
      await page.screenshot({ path: join(SHOTS, `${slug}-${theme}-1-first.png`) });

      // the pointer on a def's box: its calls lit, the rest dimmed, a tooltip
      const defBox = await inPane(page, 'rect[data-id]', `(el) => { const id = el.getAttribute('data-id'); return !id.includes('/') && [...document.querySelectorAll('.pg-canvas .dg-arrow')].some((a) => a.getAttribute('data-id').startsWith('call:' + id)); }`);
      if (defBox) {
        await page.mouse.move(defBox.x, defBox.y);
        await page.waitForTimeout(200);
        const lit = await page.evaluate(() => ({
          dimming: document.querySelector('.pg-canvas svg').classList.contains('pg-dimming'),
          arrows: document.querySelectorAll('.pg-canvas .dg-arrow.pg-lit').length,
          tip: document.querySelector('.pg-tip').hidden ? '' : document.querySelector('.pg-tip').textContent,
        }));
        ok(lit.dimming && lit.arrows > 0 && lit.tip.length > 0, `${slug} (${theme}): the pointer on ${defBox.id} lights ${lit.arrows} arrows, dims the rest, and says "${lit.tip.split('\n').join(' / ').slice(0, 90)}"`);
        await page.screenshot({ path: join(SHOTS, `${slug}-${theme}-2-hover-box.png`) });
      } else ok(false, `${slug} (${theme}): a def's box with calls in the pane`);
      // on an arrow
      const arrow = await page.evaluate(() => {
        const pane = document.querySelector('.pg-canvas').getBoundingClientRect();
        for (const g of document.querySelectorAll('.pg-canvas .dg-arrow')) {
          const id = g.getAttribute('data-id');
          if (!/^(call|data):/.test(id)) continue;
          const path = g.querySelector('path.dg-hit');
          const len = path.getTotalLength();
          for (const f of [0.5, 0.35, 0.65]) {
            const pt = path.getPointAtLength(len * f);
            const m = path.getScreenCTM();
            const x = m.a * pt.x + m.c * pt.y + m.e;
            const y = m.b * pt.x + m.d * pt.y + m.f;
            const top = document.elementFromPoint(x, y);
            if (x > pane.left + 4 && x < pane.right - 4 && y > pane.top + 4 && y < pane.bottom - 4 && top && top.closest('.dg-arrow') === g) return { x, y, id };
          }
        }
        return null;
      });
      if (arrow) {
        await page.mouse.move(arrow.x, arrow.y);
        await page.waitForTimeout(200);
        const lit = await page.evaluate(() => ({
          arrows: [...document.querySelectorAll('.pg-canvas .dg-arrow.pg-lit')].map((a) => a.getAttribute('data-id')),
          tip: document.querySelector('.pg-tip').hidden ? '' : document.querySelector('.pg-tip').textContent,
        }));
        ok(lit.arrows.length === 1 && lit.arrows[0] === arrow.id && /calls|hands/.test(lit.tip), `${slug} (${theme}): the pointer on ${arrow.id} lights it alone and says "${lit.tip}"`);
        await page.screenshot({ path: join(SHOTS, `${slug}-${theme}-3-hover-arrow.png`) });
      } else ok(false, `${slug} (${theme}): an arrow in the pane`);
      await page.mouse.move(2, 2);

      // a click on a def's box: its code in the panel; the picture stays put
      const pick = await inPane(page, 'rect[data-id]', `(el) => { const id = el.getAttribute('data-id'); return !id.includes('/') && id !== '<main>'; }`);
      if (pick) {
        const before = await paneBox(page);
        await page.mouse.click(pick.x, pick.y);
        await page.waitForTimeout(500);
        const got = await page.evaluate(() => {
          const c = document.querySelector('.pg-panel .pg-code');
          return { sel: document.querySelector('.pg').dataset.sel, from: c ? Number(c.dataset.from) : null, title: document.querySelector('.pg-panel .pg-title')?.textContent || '' };
        });
        const after = await paneBox(page);
        ok(got.sel === pick.id && got.from === nodes.get(pick.id).line && Math.abs(after.top - before.top) < 1, `${slug} (${theme}): a click on ${pick.id} shows its code in the panel from line ${got.from} ("${got.title.slice(0, 60)}"), and the picture stays where it was`);
        await page.screenshot({ path: join(SHOTS, `${slug}-${theme}-4-click-def.png`) });
      } else ok(false, `${slug} (${theme}): a def's box to click`);
      // the program's key def, found by the search: its code in the panel with its intent
      if (KEY_DEF[slug] && theme === 'light') {
        const key = KEY_DEF[slug];
        await bringIntoView(page, key);
        await page.waitForFunction((t) => document.querySelector('.pg').dataset.sel === t, key, { timeout: 20000 }).catch(() => {});
        await page.waitForTimeout(400);
        const got = await page.evaluate(() => ({ sel: document.querySelector('.pg').dataset.sel, title: document.querySelector('.pg-panel .pg-title')?.textContent || '', from: Number(document.querySelector('.pg-panel .pg-code')?.dataset.from || 0), text: document.querySelector('.pg-panel')?.textContent || '' }));
        ok(got.sel === key && got.title.includes(key) && got.from === nodes.get(key).line && got.text.length > 200, `${slug}: the search opens ${key} in the panel: its code from line ${got.from}, with its intent (${got.text.length} characters)`);
        await page.screenshot({ path: join(SHOTS, `${slug}-light-4b-key-def.png`) });
      }
      // a click on a step's box: its lines marked and in the panel's view
      const isStep = `(el) => el.getAttribute('data-id').includes('/') && !el.getAttribute('data-id').includes('loop@')`;
      if (!(await inPane(page, 'rect[data-id]', isStep))) {
        const any = await page.evaluate(() => [...document.querySelectorAll('.pg-canvas rect[data-id]')].map((r) => r.getAttribute('data-id')).find((id) => id.includes('/') && !id.includes('loop@')));
        if (any) await bringIntoView(page, any);
      }
      const stepBox = await inPane(page, 'rect[data-id]', isStep);
      if (stepBox) {
        await page.mouse.click(stepBox.x, stepBox.y);
        await page.waitForTimeout(1200);
        const n = nodes.get(stepBox.id);
        const got = await page.evaluate(() => {
          const on = [...document.querySelectorAll('.pg-panel .pg-cl.on')];
          const panel = document.querySelector('.pg-panel').getBoundingClientRect();
          const r = on.length ? on[0].getBoundingClientRect() : null;
          return { lines: on.map((x) => Number(x.dataset.line)), visible: !!r && r.top >= panel.top - 1 && r.top < panel.bottom - 10 };
        });
        const pb2 = await paneBox(page);
        // code only: the step's comment lines, if it has any, are not among them
        ok(got.lines.length > 0 && got.lines[0] === n.line && got.lines[got.lines.length - 1] === n.end && got.lines.every((x) => x >= n.line && x <= n.end) && got.visible && pb2.top >= 0 && pb2.bottom <= pb2.vh + 1,
          `${slug} (${theme}): a click on the step ${stepBox.id} marks its lines ${n.line}-${n.end} in the panel, scrolled into view, with the picture still on the screen`);
        await page.screenshot({ path: join(SHOTS, `${slug}-${theme}-5-click-step.png`) });
      } else ok(false, `${slug} (${theme}): a step's box to click`);

      // the panel's code: code only by default, the comments with "show comments"
      {
        const panelLines = () => page.evaluate(() => [...document.querySelectorAll('.pg-panel .pg-code .pg-cl')].map((c) => ({ x: Number(c.dataset.line), t: c.querySelector('code').textContent })));
        const prose = (ls) => ls.filter((l) => /^\s*(###|#\||#:)/.test(l.t));
        // a def that ran with `#|` lines right above it, else any def that ran
        const P = bundle.program.split('\n');
        const ran = new Set(bundle.calls.flatMap(([a, c]) => [a, c]));
        const cands = bundle.nodes.filter((d) => d.kind === 'def' && d.id !== '<main>' && ran.has(d.id));
        const doc = cands.find((d) => /^\s*#\|/.test(P[d.line - 2] || '')) || cands.find((d) => /^\s*#@/.test(P[d.line - 2] || '') && /^\s*#\|/.test(P[d.line - 3] || ''));
        for (const id of ['<main>', doc ? doc.id : null].filter(Boolean)) {
          await bringIntoView(page, id);
          await page.waitForFunction((t) => document.querySelector('.pg').dataset.sel === t, id, { timeout: 20000 }).catch(() => {});
          await page.waitForTimeout(300);
          const style = await page.evaluate(() => {
            const c = document.querySelector('.pg-panel .pg-cl code');
            if (!c) return null;
            const cs = getComputedStyle(c);
            return { size: parseFloat(cs.fontSize), lh: parseFloat(cs.lineHeight) / parseFloat(cs.fontSize), ws: cs.whiteSpace, scroll: getComputedStyle(document.querySelector('.pg-panel .pg-code')).overflowX };
          });
          const plain = await panelLines();
          const box = await page.evaluate(() => document.querySelector('.pg-panel .pg-code')?.dataset.full);
          ok(plain.length > 0 && prose(plain).length === 0 && box === '', `${slug} (${theme}): ${id} in the panel is code only: ${plain.length} lines, none of them \`###\`, \`#|\` or \`#:\``);
          ok(style && style.size >= 13 && style.lh >= 1.4 && style.ws === 'pre' && style.scroll === 'auto', `${slug} (${theme}): the code is ${style?.size}px monospace, line height ${style?.lh.toFixed(2)}, lines not wrapped (${style?.ws}), the block scrolls sideways (${style?.scroll})`);
          if (theme !== 'light') continue;
          await page.click('.pg-panel .pg-comments');
          await page.waitForTimeout(300);
          const all = await panelLines();
          ok(all.length > plain.length && (id === '<main>' || prose(all).length > 0),
            `${slug}: "show comments" on ${id} brings back the full text: ${plain.length} -> ${all.length} lines, ${prose(all).length} of them \`###\`, \`#|\` or \`#:\``);
          await page.screenshot({ path: join(SHOTS, `${slug}-light-5b-comments-${id.replace(/[<>]/g, '')}.png`) });
          await page.click('.pg-panel .pg-comments');
          await page.waitForTimeout(300);
          const back = await panelLines();
          ok(back.length === plain.length, `${slug}: "show comments" off gives the code only again (${back.length} lines)`);
        }
      }

      if (theme === 'light') {
        // the divider: a drag gives the panel more room, a double-click puts it back
        const d0 = await paneBox(page);
        const div = await page.evaluate(() => { const r = document.querySelector('.pg-divider').getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; });
        await page.mouse.move(div.x, div.y);
        await page.mouse.down();
        await page.mouse.move(div.x - 150, div.y, { steps: 6 });
        await page.mouse.up();
        await page.waitForTimeout(300);
        const d1 = await paneBox(page);
        await page.mouse.dblclick(div.x - 150, div.y);
        await page.waitForTimeout(300);
        const d2 = await paneBox(page);
        ok(d1.pW > d0.pW + 100 && Math.abs(d2.pW - d0.pW) < 2, `${slug}: dragging the divider widens the panel (${Math.round(d0.pW)} -> ${Math.round(d1.pW)} px) and a double-click puts it back (${Math.round(d2.pW)} px)`);

        // open a box by its +n mark, and close it by its − mark
        let more = await inPane(page, 'text[data-id$="#more"]');
        if (!more) {
          const any = await page.evaluate(() => document.querySelector('.pg-canvas text[data-id$="#more"]')?.getAttribute('data-id'));
          if (any) {
            await bringIntoView(page, any.split('#')[0]);
            more = await inPane(page, 'text[data-id$="#more"]');
          }
        }
        if (!more) {
          // everything is open at first in a small program: close one, then open it again
          const less = await inPane(page, 'text[data-id$="#less"]');
          if (less) {
            const s0 = (await state(page)).draws;
            await page.mouse.click(less.x, less.y);
            await redrawn(page, s0);
            more = await inPane(page, 'text[data-id$="#more"]');
          }
        }
        if (more) {
          const d = more.id.split('#')[0];
          const s0 = await state(page);
          const r0 = await rectOfId(page, d);
          const t0 = Date.now();
          await page.mouse.click(more.x, more.y);
          await redrawn(page, s0.draws);
          const wall = Date.now() - t0;
          const s1 = await state(page);
          const r1 = await rectOfId(page, d);
          const moved = r0 && r1 ? Math.hypot(r1.x - r0.x, r1.y - r0.y) : Infinity;
          timings.push(`${slug}: opened ${d} (${s0.shown} -> ${s1.shown} defs) in ${s1.redrawMs} ms (${wall} ms from the click to the new picture)`);
          ok(Number(s1.shown) > Number(s0.shown) && moved < 2, `${slug}: the +n mark of ${d} opens it: ${s0.shown} -> ${s1.shown} defs drawn, redrawn in ${s1.redrawMs} ms, the opened box moved ${moved.toFixed(1)} px`);
          if (slug === 'water-rhf') ok(Number(s1.redrawMs) < REDRAW_TARGET_MS, `${slug}: opening a box took ${s1.redrawMs} ms, under ${REDRAW_TARGET_MS} ms`);
          await page.screenshot({ path: join(SHOTS, `${slug}-light-6-opened.png`) });
          const less = await inPane(page, `text[data-id="${d}#less"]`);
          if (less) {
            const t1 = Date.now();
            await page.mouse.click(less.x, less.y);
            await redrawn(page, s1.draws);
            const s2 = await state(page);
            timings.push(`${slug}: closed ${d} in ${s2.redrawMs} ms (${Date.now() - t1} ms from the click)`);
            ok(s2.shown === s0.shown, `${slug}: the − mark of ${d} closes it again: ${s2.shown} defs drawn, in ${s2.redrawMs} ms`);
          } else ok(false, `${slug}: ${d}'s − mark in the pane after opening it`);
        } else ok(slug !== 'water-rhf', `${slug}: a +n mark in the pane`);

        // search for a def that is not drawn
        const s0 = await state(page);
        const target = await page.evaluate(async (slug) => {
          const b = await (await fetch(`/explore/${slug}.json`)).json();
          const drawn = new Set([...document.querySelectorAll('.pg-canvas rect[data-id]')].map((r) => r.getAttribute('data-id')));
          const ran = new Set(b.calls.flatMap(([a, c]) => [a, c]));
          const notDrawn = [...ran].filter((d) => !drawn.has(d) && d !== '<main>');
          return notDrawn.sort((a, c) => c.length - a.length)[0] || [...drawn].find((d) => !d.includes('/') && d !== '<main>');
        }, slug);
        await page.click('#pg-search');
        await page.keyboard.type(target);
        await page.waitForTimeout(200);
        await page.screenshot({ path: join(SHOTS, `${slug}-light-7-search.png`) });
        await page.keyboard.press('Enter');
        await page.waitForFunction((t) => document.querySelector('.pg').dataset.sel === t && [...document.querySelectorAll('.pg-canvas rect[data-id]')].some((r) => r.getAttribute('data-id') === t), target, { timeout: 20000 });
        await page.waitForTimeout(700);
        const found = await page.evaluate((t) => {
          const r = [...document.querySelectorAll('.pg-canvas rect[data-id]')].find((x) => x.getAttribute('data-id') === t).getBoundingClientRect();
          const p = document.querySelector('.pg-canvas').getBoundingClientRect();
          return { inView: r.left >= p.left - 1 && r.right <= p.right + 1 && r.top >= p.top - 1 && r.bottom <= p.bottom + 1, shown: document.querySelector('.pg').dataset.shown };
        }, target);
        ok(found.inView, `${slug}: the search finds ${target} (drawn defs ${s0.shown} -> ${found.shown}), picks it and brings its box into view`);
        await page.screenshot({ path: join(SHOTS, `${slug}-light-8-found.png`) });

        // zoom: buttons, Fit, the wheel, a drag
        const z0 = Number((await state(page)).scale);
        await page.click('.pg-zoomin');
        await page.waitForTimeout(100);
        const z1 = Number((await state(page)).scale);
        await page.click('.pg-fit');
        await page.waitForTimeout(500);
        const z2 = Number((await state(page)).scale);
        const pb3 = await paneBox(page);
        await page.mouse.move(pb3.left + pb3.w / 2, pb3.top + pb3.h / 2);
        await page.mouse.wheel(0, -100);
        await page.waitForTimeout(150);
        const z3 = Number((await state(page)).scale);
        const vb0 = await page.$eval('.pg-canvas svg', (s) => s.getAttribute('viewBox'));
        await page.mouse.move(pb3.left + 60, pb3.bottom - 40);
        await page.mouse.down();
        await page.mouse.move(pb3.left + 160, pb3.bottom - 90, { steps: 5 });
        await page.mouse.up();
        const vb1 = await page.$eval('.pg-canvas svg', (s) => s.getAttribute('viewBox'));
        ok(z1 > z0 && z2 <= z1 && z3 > z2 && vb1 !== vb0, `${slug}: + zooms in (${z0} -> ${z1}), Fit shows the whole picture (${z2}), the wheel zooms (${z3}), a drag moves it`);
        await page.click('.pg-fit');
        await page.waitForTimeout(500);
        await page.screenshot({ path: join(SHOTS, `${slug}-light-9-fit.png`) });

        // the keyboard: from box to box, Enter opens
        await page.focus('.pg-canvas');
        const k0 = (await state(page)).sel;
        const press = async (keys) => {
          const before = (await state(page)).sel;
          for (const k of keys) {
            await page.keyboard.press(k);
            await page.waitForTimeout(350);
            const now = (await state(page)).sel;
            if (now !== before) return now;
          }
          return before;
        };
        const k1 = await press(['ArrowRight', 'ArrowLeft']);
        const k2 = await press(['ArrowDown', 'ArrowUp']);
        ok(k1 && k1 !== k0 && k2 && k2 !== k1, `${slug}: the arrow keys move the pick from box to box (${k0} -> ${k1} -> ${k2})`);

        // the full reading below: the def at the reading line is outlined
        await page.evaluate(() => {
          const r = [...document.querySelectorAll('.rv-row')].find((x) => x.dataset.def && x.dataset.def !== '<main>' && document.querySelector(`.pg-canvas rect[data-id="${CSS.escape(x.dataset.def)}"]`));
          if (r) window.scrollTo(0, r.getBoundingClientRect().top + window.scrollY - window.innerHeight * 0.35 + 4);
        });
        await page.waitForTimeout(900);
        const hl = await page.$$eval('.pg-canvas rect[data-hl].dg-hl', (rs) => rs.map((r) => r.getAttribute('data-hl')));
        ok(hl.length > 0, `${slug}: the reading line's def is outlined in the picture (${hl.join(', ')})`);
      }
      await ctx.close();
    }


    // the walkthrough of the recorded run (lib/tour.js): Run, Pause, Next,
    // Back, speed and the keys; every stop narrated, its box picked and in
    // view, its lines marked; the passes opened on demand, a def opened from
    // a stop and back, Exit; then the Timeline tab still works
    {
      const ctx = await browser.newContext({ viewport: { width: 1400, height: 1000 }, colorScheme: 'light' });
      const page = await ctx.newPage();
      page.on('pageerror', (e) => errors.push(`${slug} (walk): ${e}`));
      await page.goto(`${BASE}/explore/${slug}`);
      await page.waitForSelector('.pg-canvas svg', { timeout: 60000 });
      await page.waitForFunction(() => !document.querySelector('.rv-wait'), null, { timeout: 120000 });
      await page.waitForSelector('.pg-tour .pg-run', { timeout: 20000 });
      const tour = () => page.evaluate(() => {
        const d = document.querySelector('.pg-split').dataset;
        const t = document.querySelector('.pg-tourstop');
        const sel = document.querySelector('.pg').dataset.sel;
        const pane = document.querySelector('.pg-canvas').getBoundingClientRect();
        const box = [...document.querySelectorAll('.pg-canvas rect[data-id]')].find((r) => r.getAttribute('data-id') === sel);
        const r = box ? box.getBoundingClientRect() : null;
        return {
          k: d.tour === '' ? -1 : Number(d.tour), kind: d.tourKind, playing: d.playing === '1', sel, busy: !!document.querySelector('.pg').dataset.busy,
          title: t?.querySelector('.pg-title')?.textContent || '', narr: (t?.querySelector('.pg-narr')?.textContent || '').trim(),
          marked: document.querySelectorAll('.pg-panel .pg-cl.on').length, code: !!document.querySelector('.pg-panel .pg-tourcode .pg-code'),
          inView: !!r && r.left < pane.right - 4 && r.right > pane.left + 4 && r.top < pane.bottom - 4 && r.bottom > pane.top + 4,
          count: document.querySelector('.pg-count')?.textContent || '',
        };
      });
      const settle = async () => {
        await page.waitForFunction(() => !document.querySelector('.pg').dataset.busy, null, { timeout: 20000 });
        await page.waitForTimeout(380);
      };
      await page.click('.pg-tour .pg-run');
      await page.waitForTimeout(300);
      await page.selectOption('.pg-speed select', '4');
      const t0 = await tour();
      await page.waitForTimeout(2600);
      const t1 = await tour();
      ok(t0.playing && t1.playing && t1.k > t0.k, `${slug}: Run plays the walk: from stop ${t0.k + 1} to stop ${t1.k + 1} by itself at 4x (${t1.count})`);
      await page.click('.pg-tour .pg-run');
      await page.waitForTimeout(200);
      const t2 = await tour();
      await page.waitForTimeout(1200);
      const t3 = await tour();
      ok(!t2.playing && t3.k === t2.k, `${slug}: Pause stops it at stop ${t2.k + 1}`);
      // the keys: space plays and pauses, the arrows step
      await page.focus('.pg-canvas');
      await page.keyboard.press(' ');
      await page.waitForTimeout(100);
      const k1 = await tour();
      await page.keyboard.press(' ');
      await page.waitForTimeout(100);
      const k2 = await tour();
      const kb = k2.k;
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(150);
      const k3 = await tour();
      await page.keyboard.press('ArrowLeft');
      await page.waitForTimeout(150);
      const k4 = await tour();
      ok(k1.playing && !k2.playing && k3.k === kb + 1 && k4.k === kb, `${slug}: space plays and pauses, → and ← move one stop (${kb + 1} -> ${k3.k + 1} -> ${k4.k + 1})`);
      // Back to the start, then Next to the end: every stop narrated, its box picked and in view, its lines marked
      for (let i = 0; i < 400 && (await tour()).k > 0; i++) await page.click('.pg-tour .pg-tback');
      await settle();
      const first = await tour();
      ok(first.k === 0 && first.kind === 'intro' && first.narr.length > 40, `${slug}: the first stop says what the program is for: "${first.narr.slice(0, 90)}…"`);
      await page.screenshot({ path: join(SHOTS, `${slug}-walk-1-start.png`) });
      const kinds = {};
      let bad = [];
      let shot = { step: false, pass1: false, pass4: false };
      let st = first;
      let opened = false;
      let peeked = false;
      for (let guard = 0; guard < 500 && st.kind !== 'end'; guard++) {
        await page.click('.pg-tour .pg-tnext');
        await settle();
        st = await tour();
        kinds[st.kind] = (kinds[st.kind] || 0) + 1;
        const needsMark = ['call', 'step', 'lines', 'loop', 'pass'].includes(st.kind);
        if (!st.narr || (needsMark && (!st.code || !st.marked)) || !st.inView) bad.push(`${st.k + 1} ${st.kind} "${st.title}" narr=${!!st.narr} marked=${st.marked} inView=${st.inView} sel=${st.sel}`);
        if (!shot.step && st.kind === 'step' && (STEP_SHOT[slug] || /two_electron|jacobian|integrals/).test(st.title)) {
          shot.step = true;
          await page.screenshot({ path: join(SHOTS, `${slug}-walk-2-step.png`) });
        }
        if (!shot.pass1 && st.kind === 'pass' && /^Pass 1 of/.test(st.title)) {
          shot.pass1 = true;
          await page.screenshot({ path: join(SHOTS, `${slug}-walk-3-pass1.png`) });
        }
        if (!shot.pass4 && st.kind === 'pass' && /^Pass 4 of/.test(st.title)) {
          shot.pass4 = true;
          await page.screenshot({ path: join(SHOTS, `${slug}-walk-4-pass4.png`) });
          // a later pass is one stop; it opens step by step on demand
          const open = await page.$('.pg-panel .pg-openpass');
          if (open) {
            await open.click();
            await settle();
            const inner = await tour();
            ok(inner.k === st.k + 1 && inner.kind !== 'pass', `${slug}: "Walk through this pass step by step" opens pass 4: the next stop is ${inner.kind} "${inner.title}"`);
            opened = true;
            st = inner;
          }
        }
        if (!peeked && st.kind === 'step') {
          const hot = await page.$('.pg-panel .pg-hot .pg-link');
          if (hot) {
            const id = await hot.evaluate((b) => b.querySelector('code').textContent);
            await hot.click();
            await settle();
            const p = await page.evaluate(() => ({ back: !!document.querySelector('.pg-panel .pg-tourback'), title: document.querySelector('.pg-panel .pg-title')?.textContent || '', stop: !!document.querySelector('.pg-tourstop') }));
            await page.click('.pg-panel .pg-tourback');
            await settle();
            const again = await tour();
            ok(p.back && !p.stop && p.title.includes(id) && again.k === st.k && !!again.narr, `${slug}: a busy def named at a stop (${id}) opens in the panel with the walk paused, and "Back to the walk" returns to stop ${st.k + 1}`);
            peeked = true;
          }
        }
      }
      ok(st.kind === 'end' && bad.length === 0, `${slug}: Next goes through all the stops to the end (${Object.entries(kinds).map(([k, n]) => `${n} ${k}`).join(', ')}), each narrated with its box picked and in view and its lines marked${bad.length ? `; not: ${bad.slice(0, 3).join(' | ')}` : ''}`);
      const end = await page.evaluate(() => ({ printed: document.querySelector('.pg-tourstop .pg-printed')?.textContent || '', holds: document.querySelectorAll('.pg-tourstop .pg-holds li').length, text: document.querySelector('.pg-tourstop .pg-narr')?.textContent || '' }));
      ok(end.printed.length > 0 && /held|checks|checks nothing/.test(end.text), `${slug}: the last stop shows what the run printed ("${end.printed.split('\n')[0]}") and what was checked (${end.holds} kinds of holds)`);
      await page.screenshot({ path: join(SHOTS, `${slug}-walk-5-end.png`) });
      if (WALK_FULL.includes(slug)) ok(shot.step && shot.pass1 && shot.pass4 && opened && peeked, `${slug}: the walk reached a step of the core path, passes 1 and 4, opened a pass and a busy def`);
      await page.click('.pg-tour .pg-texit');
      await page.waitForTimeout(300);
      const gone = await page.evaluate(() => ({ stop: !!document.querySelector('.pg-tourstop'), run: !!document.querySelector('.pg-tour .pg-run'), tour: document.querySelector('.pg-split').dataset.tour }));
      ok(!gone.stop && gone.run && gone.tour === '', `${slug}: Exit leaves the walk and the panel is the picker's again`);
      // the Timeline view of the same recording
      await page.click('.ex-modes button:nth-child(2)');
      await page.waitForTimeout(800);
      const tl = await page.evaluate(() => ({ grid: !!document.querySelector('.ex-grid'), code: !!document.querySelector('.ex-code-pane') }));
      ok(tl.grid && tl.code, `${slug}: the Timeline tab still shows its panes`);
      await ctx.close();
    }

    // a phone
    const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, hasTouch: true, isMobile: true });
    const page = await ctx.newPage();
    page.on('pageerror', (e) => errors.push(`${slug} (phone): ${e}`));
    await page.goto(`${BASE}/explore/${slug}`);
    await page.waitForSelector('.pg-canvas svg', { timeout: 60000 });
    await page.evaluate(() => document.querySelector('.pg').scrollIntoView({ block: 'start', behavior: 'instant' }));
    await page.waitForTimeout(600);
    const wide = await page.evaluate(() => document.scrollingElement.scrollWidth);
    const pb = await paneBox(page);
    ok(wide <= 391 && pb.w > 300 && pb.h > 250 && pb.pTop >= pb.bottom - 1, `${slug} (phone): nothing wider than the screen (${wide} px), the picture ${Math.round(pb.w)} x ${Math.round(pb.h)} above the panel`);
    await page.screenshot({ path: join(SHOTS, `${slug}-phone-1.png`) });
    const tap = (await inPane(page, 'rect[data-id]', `(el) => !el.getAttribute('data-id').includes('/') && el.getAttribute('data-id') !== '<main>'`)) || (await inPane(page, 'rect[data-id]'));
    if (tap) {
      await page.touchscreen.tap(tap.x, tap.y);
      await page.waitForTimeout(800);
      const got = await page.evaluate(() => {
        const t = document.querySelector('.pg-panel .pg-title');
        const g = document.querySelector('.pg-canvas').getBoundingClientRect();
        const r = t ? t.getBoundingClientRect() : null;
        return { sel: document.querySelector('.pg').dataset.sel, title: r ? r.top : -1, graphTop: g.top, graphBottom: g.bottom, vh: innerHeight };
      });
      ok(got.sel === tap.id && got.title > got.graphBottom && got.title < got.vh && got.graphTop >= 0, `${slug} (phone): a tap on ${tap.id} shows its code under the picture, which stays on the screen`);
      await page.screenshot({ path: join(SHOTS, `${slug}-phone-2-tap.png`) });
    } else ok(false, `${slug} (phone): a def's box to tap`);
    // the walk on a phone: its buttons stay on the screen, the stop's narration right under the picture
    await page.waitForFunction(() => !document.querySelector('.rv-wait'), null, { timeout: 120000 });
    await page.tap('.pg-tour .pg-run');
    await page.waitForTimeout(400);
    await page.tap('.pg-tour .pg-run');
    for (let i = 0; i < 3; i++) {
      await page.tap('.pg-tour .pg-tnext');
      await page.waitForTimeout(700);
    }
    const ph = await page.evaluate(() => {
      const bar = document.querySelector('.pg-tour').getBoundingClientRect();
      const g = document.querySelector('.pg-canvas').getBoundingClientRect();
      const t = document.querySelector('.pg-tourstop').getBoundingClientRect();
      return { bar: bar.top >= 0 && bar.bottom <= innerHeight, g: g.bottom, t: t.top, vh: innerHeight, wide: document.scrollingElement.scrollWidth };
    });
    ok(ph.bar && ph.t >= ph.g - 1 && ph.t < ph.vh * 0.8 && ph.wide <= 391, `${slug} (phone): during the walk the buttons are on the screen and the stop's narration starts right under the picture (${Math.round(ph.t)} px, the picture ends at ${Math.round(ph.g)} px)`);
    await page.screenshot({ path: join(SHOTS, `${slug}-phone-3-walk.png`) });
    await ctx.close();
  }
  ok(errors.length === 0, `no page errors${errors.length ? `: ${errors.slice(0, 3).join(' | ')}` : ''}`);
} finally {
  await browser.close();
  server.kill();
}
for (const t of timings) console.log(`time: ${t}`);
console.log(`graph-browsertest: ${checks - fails} of ${checks} checks pass; screenshots in ${SHOTS}`);
process.exit(fails ? 1 : 0);
