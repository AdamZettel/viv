// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The explorer (/explore) in a real browser.
//
//     npm run build && npm run explorebrowsertest
//     EXPLORE_SHOTS=/some/dir npm run explorebrowsertest    (keep screenshots there)
//
// Against the built site, served by scripts/serve-dist.mjs on port 4398:
//   1. /explore lists every program in src/data/explorables.mjs.
//   2. On the headline program (MINPACK on Rosenbrock), in dark and in light:
//      the run is recorded in the tab; Next and Back move one position and
//      back; one keypress moves past a loop of 16 passes; Enter opens it and
//      the next position is its first pass; at a step, the graph lights that
//      step's box and the intent pane shows that step's text from the bundle;
//      the recording of every pass stays small; the notebook lists the steps in the order
//      they ran and takes you back to the timeline.
//   2b. Play: Play moves a stop at a time at the chosen speed and stops at the
//      end; Back, Next, a click or a key pauses it; Space plays and pauses;
//      the speed is remembered; with reduced motion the panes do not scroll.
//   3. At 390 px wide, in both themes: no horizontal scroll.
//   4. Water RHF: the SCF loop is one position with 14 passes, and the
//      recording of a run with millions of integral evaluations stays under
//      5,000 events.
//   5. The small GPT, whose gradient is `value_and_grad`: the gradient is one
//      box, "gradient of sequence_loss", that calls the forward pass's defs,
//      the run's call graph goes through them, the timeline goes down into
//      the gradient to a block's step, and every position of the run can be
//      reached to the last.
//   6. No page errors anywhere.
// A program with intent opens in the Read view; each test here switches to
// the Timeline first. The Read view has its own test, read-browsertest.mjs.
import { mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { chromium } from 'playwright-core';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const PORT = Number(process.env.EXPLORE_PORT || 4398);
const BASE = `http://127.0.0.1:${PORT}`;
const SHOTS = process.env.EXPLORE_SHOTS || join(WWW, 'explore-shots');
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

const browser = await chromium.launch();
const errors = [];
async function open(path, { theme = 'dark', width = 1400 } = {}) {
  const ctx = await browser.newContext({ viewport: { width, height: 1000 }, colorScheme: theme });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => errors.push(`${path} (${theme}, ${width}): ${e}`));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(`${path} (${theme}, ${width}): console: ${m.text()}`);
  });
  await page.goto(BASE + path);
  // a program with intent opens in the Read view (scripts/read-browsertest.mjs); this test is about the timeline
  if (path.startsWith('/explore/')) {
    await page.waitForSelector('button[role=tab]:text("Timeline")', { timeout: 60000 });
    await page.click('button[role=tab]:text("Timeline")');
    // the keys go to the page, not to the tab just clicked
    await page.evaluate(() => document.activeElement && document.activeElement.blur());
  }
  return { ctx, page };
}
const where = (page) => page.$eval('.ex-where b', (e) => e.textContent);
const events = (page) => page.$eval('.ex-recorded', (e) => Number(e.textContent.match(/([\d,]+) events/)[1].replace(/,/g, '')));
const noHScroll = (page) => page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth);

try {
  // ---- 1. the index ------------------------------------------------------
  {
    const { ctx, page } = await open('/explore');
    const n = await page.$$eval('.ex-index li', (ls) => ls.length);
    ok(n >= 4, `/explore lists the programs (${n})`);
    await ctx.close();
  }

  // ---- 2. the headline program, both themes --------------------------------
  for (const theme of ['dark', 'light']) {
    const { ctx, page } = await open('/explore/minpack-rosenbrock', { theme });
    await page.waitForSelector('.ex-grid', { timeout: 120000 });
    const bundle = await page.evaluate(() => fetch('/explore/minpack-rosenbrock.json').then((r) => r.json()));
    ok((await page.getAttribute('html', 'data-theme')) === theme, `${theme}: the page is in ${theme} mode`);
    const bg = await page.$eval('.ex-pane', (e) => getComputedStyle(e).backgroundColor);
    ok(theme === 'dark' ? bg !== 'rgb(255, 255, 255)' : bg === 'rgb(255, 255, 255)', `${theme}: panes take the theme's card color (${bg})`);
    const first = await where(page);
    await page.keyboard.press('ArrowRight');
    const second = await where(page);
    await page.keyboard.press('ArrowLeft');
    ok(first !== second && (await where(page)) === first, `${theme}: Next moves (${first} -> ${second}) and Back returns`);

    // by step to the loop
    let atLoop = false;
    for (let k = 0; k < 20 && !atLoop; k++) {
      await page.keyboard.press('Shift+ArrowRight');
      atLoop = /passes$/.test(await where(page));
    }
    ok(atLoop, `${theme}: stepping by step reaches the loop (${await where(page)})`);
    ok(/16 passes/.test(await where(page)), `${theme}: the loop is one position, with its 16 passes`);
    ok((await page.$$('.ex-gloop.here')).length === 1, `${theme}: the graph lights the loop's box`);
    // the passes worth reading are a short list, not most of the loop
    const notable = await page.$$eval('.ex-notable > li', (els) => els.map((e) => e.textContent.trim()));
    console.log(`     ${theme}: passes worth reading: ${notable.join(' | ')}`);
    ok(notable.length >= 2 && notable.length <= 6, `${theme}: the loop lists ${notable.length} of 16 passes as worth reading (2 to 6)`);
    await page.screenshot({ path: join(SHOTS, `rosenbrock-loop-${theme}.png`), fullPage: true });
    await page.keyboard.press('ArrowRight');
    const after = await where(page);
    const afterBox = await page.$eval('.ex-gbox.here .ex-gbox-head b', (e) => e.textContent).catch(() => null);
    ok(afterBox === 'finish' && !(await page.$eval('.ex-path', (e) => e.textContent)).includes('pass'), `${theme}: one keypress moves past the whole loop, to the finish step (${after})`);
    await page.keyboard.press('ArrowLeft');
    await page.keyboard.press('Enter');
    await page.keyboard.press('ArrowRight');
    ok(/^pass 1 of 16$/.test(await where(page)), `${theme}: Enter opens the loop, and the next position is its first pass (${await where(page)})`);
    const intentLoop = await page.$eval('.ex-intent-pane', (e) => e.textContent);
    ok(/this is pass 1/.test(intentLoop) && /Passes worth reading/.test(intentLoop), `${theme}: the intent pane summarizes the loop and the passes worth reading`);

    // into the first pass, step by step: graph and intent follow
    await page.keyboard.press('Shift+ArrowRight');
    const crumbs = await page.$$eval('.ex-path li', (ls) => ls.map((l) => l.textContent));
    const stepId = crumbs[crumbs.length - 1];
    const lit = await page.$eval('.ex-gbox.here .ex-gbox-head b', (e) => e.textContent).catch(() => null);
    ok(lit === stepId, `${theme}: at step ${stepId}, the graph lights its box (${lit})`);
    const text = bundle.steps[`lmder/${stepId}`];
    const shown = await page.$eval('.ex-steptext .ex-big', (e) => e.textContent);
    ok(!!text && shown === text, `${theme}: the intent pane shows the step's text: "${shown}"`);
    const codeCur = await page.$$eval('.ex-cl.cur', (ls) => ls.length);
    ok(codeCur >= 1, `${theme}: the code pane highlights the step's lines`);
    await page.keyboard.press('ArrowRight');
    const lineLabel = await where(page);
    ok(/line \d+$/.test(lineLabel), `${theme}: Next goes into the step, one line at a time (${lineLabel})`);
    const vals = await page.$$eval('.ex-value', (ls) => ls.length);
    ok(vals > 0, `${theme}: the values pane shows the watched values (${vals})`);
    await page.screenshot({ path: join(SHOTS, `rosenbrock-line-${theme}.png`), fullPage: true });

    // the next pass: skipping to it with ] does not run anything again
    const before = await events(page);
    await page.keyboard.press(']');
    ok(/^pass \d+ of 16$/.test(await where(page)) && (await events(page)) === before, `${theme}: ] moves to the next pass shown (${await where(page)}), from the same recording`);

    // a run this small is recorded whole, every pass and trial line by line
    // (the Read view's pass selector reads them), and stays small
    const n = await events(page);
    ok(n < 8000, `${theme}: the recording of every pass stays small (${n} events)`);
    ok(/every pass line by line/.test(await page.$eval('.ex-recorded', (e) => e.textContent)), `${theme}: the page says the run was recorded whole`);

    // clicking a box in the graph jumps to that step
    await page.click('.ex-gbox:has(.ex-gbox-head b:text-is("finish")) .ex-gbox-head');
    ok((await page.$eval('.ex-gbox.here .ex-gbox-head b', (e) => e.textContent)) === 'finish', `${theme}: clicking a box in the graph goes to that step`);

    // the notebook
    await page.click('button[role=tab]:text("Notebook")');
    const cells = await page.$$eval('.ex-cell', (cs) => cs.length);
    ok(cells > 10, `${theme}: the notebook has a cell per step, in the order they ran (${cells})`);
    await page.screenshot({ path: join(SHOTS, `rosenbrock-notebook-${theme}.png`), fullPage: true });
    await page.click('.ex-cell:not(.ex-cell-call):not(.ex-cell-loop) button');
    ok((await page.$$('.ex-grid')).length === 1, `${theme}: a cell's button goes back to the timeline at that step`);
    await ctx.close();
  }

  // ---- 2b. Play ---------------------------------------------------------------
  {
    const { ctx, page } = await open('/explore/minpack-rosenbrock', { theme: 'light' });
    await page.waitForSelector('.ex-grid', { timeout: 120000 });
    const pos = () => page.$eval('.ex-where .ex-quiet', (e) => Number(e.textContent.match(/position (\d+)/)[1]));
    const pressed = () => page.getAttribute('.ex-playbtn', 'aria-pressed');
    ok((await page.$eval('.ex-speed', (e) => e.value)) === '1', 'Play: the speed starts at 1x');
    await page.selectOption('.ex-speed', '4');
    const p0 = await pos();
    await page.click('.ex-playbtn');
    ok((await pressed()) === 'true' && (await page.textContent('.ex-playbtn')).trim() === 'Pause', 'Play: the button turns into Pause');
    await page.waitForTimeout(1500);
    const p1 = await pos();
    ok(p1 > p0, `Play: it moves on by itself (${p0} -> ${p1} in 1.5 s at 4x)`);
    await page.click('.ex-gbox-head');
    ok((await pressed()) === 'false', 'Play: a click in the graph pauses it');
    const p2 = await pos();
    await page.waitForTimeout(800);
    ok((await pos()) === p2, 'Play: paused, it stays put');
    await page.evaluate(() => document.activeElement && document.activeElement.blur());
    await page.keyboard.press('Space');
    ok((await pressed()) === 'true', 'Play: Space plays');
    await page.keyboard.press('ArrowRight');
    ok((await pressed()) === 'false', 'Play: a key pauses it');
    await page.click('.ex-playbtn');
    await page.click('button:text-is("Back")');
    ok((await pressed()) === 'false', 'Play: Back pauses it');
    // the end: from the last stop Play starts over; reaching the end stops it
    await page.keyboard.press('End');
    const lastPos = await pos();
    await page.keyboard.press('ArrowLeft');
    await page.click('.ex-playbtn');
    await page.waitForTimeout(2500);
    ok((await pressed()) === 'false' && (await pos()) === lastPos, `Play: it stops at the last stop (${await pos()} of ${lastPos})`);
    await page.reload();
    await page.waitForSelector('button[role=tab]:text("Timeline")', { timeout: 60000 });
    await page.click('button[role=tab]:text("Timeline")');
    await page.waitForSelector('.ex-speed');
    ok((await page.$eval('.ex-speed', (e) => e.value)) === '4', 'Play: the speed is remembered');
    await ctx.close();
  }
  {
    const ctx = await browser.newContext({ viewport: { width: 1400, height: 1000 }, reducedMotion: 'reduce' });
    const page = await ctx.newPage();
    page.on('pageerror', (e) => errors.push(`reduced motion: ${e}`));
    await page.goto(BASE + '/explore/minpack-rosenbrock');
    await page.waitForSelector('button[role=tab]:text("Timeline")', { timeout: 60000 });
    await page.click('button[role=tab]:text("Timeline")');
    await page.waitForSelector('.ex-grid', { timeout: 120000 });
    const tops = () => page.evaluate(() => [document.querySelector('.ex-graph-pane').scrollTop, document.querySelector('.ex-code-pane .ex-code, .ex-code-pane > div')?.scrollTop || 0]);
    const before = await tops();
    await page.selectOption('.ex-speed', '4');
    await page.click('.ex-playbtn');
    await page.waitForTimeout(2000);
    const after = await tops();
    await page.click('.ex-playbtn');
    ok(before[0] === after[0], `Play, reduced motion: the graph pane does not scroll (${before[0]} -> ${after[0]})`);
    await ctx.close();
  }

  // ---- 3. phone width ------------------------------------------------------
  for (const theme of ['dark', 'light']) {
    const { ctx, page } = await open('/explore/minpack-rosenbrock', { theme, width: 390 });
    await page.waitForSelector('.ex-grid', { timeout: 120000 });
    for (let k = 0; k < 5; k++) await page.keyboard.press('Shift+ArrowRight');
    await page.keyboard.press('Enter');
    for (let k = 0; k < 6; k++) await page.keyboard.press('ArrowRight');
    ok(await noHScroll(page), `390 px, ${theme}: no horizontal scroll`);
    await page.screenshot({ path: join(SHOTS, `rosenbrock-390-${theme}.png`), fullPage: true });
    await page.click('button[role=tab]:text("Notebook")');
    ok(await noHScroll(page), `390 px, ${theme}: no horizontal scroll in the notebook`);
    await ctx.close();
    const idx = await open('/explore', { theme, width: 390 });
    ok(await noHScroll(idx.page), `390 px, ${theme}: /explore has no horizontal scroll`);
    await idx.ctx.close();
  }

  // ---- 4. water RHF --------------------------------------------------------
  {
    const { ctx, page } = await open('/explore/water-rhf');
    await page.waitForSelector('.ex-grid', { timeout: 180000 });
    let found = false;
    for (let k = 0; k < 40 && !found; k++) {
      await page.keyboard.press('Shift+ArrowRight');
      found = /passes$/.test(await where(page));
    }
    ok(found && /14 passes/.test(await where(page)), `water: the SCF loop is one position with its 14 passes (${await where(page)})`);
    const wnotable = await page.$$eval('.ex-notable > li', (els) => els.map((e) => e.textContent.trim()));
    console.log(`     water: passes worth reading: ${wnotable.join(' | ')}`);
    ok(wnotable.length >= 2 && wnotable.length <= 6, `water: the SCF loop lists ${wnotable.length} of 14 passes as worth reading (2 to 6)`);
    const n = await events(page);
    ok(n < 5000, `water: the recording of the whole run stays under 5,000 events (${n})`);
    await page.screenshot({ path: join(SHOTS, 'water-loop-dark.png'), fullPage: true });
    await ctx.close();
  }

  // ---- 5. the small GPT: a gradient's forward pass is its own defs --------
  {
    const { ctx, page } = await open('/explore/tiny-gpt');
    await page.waitForSelector('.ex-grid', { timeout: 180000 });
    const bundle = await page.evaluate(() => fetch('/explore/tiny-gpt.json').then((r) => r.json()));
    const grad = bundle.nodes.find((n) => n.id === 'viv__ad_grad_sequence_loss');
    ok(!!grad && grad.label === 'gradient of sequence_loss' && !grad.stepped, 'tiny-gpt: the gradient is one def, labeled "gradient of sequence_loss", with no steps of its own');
    ok(bundle.nodes.some((n) => n.kind === 'step' && n.def === 'block_forward') && bundle.nodes.some((n) => n.kind === 'step' && n.def === 'attention'), 'tiny-gpt: block_forward and attention keep their steps');
    const ran = new Set(bundle.calls.map(([a, b]) => `${a}>${b}`));
    for (const e of ['train>viv__ad_grad_sequence_loss', 'viv__ad_grad_sequence_loss>sequence_loss', 'sequence_loss>gpt_forward', 'gpt_forward>block_forward', 'block_forward>self_attention', 'self_attention>attention']) {
      ok(ran.has(e), `tiny-gpt: the run's call graph has ${e.replace('>', ' -> ')}`);
    }
    const boxes = new Set((bundle.drawing.match(/^box: [^ ;]+/gm) || []).map((l) => l.slice(5)));
    ok(['viv__ad_grad_sequence_loss', 'sequence_loss', 'gpt_forward', 'block_forward'].every((d) => boxes.has(d)), `tiny-gpt: the picture draws the gradient and the forward defs (${[...boxes].filter((b) => !b.includes('/')).join(', ')})`);
    ok(bundle.drawing.includes('gradient of sequence_loss'), 'tiny-gpt: the picture labels the gradient box "gradient of sequence_loss"');
    // to the training loop, open it, and from its first pass into the
    // gradient, down to a step of a block
    for (let k = 0; k < 40 && !/^for t to steps/.test(await where(page)); k++) await page.keyboard.press('Shift+ArrowRight');
    ok(/^for t to steps: 30 passes/.test(await where(page)), `tiny-gpt: the training loop is one position with its 30 passes (${await where(page)})`);
    await page.keyboard.press('Enter');
    await page.keyboard.press('ArrowRight');
    let crumbs = [];
    for (let k = 0; k < 80; k++) {
      // gpt_forward's loop over the blocks is one position too: open it
      if (/^for layer to/.test(await where(page))) await page.keyboard.press('Enter');
      await page.keyboard.press('Shift+ArrowRight');
      crumbs = await page.$$eval('.ex-path li', (ls) => ls.map((l) => l.textContent));
      if (crumbs.includes('gradient of sequence_loss') && crumbs.includes('block_forward') && crumbs.includes('attn_norm')) break;
    }
    ok(crumbs.includes('gradient of sequence_loss') && crumbs.includes('block_forward') && crumbs.includes('attn_norm'), `tiny-gpt: the timeline goes into the gradient and down to a block's step (${crumbs.join(' > ')})`);
    const graphHead = await page.$eval('.ex-graph-pane .ex-h', (e) => e.textContent);
    const lit = await page.$eval('.ex-gbox.here .ex-gbox-head b', (e) => e.textContent).catch(() => null);
    ok(/block_forward/.test(graphHead) && lit === 'attn_norm', `tiny-gpt: the graph pane shows block_forward with attn_norm lit (${graphHead}; ${lit})`);
    await page.screenshot({ path: join(SHOTS, 'tiny-gpt-gradient-dark.png'), fullPage: true });
    // every position, to the last
    await page.keyboard.press('End');
    const last = await page.$eval('.ex-where', (e) => e.textContent);
    const m = last.match(/position ([\d,]+) of ([\d,]+)/);
    ok(!!m && m[1] === m[2], `tiny-gpt: the run can be followed to its last position (${last.trim()})`);
    await page.keyboard.press('Home');
    let moved = 0;
    for (let k = 0; k < 3000; k++) {
      await page.keyboard.press('ArrowRight');
      moved++;
      const w = await page.$eval('.ex-where', (e) => e.textContent);
      const p = w.match(/position ([\d,]+) of ([\d,]+)/);
      if (p && p[1] === p[2]) break;
    }
    const end = await page.$eval('.ex-where', (e) => e.textContent);
    const q = end.match(/position ([\d,]+) of ([\d,]+)/);
    ok(!!q && q[1] === q[2], `tiny-gpt: one position at a time, the walk reaches the end in ${moved} moves (${end.trim()})`);
    await page.screenshot({ path: join(SHOTS, 'tiny-gpt-end-dark.png'), fullPage: true });
    await ctx.close();
  }
} catch (e) {
  ok(false, `the test ran to the end: ${e.message}`);
}

ok(errors.length === 0, `no page errors${errors.length ? `:\n  ${errors.join('\n  ')}` : ''}`);
await browser.close();
server.kill();
console.log(`\nexplorebrowsertest: ${checks - fails} of ${checks} checks passed; screenshots in ${SHOTS}`);
process.exit(fails ? 1 : 0);
