#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Checks the viewer's lenses (tools/viewer.mjs)
// by starting it on a free port and reading the HTML it serves for
// lang/viv/examples/lenses/scf_demo.viv:
//
//     node tools/viewer.test.mjs          (VIV=<path> picks the viv binary)
//
// It needs a branch whose worktree holds this checkout's commit, since the
// viewer reads a branch from git and its lenses from that branch's worktree;
// without one it prints `skip` and exits 0.
//
// The `l` key, the split and the scroll lock are page script. With
// CHROME=<path to a headless Chromium> they are driven in the browser over the
// DevTools protocol; without it only the anchors they move by are checked.
import { spawn, execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, chmodSync, realpathSync, rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import os from 'node:os';
import path from 'node:path';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const FILE = 'lang/viv/examples/lenses/scf_demo.viv';
const git = (...a) => execFileSync('git', ['-C', ROOT, ...a], { encoding: 'utf8' }).trim();

// the branch this checkout is on, else any branch on the same commit
let ref = '';
try {
  ref = git('symbolic-ref', '--short', 'HEAD');
} catch {
  const head = git('rev-parse', 'HEAD');
  ref = git('for-each-ref', '--format=%(objectname) %(refname:short)', 'refs/heads')
    .split('\n')
    .map((l) => l.split(' '))
    .filter(([sha]) => sha === head)
    .map(([, name]) => name)[0] || '';
}
if (!ref) {
  console.log('skip: no branch points at this commit, so the viewer has nothing to show it by');
  process.exit(0);
}

let failures = 0;
const check = (ok, what) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}`);
  if (!ok) failures++;
};

async function viewer(env) {
  const p = spawn(process.execPath, [path.join(ROOT, 'tools/viewer.mjs'), '0'], { env: { ...process.env, ...env }, stdio: ['ignore', 'pipe', 'inherit'] });
  const port = await new Promise((resolve, reject) => {
    let buf = '';
    p.stdout.on('data', (d) => {
      buf += d;
      const m = /localhost:(\d+)/.exec(buf);
      if (m) resolve(Number(m[1]));
    });
    p.on('exit', (c) => reject(new Error(`viewer exited with ${c}`)));
  });
  const get = async (u) => {
    const r = await fetch(`http://127.0.0.1:${port}${u}`);
    return { status: r.status, text: await r.text() };
  };
  return { get, base: `http://127.0.0.1:${port}`, stop: () => p.kill() };
}

const q = (o) => '/?' + new URLSearchParams({ ref, all: '1', file: FILE, ...o });
const lq = (o) => '/lens?' + new URLSearchParams({ ref, file: FILE, ...o });
const anchors = (html) => new Set([...html.matchAll(/data-a="([^"]*)"/g)].map((m) => m[1]));
// the lens panes of a page, without the File view around them
const panesOf = (html) => html.slice(html.indexOf('<div class="lenses'), html.indexOf('<script type="application/json" id="lens-data">'));

const v = await viewer({});
try {
  const file = await v.get(q({}));
  check(file.status === 200, 'the File view of a .viv file serves');
  check(['code', 'intent', 'debug', 'story'].every((l) => file.text.includes(`data-lens="${l}"`)), 'it has the four lens tabs');
  check(file.text.includes('<table class="code">') && !file.text.includes('<div class="lenses'), 'it still shows the file itself, not a lens');

  const other = await v.get('/?' + new URLSearchParams({ ref, all: '1', file: 'tools/viewer.mjs' }));
  // Look for a lens-tab link, not the bare attribute text: the file shown here is
  // the viewer's own source, which contains `data-lens=` as code, escaped.
  check(other.status === 200 && !/<a[^>]* data-lens="/.test(other.text), 'a file that is not .viv gets no lens tabs');

  // every lens has the anchor of the Fock step, which is what `l` keeps in view
  const pane = {};
  for (const lens of ['code', 'intent', 'debug', 'story']) {
    const r = await v.get(q({ lens }));
    pane[lens] = panesOf(r.text);
    check(r.status === 200 && pane[lens].includes(`data-lens="${lens}"`), `the ${lens} lens serves`);
    check(!pane[lens].includes('class="lnote"'), `the ${lens} lens has no warning`);
    check(anchors(pane[lens]).has('scf/fock'), `the ${lens} lens has the anchor scf/fock`);
  }
  // code: each row knows its line and its innermost def or step
  check(/<tr data-l="60" data-a="scf\/fock">/.test(pane.code), 'code: line 60, `step fock {`, is anchored at scf/fock');
  check(/<tr data-l="58" data-a="scf">/.test(pane.code), 'code: line 58, the while loop, belongs to scf itself');
  check(/<tr class="ic" data-l="15">/.test(pane.code), 'code: the intent comment on line 15 is marked, to be hidden');
  const want = ['fock', 'density', 'energy', 'diis', 'scf', 'scf/guess', 'scf/fock', 'scf/solve', 'scf/density', 'scf/mix', 'scf/energy', 'scf/check'];
  const intentA = [...anchors(pane.intent)];
  check(JSON.stringify(intentA) === JSON.stringify(want), `intent: anchors in file order (${intentA.join(' ')})`);
  check(pane.intent.includes('form the Fock matrix from the current density') && pane.intent.includes('placeholder: no body yet'), 'intent: step text and the placeholder mark');
  check(JSON.stringify([...anchors(pane.debug)]) === JSON.stringify(['scf', 'scf/fock', 'scf/density', 'scf/guess', 'scf/check']), 'debug: `hold at` and `watch at` lines anchor to their steps');
  // story: one line per iteration, the first loop iteration open
  check((pane.story.match(/<details class="sit"/g) || []).length === 33, 'story: 33 iterations');
  check((pane.story.match(/<details class="sit" open>/g) || []).length === 1, 'story: only the first iteration is open');
  check(pane.story.includes('holds: 99 checked, <span class="ok">0 failed</span>'), 'story: 99 holds checked, none failed');
  check(/<summary data-a="scf\/fock"><span class="lid">iteration 7<\/span>/.test(pane.story), 'story: an iteration line is anchored at its first step');

  const data = JSON.parse(/<script type="application\/json" id="lens-data">(.*?)<\/script>/s.exec((await v.get(q({ lens: 'intent' }))).text)[1]);
  check(JSON.stringify(data.order) === JSON.stringify(want), 'the page script gets the anchors in file order');
  check(data.ranges.some((r) => r.a === 'scf/fock' && r.line === 60 && r.end === 62), 'and the line range of each');

  const split = await v.get(q({ lens: 'code', split: 'story' }));
  check(split.text.includes('<div class="lenses two">') && (split.text.match(/<section class="lpane/g) || []).length === 2, 'split: two panes side by side');
  check(panesOf(split.text).includes('data-lens="code"') && panesOf(split.text).includes('data-lens="story"'), 'split: code beside story');

  const run = await v.get(lq({ lens: 'intent', order: 'run' }));
  check(run.status === 200 && anchors(run.text).has('scf/fock') && run.text.includes('lx-call'), 'intent in run order (viv explain) keeps the anchors');
  check((await v.get(lq({ lens: 'nope' }))).status === 400, '/lens refuses an unknown lens');
  check((await v.get('/lens?' + new URLSearchParams({ ref, file: 'tools/viewer.mjs', lens: 'code' }))).status === 400, '/lens refuses a file that is not .viv');
  check((await v.get('/hf')).status === 200, 'the Hartree-Fock map still serves');
  if (process.env.CHROME) await browser(v.base);
} finally {
  v.stop();
}

// In a real browser: rotate with `l`, split with `s`, scroll one pane and
// read which anchor sits at the top of each.
async function browser(base) {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const prof = mkdtempSync(path.join(os.tmpdir(), 'viv-viewer-chrome-'));
  const ch = spawn(process.env.CHROME, ['--headless', '--remote-debugging-port=0', `--user-data-dir=${prof}`, 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });
  try {
    const wsUrl = await new Promise((resolve) => {
      let buf = '';
      ch.stderr.on('data', (d) => {
        buf += d;
        const m = /DevTools listening on (ws:\S+)/.exec(buf);
        if (m) resolve(m[1]);
      });
    });
    const port = new URL(wsUrl).port;
    let page;
    for (let i = 0; i < 50 && !page; i++) {
      page = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === 'page');
      if (!page) await sleep(100);
    }
    const sock = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise((r) => (sock.onopen = r));
    let id = 0;
    const pend = new Map();
    const thrown = [];
    sock.onmessage = (m) => {
      const d = JSON.parse(m.data);
      if (d.id && pend.has(d.id)) (pend.get(d.id)(d), pend.delete(d.id));
      if (d.method === 'Runtime.exceptionThrown') thrown.push(d.params.exceptionDetails.text);
    };
    const send = (method, params = {}) => new Promise((r) => { pend.set(++id, r); sock.send(JSON.stringify({ id, method, params })); });
    const ev = async (expression) => (await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true })).result.result.value;
    await send('Runtime.enable');
    await send('Emulation.setDeviceMetricsOverride', { width: 1600, height: 1000, deviceScaleFactor: 1, mobile: false });
    const settle = async () => {
      await sleep(250);
      for (let i = 0; i < 300 && (await ev("!!document.querySelector('.lpane.loading')")); i++) await sleep(100);
      await sleep(250);
    };
    const nav = async (u) => {
      await send('Page.navigate', { url: base + u });
      for (let i = 0; i < 100 && (await ev('document.readyState')) !== 'complete'; i++) await sleep(100);
      await settle();
    };
    const key = async (k) => { await ev(`document.dispatchEvent(new KeyboardEvent('keydown', {key: '${k}', bubbles: true}))`); await settle(); };
    // each pane as lens:anchor, the anchor being that of the last visible anchored element at or above the pane's top
    const at = () => ev(`[...document.querySelectorAll('.lbody')].map((b) => { const top = b.getBoundingClientRect().top; let a = null; for (const e of b.querySelectorAll('[data-a]')) { if (!e.checkVisibility()) continue; if (e.getBoundingClientRect().top <= top + 1) a = e.dataset.a; else { a = a || e.dataset.a; break; } } return b.dataset.lens + ':' + a; }).join(' | ')`);
    const scrollTo = (pane, sel) => ev(`(() => { const b = document.querySelectorAll('.lbody')[${pane}]; const e = b.querySelectorAll('${sel}')[0]; b.scrollTop += e.getBoundingClientRect().top - b.getBoundingClientRect().top; })()`);

    await nav(q({ lens: 'code' }) + '#a=scf%2Ffock');
    check((await at()) === 'code:scf/fock', 'browser: #a=scf/fock opens the code lens on the Fock step');
    for (const lens of ['intent', 'debug', 'story', 'code']) {
      await key('l');
      check((await at()) === `${lens}:scf/fock`, `browser: l rotates to ${lens} with scf/fock in view`);
    }
    await key('L');
    check((await at()) === 'story:scf/fock', 'browser: L rotates back');
    await key('l');
    await key('s');
    check((await at()) === 'code:scf/fock | intent:scf/fock', 'browser: s splits, the new lens on the same step');
    await ev(`(() => { const s = document.querySelector('.lsplit-sel'); s.value = 'story'; s.dispatchEvent(new Event('change')); })()`);
    await settle();
    for (const a of ['scf/guess', 'scf/density', 'scf/check', 'energy']) {
      await scrollTo(0, `[data-a="${a}"]`);
      await sleep(300);
      check((await at()) === `code:${a} | story:${a}`, `browser: scrolling the code to ${a} brings the story to it`);
    }
    await ev(`(() => { const b = document.querySelectorAll('.lbody')[1]; const e = b.querySelectorAll('summary')[6]; b.scrollTop += e.getBoundingClientRect().top - b.getBoundingClientRect().top; })()`);
    await sleep(300);
    check((await at()) === 'code:scf/fock | story:scf/fock', 'browser: scrolling the story to iteration 7 brings the code to the loop');
    check((await ev('location.hash')) === '#a=scf%2Ffock', 'browser: the URL keeps the anchor');
    check(thrown.length === 0, `browser: no script errors ${JSON.stringify(thrown)}`);
    sock.close();
  } finally {
    ch.kill();
    await sleep(200);
    rmSync(prof, { recursive: true, force: true });
  }
}

// Without a binary, and with one built before the lenses, the File view works
// and each lens pane says what is missing.
const tmp = realpathSync(mkdtempSync(path.join(os.tmpdir(), 'viv-viewer-test-')));
const old = path.join(tmp, 'viv-old');
writeFileSync(old, '#!/bin/sh\necho "usage: viv <input.viv> [-o <output.c|output.wasm>]" >&2\nexit 1\n');
chmodSync(old, 0o755);
for (const [bin, says] of [
  [path.join(tmp, 'absent'), 'No viv binary at'],
  [old, 'it predates the lenses'],
]) {
  const w = await viewer({ VIV: bin });
  try {
    check((await w.get(q({}))).status === 200, `File view with ${path.basename(bin)}`);
    for (const lens of ['code', 'intent', 'debug', 'story']) {
      const p = panesOf((await w.get(q({ lens }))).text);
      check(p.includes(says), `${lens} lens with ${path.basename(bin)} says "${says}"`);
    }
    check(/<tr data-l="60">/.test(panesOf((await w.get(q({ lens: 'code' }))).text)), `the code lens still shows the code with ${path.basename(bin)}`);
  } finally {
    w.stop();
  }
}

rmSync(tmp, { recursive: true, force: true });
console.log(failures ? `${failures} failed` : 'all passed');
process.exit(failures ? 1 : 0);
