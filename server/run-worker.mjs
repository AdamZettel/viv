// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// One program, one worker. Runs it and reports its output; the parent decides
// what the output means and terminates this thread either way.
import { parentPort, workerData } from 'node:worker_threads';
import { ladderView } from '../www/src/lib/ladderView.js';

const { module, src, ladder: wantLadder = false } = workerData;
const instance = await WebAssembly.instantiate(module, {});
const ex = instance.exports;
const put = (text) => {
  const enc = new TextEncoder().encode(text);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  return [ptr, enc.length];
};
const take = (packed) => new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
const t0 = performance.now();
let out;
try {
  out = take(ex.viv_run(...put(src)));
} catch (e) {
  // a wasm trap: most often an allocation past the memory maximum (run.mjs)
  parentPort.postMessage({ trap: String(e && e.message || e), memoryBytes: ex.memory.buffer.byteLength });
  process.exit(0);
}
const seconds = (performance.now() - t0) / 1000;
const hash = take(ex.viv_hash(...put(src)));
let functions = [];
try { functions = JSON.parse(take(ex.viv_doc(...put(src)))).functions.map((f) => ({ name: f.name, what: f.what || '' })); } catch { /* a program that does not parse has no docs */ }
// a ladder: every rung run and reported, and the view a page draws
let ladder = null;
if (wantLadder) {
  const text = take(ex.viv_ladder(...put(src)));
  if (text.startsWith('error')) ladder = { error: text, rungs: [], view: null };
  else {
    const report = JSON.parse(text);
    const docOf = (program) => { try { return JSON.parse(take(ex.viv_doc(...put(program)))); } catch { return null; } };
    const view = ladderView(report, docOf);
    for (const r of view.rungs) delete r.output;
    ladder = { rungs: report.rungs.map((r) => ({ n: r.n, title: r.title, broken: r.broken })), view: { top: view.top, rungs: view.rungs } };
  }
}
parentPort.postMessage({ out, seconds, hash, functions, ladder });
