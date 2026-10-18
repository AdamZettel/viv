// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Runs a claim's program off the page's thread (src/components/ClaimPost.jsx),
// so a run that takes minutes in the interpreter leaves the page scrolling and
// its Stop button working. The page posts its compiled viv.wasm module with
// each run; one instance is kept for the worker's life, and Stop ends the
// worker.
//
//   in:  { id, src, vivModule }
//   out: { id, out, ms }  |  { id, error }
import { runProgram } from './runProgram.js';
import { attachFiles } from './programFiles.js';

let ready = null;

self.onmessage = async (e) => {
  const { id, src, vivModule } = e.data;
  try {
    if (!ready) ready = WebAssembly.instantiate(vivModule, {}).then((instance) => instance.exports);
    const ex = await ready;
    await attachFiles(ex, src);
    const t0 = performance.now();
    const out = await runProgram(ex, src);
    self.postMessage({ id, out, ms: performance.now() - t0 });
  } catch (err) {
    self.postMessage({ id, error: String((err && err.message) || err) });
  }
};
