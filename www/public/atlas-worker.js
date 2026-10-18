// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Builds a program's atlas in the reader's browser, off the page's thread
// (src/lib/atlas.js): viv.wasm's `viv_explore_bundle` makes the explorer
// bundle from the program and the module texts the page fetched (the code
// `viv graph --bundle` runs, lang/viv/src/graph.rs), and `viv_run_release`
// runs a program plainly, for the run's call graph or for lib/graphdraw's
// picture. One interpreter instance per worker; the page ends a run that
// goes on too long by ending the worker.
//
//   in:  { id, op: 'bundle' | 'run', text }   text: the bundle request (JSON) or the program
//   out: { id, out, ms }  |  { id, error }
// viv.wasm: a page that made this worker with src/lib/vivWasm.js vivWorker()
// posts its compiled module first ({ vivModule }), so the page fetches the
// compiler once however many workers it starts; otherwise this fetches it.
const VIV_Q = new URL(self.location.href).searchParams;
const VIV_WASM_URL = VIV_Q.get('wasm') || '/viv/viv.wasm';
let takeModule = null;
const pageModule = VIV_Q.has('module') ? new Promise((r) => (takeModule = r)) : Promise.resolve(null);
self.addEventListener('message', (e) => {
  if (e.data && 'vivModule' in e.data && takeModule) {
    e.stopImmediatePropagation();
    takeModule(e.data.vivModule);
    takeModule = null;
  }
});
const vivExports = () => pageModule.then((m) => (m
  ? WebAssembly.instantiate(m, {})
  : fetch(VIV_WASM_URL).then((r) => r.arrayBuffer()).then((bytes) => WebAssembly.instantiate(bytes, {})).then(({ instance }) => instance))
).then((instance) => instance.exports);
let ready = null;
function boot() {
  if (!ready) {
    ready = vivExports();
  }
  return ready;
}
boot();
const call = (ex, fn, text) => {
  const enc = new TextEncoder().encode(text);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const packed = fn(ptr, enc.length);
  return new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
};
self.onmessage = async (e) => {
  const { id, op, text } = e.data;
  try {
    const ex = await boot();
    const fn = op === 'bundle' ? ex.viv_explore_bundle : ex.viv_run_release;
    if (!fn) throw new Error(`this viv.wasm has no ${op === 'bundle' ? 'viv_explore_bundle' : 'viv_run_release'}`);
    const t0 = performance.now();
    const out = call(ex, fn, text);
    self.postMessage({ id, out, ms: performance.now() - t0 });
  } catch (err) {
    self.postMessage({ id, error: String((err && err.message) || err) });
  }
};
