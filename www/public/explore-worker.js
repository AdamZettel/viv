// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Records a program's run for /explore, off the page's thread: viv.wasm's
// `viv_explore_run` runs the program in the interpreter, in debug mode, with
// line-level recording on (lang/viv/src/graph.rs). One interpreter instance,
// kept; each message is one recording.
//
//   in:  { id, src, iters, calls, defs, every }   iters / calls: { def: [n, ...] } passes and
//        calls to record line by line besides the first; defs: defs without
//        steps to record line by line; every: defs whose nested loops are
//        recorded in every pass
//   out: { id, out, seconds }   |  { id, error }
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
const call = (ex, fn, text) => {
  const enc = new TextEncoder().encode(text);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const packed = fn(ptr, enc.length);
  return new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
};
self.onmessage = async (e) => {
  const { id, src, iters = {}, calls = {}, defs = [], every = [] } = e.data;
  try {
    const ex = await boot();
    if (!ex.viv_explore_run) throw new Error('this viv.wasm cannot record for the explorer (it has no viv_explore_run)');
    const t0 = performance.now();
    const out = call(ex, ex.viv_explore_run, JSON.stringify({ src, iters, calls, defs, every }));
    self.postMessage({ id, out, seconds: (performance.now() - t0) / 1000 });
  } catch (err) {
    self.postMessage({ id, error: String((err && err.message) || err) });
  }
};
