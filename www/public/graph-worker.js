// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Redraws the program graph for /explore, off the page's thread: viv.wasm's
// interpreter runs lib/graphdraw's `draw_view` on the view the page asks for
// (src/lib/graphView.js writes the program) and returns the figure records.
// One interpreter instance, loaded when the worker starts.
//
//   in:  { id, src }            the drawing library with the view's call appended
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
  const { id, src } = e.data;
  try {
    const ex = await boot();
    const t0 = performance.now();
    const out = call(ex, ex.viv_run_release, src);
    if (out.startsWith('error')) throw new Error(out.slice(0, 400));
    self.postMessage({ id, out, ms: performance.now() - t0 });
  } catch (err) {
    self.postMessage({ id, error: String((err && err.message) || err) });
  }
};
