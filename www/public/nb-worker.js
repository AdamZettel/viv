// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Runs a notebook's cells off the page's thread. One interpreter instance per
// notebook, kept for the life of the page, because the interpreter's
// incremental session lives in it: when a cell's prefix is exactly what the
// session last ran, only the cell itself executes (lang/viv/src/interp.rs,
// run_cell). Stopping a run terminates this worker, and the session with it;
// the page starts a new one, and its first run rebuilds the prefix.
//
//   in:  { id, prefix, cell }      the cells above, joined by "\n"; this cell
//   out: { id, out, seconds }  |  { id, error }
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
self.onmessage = async (e) => {
  const { id, prefix, cell } = e.data;
  try {
    const ex = await boot();
    const enc = new TextEncoder();
    const pb = enc.encode(prefix);
    const cb = enc.encode(cell);
    const ptr = ex.viv_alloc(pb.length + cb.length);
    const mem = new Uint8Array(ex.memory.buffer, ptr, pb.length + cb.length);
    mem.set(pb, 0);
    mem.set(cb, pb.length);
    const t0 = performance.now();
    const packed = ex.viv_run_cell(ptr, pb.length + cb.length, pb.length);
    const seconds = (performance.now() - t0) / 1000;
    const out = new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
    self.postMessage({ id, out, seconds });
  } catch (err) {
    self.postMessage({ id, error: String((err && err.message) || err) });
  }
};
