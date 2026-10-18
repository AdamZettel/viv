// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Runs Viv programs off the page's thread, for the composer's live preview: a
// program that takes 200 ms would otherwise freeze typing for 200 ms on every
// keystroke. One interpreter instance, kept; each message is one run.
//
//   in:  { id, src, ladder? }          ladder: also run every rung (viv_ladder)
//   out: { id, out, seconds, ladder? }  |  { id, error }
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
  const { id, src, ladder: wantLadder = false } = e.data;
  try {
    const ex = await boot();
    const t0 = performance.now();
    const out = call(ex, ex.viv_run, src);
    const seconds = (performance.now() - t0) / 1000;
    let ladder = null;
    if (wantLadder) {
      const text = call(ex, ex.viv_ladder, src);
      if (text.startsWith('error')) ladder = { error: text, rungs: [] };
      else {
        ladder = JSON.parse(text);
        // the docs of each rung's program, for the preview's ledger
        ladder.docs = ladder.rungs.map((r) => { try { return JSON.parse(call(ex, ex.viv_doc, r.program)); } catch { return null; } });
      }
    }
    self.postMessage({ id, out, seconds, ladder });
  } catch (err) {
    self.postMessage({ id, error: String(err && err.message || err) });
  }
};
