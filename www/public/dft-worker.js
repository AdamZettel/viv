// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Runs one material's plane-wave DFT program off the main thread (a
// self-consistent calculation takes seconds; on the page's own thread that
// would freeze the tab for exactly as long). Two engines, the reader's choice:
//
//   compiled     /dft/<mat>.wasm  -- the program itself, compiled Viv -> C -> wasm
//   interpreted  /viv/viv.wasm    -- the Viv interpreter, handed /dft/<mat>.viv to read
//
//   in:  { mat: 'si', engine: 'compiled' | 'interpreted' }
//   out: { ok: true, text, seconds, engine }  |  { ok: false, error }
import { runWasi } from '/dft-wasi.mjs';

const VIV_WASM_URL = new URL(self.location.href).searchParams.get('wasm') || '/viv/viv.wasm';
const instanceOf = async (viv) => (viv instanceof WebAssembly.Module
  ? WebAssembly.instantiate(viv, {})
  : (await WebAssembly.instantiate(viv, {})).instance);

const get = async (url, as) => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`could not load ${url} (${r.status})`);
  return as === 'text' ? r.text() : r.arrayBuffer();
};

self.onmessage = async (e) => {
  const { mat, engine } = e.data;
  try {
    if (engine === 'compiled') {
      const bytes = await get(`/dft/${mat}.wasm`);
      self.postMessage({ started: true });
      const { text, seconds } = await runWasi(bytes);
      self.postMessage({ ok: true, text, seconds, engine });
      return;
    }
    // the page's compiled viv.wasm when it sent one (src/lib/vivWasm.js), else fetched
    const [viv, src] = await Promise.all([e.data.vivModule || get(VIV_WASM_URL), get(`/dft/${mat}.viv`, 'text')]);
    const instance = await instanceOf(viv);
    const ex = instance.exports;
    const enc = new TextEncoder().encode(src);
    const ptr = ex.viv_alloc(enc.length);
    new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
    self.postMessage({ started: true });
    const t0 = performance.now();
    const packed = ex.viv_run(ptr, enc.length);
    const seconds = (performance.now() - t0) / 1000;
    const text = new TextDecoder().decode(
      new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
    self.postMessage({ ok: true, text, seconds, engine });
  } catch (err) {
    self.postMessage({ ok: false, error: String((err && err.message) || err) });
  }
};
