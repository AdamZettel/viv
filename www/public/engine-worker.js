// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Runs an engine off the page's thread: a DFT calculation takes seconds, and on
// the page's own thread the tab would freeze for exactly as long.
//
//   in:  { mode: 'compiled', wasm: '/engines/dft-bands.wasm', stdin: Uint8Array }
//        the compiled engine, the reader's deck on its standard input
//   in:  { mode: 'interpreted', program: '/engines/hf-energy.viv', block: '...' }
//        viv.wasm reading the engine's source with its input block replaced
//   out: { line }                            each line as the compiled engine prints it
//        { done: true, text, seconds, memoryMB }
//        { error }
import { runEngine } from '/engine-wasi.mjs';

const VIV_WASM_URL = new URL(self.location.href).searchParams.get('wasm') || '/viv/viv.wasm';
const instanceOf = async (viv) => (viv instanceof WebAssembly.Module
  ? WebAssembly.instantiate(viv, {})
  : (await WebAssembly.instantiate(viv, {})).instance);

const get = async (url, as) => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`could not load ${url} (${r.status})`);
  return as === 'text' ? r.text() : r.arrayBuffer();
};

// The same replacement as src/lib/engines/report.js withInput(): from the line
// starting "# ---- input:" through the line starting "# ---- end of input".
function withInput(program, block) {
  const a = program.search(/^# ---- input:/m);
  const endAt = program.search(/^# ---- end of input.*$/m);
  if (a < 0 || endAt < 0) throw new Error('this engine program has no input block');
  const b = program.indexOf('\n', endAt);
  return program.slice(0, a) + block.replace(/\n?$/, '\n') + program.slice(b + 1);
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.mode === 'compiled') {
      const bytes = await get(msg.wasm);
      const r = await runEngine(bytes, { stdin: msg.stdin, onLine: (line) => self.postMessage({ line }) });
      self.postMessage({ done: true, text: r.text, seconds: r.seconds, memoryMB: r.memoryMB });
      return;
    }
    // the page's compiled viv.wasm when it sent one (src/lib/vivWasm.js), else fetched
    const [viv, program] = await Promise.all([msg.vivModule || get(VIV_WASM_URL), get(msg.program, 'text')]);
    const src = withInput(program, msg.block);
    const instance = await instanceOf(viv);
    const ex = instance.exports;
    const enc = new TextEncoder().encode(src);
    const ptr = ex.viv_alloc(enc.length);
    new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
    self.postMessage({ line: 'narrate: reading the program' });
    const t0 = performance.now();
    const packed = ex.viv_run(ptr, enc.length);
    const seconds = (performance.now() - t0) / 1000;
    const text = new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
    self.postMessage({ done: true, text, seconds, memoryMB: ex.memory.buffer.byteLength / 1048576 });
  } catch (err) {
    self.postMessage({ error: String((err && err.message) || err) });
  }
};
