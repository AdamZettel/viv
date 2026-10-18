// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The Netlify Function's run worker: server/run-worker.mjs with a limit on how
// much a run may print. build-function.mjs bundles this file as run-worker.mjs.
//
// The memory cap (wasmcap.mjs) bounds the interpreter, but a program that stays
// under it can still print a long output (a plot of two million points is about
// 100 MB of text). run-worker.mjs decodes that output into a string, the main
// thread receives a copy, and summary.mjs splits it into lines: several times the
// output's size in the function's heap, which is enough to stop a 1024 MB
// invocation with no answer. So the interpreter's outputs are measured where they
// are still bytes in wasm memory, and one over the limit is replaced by an
// `error` line saying so. summary.mjs and postCheck.js both report an `error`
// line as the program's error.
//
// It wraps the exports run-worker.mjs gets from WebAssembly.instantiate and then
// loads run-worker.mjs unchanged, so the server and the function run programs
// with the same worker code.
export const OUTPUT_LIMIT_BYTES = 8 * 1024 * 1024;

const MB = 1024 * 1024;
const MEASURED = ['viv_run', 'viv_ladder']; // the outputs that grow with what the program prints
const instantiate = WebAssembly.instantiate.bind(WebAssembly);

WebAssembly.instantiate = async (...args) => {
  const result = await instantiate(...args);
  const instance = result.instance || result;
  const ex = instance.exports;
  const wrapped = { ...ex };
  for (const name of MEASURED) {
    if (typeof ex[name] !== 'function') continue;
    wrapped[name] = (...a) => {
      const packed = ex[name](...a);
      const bytes = Number(packed & 0xffffffffn);
      if (bytes <= OUTPUT_LIMIT_BYTES) return packed;
      const text = new TextEncoder().encode(`error: the program printed ${(bytes / MB).toFixed(1)} MB of output, and a run on this endpoint may print at most ${OUTPUT_LIMIT_BYTES / MB} MB. Print less: a figure carries every point it plots, so plot fewer points (a few thousand draw a smooth curve), or run the program at https://viv-lang.org/workbench\n`);
      const ptr = ex.viv_alloc(text.length);
      new Uint8Array(ex.memory.buffer, ptr, text.length).set(text);
      return (BigInt(ptr) << 32n) | BigInt(text.length);
    };
  }
  const swapped = { exports: wrapped };
  return result.instance ? { module: result.module, instance: swapped } : swapped;
};

await import('../run-worker.mjs');
