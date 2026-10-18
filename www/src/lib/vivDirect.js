// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The direct WebAssembly backend, in the tab: compile a Viv program to a wasm
// module with viv.wasm's `viv_emit_wasm` export, then run that module against
// the runtime viv.wasm also exports (lang/viv/src/interp/wasm_rt.rs).
//
//   const viv = await loadViv(bytesOrUrl);         // viv.wasm, instantiated once
//   const mod = compileDirect(viv, source);        // { bytes } or { error }
//   const r = await runDirect(viv, mod.bytes, { onOutput });
//   // r = { output, error, fails, holds, seconds }
//   // or both steps at once: await compileAndRun(viv, source, { onOutput })
//
// The compiled module imports viv.wasm's memory and its `vrt_*` functions,
// which is where arrays live and where every builtin runs (the interpreter's
// own code, so the output matches the interpreter's byte for byte). It also
// imports `host.flush`, called after each top-level statement: `onOutput`
// then receives the new text, so a long run shows its progress.
//
// Output on error follows `viv run`: the interpreter prints nothing but the
// error, so `output` is empty and `error` holds the message. `partial` keeps
// what the program printed before it stopped, for a page that wants to show it.
// tools/run-wasm.mjs runs a module the same way from node.

/** Instantiate viv.wasm from bytes (an ArrayBuffer/Uint8Array), a compiled WebAssembly.Module, or a URL. */
export async function loadViv(source) {
  let instance;
  if (source instanceof WebAssembly.Module) {
    instance = await WebAssembly.instantiate(source, {});
  } else if (typeof source === 'string' || source instanceof URL) {
    const res = await fetch(source);
    if (WebAssembly.instantiateStreaming && res.headers.get('content-type') === 'application/wasm') {
      ({ instance } = await WebAssembly.instantiateStreaming(res, {}));
    } else {
      ({ instance } = await WebAssembly.instantiate(await res.arrayBuffer(), {}));
    }
  } else {
    ({ instance } = await WebAssembly.instantiate(source, {}));
  }
  return instance.exports;
}

const text = (ex, ptr, len) => new TextDecoder().decode(new Uint8Array(ex.memory.buffer, ptr, len));

/**
 * Compile Viv source (with its imports already expanded) to a module.
 * Returns `{ bytes }` (a copy, safe to keep) or `{ error }`: the compiler's
 * message, e.g. what the direct backend does not compile yet.
 */
export function compileDirect(ex, source) {
  if (!ex.viv_emit_wasm) return { error: 'this build of viv.wasm has no direct wasm backend' };
  const enc = new TextEncoder().encode(source);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const packed = ex.viv_emit_wasm(ptr, enc.length);
  const p = Number(packed >> 32n);
  const n = Number(packed & 0xffffffffn);
  const bytes = new Uint8Array(ex.memory.buffer, p, n).slice();
  // a module starts with the magic \0asm, which no error message does
  if (bytes[0] === 0x00 && bytes[1] === 0x61 && bytes[2] === 0x73 && bytes[3] === 0x6d) return { bytes };
  const msg = new TextDecoder().decode(bytes);
  return { error: msg.startsWith('error: ') ? msg.slice(7) : msg };
}

/**
 * Run a compiled module. `onOutput(text)` (optional) receives output as the
 * program flushes it. Resolves when the program ends or stops with an error.
 */
export async function runDirect(ex, bytes, { onOutput } = {}) {
  const chunks = [];
  const flush = () => {
    const len = ex.vrt_out_len();
    if (!len) return;
    const s = text(ex, ex.vrt_out_ptr(), len);
    ex.vrt_out_clear();
    chunks.push(s);
    if (onOutput) onOutput(s);
  };
  const rt = { memory: ex.memory };
  for (const [name, f] of Object.entries(ex)) if (name.startsWith('vrt_')) rt[name] = f;
  // host.clock: imported only by a `#! timing` program (clock())
  const clock = () => performance.now() / 1000;
  const { instance } = await WebAssembly.instantiate(bytes, { rt, host: { flush, clock } });
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  let error = null;
  try {
    instance.exports.run();
    flush();
  } catch (e) {
    flush();
    const len = ex.vrt_err_len();
    // a message from the runtime is the interpreter's; anything else is a
    // trap of the module's own (a stack overflow from deep recursion, say)
    error = len ? text(ex, ex.vrt_err_ptr(), len) : String(e && e.message ? e.message : e);
  }
  const seconds = ((typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0) / 1000;
  const partial = chunks.join('');
  return {
    output: error ? '' : partial,
    partial,
    error,
    fails: error ? 0 : ex.vrt_fails(),
    holds: error ? 0 : ex.vrt_holds(),
    seconds,
  };
}

/**
 * Compile and run in one step. Resolves to `runDirect`'s result, or to
 * `{ error, compileError: true }` when the direct backend refused the program
 * (a page can then fall back to the interpreter, `viv_run`).
 */
export async function compileAndRun(ex, source, opts = {}) {
  const mod = compileDirect(ex, source);
  if (mod.error) return { output: '', partial: '', error: mod.error, compileError: true, fails: 0, holds: 0, seconds: 0 };
  return runDirect(ex, mod.bytes, opts);
}
