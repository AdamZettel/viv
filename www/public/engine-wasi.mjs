// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Run a compiled engine: a standalone WebAssembly module built from
// Viv-generated C (engines/build-wasm.mjs), with the reader's input deck on
// its standard input and its printout streamed back line by line.
//
// public/dft-wasi.mjs runs the per-material modules, which read nothing. An
// engine reads its deck, so this runner also answers fd_read on fd 0 from a
// byte array. Everything else is the same handful of WASI calls a printf needs.
//
// Used by public/engine-worker.js in the tab and by www/scripts/enginetest.mjs
// in node: one runner, so the test and the reader's tab run a module the same way.
class Exit extends Error {
  constructor(code) { super(`exit ${code}`); this.code = code; }
}

/**
 * runEngine(bytes, { stdin, onLine }) -> { text, seconds, code, memoryMB }
 *   stdin   Uint8Array handed to the module on fd 0 (the deck)
 *   onLine  called with each complete line the module prints, as it prints it
 */
export async function runEngine(bytes, { stdin = new Uint8Array(0), onLine = null } = {}) {
  let memory;
  let text = '';
  let partial = '';
  let at = 0;
  const dec = new TextDecoder();
  const view = () => new DataView(memory.buffer);
  const wasi = {
    fd_write(fd, iovs, n, nwritten) {
      const v = view();
      let total = 0;
      let chunk = '';
      for (let i = 0; i < n; i++) {
        const ptr = v.getUint32(iovs + 8 * i, true);
        const len = v.getUint32(iovs + 8 * i + 4, true);
        chunk += dec.decode(new Uint8Array(memory.buffer, ptr, len));
        total += len;
      }
      if (fd === 1) {
        text += chunk;
        if (onLine) {
          const lines = (partial + chunk).split('\n');
          partial = lines.pop();
          for (const l of lines) onLine(l);
        }
      }
      v.setUint32(nwritten, total, true);
      return 0;
    },
    fd_read(fd, iovs, n, nread) {
      const v = view();
      let total = 0;
      if (fd === 0) {
        for (let i = 0; i < n && at < stdin.length; i++) {
          const ptr = v.getUint32(iovs + 8 * i, true);
          const len = Math.min(v.getUint32(iovs + 8 * i + 4, true), stdin.length - at);
          new Uint8Array(memory.buffer, ptr, len).set(stdin.subarray(at, at + len));
          at += len;
          total += len;
        }
      }
      v.setUint32(nread, total, true);
      return 0;
    },
    fd_close: () => 0,
    fd_seek: () => 70, // ESPIPE: neither stream is seekable
    fd_fdstat_get: () => 0,
    environ_sizes_get: (a, b) => { const v = view(); v.setUint32(a, 0, true); v.setUint32(b, 0, true); return 0; },
    environ_get: () => 0,
    args_sizes_get: (a, b) => { const v = view(); v.setUint32(a, 0, true); v.setUint32(b, 0, true); return 0; },
    args_get: () => 0,
    clock_time_get(id, precision, out) { view().setBigUint64(out, BigInt(Date.now()) * 1000000n, true); return 0; },
    proc_exit(code) { throw new Exit(code); },
  };
  const { instance } = await WebAssembly.instantiate(bytes, {
    wasi_snapshot_preview1: wasi,
    env: { emscripten_notify_memory_growth: () => {} },
  });
  memory = instance.exports.memory;
  const t0 = performance.now();
  let code = 0;
  try {
    instance.exports._start();
  } catch (e) {
    if (!(e instanceof Exit)) throw e;
    code = e.code;
  }
  if (partial && onLine) onLine(partial);
  const seconds = (performance.now() - t0) / 1000;
  return { text, seconds, code, memoryMB: memory.buffer.byteLength / 1048576 };
}
