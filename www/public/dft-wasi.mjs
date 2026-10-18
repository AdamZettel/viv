// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Run a standalone WebAssembly module built from Viv-generated C, and return
// what it printed.
//
// dft/build-wasm.sh compiles each program Viv -> C -> wasm with emscripten's
// STANDALONE mode, so the module needs no JavaScript glue: it imports a handful
// of WASI calls (printf ends in fd_write; main's return ends in proc_exit) and
// one notification when its memory grows. That is all this file provides.
//
// Used by the page's worker (public/dft-worker.js) and, from node, by
// scripts/dfttest.mjs and scripts/dft-precompute.mjs -- one runner, so the
// build, the test and the reader's tab execute a module the same way.
class Exit extends Error {
  constructor(code) { super(`exit ${code}`); this.code = code; }
}

export async function runWasi(bytes) {
  let memory;
  let text = '';
  const dec = new TextDecoder();
  const view = () => new DataView(memory.buffer);
  const zero2 = (a, b) => { const v = view(); v.setUint32(a, 0, true); v.setUint32(b, 0, true); return 0; };
  const wasi = {
    fd_write(fd, iovs, n, nwritten) {
      const v = view();
      let total = 0;
      for (let i = 0; i < n; i++) {
        const ptr = v.getUint32(iovs + 8 * i, true);
        const len = v.getUint32(iovs + 8 * i + 4, true);
        text += dec.decode(new Uint8Array(memory.buffer, ptr, len));
        total += len;
      }
      v.setUint32(nwritten, total, true);
      return 0;
    },
    fd_close: () => 0,
    fd_seek: () => 70, // ESPIPE: stdout is not seekable
    fd_read: () => 0,
    fd_fdstat_get: () => 0,
    environ_sizes_get: zero2,
    environ_get: () => 0,
    args_sizes_get: zero2,
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
  const seconds = (performance.now() - t0) / 1000;
  return { text, seconds, code, memoryMB: memory.buffer.byteLength / 1048576 };
}
