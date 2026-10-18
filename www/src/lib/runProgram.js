// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Run a program for a Run button, and return what the interpreter's
// `viv_run_keep` returns: the output, and on an error what was printed before
// it followed by the `error: ...` line.
//
// A `#! timing` program calls clock(), and the interpreter compiled to
// viv.wasm has no clock to read. The direct WebAssembly backend imports one
// from the page (host.clock in vivDirect.js), and its runtime is the
// interpreter's own code, so such a program is compiled and run that way. If
// the direct backend refuses the program, the interpreter runs it and reports
// the missing clock itself.
import { compileDirect, runDirect } from './vivDirect.js';

const TIMING_MARK = /^#! timing\s*$/m; // lang/viv/src/lib.rs is_timing

/** The text a Run shows for `src`, from the viv.wasm exports `ex`. */
export async function runProgram(ex, src) {
  if (TIMING_MARK.test(src) && ex.viv_emit_wasm) {
    const mod = compileDirect(ex, src);
    if (!mod.error) {
      const r = await runDirect(ex, mod.bytes);
      if (!r.error) return r.output;
      const head = r.partial && !r.partial.endsWith('\n') ? `${r.partial}\n` : r.partial;
      return `${head}error: ${r.error}`;
    }
  }
  const fn = ex.viv_run_keep || ex.viv_run;
  const enc = new TextEncoder().encode(src);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const packed = fn(ptr, enc.length);
  const text = new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
  // viv_run_keep separates the kept output from the error with a NUL
  const i = text.indexOf('\0');
  if (i < 0) return text;
  const head = text.slice(0, i);
  return head + (head && !head.endsWith('\n') ? '\n' : '') + text.slice(i + 1);
}
