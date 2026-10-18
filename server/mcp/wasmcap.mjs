// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// A memory limit for the interpreter, set in the module itself.
//
// viv.wasm declares its memory with a minimum and no maximum, so a program
// can grow it to wasm32's 4 GB: ten 15-million-element arrays in one run
// took the process to 1.0 GB resident (measured 2026-09-24). A worker's
// resourceLimits cap the JavaScript heap, not wasm memory. So the MCP server
// compiles its own copy of the module with a maximum written into the memory
// section; growing past it fails inside the interpreter, which traps, and the
// run ends with an error instead of taking the host's memory.
//
// Only the limits of the one defined memory change. Everything else in the
// binary is copied byte for byte.

const PAGE = 65536;

function readLeb(bytes, pos) {
  let result = 0;
  let shift = 0;
  let byte;
  do {
    byte = bytes[pos++];
    result += (byte & 0x7f) * 2 ** shift;
    shift += 7;
  } while (byte & 0x80);
  return { value: result, pos };
}

function leb(value) {
  const out = [];
  do {
    let byte = value & 0x7f;
    value = Math.floor(value / 128);
    if (value) byte |= 0x80;
    out.push(byte);
  } while (value);
  return out;
}

/** The module's bytes with its memory's maximum set to `maxBytes` (rounded down to whole pages). */
export function capMemory(bytes, maxBytes) {
  const maxPages = Math.floor(maxBytes / PAGE);
  let pos = 8; // magic and version
  while (pos < bytes.length) {
    const id = bytes[pos];
    const size = readLeb(bytes, pos + 1);
    const bodyStart = size.pos;
    const bodyEnd = bodyStart + size.value;
    if (id === 5) {
      const count = readLeb(bytes, bodyStart);
      if (count.value !== 1) throw new Error(`the module defines ${count.value} memories; expected one`);
      const flags = bytes[count.pos];
      if (flags & ~1) throw new Error(`the memory's limits use flags 0x${flags.toString(16)} (shared or 64-bit), which this does not rewrite`);
      const min = readLeb(bytes, count.pos + 1);
      const oldMax = flags & 1 ? readLeb(bytes, min.pos).value : Infinity;
      if (min.value > maxPages) throw new Error(`the module starts with ${min.value * PAGE} bytes of memory, over the ${maxBytes}-byte limit`);
      const body = [...leb(1), 0x01, ...leb(min.value), ...leb(Math.min(oldMax, maxPages))];
      const section = [5, ...leb(body.length), ...body];
      const out = new Uint8Array(pos + section.length + (bytes.length - bodyEnd));
      out.set(bytes.subarray(0, pos), 0);
      out.set(section, pos);
      out.set(bytes.subarray(bodyEnd), pos + section.length);
      return out;
    }
    pos = bodyEnd;
  }
  throw new Error('the module defines no memory (an imported memory would be capped where it is created)');
}
