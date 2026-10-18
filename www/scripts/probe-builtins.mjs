// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Measure the builtin surface instead of transcribing it.
//
// For every candidate below, compile and run a one-line program through
// viv.wasm and record what each engine did: accepted, rejected-with-a-message,
// or crashed. The reference page renders THIS output, so the tables cannot
// drift from the compiler — if a builtin is added, removed or renamed, the page
// changes on the next build without anyone editing prose.
//
// Writes www/src/data/builtins.json.

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const wasmBytes = readFileSync(join(WWW, 'public/viv/viv.wasm'));
const { instance } = await WebAssembly.instantiate(wasmBytes, {});
const ex = instance.exports;

function call(fn, text) {
  const enc = new TextEncoder().encode(text);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const packed = fn(ptr, enc.length);
  return new TextDecoder().decode(
    new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn))
  );
}

// The rows are the compiler's: lang/viv/features/builtins.txt, which the
// compiler carries and `viv_features --json` returns as `builtin_groups` (one
// row per form: name, signature, what it does, and `probe`, a full program that
// must run clean if the builtin exists). tests/features.rs fails when a builtin
// the compiler's tables know has no row, and runs every probe in three engines.
if (!ex.viv_features) {
  console.error('probe-builtins: this viv.wasm has no viv_features (rebuild it with www/build-wasm.sh)');
  process.exit(1);
}
const GROUPS = JSON.parse(call(ex.viv_features, '--json')).builtin_groups;

const MAGIC = [0x00, 0x61, 0x73, 0x6d];
function probeWasmBackend(src) {
  const enc = new TextEncoder().encode(src);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const packed = ex.viv_emit_wasm(ptr, enc.length);
  const bytes = new Uint8Array(
    ex.memory.buffer,
    Number(packed >> 32n),
    Number(packed & 0xffffffffn)
  );
  return MAGIC.every((b, i) => bytes[i] === b);
}

let ok = 0;
let partial = 0;
const groups = GROUPS.map((g) => ({
  title: g.title,
  note: g.note,
  items: g.items.map(([name, sig, blurb, probe]) => {
    const interp = call(ex.viv_run, probe);
    const c = call(ex.viv_compile, probe);
    const iOk = !/^error/m.test(interp);
    const cOk = !c.startsWith('error');
    const wOk = probeWasmBackend(probe);
    if (iOk && cOk) ok++;
    else if (iOk || cOk) partial++;
    return {
      name,
      sig,
      blurb,
      probe,
      interp: iOk,
      c: cOk,
      wasm: wOk,
      interpErr: iOk ? null : interp.trim().split('\n')[0].slice(0, 160),
      cErr: cOk ? null : c.trim().split('\n')[0].slice(0, 160),
    };
  }),
}));

const total = groups.reduce((s, g) => s + g.items.length, 0);
const missing = groups.flatMap((g) =>
  g.items.filter((i) => !i.interp && !i.c).map((i) => i.name)
);

writeFileSync(
  join(WWW, 'src/data/builtins.json'),
  JSON.stringify(
    {
      groups,
      total,
      bothEngines: ok,
      oneEngine: partial,
      wasmBackend: groups.reduce((s, g) => s + g.items.filter((i) => i.wasm).length, 0),
      missing,
      probedAt: new Date().toISOString().slice(0, 10),
    },
    null,
    2
  )
);

console.log(
  `probed ${total} builtins: ${ok} in both engines, ${partial} in one, ` +
    `${groups.reduce((s, g) => s + g.items.filter((i) => i.wasm).length, 0)} through the direct wasm backend` +
    (missing.length ? `\n  NOT FOUND in either engine: ${missing.join(', ')}` : '')
);
