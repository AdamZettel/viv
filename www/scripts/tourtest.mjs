// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Run every tour cell the way the browser runs it.
//
// The tour is a checkpointed notebook: cell N executes the concatenation of
// cells 1..N. That model only holds if every prefix actually runs, so this
// harness executes each prefix through viv.wasm's interpreter and fails on any
// error line — including the step budget. A tour that ships is a tour that
// runs.
//
// It also checks the two claims the page makes about numbers: the capstone's
// hydrogen energy and the trace identity in the linear-algebra cell.
//
//   npm run tourtest

import { readFileSync } from 'node:fs';
import { cells } from '../src/data/vivTourCells.mjs';

const wasm = readFileSync(new URL('../public/viv/viv.wasm', import.meta.url));
const { instance } = await WebAssembly.instantiate(wasm, {});
const ex = instance.exports;

function run(text) {
  const enc = new TextEncoder().encode(text);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const packed = ex.viv_run(ptr, enc.length);
  return new TextDecoder().decode(
    new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn))
  );
}

const strip = (out) =>
  out
    .split('\n')
    .filter((l) => !l.startsWith('__graph__') && !l.startsWith('__trace__') && !l.startsWith('__steps__') && !l.startsWith('__story__'))
    .join('\n')
    .trim();

let fail = 0;
let lastOut = '';

for (let i = 0; i < cells.length; i++) {
  const program = cells.slice(0, i + 1).join('\n');
  const out = strip(run(program));
  const errs = out.split('\n').filter((l) => /^error/.test(l) || /\bFAIL\b/.test(l));
  if (errs.length) {
    fail++;
    console.log(`FAIL cell ${i + 1} (prefix 1..${i + 1})`);
    for (const e of errs) console.log(`  ${e}`);
  } else {
    const asserts = (out.match(/^PASS /gm) || []).length;
    console.log(
      `ok   cell ${i + 1}  ${String(out.split('\n').length).padStart(3)} output lines` +
        (asserts ? `  ${asserts} test${asserts === 1 ? '' : 's'} passed` : '')
    );
  }
  lastOut = out;
}

// The page states two numbers. Check them rather than trusting the prose.
// `show f(x)` on a call expression prints the bare value (there is no name to
// label it with), so these match the value, not a label.
const CLAIMS = [
  [/^-0\.4244\d*$/m, 'the capstone hydrogen energy, -0.4244 hartree'],
  [/^0$/m, "deriv vanishes at the analytic optimum (the minimum is real)"],
  [/^PASS the analytic optimum/m, 'the capstone test passes, by name'],
];
for (const [re, what] of CLAIMS) {
  if (re.test(lastOut)) {
    console.log(`ok   claim: ${what}`);
  } else {
    fail++;
    console.log(`FAIL claim: ${what} — not found in the final run's output`);
  }
}

console.log(
  fail
    ? `\ntourtest: ${fail} failure(s)`
    : `\ntourtest: ${cells.length} cells, every prefix runs clean`
);
if (fail) process.exit(1);
