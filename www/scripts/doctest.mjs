// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The Viv doctest harness: every example in src/data/vivDocExamples.json runs
// against the shipped interpreter (www/public/viv/viv.wasm) and its output must
// match the recorded `expect` exactly (error examples match `expectError` as a
// substring). The same corpus renders the /docs pages — the docs and the
// test suite are one artifact, so a documented behavior cannot silently drift.
//
//   npm run doctest           # verify (exit 1 on any mismatch)
//   npm run doctest --bless   # record current outputs as expected
import { readFileSync, writeFileSync } from 'node:fs';

const CORPUS = new URL('../src/data/vivDocExamples.json', import.meta.url);
const WASM = new URL('../public/viv/viv.wasm', import.meta.url);

const corpus = JSON.parse(readFileSync(CORPUS, 'utf8'));
const { instance } = await WebAssembly.instantiate(readFileSync(WASM), {});
const ex = instance.exports;

function run(text) {
  const enc = new TextEncoder().encode(text);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const packed = ex.viv_run(ptr, enc.length);
  return new TextDecoder().decode(
    new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
}

// The call-graph lines vary with the example's structure and are the graph
// panel's concern, not the documented output — strip them before comparing.
const normalize = (out) =>
  out.split('\n').filter((l) => !l.startsWith('__graph__') && !l.startsWith('__trace__') && !l.startsWith('__steps__') && !l.startsWith('__story__'))
    .join('\n').trim();

const bless = process.argv.includes('--bless');
let pass = 0, fail = 0, blessed = 0;

for (const e of corpus.examples) {
  const out = normalize(run(e.code));
  const isError = out.startsWith('error:');
  const id = `${e.topic} / ${e.title}`;

  if (e.expectError !== undefined) {
    if (isError && out.includes(e.expectError)) { pass++; continue; }
    fail++;
    console.log(`FAIL ${id}\n  want error containing: ${e.expectError}\n  got: ${out.split('\n')[0]}`);
    continue;
  }
  if (isError) {
    fail++;
    console.log(`FAIL ${id}\n  unexpected error: ${out.split('\n')[0]}`);
    continue;
  }
  if (bless) {
    if (e.expect !== out) { e.expect = out; blessed++; }
    continue;
  }
  if (e.expect === out) { pass++; continue; }
  fail++;
  console.log(`FAIL ${id}\n  want: ${JSON.stringify(e.expect)}\n  got:  ${JSON.stringify(out)}`);
}

if (bless) {
  writeFileSync(CORPUS, JSON.stringify(corpus, null, 2) + '\n');
  console.log(`blessed ${blessed} example(s); corpus rewritten`);
} else {
  console.log(`doctest: ${pass} passed, ${fail} failed, ${corpus.examples.length} total`);
  if (fail) process.exit(1);
}
