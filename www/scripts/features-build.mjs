// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The feature catalog on the reference page, read from the compiler.
//
// viv.wasm's `viv_features` returns the catalog `viv features --json` prints
// (lang/viv/src/features.rs): the hand-written entries of lang/viv/features/
// with their examples and recorded outputs, and the entries generated from the
// compiler's tables. tests/features.rs runs every example in three engines, so
// the page shows what the compiler checked. A catalog with problems fails the
// build.
//
// Writes www/src/data/features.json, and www/src/data/vivKeywords.json: the
// lexer's keywords and the parser's soft keywords (features.rs KEYWORDS), which
// the highlighter (src/lib/highlight.js) colors.

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const wasmBytes = readFileSync(join(WWW, 'public/viv/viv.wasm'));
const { instance } = await WebAssembly.instantiate(wasmBytes, {});
const ex = instance.exports;
if (!ex.viv_features) {
  console.error('features-build: this viv.wasm has no viv_features (rebuild it with www/build-wasm.sh)');
  process.exit(1);
}

function call(fn, text) {
  const enc = new TextEncoder().encode(text);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const packed = fn(ptr, enc.length);
  return new TextDecoder().decode(
    new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn))
  );
}

const cat = JSON.parse(call(ex.viv_features, '--json'));
if (cat.errors.length) {
  console.error(`features-build: the catalog has problems:\n  ${cat.errors.join('\n  ')}`);
  process.exit(1);
}

const hand = cat.entries.filter((e) => e.hand);
const areas = cat.areas
  .filter((a) => hand.some((e) => e.area === a.key))
  .map((a) => ({
    key: a.key,
    heading: a.heading,
    what: a.what,
    entries: hand
      .filter((e) => e.area === a.key)
      .map((e) => ({
        id: e.id,
        title: e.title,
        what: e.what,
        when: e.when,
        doc: e.doc,
        see: e.see,
        verb: e.verb,
        example: e.examples[0] ? { code: e.examples[0].code, output: e.examples[0].output, cmd: e.examples[0].cmd } : null,
        noExample: e.no_example,
      })),
  }));
const generated = cat.areas
  .filter((a) => !hand.some((e) => e.area === a.key) && a.count > 0)
  .map((a) => ({ key: a.key, heading: a.heading, what: a.what, count: a.count }));

writeFileSync(
  join(WWW, 'src/data/features.json'),
  JSON.stringify({ count: cat.count, hand: cat.hand, generated: cat.generated, areas, generatedAreas: generated }, null, 2) + '\n'
);
const keywords = cat.entries.filter((e) => e.area === 'keywords').map((e) => e.keywords[0] || e.id.replace(/^keyword-/, ''));
writeFileSync(join(WWW, 'src/data/vivKeywords.json'), JSON.stringify({ source: 'lang/viv/src/features.rs KEYWORDS', keywords }, null, 1) + '\n');
console.log(`features-build: ${cat.count} entries, ${cat.hand} written by hand in ${areas.length} areas, ${cat.generated} generated`);
