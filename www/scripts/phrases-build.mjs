// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The phrase table on the reference page, read from the compiler.
//
// viv.wasm's `viv_phrases` returns the table the parser matches against
// (lang/viv/src/phrases.rs): every pattern, its group and the Viv expression it
// lowers to. This script adds, for each pattern, an example from the generated
// test program lang/viv/tests/phrases/phrases.viv (one `test` per phrase, each
// against the value NumPy or a closed form gives), and runs that program in the
// same viv.wasm to record which tests passed. So the table cannot drift from the
// parser: a pattern added, removed or reworded changes the page on the next
// build, and a pattern with no passing example fails the build.
//
// Writes www/src/data/phrases.json.

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const wasmBytes = readFileSync(join(WWW, 'public/viv/viv.wasm'));
const { instance } = await WebAssembly.instantiate(wasmBytes, {});
const ex = instance.exports;
if (!ex.viv_phrases) {
  console.error('phrases-build: this viv.wasm has no viv_phrases (rebuild it with www/build-wasm.sh)');
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

const table = JSON.parse(call(ex.viv_phrases, ''));
const casesPath = join(ROOT, 'lang/viv/tests/phrases/phrases.viv');
const cases = readFileSync(casesPath, 'utf8');
const names = [...cases.matchAll(/^test "(.*)" \{$/gm)].map((m) => m[1]);
const out = call(ex.viv_run, cases);
const passed = new Set([...out.matchAll(/^PASS (.*)$/gm)].map((m) => m[1]));

// a pattern as a regular expression over a phrase as written
function patternRe(p) {
  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const parts = p.words.split(/\s+/).map((w) => {
    if (w.startsWith('{')) return '(.+)';
    if (w.includes('|')) return `(?:${w.split('|').map(esc).join('|')})`;
    return w.split('-').map(esc).join(' ?- ?');
  });
  return new RegExp(`^${p.bare ? '' : 'take '}${parts.join(' ')}$`);
}

const missing = [];
const rows = table.patterns.map((p) => {
  const re = patternRe(p);
  // the shortest test the pattern reads: a longer one may be a longer pattern's
  const example = names.filter((n) => re.test(n)).sort((a, b) => a.length - b.length)[0] || null;
  if (!example || !passed.has(example)) missing.push(p.phrase);
  // the lowering as the phrase column writes holes: `{v}` is V, `{f'}` is F'
  const call = p.lowers.replace(/\{([A-Za-z0-9_]+)('?)\}/g, (_, n, d) => n.toUpperCase() + d).replace(/\{([^}]*\|[^}]*)\}/g, '$1');
  return { ...p, call, example, passed: example ? passed.has(example) : false };
});

const groups = [];
for (const r of rows) {
  let g = groups.find((x) => x.title === r.group);
  if (!g) groups.push((g = { title: r.group, rows: [] }));
  g.rows.push(r);
}

writeFileSync(
  join(WWW, 'src/data/phrases.json'),
  JSON.stringify(
    {
      groups,
      count: table.count,
      knot: table.knot,
      verbs: table.verbs,
      std: table.std,
      tests: names.length,
      passed: names.filter((n) => passed.has(n)).length,
    },
    null,
    2
  ) + '\n'
);

console.log(`phrases-build: ${table.count} patterns (${table.knot} from Knot), ${passed.size} of ${names.length} example tests pass in viv.wasm`);
if (missing.length) {
  console.error(`phrases-build: no passing example for: ${missing.join('; ')}`);
  process.exit(1);
}
