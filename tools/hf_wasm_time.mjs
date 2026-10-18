// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// hf_wasm_time -- time a Viv program the way a reader's tab runs it, and hold
// it to the feed's budget.
//
//     node tools/hf_wasm_time.mjs posts/2026-09-24-hartree-fock-water.viv
//     node tools/hf_wasm_time.mjs --runs 3 bench/hf_budget.viv
//
// The browser runs a post in viv.wasm, the interpreter compiled to
// WebAssembly, after expanding its imports. This tool does the same
// two steps with the same code the site build uses (www/src/lib/uses.js and
// www/public/viv/viv.wasm), so the seconds it prints are the seconds
// www/scripts/feed-build.mjs measures. The native binary is no guide here: a
// Hartree-Fock post that finishes in 0.05 s compiled can take seconds in the
// interpreter.
//
// It prints the best time over --runs runs (default 1) and exits 1 when that
// time is over the budget (--budget, default 10 s, the feed's BUDGET_SECONDS),
// when a test fails, or when no test ran. --print shows the program's output
// first. VIV_WASM=<path> times another build
// of viv.wasm (`npm run wasm` in www/ writes the default one).
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { dirname, join, resolve as resolvePath } from 'node:path';
import { fileURLToPath } from 'node:url';
import { expandUses } from '../www/src/lib/uses.js';
import { BUDGET_SECONDS } from '../www/src/lib/postCheck.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const TARGET_SECONDS = 5.0;

const args = process.argv.slice(2);
let runs = 1;
let budget = BUDGET_SECONDS;
let file = null;
let print = false;
for (let i = 0; i < args.length; i += 1) {
  if (args[i] === '--runs') runs = Number(args[(i += 1)]);
  else if (args[i] === '--budget') budget = Number(args[(i += 1)]);
  else if (args[i] === '--print') print = true;
  else file = args[i];
}
if (!file || !(runs >= 1) || !(budget > 0)) {
  console.error('usage: node tools/hf_wasm_time.mjs [--runs N] [--budget SECONDS] [--print] <program.viv>');
  process.exit(2);
}
const path = resolvePath(process.cwd(), file);
if (!existsSync(path)) {
  console.error(`hf_wasm_time: no such file: ${file}`);
  process.exit(2);
}
const wasmPath = process.env.VIV_WASM || join(ROOT, 'www/public/viv/viv.wasm');
if (!existsSync(wasmPath)) {
  console.error('hf_wasm_time: no viv.wasm; build it with `npm run wasm` in www/, or set VIV_WASM');
  process.exit(2);
}

// Module names resolve the way feed-build resolves them: lib/, pde/, dft/,
// examples/ (lang/viv/examples) and post:<slug>.
const DIRS = { dft: 'dft', pde: 'pde', lib: 'lib', examples: 'lang/viv/examples' };
const resolveModule = async (name) => {
  if (name.startsWith('post:')) {
    const slug = name.slice(5);
    const hit = readdirSync(join(ROOT, 'posts')).find((f) => f.replace(/^\d{4}-\d{2}-\d{2}-/, '') === `${slug}.viv`);
    if (!hit) throw new Error(`module \`${name}\`: no post by that slug in posts/`);
    return readFileSync(join(ROOT, 'posts', hit), 'utf8');
  }
  const [head, ...rest] = name.split('/');
  const f = DIRS[head] && rest.length ? join(ROOT, DIRS[head], `${rest.join('/')}.viv`) : null;
  if (!f || !existsSync(f)) throw new Error(`module \`${name}\`: no module by that name`);
  return readFileSync(f, 'utf8');
};

let expanded;
try {
  expanded = await expandUses(readFileSync(path, 'utf8'), resolveModule);
} catch (e) {
  console.error(`hf_wasm_time: ${file}: ${e.message}`);
  process.exit(1);
}
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const ex = instance.exports;
function call(fn, text) {
  const enc = new TextEncoder().encode(text);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const packed = fn(ptr, enc.length);
  return new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
}

let best = Infinity;
let out = '';
for (let r = 0; r < runs; r += 1) {
  const t0 = performance.now();
  out = call(ex.viv_run, expanded);
  best = Math.min(best, (performance.now() - t0) / 1000);
}
const lines = out.split('\n');
// --print: the program's own output, as the reader's tab would show it
if (print) for (const l of lines) if (l && !/^__(graph|trace)__/.test(l)) console.log(l);
const pass = lines.filter((l) => l.startsWith('PASS ')).length;
const fail = lines.filter((l) => l.startsWith('FAIL ') || l.startsWith('error')).length;
const verdict = best > budget ? 'OVER BUDGET' : best > TARGET_SECONDS ? 'within budget, over target' : 'within target';
console.log(`${best.toFixed(2)} s  ${file}  (best of ${runs}; budget ${budget.toFixed(1)} s, target ${TARGET_SECONDS.toFixed(1)} s: ${verdict}; ${pass} tests pass, ${fail} failed)`);
if (fail) for (const l of lines.filter((x) => x.startsWith('FAIL ') || x.startsWith('error')).slice(0, 6)) console.log(`  ${l.slice(0, 170)}`);
process.exit(best > budget || fail || !pass ? 1 : 0);
