// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The plane-wave DFT code through BOTH of the browser's engines, exactly as a
// tab runs them:
//
//   compiled     dft/wasm/<mat>.wasm  the program itself, Viv -> C -> wasm (dft/build-wasm.sh)
//   interpreted  viv.wasm             the Viv interpreter, reading the .viv source
//
// Every material must finish, pass all of its own tests on both, and do it in
// a time a person will wait.
//
//     node scripts/dfttest.mjs                 the five that the page offers, both engines
//     node scripts/dfttest.mjs si al           just these
//     node scripts/dfttest.mjs --compiled cu   copper, compiled only (minutes)
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { runWasi } from '../public/dft-wasi.mjs';
import { expandUses } from '../src/lib/uses.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..', '..');
const DFT = join(root, 'dft');
const args = process.argv.slice(2);
const compiledOnly = args.includes('--compiled');
const wanted = args.filter((a) => !a.startsWith('--'));
const mats = wanted.length ? wanted : ['si', 'al', 'ge', 'c', 'gaas'];
const BUDGET_S = { compiled: 60, interpreted: 120 };

// what `viv run --use deck --use kmesh --use common --use engine check` does: concatenate,
// and expand the engine's `from lobpcg import lobpcg` as the compiler does
const fromRepo = async (name) => readFileSync(join(root, `${name}.viv`), 'utf8');
const source = (mat) => expandUses([`${mat}.viv`, 'kmesh.viv', 'pw_common.viv', 'pwdft_it.viv', `${mat}_it_check.viv`].map((f) => readFileSync(join(DFT, f), 'utf8')).join('\n'), fromRepo);

async function interpreted(mat) {
  const { instance } = await WebAssembly.instantiate(readFileSync(join(here, '..', 'public', 'viv', 'viv.wasm')), {});
  const ex = instance.exports;
  const enc = new TextEncoder().encode(await source(mat));
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const t0 = performance.now();
  const packed = ex.viv_run(ptr, enc.length);
  const seconds = (performance.now() - t0) / 1000;
  const text = new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
  return { text, seconds, memoryMB: ex.memory.buffer.byteLength / 1048576 };
}
const compiled = (mat) => runWasi(readFileSync(join(DFT, 'wasm', `${mat}.wasm`)));

let failed = 0;
for (const mat of mats) {
  for (const engine of compiledOnly || mat === 'cu' ? ['compiled'] : ['compiled', 'interpreted']) {
    const { text, seconds, memoryMB } = await (engine === 'compiled' ? compiled(mat) : interpreted(mat));
    const lines = text.split('\n');
    const pass = lines.filter((l) => l.startsWith('PASS')).length;
    const fails = lines.filter((l) => l.startsWith('FAIL') || l.startsWith('error'));
    const iters = (lines.find((l) => l.startsWith('ITER: ')) || '').slice(6);
    const ok = fails.length === 0 && pass > 0 && (mat === 'cu' || seconds < BUDGET_S[engine]);
    if (!ok) failed++;
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${mat.padEnd(5)} ${engine.padEnd(11)} ${String(pass).padStart(2)} pass ${fails.length} fail   ${iters.padStart(2)} SCF iterations   ${seconds.toFixed(1).padStart(6)} s   ${memoryMB.toFixed(0).padStart(4)} MB`);
    for (const f of fails.slice(0, 5)) console.log(`       ${f}`);
    if (pass === 0 && fails.length === 0) console.log(lines.slice(-5).map((l) => `       ${l}`).join('\n'));
  }
}
process.exit(failed ? 1 : 0);
