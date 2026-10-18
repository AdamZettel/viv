// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Run every browser-sized material at BUILD time, through BOTH of the engines the
// page offers, so it opens on a real band structure instead of an empty frame
// and can say truthfully how long each engine takes:
//
//   compiled     dft/wasm/<mat>.wasm  the program itself, compiled Viv -> C -> wasm
//                                     ahead of time by dft/build-wasm.sh (committed,
//                                     and tied to the sources by a manifest of hashes)
//   interpreted  viv.wasm             the Viv interpreter, reading the .viv source
//
// Writes:
//   www/public/dft/<mat>.wasm       the compiled program the page's worker runs
//   www/public/dft/<mat>.viv        the source the interpreter is handed
//   www/src/data/dft.json           per material: settings, results, bands, timings
//
// A material is recomputed only when its program, its module or viv.wasm
// changed (a hash is kept in the JSON), so `npm run dev` does not pay a minute
// every time. If ANY test fails on either engine, the build fails.
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { parseDft } from '../src/lib/dftParse.js';
import { expandUses } from '../src/lib/uses.js';
import { runWasi } from '../public/dft-wasi.mjs';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const DFT = join(ROOT, 'dft');
const OUT_PUBLIC = join(WWW, 'public/dft');
const OUT_JSON = join(WWW, 'src/data/dft.json');
const WASM = join(WWW, 'public/viv/viv.wasm');

// order is the order on the page
const MATERIALS = [
  { id: 'si', name: 'Silicon', formula: 'Si', structure: 'diamond' },
  { id: 'ge', name: 'Germanium', formula: 'Ge', structure: 'diamond' },
  { id: 'c', name: 'Diamond', formula: 'C', structure: 'diamond' },
  { id: 'gaas', name: 'Gallium arsenide', formula: 'GaAs', structure: 'zincblende' },
  { id: 'al', name: 'Aluminum', formula: 'Al', structure: 'fcc metal' },
  { id: 'cu', name: 'Copper', formula: 'Cu', structure: 'fcc metal, 3d in the valence', recorded: true },
];

const read = (f) => readFileSync(join(DFT, f), 'utf8');
// what `viv --use deck --use kmesh --use common --use engine check` does: concatenate,
// with the engine's own import of lib/lobpcg expanded the way the compiler expands it
const fromRepo = async (name) => readFileSync(join(ROOT, `${name}.viv`), 'utf8');
const program = (id) => expandUses([`${id}.viv`, 'kmesh.viv', 'pw_common.viv', 'pwdft_it.viv', `${id}_it_check.viv`].map(read).join('\n'), fromRepo);
const setting = (deck, key) => Number((deck.match(new RegExp(`^${key} = ([-0-9.e]+)`, 'm')) || [])[1]);
// `assert_near(NAME, value, tol)` in a check file's physics section, where NAME is
// a quantity the program also shows: the published value it is held to.
const published = (check) => Object.fromEntries(
  [...check.matchAll(/^\s+assert_near\(([A-Z][A-Z0-9_]*), (-?[0-9.]+), ([0-9.e-]+)\)/gm)].map((m) => [m[1], { value: Number(m[2]), tol: Number(m[3]) }]));
const pseudos = (deck) => [...deck.matchAll(/^# +species \d+: ([A-Z][a-z]?) GTH-PADE-q(\d+)/gm)].map((m) => `${m[1]} GTH-PADE-q${m[2]}`);

mkdirSync(OUT_PUBLIC, { recursive: true });
mkdirSync(dirname(OUT_JSON), { recursive: true });
const previous = existsSync(OUT_JSON) ? JSON.parse(readFileSync(OUT_JSON, 'utf8')) : { materials: [] };
const wasmBytes = existsSync(WASM) ? readFileSync(WASM) : null;
const wasmHash = wasmBytes ? createHash('sha256').update(wasmBytes).digest('hex').slice(0, 16) : 'none';

const modulePath = (id) => join(DFT, 'wasm', `${id}.wasm`);

async function interpret(src) {
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const ex = instance.exports;
  const enc = new TextEncoder().encode(src);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const t0 = performance.now();
  const packed = ex.viv_run(ptr, enc.length);
  const seconds = (performance.now() - t0) / 1000;
  const text = new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
  return { text, seconds };
}
const round1 = (x) => Math.round(x * 10) / 10;
function mustPass(id, engine, parsed, text) {
  if (parsed.fail.length || !parsed.pass.length) {
    console.error(`dft-precompute: ${id} FAILED (${engine}):\n  ${parsed.fail.join('\n  ') || text.split('\n').slice(-5).join('\n  ')}`);
    process.exit(1);
  }
}

const out = [];
for (const mat of MATERIALS) {
  const deck = read(`${mat.id}.viv`);
  const src = await program(mat.id);
  const hasModule = existsSync(modulePath(mat.id));
  const moduleBytes = hasModule ? readFileSync(modulePath(mat.id)) : null;
  const hash = createHash('sha256').update(src).update(wasmHash).update(moduleBytes || '').digest('hex').slice(0, 16);
  const entry = {
    ...mat,
    ecut: setting(deck, 'ECUT'),
    nkmesh: setting(deck, 'NKMESH'),
    smear: setting(deck, 'SMEAR'),
    nbands: setting(deck, 'NBANDS'),
    pseudos: pseudos(deck),
    published: published(read(`${mat.id}_it_check.viv`)),
    lines: src.replace(/\n$/, '').split('\n').length, // the newline that ends the last line opens no new one
    moduleBytes: moduleBytes ? moduleBytes.length : 0,
    hash,
  };
  writeFileSync(join(OUT_PUBLIC, `${mat.id}.viv`), src);
  if (moduleBytes) writeFileSync(join(OUT_PUBLIC, `${mat.id}.wasm`), moduleBytes);
  if (mat.recorded) {
    // minutes, not seconds: too long for a site build. The native run's recorded
    // output, plus how long the SAME program took as compiled wasm, measured once.
    const meta = JSON.parse(read(`results/${mat.id}.meta.json`));
    entry.result = { ...parseDft(read(`results/${mat.id}.out`)), seconds: meta.seconds, machine: meta.machine, engine: 'native' };
    entry.wasmSeconds = meta.wasm_seconds || null;
    entry.wasmMemoryMB = meta.wasm_memory_mb || null;
  } else {
    const old = previous.materials.find((m) => m.id === mat.id);
    if (old && old.hash === hash && old.result && old.interpreted) {
      entry.result = old.result;
      entry.interpreted = old.interpreted;
    } else if (!wasmBytes || !moduleBytes) {
      console.warn(`dft-precompute: ${mat.id}: missing ${!moduleBytes ? 'dft/wasm module (run dft/build-wasm.sh)' : 'viv.wasm (run `npm run wasm`)'}; no build-time result`);
      if (old) { entry.result = old.result; entry.interpreted = old.interpreted; }
    } else {
      const c = await runWasi(moduleBytes);
      const cp = parseDft(c.text);
      mustPass(mat.id, 'compiled', cp, c.text);
      const i = await interpret(src);
      const ip = parseDft(i.text);
      mustPass(mat.id, 'interpreted', ip, i.text);
      entry.result = { ...cp, seconds: round1(c.seconds), engine: 'compiled' };
      entry.interpreted = { seconds: round1(i.seconds), pass: ip.pass.length };
      console.log(`dft-precompute: ${mat.id.padEnd(5)} ${cp.pass.length} tests pass on both engines, ${cp.shown.ITER} SCF iterations: compiled ${c.seconds.toFixed(1)} s, interpreted ${i.seconds.toFixed(1)} s`);
    }
  }
  out.push(entry);
}
writeFileSync(OUT_JSON, JSON.stringify({ materials: out }, null, 1));

// ---- the source, for /dft/source/<file> --------------------------------------
// Each file as written, plus what `viv doc` -- the compiler's own annotation
// extractor, called here through viv.wasm -- makes of it: every function with its
// what / how / why, the variable glossary, the notes on individual lines.
const SOURCES = [
  { slug: 'pwdft_it', file: 'pwdft_it.viv', group: 'The solver', title: 'The iterative engine', blurb: 'What this site runs. H is never formed: it is applied by FFT, and the lowest bands come from LOBPCG. Storage, the projectors, hpsi, the operator and preconditioner it hands the eigensolver, the SCF loop, the band path.' },
  { slug: 'lobpcg', file: 'lib/lobpcg.viv', path: 'lib/lobpcg.viv', group: 'The solver', title: 'The eigensolver', blurb: 'LOBPCG, as a library any program can use: the lowest eigenpairs of an operator handed to it as a function. The engine hands it H at one k-point and the Teter-Payne-Allan preconditioner.' },
  { slug: 'pw_common', file: 'pw_common.viv', group: 'The solver', title: 'What both engines share', blurb: 'Geometry and reciprocal lattice, the GTH pseudopotential closed forms, plane-wave counting, the FFT grid, the exchange-correlation constants, occupations and the Fermi level, Anderson mixing.' },
  { slug: 'kmesh', file: 'kmesh.viv', group: 'The solver', title: 'The k-mesh', blurb: 'The four-shift fcc mesh, halved by time reversal -- and the measurement that says why four shifts.' },
  { slug: 'pwdft', file: 'pwdft.viv', group: 'The solver', title: 'The dense engine', blurb: 'The same physics with H built as a matrix and diagonalized by eig_herm at every k-point: simple, O(N^3), and the step-for-step twin of the numpy reference.' },
  ...MATERIALS.map((m) => ({ slug: m.id, file: `${m.id}.viv`, group: 'Input decks', title: `${m.name}: the deck`, blurb: 'Ordinary Viv that sets the globals the engine reads. Generated by dft/ref/make_deck.py; the pseudopotential parameters come from CP2K\'s table, never typed.' })),
  ...MATERIALS.map((m) => ({ slug: `${m.id}_check`, file: `${m.id}_it_check.viv`, group: 'What each must reproduce', title: `${m.name}: the checks`, blurb: 'Generated by RUNNING the independent numpy reference at the deck\'s settings and pinning what it printed; then the published LDA values; then the figure.' })),
];
let docOf = () => null;
if (wasmBytes) {
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const ex = instance.exports;
  docOf = (text) => {
    const enc = new TextEncoder().encode(text);
    const ptr = ex.viv_alloc(enc.length);
    new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
    const packed = ex.viv_doc(ptr, enc.length);
    const json = new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
    try { return JSON.parse(json); } catch { return null; }
  };
}
const sources = SOURCES.map((f) => {
  const raw = f.path ? readFileSync(join(ROOT, f.path), 'utf8') : read(f.file);
  return { ...f, raw, lines: raw.replace(/\n$/, '').split('\n').length, doc: docOf(raw) };
});
writeFileSync(join(WWW, 'src/data/dftSource.json'), JSON.stringify({ sources }, null, 1));
console.log(`dft-precompute: wrote ${sources.length} source files (${sources.filter((f) => f.doc).length} with viv doc output) to src/data/dftSource.json`);
console.log(`dft-precompute: wrote ${out.length} materials to src/data/dft.json`);
