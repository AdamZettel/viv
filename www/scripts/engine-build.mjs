// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Stage the engines: Viv programs with a declared interface, each run from an input file the reader supplies.
//
// An engine is a file in engines/ whose header declares it:
//
//   #! engine dft-bands                      its name, and its page: /engines/dft-bands
//   #! title  Band structure of a crystal    one line
//   #! program dft/pwdft_it                  the program it runs, for the "source" link
//   #! input  crystal: poscar | cif | ...    what it reads (one line per input)
//   #! output bands, total-energy            what it produces
//   #! limit  tab: ...                       what the tab can take; the rest runs natively
//   #! validated si: dft/si_it_check.viv     a check it has passed, and the file that holds it
//
// Each must also mark the block the page replaces with the reader's input:
// a line starting "# ---- input:" and one starting "# ---- end of input".
// A header that names a validated file that does not exist, or an engine
// without the input block, fails the build.
//
// Writes:
//   www/public/engines/<name>.viv      the program, imports expanded, for viv.wasm
//   www/public/engines/<name>.wasm     the compiled module, when engines/wasm/ has one
//   www/public/engines/examples/*      the example inputs the pages offer
//   www/src/data/engines.json          every manifest, plus the reference tables:
//     dftRefs  per material in dft/: the deck text and what dft/<m>_it_check.viv
//              pins (the numpy reference's total energy and bands)
//     hfRefs   per calculation in lib/hfref: atoms, basis, state and PySCF's energy
import { readFileSync, writeFileSync, mkdirSync, readdirSync, existsSync, copyFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { expandUses } from '../src/lib/uses.js';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const ENGINES = join(ROOT, 'engines');
const OUT = join(WWW, 'public/engines');
const read = (p) => readFileSync(join(ROOT, p), 'utf8');

/** The `#! key value` header of an engine file. */
export function parseManifest(src, file) {
  const m = { file, inputs: [], outputs: [], limits: [], validated: [] };
  for (const line of src.split('\n')) {
    const h = line.match(/^#!\s*(\w+)\s+(.*?)\s*$/);
    if (!h) continue;
    const [, key, val] = h;
    if (key === 'engine') m.name = val;
    else if (key === 'title') m.title = val;
    else if (key === 'program') m.program = val;
    else if (key === 'input') { const [what, how] = val.split(/:\s*/); m.inputs.push({ what, how: how || '' }); }
    else if (key === 'output') m.outputs.push(...val.split(/,\s*/));
    else if (key === 'limit') { const [where, what] = val.split(/:\s*/); m.limits.push({ where, what: what || '' }); }
    else if (key === 'validated') { const [what, where] = val.split(/:\s*/); m.validated.push({ what, file: where }); }
  }
  return m;
}

mkdirSync(join(OUT, 'examples'), { recursive: true });
const engines = [];
for (const f of readdirSync(ENGINES).filter((x) => x.endsWith('.viv')).sort()) {
  const src = readFileSync(join(ENGINES, f), 'utf8');
  if (!/^#! engine /m.test(src)) continue; // a helper module such as precise.viv
  const m = parseManifest(src, `engines/${f}`);
  if (!m.name || !m.title) throw new Error(`engine-build: engines/${f} needs #! engine and #! title lines`);
  for (const v of m.validated) {
    if (!existsSync(join(ROOT, v.file))) throw new Error(`engine-build: ${m.name} claims "${v.what}" is validated by ${v.file}, which does not exist`);
  }
  if (!/^# ---- input:/m.test(src) || !/^# ---- end of input/m.test(src)) throw new Error(`engine-build: engines/${f} has no "# ---- input:" ... "# ---- end of input" block`);
  // the program as viv.wasm will read it: every import expanded, the way the
  // compiler expands it, so the tab needs no further fetches
  const staged = await expandUses(src, async (name) => read(`${name}.viv`));
  writeFileSync(join(OUT, `${m.name}.viv`), staged);
  const wasm = join(ENGINES, 'wasm', `${m.name}.wasm`);
  m.compiled = existsSync(wasm);
  if (m.compiled) {
    copyFileSync(wasm, join(OUT, `${m.name}.wasm`));
    m.wasmBytes = readFileSync(wasm).length;
  }
  m.source = src;
  m.lines = src.split('\n').length;
  m.stagedLines = staged.split('\n').length;
  engines.push(m);
}
const exampleFiles = readdirSync(join(ENGINES, 'examples')).sort();
for (const f of exampleFiles) copyFileSync(join(ENGINES, 'examples', f), join(OUT, 'examples', f));

// ---- dft/: each material's deck and the reference its check file pins ------
const DFT_MATERIALS = ['si', 'ge', 'c', 'gaas', 'al', 'cu'];
const dftRefs = {};
for (const id of DFT_MATERIALS) {
  const deck = read(`dft/${id}.viv`);
  const check = read(`dft/${id}_it_check.viv`);
  const etot = check.match(/assert_near\((?:ETOT), (-?[0-9.]+), ([0-9.e-]+)\)/);
  const btol = check.match(/assert_lt\(max\(abs\(EVB\[:, 0:shape\(REFPATH\)\[1\]\] - REFPATH\)\), ([0-9.e-]+)\)/);
  const path = check.match(/^REFPATH = \[([\s\S]*?)\]\n/m);
  if (!etot || !btol || !path) throw new Error(`engine-build: cannot read the reference out of dft/${id}_it_check.viv`);
  const rows = [...path[1].matchAll(/\[([^\]]*)\]/g)].map((r) => r[1].split(',').map(Number));
  dftRefs[id] = {
    deckBody: deck.slice(deck.indexOf('# GTH/HGH')),
    etot: Number(etot[1]), etol: Number(etot[2]), btol: Number(btol[1]),
    nb: rows[0].length, refpath: rows,
  };
}

// ---- lib/hfref: the calculations PySCF was run on (tools/hf_reference.py's SPECS)
// module, the constants' prefix, basis, AO kind ('pure' has no d functions, so
// both kinds agree), and each state: [constant suffix, method, charge, spin]
const CLOSED = [['RHF', 'RHF', 0, 0]];
const OPEN1 = [['UHF', 'UHF', 0, 1], ['ROHF', 'ROHF', 0, 1]];
const OPEN2 = [['UHF', 'UHF', 0, 2], ['ROHF', 'ROHF', 0, 2]];
const HF_SPECS = [
  ['water_sto3g', 'WATERSTO3G', 'sto-3g', 'pure', [...CLOSED, ['UHF', 'UHF', 0, 0]]],
  ['water_631g', 'WATER631G', '6-31g', 'pure', [...CLOSED, ['UHF', 'UHF', 0, 0], ['CATION_ROHF', 'ROHF', 1, 1], ['CATION_UHF', 'UHF', 1, 1], ['SPIN2_UHF', 'UHF', 0, 2], ['SPIN2_ROHF', 'ROHF', 0, 2]]],
  ['water_631g_stretched', 'WATER631G_STRETCHED', '6-31g', 'pure', CLOSED],
  ['water_631gs_cart', 'WATER631GS_CART', '6-31g*', 'cart', CLOSED],
  ['water_631gs_sph', 'WATER631GS_SPH', '6-31g*', 'sph', CLOSED],
  ['water_ccpvdz', 'WATERCCPVDZ', 'cc-pvdz', 'mixed', CLOSED],
  ['water_svp', 'WATERSVP', 'def2-svp', 'mixed', CLOSED],
  ['hf_631gs', 'HF_631GS', '6-31g*', 'mixed', CLOSED],
  ['oh_sto3g', 'OH_STO3G', 'sto-3g', 'pure', OPEN1],
  ['oh_631g', 'OH_631G', '6-31g', 'pure', OPEN1],
  ['oh_631gs', 'OH_631GS', '6-31g*', 'mixed', OPEN1],
  ['o2_sto3g', 'O2_STO3G', 'sto-3g', 'pure', OPEN2],
  ['o2_631g', 'O2_631G', '6-31g', 'pure', OPEN2],
  ['o2_631gs', 'O2_631GS', '6-31g*', 'mixed', OPEN2],
  ['breadth_631gs', 'NH3_631GS', '6-31g*', 'mixed', CLOSED],
  ['breadth_631gs', 'CH4_631GS', '6-31g*', 'mixed', CLOSED],
  ['breadth_631gs', 'N2_631GS', '6-31g*', 'mixed', CLOSED],
];
const hfRefs = [];
for (const [mod, P, basis, kind, states] of HF_SPECS) {
  const text = read(`lib/hfref/${mod}.viv`);
  const p = P.toLowerCase();
  const am = text.match(new RegExp(`def ${p}_atoms_flat\\(\\) -> Arr \\{\\n\\s*return \\[([^\\]]*)\\]`));
  if (!am) throw new Error(`engine-build: no ${p}_atoms_flat in lib/hfref/${mod}.viv`);
  const flat = am[1].split(',').map(Number);
  const atoms = [];
  for (let i = 0; i < flat.length; i += 4) atoms.push(flat.slice(i, i + 4));
  // constants: <P>_E_<state> is spherical (or the only one); <P>_E_<state>_CART is Cartesian
  const kinds = kind === 'pure' ? [[null, '']] : kind === 'cart' ? [[false, '']] : kind === 'sph' ? [[true, '']] : [[true, ''], [false, '_CART']];
  for (const [suffix, method, charge, spin] of states) {
    for (const [spherical, cs] of kinds) {
      const c = text.match(new RegExp(`^${P}_E_${suffix}${cs} = (-?[0-9.e-]+)$`, 'm'));
      if (!c) continue;
      hfRefs.push({ module: `lib/hfref/${mod}.viv`, name: `${P}_E_${suffix}${cs}`, basis, spherical, method, charge, spin, atoms, energy: Number(c[1]) });
    }
  }
}

writeFileSync(join(WWW, 'src/data/engines.json'), JSON.stringify({ engines, examples: exampleFiles, dftRefs, hfRefs }, null, 1));
console.log(`engine-build: ${engines.length} engines (${engines.map((e) => `${e.name}${e.compiled ? ' +wasm' : ''}`).join(', ')}), ${exampleFiles.length} example inputs, ${Object.keys(dftRefs).length} dft references, ${hfRefs.length} lib/hfref energies`);
