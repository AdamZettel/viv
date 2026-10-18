// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The engines' tests: the adapters, the decks and packs they write, the
// refusals, and (with --run) the engines themselves on every example input.
//
//     node scripts/enginetest.mjs          adapters, decks, packs, refusals, sizes: a few seconds
//     node scripts/enginetest.mjs --run    also run every example through the engines: about 3 minutes
//     node scripts/enginetest.mjs --fit    also measure the compiled DFT engine on dft/'s five
//                                          tab-sized materials and write the cost model's constant
//
// What "matches" means here, measured rather than asserted by eye:
//   deck      the adapter's deck is dft/<m>.viv character for character from the
//             pseudopotential block down: every field identical, tolerance zero
//   packs     the HF adapter's atoms, shells and primitives equal lib/hfref's
//             (which tools/hf_reference.py took from PySCF), tolerance zero
//   energies  a DFT run lands on dft/<m>_it_check.viv's numpy reference to its
//             own tolerances (1e-7 Ha, 1e-5 eV); an HF run on lib/hfref's PySCF
//             energy to 1e-9 Ha, and on PySCF's orbital energies to the six
//             figures the engine prints
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseStructure } from '../src/lib/engines/formats/index.js';
import { buildDeck, deckText, encodeVariables, DECK_VARIABLES, TABLES } from '../src/lib/engines/deck.js';
import { sizes, work } from '../src/lib/engines/estimate.js';
import { dftInputBlock, dftReferenceVariables, withInput, parseReport, precise } from '../src/lib/engines/report.js';
import { parseMolecule, buildMolecule, moleculeBlock, hfReferenceFor, primitiveFunctions, hfSeconds, BASES } from '../src/lib/engines/molecule.js';
import { pyRepr } from '../src/lib/engines/pyrepr.js';
import { runEngine } from '../public/engine-wasi.mjs';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const RUN = process.argv.includes('--run');
const FIT = process.argv.includes('--fit');
const read = (p) => readFileSync(join(ROOT, p), 'utf8');
const example = (f) => read(`engines/examples/${f}`);
const data = JSON.parse(readFileSync(join(WWW, 'src/data/engines.json'), 'utf8'));

let failures = 0;
let checks = 0;
const runs = []; // with --run: what each engine run measured, for the engine pages
function check(name, ok, detail = '') {
  checks += 1;
  if (!ok) failures += 1;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? `  (${detail})` : ''}`);
}
const body = (t) => t.slice(t.indexOf('# GTH/HGH'));
function firstDiff(a, b) {
  const x = a.split('\n'), y = b.split('\n');
  for (let i = 0; i < Math.max(x.length, y.length); i++) if (x[i] !== y[i]) return `line ${i + 1}: got "${x[i]}", want "${y[i]}"`;
  return '';
}

// ---- Python's float repr, which the decks are written in -----------------
const reprCases = [[10.0, '10.0'], [0.125, '0.125'], [-0.125, '-0.125'], [1e-5, '1e-05'], [1.5e-5, '1.5e-05'], [0.0001, '0.0001'],
  [-7.33610297, '-7.33610297'], [1e16, '1e+16'], [123456789012345.6, '123456789012345.6'], [0.1 + 0.2, '0.30000000000000004'], [0, '0.0']];
check('pyRepr prints floats as Python\'s repr does', reprCases.every(([x, s]) => pyRepr(x) === s),
  reprCases.filter(([x, s]) => pyRepr(x) !== s).map(([x, s]) => `${x}: ${pyRepr(x)} vs ${s}`).join('; '));

// ---- every crystal example becomes dft/'s own deck -------------------------
const CRYSTALS = [['si.poscar', 'si'], ['si.cif', 'si'], ['si.pw.in', 'si'], ['si.xyz', 'si'], ['gaas.poscar', 'gaas'], ['gaas.cif', 'gaas'], ['gaas.pw.in', 'gaas']];
const decks = {};
for (const [f, mat] of CRYSTALS) {
  let built;
  try {
    built = buildDeck(parseStructure(example(f), f));
  } catch (e) {
    check(`${f} is read`, false, e.message);
    continue;
  }
  if (built.refusals.length) { check(`${f} is accepted`, false, built.refusals.join(' ')); continue; }
  const text = deckText(built, f);
  const want = read(`dft/${mat}.viv`);
  check(`${f} -> the deck dft/${mat}.viv, every field identical (${built.formatName}, ${built.crystal.nInput} atoms in the file's cell)`,
    body(text) === body(want), firstDiff(body(text), body(want)));
  decks[f] = { built, mat };
}

// ---- all six materials of dft/ref/materials.py, as POSCARs ------------------
// The table's own crystal written as a POSCAR (the fcc primitive cell in QE's
// orientation, positions shifted so that no atom sits where materials.py puts
// it): the adapter has to find dft/'s orientation and origin itself.
const B = 0.529177210903;
for (const [name, m] of Object.entries(TABLES.materials)) {
  const a = m.alat * B;
  const cell = [[-a / 2, 0, a / 2], [0, a / 2, a / 2], [-a / 2, a / 2, 0]];
  const els = m.atoms.map((x) => x.entry.split(' ')[0]);
  const species = [...new Set(els)];
  // materials.py's fractions are of the dft/ fcc cell; take them to cartesian, shift by (0.1, 0.2, 0.3) of a
  const A = [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]];
  const cart = m.atoms.map((x) => [0, 1, 2].map((c) => x.frac[0] * A[0][c] + x.frac[1] * A[1][c] + x.frac[2] * A[2][c] + [0.1, 0.2, 0.3][c] * a));
  const lines = [`${name} from materials.py`, '1.0', ...cell.map((v) => v.map((x) => x.toFixed(12)).join(' ')), species.join(' '),
    species.map((s) => els.filter((e) => e === s).length).join(' '), 'Cartesian',
    ...species.flatMap((s) => cart.filter((_, i) => els[i] === s).map((v) => v.map((x) => x.toFixed(12)).join(' ')))];
  const built = buildDeck(parseStructure(`${lines.join('\n')}\n`, 'POSCAR'));
  const mat = name.toLowerCase();
  const text = built.deck ? deckText(built, 'POSCAR') : '';
  check(`materials.py ${name}, shifted and in QE's cell vectors -> dft/${mat}.viv, every field identical`,
    !built.refusals.length && body(text) === body(read(`dft/${mat}.viv`)), built.refusals.join(' ') || firstDiff(body(text), body(read(`dft/${mat}.viv`))));
  // the engine's sizes, computed here the way pw_common.viv computes them, against the sizes the site's
  // build-time run of that material printed (www/src/data/dft.json)
  const rec = JSON.parse(readFileSync(join(WWW, 'src/data/dft.json'), 'utf8')).materials.find((x) => x.id === mat);
  if (built.deck && rec && rec.result) {
    const s = sizes(built.deck);
    const sh = rec.result.shown;
    check(`${name}: NKPT ${s.nkpt}, NPWMAX ${s.npwmax}, NG ${s.ng} as the engine printed them`, s.nkpt === sh.NKPT && s.npwmax === sh.NPWMAX && s.ng === sh.NG,
      `engine: ${sh.NKPT}, ${sh.NPWMAX}, ${sh.NG}`);
  }
}

// ---- other ways of writing silicon give the same deck -----------------------
{
  const a = 10.26 * B;
  const variants = {
    'the 8-atom cubic cell': `Si8\n1.0\n${a} 0 0\n0 ${a} 0\n0 0 ${a}\nSi\n8\nDirect\n0 0 0\n0 .5 .5\n.5 0 .5\n.5 .5 0\n.25 .25 .25\n.25 .75 .75\n.75 .25 .75\n.75 .75 .25\n`,
    'a 2x1x1 supercell of the primitive cell': `Si4\n1.0\n0 ${a} ${a}\n${a / 2} 0 ${a / 2}\n${a / 2} ${a / 2} 0\nSi\n4\nDirect\n0 0 0\n.125 .25 .25\n.5 0 0\n.625 .25 .25\n`,
    'the cell rotated 30 degrees about z': (() => {
      const c = Math.cos(Math.PI / 6), s = Math.sin(Math.PI / 6);
      const rot = (v) => [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]];
      const cell = [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]].map(rot);
      return `Si2\n1.0\n${cell.map((v) => v.join(' ')).join('\n')}\nSi\n2\nDirect\n0 0 0\n.25 .25 .25\n`;
    })(),
    'a scale factor of -volume (VASP\'s other convention)': `Si2\n${-(a ** 3) / 4}\n0 .5 .5\n.5 0 .5\n.5 .5 0\nSi\n2\nDirect\n0 0 0\n.25 .25 .25\n`,
  };
  for (const [what, text] of Object.entries(variants)) {
    const built = buildDeck(parseStructure(text, 'POSCAR'));
    const t = built.deck ? deckText(built, 'POSCAR') : '';
    check(`silicon as ${what} -> dft/si.viv`, body(t) === body(read('dft/si.viv')), built.refusals.join(' ') || firstDiff(body(t), body(read('dft/si.viv'))));
  }
}

// ---- what the engine must refuse, and say why --------------------------------
function refusal(what, text, name, want, overrides) {
  let msg = '';
  try {
    const built = buildDeck(parseStructure(text, name), overrides);
    msg = built.refusals.join(' ');
  } catch (e) {
    msg = e.refusal ? e.message : `CRASH: ${e.message}`;
  }
  check(`refuses ${what}`, want.test(msg), msg || 'accepted');
}
const a = 10.26 * B;
const prim = (els, fr, alat = a) => `x\n1.0\n0 ${alat / 2} ${alat / 2}\n${alat / 2} 0 ${alat / 2}\n${alat / 2} ${alat / 2} 0\n${[...new Set(els)].join(' ')}\n${[...new Set(els)].map((s) => els.filter((e) => e === s).length).join(' ')}\nDirect\n${fr.map((f) => f.join(' ')).join('\n')}\n`;
refusal('an element with no pseudopotential (uranium)', prim(['U'], [[0, 0, 0]]), 'POSCAR', /No pseudopotential for U/);
refusal('an element whose pseudopotential has f projectors (lanthanum)', prim(['La'], [[0, 0, 0]]), 'POSCAR', /f-channel projectors/);
refusal('an element with no measured cutoff (sodium chloride), unless a cutoff is given', prim(['Na', 'Cl'], [[0, 0, 0], [0.5, 0.5, 0.5]], 10.6), 'POSCAR', /No measured cutoff for Na, Cl/);
refusal('a spin-polarized pw.x input (ferromagnetic iron)', example('fe.pw.in'), 'fe.pw.in', /spin-polarized/);
refusal('a body-centered cubic lattice', `Fe\n1.0\n2.87 0 0\n0 2.87 0\n0 0 2.87\nFe\n2\nDirect\n0 0 0\n.5 .5 .5\n`, 'POSCAR', /body-centered cubic, not face-centered/);
refusal('a hexagonal cell', `Mg\n1.0\n3.21 0 0\n-1.605 2.78 0\n0 0 5.21\nMg\n2\nDirect\n0.333333 0.666667 0.25\n0.666667 0.333333 0.75\n`, 'POSCAR', /not face-centered cubic/);
refusal('a partially occupied CIF site', example('si.cif').replace('Si1  Si  0.00000  0.00000  0.00000  1', 'Si1  Si  0.00000  0.00000  0.00000  0.5'), 'x.cif', /partially occupied/);
refusal('a CIF that names a space group but lists no operations', example('si.cif').replace(/loop_\n _symmetry_equiv_pos_site_id[\s\S]*?(?=loop_\n _atom_site_label)/, ''), 'x.cif', /lists no symmetry operations/);
refusal('an extended XYZ with no cell', '2\nsilicon\nSi 0 0 0\nSi 1.35 1.35 1.35\n', 'x.xyz', /no cell/);
refusal('a magnetic extended XYZ', `2\nLattice="0 2.7 2.7 2.7 0 2.7 2.7 2.7 0" Properties=species:S:1:pos:R:3:magmoms:R:1\nSi 0 0 0 1\nSi 1.35 1.35 1.35 1\n`, 'x.xyz', /magnetic/);
refusal('an odd electron count with no smearing', example('si.pw.in').replace('Si 0.25 0.25 0.25', 'Al 0.25 0.25 0.25').replace('Si  28.086  Si.pz-vbc.UPF', 'Si  28.086  Si.pz-vbc.UPF\n  Al  26.98  Al.UPF').replace('ntyp = 1', 'ntyp = 2'), 'x.in', /7 electrons with no smearing/);
refusal('more than 8 atoms in the primitive cell', prim(Array(9).fill('Si'), Array.from({ length: 9 }, (_, i) => [i * 0.1, i * 0.05, i * 0.02])), 'POSCAR', /9 atoms in the primitive cell/);
refusal('a file in no format it reads', 'hello\nworld\n', 'notes.txt', /Not a format this engine reads/);
{
  const built = buildDeck(parseStructure(prim(['Na', 'Cl'], [[0, 0, 0], [0.5, 0.5, 0.5]], 10.6), 'POSCAR'), { ecut: 20 });
  check('...and sodium chloride runs, marked unvalidated, once a cutoff is given', !built.refusals.length && built.unvalidated && built.settings.ecut.from === 'set by you', built.refusals.join(' '));
}

// ---- too large for the tab, not for the engine: what a worker is offered ------
{
  // copper: dft/cu.viv exactly, so its reference applies to a worker's run, and
  // an estimated time over the 90 s the page gives the tab (DftBandsEngine.jsx)
  const built = buildDeck(parseStructure(example('cu.pw.in'), 'cu.pw.in'));
  const t = built.deck ? deckText(built, 'cu.pw.in') : '';
  check('cu.pw.in -> dft/cu.viv, from the pseudopotentials down', body(t) === body(read('dft/cu.viv')), built.refusals.join(' ') || firstDiff(body(t), body(read('dft/cu.viv'))));
  const cost = JSON.parse(readFileSync(join(WWW, 'src/lib/engines/dft-cost.json'), 'utf8'));
  const est = cost.c * work(sizes(built.deck), built.deck.NBANDS);
  check(`cu.pw.in is refused for the tab only by its size: estimated ${est.toFixed(0)} s, over 90 s`, est > 90 && !built.refusals.length);
  // a molecule the tab cannot take in time is marked tooBig, which is what offers a worker
  const bz = '12\nbenzene basis=sto-3g\nC 0.000 1.396 0.000\nC 1.209 0.698 0.000\nC 1.209 -0.698 0.000\nC 0.000 -1.396 0.000\nC -1.209 -0.698 0.000\nC -1.209 0.698 0.000\nH 0.000 2.479 0.000\nH 2.147 1.240 0.000\nH 2.147 -1.240 0.000\nH 0.000 -2.479 0.000\nH -2.147 -1.240 0.000\nH -2.147 1.240 0.000\n';
  const b = buildMolecule(parseMolecule(bz));
  check(`benzene in STO-3G: refused for the tab by its size alone (${b.nao} functions, about ${Math.round(b.seconds)} s), so a worker is offered`, b.tooBig && b.refusals.length === 1, b.refusals.join(' '));
  const w = buildMolecule(parseMolecule(example('water.xyz')));
  check('water in 6-31G is not too big for the tab', !w.tooBig && !w.refusals.length);
}

// ---- the Hartree-Fock adapter: packs identical to lib/hfref's ---------------
function hfrefPacks(mod, p) {
  const t = read(`lib/hfref/${mod}.viv`);
  const flat = (what) => {
    const m = t.match(new RegExp(`def ${p}_${what}_flat\\(\\) -> Arr \\{\\n\\s*return \\[([^\\]]*)\\]`));
    return m[1].split(',').map(Number);
  };
  return { atoms: flat('atoms'), shells: flat('shells'), prims: flat('prims') };
}
const MOLECULES = [['water.xyz', 'water_631g', 'water631g', 'water 6-31g RHF'], ['oh.xyz', 'oh_631g', 'oh_631g', 'OH 6-31g UHF'], ['ammonia.xyz', 'breadth_631gs', 'nh3_631gs', null]];
const mols = {};
for (const [f, mod, p, orb] of MOLECULES) {
  const built = buildMolecule(parseMolecule(example(f)));
  if (built.refusals.length) { check(`${f} is accepted`, false, built.refusals.join(' ')); continue; }
  const want = hfrefPacks(mod, p);
  const got = { atoms: built.packs.atoms.flat(), shells: built.packs.shells.flat(), prims: built.packs.prims.flat() };
  const same = ['atoms', 'shells', 'prims'].every((k) => got[k].length === want[k].length && got[k].every((v, i) => v === want[k][i]));
  check(`${f} (${built.basis}) -> lib/hfref/${mod}.viv's atoms, shells and primitives, every number identical`, same,
    ['atoms', 'shells', 'prims'].map((k) => `${k} ${got[k].length}/${want[k].length}`).join(', '));
  const ref = hfReferenceFor(built, data.hfRefs);
  check(`${f}: the reference is found (${ref ? ref.name : 'none'})`, !!ref);
  mols[f] = { built, ref, orb };
}
function hfRefusal(what, text, opts, want) {
  let msg = '';
  try { msg = buildMolecule(parseMolecule(text), opts).refusals.join(' '); } catch (e) { msg = e.refusal ? e.message : `CRASH: ${e.message}`; }
  check(`refuses ${what}`, want.test(msg), msg || 'accepted');
}
hfRefusal('an element past argon (potassium)', '1\n\nK 0 0 0\n', {}, /hydrogen through argon/);
hfRefusal('a basis it does not have', example('water.xyz'), { basis: 'aug-cc-pvqz' }, /No basis named "aug-cc-pvqz"/);
hfRefusal('RHF on an open shell', example('oh.xyz'), { method: 'rhf' }, /RHF needs a closed shell/);
hfRefusal('a spin the electron count cannot have', example('water.xyz'), { spin: 1 }, /cannot have Na - Nb = 1/);
hfRefusal('a molecule too large for the tab (octane in cc-pVDZ)', `26\noctane\n${Array.from({ length: 8 }, (_, i) => `C ${(i * 1.54).toFixed(2)} 0 0`).join('\n')}\n${Array.from({ length: 18 }, (_, i) => `H ${(i * 0.7).toFixed(2)} 1.0 ${i % 2 ? 1 : -1}`).join('\n')}\n`, { basis: 'cc-pvdz' }, /estimated at about \d+ s: more than the 60 s/);
hfRefusal('a crystal handed to the molecule engine', example('si.xyz'), {}, /describes a crystal/);

// ---- running the engines ------------------------------------------------------
const vivWasm = join(WWW, 'public/viv/viv.wasm');
async function interpret(src) {
  const { instance } = await WebAssembly.instantiate(readFileSync(vivWasm), {});
  const ex = instance.exports;
  const enc = new TextEncoder().encode(src);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const t0 = performance.now();
  const packed = ex.viv_run(ptr, enc.length);
  const seconds = (performance.now() - t0) / 1000;
  return { text: new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn))), seconds };
}
const dftWasm = join(WWW, 'public/engines/dft-bands.wasm');
async function runCompiled(built, ref) {
  const stdin = encodeVariables({ ...built.deck, ...dftReferenceVariables(ref) }, [...DECK_VARIABLES, 'HAS_REF', 'REF_ETOT', 'REF_NB', 'REFPATH']);
  return runEngine(readFileSync(dftWasm), { stdin });
}
function judgeDft(label, r, ref, seconds) {
  const rep = parseReport(r.text);
  const e = precise(rep.shown, 'ETOT');
  const ok = rep.fail.length === 0 && rep.pass.length === 2 && rep.shown.REF_DE < ref.etol && rep.shown.REF_DBAND < ref.btol;
  runs.push({ engine: 'dft-bands', label, deck: `dft/${ref.id}.viv`, seconds: Math.round(seconds * 10) / 10, energy: e, energyError: rep.shown.REF_DE, bandError: rep.shown.REF_DBAND, etol: ref.etol, btol: ref.btol, pass: rep.pass.length, fail: rep.fail.length });
  check(`${label}: E = ${e} Ha, ${Math.abs(e - ref.etot).toExponential(1)} from dft/'s reference (tolerance ${ref.etol}), bands within ${rep.shown.REF_DBAND} eV (tolerance ${ref.btol}), ${seconds.toFixed(1)} s`,
    ok && Math.abs(e - ref.etot) < ref.etol, rep.fail.join('; '));
  return rep;
}
if (RUN) {
  if (!existsSync(vivWasm) || !existsSync(dftWasm)) {
    check('the engines are staged (viv.wasm, and public/engines/ from npm run stage)', false);
  } else {
    const program = readFileSync(join(WWW, 'public/engines/dft-bands.viv'), 'utf8');
    for (const [f, { built }] of Object.entries(decks)) {
      const { ref, text } = dftInputBlock(built, f, data.dftRefs);
      check(`${f}: its deck is dft/${ref ? ref.id : '?'}.viv, so that reference applies`, !!ref);
      if (!ref) continue;
      const r = await runCompiled(built, ref);
      const rep = judgeDft(`${f}, compiled`, r, ref, r.seconds);
      const s = sizes(built.deck);
      check(`${f}: the page's sizes are the engine's (NKPT ${s.nkpt}, NPWMAX ${s.npwmax}, NG ${s.ng})`, s.nkpt === rep.shown.NKPT && s.npwmax === rep.shown.NPWMAX && s.ng === rep.shown.NG);
      void text;
    }
    // the interpreter, once per material: the same deck text the page would hand viv.wasm
    for (const f of ['si.poscar', 'gaas.cif']) {
      const { built } = decks[f];
      const { block, ref } = dftInputBlock(built, f, data.dftRefs);
      const r = await interpret(withInput(program, block));
      judgeDft(`${f}, interpreted`, r, ref, r.seconds);
    }
    const hfProgram = readFileSync(join(WWW, 'public/engines/hf-energy.viv'), 'utf8');
    for (const [f, { built, ref, orb }] of Object.entries(mols)) {
      const r = await interpret(withInput(hfProgram, moleculeBlock(built, ref)));
      const rep = parseReport(r.text);
      const e = precise(rep.shown, 'E');
      runs.push({ engine: 'hf-energy', label: `${f}, ${built.basis}`, reference: ref.name, module: ref.module, seconds: Math.round(r.seconds * 10) / 10, energy: e, refEnergy: ref.energy, energyError: rep.shown.REF_DE, pass: rep.pass.length, fail: rep.fail.length });
      check(`${f} (${built.basis}): E = ${e} Ha, ${Math.abs(e - ref.energy).toExponential(1)} from PySCF's ${ref.energy} (${ref.name}; tolerance 1e-9), ${r.seconds.toFixed(1)} s`,
        rep.fail.length === 0 && rep.shown.REF_DE < 1e-9 && Math.abs(e - ref.energy) < 1e-9, rep.fail.join('; '));
      if (orb) {
        const want = BASES.orbitals[orb];
        // the engine prints six significant figures: compare each at half a unit of its last printed digit
        const close = (got, w) => got && got.length === w.length && got.every((v, i) => Math.abs(v - w[i]) <= 0.5 * 10 ** (Math.floor(Math.log10(Math.abs(w[i]))) - 5) * 1.0001);
        check(`${f}: orbital energies are PySCF's to the six figures printed (${want.alpha.length} alpha${want.beta ? `, ${want.beta.length} beta` : ''})`,
          close(rep.shown.MO_ALPHA, want.alpha) && (!want.beta || close(rep.shown.MO_BETA, want.beta)));
      }
    }
  }
}

// ---- the cost model's constant ------------------------------------------------
if (FIT) {
  const rows = [];
  for (const [name, m] of Object.entries(TABLES.materials)) {
    if (name === 'Cu') continue; // minutes; it is the model's out-of-sample check below
    const cell = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]].map((v) => v.map((x) => x * m.alat * B));
    const els = m.atoms.map((x) => x.entry.split(' ')[0]);
    const sp = [...new Set(els)];
    const text = `${name}\n1.0\n${cell.map((v) => v.join(' ')).join('\n')}\n${sp.join(' ')}\n${sp.map((s) => els.filter((e) => e === s).length).join(' ')}\nDirect\n${sp.flatMap((s) => m.atoms.filter((x) => x.entry.startsWith(`${s} `)).map((x) => x.frac.join(' '))).join('\n')}\n`;
    const built = buildDeck(parseStructure(text, 'POSCAR'));
    const r = await runCompiled(built, null);
    const s = sizes(built.deck);
    rows.push({ name, seconds: r.seconds, work: work(s, built.deck.NBANDS) });
    console.log(`fit: ${name} ${r.seconds.toFixed(1)} s, work ${rows.at(-1).work.toExponential(3)}`);
  }
  // least squares through the origin: seconds = c * work
  const c = rows.reduce((s, r) => s + r.seconds * r.work, 0) / rows.reduce((s, r) => s + r.work * r.work, 0);
  const worst = Math.max(...rows.map((r) => Math.abs(c * r.work - r.seconds) / r.seconds));
  const cu = TABLES.materials.Cu;
  const cuBuilt = buildDeck({ lattice: [[0, cu.alat / 2, cu.alat / 2], [cu.alat / 2, 0, cu.alat / 2], [cu.alat / 2, cu.alat / 2, 0]], atoms: [{ el: 'Cu', frac: [0, 0, 0] }], settings: {}, notes: [] });
  const cuPred = c * work(sizes(cuBuilt.deck), cuBuilt.deck.NBANDS);
  const cuMeta = JSON.parse(read('dft/results/cu.meta.json'));
  const out = {
    _: 'written by www/scripts/enginetest.mjs --fit: seconds = c * work (www/src/lib/engines/estimate.js), fitted to the compiled dft-bands engine in node',
    c, fitted: rows.map((r) => ({ name: r.name, seconds: Math.round(r.seconds * 10) / 10, predicted: Math.round(c * r.work * 10) / 10 })),
    worst_relative_error: Math.round(worst * 100) / 100,
    copper: { predicted: Math.round(cuPred), measured_wasm: cuMeta.wasm_seconds || null },
    machine: process.env.ENGINE_FIT_MACHINE || 'unrecorded',
  };
  writeFileSync(join(WWW, 'src/lib/engines/dft-cost.json'), `${JSON.stringify(out, null, 1)}\n`);
  console.log(`fit: c = ${c.toExponential(4)}, worst relative error ${(worst * 100).toFixed(0)}%, copper predicted ${cuPred.toFixed(0)} s against ${cuMeta.wasm_seconds} s measured`);
}

// ---- the Hartree-Fock cost model: seconds = c * P^a * nao^b, fitted in logs,
// where P counts primitive Cartesian functions (the two-electron integrals'
// work grows with it) and nao the basis functions (the SCF's matrices)
if (FIT) {
  const MOLS = {
    water: example('water.xyz'), oh: example('oh.xyz'), ammonia: example('ammonia.xyz'),
    ethanol: '9\nethanol\nC -0.0017 -0.0006 0\nC 1.5117 0.0012 0\nO -0.4700 1.3469 0\nH -0.3906 -0.5120 0.8839\nH -0.3906 -0.5120 -0.8839\nH 1.8849 -1.0253 0\nH 1.8849 0.5147 0.8927\nH 1.8849 0.5147 -0.8927\nH -1.4300 1.3200 0\n',
    benzene: `12\nbenzene\n${Array.from({ length: 6 }, (_, i) => { const t = (i * Math.PI) / 3; return `C ${(1.39 * Math.cos(t)).toFixed(4)} ${(1.39 * Math.sin(t)).toFixed(4)} 0\nH ${(2.47 * Math.cos(t)).toFixed(4)} ${(2.47 * Math.sin(t)).toFixed(4)} 0`; }).join('\n')}\n`,
  };
  const SET = [['water', '6-31g'], ['oh', '6-31g'], ['ammonia', '6-31g*'], ['water', 'cc-pvdz'], ['water', 'def2-svp'], ['ethanol', 'sto-3g'], ['benzene', 'sto-3g'], ['ethanol', '6-31g'], ['ethanol', '6-31g*']];
  const hfProgram = readFileSync(join(WWW, 'public/engines/hf-energy.viv'), 'utf8');
  const rows = [];
  for (const [m, basis] of SET) {
    const built = buildMolecule(parseMolecule(MOLS[m]), { basis }, { maxAO: 1000, maxSeconds: Infinity });
    const r = await interpret(withInput(hfProgram, moleculeBlock(built, null)));
    rows.push({ molecule: m, basis, P: primitiveFunctions(built), nao: built.nao, seconds: Math.round(r.seconds * 10) / 10 });
    console.log(`fit: ${m} ${basis}: P ${rows.at(-1).P}, nao ${built.nao}, ${r.seconds.toFixed(1)} s`);
  }
  // least squares for log t = log c + a log P + b log nao
  const A = rows.map((r) => [1, Math.log(r.P), Math.log(r.nao)]);
  const y = rows.map((r) => Math.log(r.seconds));
  const M = [0, 1, 2].map((i) => [...[0, 1, 2].map((j) => A.reduce((s, r) => s + r[i] * r[j], 0)), A.reduce((s, r, k) => s + r[i] * y[k], 0)]);
  for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) { const f = M[j][i] / M[i][i]; for (let k = i; k < 4; k++) M[j][k] -= f * M[i][k]; }
  const x = [0, 0, 0];
  for (let i = 2; i >= 0; i--) x[i] = (M[i][3] - [0, 1, 2].slice(i + 1).reduce((s, k) => s + M[i][k] * x[k], 0)) / M[i][i];
  const model = { c: Math.exp(x[0]), a: x[1], b: x[2] };
  for (const r of rows) r.predicted = Math.round(hfSeconds(r.P, r.nao, model) * 10) / 10;
  const worst = Math.max(...rows.map((r) => Math.abs(r.predicted - r.seconds) / r.seconds));
  writeFileSync(join(WWW, 'src/lib/engines/hf-cost.json'), `${JSON.stringify({
    _: 'written by www/scripts/enginetest.mjs --fit: seconds = c * P^a * nao^b for the hf-energy engine in viv.wasm under node (www/src/lib/engines/molecule.js)',
    ...model, worst_relative_error: Math.round(worst * 100) / 100, machine: process.env.ENGINE_FIT_MACHINE || 'unrecorded', measured: rows,
  }, null, 1)}\n`);
  console.log(`fit: hf seconds = ${model.c.toExponential(3)} P^${model.a.toFixed(2)} nao^${model.b.toFixed(2)}, worst relative error ${(worst * 100).toFixed(0)}%`);
}

if (RUN && !failures) {
  writeFileSync(join(WWW, 'src/data/engineRuns.json'), `${JSON.stringify({ _: 'written by www/scripts/enginetest.mjs --run when every check passes', node: process.version, runs }, null, 1)}\n`);
}

console.log(`\nenginetest: ${checks - failures} of ${checks} checks pass${RUN ? '' : ' (add --run to run the engines)'}`);
process.exit(failures ? 1 : 0);
