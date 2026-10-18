// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// From an XYZ file and a basis name to the Hartree-Fock engine's input: the
// three basis packs lib/hf takes, written as Viv.
//
//   parseMolecule(text)          -> { atoms: [{ el, x }], keys }   (x in angstrom)
//   buildMolecule(parsed, opts)  -> { packs, nao, nelec, settings, notes, refusals, ... }
//   moleculeBlock(built)         -> the engine's input block, Viv source
//
// The packs are built exactly as tools/hf_reference.py builds lib/hfref's
// from PySCF: the basis is PySCF's (hf-bases.json, written by
// www/scripts/hf-tables.py), general contractions are split into one shell
// per contraction column, primitives whose coefficient is exactly zero are
// dropped, and angstroms become bohr with PySCF's own constant. So a molecule
// that lib/hfref also holds gets bit-for-bit the packs PySCF's integrals were
// computed from, and the engine can be held to PySCF's energy.
import BASES from './hf-bases.json' with { type: 'json' };
import HF_COST from './hf-cost.json' with { type: 'json' };
import { atomicNumber } from './elements.js';
import { Refusal } from './crystal.js';
import { readXyz, commentKeys } from './formats/xyz.js';
import { pyRepr } from './pyrepr.js';

export { BASES, HF_COST };
export const MAX_SECONDS = 60; // what the tab is offered, by the cost model below
export const MAX_AO = 60; // and a hard cap, whatever the model says

/** Primitive Cartesian functions: sum over shells of primitives times components. */
export function primitiveFunctions(built) {
  return built.packs.shells.reduce((s, sh) => s + sh[2] * sh[6], 0);
}

/**
 * Seconds for viv.wasm to run the engine: c * P^a * nao^b, fitted to measured
 * runs (hf-cost.json, from www/scripts/enginetest.mjs --fit). P grows the
 * two-electron integrals' work; nao the SCF's matrix work.
 */
export function hfSeconds(P, nao, model = HF_COST) {
  return model.c * P ** model.a * nao ** model.b;
}
export const BASIS_NAMES = Object.keys(BASES.bases);
const BOHR = BASES.bohr_angstrom;
const METHODS = { auto: 0, rhf: 1, uhf: 2, rohf: 3 };
export const METHOD_NAMES = ['auto', 'RHF', 'UHF', 'ROHF'];

/**
 * An XYZ file. The comment line may carry charge=, multiplicity= (or spin=,
 * as Na - Nb) and basis= in extended-XYZ key=value form; the page's fields
 * start from those.
 */
export function parseMolecule(text) {
  if (!text || !text.trim()) throw new Refusal('The file is empty.');
  const { atoms, comment } = readXyz(text, 'XYZ');
  const keys = commentKeys(comment);
  if (keys.lattice) {
    throw new Refusal('This XYZ file has a Lattice="...": it describes a crystal. The Hartree-Fock engine computes molecules; a crystal belongs in the DFT engine.');
  }
  return { atoms, comment: comment.trim(), keys };
}

const ncart = (l) => ((l + 1) * (l + 2)) / 2;

/** The packs for these atoms in this basis, split as tools/hf_reference.py's packs() splits them. */
function packs(atoms, table) {
  const A = [];
  const shells = [];
  const prims = [];
  let p0 = 0, cart0 = 0, sph0 = 0;
  atoms.forEach((at, ia) => {
    const z = atomicNumber(at.el);
    A.push([z, ...at.x.map((v) => v * (1 / BOHR))]);
    for (const sh of table[at.el]) {
      for (const col of sh.coefs) {
        const keep = [];
        sh.exps.forEach((e, p) => { if (col[p] !== 0) keep.push([e, col[p]]); });
        prims.push(...keep);
        const nc = ncart(sh.l), ns = 2 * sh.l + 1;
        shells.push([ia, sh.l, keep.length, p0, cart0, sph0, nc, ns]);
        p0 += keep.length;
        cart0 += nc;
        sph0 += ns;
      }
    }
  });
  return { atoms: A, shells, prims, naoCart: cart0, naoSph: sph0 };
}

/** Parse "rhf", "UHF", 2, ... into the method index lib/hf uses. */
function methodIndex(m) {
  if (m === undefined || m === null || m === '') return 0;
  if (typeof m === 'number') return m;
  const k = String(m).toLowerCase();
  if (!(k in METHODS)) throw new Refusal(`Method "${m}": the engine runs RHF, UHF and ROHF.`);
  return METHODS[k];
}

/**
 * Build the engine input. `opts` = { basis, charge, spin, method, spherical }
 * (any subset; the file's comment keys and then defaults fill the rest).
 * `limits` = { maxAO, maxSeconds } refuses molecules too large for the tab.
 */
export function buildMolecule(parsed, opts = {}, limits = { maxAO: MAX_AO, maxSeconds: MAX_SECONDS }) {
  const refusals = [];
  const notes = [];
  const settings = {};
  const pick = (k, fromFile, dflt, why) => {
    if (opts[k] !== undefined && opts[k] !== null && opts[k] !== '') settings[k] = { value: opts[k], from: 'set by you' };
    else if (fromFile !== undefined) settings[k] = { value: fromFile, from: 'the XYZ comment line' };
    else settings[k] = { value: dflt, from: why };
  };
  const keys = parsed.keys || {};
  pick('basis', keys.basis, '6-31g', 'the default');
  pick('charge', keys.charge !== undefined ? Number(keys.charge) : undefined, 0, 'neutral, the default');
  const basis = String(settings.basis.value).toLowerCase();
  const table = BASES.bases[basis];
  if (!table) refusals.push(`No basis named "${settings.basis.value}": the engine has ${BASIS_NAMES.join(', ')} (from PySCF ${BASES.pyscf}).`);
  const missing = [...new Set(parsed.atoms.map((a) => a.el))].filter((el) => !table || !table[el]);
  if (table && missing.length) {
    const beyond = missing.filter((el) => atomicNumber(el) > 18);
    refusals.push(beyond.length
      ? `${missing.join(', ')}: the engine carries basis sets for hydrogen through argon only.`
      : `${missing.join(', ')}: ${basis} has no functions for ${missing.length > 1 ? 'these elements' : 'this element'} in PySCF's library.`);
  }
  const charge = Number(settings.charge.value);
  if (!Number.isInteger(charge)) refusals.push(`The charge must be a whole number; it is ${settings.charge.value}.`);
  const zsum = parsed.atoms.reduce((s, a) => s + atomicNumber(a.el), 0);
  const nelec = zsum - charge;
  if (nelec <= 0) refusals.push(`With charge ${charge} the molecule has ${nelec} electrons; it needs at least one.`);
  // spin: Na - Nb, as PySCF counts it; multiplicity 2S + 1 is also accepted from the file
  let fileSpin;
  if (keys.spin !== undefined) fileSpin = Number(keys.spin);
  else if (keys.multiplicity !== undefined) fileSpin = Number(keys.multiplicity) - 1;
  pick('spin', fileSpin, nelec % 2, nelec % 2 ? 'the lowest spin an odd electron count allows' : 'closed shell, the default');
  const spin = Number(settings.spin.value);
  if (!Number.isInteger(spin) || spin < 0) refusals.push(`The spin (Na - Nb) must be a whole number, 0 or more; it is ${settings.spin.value}.`);
  else if ((nelec - spin) % 2 !== 0) refusals.push(`${nelec} electrons cannot have Na - Nb = ${spin}: the two must be both even or both odd.`);
  else if (spin > nelec) refusals.push(`Na - Nb = ${spin} is more than the ${nelec} electrons.`);
  pick('method', keys.method, 'auto', 'RHF for a closed shell, UHF otherwise');
  let method = 0;
  try { method = methodIndex(settings.method.value); } catch (e) { refusals.push(e.message); }
  if (method === 1 && spin !== 0) refusals.push(`RHF needs a closed shell; this one has Na - Nb = ${spin}. Use UHF or ROHF.`);
  pick('spherical', keys.spherical !== undefined ? keys.spherical !== '0' && keys.spherical !== 'false' : undefined, true, 'spherical harmonics, PySCF\'s default');
  const spherical = settings.spherical.value === true || settings.spherical.value === 'true' || settings.spherical.value === 1;
  // two atoms on one spot give a singular overlap and an infinite repulsion
  for (let i = 0; i < parsed.atoms.length; i++) {
    for (let j = 0; j < i; j++) {
      const d = Math.hypot(...parsed.atoms[i].x.map((v, c) => v - parsed.atoms[j].x[c]));
      if (d < 0.1) refusals.push(`Atoms ${j + 1} and ${i + 1} are ${d.toFixed(3)} angstrom apart; the engine refuses atoms closer than 0.1 angstrom. Check the coordinates.`);
    }
  }
  let pk = null;
  let nao = null;
  let seconds = null;
  // refused because of its size alone: the program is fine, the tab is too small
  let tooBig = false;
  if (table && !missing.length) {
    pk = packs(parsed.atoms, table);
    nao = spherical ? pk.naoSph : pk.naoCart;
    seconds = hfSeconds(pk.shells.reduce((s, sh) => s + sh[2] * sh[6], 0), nao);
    tooBig = seconds > limits.maxSeconds || nao > limits.maxAO;
    if (seconds > limits.maxSeconds) refusals.push(`${nao} basis functions, estimated at about ${Math.round(seconds)} s: more than the ${limits.maxSeconds} s a browser run allows (the two-electron integrals grow as the fourth power of the basis). Run it on your own machine with viv run.`);
    else if (nao > limits.maxAO) refusals.push(`${nao} basis functions: more than the ${limits.maxAO} a browser run allows. Run it on your own machine with viv run.`);
    const maxl = Math.max(...pk.shells.map((s) => s[1]));
    if (maxl >= 2 && !spherical) notes.push('Cartesian d functions: six per shell where spherical ones have five, so the energy differs from a spherical run (lib/hfref holds both for water).');
  }
  const na = (nelec + spin) / 2, nb = (nelec - spin) / 2;
  return {
    atoms: parsed.atoms, comment: parsed.comment, basis, charge, spin, method, spherical, nelec, nalpha: na, nbeta: nb,
    packs: pk, nao, seconds, settings, notes, refusals, tooBig,
  };
}

const row = (r) => `[${r.map((v) => pyRepr(Number(v))).join(', ')}]`;
const mat = (rows, indent) => `[${rows.map(row).join(`,\n${' '.repeat(indent)}`)}]`;

/**
 * The engine's input block: everything between its "input" and "end of input"
 * lines. `ref` is { energy } when lib/hfref holds this exact calculation.
 */
export function moleculeBlock(built, ref = null) {
  const p = built.packs;
  const formula = built.atoms.map((a) => a.el).join(' ');
  return [
    `# ---- input: ${formula}, ${built.basis}, charge ${built.charge}, Na - Nb = ${built.spin},`,
    `# written by the hf-energy engine's XYZ adapter (www/src/lib/engines/molecule.js)`,
    `MOL_ATOMS = ${mat(p.atoms, 13)}`,
    `MOL_SHELLS = ${mat(p.shells, 14)}`,
    `MOL_PRIMS = ${mat(p.prims, 13)}`,
    `MOL_CHARGE = ${pyRepr(built.charge)}`,
    `MOL_SPIN = ${pyRepr(built.spin)}`,
    `MOL_SPHERICAL = ${built.spherical ? '1.0' : '0.0'}`,
    `MOL_METHOD = ${pyRepr(built.method)}`,
    `HAS_REF = ${ref ? '1.0' : '0.0'}`,
    `REF_E = ${ref ? pyRepr(ref.energy) : '0.0'}`,
    '# ---- end of input',
  ].join('\n');
}

/** The resolved method for a built molecule: lib/hf's scf_method(o, na, nb) (lib/hf/scf_loop), which scf_setup stores as the setup's method. */
export function resolvedMethod(built) {
  if (built.method > 0) return built.method;
  return built.nalpha === built.nbeta ? 1 : 2;
}

/**
 * The lib/hfref calculation this one is, if any: same atoms (to 1e-10 bohr),
 * basis, AO kind, charge, spin and method. `refs` is engines.json's hf table.
 */
export function hfReferenceFor(built, refs) {
  if (!built.packs) return null;
  const m = resolvedMethod(built);
  const methodName = METHOD_NAMES[m];
  for (const r of refs) {
    if (r.basis !== built.basis || r.charge !== built.charge || r.spin !== built.spin || r.method !== methodName) continue;
    if (r.spherical !== null && r.spherical !== built.spherical) continue;
    if (r.atoms.length !== built.packs.atoms.length) continue;
    const same = r.atoms.every((a, i) => a.every((v, c) => Math.abs(v - built.packs.atoms[i][c]) < 1e-10));
    if (same) return r;
  }
  return null;
}
