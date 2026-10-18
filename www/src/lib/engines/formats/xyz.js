// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// XYZ files: an atom count, a comment line, then "symbol x y z" in angstrom.
//
// A crystal needs a cell, and plain XYZ has none. The extended XYZ convention
// (ASE, and most tools that write periodic structures) puts it on the comment
// line, as nine numbers, a1 then a2 then a3:
//
//   2
//   Lattice="0 2.715 2.715 2.715 0 2.715 2.715 2.715 0" Properties=species:S:1:pos:R:3 pbc="T T T"
//   Si 0.0 0.0 0.0
//   Si 1.3575 1.3575 1.3575
//
// parseXyz reads that for the DFT engine (xyz+cell); parseMolecule reads a
// plain XYZ for the Hartree-Fock engine, where there is no cell.
import { BOHR_ANGSTROM, Refusal, toFrac } from '../crystal.js';
import { elementOf, symbolOf } from '../elements.js';

/** key=value pairs on an extended-XYZ comment line; values may be quoted. */
export function commentKeys(line) {
  const out = {};
  const re = /([A-Za-z_][A-Za-z0-9_-]*)\s*=\s*("([^"]*)"|'([^']*)'|(\S+))/g;
  let m;
  while ((m = re.exec(line))) out[m[1].toLowerCase()] = m[3] ?? m[4] ?? m[5];
  return out;
}

/** The atoms of an XYZ file: [{ el, x: [x, y, z] angstrom, label }] and its comment. */
export function readXyz(text, who = 'XYZ') {
  const lines = text.replace(/\r/g, '').split('\n');
  const n = Number(lines[0].trim());
  if (!Number.isInteger(n) || n < 1) throw new Refusal(`${who}: the first line should be the number of atoms; it reads "${lines[0].trim()}".`);
  const comment = lines[1] || '';
  const atoms = [];
  for (let i = 0; i < n; i++) {
    const line = lines[2 + i];
    if (line === undefined || !line.trim()) throw new Refusal(`${who}: the first line promises ${n} atoms but the file lists ${i}.`);
    const t = line.trim().split(/\s+/);
    // some writers give the atomic number instead of the symbol
    const el = /^\d+$/.test(t[0]) ? symbolOf(Number(t[0])) : elementOf(t[0]);
    if (!el) throw new Refusal(`${who}: "${t[0]}" on line ${i + 3} is not an element.`);
    const x = t.slice(1, 4).map(Number);
    if (x.length < 3 || x.some((v) => !Number.isFinite(v))) throw new Refusal(`${who}: cannot read the position on line ${i + 3}.`);
    atoms.push({ el, x, label: t[0] });
  }
  return { atoms, comment };
}

export function parseXyz(text) {
  const { atoms, comment } = readXyz(text, 'XYZ');
  const keys = commentKeys(comment);
  if (!keys.lattice) {
    throw new Refusal('XYZ: no cell. A crystal needs one: write it on the second line as Lattice="a1x a1y a1z a2x a2y a2z a3x a3y a3z" in angstrom (extended XYZ).');
  }
  const v = keys.lattice.trim().split(/\s+/).map(Number);
  if (v.length !== 9 || v.some((x) => !Number.isFinite(x))) throw new Refusal(`XYZ: Lattice="${keys.lattice}" should be nine numbers.`);
  if (keys.properties && /magmom/i.test(keys.properties)) {
    throw new Refusal('XYZ: the atoms carry magnetic moments (a magmoms property). The engine is spin-restricted LDA and cannot do a magnetic calculation.');
  }
  if (keys.pbc && !/^\s*(t|true|1)\s+(t|true|1)\s+(t|true|1)\s*$/i.test(keys.pbc)) {
    throw new Refusal(`XYZ: pbc="${keys.pbc}" is not periodic in all three directions; the engine computes bulk crystals.`);
  }
  const L = [v.slice(0, 3), v.slice(3, 6), v.slice(6, 9)].map((r) => r.map((x) => x / BOHR_ANGSTROM));
  return {
    format: 'xyz',
    formatName: 'extended XYZ (xyz + cell)',
    title: '',
    lattice: L,
    atoms: atoms.map((a) => ({ el: a.el, label: a.label, frac: toFrac(a.x.map((x) => x / BOHR_ANGSTROM), L) })),
    settings: {},
    notes: [],
  };
}

export const looksLikeXyz = (text) => /^\s*\d+\s*\n/.test(text.replace(/\r/g, ''));
