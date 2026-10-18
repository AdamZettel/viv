// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// VASP's POSCAR (and CONTCAR): a comment, a scale, three lattice vectors, the
// species and their counts, then the positions, fractional ("Direct") or
// cartesian. Lengths are in angstrom.
//
//   Si2                          <- comment
//   1.0                          <- scale; a negative one is a cell volume, A^3
//   0.0 2.715 2.715              <- lattice vectors, one per line
//   2.715 0.0 2.715
//   2.715 2.715 0.0
//   Si                           <- species (VASP 5; VASP 4 files omit this line)
//   2                            <- how many of each
//   Direct                       <- optional "Selective dynamics" line before it
//   0.00 0.00 0.00
//   0.25 0.25 0.25
import { BOHR_ANGSTROM, Refusal, det3, toFrac } from '../crystal.js';
import { elementOf } from '../elements.js';

const nums = (line, n, what) => {
  const t = line.trim().split(/\s+/).slice(0, n).map(Number);
  if (t.length < n || t.some((v) => !Number.isFinite(v))) throw new Refusal(`POSCAR: expected ${n} numbers for ${what}, found "${line.trim()}".`);
  return t;
};

export function parsePoscar(text) {
  const lines = text.replace(/\r/g, '').split('\n');
  if (lines.length < 8) throw new Refusal('POSCAR: too short to hold a cell and its atoms.');
  const comment = lines[0].trim();
  const scaleTok = nums(lines[1], 1, 'the scale factor')[0];
  let L = [2, 3, 4].map((i) => nums(lines[i], 3, `lattice vector ${i - 1}`));
  let k = 5;
  let species;
  const notes = [];
  if (/^\s*[A-Za-z]/.test(lines[k])) {
    species = lines[k].trim().split(/\s+/);
    k += 1;
  } else {
    // VASP 4: the species are not in the file. The comment line is where
    // writers conventionally put them; use it only if it is exactly a list of
    // element symbols, and say so.
    const guess = comment.split(/\s+/);
    if (guess.length && guess.every((s) => elementOf(s) && elementOf(s) === s.replace(/[_\d].*$/, ''))) {
      species = guess;
      notes.push(`This is a VASP 4 POSCAR with no species line; the species were read from its first line, "${comment}".`);
    } else {
      throw new Refusal('POSCAR: no species line (a VASP 4 file). Add a line naming the elements, in order, above the counts line.');
    }
  }
  const counts = lines[k].trim().split(/\s+/).map(Number);
  if (counts.length !== species.length || counts.some((c) => !Number.isInteger(c) || c < 0)) {
    throw new Refusal(`POSCAR: ${species.length} species (${species.join(' ')}) but the counts line reads "${lines[k].trim()}".`);
  }
  k += 1;
  if (/^\s*s/i.test(lines[k])) k += 1; // Selective dynamics: the T/F flags after each position are ignored
  const mode = lines[k].trim();
  const cartesian = /^[ck]/i.test(mode);
  if (!/^[dck]/i.test(mode)) throw new Refusal(`POSCAR: expected "Direct" or "Cartesian", found "${mode}".`);
  k += 1;
  // the scale: a multiplier, or (negative) the volume the cell is scaled to
  let s = scaleTok;
  if (scaleTok < 0) s = Math.cbrt(-scaleTok / Math.abs(det3(L)));
  L = L.map((v) => v.map((x) => (x * s) / BOHR_ANGSTROM));
  const atoms = [];
  species.forEach((sp, i) => {
    const el = elementOf(sp);
    if (!el) throw new Refusal(`POSCAR: "${sp}" is not an element symbol.`);
    for (let n = 0; n < counts[i]; n++) {
      const line = lines[k];
      if (line === undefined || !line.trim()) throw new Refusal(`POSCAR: the counts promise ${counts.reduce((a, b) => a + b, 0)} atoms but the file ends after ${atoms.length}.`);
      let p = nums(line, 3, `atom ${atoms.length + 1}`);
      if (cartesian) p = toFrac(p.map((x) => (x * s) / BOHR_ANGSTROM), L);
      atoms.push({ el, frac: p, label: sp });
      k += 1;
    }
  });
  return { format: 'poscar', formatName: 'POSCAR (VASP)', title: comment, lattice: L, atoms, settings: {}, notes };
}

/** Does this text look like a POSCAR? Line 2 is one number, lines 3-5 three. */
export function looksLikePoscar(text) {
  const l = text.replace(/\r/g, '').split('\n');
  const n = (s, k) => s !== undefined && s.trim().split(/\s+/).length >= k && s.trim().split(/\s+/).slice(0, k).every((x) => Number.isFinite(Number(x)));
  return l.length > 7 && n(l[1], 1) && l[1].trim().split(/\s+/).length === 1 && n(l[2], 3) && n(l[3], 3) && n(l[4], 3);
}
