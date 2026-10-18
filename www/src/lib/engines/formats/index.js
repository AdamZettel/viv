// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Which format a pasted or dropped file is, and its parse.
//
// The file name decides when it says something (.cif, POSCAR, .in, .xyz);
// otherwise the text does. Each parser returns the same shape:
//   { format, formatName, title, lattice (rows, bohr), atoms: [{ el, frac }], settings, notes }
// and throws a Refusal (crystal.js) when the file cannot be read.
import { parseCif, looksLikeCif } from './cif.js';
import { parsePoscar, looksLikePoscar } from './poscar.js';
import { parseQe, looksLikeQe } from './qe.js';
import { parseXyz, looksLikeXyz } from './xyz.js';
import { Refusal } from '../crystal.js';

export const FORMATS = [
  { id: 'poscar', name: 'POSCAR', parse: parsePoscar, looks: looksLikePoscar, ext: /(^|\/)(POSCAR|CONTCAR)[^/]*$|\.(poscar|vasp)$/i },
  { id: 'cif', name: 'CIF', parse: parseCif, looks: looksLikeCif, ext: /\.cif$/i },
  { id: 'qe-pw', name: 'pw.x input', parse: parseQe, looks: looksLikeQe, ext: /\.(in|pwi|pw)$/i },
  { id: 'xyz', name: 'extended XYZ', parse: parseXyz, looks: looksLikeXyz, ext: /\.(xyz|extxyz)$/i },
];

export function detectFormat(text, name = '') {
  const byName = FORMATS.find((f) => name && f.ext.test(name));
  if (byName) return byName;
  // the text: the most specific signatures first
  return [FORMATS[1], FORMATS[2], FORMATS[3], FORMATS[0]].find((f) => f.looks(text)) || null;
}

export function parseStructure(text, name = '', formatId = null) {
  if (!text || !text.trim()) throw new Refusal('The file is empty.');
  const f = formatId ? FORMATS.find((x) => x.id === formatId) : detectFormat(text, name);
  if (!f) throw new Refusal('Not a format this engine reads. It reads POSCAR, CIF, Quantum ESPRESSO pw.x input, and extended XYZ with a Lattice="..." cell.');
  return f.parse(text);
}
