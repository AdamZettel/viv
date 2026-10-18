// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Element symbols and atomic numbers, all 118: enough to recognize a species
// written in any of the formats the adapters read, and to name what the
// engines cannot do rather than guess.
export const SYMBOLS = [
  'H', 'He',
  'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
  'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
  'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
  'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
  'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
  'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
  'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
  'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
];

const BY_LOWER = new Map(SYMBOLS.map((s, i) => [s.toLowerCase(), i + 1]));

/** Atomic number of a symbol, case-insensitive; 0 if it is not an element. */
export function atomicNumber(sym) {
  return BY_LOWER.get(String(sym).toLowerCase()) || 0;
}

/** The symbol for an atomic number, or null. */
export function symbolOf(z) {
  return SYMBOLS[z - 1] || null;
}

/**
 * The element a label names. Formats decorate labels: CIF writes `Si1` or
 * `Ga1+`, QE species labels are often `Si1` or `Fe_up`, XYZ files sometimes
 * write `SI`. The rule: the leading letters, first uppercase then an optional
 * lowercase one, normalized; `null` when they are not an element.
 */
export function elementOf(label) {
  const m = String(label).trim().match(/^([A-Za-z])([a-z]?)/);
  if (!m) return null;
  const two = (m[1].toUpperCase() + m[2]);
  if (m[2] && atomicNumber(two)) return two;
  // a label like "SI" or "CL1" written all in capitals
  const caps = String(label).trim().match(/^([A-Z])([A-Z])(?![A-Za-z])/);
  if (caps && atomicNumber(caps[1] + caps[2].toLowerCase())) return caps[1] + caps[2].toLowerCase();
  const one = m[1].toUpperCase();
  if (atomicNumber(one)) return one;
  return null;
}
