// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// CIF, the crystallographer's format: cell lengths and angles, the
// asymmetric unit's sites, and the space group's symmetry operations, which
// generate every other atom from those sites.
//
//   _cell_length_a 5.431
//   _cell_angle_alpha 90
//   loop_
//   _symmetry_equiv_pos_as_xyz        (or _space_group_symop_operation_xyz)
//   'x, y, z'
//   '-x+1/4, -y+1/4, z'  ...
//   loop_
//   _atom_site_label _atom_site_type_symbol _atom_site_fract_x ...
//   Si1 Si 0 0 0
//
// This reads the first data block. It expands the sites with the listed
// operations; a CIF that names a space group but lists no operations is
// refused, because generating them from the name would need the whole of the
// International Tables and this adapter carries none of it.
import { BOHR_ANGSTROM, Refusal } from '../crystal.js';
import { elementOf } from '../elements.js';

/** CIF tokens: bare words, quoted strings, and ;-delimited text fields. */
function tokenize(text) {
  const toks = [];
  const lines = text.replace(/\r/g, '').split('\n');
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith(';')) {
      // a text field runs to the next line that starts with ';'
      const body = [line.slice(1)];
      i += 1;
      while (i < lines.length && !lines[i].startsWith(';')) body.push(lines[i++]);
      toks.push({ v: body.join('\n'), quoted: true });
      continue;
    }
    let j = 0;
    while (j < line.length) {
      const c = line[j];
      if (c === ' ' || c === '\t') { j += 1; continue; }
      if (c === '#') break;
      if (c === "'" || c === '"') {
        // a quote closes only when followed by whitespace or the end of the line
        let e = j + 1;
        while (e < line.length && !(line[e] === c && (e + 1 === line.length || /\s/.test(line[e + 1])))) e += 1;
        toks.push({ v: line.slice(j + 1, e), quoted: true });
        j = e + 1;
        continue;
      }
      let e = j;
      while (e < line.length && !/\s/.test(line[e])) e += 1;
      toks.push({ v: line.slice(j, e), quoted: false });
      j = e;
    }
  }
  return toks;
}

/** The first data block as { tags: {name: value}, loops: [{ names, rows }] }. */
function parseBlock(text) {
  const toks = tokenize(text);
  const tags = {};
  const loops = [];
  let i = 0;
  let blocks = 0;
  while (i < toks.length) {
    const t = toks[i];
    const low = t.quoted ? '' : t.v.toLowerCase();
    if (low.startsWith('data_')) {
      blocks += 1;
      if (blocks > 1) break;
      i += 1;
    } else if (low === 'loop_') {
      i += 1;
      const names = [];
      while (i < toks.length && !toks[i].quoted && toks[i].v.startsWith('_')) names.push(toks[i++].v.toLowerCase());
      const vals = [];
      while (i < toks.length && (toks[i].quoted || !(toks[i].v.startsWith('_') || /^(loop_|data_)/i.test(toks[i].v)))) vals.push(toks[i++].v);
      const rows = [];
      for (let r = 0; r + names.length <= vals.length; r += names.length) rows.push(vals.slice(r, r + names.length));
      loops.push({ names, rows });
    } else if (!t.quoted && t.v.startsWith('_')) {
      tags[low] = toks[i + 1] ? toks[i + 1].v : '';
      i += 2;
    } else {
      i += 1;
    }
  }
  return { tags, loops, moreBlocks: blocks > 1 };
}

/** A CIF number: "5.431(2)" is 5.431, with its uncertainty dropped. */
const num = (s) => {
  if (s === undefined || s === '?' || s === '.') return NaN;
  return Number(String(s).replace(/\(\d+\)$/, ''));
};

/**
 * A symmetry operation, "-x+1/2, y, z+1/4", as a function on fractions.
 * Each component is a sum of signed terms: x, y or z with an optional
 * coefficient, or a constant (a fraction or a decimal).
 */
export function parseSymop(s) {
  const parts = s.toLowerCase().replace(/\s+/g, '').split(',');
  if (parts.length !== 3) throw new Refusal(`CIF: cannot read the symmetry operation "${s}".`);
  const rows = parts.map((p) => {
    const row = [0, 0, 0, 0];
    const terms = p.match(/[+-]?[^+-]+/g);
    if (!terms) throw new Refusal(`CIF: cannot read the symmetry operation "${s}".`);
    for (const term of terms) {
      const m = term.match(/^([+-]?)(\d*\.?\d*(?:\/\d+)?)\*?([xyz]?)$/);
      if (!m || (!m[2] && !m[3])) throw new Refusal(`CIF: cannot read "${term}" in the symmetry operation "${s}".`);
      const sign = m[1] === '-' ? -1 : 1;
      let c = 1;
      if (m[2]) {
        const [a, b] = m[2].split('/');
        c = b ? Number(a) / Number(b) : Number(a);
      }
      if (m[3]) row['xyz'.indexOf(m[3])] += sign * c;
      else row[3] += sign * c;
    }
    return row;
  });
  return (f) => rows.map((r) => r[0] * f[0] + r[1] * f[1] + r[2] * f[2] + r[3]);
}

/** The cell as three row vectors (angstrom), a along x and b in the xy plane. */
export function cellFromParameters(a, b, c, alpha, beta, gamma) {
  const r = Math.PI / 180;
  const ca = Math.cos(alpha * r), cb = Math.cos(beta * r), cg = Math.cos(gamma * r), sg = Math.sin(gamma * r);
  // exact zeros for right angles, so a cubic cell stays exactly cubic
  const z = (x) => (Math.abs(x) < 1e-12 ? 0 : x);
  const cx = c * cb;
  const cy = (c * (ca - cb * cg)) / sg;
  const cz = Math.sqrt(Math.max(0, c * c - cx * cx - cy * cy));
  return [[a, 0, 0], [z(b * cg), z(b * sg), 0], [z(cx), z(cy), cz]];
}

export function parseCif(text) {
  const { tags, loops, moreBlocks } = parseBlock(text);
  const notes = [];
  if (moreBlocks) notes.push('The file has more than one data block; the first was read.');
  const p = ['a', 'b', 'c'].map((x) => num(tags[`_cell_length_${x}`]));
  const ang = ['alpha', 'beta', 'gamma'].map((x) => num(tags[`_cell_angle_${x}`]));
  if (p.some((v) => !(v > 0))) throw new Refusal('CIF: the cell lengths (_cell_length_a, _b, _c) are missing.');
  ang.forEach((v, i) => { if (!Number.isFinite(v)) ang[i] = 90; });
  const Lang = cellFromParameters(...p, ...ang);
  const L = Lang.map((v) => v.map((x) => x / BOHR_ANGSTROM));

  const site = loops.find((l) => l.names.some((n) => n.startsWith('_atom_site_fract_')));
  if (!site) {
    if (loops.some((l) => l.names.some((n) => n.startsWith('_atom_site_cartn_')))) throw new Refusal('CIF: positions are given only as cartesian coordinates; this page reads fractional ones (_atom_site_fract_x/y/z).');
    throw new Refusal('CIF: no atom sites (_atom_site_fract_x ...) found.');
  }
  const col = (n) => site.names.indexOf(n);
  const iSym = col('_atom_site_type_symbol'), iLab = col('_atom_site_label');
  const ix = col('_atom_site_fract_x'), iy = col('_atom_site_fract_y'), iz = col('_atom_site_fract_z');
  const iOcc = col('_atom_site_occupancy');
  if (ix < 0 || iy < 0 || iz < 0) throw new Refusal('CIF: the atom-site loop lacks one of _atom_site_fract_x, _y, _z.');
  // a magnetic CIF (mCIF) gives each site a moment; the engine has no spin
  if (loops.some((l) => l.names.some((n) => n.startsWith('_atom_site_moment')))) {
    throw new Refusal('CIF: the sites carry magnetic moments (_atom_site_moment_...). The engine is spin-restricted LDA and cannot do a magnetic calculation.');
  }
  const sites = site.rows.map((row) => {
    const label = iSym >= 0 ? row[iSym] : row[iLab];
    const el = elementOf(label);
    if (!el) throw new Refusal(`CIF: "${label}" does not name an element.`);
    const occ = iOcc >= 0 ? num(row[iOcc]) : 1;
    if (Number.isFinite(occ) && Math.abs(occ - 1) > 1e-3) {
      throw new Refusal(`CIF: site ${row[iLab] || label} is partially occupied (occupancy ${row[iOcc]}). A disordered crystal needs a supercell model, which this page does not build.`);
    }
    const f = [row[ix], row[iy], row[iz]].map(num);
    if (f.some((v) => !Number.isFinite(v))) throw new Refusal(`CIF: site ${row[iLab] || label} has an unreadable position.`);
    return { el, frac: f, label: iLab >= 0 ? row[iLab] : label };
  });

  const symLoop = loops.find((l) => l.names.includes('_symmetry_equiv_pos_as_xyz') || l.names.includes('_space_group_symop_operation_xyz'));
  const sg = tags['_symmetry_space_group_name_h-m'] || tags['_space_group_name_h-m_alt'] || '';
  let ops;
  if (symLoop) {
    const k = symLoop.names.findIndex((n) => n === '_symmetry_equiv_pos_as_xyz' || n === '_space_group_symop_operation_xyz');
    ops = symLoop.rows.map((r) => parseSymop(r[k]));
  } else if (!sg || /^p\s*1$/i.test(sg.trim())) {
    ops = [(f) => f];
  } else {
    throw new Refusal(`CIF: the space group is ${sg.trim()} but the file lists no symmetry operations (_space_group_symop_operation_xyz), so the full set of atoms cannot be built. Add the symmetry operations, or list every atom in the cell with space group P 1.`);
  }
  // expand, then drop images that coincide (within 1e-3 of a cell edge)
  const atoms = [];
  const same = (f, g) => f.every((x, i) => { const d = Math.abs(x - g[i]); return Math.min(d, 1 - d) < 1e-3; });
  for (const s of sites) {
    for (const op of ops) {
      const f = op(s.frac).map((x) => x - Math.floor(x));
      if (!atoms.some((a) => a.el === s.el && same(a.frac, f))) atoms.push({ el: s.el, frac: f, label: s.label });
    }
  }
  if (ops.length > 1) notes.push(`${sites.length} site${sites.length > 1 ? 's' : ''} and ${ops.length} symmetry operations gave ${atoms.length} atoms in the cell.`);
  const title = tags['_chemical_formula_sum'] || tags['_chemical_name_common'] || tags['_chemical_formula_structural'] || '';
  return { format: 'cif', formatName: 'CIF', title: title.trim(), spaceGroup: sg.trim(), lattice: L, atoms, settings: {}, notes };
}

export const looksLikeCif = (text) => /^\s*data_/m.test(text) && /_cell_length_a/.test(text);
