// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// From "a cell and some atoms", as any file format gives them, to the one
// description the DFT engine takes: the primitive fcc cell in dft/'s fixed
// orientation, with the atoms placed the way dft/ref/materials.py places them.
//
// The engine can only do fcc lattices (its k-mesh, dft/kmesh.viv, is the
// four-shift fcc mesh, and its band path is L - Gamma - X in fcc coordinates),
// and a file can describe an fcc crystal in many ways: the 8-atom cubic cell,
// the 2-atom primitive cell in any of several vector conventions, a rotated
// cell, a supercell. So this module:
//
//   1. finds every pure translation of the structure (a vector that maps each
//      atom onto an atom of the same element), which is how a conventional or
//      super cell reveals its smaller primitive cell;
//   2. builds a primitive basis from them and asks whether the lattice is fcc,
//      refusing (with the lattice it did find) if not;
//   3. finds the cube's edges among the lattice vectors, rotates everything
//      into that cubic frame, and writes the cell as dft/ does:
//         A = [[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]]
//   4. places the atoms as one compact cluster centered on the origin, and of
//      the cube's 24 rotations picks the one that puts the first atom furthest
//      along (1, 1, 1). For diamond and zincblende that is exactly the
//      materials.py convention (first atom at +1/8, second at -1/8); the rule
//      is written generally so that every input gets one deterministic answer.
//
// Lengths are in bohr throughout. Rounding at the very end (fractions to 1e-10
// of the cell, the lattice constant to 1e-9 bohr) removes the noise of unit
// conversions and rotations, so that the same crystal written in two formats
// gives the same deck text. Both are at or below the last digit structure
// files carry (1e-10 angstrom is about 2e-10 bohr).

export const BOHR_ANGSTROM = 0.529177210903; // CODATA 2018: 1 bohr in angstrom

const TOL = 2.0e-3; // bohr: two positions closer than this are the same site

// ---- small vector helpers --------------------------------------------------
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const norm = (a) => Math.sqrt(dot(a, a));
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const det3 = (m) => dot(m[0], cross(m[1], m[2]));
/** frac (row) times the lattice (rows): cartesian */
export const toCart = (f, L) => [0, 1, 2].map((c) => f[0] * L[0][c] + f[1] * L[1][c] + f[2] * L[2][c]);
/** cartesian to fractions of the lattice rows */
export function toFrac(x, L) {
  const d = det3(L);
  // Cramer's rule on x = f . L, i.e. L^T f = x
  const c0 = cross(L[1], L[2]), c1 = cross(L[2], L[0]), c2 = cross(L[0], L[1]);
  return [dot(x, c0) / d, dot(x, c1) / d, dot(x, c2) / d];
}
const wrap01 = (v) => v.map((x) => { const y = x - Math.floor(x); return y > 1 - 1e-9 ? 0 : y; });
// round to `places` decimals through the decimal string, so 10.68 stays 10.68
// (Math.round(x / q) * q would give 10.680000000000001)
const roundTo = (x, places) => { const r = Number(x.toFixed(places)); return Object.is(r, -0) ? 0 : r; };

/** An error the page shows as a refusal: what could not be done, and why. */
export class Refusal extends Error {
  constructor(message, detail) { super(message); this.refusal = true; this.detail = detail || null; }
}

/** Is fractional displacement d (mod 1) a translation of the lattice L within TOL? */
function isLatticeVector(d, L) {
  const r = d.map((x) => x - Math.round(x));
  return norm(toCart(r, L)) < TOL;
}

/**
 * The pure translations of a structure, as fractions of its own cell: every t
 * with atom i + t = an atom of the same element, for all i. Always includes 0.
 */
export function pureTranslations(L, atoms) {
  const counts = new Map();
  for (const a of atoms) counts.set(a.el, (counts.get(a.el) || 0) + 1);
  // the rarest element has the fewest candidates to try
  const rare = [...counts.entries()].sort((x, y) => x[1] - y[1])[0][0];
  const i0 = atoms.findIndex((a) => a.el === rare);
  const out = [];
  for (const cand of atoms) {
    if (cand.el !== rare) continue;
    const t = wrap01(sub(cand.frac, atoms[i0].frac));
    if (out.some((u) => isLatticeVector(sub(u, t), L))) continue;
    const maps = atoms.every((a) => atoms.some((b) => b.el === a.el && isLatticeVector(sub(add(a.frac, t), b.frac), L)));
    if (maps) out.push(t);
  }
  return out;
}

/** Lattice points (n + t) . L for small integer n, as cartesian vectors, shortest first. */
function latticePoints(L, trans, range) {
  const pts = [];
  for (let i = -range; i <= range; i++) {
    for (let j = -range; j <= range; j++) {
      for (let k = -range; k <= range; k++) {
        for (const t of trans) {
          const v = toCart([i + t[0], j + t[1], k + t[2]], L);
          const n = norm(v);
          if (n > TOL) pts.push({ v, n });
        }
      }
    }
  }
  pts.sort((a, b) => a.n - b.n);
  return pts;
}

/**
 * A primitive basis of the lattice generated by the cell L and the extra
 * translations: the shortest vector, the next shortest not parallel to it,
 * and the shortest that completes a cell of the primitive volume.
 */
function primitiveBasis(L, trans) {
  const vp = Math.abs(det3(L)) / trans.length;
  const pts = latticePoints(L, trans, 2);
  const v1 = pts[0].v;
  const v2 = pts.find((p) => norm(cross(v1, p.v)) > 1e-6 * p.n * pts[0].n).v;
  for (const p of pts) {
    const d = Math.abs(det3([v1, v2, p.v]));
    if (Math.abs(d - vp) < 1e-6 * vp) return { basis: [v1, v2, p.v], volume: vp, points: pts };
  }
  throw new Refusal('Could not find a primitive cell for this structure.', 'The lattice vectors may be nearly linearly dependent.');
}

/** Name the lattice, for a refusal that says what was found instead of fcc. */
function describeLattice(points, volume) {
  const d = points[0].n;
  const nn = points.filter((p) => Math.abs(p.n - d) < 1e-4 * d).length;
  const cubicA = { fcc: d * Math.SQRT2, bcc: (2 * d) / Math.sqrt(3), sc: d };
  if (nn === 12 && Math.abs(volume - cubicA.fcc ** 3 / 4) < 1e-4 * volume) return 'fcc';
  if (nn === 8 && Math.abs(volume - cubicA.bcc ** 3 / 2) < 1e-4 * volume) return 'body-centered cubic';
  if (nn === 6 && Math.abs(volume - cubicA.sc ** 3) < 1e-4 * volume) return 'simple cubic';
  if (nn === 12) return 'close-packed but not cubic';
  return 'not cubic';
}

// The 24 proper rotations of a cube, as signed permutation matrices with determinant +1.
const CUBE_ROTATIONS = (() => {
  const perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
  const out = [];
  for (const p of perms) {
    for (let s = 0; s < 8; s++) {
      const R = [0, 1, 2].map((r) => { const row = [0, 0, 0]; row[p[r]] = (s >> r) & 1 ? -1 : 1; return row; });
      if (det3(R) > 0) out.push(R);
    }
  }
  return out;
})();
const rotate = (R, x) => [dot(R[0], x), dot(R[1], x), dot(R[2], x)];

/**
 * The fcc primitive cell dft/ uses, for cubic lattice constant a (bohr).
 * Rows are the primitive vectors.
 */
export function fccCell(a) {
  return [[0.0, a / 2.0, a / 2.0], [a / 2.0, 0.0, a / 2.0], [a / 2.0, a / 2.0, 0.0]];
}

/**
 * Reduce a structure to the engine's canonical fcc description.
 *
 *   L      the cell, rows = lattice vectors, bohr
 *   atoms  [{ el, frac: [f1, f2, f3] }] in fractions of L
 *
 * Returns { alat, A, atoms: [{ el, frac }], nInput, nPrimitive, lattice: 'fcc' }
 * or throws a Refusal naming what the structure is instead.
 */
export function canonicalFcc(L, atomsIn) {
  if (!atomsIn.length) throw new Refusal('The structure has no atoms.');
  const vol = Math.abs(det3(L));
  if (!(vol > 1e-6)) throw new Refusal('The cell has no volume: its three vectors are linearly dependent.');
  const atoms = atomsIn.map((a) => ({ el: a.el, frac: wrap01(a.frac) }));
  // two atoms on one site is a broken file, not a structure
  for (let i = 0; i < atoms.length; i++) {
    for (let j = 0; j < i; j++) {
      if (isLatticeVector(sub(atoms[i].frac, atoms[j].frac), L)) {
        throw new Refusal(`Atoms ${j + 1} and ${i + 1} sit on the same site.`, 'Duplicate positions usually mean symmetry operations were applied to a file that already listed every atom.');
      }
    }
  }
  const trans = pureTranslations(L, atoms);
  if (atoms.length % trans.length !== 0) throw new Refusal(`The cell holds ${trans.length} copies of a smaller repeating unit, but its ${atoms.length} atoms do not divide evenly among them.`);
  const { basis, volume, points } = primitiveBasis(L, trans);
  const kind = describeLattice(points, volume);
  if (kind !== 'fcc') {
    throw new Refusal(`The lattice is ${kind}${/^not/.test(kind) ? ', so' : ','} not face-centered cubic.`,
      'The engine\'s k-mesh (dft/kmesh.viv) and band path (L - Gamma - X) are written for fcc crystals only: diamond, zincblende, rock salt, and fcc metals.');
  }
  // the cube's edges: lattice vectors of length a, three of them orthogonal
  const a = points[0].n * Math.SQRT2;
  const edges = points.filter((p) => Math.abs(p.n - a) < 1e-4 * a).map((p) => p.v);
  // prefer the edges closest to the input's own x and y axes, so an input that
  // is already cubic keeps its orientation
  const best = (dir, cands) => cands.reduce((m, v) => (dot(v, dir) > dot(m, dir) + 1e-9 ? v : m));
  const e1 = best([1, 0, 0], edges);
  const e2 = best([0, 1, 0], edges.filter((v) => Math.abs(dot(v, e1)) < 1e-4 * a * a));
  const e3 = scale(cross(e1, e2), 1 / a);
  const frame = [scale(e1, 1 / a), scale(e2, 1 / a), scale(e3, 1 / a)];
  const A = fccCell(a);
  // the primitive set: one atom per class of positions equal modulo the lattice
  const cart = [];
  for (const at of atoms) {
    const x = rotate(frame, toCart(at.frac, L));
    const f = toFrac(x, A);
    if (!cart.some((c) => c.el === at.el && isLatticeVector(sub(toFrac(c.x, A), f), A))) cart.push({ el: at.el, x });
  }
  const nPrim = atoms.length / trans.length;
  if (cart.length !== nPrim) throw new Refusal(`Reducing to the primitive cell left ${cart.length} atoms where ${nPrim} were expected.`);
  // one compact cluster: every atom at its periodic image nearest the first
  const images = [];
  for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) for (let k = -1; k <= 1; k++) images.push(toCart([i, j, k], A));
  // of the 24 cube rotations, the one that sends the first atom (relative to
  // the cluster's centroid) furthest along (1,1,1), ties broken toward +x, +y
  let chosen = null;
  for (const R of CUBE_ROTATIONS) {
    const xs = cart.map((c) => rotate(R, c.x));
    const x0 = xs[0];
    const cl = xs.map((x, n) => {
      if (n === 0) return x;
      let bestX = null;
      for (const g of images) {
        const f = toFrac(sub(x, x0), A).map((u) => u - Math.round(u));
        const y = add(add(x0, toCart(f, A)), g);
        const d = norm(sub(y, x0));
        // equally near images (diamond's four bonds are) are broken toward
        // -(1,1,1), which leaves the first atom ahead along +(1,1,1)
        if (!bestX || d < bestX.d - 1e-9 || (Math.abs(d - bestX.d) <= 1e-9 && y[0] + y[1] + y[2] < bestX.s - 1e-9)) bestX = { y, d, s: y[0] + y[1] + y[2] };
      }
      return bestX.y;
    });
    const c = scale(cl.reduce(add, [0, 0, 0]), 1 / cl.length);
    const rel = cl.map((x) => sub(x, c));
    const key = [rel[0][0] + rel[0][1] + rel[0][2], rel[0][0], rel[0][1]];
    const better = !chosen || key.some((v, n) => key.slice(0, n).every((u, m) => Math.abs(u - chosen.key[m]) < 1e-9) && v > chosen.key[n] + 1e-9);
    if (better) chosen = { key, rel };
  }
  const alat = roundTo(a, 9);
  const Ar = fccCell(alat);
  const out = chosen.rel.map((x, n) => ({ el: cart[n].el, frac: toFrac(x, A).map((v) => roundTo(v, 10)) }));
  return { lattice: 'fcc', alat, A: Ar, atoms: out, nInput: atoms.length, nPrimitive: out.length, frame };
}
