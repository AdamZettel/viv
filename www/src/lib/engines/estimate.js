// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// What a deck will cost, before it runs: the engine's own sizes, computed the
// way the engine computes them, and a time from those sizes.
//
// The sizes are exact, not estimated. The k-mesh is dft/kmesh.viv's, the
// plane-wave count and the FFT grid are dft/pw_common.viv's pw_extent and
// fft_nice, ported loop for loop; enginetest.mjs checks them
// against what the engine prints (NKPT, NPWMAX, NG) for every example.
//
// The time is a model with one fitted constant. Nearly all the work is
// applying H to a band, which is two FFTs of the NG^3 grid, and that happens
// for every band at every k-point on every SCF iteration and at every point of
// the band path, a few LOBPCG steps each:
//
//     seconds ~ C * NBANDS * NG^3 log2(NG^3) * (NKPT * SCF_STEPS + NPTS * PATH_STEPS)
//
// SCF_STEPS and PATH_STEPS are typical counts (12 SCF iterations of NITER = 3
// steps; about 12 LOBPCG steps per path point from a cold start); C is fitted
// to measured runs of the compiled module in engines/cost.json (written by
// www/scripts/enginetest.mjs --fit), so the estimate says where it came from.
import { CORNER } from './deck.js';

const PI = 3.141592653589793;

function kwrap(x) {
  let y = x - Math.round(x);
  if (y < -0.5 + 1.0e-9) y = 0.5;
  return y;
}

/** dft/kmesh.viv: the four-shift fcc mesh, halved by time reversal. */
export function kmesh(n) {
  const shifts = [[0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]];
  const pts = [];
  const wts = [];
  for (const s of shifts) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        for (let l = 0; l < n; l++) {
          const k = [kwrap((i + s[0]) / n), kwrap((j + s[1]) / n), kwrap((l + s[2]) / n)];
          const m = k.map((x) => kwrap(0 - x));
          let found = -1;
          pts.forEach((q, qi) => { if (Math.abs(q[0] - m[0]) + Math.abs(q[1] - m[1]) + Math.abs(q[2] - m[2]) < 1e-8) found = qi; });
          const self = Math.abs(k[0] - m[0]) + Math.abs(k[1] - m[1]) + Math.abs(k[2] - m[2]);
          if (found >= 0 && self > 1e-8) wts[found] += 1;
          else { pts.push(k); wts.push(1); }
        }
      }
    }
  }
  return { kfrac: pts, weights: wts.map((w) => w / (4 * n * n * n)) };
}

/** dft/pw_common.viv: the band path, NSEG points per segment between corners. */
export function pathPoints(nseg) {
  const out = [];
  for (let c = 0; c < CORNER.length - 1; c++) {
    for (let t = 0; t < nseg; t++) out.push([0, 1, 2].map((d) => CORNER[c][d] + ((CORNER[c + 1][d] - CORNER[c][d]) * t) / nseg));
  }
  out.push(CORNER[CORNER.length - 1].slice());
  return out;
}

/** Reciprocal vectors, one per row, b_i . a_j = 2 pi delta_ij. */
function reciprocal(A) {
  const os = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
  const B = [];
  for (let i = 0; i < 3; i++) {
    const j = (i + 1) % 3, k = (i + 2) % 3;
    B.push([
      (2 * PI * (A[j][1] * A[k][2] - A[j][2] * A[k][1])) / os,
      (2 * PI * (A[j][2] * A[k][0] - A[j][0] * A[k][2])) / os,
      (2 * PI * (A[j][0] * A[k][1] - A[j][1] * A[k][0])) / os,
    ]);
  }
  return { B, omega: Math.abs(os) };
}

function fftNice(n) {
  let m = n;
  for (;;) {
    let r = m;
    for (const p of [2, 3, 5]) while (r % p === 0) r /= p;
    if (r === 1) return m;
    m += 1;
  }
}

/** The engine's sizes for a deck: NKPT, NPTS, NPWMAX, NG, NPROJ, and memory. */
export function sizes(deck) {
  const a = deck.ALAT;
  const A = [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]];
  const { B, omega } = reciprocal(A);
  const ecut = deck.ECUT;
  const amax = Math.max(...A.map((v) => Math.hypot(...v)));
  const bmax = Math.max(...B.map((v) => Math.hypot(...v)));
  const MM = Math.floor(((Math.sqrt(2 * ecut) + bmax) * amax) / (2 * PI)) + 1;
  const NM = 2 * MM + 1;
  // pw_extent: the count and the largest |m| in one pass per k-point
  const count = (k) => {
    let n = 0, big = 0;
    for (let i1 = 0; i1 < NM; i1++) {
      for (let i2 = 0; i2 < NM; i2++) {
        for (let i3 = 0; i3 < NM; i3++) {
          const m1 = i1 - MM, m2 = i2 - MM, m3 = i3 - MM;
          const f1 = m1 + k[0], f2 = m2 + k[1], f3 = m3 + k[2];
          const gx = f1 * B[0][0] + f2 * B[1][0] + f3 * B[2][0];
          const gy = f1 * B[0][1] + f2 * B[1][1] + f3 * B[2][1];
          const gz = f1 * B[0][2] + f2 * B[1][2] + f3 * B[2][2];
          if (0.5 * (gx * gx + gy * gy + gz * gz) <= ecut) {
            n += 1;
            big = Math.max(big, Math.abs(m1), Math.abs(m2), Math.abs(m3));
          }
        }
      }
    }
    return { n, big };
  };
  const mesh = kmesh(deck.NKMESH);
  const path = pathPoints(deck.NSEG);
  let npwmax = 0, mbasis = 0;
  for (const k of [...mesh.kfrac, ...path]) {
    const c = count(k);
    npwmax = Math.max(npwmax, c.n);
    mbasis = Math.max(mbasis, c.big);
  }
  const ng = fftNice(4 * mbasis + 2);
  // projector functions per k-point: every atom's every (l, m, i), l = 0..2
  const nprj = (s, l) => deck.NPRJ.data[s * 4 + l];
  let nproj = 0;
  for (const s of deck.ATSPEC.data) for (let l = 0; l < 3; l++) nproj += (2 * l + 1) * nprj(s, l);
  const nkpt = mesh.kfrac.length;
  const nslot = nkpt + 1;
  // the big arrays of dft/pwdft_it.viv: PSI and BETA (complex, 16 bytes), and
  // TK, IDX, MASK (real), per slot; the grids are small next to them
  const bytes = nslot * npwmax * (16 * (deck.NBANDS + nproj) + 24) + 12 * ng * ng * ng * 16;
  return { nkpt, npts: path.length, npwmax, ng, nproj, omega, memoryMB: bytes / 1048576 };
}

export const SCF_STEPS = 36; // 12 iterations x NITER = 3 LOBPCG steps
export const PATH_STEPS = 12;

/** The work measure the time model is proportional to. */
export function work(s, nbands) {
  const n3 = s.ng ** 3;
  return nbands * n3 * Math.log2(n3) * (s.nkpt * SCF_STEPS + s.npts * PATH_STEPS);
}

/** Estimated seconds for the compiled engine, from the fitted constant. */
export function estimateSeconds(s, nbands, cost) {
  return cost.c * work(s, nbands);
}
