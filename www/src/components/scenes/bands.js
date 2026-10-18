// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Silicon's band structure: the readout steps through the self-consistent
// iterations, then the bands are drawn along L - Γ - X and the gap between
// the highest filled band and the lowest empty one lights up.
//
// Records read: the recorded run of dft/ in src/data/dft.json (scripts/dft-
// build.mjs runs the program and keeps its records): `x` and `bands` (the
// band-path records, eV from the valence-band maximum), `drho` (the density
// change per SCF iteration) and `shown.GAP`.
import { P, fit, clear, clock, fmt, label, alpha } from './kit.js';

const ITER = 0.28; // seconds per SCF iteration in the readout
const DRAW = 2.6; // seconds to draw the bands along the path
const HOLD = 3;

export default function create(canvas, { data, readout, reduced }) {
  const ctx = canvas.getContext('2d');
  const R = data;

  function draw(t) {
    const { w, h } = fit(canvas, ctx);
    clear(ctx, w, h);
    const { x, bands, drho } = R;
    const nIt = drho.length;
    const cycle = nIt * ITER + DRAW + HOLD;
    const tt = reduced ? cycle : t % cycle;
    const pad = { l: 40, r: 14, t: 14, b: 26 };
    const lo = Math.floor(Math.min(...bands[0]) - 0.8);
    const hi = 7;
    const sx = (v) => pad.l + ((v - x[0]) / (x[x.length - 1] - x[0])) * (w - pad.l - pad.r);
    const sy = (v) => pad.t + ((hi - v) / (hi - lo)) * (h - pad.t - pad.b);
    // the symmetry points
    const nseg = (x.length - 1) / 2;
    for (const [i, name] of [[0, 'L'], [nseg, 'Γ'], [x.length - 1, 'X']]) {
      ctx.strokeStyle = P.grid;
      ctx.beginPath();
      ctx.moveTo(sx(x[i]), pad.t);
      ctx.lineTo(sx(x[i]), h - pad.b);
      ctx.stroke();
      label(ctx, name, sx(x[i]), h - pad.b + 13, 'center', P.body);
    }
    for (let e = Math.ceil(lo / 4) * 4; e <= hi; e += 4) label(ctx, `${e}`, pad.l - 8, sy(e), 'right');
    label(ctx, 'eV', pad.l - 8, pad.t, 'right');
    const filled = bands.filter((b) => Math.max(...b) <= 1e-6);
    const empty = bands.filter((b) => Math.min(...b) > 1e-6);
    const cbm = empty.length ? Math.min(...empty.map((b) => Math.min(...b))) : null;
    const prog = Math.max(0, Math.min(1, (tt - nIt * ITER) / DRAW));
    const upto = Math.round(prog * (x.length - 1));
    // the gap, once the bands are drawn
    if (cbm !== null && prog >= 1) {
      ctx.fillStyle = alpha(P.amber, 0.1);
      ctx.fillRect(pad.l, sy(cbm), w - pad.l - pad.r, sy(0) - sy(cbm));
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = alpha(P.amber, 0.7);
      for (const v of [0, cbm]) {
        ctx.beginPath();
        ctx.moveTo(pad.l, sy(v));
        ctx.lineTo(w - pad.r, sy(v));
        ctx.stroke();
      }
      ctx.restore();
    }
    // before the bands are traced, all of them faintly: the quiet baseline
    if (prog < 1) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(pad.l, pad.t, w - pad.l - pad.r, h - pad.t - pad.b);
      ctx.clip();
      ctx.strokeStyle = alpha(P.body, 0.25);
      ctx.lineWidth = 1;
      for (const b of bands) {
        ctx.beginPath();
        b.forEach((e, i) => (i ? ctx.lineTo(sx(x[i]), sy(e)) : ctx.moveTo(sx(x[i]), sy(e))));
        ctx.stroke();
      }
      ctx.restore();
    }
    if (upto > 0) {
      for (const b of bands) {
        const occ = filled.includes(b);
        ctx.save();
        ctx.beginPath();
        ctx.rect(pad.l, pad.t, w - pad.l - pad.r, h - pad.t - pad.b);
        ctx.clip();
        ctx.beginPath();
        for (let i = 0; i <= upto; i++) (i ? ctx.lineTo : ctx.moveTo).call(ctx, sx(x[i]), sy(Math.max(lo - 1, Math.min(hi + 1, b[i]))));
        ctx.strokeStyle = occ ? P.cyan : P.violet;
        ctx.lineWidth = 1.8;
        ctx.stroke();
        ctx.restore();
      }
    }
    if (tt < nIt * ITER) {
      const k = Math.min(nIt - 1, Math.floor(tt / ITER));
      readout(`SCF iteration ${k + 1} · density change ${fmt(drho[k], 2)}`);
    } else if (prog < 1) readout(`converged in ${nIt} iterations · drawing the bands`);
    else readout(`gap ${fmt(R.shown.GAP, 3)} eV, LDA`);
  }

  const c = clock(draw);
  return {
    start: () => (reduced ? draw(0) : c.start()),
    stop: c.stop,
    redraw: c.redraw,
    setText() {},
    ok: () => true,
  };
}
