// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Water's electron density, one Roothaan step per frame, as glowing
// iso-contours that morph from frame to frame, with the energy of each
// frame's density as a sparkline underneath.
//
// The contour technique is ported from the owner's ScfConvergence.astro
// (notesoncomputing.com), whose header reads: "ScfConvergence — a home
// flagship section: a real SCF's electron density converging, drawn as glowing
// iso-contours (marching squares) that morph between the per-iteration density
// matrices emitted by a Viv program, with the energy trace settling alongside."
//
// Records read (posts/2026-09-24-hartree-fock-water.viv): the plot2d frames of
// the figure "Water's electron density, one plain Roothaan step per frame"
// (log10 of the density, bohr^-3, row 0 at the bottom), its label marks (the
// atoms), and the plot of "The energy of each step's density".
import { P, fit, clear, clock, ease, fmt, label, alpha } from './kit.js';
import { figuresOf, figureNamed } from '../../lib/sceneRecords.js';

// the contour levels, log10 of the density: every quarter decade from -2 to 1.5
const LEVEL_STEP = 0.25;
const LEVELS = Array.from({ length: Math.round((1.5 - -2) / LEVEL_STEP) + 1 }, (_, i) => -2 + i * LEVEL_STEP);
const SEG = 1.2; // seconds per step
const HOLD = 2.4; // seconds on the converged density before it starts again

function parse(text) {
  const figs = figuresOf(text);
  const dens = figureNamed(figs, "Water's electron density");
  const en = figureNamed(figs, "The energy of each step's density");
  if (!dens || !dens.frames.length || !en || !en.plots.length) return null;
  return { win: dens.window, frames: dens.frames, atoms: dens.lines.filter((l) => l.kind === 'label'), energy: en.plots[0].ys };
}

export default function create(canvas, { text, readout, reduced }) {
  const ctx = canvas.getContext('2d');
  let D = parse(text);
  let buf = null;
  const field = (p) => {
    const n = D.frames.length;
    const k = Math.min(Math.floor(p), n - 1);
    const A = D.frames[k].z;
    const B = D.frames[Math.min(k + 1, n - 1)].z;
    const f = ease(p - k);
    if (!buf || buf.length !== A.length) buf = new Float32Array(A.length);
    for (let i = 0; i < A.length; i++) buf[i] = A[i] + (B[i] - A[i]) * f;
    return buf;
  };

  function draw(t) {
    if (!D) return;
    const { w, h } = fit(canvas, ctx);
    clear(ctx, w, h);
    const n = D.frames.length;
    const cycle = (n - 1) * SEG + HOLD;
    const p = reduced ? n - 1 : Math.min(n - 1, (t % cycle) / SEG);
    const spark = Math.min(84, h * 0.2);
    const side = Math.min(w, h - spark - 10);
    const ox = (w - side) / 2;
    const { nx: rows, ny: cols } = D.frames[0];
    const fr = field(p);
    const px = (gx) => ox + ((gx + 0.5) / cols) * side;
    const py = (gy) => side - ((gy + 0.5) / rows) * side;
    ctx.globalCompositeOperation = P.dark ? 'lighter' : 'source-over';
    ctx.lineJoin = 'round';
    for (let li = 0; li < LEVELS.length; li++) {
      const L = LEVELS[li];
      const tt = li / (LEVELS.length - 1);
      // cyan outside toward the indigo of the core; deeper on a light background
      const hue = 192 + tt * 40;
      const light = P.dark ? 58 + tt * 14 : 34 - tt * 6;
      ctx.strokeStyle = `hsla(${hue},${P.dark ? 95 : 80}%,${light}%,0.95)`;
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      for (let j = 0; j < rows - 1; j++) {
        const rA = j * cols;
        const rB = (j + 1) * cols;
        for (let i = 0; i < cols - 1; i++) {
          const a = fr[rA + i], b = fr[rA + i + 1], c = fr[rB + i + 1], d = fr[rB + i];
          let idx = 0;
          if (a > L) idx |= 1;
          if (b > L) idx |= 2;
          if (c > L) idx |= 4;
          if (d > L) idx |= 8;
          if (idx === 0 || idx === 15) continue;
          const Tx = px(i + (L - a) / (b - a)), Ty = py(j);
          const Rx = px(i + 1), Ry = py(j + (L - b) / (c - b));
          const Bx = px(i + (L - d) / (c - d)), By = py(j + 1);
          const Lx = px(i), Ly = py(j + (L - a) / (d - a));
          const seg = (x0, y0, x1, y1) => {
            ctx.moveTo(x0, y0);
            ctx.lineTo(x1, y1);
          };
          switch (idx) {
            case 1: case 14: seg(Lx, Ly, Tx, Ty); break;
            case 2: case 13: seg(Tx, Ty, Rx, Ry); break;
            case 3: case 12: seg(Lx, Ly, Rx, Ry); break;
            case 4: case 11: seg(Rx, Ry, Bx, By); break;
            case 6: case 9: seg(Tx, Ty, Bx, By); break;
            case 7: case 8: seg(Lx, Ly, Bx, By); break;
            case 5: seg(Lx, Ly, Tx, Ty); seg(Rx, Ry, Bx, By); break;
            case 10: seg(Tx, Ty, Rx, Ry); seg(Lx, Ly, Bx, By); break;
          }
        }
      }
      ctx.stroke();
    }
    ctx.shadowBlur = 0;
    ctx.globalCompositeOperation = 'source-over';
    // the nuclei, where the program placed them
    const [x0, x1, y0, y1] = D.win;
    for (const a of D.atoms) {
      const ax = ox + ((a.x - x0) / (x1 - x0)) * side;
      const ay = side - ((a.y - y0) / (y1 - y0)) * side;
      ctx.strokeStyle = P.aqua;
      ctx.lineWidth = 1.3;
      ctx.beginPath();
      ctx.arc(ax, ay, a.text === 'O' ? 7 : 4.5, 0, 7);
      ctx.stroke();
      label(ctx, a.text, ax + 11, ay - 9, 'left', P.aqua);
    }
    // the energy of each step's density above the last step's, on a log
    // scale, with the frame's step lit
    const E = D.energy;
    const last = E[E.length - 1];
    const gap = E.map((e) => Math.log10(Math.max(e - last, 1e-12)));
    const lo = Math.min(...gap.slice(0, -1)), hi = Math.max(...gap);
    const top = h - spark + 16, bot = h - 8;
    const sx = (i) => ox + 8 + (i / (E.length - 2)) * (side - 16);
    const sy = (g) => top + (1 - (g - lo) / (hi - lo || 1)) * (bot - top);
    ctx.beginPath();
    gap.slice(0, -1).forEach((g, i) => (i ? ctx.lineTo(sx(i), sy(g)) : ctx.moveTo(sx(i), sy(g))));
    ctx.strokeStyle = alpha(P.cyan, 0.8);
    ctx.lineWidth = 1.5;
    ctx.stroke();
    label(ctx, 'energy above the last step, log scale', ox + 8, top - 9);
    const k = Math.round(p);
    ctx.fillStyle = P.mint;
    ctx.shadowColor = P.mint;
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(sx(k), sy(gap[k]), 3.5, 0, 7);
    ctx.fill();
    ctx.shadowBlur = 0;
    readout(`step ${k + 1} · E = ${fmt(E[k], 10)} hartree`);
  }

  const c = clock(draw);
  return {
    start: () => (reduced ? draw(0) : c.start()),
    stop: c.stop,
    redraw: c.redraw,
    setText(t) {
      const d = parse(t);
      if (d) D = d;
      c.reset();
      c.redraw();
    },
    ok: () => !!D,
  };
}
