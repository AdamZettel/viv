// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The Ising grid at its critical temperature, frame by frame: up spins glow,
// down spins are dark, and a frame fades into the next.
//
// Records read (posts/2026-09-25-metropolis-ising-against-onsager.viv): the
// plot2d frames of "A 32 x 32 grid at T_c, every two sweeps" (1 up, 0 down)
// and the plot of "|m| of the 32 x 32 grid at T_c" (x is the sweep).
import { P, fit, clear, clock, ease, fmt, alpha, rgb } from './kit.js';
import { figuresOf, figureNamed } from '../../lib/sceneRecords.js';

const PER = 0.42; // seconds per frame
const HOLD = 1.6;

function parse(text) {
  const figs = figuresOf(text);
  const g = figureNamed(figs, 'A 32 x 32 grid at T_c');
  const m = figureNamed(figs, '|m| of the 32 x 32 grid');
  if (!g || !g.frames.length || !m || !m.plots.length) return null;
  return { frames: g.frames, sweep: m.plots[0].xs, mag: m.plots[0].ys };
}

export default function create(canvas, { text, readout, reduced }) {
  const ctx = canvas.getContext('2d');
  let D = parse(text);
  const off = document.createElement('canvas');
  const octx = off.getContext('2d');

  function draw(t) {
    if (!D) return;
    const { w, h } = fit(canvas, ctx);
    clear(ctx, w, h);
    const n = D.frames.length;
    const cycle = (n - 1) * PER + HOLD;
    const p = reduced ? n - 1 : Math.min(n - 1, (t % cycle) / PER);
    const k = Math.floor(p);
    const f = ease((p - k) * 1.6);
    const A = D.frames[k].z;
    const B = D.frames[Math.min(k + 1, n - 1)].z;
    const { nx: rows, ny: cols } = D.frames[0];
    off.width = cols;
    off.height = rows;
    const img = octx.createImageData(cols, rows);
    // up: the cyan accent; down: the background, a little lifted toward the text
    const up = rgb(P.cyan);
    const bg = rgb(P.bg);
    const tx = rgb(P.text);
    const dn = bg.map((b, j) => b + 0.08 * (tx[j] - b));
    for (let i = 0; i < A.length; i++) {
      const v = A[i] + (B[i] - A[i]) * f;
      img.data[4 * i] = dn[0] + v * (up[0] - dn[0]);
      img.data[4 * i + 1] = dn[1] + v * (up[1] - dn[1]);
      img.data[4 * i + 2] = dn[2] + v * (up[2] - dn[2]);
      img.data[4 * i + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    const side = Math.min(w, h) - 8;
    const ox = (w - side) / 2;
    const oy = (h - side) / 2;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, ox, oy, side, side);
    // the glow: the same picture, blurred and added
    if (P.dark && 'filter' in ctx) {
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      ctx.globalAlpha = 0.55;
      ctx.filter = `blur(${Math.max(4, side / 70)}px)`;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, ox, oy, side, side);
      ctx.restore();
    }
    // the cell lines, faint
    ctx.strokeStyle = alpha(P.bg, 0.55);
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 1; i < cols; i++) {
      const x = Math.round(ox + (i * side) / cols) + 0.5;
      ctx.moveTo(x, oy);
      ctx.lineTo(x, oy + side);
    }
    for (let j = 1; j < rows; j++) {
      const y = Math.round(oy + (j * side) / rows) + 0.5;
      ctx.moveTo(ox, y);
      ctx.lineTo(ox + side, y);
    }
    ctx.stroke();
    ctx.strokeStyle = P.rule;
    ctx.strokeRect(ox + 0.5, oy + 0.5, side - 1, side - 1);
    const q = Math.round(p);
    readout(`sweep ${D.sweep[q]} · |m| = ${fmt(D.mag[q], 3)}`);
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
