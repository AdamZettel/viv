// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Two KdV solitons as a waterfall: one ridge line per recorded time, the
// newest at the back and lit, each hiding what is behind it.
//
// Records read (pde/kdv_soliton.viv): the plot2d of "Two solitons collide and
// pass through each other" (rows are times from 0 to the window's top, columns
// are x) and the shown value collision_error.
import { P, fit, clear, clock, fmt, alpha } from './kit.js';
import { figuresOf, figureNamed, shownOf } from '../../lib/sceneRecords.js';

const PER = 0.16; // seconds per row
const HOLD = 2.4;

function parse(text) {
  const f = figureNamed(figuresOf(text), 'Two solitons collide');
  if (!f || !f.frames.length) return null;
  return { fr: f.frames[0], win: f.window, err: shownOf(text).collision_error };
}

export default function create(canvas, { text, readout, reduced }) {
  const ctx = canvas.getContext('2d');
  let D = parse(text);

  function draw(t) {
    if (!D) return;
    const { w, h } = fit(canvas, ctx);
    clear(ctx, w, h);
    const { nx: rows, ny: cols, z } = D.fr;
    const cycle = rows * PER + HOLD;
    const shown = reduced ? rows : Math.min(rows, 1 + Math.floor((t % cycle) / PER));
    let umax = 0;
    for (const v of z) umax = Math.max(umax, v);
    const pad = 14;
    const depth = h * 0.5; // how far up the last row sits
    const amp = h * 0.32; // the height of the tallest wave
    const skew = w * 0.12; // later rows shift right, for depth
    const width = w - 2 * pad - skew;
    const base = h - pad;
    const X = (c, r) => pad + (c / (cols - 1)) * width + (r / (rows - 1)) * skew;
    const Y = (v, r) => base - (r / (rows - 1)) * depth - (v / umax) * amp;
    for (let r = shown - 1; r >= 0; r--) {
      const row = r * cols;
      ctx.beginPath();
      ctx.moveTo(X(0, r), Y(0, r) + 2);
      for (let c = 0; c < cols; c++) ctx.lineTo(X(c, r), Y(z[row + c], r));
      ctx.lineTo(X(cols - 1, r), Y(0, r) + 2);
      ctx.closePath();
      ctx.fillStyle = P.bg;
      ctx.fill();
      ctx.beginPath();
      for (let c = 0; c < cols; c++) (c ? ctx.lineTo : ctx.moveTo).call(ctx, X(c, r), Y(z[row + c], r));
      const lead = r === shown - 1;
      const age = (shown - 1 - r) / rows;
      ctx.save();
      ctx.strokeStyle = lead ? P.mint : alpha(P.cyan, 0.85 - 0.6 * age);
      ctx.shadowColor = lead ? P.mint : P.cyan;
      ctx.shadowBlur = lead ? 14 : 0;
      ctx.lineWidth = lead ? 2 : 1.1;
      ctx.stroke();
      ctx.restore();
    }
    const tEnd = D.win[3];
    const tNow = ((shown - 1) / (rows - 1)) * tEnd;
    readout(shown >= rows && Number.isFinite(D.err) ? `t = ${tNow.toFixed(1)} · largest gap to Hirota's exact solution ${fmt(D.err, 2)}` : `t = ${tNow.toFixed(1)}`);
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
