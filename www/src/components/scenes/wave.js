// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// A Langmuir wave decaying: above, the field across one wavelength,
// E(x, t) = Re(E1(t) e^{ikx}); below, |E1| on a log scale as it is traced,
// with the decay at the dispersion root's rate dashed through its peaks.
//
// Records read (posts/2026-09-25-landau-damping-from-the-dispersion-relation.viv):
// the plots "Re E1", "Im E1" and "decay at the root's rate" of the figure
// "The field of a Langmuir wave ...", and the shown values gamma_measured and
// gamma_root.
import { P, fit, clear, clock, fmt, label, alpha } from './kit.js';
import { figuresOf, figureNamed, shownOf } from '../../lib/sceneRecords.js';

const SPEED = 4; // time units per second
const HOLD = 2.5;

function parse(text) {
  const f = figureNamed(figuresOf(text), 'The field of a Langmuir wave');
  if (!f) return null;
  const get = (l) => f.plots.find((p) => p.label === l);
  const re = get('Re E1'), im = get('Im E1'), line = get("decay at the root's rate");
  if (!re || !im || !line) return null;
  const s = shownOf(text);
  return { t: re.xs, re: re.ys, im: im.ys, line: line.ys, gm: s.gamma_measured, gr: s.gamma_root };
}

export default function create(canvas, { text, readout, reduced }) {
  const ctx = canvas.getContext('2d');
  let D = parse(text);

  function draw(time) {
    if (!D) return;
    const { w, h } = fit(canvas, ctx);
    clear(ctx, w, h);
    const n = D.t.length;
    const tEnd = D.t[n - 1];
    const cycle = tEnd / SPEED + HOLD;
    const tt = reduced ? tEnd : Math.min(tEnd, (time % cycle) * SPEED);
    const k = Math.min(n - 1, Math.round((tt / tEnd) * (n - 1)));
    const mag = D.re.map((r, i) => Math.hypot(r, D.im[i]));
    const m0 = mag[0];
    // the field across one wavelength
    const top = h * 0.46;
    const mid = top / 2 + 6;
    const pad = 16;
    ctx.strokeStyle = P.grid;
    ctx.beginPath();
    ctx.moveTo(pad, mid);
    ctx.lineTo(w - pad, mid);
    ctx.stroke();
    const N = 160;
    const lTop = Math.log10(m0) + 0.2;
    const lBot = Math.log10(Math.min(...mag.filter((v) => v > 0))) - 0.2;
    const amp = Math.max(0.03, (Math.log10(Math.max(mag[k], 1e-300)) - lBot) / (lTop - lBot));
    ctx.beginPath();
    for (let i = 0; i <= N; i++) {
      const ph = (2 * Math.PI * i) / N;
      // the shape is Re(E1 e^{ikx}); its height is |E1| on the log scale below
      const e = (D.re[k] * Math.cos(ph) - D.im[k] * Math.sin(ph)) / (mag[k] || 1);
      const x = pad + (i / N) * (w - 2 * pad);
      const y = mid - e * amp * (top / 2 - 10);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.save();
    ctx.strokeStyle = P.cyan;
    ctx.lineWidth = 2.2;
    ctx.stroke();
    ctx.restore();
    label(ctx, 'E(x) over one wavelength, height on the log scale below', pad, 10);
    // |E1| on a log scale, traced up to now
    const y0 = top + 18, y1 = h - 14;
    const lmax = Math.log10(m0) + 0.2;
    const lmin = Math.log10(Math.min(...mag.filter((v) => v > 0))) - 0.2;
    const sx = (i) => pad + (D.t[i] / tEnd) * (w - 2 * pad);
    const sy = (v) => y0 + ((lmax - Math.log10(Math.max(v, 1e-300))) / (lmax - lmin)) * (y1 - y0);
    ctx.save();
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = alpha(P.amber, 0.8);
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    for (let i = 0; i < n; i++) (i ? ctx.lineTo : ctx.moveTo).call(ctx, sx(i), sy(D.line[i]));
    ctx.stroke();
    ctx.restore();
    ctx.beginPath();
    for (let i = 0; i <= k; i++) (i ? ctx.lineTo : ctx.moveTo).call(ctx, sx(i), sy(mag[i]));
    ctx.save();
    ctx.strokeStyle = P.mint;
    ctx.shadowColor = P.mint;
    ctx.shadowBlur = 8;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();
    label(ctx, '|E1|, log scale', pad, y0 - 6);
    label(ctx, "the root's rate", w - pad, y0 - 6, 'right', P.amber);
    readout(`t ω_pe = ${D.t[k].toFixed(1)} · γ = ${fmt(D.gm, 6)} measured, ${fmt(D.gr, 6)} from the root`);
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
