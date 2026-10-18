// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// One figure() of a run drawn as glowing curves that trace in from left to
// right, with the values at the tracing point in the readout. Linear or log
// axes, as the program set them; lines and dots; the figure's hline/vline.
//
// Records read: the figure whose title starts with `fig` (default: the first),
// its plot and scatter records, xscale/yscale, hline and vline. Robertson's
// kinetics (show/robertson.viv) is drawn this way on the home page, and the
// /show pages without a scene of their own use it for their hero.
import { P, fit, clear, clock, fmt, label, alpha } from './kit.js';
import { figuresOf, figureNamed } from '../../lib/sceneRecords.js';

const DRAW = 4.5;
const HOLD = 2.5;
// the data series in order (cyan, amber, indigo, green): two curves are
// never the same color
const COLORS = () => [P.cyan, P.amber, P.violet, P.rose];

function parse(text, fig) {
  const figs = figuresOf(text).filter((f) => f.plots.length);
  const f = fig ? figureNamed(figs, fig) : figs[0];
  return f || null;
}

// a tick's label: the shortest spelling, a true minus sign
const tick = (v) => String(+(Math.abs(v) < 1e-12 ? 0 : v).toPrecision(4)).replace(/^-/, '−');

const niceStep = (span) => {
  const raw = span / 5;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
};

export default function create(canvas, { text, readout, reduced, fig }) {
  const ctx = canvas.getContext('2d');
  let F = parse(text, fig);

  function draw(t) {
    if (!F) return;
    const { w, h } = fit(canvas, ctx);
    clear(ctx, w, h);
    const lx = F.xscale === 'log', ly = F.yscale === 'log';
    const tx = (v) => (lx ? Math.log10(v) : v);
    const ty = (v) => (ly ? Math.log10(v) : v);
    const xs = F.plots.flatMap((p) => p.xs).filter((v) => !lx || v > 0);
    const ys = F.plots.flatMap((p) => p.ys).filter((v) => Number.isFinite(v) && (!ly || v > 0));
    let [x0, x1] = [Math.min(...xs.map(tx)), Math.max(...xs.map(tx))];
    let [y0, y1] = [Math.min(...ys.map(ty)), Math.max(...ys.map(ty))];
    if (F.window && !lx) [x0, x1] = [F.window[0], F.window[1]];
    if (F.window && !ly && F.window[3] > F.window[2]) [y0, y1] = [F.window[2], F.window[3]];
    // dots sit inside the frame, not on its edge
    if (F.plots.some((p) => p.dots)) {
      const xr = x1 - x0 || 1;
      x0 -= xr * 0.05;
      x1 += xr * 0.05;
    }
    const yr = y1 - y0 || 1;
    y0 -= yr * 0.06;
    y1 += yr * 0.06;
    const pad = { l: 46, r: 14, t: 16, b: 30 };
    const sx = (v) => pad.l + ((tx(v) - x0) / (x1 - x0 || 1)) * (w - pad.l - pad.r);
    const sy = (v) => pad.t + ((y1 - ty(v)) / (y1 - y0)) * (h - pad.t - pad.b);
    // ticks
    const xStep = lx ? Math.max(1, Math.round((x1 - x0) / 5)) : niceStep(x1 - x0);
    for (let v = Math.ceil(x0 / xStep) * xStep; v <= x1 + 1e-9; v += xStep) {
      const X = pad.l + ((v - x0) / (x1 - x0)) * (w - pad.l - pad.r);
      ctx.strokeStyle = P.grid;
      ctx.beginPath();
      ctx.moveTo(X, pad.t);
      ctx.lineTo(X, h - pad.b);
      ctx.stroke();
      label(ctx, lx ? `1e${Math.round(v)}` : tick(v), X, h - pad.b + 13, 'center');
    }
    const yStep = ly ? Math.max(1, Math.round((y1 - y0) / 4)) : niceStep(y1 - y0);
    for (let v = Math.ceil(y0 / yStep) * yStep; v <= y1 + 1e-9; v += yStep) {
      const Y = pad.t + ((y1 - v) / (y1 - y0)) * (h - pad.t - pad.b);
      ctx.strokeStyle = P.grid;
      ctx.beginPath();
      ctx.moveTo(pad.l, Y);
      ctx.lineTo(w - pad.r, Y);
      ctx.stroke();
      label(ctx, ly ? `1e${Math.round(v)}` : tick(v), pad.l - 6, Y, 'right');
    }
    const cycle = DRAW + HOLD;
    const prog = reduced ? 1 : Math.min(1, (t % cycle) / DRAW);
    const xNow = x0 + prog * (x1 - x0);
    ctx.save();
    ctx.beginPath();
    ctx.rect(pad.l, pad.t, w - pad.l - pad.r, h - pad.t - pad.b);
    ctx.clip();
    for (const l of F.lines) {
      if (l.kind !== 'hline' && l.kind !== 'vline') continue;
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = alpha(P.amber, 0.7);
      ctx.beginPath();
      if (l.kind === 'hline') {
        ctx.moveTo(pad.l, sy(l.at));
        ctx.lineTo(w - pad.r, sy(l.at));
      } else {
        ctx.moveTo(sx(l.at), pad.t);
        ctx.lineTo(sx(l.at), h - pad.b);
      }
      ctx.stroke();
      ctx.restore();
    }
    const now = [];
    F.plots.forEach((p, s) => {
      const col = COLORS()[s % COLORS().length];
      const pts = p.xs.map((x, i) => [x, p.ys[i]]).filter(([x, y]) => Number.isFinite(y) && (!lx || x > 0) && (!ly || y > 0) && tx(x) <= xNow + 1e-12);
      if (!pts.length) return;
      ctx.save();
      ctx.strokeStyle = col;
      ctx.fillStyle = col;
      ctx.lineWidth = 1.8;
      if (p.dots) {
        for (const [x, y] of pts) {
          ctx.beginPath();
          ctx.arc(sx(x), sy(y), 3.2, 0, 7);
          ctx.fill();
        }
      } else {
        ctx.beginPath();
        pts.forEach(([x, y], i) => (i ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y))));
        ctx.stroke();
      }
      ctx.restore();
      now.push([p.label, pts[pts.length - 1], !!p.dots]);
    });
    ctx.restore();
    // the lines at the tracing point; a figure of dots only, its last points
    const lines = now.filter((n) => !n[2]);
    const pick = lines.length ? lines : now;
    if (pick.length) {
      const x = pick[0][1][0];
      readout([`${F.xlabel || 'x'} = ${fmt(x, 3)}`, ...pick.slice(0, 3).map(([l, [, y]]) => `${l} = ${fmt(y, 4)}`)].join(' · '));
    }
  }

  const c = clock(draw);
  return {
    start: () => (reduced ? draw(0) : c.start()),
    stop: c.stop,
    redraw: c.redraw,
    setText(t) {
      const f = parse(t, fig);
      if (f) F = f;
      c.reset();
      c.redraw();
    },
    ok: () => !!F,
  };
}
