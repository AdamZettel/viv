// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The cart-pole under LQR: the cart on its track, the pole, and a fading
// trail of where the pole's tip has been.
//
// Records read (show/cartpole_lqr.viv): the animation's stage window and,
// per keyframe, the `rect: cart`, `segment: pole` and `circle: bob` records
// with each `frame:` duration. The angle in the readout is the pole's own
// segment, atan2 of its run over its rise.
import { P, fit, clear, clock, fmt, alpha } from './kit.js';
import { stageFrames } from '../../lib/sceneRecords.js';

const HOLD = 1.5;
const TRAIL = 46;

function parse(text) {
  const s = stageFrames(text);
  const frames = s.frames.filter((f) => f.cart && f.pole && f.bob);
  if (!s.stage || frames.length < 2) return null;
  const times = [];
  let acc = 0;
  for (const f of frames) {
    times.push(acc);
    acc += f.dt;
  }
  return { win: s.stage.window, frames, times, total: acc };
}

export default function create(canvas, { text, readout, reduced }) {
  const ctx = canvas.getContext('2d');
  let D = parse(text);

  function draw(t) {
    if (!D) return;
    const { w, h } = fit(canvas, ctx);
    clear(ctx, w, h);
    const [x0, x1, y0, y1] = D.win;
    const k0 = Math.min(w / (x1 - x0), h / (y1 - y0)) * 0.94;
    const cx = w / 2 - ((x0 + x1) / 2) * k0;
    const cy = h / 2 + ((y0 + y1) / 2) * k0;
    const X = (x) => cx + x * k0;
    const Y = (y) => cy - y * k0;
    const cycle = D.total + HOLD;
    const tt = reduced ? 0 : Math.min(D.total, t % cycle);
    let i = 0;
    while (i < D.frames.length - 1 && D.times[i + 1] <= tt) i++;
    const f = D.frames[i];
    // the track
    ctx.strokeStyle = P.rule;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(X(x0), Y(0));
    ctx.lineTo(X(x1), Y(0));
    ctx.stroke();
    for (let x = Math.ceil(x0 * 4) / 4; x <= x1; x += 0.25) {
      ctx.fillStyle = P.muted;
      ctx.fillRect(X(x) - 0.5, Y(0) + 4, 1, 5);
    }
    // where the tip has been
    for (let j = Math.max(0, i - TRAIL); j < i; j++) {
      const a = D.frames[j].bob, b = D.frames[j + 1].bob;
      const age = (i - j) / TRAIL;
      ctx.strokeStyle = alpha(P.mint, 0.75 * (1 - age));
      ctx.lineWidth = 2.4 * (1 - age) + 0.4;
      ctx.beginPath();
      ctx.moveTo(X(a[0]), Y(a[1]));
      ctx.lineTo(X(b[0]), Y(b[1]));
      ctx.stroke();
    }
    // the cart
    const [rx, ry, rw, rh] = f.cart;
    ctx.save();
    ctx.fillStyle = P.panel;
    ctx.strokeStyle = P.cyan;
    ctx.shadowColor = P.cyan;
    ctx.shadowBlur = 12;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(X(rx - rw / 2), Y(ry + rh / 2), rw * k0, rh * k0, 4) : ctx.rect(X(rx - rw / 2), Y(ry + rh / 2), rw * k0, rh * k0);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
    // the pole and its tip
    const [px, py, qx, qy] = f.pole;
    ctx.save();
    ctx.strokeStyle = P.text;
    ctx.shadowColor = P.aqua;
    ctx.shadowBlur = 16;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(X(px), Y(py));
    ctx.lineTo(X(qx), Y(qy));
    ctx.stroke();
    const [bx, by, br] = f.bob;
    ctx.fillStyle = P.mint;
    ctx.shadowColor = P.mint;
    ctx.shadowBlur = 20;
    ctx.beginPath();
    ctx.arc(X(bx), Y(by), Math.max(4, br * k0), 0, 7);
    ctx.fill();
    ctx.restore();
    const deg = (Math.atan2(qx - px, qy - py) * 180) / Math.PI;
    readout(`t = ${D.times[i].toFixed(2)} s · pole ${fmt(deg, 3)}° from upright · cart at x = ${fmt(rx, 3)}`);
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
