// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// What every scene shares: a palette read from the site's own tokens in the
// reader's light or dark setting, a canvas sized to its box at the device's
// pixel ratio, strokes that glow in dark mode and are crisp in light mode,
// and a clock that runs only while the scene is on screen.
//
// A scene module's default export is create(canvas, {text, readout, reduced})
// and returns {start, stop, redraw, setText, ok}: `text` is the program's
// output, `readout(str)` sets the scene's readout line, `reduced` draws one
// still frame.

// Data is drawn in the data palette (src/lib/dataPalette.mjs, as the CSS
// tokens --data-1 .. --data-4 of src/styles/data-palette.css): cyan, amber,
// indigo, green. P.cyan is series 1 and the active highlight; P.aqua a
// lighter or deeper step of it for marks; P.mint, the "now" mark, is cyan too
// (one accent for the active thing); P.violet and P.rose are series 3 and 4.
import { DATA_PALETTE } from '../../lib/dataPalette.mjs';
const role = (m) => {
  const [c, a, i, g] = DATA_PALETTE[m].series;
  return { cyan: c, aqua: DATA_PALETTE[m].text, mint: c, amber2: a, violet: i, rose: g, glow: DATA_PALETTE[m].glow };
};

/** The scene palette, refreshed from the site's tokens by fit(). */
export const P = { dark: true, bg: '#150e21', panel: '#221834', rule: '#33254c', grid: 'rgba(160,150,190,0.12)', text: '#e7e1f2', body: '#a99cc0', muted: '#7b6d94', amber: '#fbbf24', ...role('dark') };

let themeKey = '';
function refresh(el) {
  const root = document.documentElement;
  const key = `${root.dataset.theme || ''}|${root.dataset.side || ''}`;
  if (key === themeKey) return;
  themeKey = key;
  const cs = getComputedStyle(el);
  const v = (n, d) => (cs.getPropertyValue(n) || '').trim() || d;
  const dark = root.dataset.theme ? root.dataset.theme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
  const r = role(dark ? 'dark' : 'light');
  const data = (i) => v(`--data-${i}`, '');
  Object.assign(P, r, {
    dark,
    bg: v('--bg', dark ? '#150e21' : '#fbfafd'),
    panel: v('--bg-card', dark ? '#221834' : '#ffffff'),
    rule: v('--border', dark ? '#33254c' : '#e0daea'),
    text: v('--text', dark ? '#e7e1f2' : '#2c2338'),
    body: v('--text-muted', dark ? '#a99cc0' : '#6b5f80'),
    muted: v('--text-faint', dark ? '#7b6d94' : '#8d82a1'),
  });
  // the CSS tokens win when the page has them (one source, two readers)
  P.cyan = data(1) || r.cyan;
  P.mint = P.cyan;
  P.amber = data(2) || r.amber2;
  P.violet = data(3) || r.violet;
  P.rose = data(4) || r.rose;
  P.grid = alpha(P.text, dark ? 0.09 : 0.08);
}

/** A color with an alpha: '#rrggbb' or 'rgb(...)' to 'rgba(...)'. */
export function alpha(color, a) {
  const c = String(color).trim();
  if (c.startsWith('#')) {
    const h = c.length === 4 ? c.slice(1).split('').map((x) => x + x).join('') : c.slice(1, 7);
    return `rgba(${parseInt(h.slice(0, 2), 16)},${parseInt(h.slice(2, 4), 16)},${parseInt(h.slice(4, 6), 16)},${a})`;
  }
  const m = c.match(/rgba?\(([^)]+)\)/);
  if (m) return `rgba(${m[1].split(',').slice(0, 3).join(',')},${a})`;
  return c;
}

/** A color as [r, g, b]. */
export function rgb(color) {
  const m = alpha(color, 1).match(/rgba\((\d+),(\d+),(\d+)/);
  return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : [128, 128, 128];
}

/** In light mode a context draws no glow: shadowBlur is held at zero. */
function glowFor(ctx) {
  if (P.dark) delete ctx.shadowBlur;
  else if (!Object.prototype.hasOwnProperty.call(ctx, 'shadowBlur')) {
    Object.defineProperty(ctx, 'shadowBlur', { configurable: true, get: () => 0, set: () => {} });
  }
}

/** Size a canvas to its CSS box; returns {w, h} in CSS pixels with the context scaled. */
export function fit(canvas, ctx) {
  refresh(canvas);
  glowFor(ctx);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const r = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width));
  const h = Math.max(1, Math.round(r.height));
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w, h };
}

/** Clear to the scene background. */
export function clear(ctx, w, h) {
  ctx.globalCompositeOperation = 'source-over';
  ctx.shadowBlur = 0;
  ctx.fillStyle = P.bg;
  ctx.fillRect(0, 0, w, h);
}

/** Stroke the current path twice: a wide soft halo, then the line. */
export function glowStroke(ctx, color, width = 1.6, blur = 10) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = blur;
  ctx.lineWidth = width;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.stroke();
  ctx.restore();
}

/** Small axis text. */
export function label(ctx, s, x, y, align = 'left', color = P.muted) {
  ctx.save();
  ctx.font = '500 11px "IBM Plex Mono", ui-monospace, monospace';
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.textBaseline = 'middle';
  ctx.fillText(s, x, y);
  ctx.restore();
}

/**
 * A clock for a scene: draw(t) with t in seconds since the scene first
 * started, advancing only while started; a resize redraws the current t.
 */
export function clock(draw) {
  let raf = 0;
  let t = 0;
  let last = 0;
  const tick = (ts) => {
    if (last) t += Math.min(0.1, (ts - last) / 1000);
    last = ts;
    draw(t);
    raf = requestAnimationFrame(tick);
  };
  return {
    start() {
      if (raf) return;
      last = 0;
      raf = requestAnimationFrame(tick);
    },
    stop() {
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
    },
    redraw() {
      draw(t);
    },
    reset() {
      t = 0;
    },
    get t() {
      return t;
    },
  };
}

/** Smoothstep on [0, 1]. */
export const ease = (f) => (f <= 0 ? 0 : f >= 1 ? 1 : f * f * (3 - 2 * f));

/** A number in the readout's style: n significant digits, a true minus sign. */
export const fmt = (v, n = 4) => {
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  const s = a !== 0 && (a < 1e-3 || a >= 1e5) ? v.toExponential(Math.max(0, n - 1)).replace('e+', 'e') : v.toPrecision(n);
  return s.replace(/^-/, '−');
};

/** A scene whose records have not arrived: the frame, faint grid and its title, never blank. */
export function placeholder(canvas, title = '') {
  const ctx = canvas.getContext('2d');
  const { w, h } = fit(canvas, ctx);
  clear(ctx, w, h);
  ctx.strokeStyle = P.grid;
  ctx.beginPath();
  for (let x = 0; x <= w; x += 40) {
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, h);
  }
  for (let y = 0; y <= h; y += 40) {
    ctx.moveTo(0, y + 0.5);
    ctx.lineTo(w, y + 0.5);
  }
  ctx.stroke();
  label(ctx, title, w / 2, h / 2, 'center', P.body);
}
