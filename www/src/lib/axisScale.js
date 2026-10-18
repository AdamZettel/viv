// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Axis scales and ticks for the figure renderers. Plain JavaScript with no dependencies, so the charts,
// the tests and the build can all use it.
//
// A chart computes in "t-space": the value itself on a linear axis, log10 of
// it on a log axis. A value a log axis cannot show (zero, negative, nan) maps
// to NaN, and the chart draws NaN as a gap.

/** The map from a data value to t-space for an axis of this scale. */
export function scaleOf(kind) {
  return kind === 'log' ? (v) => (v > 0 ? Math.log10(v) : NaN) : (v) => v;
}

/** Ticks at 1, 2 or 5 times a power of ten, about n of them over [a, b]. */
export function niceTicks(a, b, n) {
  const span = b - a || 1;
  const step0 = span / Math.max(1, n);
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const out = [];
  for (let t = Math.ceil(a / step - 1e-9) * step; t <= b + step * 1e-9; t += step) out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
  return { ticks: out, step };
}

/** Tick labels for ticks `step` apart: enough decimals, and e-notation for the very large or small. */
export function tickFormat(step) {
  const decimals = Math.max(0, Math.min(8, -Math.floor(Math.log10(step) + 1e-9)));
  return (v) => {
    if (v === 0) return '0';
    const a = Math.abs(v);
    if (a >= 1e5 || a < 1e-4) return v.toExponential(1).replace('e+', 'e').replace('.0e', 'e');
    return v.toFixed(decimals).replace('-', '−');
  };
}

const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
/** 10 to the k, written with a superscript: `10⁻⁶`. */
export const pow10Label = (k) => `10${String(k).split('').map((c) => SUP[c]).join('')}`;

/**
 * The ticks of a log axis whose t-space range is [lo, hi] (log10 units), about
 * n labeled ones. Returns { ticks: [{ t, label }], minor: [t] }.
 *   - Two or more whole decades in range: a tick at each decade, labeled 10ᵏ
 *     (every m-th decade when there are more than n), and unlabeled minor
 *     ticks at 2..9 times each decade when every decade is labeled.
 *   - Less: ticks at 1, 2 and 5 times each power of ten in range (1 to 9
 *     times when that gives fewer than three), labeled as numbers.
 */
export function logTicks(lo, hi, n) {
  const k0 = Math.ceil(lo - 1e-9);
  const k1 = Math.floor(hi + 1e-9);
  if (k1 - k0 + 1 >= 2) {
    const m = Math.max(1, Math.ceil((k1 - k0 + 1) / Math.max(1, n)));
    const ticks = [];
    for (let k = Math.ceil(k0 / m) * m; k <= k1; k += m) ticks.push({ t: k, label: pow10Label(k) });
    const minor = [];
    if (m === 1) {
      for (let k = Math.floor(lo); k <= k1; k++) {
        for (let d = 2; d <= 9; d++) {
          const t = k + Math.log10(d);
          if (t >= lo && t <= hi) minor.push(t);
        }
      }
    }
    return { ticks, minor };
  }
  const within = (digits) => {
    const out = [];
    for (let k = Math.floor(lo); k <= Math.ceil(hi); k++) {
      for (const d of digits) {
        const t = k + Math.log10(d);
        if (t >= lo - 1e-9 && t <= hi + 1e-9) out.push({ t, label: numLabel(d, k) });
      }
    }
    return out;
  };
  let ticks = within([1, 2, 5]);
  if (ticks.length < 3) ticks = within([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  return { ticks, minor: [] };
}

// d times 10^k as a plain number: 0.02, 5, 300; e-notation beyond 1e5 or below 1e-4
function numLabel(d, k) {
  if (k >= 5 || k < -4) return `${d}e${k}`;
  return k >= 0 ? String(d * Math.pow(10, k)) : (d / Math.pow(10, -k)).toFixed(-k);
}

/**
 * The ticks of an axis over the t-space range [lo, hi]: { ticks: [{ t, label }], minor: [t] }.
 * Linear: niceTicks and tickFormat. Log: logTicks.
 */
export function axisTicks(kind, lo, hi, n) {
  if (kind === 'log') return logTicks(lo, hi, n);
  const { ticks, step } = niceTicks(lo, hi, n);
  const fmt = tickFormat(step);
  return { ticks: ticks.map((t) => ({ t, label: fmt(t) })), minor: [] };
}

/**
 * aspect("equal"): one unit of x is as long on the screen as one unit of y (in
 * t-space, so on two log axes a decade is as long across as up). The plot keeps
 * its width and takes the height plotW * (y span) / (x span). When that height
 * would fall outside [minH, maxH], it is held at the limit and the other axis's
 * range is widened about its center instead, so the scales still match. A range
 * is only ever widened, never cut: a window the program set is the least that is
 * shown. Returns { xlo, xhi, ylo, yhi, plotH }.
 */
export function equalAspect(xlo, xhi, ylo, yhi, plotW, minH, maxH) {
  const xs = xhi - xlo || 1;
  const ys = yhi - ylo || 1;
  let plotH = (plotW * ys) / xs;
  if (plotH > maxH) {
    plotH = maxH;
    const c = (xlo + xhi) / 2;
    const half = (ys * plotW) / maxH / 2;
    xlo = c - half; xhi = c + half;
  } else if (plotH < minH) {
    plotH = minH;
    const c = (ylo + yhi) / 2;
    const half = (xs * minH) / plotW / 2;
    ylo = c - half; yhi = c + half;
  }
  return { xlo, xhi, ylo, yhi, plotH };
}

/**
 * A mark (animParse.js parseMark) with its coordinates in t-space. A rect's
 * corners are mapped and it stays centered on the middle of them, so a band
 * from 1e-3 to 1e-1 on a log axis covers exactly those decades.
 */
export function markToT(k, tx, ty) {
  if (k.kind === 'hline') return { ...k, y: ty(k.y) };
  if (k.kind === 'vline' || k.kind === 'xtick') return { ...k, x: tx(k.x) };
  if (k.kind === 'ytick') return { ...k, y: ty(k.y) };
  if (k.kind === 'label') return { ...k, x: tx(k.x), y: ty(k.y) };
  if (k.kind === 'segment') return { ...k, x1: tx(k.x1), y1: ty(k.y1), x2: tx(k.x2), y2: ty(k.y2) };
  if (k.kind === 'rect') {
    const [x0, x1, y0, y1] = [tx(k.x - k.w / 2), tx(k.x + k.w / 2), ty(k.y - k.h / 2), ty(k.y + k.h / 2)];
    return { ...k, x: (x0 + x1) / 2, y: (y0 + y1) / 2, w: Math.abs(x1 - x0), h: Math.abs(y1 - y0) };
  }
  return k;
}

/**
 * Where a heatmap's marks go. A
 * figure window with both pairs set is the picture's extent, and a mark's
 * (x, y) is in those units, ylo at the bottom whichever row is there. With no
 * window a mark is in cell coordinates: x is the column index j and y the row
 * index i, cell centers at whole numbers, so y runs down the picture by
 * default and up it under origin("lower"). Returns { xlo, xhi, ylo, yhi,
 * place }, where place(x, y) is [fx, fy], the point's fraction of the picture
 * from the left and from the top.
 */
export function heatExtent(nx, ny, fig = {}) {
  const win = fig.xlim && fig.ylim;
  const [xlo, xhi] = win ? fig.xlim : [-0.5, ny - 0.5];
  const [ylo, yhi] = win ? fig.ylim : [-0.5, nx - 0.5];
  const down = !win && fig.origin !== 'lower';
  const place = (x, y) => [(x - xlo) / (xhi - xlo || 1), down ? (y - ylo) / (yhi - ylo || 1) : (yhi - y) / (yhi - ylo || 1)];
  return { xlo, xhi, ylo, yhi, place };
}

// ---- tick labels and bar groups (lang/viv/src/figsvg.rs does the same) ----------

/** Tick label text size, and the least room between two flat x tick labels. */
export const TICK_SIZE = 11;
const TICK_PAD = 6;
/** x tick labels that do not fit side by side are turned this many degrees ... */
export const TICK_TURN_DEG = 40;
/** ... and kept this far apart, across their baselines. */
const TURN_ROW = 13;
/** Where a turned label's baseline ends, from its tick at the bottom of the plot. */
export const TURN_DX = 4;
export const TURN_DY = 14;
/** How far below the plot a flat tick label reaches (baseline 17, descent 2). */
const FLAT_DROP = 19;
/** The least spacing of the program's own y ticks, and the tallest plot that buys it. */
export const YTICK_ROW = 14;
export const MAX_PLOT_H = 900;
/** A bar group fills this part of the space between categories; a bar in a group, this part of its slot. */
const GROUP_FILL = 0.8;
const BAR_FILL = 0.88;
/** A lone bar series: its bars fill this part of the space between them. */
const SOLO_FILL = 0.86;
const rad = (d) => (d * Math.PI) / 180;

/**
 * Lays out x tick labels at pixel positions `px` with text widths `w`:
 * { turn, show }. Flat when every neighbor pair fits; otherwise a category
 * axis (the program's own ticks) turns its labels, and a number axis drops
 * every other label (then two of three ...) until the rest fit.
 */
export function xTickLayout(px, w, categories) {
  const n = px.length;
  const ord = [...Array(n).keys()].filter((i) => Number.isFinite(px[i])).sort((a, b) => px[a] - px[b]);
  const picked = (step) => ord.filter((_, j) => j % step === 0);
  const pairs = (idx, ok) => idx.every((a, j) => j === 0 || ok(idx[j - 1], a));
  const flatFits = (idx) => pairs(idx, (a, b) => px[b] - px[a] >= (w[a] + w[b]) / 2 + TICK_PAD);
  const turnFits = (idx) => pairs(idx, (a, b) => (px[b] - px[a]) * Math.sin(rad(TICK_TURN_DEG)) >= TURN_ROW);
  const shown = (idx) => { const s = new Set(idx); return [...Array(n).keys()].map((i) => s.has(i)); };
  if (flatFits(ord)) return { turn: false, show: shown(ord) };
  for (let step = 1; step <= Math.max(ord.length, 1); step++) {
    const idx = picked(step);
    if (categories && turnFits(idx)) return { turn: true, show: shown(idx) };
    if (!categories && step > 1 && flatFits(idx)) return { turn: false, show: shown(idx) };
  }
  return { turn: categories, show: shown(picked(Math.max(ord.length, 1))) };
}

/** The extra room under the plot that turned labels of these widths need, past a flat label's. */
export function turnedDrop(w) {
  const widest = w.reduce((m, v) => Math.max(m, v), 0);
  return Math.max(TURN_DY + widest * Math.sin(rad(TICK_TURN_DEG)) + 0.2 * TICK_SIZE * Math.cos(rad(TICK_TURN_DEG)) - FLAT_DROP, 0);
}

/** How far a turned label reaches left of its tick. */
export const turnedReach = (w) => w * Math.cos(rad(TICK_TURN_DEG)) - TURN_DX;

/**
 * Where each bar series' bars stand, in pixels: [offset from its x, width],
 * null for a series that is not bars. `xsOf(k)` gives series k's x values (in
 * t-space). Bar series that share an x stand side by side, a group centered
 * on the x; `pxPerX` turns x units into pixels.
 */
export function barSlots(series, xsOf, pxPerX) {
  const bars = series.map((s, k) => (s.bars ? k : -1)).filter((k) => k >= 0);
  const finXs = (k) => xsOf(k).filter(Number.isFinite);
  const minGap = (xs) => {
    const v = [...xs].sort((a, b) => a - b);
    let g = Infinity;
    for (let i = 1; i < v.length; i++) { const d = v[i] - v[i - 1]; if (d > 0 && d < g) g = d; }
    return g;
  };
  const shared = bars.length > 1 && bars.some((a, j) => bars.slice(j + 1).some((b) => { const xb = xsOf(b); return finXs(a).some((x) => xb.includes(x)); }));
  const out = series.map(() => null);
  const all = bars.flatMap(finXs);
  if (shared) {
    const gap = minGap(all);
    const group = Number.isFinite(gap) ? gap * pxPerX * GROUP_FILL : 12 * bars.length;
    const slot = group / bars.length;
    bars.forEach((k, j) => { out[k] = [(j - (bars.length - 1) / 2) * slot, Math.max(slot * BAR_FILL, 1.5)]; });
    return out;
  }
  // bars that do not share an x: each as wide as the closest two of all of them allow
  const unionGap = minGap(all);
  for (const k of bars) {
    const gap = bars.length > 1 ? unionGap : minGap(finXs(k));
    out[k] = [0, Number.isFinite(gap) ? Math.max(gap * pxPerX * SOLO_FILL, 2) : 12];
  }
  return out;
}
