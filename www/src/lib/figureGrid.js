// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// A grid of figures: what each
// panel's chart is told about its neighbors. Plain JavaScript with no
// dependencies beyond axisScale.js, so the charts and the tests can use it.
//
// grid("title", rows, cols) makes the next rows * cols figures its panels, row
// by row; share("x") gives the panels of each column one x range and draws the
// x tick labels and the x label only under the lowest panel of the column;
// share("y") does the same for the y axis along each row, with the labels on
// the left-most panel. The shared range is the union of the panels' data (and
// of any window a panel's figure set), in the panels' own units; nothing else
// about a panel changes.

import { scaleOf } from './axisScale.js';

// a chart that has x and y axes (LineChart, ScatterPlot); a heatmap does not share
const isXY = (c) => c && (c.kind === undefined || c.kind === 'scatter');

// the values a chart draws on one axis, left y axis only (a yright series has its own)
function values(c, axis) {
  const right = new Set(((c.fig && c.fig.yright) || {}).series || []);
  if (axis === 'x') return c.x ? c.x.values : c.series.flatMap((s) => s.xs);
  // an interval (fill_between, rangebars) reaches from its lo to its hi
  const ys = c.series.filter((s) => !right.has(s.name)).flatMap((s) => (s.interval ? [...s.lo, ...s.hi] : c.x ? s.values : s.ys));
  // bars stand on zero, so zero is part of what they draw
  return c.series.some((s) => s.bars) ? [...ys, 0] : ys;
}

/** The union of the charts' ranges on one axis, in data units, or null. */
export function sharedRange(charts, axis) {
  let lo = Infinity;
  let hi = -Infinity;
  let fixed = true;
  for (const c of charts) {
    const fig = c.fig || {};
    const kind = axis === 'x' ? fig.xscale : fig.yscale;
    const t = scaleOf(kind);
    const lim = axis === 'x' ? fig.xlim : fig.ylim;
    const take = (v) => { if (Number.isFinite(t(v))) { if (v < lo) lo = v; if (v > hi) hi = v; } };
    if (lim) { take(lim[0]); take(lim[1]); } else { fixed = false; values(c, axis).forEach(take); }
  }
  if (!Number.isFinite(lo)) return null;
  if (hi === lo) return null;
  // a y range the charts would have padded (7%, as LineChart does) keeps its padding;
  // on a log axis the padding is in decades
  if (axis === 'y' && !fixed) {
    const log = charts.some((c) => c.fig && c.fig.yscale === 'log');
    if (log) {
      const [a, b] = [Math.log10(lo), Math.log10(hi)];
      const pad = (b - a) * 0.07;
      [lo, hi] = [Math.pow(10, a - pad), Math.pow(10, b + pad)];
    } else {
      const pad = (hi - lo) * 0.07;
      [lo, hi] = [lo - pad, hi + pad];
    }
  }
  return [lo, hi];
}

/**
 * The panels of a grid chart, ready to draw: each chart with its figure told
 * the shared ranges (xlim, ylim) and which of its tick labels a neighbor
 * already shows (`inner: { x, y }`). `stacked` is true when the page is too
 * narrow for the grid's columns and the panels are drawn one under another:
 * the ranges are still shared, and every panel keeps its labels.
 */
export function layoutPanels(grid, stacked = false) {
  const { cols, share } = grid;
  const n = grid.panels.length;
  const out = grid.panels.map((p) => p.map((c) => (isXY(c) ? { ...c, fig: { ...(c.fig || {}), inGrid: true } } : c)));
  const shareAxis = (axis, groupOf) => {
    const groups = new Map();
    out.forEach((p, k) => {
      const g = groupOf(k);
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(...p.filter(isXY));
    });
    for (const charts of groups.values()) {
      const r = sharedRange(charts, axis);
      if (!r) continue;
      for (const c of charts) c.fig[axis === 'x' ? 'xlim' : 'ylim'] = r;
    }
  };
  if (share.includes('x')) shareAxis('x', (k) => k % cols);
  if (share.includes('y')) shareAxis('y', (k) => Math.floor(k / cols));
  out.forEach((p, k) => {
    const below = k + cols < n; // another panel under this one, in its column
    const col = k % cols;
    // a column with a right axis anywhere keeps room for it in every panel
    let right = false;
    for (let j = col; j < n; j += cols) if (out[j].some((c) => isXY(c) && c.fig.yright)) right = true;
    for (const c of p) {
      if (!isXY(c)) continue;
      c.fig.inner = {
        x: !stacked && share.includes('x') && below,
        y: !stacked && share.includes('y') && col > 0,
      };
      c.fig.roomRight = right;
    }
  });
  return out;
}
