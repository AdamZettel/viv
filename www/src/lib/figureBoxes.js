// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// At build: the size of each figure a card draws, for the empty boxes its
// stub shows until the card loads (src/components/FigureBox.jsx). Each chart
// is rendered to markup the way the card renders it, and read back:
//   line   a LineChart, drawn at the width it is shown: its plot is
//          clamp(200, 0.52 x plot width, 340) px tall (LineChart.jsx), so the
//          box keeps the margins around the plot and that rule
//   ratio  anything else: the proportions of its first <svg> (or canvas),
//          scaled to the card's width up to `max`
// `head` is the HTML above the drawing: the title line and one legend row.
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { Chart, parseOwnOutput } from '../components/VivCell.jsx';

const PLOT_MAX = 540; // .kcell-plot { width: min(540px, 100%) } in VivCell.jsx
const TITLE_PX = 23; // .vc-title: 14px at line-height 1.35, 4px below
const LEGEND_PX = 24; // .vc-legend: one 18px row, 6px below

function boxOf(chart) {
  let html = '';
  try {
    html = renderToStaticMarkup(createElement(Chart, { chart }));
  } catch {
    return null;
  }
  const head = (/<figcaption class="vc-title"/.test(html) ? TITLE_PX : 0) + (/class="vc-legend/.test(html) ? LEGEND_PX : 0);
  // the drawing's own <svg>: not the small ones that set math in the title or legend
  const svg = [...html.matchAll(/<svg[^>]*\bviewBox="0 0 ([\d.]+) ([\d.]+)"[^>]*>/g)].find((m) => !/class="math-inline"/.test(m[0]));
  if (svg) {
    const w = Number(svg[1]);
    const h = Number(svg[2]);
    const clip = html.slice(svg.index).match(/<clipPath[^>]*><rect[^>]*\bwidth="([\d.]+)"[^>]*\bheight="([\d.]+)"/);
    if (/<figure class="vc"/.test(html) && clip) {
      const pw = Number(clip[1]);
      const ph = Number(clip[2]) - 2;
      // the default height rule only; an equal-aspect or square plot falls through
      if (Math.abs(ph - Math.round(Math.max(200, Math.min(340, pw * 0.52)))) <= 1) return { kind: 'line', wx: w - pw, hx: h - ph, head };
    }
    const cap = (svg[0].match(/max-width:\s*([\d.]+)px/) || [])[1];
    return { kind: 'ratio', w, h, max: Math.min(PLOT_MAX, cap ? Number(cap) : PLOT_MAX), head };
  }
  const canvas = html.match(/<canvas[^>]*\bwidth="(\d+)"[^>]*\bheight="(\d+)"/);
  if (canvas) return { kind: 'ratio', w: Number(canvas[1]), h: Number(canvas[2]), max: PLOT_MAX, head };
  return null;
}

/** The boxes for the figures of a card's recorded output; [] when it draws none. */
export function figureBoxes(figure, first = Infinity) {
  const charts = parseOwnOutput(figure || '').charts.slice(0, first);
  return charts.map(boxOf).map((b) => b || { kind: 'ratio', w: 4, h: 3, max: PLOT_MAX, head: 0 });
}
