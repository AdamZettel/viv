// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The records a scene draws, read from a program's output: its figure()
// blocks (title, window, scales, plots, plot2d frames, lines) and its shown
// values (`name: value`). The scenes only draw these; no number on a scene
// is computed anywhere but in the program.
import { recArr, recNum } from './animParse.js';

const arr = (s) => {
  const t = (s || '').trim();
  return t.startsWith('[') && t.endsWith(']') ? recArr(t.slice(1, -1)) : [];
};
const nums = (s) => (s || '').trim().split(/\s+/).filter(Boolean).map(recNum);

/** Every figure() of a run: {title, xlabel, ylabel, window, xscale, yscale, plots, frames, lines, marks}. */
export function figuresOf(text) {
  const figs = [];
  let cur = null;
  for (const line of String(text || '').split('\n')) {
    const c = line.indexOf(': ');
    if (c < 0) continue;
    const kind = line.slice(0, c);
    const parts = line.slice(c + 2).split(' ; ');
    if (kind === 'figure') {
      const w = nums(parts[3]);
      cur = { title: parts[0].trim(), xlabel: (parts[1] || '').trim(), ylabel: (parts[2] || '').trim(), window: w.length === 4 ? w : null, xscale: 'linear', yscale: 'linear', plots: [], frames: [], lines: [] };
      figs.push(cur);
      continue;
    }
    if (!cur) continue;
    if (kind === 'xscale') cur.xscale = parts[0].trim();
    else if (kind === 'yscale') cur.yscale = parts[0].trim();
    else if (kind === 'plot') cur.plots.push({ label: parts[0].trim(), xs: arr(parts[1]), ys: arr(parts[2]) });
    else if (kind === 'scatter') cur.plots.push({ label: parts[0].trim(), xs: arr(parts[2]), ys: arr(parts[3]), dots: true });
    else if (kind === 'plot2d') {
      const [nx, ny] = nums(parts[1]);
      const z = arr(parts[2]);
      if (z.length === nx * ny) cur.frames.push({ label: parts[0].trim(), nx, ny, z });
    } else if (kind === 'hline' || kind === 'vline') cur.lines.push({ kind, label: parts[0].trim(), at: nums(parts[1])[0] });
    else if (kind === 'label') {
      const [x, y] = nums(parts[2]);
      cur.lines.push({ kind, id: parts[0].trim(), text: (parts[1] || '').trim(), x, y });
    }
  }
  return figs;
}

/** The figure whose title starts with `start`, or null. */
export const figureNamed = (figs, start) => figs.find((f) => f.title.startsWith(start)) || null;

/** The program's shown values, `name: number` lines, as {name: number}. */
export function shownOf(text) {
  const out = {};
  for (const line of String(text || '').split('\n')) {
    const m = line.match(/^([A-Za-z_]\w*): (-?[\d.]+(?:e[+-]?\d+)?)$/);
    if (m) out[m[1]] = Number(m[2]);
  }
  return out;
}

/** An animation's stage records grouped per keyframe: [{rect, segment, circle}] by object id. */
export function stageFrames(text) {
  const frames = [];
  let cur = {};
  let stage = null;
  for (const line of String(text || '').split('\n')) {
    const c = line.indexOf(': ');
    if (c < 0) continue;
    const kind = line.slice(0, c);
    const parts = line.slice(c + 2).split(' ; ');
    if (kind === 'stage') stage = { title: parts[0].trim(), window: nums(parts[1]) };
    else if (kind === 'frame') {
      frames.push({ ...cur, dt: nums(parts[0])[0] });
    } else if (kind === 'rect' || kind === 'segment' || kind === 'circle') cur = { ...cur, [parts[0].trim()]: nums(parts[1]) };
  }
  if (Object.keys(cur).length) frames.push({ ...cur, dt: 0 });
  return { stage, frames };
}
