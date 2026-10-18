// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// A diagram: the boxes, arrows and notes
// a program printed with box(), arrow() and note(), drawn where the records
// put them, one unit to a pixel at full size, with no axes. The same drawing
// as lang/viv/src/figsvg.rs render_diagram, in the page's theme.
//
//   chart      a parsed chart of kind 'diagram' (VivCell parseOwnOutput)
//   highlight  ids to mark (a reader's position: the def and the step it is in)
//   fit        true: as wide as the page, scaled down; false: full size
//   onPick     called with the id of a box the reader clicks
//
// A page that pans and zooms the picture itself (ProgramGraph.jsx) passes
//   viewBox    the part of the picture to show, in its units (diagramFrame)
//   outlines   [{ id, kind }]: an outline around each box, classed by kind
//   hits       true: every arrow gets a wide invisible stroke, easier to point at
//   svgRef     a ref to the <svg>
// and handles the pointer itself. Every mark carries its id (`data-id`), and
// a note also its owner's (`data-of`: the id before `#`), for the page to
// light and dim by.
import { useMemo } from 'react';

// the text sizes and baselines figsvg.rs uses, so both draw the same picture
const SIZE = { small: 11, normal: 13, large: 15 };
const MARGIN = 12;
const HEAD_LEN = 7;
const HEAD_HALF = 3.2;
// the invisible stroke that makes an arrow easier to point at
const HIT_WIDTH = 9;

function colorOf(words, dflt) {
  for (const w of words) {
    if (/^s[1-6]$/.test(w)) return `var(--${w})`;
    if (w === 'ink') return 'var(--heading)';
    if (w === 'muted') return 'var(--text-faint)';
  }
  return dflt;
}

const dashOf = (words) => (words.includes('dash') ? '5 3' : words.includes('dot') ? '1.5 2.5' : undefined);

/** A smooth path through the points (Catmull-Rom as cubic Béziers), or straight pieces. */
function pathD(xs, ys, curved) {
  const n = xs.length;
  let d = `M${xs[0].toFixed(2)} ${ys[0].toFixed(2)}`;
  for (let i = 1; i < n; i++) {
    if (curved && n > 2) {
      const [x0, y0] = i >= 2 ? [xs[i - 2], ys[i - 2]] : [xs[i - 1], ys[i - 1]];
      const [x3, y3] = i + 1 < n ? [xs[i + 1], ys[i + 1]] : [xs[i], ys[i]];
      const [x1, y1, x2, y2] = [xs[i - 1], ys[i - 1], xs[i], ys[i]];
      d += `C${(x1 + (x2 - x0) / 6).toFixed(2)} ${(y1 + (y2 - y0) / 6).toFixed(2)} ${(x2 - (x3 - x1) / 6).toFixed(2)} ${(y2 - (y3 - y1) / 6).toFixed(2)} ${x2.toFixed(2)} ${y2.toFixed(2)}`;
    } else d += `L${xs[i].toFixed(2)} ${ys[i].toFixed(2)}`;
  }
  return d;
}

/** The window: the figure's, or the marks' extent. */
function windowOf(chart) {
  const f = chart.fig || {};
  let [xlo, xhi, ylo, yhi] = [Infinity, -Infinity, Infinity, -Infinity];
  const take = (x, y) => {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    xlo = Math.min(xlo, x); xhi = Math.max(xhi, x); ylo = Math.min(ylo, y); yhi = Math.max(yhi, y);
  };
  for (const m of chart.marks || []) {
    if (m.kind === 'box') { take(m.x0, m.y0); take(m.x1, m.y1); } else if (m.kind === 'arrow') m.x.forEach((x, i) => take(x, m.y[i]));
    else if (m.kind === 'note') take(m.x, m.y);
  }
  if (f.xlim) [xlo, xhi] = f.xlim;
  if (f.ylim) [ylo, yhi] = f.ylim;
  if (![xlo, xhi, ylo, yhi].every(Number.isFinite)) return [0, 1, 0, 1];
  return [xlo, xhi, ylo, yhi];
}

/**
 * Where the picture's units land in the SVG: its size (W, H) and the maps
 * from the records' x and y (y up) to the SVG's (y down).
 */
export function diagramFrame(chart) {
  const [xlo, xhi, ylo, yhi] = windowOf(chart);
  const W = xhi - xlo + 2 * MARGIN;
  const H = yhi - ylo + 2 * MARGIN;
  const px = (x) => MARGIN + (x - xlo);
  const py = (y) => MARGIN + (yhi - y);
  return { W, H, px, py };
}

/** A box mark's rectangle in the SVG's units: { x, y, w, h }. */
export function boxRect(frame, m) {
  return { x: frame.px(Math.min(m.x0, m.x1)), y: frame.py(Math.max(m.y0, m.y1)), w: Math.abs(m.x1 - m.x0), h: Math.abs(m.y1 - m.y0) };
}

export default function DiagramView({ chart, highlight = [], fit = true, onPick = null, viewBox = null, outlines = [], hits = false, svgRef = null }) {
  const [xlo, xhi, ylo, yhi] = useMemo(() => windowOf(chart), [chart]);
  const W = xhi - xlo + 2 * MARGIN;
  const H = yhi - ylo + 2 * MARGIN;
  const px = (x) => MARGIN + (x - xlo);
  const py = (y) => MARGIN + (yhi - y);
  const lit = new Set(highlight);
  const title = chart.fig && chart.fig.title;
  const body = useMemo(() => (chart.marks || []).map((m, k) => {
    const words = (m.style || '').split(/\s+/).filter(Boolean);
    if (m.kind === 'box') {
      const color = colorOf(words, 'var(--text-faint)');
      const fill = words.includes('tint') ? color : words.includes('clear') ? 'none' : 'var(--bg-card)';
      if (words.includes('diamond')) {
        // the rhombus through the midpoints of the box's sides
        const [l, r, t, b] = [px(Math.min(m.x0, m.x1)), px(Math.max(m.x0, m.x1)), py(Math.max(m.y0, m.y1)), py(Math.min(m.y0, m.y1))];
        const [mx, my] = [(l + r) / 2, (t + b) / 2];
        return (
          <path key={k} data-id={m.id} className="dg-box" d={`M${mx} ${t}L${r} ${my}L${mx} ${b}L${l} ${my}Z`}
            fill={fill} fillOpacity={words.includes('tint') ? 0.07 : 1} stroke={words.includes('noline') ? 'none' : color} strokeLinejoin="round"
            strokeWidth={words.includes('thin') ? 0.8 : words.includes('thick') ? 2 : 1.2} strokeDasharray={dashOf(words)} />
        );
      }
      return (
        <rect key={k} data-id={m.id} className="dg-box" x={px(Math.min(m.x0, m.x1))} y={py(Math.max(m.y0, m.y1))}
          width={Math.abs(m.x1 - m.x0)} height={Math.abs(m.y1 - m.y0)} rx={words.includes('square') ? 0 : 5}
          fill={fill} fillOpacity={words.includes('tint') ? 0.07 : 1} stroke={words.includes('noline') ? 'none' : color}
          strokeWidth={words.includes('thin') ? 0.8 : words.includes('thick') ? 2 : 1.2} strokeDasharray={dashOf(words)} />
      );
    }
    if (m.kind === 'arrow') {
      const pts = m.x.map((x, i) => [x, m.y[i]]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y)).map(([x, y]) => [px(x), py(y)]);
      if (pts.length < 2) return null;
      const color = colorOf(words, 'var(--heading)');
      const xs = pts.map((p) => p[0]);
      const ys = pts.map((p) => p[1]);
      const n = xs.length;
      const head = !words.includes('nohead');
      let ux = xs[n - 1] - xs[n - 2];
      let uy = ys[n - 1] - ys[n - 2];
      const len = Math.hypot(ux, uy);
      const [tx, ty] = [xs[n - 1], ys[n - 1]];
      const whole = pathD(xs, ys, words.includes('curved'));
      if (head && len > 0) {
        ux /= len; uy /= len;
        const back = Math.min(HEAD_LEN, len);
        xs[n - 1] -= ux * back; ys[n - 1] -= uy * back;
      }
      const bx = tx - ux * HEAD_LEN;
      const by = ty - uy * HEAD_LEN;
      return (
        <g key={k} data-id={m.id} className="dg-arrow">
          {hits && <path className="dg-hit" d={whole} fill="none" stroke="transparent" strokeWidth={HIT_WIDTH} />}
          <path d={pathD(xs, ys, words.includes('curved'))} fill="none" stroke={color} strokeLinejoin="round"
            strokeWidth={words.includes('thin') ? 0.9 : words.includes('thick') ? 2.2 : 1.3} strokeDasharray={dashOf(words)} />
          {head && len > 0 && <path d={`M${tx} ${ty}L${bx - uy * HEAD_HALF} ${by + ux * HEAD_HALF}L${bx + uy * HEAD_HALF} ${by - ux * HEAD_HALF}Z`} fill={color} />}
        </g>
      );
    }
    if (m.kind === 'note') {
      const size = SIZE[words.find((w) => SIZE[w])] || SIZE.small;
      const anchor = words.includes('left') ? 'start' : words.includes('right') ? 'end' : 'middle';
      const base = words.includes('top') ? 0.78 * size : words.includes('bottom') ? -0.22 * size : 0.35 * size;
      return (
        <text key={k} data-id={m.id} data-of={String(m.id).split('#')[0]} className="dg-note" x={px(m.x)} y={py(m.y) + base} fontSize={size} textAnchor={anchor}
          fill={colorOf(words, 'var(--heading)')} fontFamily={words.includes('mono') ? 'var(--font-mono)' : undefined}
          fontWeight={words.includes('bold') ? 600 : undefined}>{m.text}</text>
      );
    }
    return null;
  }), [chart, xlo, yhi, hits]);
  const boxes = useMemo(() => new Map((chart.marks || []).filter((m) => m.kind === 'box').map((m) => [m.id, m])), [chart]);
  // outlines over boxes, drawn last so nothing hides them: the reader's place
  // (highlight), and the page's own (outlines)
  const ring = (m, key, cls, pad, extra = {}) => (
    <rect key={key} data-hl={m.id} className={cls} x={px(Math.min(m.x0, m.x1)) - pad} y={py(Math.max(m.y0, m.y1)) - pad}
      width={Math.abs(m.x1 - m.x0) + 2 * pad} height={Math.abs(m.y1 - m.y0) + 2 * pad} rx={5 + pad} fill="none" {...extra} />
  );
  const svg = (
    <svg ref={svgRef} viewBox={viewBox || `0 0 ${W} ${H}`} width={viewBox || fit ? '100%' : W} height={viewBox ? '100%' : fit ? undefined : H}
      style={!viewBox && fit ? { maxWidth: `${W}px` } : undefined} preserveAspectRatio={viewBox ? 'xMinYMin meet' : undefined}
      role="img" aria-label={title || 'diagram'} data-nodes={boxes.size}
      onClick={onPick ? (e) => { const el = e.target.closest('[data-id]'); if (el) onPick(el.getAttribute('data-id')); } : undefined}>
      {body}
      {[...lit].filter((id) => boxes.has(id)).map((id, k) => ring(boxes.get(id), `hl${k}`, 'dg-hl', 2.5, { stroke: 'var(--hot)', strokeWidth: 2.5 }))}
      {outlines.filter((o) => boxes.has(o.id)).map((o, k) => ring(boxes.get(o.id), `ol${k}`, `dg-ol dg-ol-${o.kind}`, o.kind === 'cursor' ? 5.5 : 3.5))}
    </svg>
  );
  if (viewBox) return svg;
  return (
    <figure className="vc vc-diagram">
      {title && <figcaption className="vc-title">{title}</figcaption>}
      <div className={fit ? 'dg-fit' : 'dg-scroll'}>{svg}</div>
    </figure>
  );
}
