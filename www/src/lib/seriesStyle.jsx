// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// How a series is drawn, from the program's style() record or, where it chose nothing, the renderer's documented defaults:
// palette slot in order of first appearance, a solid 1.8 px line, and a dot at
// each point of a line of 24 points or fewer.

export const SERIES_VARS = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)', 'var(--s6)'];

const DASH = { dash: '6 4', dot: '1.5 3.5' };
const WIDTH = { thin: 1.1, thick: 2.8 };
/** The primary series' line width (lang/viv/src/figsvg.rs PRIMARY_WIDTH). */
const PRIMARY_WIDTH = 2.4;
/** Bars: nearly opaque, slightly rounded, no outline (figsvg.rs BAR_OPACITY, BAR_RX). */
export const BAR_OPACITY = 0.88;
export const BAR_RX = 1.5;

/**
 * The drawing of one series. `style` is the parsed record (or undefined),
 * `slot` the palette slot the renderer assigned, `sparse` whether the default
 * puts a dot at each point.
 */
export function drawStyle(style, slot, sparse) {
  const st = style || {};
  const color = st.color ? `var(--${st.color})` : SERIES_VARS[slot % SERIES_VARS.length];
  const marker = st.marker ? (st.marker === 'nomarker' ? null : st.marker) : sparse ? 'circle' : null;
  // the primary series: the first palette slot's color, a solid line; it is
  // drawn heavier (class vc-primary)
  const primary = (st.color ? st.color === 's1' : slot % SERIES_VARS.length === 0) && !DASH[st.line];
  return {
    color,
    line: st.line !== 'noline',
    dash: DASH[st.line],
    width: WIDTH[st.width] || (primary ? PRIMARY_WIDTH : 1.8),
    marker,
    primary,
  };
}

/**
 * Each series' palette slot, in order of first appearance: a label names one
 * thing (series that share a label share a slot), and an unlabeled series
 * takes a slot of its own. A slot a style() claims for a label (s1..s6) is
 * that label's, so the others skip it.
 */
export function paletteSlots(series, styles = {}) {
  const claimed = new Set(Object.values(styles).filter((st) => st && st.color).map((st) => Number(st.color.slice(1)) - 1));
  const slots = [];
  const byName = new Map();
  let next = 0;
  const take = () => {
    let tries = 0;
    while (claimed.has(next % SERIES_VARS.length) && tries++ < SERIES_VARS.length) next++;
    return next++;
  };
  for (const s of series) {
    if (s.name && byName.has(s.name)) { slots.push(byName.get(s.name)); continue; }
    const st = styles[s.name];
    const c = st && st.color ? -1 : take(); // a claimed color is set by drawStyle
    if (s.name) byName.set(s.name, c);
    slots.push(c);
  }
  return slots;
}

/** One marker at (x, y): circle (filled), ring (open), square, cross. */
export function Marker({ kind, x, y, color, r = 3, className }) {
  // rounded: SSR and hydration must print the same attribute
  x = Number.isFinite(x) ? Math.round(x * 1000) / 1000 : x;
  y = Number.isFinite(y) ? Math.round(y * 1000) / 1000 : y;
  // (a class such as vc-dot sets a stroke, which would hide a ring: a ring takes none)
  if (kind === 'ring') return <circle cx={x} cy={y} r={r} fill="none" stroke={color} strokeWidth="1.4" />;
  if (kind === 'square') return <rect className={className} x={x - r * 0.9} y={y - r * 0.9} width={r * 1.8} height={r * 1.8} fill={color} />;
  if (kind === 'cross') {
    const a = r * 0.95;
    return <path d={`M${x - a} ${y - a}L${x + a} ${y + a}M${x - a} ${y + a}L${x + a} ${y - a}`} stroke={color} strokeWidth="1.5" strokeLinecap="round" fill="none" />;
  }
  return <circle className={className} cx={x} cy={y} r={r} fill={color} />;
}

/** A legend swatch showing the series' line and marker as drawn. */
export function Swatch({ d, bars = false, interval = null }) {
  // a band (fill_between) is a light block with its two edges; an interval (rangebars) a capped bar
  if (interval === 'fill') return <svg width="14" height="10" aria-hidden="true"><rect x="1" y="1" width="12" height="8" fill={d.color} fillOpacity="0.2" /></svg>;
  if (interval === 'err') return <svg width="10" height="12" aria-hidden="true"><path d="M2 1H8M5 1V11M2 11H8" stroke={d.color} strokeWidth="1.4" fill="none" /></svg>;
  if (bars) return <svg width="14" height="10" aria-hidden="true"><rect x="1" y="1" width="12" height="8" rx={BAR_RX} fill={d.color} fillOpacity={BAR_OPACITY} /></svg>;
  return (
    <svg width="22" height="10" aria-hidden="true" style={{ overflow: 'visible' }}>
      {d.line && <line x1="1" x2="21" y1="5" y2="5" stroke={d.color} strokeWidth={Math.max(2, d.width)} strokeDasharray={d.dash} strokeLinecap={d.dash === DASH.dot ? 'round' : 'butt'} />}
      {d.marker && <Marker kind={d.marker} x={11} y={5} color={d.color} r={3} />}
    </svg>
  );
}
