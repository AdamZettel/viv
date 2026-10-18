// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { MAX_PLOT_H, TICK_SIZE, TICK_TURN_DEG, TURN_DX, TURN_DY, YTICK_ROW, axisTicks, barSlots, equalAspect, scaleOf, turnedDrop, turnedReach, xTickLayout } from '../lib/axisScale.js';
import { BAR_OPACITY, BAR_RX, SERIES_VARS, Marker, Swatch, drawStyle, paletteSlots } from '../lib/seriesStyle.jsx';
import { chartLabel, hasMath } from '../lib/mathtext.js';
import { MathInline, SvgText } from './MathText.jsx';

// The line chart every Viv figure is drawn with.
//
// It is drawn at the size it is shown: the SVG's coordinate system is the
// container's width in pixels, so an 11px tick label is 11px and a 1.75px line
// is 1.75px in a narrow notebook cell and in a wide card alike. (Scaling one
// fixed viewBox up is what made the old charts look heavy.) The title and the
// legend are HTML above the plot, so they wrap, and never sit on the data.
//
// What it shows comes from the program: title, axis labels and window from
// figure(), a log axis from xscale()/yscale(), a legend entry from each
// plot()'s label. It takes no presentation props, so a page cannot decorate a
// chart the program did not describe.
//
// Geometry is computed in t-space (lib/axisScale.js): the value on a linear
// axis, log10 of it on a log axis, where a value <= 0 is a gap like a nan.

export { SERIES_VARS };

const readout = (v) => {
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return '0';
  const a = Math.abs(v);
  const s = a >= 1e5 || a < 1e-3 ? v.toExponential(3).replace('e+', 'e') : String(+v.toPrecision(5));
  return s.replace('-', '−');
};

// how many finite values a log axis cannot show (zero or negative)
const hiddenOnLog = (vals) => vals.reduce((n, v) => n + (Number.isFinite(v) && v <= 0 ? 1 : 0), 0);

export default function LineChart({ chart }) {
  const fig = chart.fig || {};
  const wrapRef = useRef(null);
  const [width, setWidth] = useState(640);
  const [hover, setHover] = useState(null); // index into xs

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return undefined;
    const measure = () => {
      const w = Math.round(el.clientWidth);
      if (w > 0) setWidth((prev) => (Math.abs(prev - w) > 1 ? w : prev));
    };
    measure();
    if (typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // the program's own marks on this chart (animParse.js parseMark), in the order printed
  const marks = useMemo(() => [...(fig.marks || []), ...(chart.marks || [])], [chart, fig.marks]);
  // the axis scales the program set with xscale()/yscale(); linear otherwise
  const xScale = fig.xscale === 'log' ? 'log' : 'linear';
  const yScale = fig.yscale === 'log' ? 'log' : 'linear';
  // the series the program put on a right y axis (yright), by label
  const rightSet = useMemo(() => new Set((fig.yright && fig.yright.series) || []), [fig.yright]);

  const geom = useMemo(() => {
    const tx = scaleOf(xScale);
    const ty = scaleOf(yScale);
    const xs = chart.x ? chart.x.values : chart.series[0].values.map((_, i) => i);
    const txs = xs.map(tx);
    // yright("label", "axis"): those series read against a linear y axis on the right
    const right = chart.series.map((s) => rightSet.has(s.name));
    const tys = chart.series.map((s, k) => (right[k] ? s.values : s.values.map(ty)));
    // fill_between and rangebars: the two ends of each interval, in t-space
    const tlo = chart.series.map((s, k) => (s.interval ? (right[k] ? s.lo : s.lo.map(ty)) : null));
    const thi = chart.series.map((s, k) => (s.interval ? (right[k] ? s.hi : s.hi.map(ty)) : null));
    const ends = (k) => (tlo[k] ? [...tlo[k], ...thi[k]] : []);
    // the extent of the marks, so a band or a label the program placed is on the chart
    const mx = [];
    const my = [];
    for (const k of marks) {
      if (k.kind === 'hline') my.push(ty(k.y));
      else if (k.kind === 'vline' || k.kind === 'xtick') mx.push(tx(k.x));
      else if (k.kind === 'ytick') my.push(ty(k.y));
      else if (k.kind === 'label') { mx.push(tx(k.x)); my.push(ty(k.y)); }
      else if (k.kind === 'rect') { mx.push(tx(k.x - k.w / 2), tx(k.x + k.w / 2)); my.push(ty(k.y - k.h / 2), ty(k.y + k.h / 2)); }
      else if (k.kind === 'segment') { mx.push(tx(k.x1), tx(k.x2)); my.push(ty(k.y1), ty(k.y2)); }
    }
    // a window the program fixed, in t-space; on a log axis it must be positive
    const lim = (l, f) => {
      if (!l) return null;
      const a = f(l[0]);
      const b = f(l[1]);
      return Number.isFinite(a) && Number.isFinite(b) ? [a, b] : null;
    };
    const xlim = lim(fig.xlim, tx);
    const ylim = lim(fig.ylim, ty);
    const extent = (vals) => {
      let lo = Infinity;
      let hi = -Infinity;
      for (const v of vals) if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
      return Number.isFinite(lo) ? [lo, hi] : [0, 1];
    };
    let xlo, xhi, ylo, yhi;
    if (xlim) [xlo, xhi] = xlim;
    else [xlo, xhi] = extent([...txs, ...mx]);
    const hasBars = chart.series.some((s) => s.bars);
    if (ylim) [ylo, yhi] = ylim;
    else {
      [ylo, yhi] = extent([...tys.filter((_, k) => !right[k]).flat(), ...chart.series.flatMap((_, k) => (right[k] ? [] : ends(k))), ...my]);
      // bars stand on zero (on a log axis, on the bottom of the chart)
      const onZero = hasBars && yScale === 'linear';
      if (onZero && ylo > 0) ylo = 0;
      const pad = (yhi - ylo || Math.abs(yhi) || 1) * 0.07;
      ylo -= onZero && ylo === 0 ? 0 : pad; yhi += pad;
    }
    if (hasBars && !xlim) {
      // half a bar of room at each end
      let gap = Infinity;
      for (let i = 1; i < txs.length; i++) { const d = Math.abs(txs[i] - txs[i - 1]); if (d < gap) gap = d; }
      if (Number.isFinite(gap)) { xlo -= gap / 2; xhi += gap / 2; }
    }
    // the right axis: the extent of its series, padded like the left
    let y2lo = 0;
    let y2hi = 1;
    const hasRight = right.some(Boolean);
    if (hasRight) {
      [y2lo, y2hi] = extent([...tys.filter((_, k) => right[k]).flat(), ...chart.series.flatMap((_, k) => (right[k] ? ends(k) : []))]);
      const pad = (y2hi - y2lo || Math.abs(y2hi) || 1) * 0.07;
      y2lo -= pad; y2hi += pad;
    }
    const equal = fig.aspect === 'equal';
    const square = !equal && xlim && ylim && Math.abs((xhi - xlo) - (yhi - ylo)) < 1e-9 * Math.abs(xhi - xlo || 1);
    if (equal) {
      // aspect("equal"): the ranges are widened here if the plot would be too tall
      // or too flat; the height itself is set once the margins are known, below
      const pw = Math.max(40, width - 80);
      ({ xlo, xhi, ylo, yhi } = equalAspect(xlo, xhi, ylo, yhi, pw, 160, Math.min(pw, 520)));
    }
    // what a log axis cannot show is said under the chart, not dropped in silence
    const hidden = [];
    if (xScale === 'log' && hiddenOnLog(xs)) hidden.push(`${hiddenOnLog(xs)} x value${hiddenOnLog(xs) > 1 ? 's' : ''} ≤ 0`);
    const hy = chart.series.reduce((n, s, k) => n + (right[k] ? 0 : hiddenOnLog(s.interval ? [...s.lo, ...s.hi] : s.values)), 0);
    if (yScale === 'log' && hy) hidden.push(`${hy} y value${hy > 1 ? 's' : ''} ≤ 0`);
    return { xs, txs, tys, tlo, thi, tx, ty, xlo, xhi, ylo, yhi, y2lo, y2hi, right, hasRight, square, equal, hidden };
  }, [chart, fig.xlim, fig.ylim, fig.aspect, marks, xScale, yScale, width, rightSet]);

  const { xs, txs, tys, tlo, thi, tx, ty, xlo, xhi, ylo, yhi, y2lo, y2hi, right, hasRight, square, equal, hidden } = geom;
  // in a grid (lib/figureGrid.js): which tick labels a neighbor panel already shows
  const inner = fig.inner || {};
  const W = width;
  // the program's own y ticks (ytick) replace the automatic ones, as xtick does for x
  const yticks = marks.filter((k) => k.kind === 'ytick');
  const yT = yticks.length
    ? { ticks: yticks.map((k) => ({ t: ty(k.y), label: k.text })).filter((k) => Number.isFinite(k.t)), minor: [] }
    : axisTicks(yScale, ylo, yhi, 5);
  const xticks = marks.filter((k) => k.kind === 'xtick');
  // the program's own x ticks replace the automatic ones
  const xT = xticks.length
    ? { ticks: xticks.map((k) => ({ t: tx(k.x), label: k.text })).filter((k) => Number.isFinite(k.t)), minor: [] }
    : axisTicks(xScale, xlo, xhi, W < 420 ? 4 : 7);
  // a label's width in tick characters: math is measured (src/lib/mathtext.js)
  const chars = (s) => (hasMath(s) ? chartLabel(s, 11).w / 6.4 : s.length);
  const widest = Math.max(...yT.ticks.map((t) => chars(t.label)), 1);
  // a tall math label (a fraction) moves out from the plot, as in viv plot
  const xl = hasMath(fig.xlabel) ? chartLabel(fig.xlabel, 11.5) : null;
  const yl = hasMath(fig.ylabel) ? chartLabel(fig.ylabel, 11.5) : null;
  const below = xl ? Math.max(xl.d - 3, 0) : 0;
  const xlRoom = below + (xl ? Math.max(xl.h - 9, 0) : 0);
  const ylRoom = yl ? Math.max(yl.h - 9, 0) : 0;
  const y2T = hasRight ? axisTicks('linear', y2lo, y2hi, 5) : null;
  const widest2 = y2T ? Math.max(...y2T.ticks.map((t) => t.label.length), 1) : 0;
  // a panel of a grid keeps room for 8-character tick labels, so the plots of a
  // column line up whatever their numbers; a panel whose neighbor shows its
  // labels keeps none
  const GRID_ROOM = 16 + 8 * 6.4 + 18;
  const M = {
    l: inner.y ? 12 : Math.max(16 + widest * 6.4 + (fig.ylabel ? 18 + ylRoom : 0), fig.inGrid ? GRID_ROOM : 0),
    r: Math.max(hasRight ? 16 + widest2 * 6.4 + 18 : 14, fig.inGrid && fig.roomRight ? GRID_ROOM : 0),
    t: 10,
    b: inner.x ? 10 : fig.xlabel ? 44 + xlRoom : 26,
  };
  let plotW = Math.max(40, W - M.l - M.r);
  // x tick labels that would run into each other are turned (categories) or thinned (numbers)
  const xtW = xT.ticks.map(({ label }) => chartLabel(label, TICK_SIZE).w);
  const xtPx = (l, pw) => xT.ticks.map(({ t }) => l + ((t - xlo) / (xhi - xlo || 1)) * pw);
  const xLay = xTickLayout(xtPx(M.l, plotW), xtW, xticks.length > 0);
  if (xLay.turn && !inner.x) {
    M.b += turnedDrop(xtW.filter((_, i) => xLay.show[i]));
    // the first labels reach left past the plot: the margin grows to hold them
    const px = xtPx(M.l, plotW);
    const over = xtW.reduce((m, w, i) => (xLay.show[i] ? Math.max(m, turnedReach(w) - (px[i] - 4)) : m), 0);
    if (over > 0) { M.l += over; plotW = Math.max(40, W - M.l - M.r); }
  }
  // equal aspect: the height that makes a unit of y as long as a unit of x
  let plotH = equal ? Math.round((plotW * (yhi - ylo)) / (xhi - xlo || 1))
    : square ? plotW : Math.round(Math.max(200, Math.min(340, plotW * 0.52)));
  if (!equal && !square && yticks.length > 1) {
    // the program's own y ticks (categories) get a row each
    const ys = yticks.map((k) => ty(k.y)).filter(Number.isFinite).sort((a, b) => a - b);
    let gap = Infinity;
    for (let i = 1; i < ys.length; i++) { const d = ys[i] - ys[i - 1]; if (d > 0 && d < gap) gap = d; }
    if (Number.isFinite(gap)) plotH = Math.min(Math.max(plotH, Math.round((YTICK_ROW * Math.abs(yhi - ylo)) / gap)), Math.max(MAX_PLOT_H, plotH));
  }
  const H = plotH + M.t + M.b;
  // X and Y take t-space values
  // to 1/1000 px: t-space values come from a log10 or pow, whose last digit the
  // build and the browser may disagree on, and hydration compares attributes
  const r3 = (v) => (Number.isFinite(v) ? Math.round(v * 1000) / 1000 : v);
  const X = (v) => r3(M.l + ((v - xlo) / (xhi - xlo || 1)) * plotW);
  const Y = (v) => r3(M.t + (1 - (v - ylo) / (yhi - ylo || 1)) * plotH);
  const Y2 = (v) => r3(M.t + (1 - (v - y2lo) / (y2hi - y2lo || 1)) * plotH);
  // the y map of series k: the right axis's for a yright series
  const Yof = (k) => (right[k] ? Y2 : Y);
  const fin = Number.isFinite;
  // bars that share an x stand side by side (lib/axisScale.js barSlots)
  const barPx = barSlots(chart.series, () => txs, plotW / (xhi - xlo || 1));
  const sparse = xs.length <= 24;
  const clipId = `lc${useId().replace(/:/g, '')}`;
  // a label names one thing: series that share a label share its color and one
  // legend entry (eight bands printed as "band" are one entry, one color)
  // each series' drawing: the program's style() for its label, else the defaults
  const styles = fig.styles || {};
  const colorIndex = paletteSlots(chart.series, styles);
  const byName = new Set(chart.series.map((s) => s.name).filter(Boolean));
  const draws = chart.series.map((s, k) => drawStyle(styles[s.name], colorIndex[k], sparse));
  const colorOf = (k) => draws[k].color;
  const named = chart.series.filter((s) => s.name);
  // one legend entry per label, drawn like its first series
  const legend = [...byName].map((name) => {
    // drawn like the label's first series that is not an interval, if it has one
    const first = chart.series.findIndex((s) => s.name === name && !s.interval);
    const k = first >= 0 ? first : chart.series.findIndex((s) => s.name === name);
    return { name, d: draws[k], bars: !!chart.series[k].bars, interval: chart.series[k].interval || null, right: right[k] };
  });

  const paths = useMemo(() => tys.map((vals, k) => {
    const Yk = Yof(k);
    let d = '';
    let pen = false;
    for (let i = 0; i < vals.length; i++) {
      const v = vals[i];
      // a nan or inf (in y or in x), or a value a log axis cannot show, is a gap
      if (!fin(v) || !fin(txs[i])) { pen = false; continue; }
      // a value far outside a declared window is clamped, so the path stays finite
      const yy = Math.max(-1e4, Math.min(1e4, Yk(v)));
      d += `${pen ? 'L' : 'M'}${X(txs[i]).toFixed(2)} ${yy.toFixed(2)}`;
      pen = true;
    }
    return d;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [geom, W, plotH]);

  const onMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const xv = xlo + ((px - M.l) / plotW) * (xhi - xlo);
    let best = 0;
    let bestD = Infinity;
    for (let i = 0; i < txs.length; i++) {
      const dd = Math.abs(txs[i] - xv);
      if (dd < bestD) { bestD = dd; best = i; }
    }
    if (!fin(bestD)) { setHover(null); return; }
    setHover(txs[best] >= xlo && txs[best] <= xhi ? best : null);
  };

  const tipLeft = hover === null ? 0 : X(txs[hover]);
  const flip = tipLeft > W * 0.62;
  const bottom = M.t + plotH;

  return (
    <figure className="vc">
      {fig.title && <figcaption className="vc-title"><MathInline text={fig.title} size={14} /></figcaption>}
      {named.length > 0 && (
        <div className="vc-legend">
          {legend.map(({ name, d, bars, interval, right: r }) => (
            <span key={name}><Swatch d={d} bars={bars} interval={interval} /><MathInline text={name} size={12} />{r && <em className="vc-right-note"> (right axis)</em>}</span>
          ))}
        </div>
      )}
      <div className="vc-plot" ref={wrapRef} style={square ? { maxWidth: 460, margin: '0 auto' } : undefined}>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
          aria-label={fig.title || `plot of ${chart.series.map((s) => s.name).filter(Boolean).join(', ')}`}
          onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
          <defs>
            <clipPath id={clipId}><rect x={M.l} y={M.t - 1} width={plotW} height={plotH + 2} /></clipPath>
          </defs>
          {yT.ticks.map(({ t, label }) => (
            <g key={'y' + t}>
              <line className="vc-grid" x1={M.l} x2={W - M.r} y1={Y(t)} y2={Y(t)} />
              {!inner.y && <SvgText className="vc-tick" text={label} size={11} color="var(--text-faint)" x={M.l - 8} y={Y(t) + 3.8} textAnchor="end" />}
            </g>
          ))}
          {yT.minor.map((t) => <line key={'ym' + t} className="vc-axis" x1={M.l - 3} x2={M.l} y1={Y(t)} y2={Y(t)} />)}
          <line className="vc-axis" x1={M.l} x2={W - M.r} y1={bottom} y2={bottom} />
          {xT.ticks.map(({ t, label }, i) => (
            <g key={'x' + t}>
              <line className="vc-axis" x1={X(t)} x2={X(t)} y1={bottom} y2={bottom + 4} />
              {!inner.x && xLay.show[i] && (xLay.turn
                ? <SvgText className="vc-tick" text={label} size={TICK_SIZE} color="var(--text-faint)" x={X(t) + TURN_DX} y={bottom + TURN_DY} textAnchor="end" rotate={-TICK_TURN_DEG} />
                : <SvgText className="vc-tick" text={label} size={TICK_SIZE} color="var(--text-faint)" x={X(t)} y={bottom + 17} textAnchor="middle" />)}
            </g>
          ))}
          {xT.minor.map((t) => <line key={'xm' + t} className="vc-axis" x1={X(t)} x2={X(t)} y1={bottom} y2={bottom + 2.5} />)}
          {fig.xlabel && !inner.x && <SvgText className="vc-label" text={fig.xlabel} size={11.5} x={M.l + plotW / 2} y={H - 6 - below} textAnchor="middle" />}
          {y2T && y2T.ticks.map(({ t, label }) => (
            <g key={'y2' + t}>
              <line className="vc-axis" x1={W - M.r} x2={W - M.r + 4} y1={Y2(t)} y2={Y2(t)} />
              <SvgText className="vc-tick" text={label} size={11} color="var(--text-faint)" x={W - M.r + 8} y={Y2(t) + 3.8} textAnchor="start" />
            </g>
          ))}
          {y2T && <line className="vc-axis" x1={W - M.r} x2={W - M.r} y1={M.t} y2={bottom} />}
          {y2T && fig.yright.label && (
            <SvgText className="vc-label" text={fig.yright.label} size={11.5} x={W - 10} y={M.t + plotH / 2} textAnchor="middle" rotate={90} />
          )}
          {fig.ylabel && !inner.y && (
            <SvgText className="vc-label" text={fig.ylabel} size={11.5} x={12 + ylRoom} y={M.t + plotH / 2} textAnchor="middle" rotate={-90} />
          )}
          <g clipPath={`url(#${clipId})`}>
            {marks.map((k, i) => {
              if (k.kind !== 'rect') return null;
              const [x0, x1, y0, y1] = [X(tx(k.x - k.w / 2)), X(tx(k.x + k.w / 2)), Y(ty(k.y + k.h / 2)), Y(ty(k.y - k.h / 2))];
              if (![x0, x1, y0, y1].every(fin)) return null;
              return <rect key={`m${i}`} className="vc-mark-band" x={Math.min(x0, x1)} y={Math.min(y0, y1)} width={Math.abs(x1 - x0)} height={Math.abs(y1 - y0)} />;
            })}
            {chart.series.map((s, k) => {
              // fill_between: the band from lo to hi; a gap (nan, or <= 0 on a log axis) splits it
              if (s.interval !== 'fill') return null;
              const Yk = Yof(k);
              const runs = [];
              let run = [];
              for (let i = 0; i < txs.length; i++) {
                if (fin(txs[i]) && fin(tlo[k][i]) && fin(thi[k][i])) run.push(i);
                else { if (run.length) runs.push(run); run = []; }
              }
              if (run.length) runs.push(run);
              const d = runs.map((r) => `M${r.map((i) => `${X(txs[i]).toFixed(2)} ${Math.max(-1e4, Math.min(1e4, Yk(thi[k][i]))).toFixed(2)}`).join('L')}L${r.slice().reverse().map((i) => `${X(txs[i]).toFixed(2)} ${Math.max(-1e4, Math.min(1e4, Yk(tlo[k][i]))).toFixed(2)}`).join('L')}Z`).join('');
              return <path key={`f${k}`} d={d} fill={colorOf(k)} fillOpacity="0.2" stroke="none" />;
            })}
            {xT.ticks.length > 0 && xticks.length > 0 && xT.ticks.map(({ t }, i) => <line key={`xt${i}`} className="vc-grid" x1={X(t)} x2={X(t)} y1={M.t} y2={bottom} />)}
            {chart.series.map((s, k) => {
              if (!s.bars) return null;
              const [off, bw] = barPx[k];
              // a bar stands on zero; on a log axis, on the bottom of the chart
              const y0 = yScale === 'log' ? bottom : Y(Math.max(ylo, Math.min(yhi, 0)));
              return (
                <g key={`b${k}`} className="vc-bars">
                  {tys[k].map((v, i) => (fin(v) && fin(txs[i]) ? (
                    <rect key={i} x={X(txs[i]) + off - bw / 2} y={Math.min(y0, Y(v))} width={bw} height={Math.abs(y0 - Y(v))}
                      rx={BAR_RX} fill={colorOf(k)} fillOpacity={BAR_OPACITY} />
                  ) : null))}
                </g>
              );
            })}
            {chart.series.map((s, k) => {
              // rangebars: a capped bar from lo to hi at each x
              if (s.interval !== 'err') return null;
              const Yk = Yof(k);
              const cap = 3.5;
              let d = '';
              for (let i = 0; i < txs.length; i++) {
                if (!fin(txs[i]) || !fin(tlo[k][i]) || !fin(thi[k][i])) continue;
                const x = X(txs[i]).toFixed(2);
                const [a, b] = [Yk(tlo[k][i]), Yk(thi[k][i])].map((v) => Math.max(-1e4, Math.min(1e4, v)).toFixed(2));
                d += `M${x} ${a}V${b}M${(X(txs[i]) - cap).toFixed(2)} ${a}H${(X(txs[i]) + cap).toFixed(2)}M${(X(txs[i]) - cap).toFixed(2)} ${b}H${(X(txs[i]) + cap).toFixed(2)}`;
              }
              return <path key={`e${k}`} d={d} stroke={colorOf(k)} strokeWidth="1.3" fill="none" />;
            })}
            {paths.map((d, k) => (chart.series[k].bars || chart.series[k].interval || !draws[k].line ? null : (
              <path key={k} d={d} className={draws[k].primary ? 'vc-primary' : undefined} fill="none" stroke={colorOf(k)} strokeDasharray={draws[k].dash}
                strokeWidth={draws[k].width} strokeLinejoin="round" strokeLinecap={draws[k].dash ? 'butt' : 'round'} />
            )))}
            {chart.series.map((s, k) => (s.bars || s.interval || !draws[k].marker ? null : tys[k].map((v, i) => (fin(v) && fin(txs[i]) ? (
              <Marker key={`${k}-${i}`} kind={draws[k].marker} className="vc-dot" x={X(txs[i])} y={Yof(k)(v)} r={3}
                color={colorOf(k)} />
            ) : null))))}
            {marks.map((k, i) => {
              if (k.kind === 'hline' && fin(ty(k.y))) return <line key={`m${i}`} className="vc-mark" x1={M.l} x2={W - M.r} y1={Y(ty(k.y))} y2={Y(ty(k.y))} />;
              if (k.kind === 'vline' && fin(tx(k.x))) return <line key={`m${i}`} className="vc-mark" x1={X(tx(k.x))} x2={X(tx(k.x))} y1={M.t} y2={bottom} />;
              if (k.kind === 'segment' && [tx(k.x1), ty(k.y1), tx(k.x2), ty(k.y2)].every(fin)) return <line key={`m${i}`} className="vc-mark" x1={X(tx(k.x1))} y1={Y(ty(k.y1))} x2={X(tx(k.x2))} y2={Y(ty(k.y2))} />;
              return null;
            })}
          </g>
          {marks.map((k, i) => (k.kind === 'label' && fin(tx(k.x)) && fin(ty(k.y)) ? (
            hasMath(k.text)
              ? <SvgText key={`l${i}`} className="vc-mark-label" text={k.text} size={12} x={X(tx(k.x))} y={Y(ty(k.y)) + (chartLabel(k.text, 12).h - chartLabel(k.text, 12).d) / 2} textAnchor="middle" />
              : <text key={`l${i}`} className="vc-mark-label" x={X(tx(k.x))} y={Y(ty(k.y))} textAnchor="middle" dominantBaseline="middle">{k.text}</text>
          ) : null))}
          {hover !== null && (
            <g pointerEvents="none">
              <line className="vc-rule" x1={X(txs[hover])} x2={X(txs[hover])} y1={M.t} y2={bottom} />
              {tys.map((vals, k) => (fin(vals[hover]) && vals[hover] >= (right[k] ? y2lo : ylo) && vals[hover] <= (right[k] ? y2hi : yhi) ? (
                <circle key={k} className="vc-dot" cx={X(txs[hover])} cy={Yof(k)(vals[hover])} r="3.6"
                  fill={colorOf(k)} />
              ) : null))}
            </g>
          )}
        </svg>
        {hover !== null && (
          <div className="vc-tip" style={{ left: `${(tipLeft / W) * 100}%`, transform: flip ? 'translateX(calc(-100% - 12px))' : 'translateX(12px)' }}>
            <div className="vc-tip-x"><MathInline text={fig.xlabel || 'x'} size={12} /> = {readout(xs[hover])}</div>
            {chart.series.map((s, k) => (
              <div key={k} className="vc-tip-row">
                <i style={{ background: colorOf(k) }} />
                <span><MathInline text={s.name || 'y'} size={12} /></span>
                <b>{s.interval ? `${readout(s.lo[hover])} to ${readout(s.hi[hover])}` : readout(s.values[hover])}</b>
              </div>
            ))}
          </div>
        )}
      </div>
      {/* points a log axis cannot draw (y <= 0) are left out quietly: a note about it read as machinery */}
    </figure>
  );
}
