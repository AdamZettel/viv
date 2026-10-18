// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import { highlight as highlightCode } from '../lib/highlight.js';
import { VIZ } from './cellViz.jsx';
import LineChart from './LineChart.jsx';
import { MathInline, SvgText } from './MathText.jsx';
import { chartLabel, hasMath } from '../lib/mathtext.js';
import AnimPlayer from './AnimPlayer.jsx';
import DiagramView from './DiagramView.jsx';
import { isAnimRecord, parseMark, parseStyle, recArr, recNum } from '../lib/animParse.js';
import { TICK_TURN_DEG, TURN_DX, TURN_DY, YTICK_ROW, MAX_PLOT_H, equalAspect, heatExtent, logTicks, markToT, scaleOf, turnedDrop, turnedReach, xTickLayout } from '../lib/axisScale.js';
import { layoutPanels } from '../lib/figureGrid.js';
import { Marker, Swatch, drawStyle, paletteSlots } from '../lib/seriesStyle.jsx';

/** A pixel coordinate rounded to 1/1000 px, so SSR and hydration agree on it. */
const px3 = (v) => (Number.isFinite(v) ? Math.round(v * 1000) / 1000 : v);
import { expandUses, hasUses, browserResolver } from '../lib/uses.js';
import { vivExports } from '../lib/vivWasm.js';

const resolvers = new Map();
/**
 * A program with its imported modules spliced in, fetched from the site.
 * `base` is the program's staged path without `.viv`, for modules beside it.
 */
export const withModules = (src, base = null) => {
  if (!hasUses(src)) return Promise.resolve(src);
  if (!resolvers.has(base)) resolvers.set(base, browserResolver(base));
  return expandUses(src, resolvers.get(base));
};

// A notebook code cell running Viv in the browser via the same viv.wasm
// interpreter as the /viv IDE. Cells on a page share one module-scope
// registry; running cell N executes the CONCATENATION of cells 0..N (the
// Jupyter model, made honest: every run is a fresh top-to-bottom execution,
// so there is no hidden kernel state — and editing an early cell really does
// change what later cells compute). A sentinel show separates the prefix's
// output from the current cell's, so each cell shows only its own output.
//
// Output conventions (shared with the IDE): `show arr` on a 1-D array prints
// `name: [v1, v2, ...]` — rendered here as an SVG line plot instead of text.
// A shown array whose name ends in `_x` becomes the x-axis for the
// same-length arrays shown after it (one chart, multiple labeled series).

// `narrate` is commentary (dropped at parse), so the cell boundary is marked
// by showing a sentinel variable — its "name: value" line is what we split on.
const SENTINEL = '__cell_boundary__: 0';
const SENTINEL_STMT = '__cell_boundary__ = 0.0\nshow __cell_boundary__';

// ---- module-scope singletons shared by every cell island on the page ----
let wasmPromise = null;
const registry = new Map(); // idx -> current source (edits included)
const runners = new Map();  // idx -> that cell's run() — drives the Run-all bar
const viewSetters = new Map(); // idx -> setView — the bar's global offline/live switch

export function loadWasm() {
  if (!wasmPromise) {
    wasmPromise = (async () => {
      // the page's one compiled viv.wasm (src/lib/vivWasm.js); this instance is
      // shared by every cell, so the incremental session carries between them
      return vivExports();
    })();
    wasmPromise.catch(() => {
      wasmPromise = null;
    });
  }
  return wasmPromise;
}

// `viv_run_keep` and `viv_run_cell_keep` give what a program printed before
// an error, a NUL byte, then the error; this puts the error on its own line
// after that output, so the view shows both.
export function joinKept(s) {
  const i = s.indexOf('\0');
  if (i < 0) return s;
  const head = s.slice(0, i);
  return head + (head && !head.endsWith('\n') ? '\n' : '') + s.slice(i + 1);
}

export function callWasm(ex, fn, text) {
  const enc = new TextEncoder().encode(text);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const packed = fn(ptr, enc.length);
  return new TextDecoder().decode(
    new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
}

// Incremental cell run: the wasm keeps a session keyed by the prefix's hash,
// so a cell whose prefix is already the live session state executes ALONE —
// Run-all costs one pass over the chain instead of the sum of all prefixes.
// Results are identical to a full re-run by construction (any mismatch cold-
// rebuilds the prefix inside the wasm).
function callWasmCell(ex, prefix, cell) {
  const enc = new TextEncoder();
  const pb = enc.encode(prefix);
  const cb = enc.encode(cell);
  const buf = new Uint8Array(pb.length + cb.length);
  buf.set(pb, 0);
  buf.set(cb, pb.length);
  const ptr = ex.viv_alloc(buf.length);
  new Uint8Array(ex.memory.buffer, ptr, buf.length).set(buf);
  const packed = (ex.viv_run_cell_keep || ex.viv_run_cell)(ptr, buf.length, pb.length);
  return joinKept(new TextDecoder().decode(
    new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn))));
}

// ---- syntax highlighting: same token classes as the IDE's Monaco language ----
// One highlighter for the whole site: src/lib/highlight.js. It never changes the
// text (its test checks that on every program in the repo), which this overlay
// needs: the colored copy sits under a transparent textarea, glyph for glyph.
const highlight = (src) => highlightCode(src, 'viv');

// ---- output parsing ---------------------------------------------------------
// The interpreter's structured lines:
//   plot: <label> ; [x...] ; [y...]   — the explicit `plot(x, y, "label")` builtin
//   narrate: <text>                   — the live narration channel
//   __graph__ {json} / __trace__ [json] — the call graph and its event replay
// Everything else is plain text output. Shown arrays are ALWAYS text now — the
// old "3+ points auto-charts" convention is gone; figures come from plot().
export function parseOwnOutput(text) {
  const blocks = []; // ordered: {type: 'text'|'narrate', s}
  const charts = [];
  let pendingFig = null; // a `figure:` line styles the NEXT chart (title, axes, window)
  let graph = null;
  let trace = [];
  // The chart the most recent figure() opened. Every plot() until the next
  // figure() draws into it, whatever its x: a ceiling, two ropes and a box are
  // one picture, and a reference line over a sampled curve is one chart.
  let figChart = null;
  // figure() is the only grouping: from a `figure:` line to the next one, every
  // plot, scatter, bars, plot2d frame and mark is part of that one chart. With
  // no figure open, each record is its own chart. Nothing is merged by guessing
  // (same x, consecutive calls, same label).
  let figActive = false;
  // the figure most recently opened: a setting record (xscale, yscale) applies
  // to all of it, wherever in the figure the call is
  let curFig = null;
  // A grid of figures (grid: title ; rows cols): the next rows * cols figures
  // are its panels, row by row. `sink` is where a new chart goes: the open
  // panel, or the top level. `serial` keeps the keys unique across both.
  let grid = null;
  let sink = charts;
  let serial = 0;
  const takeFig = () => { const f = pendingFig; pendingFig = null; return f; };
  const own = (chart) => { sink.push(chart); if (figActive && !figChart) figChart = chart; return chart; };
  const sameX = (chart, xs) => chart.x && chart.x.values.length === xs.length && chart.x.values.every((v, i) => Object.is(v, xs[i]));
  // a figure whose plots do not share an x becomes lines that each carry their own x
  const toOwnX = (chart) => {
    if (chart.kind === 'scatter') return;
    chart.series = chart.series.map((sr) => (sr.interval
      ? { name: sr.name, xs: chart.x.values, ys: sr.values, interval: sr.interval, lo: sr.lo, hi: sr.hi }
      : { name: sr.name, xs: chart.x.values, ys: sr.values, line: true }));
    chart.kind = 'scatter';
    chart.key = `sc-${serial++}`;
    delete chart.x;
  };
  // flow: the same output in the order it was produced, a chart placed where
  // its first record came, between the lines printed before and after it
  const flow = [];
  let placed = 0;
  const placeCharts = () => {
    while (placed < charts.length) flow.push({ type: 'chart', chart: charts[placed++] });
  };
  const pushText = (line) => {
    const last = blocks[blocks.length - 1];
    if (last && last.type === 'text') last.s += '\n' + line;
    else blocks.push({ type: 'text', s: line });
    placeCharts();
    const lastF = flow[flow.length - 1];
    if (lastF && lastF.type === 'text') lastF.s += '\n' + line;
    else flow.push({ type: 'text', s: line });
  };
  for (const line of text.split('\n')) {
    // figure: title ; xlabel ; ylabel [; xlo xhi ylo yhi] — the program's own
    // presentation for whatever chart comes next
    // grid: title ; rows cols -- the next rows * cols figures are its panels
    const gr = line.match(/^grid: (.*?) ; (\S+) (\S+)\s*$/);
    if (gr) {
      const rows = Math.floor(recNum(gr[2]));
      const cols = Math.floor(recNum(gr[3]));
      if (rows >= 1 && cols >= 1 && rows * cols <= 64) {
        grid = { kind: 'grid', key: `gr-${serial++}`, title: gr[1].trim(), rows, cols, share: '', panels: [] };
        charts.push(grid);
        sink = charts;
        figChart = null;
        figActive = false;
        pendingFig = null;
      }
      continue;
    }
    // share: x | y | xy -- a setting of the most recent grid
    const sh = line.match(/^share: (x|y|xy)\s*$/);
    if (sh) {
      if (grid) grid.share = sh[1];
      continue;
    }
    const fig = line.match(/^figure: (.*?) ; (.*?) ; ([^;]*?)(?: ; ([-\d.eE+ ]+))?(?: ; ([A-Za-z][\w-]*))?\s*$/);
    if (fig) {
      // inside a grid with a panel left, a figure opens the next panel
      if (grid && grid.panels.length < grid.rows * grid.cols) {
        sink = [];
        grid.panels.push(sink);
      } else {
        grid = null;
        sink = charts;
      }
      pendingFig = { title: fig[1].trim(), xlabel: fig[2].trim(), ylabel: fig[3].trim() };
      if (fig[4]) {
        const w = fig[4].trim().split(/\s+/).map(Number);
        if (w.length === 4 && w.every(Number.isFinite)) {
          // an equal pair (e.g. 0, 0) means "auto-scale that axis"
          if (w[0] !== w[1]) pendingFig.xlim = [w[0], w[1]];
          if (w[2] !== w[3]) pendingFig.ylim = [w[2], w[3]];
        }
      }
      if (fig[5]) pendingFig.style = fig[5];
      curFig = pendingFig;
      figChart = null;
      figActive = true;
      // a diagram is a chart of marks alone
      if (pendingFig.style === 'diagram') own({ kind: 'diagram', key: `dg-${serial++}`, marks: [], fig: takeFig() });
      continue;
    }
    // xscale: log | linear, yscale: log | linear -- a setting of the open figure
    // aspect: equal | auto -- one unit as long on x as on y
    const scale = line.match(/^(xscale|yscale): (log|linear)\s*$/) || line.match(/^(aspect): (equal|auto)\s*$/);
    if (scale) {
      if (curFig) curFig[scale[1]] = scale[2];
      continue;
    }
    // a heatmap's settings: origin: lower|upper, colormap: viridis|diverging,
    // clim: lo hi, levels: [v, ...] -- they belong to the open figure
    const hs = line.match(/^(origin|colormap): (\w+)\s*$/) || line.match(/^(clim): (\S+ \S+)\s*$/) || line.match(/^(levels): \[([^\]]*)\]\s*$/);
    if (hs) {
      if (curFig) {
        if (hs[1] === 'clim') curFig.clim = hs[2].split(' ').map(recNum);
        else if (hs[1] === 'levels') curFig.levels = recArr(hs[2]);
        else curFig[hs[1]] = hs[2];
      }
      continue;
    }
    // yright: <series label> ; <axis label> -- that series reads against a right y axis
    const yr = line.match(/^yright: (.*?) ; (.*?)\s*$/);
    if (yr) {
      if (curFig) {
        const r = curFig.yright || (curFig.yright = { label: '', series: [] });
        r.label = yr[2];
        if (!r.series.includes(yr[1])) r.series.push(yr[1]);
      }
      continue;
    }
    // style: <label> ; <words> -- how the open figure draws the series of that label
    const st = parseStyle(line);
    if (st) {
      if (curFig) (curFig.styles = curFig.styles || {})[st.label] = st.style;
      continue;
    }
    // a static mark (hline, vline, xtick, label, rect, segment) inside a figure()
    // with no stage open belongs to that figure's chart, before or after its plots
    const mark = parseMark(line);
    if (mark) {
      const lastChart = sink[sink.length - 1];
      if (figChart && lastChart === figChart) { (figChart.marks = figChart.marks || []).push(mark); continue; }
      if (pendingFig) { (pendingFig.marks = pendingFig.marks || []).push(mark); continue; }
    }
    // animation protocol: `stage:` opens one, and every
    // animation record after it belongs to that stage until the next `stage:`.
    if (isAnimRecord(line)) {
      const last = sink[sink.length - 1];
      if (line.startsWith('stage:') || !last || last.kind !== 'anim') sink.push({ kind: 'anim', key: `an-${serial++}`, lines: [line] });
      else last.lines.push(line);
      continue;
    }
    // scene protocol: scene: opens a 3D scene; wire:
    // and verts: lines append grouped geometry to the most recent scene.
    const sn = line.match(/^scene: (.*)$/);
    if (sn) {
      sink.push({ kind: 'scene', key: `sn-${serial++}`, title: sn[1].trim(), groups: [], fig: takeFig() });
      continue;
    }
    const el = line.match(/^(wire|verts): (.*?) ; \[([^\]]*)\] ; \[([^\]]*)\] ; \[([^\]]*)\](?: ; \[([^\]]*)\])?\s*$/);
    if (el) {
      let scn = sink[sink.length - 1];
      if (!scn || scn.kind !== 'scene') {
        scn = { kind: 'scene', key: `sn-${serial++}`, title: '', groups: [], fig: takeFig() };
        sink.push(scn);
      }
      const gname = el[2].trim();
      const xs = recArr(el[3]);
      const ys = recArr(el[4]);
      const zs = recArr(el[5]);
      const vs = el[6] != null ? recArr(el[6]) : null;
      let g = scn.groups.find((G) => G.name === gname);
      if (!g) { g = { name: gname, wires: [], verts: [] }; scn.groups.push(g); }
      (el[1] === 'wire' ? g.wires : g.verts).push({ xs, ys, zs, vs });
      continue;
    }
    const bars = line.match(/^bars: (.*?) ; \[([^\]]*)\] ; \[([^\]]*)\]\s*$/);
    if (bars) {
      const xs = recArr(bars[2]);
      const ys = recArr(bars[3]);
      if (xs.length && xs.length === ys.length) {
        const key = `bars:${bars[2]}`;
        if (figChart && sameX(figChart, xs)) figChart.series.push({ name: bars[1].trim(), values: ys, bars: true });
        else own({ key, x: { values: xs }, series: [{ name: bars[1].trim(), values: ys, bars: true }], fig: takeFig() });
      }
      continue;
    }
    const plot = line.match(/^plot: (.*?) ; \[([^\]]*)\] ; \[([^\]]*)\]\s*$/);
    if (plot) {
      const label = plot[1].trim();
      // a nan or inf is kept and drawn as a gap: one bad value never hides the series
      const xs = recArr(plot[2]);
      const ys = recArr(plot[3]);
      if (xs.length && xs.length === ys.length) {
        const key = plot[2];
        if (figChart && sameX(figChart, xs)) figChart.series.push({ name: label, values: ys });
        else if (figChart && (figChart.x || figChart.kind === 'scatter')) {
          toOwnX(figChart);
          figChart.series.push({ name: label, xs, ys, line: true });
        } else own({ key, x: { values: xs }, series: [{ name: label, values: ys }], fig: takeFig() });
      }
      continue;
    }
    // fill_between: label ; [x] ; [lo] ; [hi] -- a band; rangebars: the same, as
    // an interval at each x. A series with no y of its own (its y is all gaps),
    // grouped like a plot: on the figure's x it joins the chart, else it carries its own x
    const iv = line.match(/^(fill_between|rangebars): (.*?) ; \[([^\]]*)\] ; \[([^\]]*)\] ; \[([^\]]*)\]\s*$/);
    if (iv) {
      const label = iv[2].trim();
      const xs = recArr(iv[3]);
      const lo = recArr(iv[4]);
      const hi = recArr(iv[5]);
      if (xs.length && xs.length === lo.length && xs.length === hi.length) {
        const interval = iv[1] === 'fill_between' ? 'fill' : 'err';
        const none = xs.map(() => NaN);
        if (figChart && sameX(figChart, xs)) figChart.series.push({ name: label, values: none, interval, lo, hi });
        else if (figChart && (figChart.x || figChart.kind === 'scatter')) {
          toOwnX(figChart);
          figChart.series.push({ name: label, xs, ys: none, interval, lo, hi });
        } else own({ key: `iv-${serial++}`, x: { values: xs }, series: [{ name: label, values: none, interval, lo, hi }], fig: takeFig() });
      }
      continue;
    }
    const sc = line.match(/^scatter: (.*?) ; ([^;]*) ; \[([^\]]*)\] ; \[([^\]]*)\]\s*$/);
    if (sc) {
      const label = sc[1].trim();
      // color token: a name plus an optional marker style, e.g. "blue o" for
      // open circles (matplotlib 'o' with mfc=none) vs plain filled dots
      const ctok = sc[2].trim().split(/\s+/);
      const color = ctok[0] || '';
      // 'o' = open circle; anything else is a filled dot
      const open = ctok[1] === 'o';
      const xs = recArr(sc[3]);
      const ys = recArr(sc[4]);
      if (xs.length && xs.length === ys.length) {
        if (figChart && (figChart.x || figChart.kind === 'scatter')) {
          toOwnX(figChart);
          figChart.series.push({ name: label, color, open, xs, ys });
        } else own({ kind: 'scatter', key: `sc-${serial++}`, series: [{ name: label, color, open, xs, ys }], fig: takeFig() });
      }
      continue;
    }
    // frames2d: an externalized frame set — the precompute repacked a big run
    // of plot2d frames into a quantized binary sidecar; the chart streams it
    const f2 = line.match(/^frames2d: (.*?) ; (\d+) (\d+) ; (\d+) ; ([-\d.eE+]+) ([-\d.eE+]+) ; (\d+) ; (\S+)\s*$/);
    if (f2) {
      sink.push({
        kind: 'heatmap', key: `hm-${serial++}`, label: f2[1].trim(),
        nx: parseInt(f2[2], 10), ny: parseInt(f2[3], 10),
        lazy: { nframes: parseInt(f2[4], 10), fmin: Number(f2[5]), fmax: Number(f2[6]), offset: parseInt(f2[7], 10), url: f2[8] },
        fig: takeFig(),
      });
      continue;
    }
    const p2 = line.match(/^plot2d: (.*?) ; (\d+) (\d+) ; \[([^\]]*)\]\s*$/);
    if (p2) {
      const z = recArr(p2[4]); // a non-finite cell is drawn empty
      const nx = parseInt(p2[2], 10);
      const ny = parseInt(p2[3], 10);
      const label = p2[1].trim();
      if (z.length === nx * ny && nx > 0 && ny > 0) {
        // inside one figure(), plot2d() calls with the same label and shape are
        // FRAMES of one animated field (a time-stepping loop that plots every K
        // steps becomes a movie); with no figure open each is its own heatmap
        if (figChart && figChart.kind === 'heatmap' && figChart.label === label && figChart.nx === nx && figChart.ny === ny) {
          figChart.frames.push(z);
        } else {
          own({ kind: 'heatmap', key: `hm-${serial++}`, label, nx, ny, z, frames: [z], fig: takeFig() });
        }
      }
      continue;
    }
    const qv = line.match(/^quiver: (.*?) ; \[([^\]]*)\] ; \[([^\]]*)\] ; \[([^\]]*)\] ; \[([^\]]*)\]\s*$/);
    if (qv) {
      const [xs, ys, us, vs] = [qv[2], qv[3], qv[4], qv[5]].map(recArr);
      if (xs.length && xs.length === ys.length && xs.length === us.length && xs.length === vs.length) {
        sink.push({ kind: 'quiver', key: `qv-${serial++}`, label: qv[1].trim(), xs, ys, us, vs, fig: takeFig() });
      }
      continue;
    }
    const p3 = line.match(/^plot3d: (.*?) ; \[([^\]]*)\] ; \[([^\]]*)\] ; \[([^\]]*)\]\s*$/);
    if (p3) {
      const xs = recArr(p3[2]);
      const ys = recArr(p3[3]);
      const zs = recArr(p3[4]);
      if (xs.length && xs.length === ys.length && xs.length === zs.length) {
        sink.push({ kind: 'curve3d', key: `c3-${serial++}`, label: p3[1].trim(), xs, ys, zs, fig: takeFig() });
      }
      continue;
    }
    const nar = line.match(/^narrate: (.*)$/);
    if (nar) {
      blocks.push({ type: 'narrate', s: nar[1] });
      placeCharts();
      flow.push({ type: 'narrate', s: nar[1] });
      continue;
    }
    const g = line.match(/^__graph__ (.*)$/);
    if (g) { try { graph = JSON.parse(g[1]); } catch { /* ignore */ } continue; }
    const t = line.match(/^__trace__ (.*)$/);
    if (t) { try { trace = JSON.parse(t[1]).map((e) => e.split('>')); } catch { /* ignore */ } continue; }
    // step events: data for the story lens, not output
    if (line.startsWith('__steps__ ')) continue;
    // the story of a debug run with a debug lens: data for the story lens, not output
    if (line.startsWith('__story__ ')) continue;
    pushText(line);
  }
  // trim leading/trailing empty text
  const clean = blocks
    .map((b) => (b.type === 'text' ? { ...b, s: b.s.replace(/^\n+|\n+$/g, '') } : b))
    .filter((b) => b.type !== 'text' || b.s !== '');
  placeCharts();
  const cleanFlow = flow
    .map((b) => (b.type === 'text' ? { ...b, s: b.s.replace(/^\n+|\n+$/g, '') } : b))
    .filter((b) => b.type !== 'text' || b.s !== '');
  return { blocks: clean, charts, graph, trace, flow: cleanFlow };
}

// ---- the live call graph ----------------------------------------------------
// Layered left-to-right layout: <main> in column 0, every function in the
// column of its call depth (longest path from <main>), rows ordered by the
// barycenter of their callers to keep edges short. Collapsed behind a
// disclosure by default — it appears when you ask, not under every output.
function layoutGraph(graph) {
  const nodes = [...graph.nodes];
  if (nodes.includes('<main>')) nodes.splice(nodes.indexOf('<main>'), 1), nodes.unshift('<main>');
  // longest-path layering from <main>, cycle-safe
  const layer = {};
  nodes.forEach((n) => { layer[n] = n === '<main>' ? 0 : 1; });
  for (let pass = 0; pass < nodes.length + 1; pass++) {
    let moved = false;
    for (const e of graph.edges) {
      if (e.from === e.to) continue;
      if (layer[e.to] < layer[e.from] + 1 && layer[e.from] + 1 <= nodes.length) {
        layer[e.to] = layer[e.from] + 1;
        moved = true;
      }
    }
    if (!moved) break;
  }
  const cols = [];
  for (const n of nodes) (cols[layer[n]] = cols[layer[n]] || []).push(n);
  const columns = cols.filter(Boolean);
  // order each column by mean row of callers (one barycenter pass)
  const row = {};
  columns.forEach((col) => col.forEach((n, i) => { row[n] = i; }));
  columns.forEach((col, ci) => {
    if (ci === 0) return;
    col.sort((a, b) => {
      const bary = (n) => {
        const rs = graph.edges.filter((e) => e.to === n && e.from !== n).map((e) => row[e.from] ?? 0);
        return rs.length ? rs.reduce((x, y) => x + y, 0) / rs.length : 0;
      };
      return bary(a) - bary(b);
    });
    col.forEach((n, i) => { row[n] = i; });
  });
  const COLW = 148, ROWH = 44, PADX = 12, PADY = 26;
  // a recursive def's self-loop is drawn up to 34 px above its node (CallGraph's
  // edgeGeom), into the row above; its row is taller by LOOPH, so the loop and
  // its count sit in space of their own
  const LOOPH = 30;
  const loops = new Set(graph.edges.filter((e) => e.from === e.to).map((e) => e.from));
  const rowH = (n) => ROWH + (loops.has(n) ? LOOPH : 0);
  const colH = (col) => col.reduce((h, n) => h + rowH(n), 0);
  const H = Math.max(...columns.map(colH)) + PADY * 2;
  const W = columns.length * COLW + PADX * 2 + 30;
  const pos = {};
  columns.forEach((col, ci) => {
    let y = (H - colH(col)) / 2;
    col.forEach((n) => {
      y += rowH(n);
      pos[n] = { x: PADX + ci * COLW + 46, y: y - ROWH / 2 };
    });
  });
  return { pos, W, H };
}

export function CallGraph({ graph, trace, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  const [step, setStep] = useState(-1); // -1 = resting (all edges lit)
  const timer = useRef(null);
  const replay = useCallback(() => {
    if (!trace.length) return;
    clearInterval(timer.current);
    setStep(0);
    let i = 0;
    timer.current = setInterval(() => {
      i += 1;
      if (i >= trace.length) { clearInterval(timer.current); setStep(-1); }
      else setStep(i);
    }, 140);
  }, [trace]);
  useEffect(() => () => clearInterval(timer.current), []);

  const totalCalls = graph.edges.reduce((a, e) => a + e.count, 0);
  if (!open) {
    return (
      <div className="kcell-graph kcell-graph-closed">
        <style>{`
        .kcell-graph-closed { padding: 0; border-top: 1px dashed var(--border); }
        .kcell-graph-disclose { font: 600 11px var(--font-mono); padding: 4px 12px; border: none; background: transparent; color: var(--text-muted); cursor: pointer; letter-spacing: 0.02em; }
        .kcell-graph-disclose:hover { color: var(--accent); }
        `}</style>
        <button className="kcell-graph-disclose" onClick={() => { setOpen(true); }}>
          ▸ call graph — {graph.nodes.length} nodes, {totalCalls} calls
        </button>
      </div>
    );
  }

  const { pos, W, H } = layoutGraph(graph);
  const maxCount = Math.max(...graph.edges.map((e) => e.count));
  const active = step >= 0 ? trace[step] : null;
  const activeKey = active ? `${active[0]}>${active[1]}` : null;
  const visited = new Set();
  if (step >= 0) for (let i = 0; i <= step; i++) visited.add(`${trace[i][0]}>${trace[i][1]}`);
  const boxW = (n) => 16 + n.length * 7.2;

  // Edge geometry + collision-aware count labels. Each edge is a cubic
  // bezier; its label tries points along the curve (midpoint first) and
  // takes the first spot clear of node boxes and already-placed labels.
  const cubic = (P0, P1, P2, P3) => (t) => {
    const u = 1 - t;
    return {
      x: u * u * u * P0.x + 3 * u * u * t * P1.x + 3 * u * t * t * P2.x + t * t * t * P3.x,
      y: u * u * u * P0.y + 3 * u * u * t * P1.y + 3 * u * t * t * P2.y + t * t * t * P3.y,
    };
  };
  const edgeGeom = graph.edges.map((e) => {
    const a = pos[e.from], b = pos[e.to];
    if (e.from === e.to) {
      const P0 = { x: a.x - 8, y: a.y - 12 }, P1 = { x: a.x - 20, y: a.y - 34 };
      const P2 = { x: a.x + 20, y: a.y - 34 }, P3 = { x: a.x + 8, y: a.y - 12 };
      return { e, self: true, d: `M ${P0.x} ${P0.y} C ${P1.x} ${P1.y}, ${P2.x} ${P2.y}, ${P3.x} ${P3.y}`, at: cubic(P0, P1, P2, P3) };
    }
    const x1 = a.x + boxW(e.from) / 2 + 2, y1 = a.y;
    const x2 = b.x - boxW(e.to) / 2 - 4, y2 = b.y;
    if (x2 < x1) {
      const P0 = { x: x1, y: y1 + 8 }, P1 = { x: x1 + 24, y: y1 + 34 };
      const P2 = { x: x2 - 24, y: y2 + 34 }, P3 = { x: x2, y: y2 + 8 };
      return { e, back: true, d: `M ${P0.x} ${P0.y} C ${P1.x} ${P1.y}, ${P2.x} ${P2.y}, ${P3.x} ${P3.y}`, at: cubic(P0, P1, P2, P3) };
    }
    const mx = (x1 + x2) / 2;
    // An edge that skips past intermediate columns bows upward so it cannot
    // thread through a node sitting on its straight-line path.
    const span = Math.abs(b.x - a.x);
    const bow = span > 190 ? Math.min(34, 12 + span / 14) : 0;
    const P0 = { x: x1, y: y1 }, P1 = { x: mx, y: y1 - bow }, P2 = { x: mx, y: y2 - bow }, P3 = { x: x2, y: y2 };
    return { e, d: `M ${P0.x} ${P0.y} C ${P1.x} ${P1.y}, ${P2.x} ${P2.y}, ${P3.x} ${P3.y}`, at: cubic(P0, P1, P2, P3) };
  });
  const taken = graph.nodes.map((n) => ({
    x: pos[n].x - boxW(n) / 2 - 2, y: pos[n].y - 14, w: boxW(n) + 4, h: 28,
  }));
  const clear = (r) => !taken.some((o) =>
    r.x < o.x + o.w && o.x < r.x + r.w && r.y < o.y + o.h && o.y < r.y + r.h);
  const CAND_T = [0.5, 0.42, 0.58, 0.34, 0.66, 0.26, 0.74, 0.18, 0.82];
  const labelPos = new Map();
  [...edgeGeom]
    .sort((g1, g2) => g2.e.count - g1.e.count)
    .forEach((g) => {
      const text = String(g.e.count);
      const w = text.length * 6.4 + 4, h = 12;
      let best = null;
      for (const t of CAND_T) {
        const c = g.at(t);
        for (const dy of g.self ? [-8, -14] : [-7, 9, -15, 17]) {
          const r = { x: c.x - w / 2, y: c.y + dy - h + 3, w, h };
          if (clear(r)) { best = { x: c.x, y: c.y + dy }; break; }
        }
        if (best) break;
      }
      if (!best) { const c = g.at(0.5); best = { x: c.x, y: c.y - 7 }; }
      taken.push({ x: best.x - w / 2, y: best.y - h + 3, w, h });
      labelPos.set(g.e, best);
    });
  return (
    <div className="kcell-graph">
      <style>{`
        .kcell-graph { border-top: 1px dashed var(--border); padding: 4px 8px 8px; }
        .kcell-graph-closed { padding: 0; }
        .kcell-graph-disclose { font: 600 11px var(--font-mono); padding: 4px 12px; border: none; background: transparent; color: var(--text-muted); cursor: pointer; letter-spacing: 0.02em; }
        .kcell-graph-disclose:hover { color: var(--accent); }
        .kcell-graph svg { display: block; width: 100%; margin: 0 auto; }
        .kcell-graph-head { display: flex; align-items: center; gap: 10px; padding: 4px 4px 0; }
        .kcell-graph-title { font: 600 11px var(--font-mono); color: var(--text-muted); letter-spacing: 0.04em; }
        .kcell-graph-replay { font: 600 11px var(--font-sans); padding: 2px 9px; border-radius: 6px; border: 1px solid var(--border); background: var(--bg); color: var(--text-muted); cursor: pointer; }
        .kcell-graph-replay:hover { border-color: var(--accent); color: var(--accent); }
        .kcell-graph-step { font: 11px var(--font-mono); color: var(--accent); }
        .kg-box { fill: color-mix(in srgb, var(--accent) 6%, var(--bg-card)); stroke: var(--border-strong, var(--border)); }
        .kg-name { font: 600 11.5px var(--font-mono); fill: var(--text); }
        .kg-line { stroke: var(--text-muted); opacity: 0.75; transition: stroke 0.12s, opacity 0.12s; }
        .kg-arrhead { fill: var(--text-muted); }
        .kg-count { font: 10px var(--font-mono); fill: var(--text-muted); paint-order: stroke; stroke: var(--bg-card); stroke-width: 3px; stroke-linejoin: round; }
        .kg-dim .kg-line { opacity: 0.18; }
        .kg-count-dim { opacity: 0.3; }
        .kg-active .kg-line { stroke: var(--accent); opacity: 1; }
        .kg-count-active { fill: var(--accent); font-weight: 700; }
        .kg-pulse .kg-box { stroke: var(--accent); stroke-width: 2; fill: color-mix(in srgb, var(--accent) 22%, var(--bg-card)); }
        .kg-from .kg-box { stroke: var(--accent); }
      `}</style>
      <div className="kcell-graph-head">
        <button className="kcell-graph-disclose" onClick={() => setOpen(false)}>▾ call graph</button>
        <button className="kcell-graph-replay" onClick={replay} title="Replay the recorded call sequence">▶ replay</button>
        {step >= 0 && <span className="kcell-graph-step">call {step + 1} / {trace.length}</span>}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: W }} role="img"
           aria-label={`Call graph: ${graph.edges.map((e) => `${e.from} calls ${e.to} ${e.count} times`).join('; ')}`}>
        <defs>
          <marker id="kg-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" className="kg-arrhead" />
          </marker>
        </defs>
        {edgeGeom.map((g) => {
          const e = g.e;
          const key = `${e.from}>${e.to}`;
          const isActive = key === activeKey;
          const lit = step < 0 || visited.has(key);
          const w = 1.1 + 2.2 * Math.log(1 + e.count) / Math.log(1 + maxCount);
          return (
            <g key={key} className={`kg-edge ${isActive ? 'kg-active' : ''} ${lit ? '' : 'kg-dim'}`}>
              <path d={g.d} fill="none" strokeWidth={w} markerEnd="url(#kg-arr)" className="kg-line" />
            </g>
          );
        })}
        {graph.nodes.map((n) => {
          const p = pos[n];
          const isCallee = active && active[1] === n;
          const isCaller = active && active[0] === n;
          return (
            <g key={n} className={`kg-node ${isCallee ? 'kg-pulse' : ''} ${isCaller ? 'kg-from' : ''}`}>
              <rect x={p.x - boxW(n) / 2} y={p.y - 12} width={boxW(n)} height={24} rx="12" className="kg-box" />
              <text x={p.x} y={p.y + 4} textAnchor="middle" className="kg-name">{n}</text>
            </g>
          );
        })}
        {edgeGeom.map((g) => {
          const key = `${g.e.from}>${g.e.to}`;
          const isActive = key === activeKey;
          const lit = step < 0 || visited.has(key);
          const lp = labelPos.get(g.e);
          return (
            <text key={key + '-c'} x={lp.x} y={lp.y} textAnchor="middle"
                  className={`kg-count ${isActive ? 'kg-count-active' : ''} ${lit ? '' : 'kg-count-dim'}`}>
              {g.e.count}
            </text>
          );
        })}
      </svg>
    </div>
  );
}

// Jupyter-style toolbar: run every registered cell on the page, top to bottom.
// Cells re-run their full prefix anyway (the honest-chaining model), so this is
// exactly "Restart & Run All". Sticks to the viewport top while scrolling and
// only takes on elevation when actually stuck (sentinel + IntersectionObserver).
export function RunAllBar({ count, cached = false, slug = '' }) {
  const [state, setState] = useState('idle');
  const [at, setAt] = useState(0);
  const [total, setTotal] = useState(count || 0);
  const [secs, setSecs] = useState(0);
  const [stuck, setStuck] = useState(false);
  const [barView, setBarView] = useState('offline'); // cached pages: which result every cell shows
  const sentinelRef = useRef(null);
  const switchAll = useCallback((v) => {
    setBarView(v);
    for (const set of viewSetters.values()) set(v);
  }, []);
  useEffect(() => {
    const el = sentinelRef.current;
    if (!el || typeof IntersectionObserver === 'undefined') return;
    const obs = new IntersectionObserver(([e]) => setStuck(!e.isIntersecting), { threshold: 0 });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);
  const runAll = useCallback(async () => {
    const idxs = [...runners.keys()].sort((a, b) => a - b);
    if (!idxs.length) return;
    setState('running');
    setTotal(idxs.length);
    const t0 = performance.now();
    for (let k = 0; k < idxs.length; k++) {
      setAt(k + 1);
      try { await runners.get(idxs[k])?.(); } catch { /* the cell shows its own error */ }
      await new Promise((r) => setTimeout(r, 30)); // let the cell paint
    }
    setSecs((performance.now() - t0) / 1000);
    setState('done');
  }, []);
  const n = total || count || 0;
  return (
    <>
      <div ref={sentinelRef} aria-hidden="true" />
      <div className={`krunall ${stuck ? 'krunall-stuck' : ''}`} role="toolbar" aria-label="Notebook controls">
        <span className="krunall-label">Notebook{n ? ` · ${n} cells` : ''}</span>
        {cached && (
          <span className="krunall-viewsw" role="group" aria-label="Which results every cell shows">
            <button className={barView === 'offline' ? 'krunall-viewon' : ''}
              title="Results computed ahead of time by the compiled C program, at full problem size"
              onClick={() => switchAll('offline')}>⚡ precomputed</button>
            <button className={barView === 'live' ? 'krunall-viewon' : ''}
              title="Results of the cells you have run on this page"
              onClick={() => switchAll('live')}>▶ run in browser</button>
          </span>
        )}
        {cached && slug && (
          <span className="krunall-dl">
            <a href={`/notebook-cache/${slug}.viv`} download title="The exact program these results came from">⬇ .viv</a>
            <a href={`/notebook-cache/${slug}.c`} download title="The generated C: compile it with cc -O3 file.c -lm to run the full problem size on your machine">⬇ .c</a>
          </span>
        )}
        <span className="krunall-spacer" />
        {state === 'running' && (
          <span className="krunall-status"><span className="krunall-dot" /> Running cell {at} of {total}</span>
        )}
        {state === 'done' && (
          <span className="krunall-status krunall-ok">✓ {total} cells · {secs.toFixed(1)} s</span>
        )}
        <button className="krunall-btn" onClick={runAll} disabled={state === 'running'}>
          {state === 'running' ? 'Running…' : state === 'done' ? 'Run again' : 'Run all'}
        </button>
        {state === 'running' && total > 0 && (
          <span className="krunall-progress" style={{ width: `${(100 * at) / total}%` }} />
        )}
        <style>{`
          .krunall { position: sticky; top: 0; z-index: 40; overflow: hidden;
            display: flex; align-items: center; gap: 0.9rem; margin: 1rem 0 0.4rem;
            padding: 7px 14px; border: 1px solid var(--border); border-radius: 10px;
            background: var(--bg-card);
            transition: box-shadow 0.18s ease, border-radius 0.18s ease, background 0.18s ease; }
          .krunall-stuck { border-radius: 0 0 10px 10px; border-top-color: transparent;
            background: color-mix(in srgb, var(--bg-card) 88%, transparent);
            backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
            box-shadow: 0 3px 14px color-mix(in srgb, var(--text) 10%, transparent); }
          .krunall-label { font: 600 11px var(--font-mono); letter-spacing: 0.06em;
            text-transform: uppercase; color: var(--text-muted); }
          .krunall-spacer { flex: 1; }
          .krunall-status { font: 12px var(--font-mono); color: var(--text-muted);
            display: inline-flex; align-items: center; gap: 0.45rem; }
          .krunall-ok { color: var(--accent); font-weight: 600; }
          .krunall-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent);
            animation: krunall-pulse 1s ease-in-out infinite; }
          @keyframes krunall-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.25; } }
          @media (prefers-reduced-motion: reduce) { .krunall-dot { animation: none; } }
          .krunall-btn { font: 600 12.5px var(--font-sans); padding: 4px 14px; border-radius: 7px;
            border: 1px solid color-mix(in srgb, var(--accent) 55%, var(--border));
            background: transparent; color: var(--accent); cursor: pointer;
            transition: background 0.12s, color 0.12s; }
          .krunall-btn:hover:not(:disabled) { background: var(--accent); color: var(--on-accent, #fff); }
          .krunall-btn:disabled { opacity: 0.55; cursor: default; }
          .krunall-progress { position: absolute; left: 0; bottom: 0; height: 2px;
            background: var(--accent); transition: width 0.3s ease; }
          .krunall-viewsw { display: inline-flex; border: 1px solid var(--border);
            border-radius: 7px; overflow: hidden; }
          .krunall-viewsw button { font: 600 11px var(--font-mono); padding: 3px 10px;
            border: none; background: transparent; color: var(--text-muted); cursor: pointer; }
          .krunall-viewsw button + button { border-left: 1px solid var(--border); }
          .krunall-viewsw .krunall-viewon { background: color-mix(in srgb, var(--accent) 14%, transparent);
            color: var(--accent); }
          .krunall-dl { display: inline-flex; gap: 0.6rem; }
          .krunall-dl a { font: 600 11px var(--font-mono); color: var(--text-muted);
            text-decoration: none; border-bottom: 1px dotted var(--border); }
          .krunall-dl a:hover { color: var(--accent); border-bottom-color: var(--accent); }
        `}</style>
      </div>
    </>
  );
}

// matplotlib-style round-to-nearest 1-2-5 tick positions
function heatTicks(a, b, n) {
  const span = b - a || 1;
  const step0 = span / Math.max(1, n);
  const mag = Math.pow(10, Math.floor(Math.log10(Math.abs(step0))));
  const norm = step0 / mag;
  const step = (norm > 7 ? 10 : norm > 3 ? 5 : norm > 1.4 ? 2 : 1) * mag;
  const t0 = Math.ceil(Math.min(a, b) / step) * step;
  const out = [];
  for (let t = t0; t <= Math.max(a, b) + step * 1e-9; t += step) out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
  return out;
}

// marching squares on grid values normalized to [0, 1]; returns segments in
// (row, col) grid coordinates (grid points at cell centers)
function contourSegs(get, nx, ny, level) {
  const segs = [];
  for (let r = 0; r < nx - 1; r++) {
    for (let c = 0; c < ny - 1; c++) {
      const v = [get(r, c), get(r, c + 1), get(r + 1, c + 1), get(r + 1, c)]; // TL TR BR BL
      let idx = 0;
      for (let k = 0; k < 4; k++) if (v[k] > level) idx |= 1 << k;
      if (idx === 0 || idx === 15) continue;
      const f = (a, b) => (level - a) / (b - a || 1e-30);
      const P = {
        0: [r, c + f(v[0], v[1])],
        1: [r + f(v[1], v[2]), c + 1],
        2: [r + 1, c + f(v[3], v[2])],
        3: [r + f(v[0], v[3]), c],
      };
      const T = {
        1: [[3, 0]], 2: [[0, 1]], 3: [[3, 1]], 4: [[1, 2]], 5: [[3, 0], [1, 2]],
        6: [[0, 2]], 7: [[3, 2]], 8: [[2, 3]], 9: [[0, 2]], 10: [[0, 1], [2, 3]],
        11: [[1, 2]], 12: [[1, 3]], 13: [[0, 1]], 14: [[0, 3]],
      };
      for (const [e1, e2] of T[idx]) segs.push([P[e1], P[e2]]);
    }
  }
  return segs;
}

// one fetch per frames.bin, shared by every chart on the page
const frameBinCache = new Map();
function fetchFrameBin(url) {
  if (!frameBinCache.has(url)) {
    frameBinCache.set(url, fetch(url).then((r) => (r.ok ? r.arrayBuffer() : null)).catch(() => null));
  }
  return frameBinCache.get(url);
}

const SERIES_COLORS = ['#2f7fd1', '#1a9e5b', '#c05fd6', '#e0913a'];
// named colors for scatter's optional 4th argument (matplotlib-familiar hues)
const SCATTER_NAMED = { red: '#d62728', blue: '#1f77b4', green: '#2ca02c', purple: '#9467bd', orange: '#ff7f0e', black: '#333333' };

// viridis, sampled at 16 stops and lerped — perceptually uniform, dark-mode safe
const VIRIDIS = [
  [68, 1, 84], [72, 26, 108], [71, 47, 125], [65, 68, 135], [57, 86, 140], [49, 104, 142],
  [42, 120, 142], [35, 136, 142], [31, 152, 139], [34, 168, 132], [53, 183, 121], [84, 197, 104],
  [122, 209, 81], [165, 219, 54], [210, 226, 27], [253, 231, 37],
];
function viridisRGB(t) {
  const x = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x));
  const f = x - i;
  const a = VIRIDIS[i], b = VIRIDIS[i + 1];
  return [
    Math.round(a[0] + f * (b[0] - a[0])),
    Math.round(a[1] + f * (b[1] - a[1])),
    Math.round(a[2] + f * (b[2] - a[2])),
  ];
}
function viridis(t) {
  const [r, g, b] = viridisRGB(t);
  return `rgb(${r},${g},${b})`;
}
// colormap("diverging"): blue below the middle, white at it, red above
// (ColorBrewer RdBu, reversed), for a signed field whose zero should show
const DIVERGING = [
  [5, 48, 97], [33, 102, 172], [67, 147, 195], [146, 197, 222], [209, 229, 240], [247, 247, 247],
  [253, 219, 199], [244, 165, 130], [214, 96, 77], [178, 24, 43], [103, 0, 31],
];
function stopsRGB(stops, t) {
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = stops[i], b = stops[i + 1];
  return [0, 1, 2].map((c) => Math.round(a[c] + f * (b[c] - a[c])));
}

// plot2d: a matrix as a heatmap, canvas-rendered with a colorbar. When the
// chart carries FRAMES (consecutive same-label plot2d calls), it becomes a
// player: rest state is the final frame (the run's result), ▶ replays the
// recorded evolution, the slider scrubs. Color scale is fixed across ALL
// frames so the animation shows the field changing, not the palette.
export function Heatmap({ chart, title, opts = {} }) {
  const canvasRef = useRef(null);
  const rowRef = useRef(null);
  const [boxW, setBoxW] = useState(640);
  useEffect(() => {
    const el = rowRef.current;
    if (!el) return undefined;
    const measure = () => { const w = Math.round(el.clientWidth); if (w > 0) setBoxW((p) => (Math.abs(p - w) > 1 ? w : p)); };
    measure();
    if (typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const { nx, ny, label } = chart;
  // lazy mode: frames live in a quantized binary sidecar, fetched once and
  // shared across every chart on the page that references the same file
  const lazy = chart.lazy || null;
  const [bin, setBin] = useState(null);
  useEffect(() => {
    if (!lazy) return;
    let alive = true;
    fetchFrameBin(lazy.url).then((buf) => { if (alive && buf) setBin(new Uint8Array(buf)); });
    return () => { alive = false; };
  }, [lazy && lazy.url]);
  const frames = !lazy && chart.frames && chart.frames.length ? chart.frames : (!lazy ? [chart.z] : null);
  const nf = lazy ? lazy.nframes : frames.length;
  const [fi, setFi] = useState(nf - 1);
  const [playing, setPlaying] = useState(false);
  let zmin = Infinity, zmax = -Infinity;
  if (lazy) {
    zmin = lazy.fmin;
    zmax = lazy.fmax;
  } else {
    for (const f of frames) {
      for (let i = 0; i < f.length; i++) {
        if (!Number.isFinite(f[i])) continue; // a nan or inf cell is drawn empty
        if (f[i] < zmin) zmin = f[i];
        if (f[i] > zmax) zmax = f[i];
      }
    }
    if (!Number.isFinite(zmin)) { zmin = 0; zmax = 1; }
  }
  // the color limits: the program's clim; else, for a diverging map, symmetric
  // about zero; else the smallest to the largest finite value over all frames
  const diverging = opts.colormap === 'diverging';
  const stops = diverging ? DIVERGING : VIRIDIS;
  let cmin = zmin, cmax = zmax;
  if (opts.clim && opts.clim.length === 2 && opts.clim.every(Number.isFinite) && opts.clim[0] !== opts.clim[1]) [cmin, cmax] = opts.clim;
  else if (diverging) { const m = Math.max(Math.abs(zmin), Math.abs(zmax)) || 1; cmin = -m; cmax = m; }
  const span = cmax - cmin || 1;
  const lower = opts.origin === 'lower'; // row 0 at the bottom
  const z = frames ? frames[Math.min(fi, nf - 1)] : null;
  const ready = lazy ? !!bin : true;
  useEffect(() => {
    if (!ready) return;
    const cv = canvasRef.current;
    if (!cv) return;
    // Render the grid into an offscreen canvas at its native resolution, then
    // upscale to display size. Real fields (32+ cells) get bilinear smoothing
    // at device-pixel resolution — matplotlib-imshow quality, not chunky
    // blocks. Tiny matrices stay pixelated on purpose: those ARE cells.
    const big = Math.max(nx, ny) >= 32;
    // "pixels": the cells ARE the data (categories, counts) and blending two of
    // them invents a third. Otherwise a field of 32+ cells is smoothed.
    const smooth = big && opts.style !== 'pixels';
    // the program's marks on this figure, in data coordinates (lib/axisScale.js heatExtent)
    const marks = [...((chart.fig && chart.fig.marks) || []), ...(chart.marks || [])];
    const placedX = marks.filter((k) => k.kind === 'xtick');
    const placedY = marks.filter((k) => k.kind === 'ytick');
    // a figure() window on a heatmap is its physical EXTENT: draw real axes
    // (and draw them in cell coordinates when the program places ticks without one)
    const axes = !!(opts.xlim && opts.ylim) || placedX.length > 0 || placedY.length > 0;
    const ML = axes ? 46 : 0, MB = axes ? 26 : 0, MT = axes ? 6 : 0, MR = axes ? 8 : 0;
    // fill the box it is in (less the colorbar and the y label), up to 600px
    // (as small as 80 px: a panel of a grid can be narrow)
    const avail = Math.max(80, boxW - 78 - (opts.ylabel ? 22 : 0) - ML - MR);
    let tw = big ? Math.min(avail, 600) : Math.min(avail, ny * Math.max(1, Math.floor(280 / Math.max(nx, ny))));
    let th = Math.round(tw * nx / ny);
    if (th > 600) {
      // tall images (e.g. a 30x60 physical domain) fit a 600px box instead
      th = 600;
      tw = Math.round(th * ny / nx);
    }
    const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
    cv.width = Math.round((tw + ML + MR) * dpr);
    cv.height = Math.round((th + MT + MB) * dpr);
    cv.style.width = (tw + ML + MR) + 'px';
    cv.style.height = (th + MT + MB) + 'px';
    cv.style.imageRendering = smooth ? 'auto' : 'pixelated';
    const off = document.createElement('canvas');
    off.width = ny;
    off.height = nx;
    const octx = off.getContext('2d');
    const img = octx.createImageData(ny, nx);
    const sz = nx * ny;
    const base = lazy ? lazy.offset + Math.min(fi, nf - 1) * sz : 0;
    // a cell's value (a quantized frame is mapped back through its own range), and where it
    // falls between the color limits: 0..1 inside them, beyond them outside
    const value = (t) => (lazy ? zmin + (bin[base + t] / 255) * (zmax - zmin) : z[t]);
    const val01 = (t) => (value(t) - cmin) / span;
    for (let r = 0; r < nx; r++) for (let c = 0; c < ny; c++) {
      const t = r * ny + c;
      const u = val01(t);
      const p = ((lower ? nx - 1 - r : r) * ny + c) * 4; // origin lower: row 0 is the bottom row
      if (!Number.isFinite(u)) { img.data[p + 3] = 0; continue; } // a gap: nothing drawn
      const [r2, g, b] = stopsRGB(stops, u);
      img.data[p] = r2;
      img.data[p + 1] = g;
      img.data[p + 2] = b;
      img.data[p + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.imageSmoothingEnabled = smooth;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(off, ML * dpr, MT * dpr, tw * dpr, th * dpr);
    // contour lines of the displayed frame: at the program's levels(); with the
    // "contour" style and no levels, at 7 evenly spaced levels between the color limits
    const levels01 = opts.levels ? opts.levels.filter(Number.isFinite).map((v) => (v - cmin) / span)
      : opts.style === 'contour' ? [1, 2, 3, 4, 5, 6, 7].map((k) => k / 8) : [];
    if (levels01.length) {
      ctx.strokeStyle = diverging ? 'rgba(40,40,40,0.55)' : 'rgba(255,255,255,0.55)';
      ctx.lineWidth = 1 * dpr;
      const get = (r, c) => val01(r * ny + c);
      const rowY = (r) => (lower ? nx - 1 - r : r);
      ctx.beginPath();
      for (const lv of levels01) {
        for (const [[r1, c1], [r2, c2]] of contourSegs(get, nx, ny, lv)) {
          ctx.moveTo((ML + ((c1 + 0.5) / ny) * tw) * dpr, (MT + ((rowY(r1) + 0.5) / nx) * th) * dpr);
          ctx.lineTo((ML + ((c2 + 0.5) / ny) * tw) * dpr, (MT + ((rowY(r2) + 0.5) / nx) * th) * dpr);
        }
      }
      ctx.stroke();
    }
    const ext = heatExtent(nx, ny, opts);
    const at = (x, y) => { const [fx, fy] = ext.place(x, y); return [(ML + fx * tw) * dpr, (MT + fy * th) * dpr]; };
    // the marks, clipped to the picture: lines dashed, a rect a translucent band,
    // light on viridis and dark on a diverging map, like the contour lines
    const drawn = marks.filter((k) => k.kind !== 'xtick' && k.kind !== 'ytick');
    if (drawn.length) {
      const ink = diverging ? 'rgba(30,30,30,0.9)' : 'rgba(255,255,255,0.92)';
      const halo = diverging ? 'rgba(255,255,255,0.8)' : 'rgba(0,0,0,0.55)';
      ctx.save();
      ctx.beginPath();
      ctx.rect(ML * dpr, MT * dpr, tw * dpr, th * dpr);
      ctx.clip();
      ctx.strokeStyle = ink;
      ctx.fillStyle = ink;
      ctx.lineWidth = 1.5 * dpr;
      for (const k of drawn) {
        if (k.kind === 'rect') {
          const [x0, y0] = at(k.x - k.w / 2, k.y - k.h / 2);
          const [x1, y1] = at(k.x + k.w / 2, k.y + k.h / 2);
          ctx.globalAlpha = 0.22;
          ctx.fillRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
          ctx.globalAlpha = 1;
          continue;
        }
        if (k.kind === 'label') {
          const [px, py] = at(k.x, k.y);
          ctx.font = `600 ${12 * dpr}px "IBM Plex Sans", system-ui, sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.lineWidth = 3 * dpr;
          ctx.strokeStyle = halo;
          ctx.strokeText(k.text, px, py);
          ctx.fillText(k.text, px, py);
          ctx.strokeStyle = ink;
          ctx.lineWidth = 1.5 * dpr;
          continue;
        }
        let a = null, b = null;
        if (k.kind === 'hline') { a = at(ext.xlo, k.y); b = at(ext.xhi, k.y); }
        else if (k.kind === 'vline') { a = at(k.x, ext.ylo); b = at(k.x, ext.yhi); }
        else if (k.kind === 'segment') { a = at(k.x1, k.y1); b = at(k.x2, k.y2); }
        if (!a) continue;
        ctx.setLineDash([5 * dpr, 4 * dpr]);
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
      }
      ctx.restore();
    }
    if (axes) {
      const { xlo, xhi, ylo, yhi } = ext;
      const mut = (typeof getComputedStyle !== 'undefined'
        && getComputedStyle(document.documentElement).getPropertyValue('--text-faint').trim()) || '#8a8f98';
      ctx.strokeStyle = 'rgba(128,128,128,0.55)';
      ctx.lineWidth = 1 * dpr;
      ctx.strokeRect(ML * dpr, MT * dpr, tw * dpr, th * dpr);
      ctx.fillStyle = mut;
      ctx.font = `${11 * dpr}px "IBM Plex Sans", system-ui, sans-serif`;
      const fmtv = (v) => (Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(0) : String(+v.toPrecision(4)));
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      // the program's xtick and ytick marks replace the automatic ticks of their axis
      const xt = placedX.length ? placedX.map((k) => [k.x, k.text]) : heatTicks(xlo, xhi, 5).map((t) => [t, fmtv(t)]);
      const yt = placedY.length ? placedY.map((k) => [k.y, k.text]) : heatTicks(ylo, yhi, 5).map((t) => [t, fmtv(t)]);
      for (const [t, text] of xt) {
        const px = at(t, ylo)[0];
        ctx.beginPath();
        ctx.moveTo(px, (MT + th) * dpr);
        ctx.lineTo(px, (MT + th + 4) * dpr);
        ctx.stroke();
        ctx.fillText(text, px, (MT + th + 7) * dpr);
      }
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (const [t, text] of yt) {
        // with a window ylo is at the bottom of the picture either way; origin only says which row is there
        const py = at(xlo, t)[1];
        ctx.beginPath();
        ctx.moveTo((ML - 4) * dpr, py);
        ctx.lineTo(ML * dpr, py);
        ctx.stroke();
        ctx.fillText(text, (ML - 7) * dpr, py);
      }
    }
  }, [chart, z, bin, fi, nf, ready, nx, ny, zmin, zmax, cmin, span, lower, diverging, opts.levels, opts.style, opts.xlim, opts.ylim, opts.ylabel, opts.origin, boxW]);
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setFi((f) => {
        if (f + 1 >= nf) { setPlaying(false); return f; }
        return f + 1;
      });
    }, 90);
    return () => clearInterval(id);
  }, [playing, nf]);
  const togglePlay = () => {
    if (!playing && fi >= nf - 1) setFi(0);
    setPlaying((p) => !p);
  };
  const fmt = (v) => (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(2) : String(Math.round(v * 1000) / 1000));
  const barTicks = heatTicks(cmin, cmax, 4).filter((t) => t >= Math.min(cmin, cmax) && t <= Math.max(cmin, cmax));
  const barLevels = (opts.levels || []).filter((v) => Number.isFinite(v) && v >= cmin && v <= cmax);
  return (
    <figure className="vc vc-heat">
      {(title || label) && <figcaption className="vc-title"><MathInline text={title || label} size={14} /></figcaption>}
      {!ready && <div className="kv-hmloading">loading {nf} frames…</div>}
      <div className="kv-hmrow" ref={rowRef} style={ready ? undefined : { visibility: 'hidden', height: 0 }}>
        {opts.ylabel && <span className="kv-hmylabel"><MathInline text={opts.ylabel} size={11.5} vertical /></span>}
        <div className="kv-hmmain">
          <canvas ref={canvasRef} className="kv-hmcanvas" />
          {opts.xlabel && <div className="kv-hmxlabel"><MathInline text={opts.xlabel} size={11.5} /></div>}
        </div>
        {/* the gradient and padding are this heatmap's own: in the shared <style>
            below, the last heatmap on a page would set them for every one */}
        <div className="kv-hmbar" aria-hidden="true" style={{ paddingBottom: opts.xlabel ? 24 : 0 }}>
          <div className="kv-hmgrad" style={{ background: `linear-gradient(to top, ${stops.map((c, i) => `rgb(${c[0]},${c[1]},${c[2]}) ${(i / (stops.length - 1)) * 100}%`).join(', ')})` }} />
          <div className="kv-hmticks">
            {barTicks.map((t) => (
              <span key={t} style={{ bottom: `${((t - cmin) / span) * 100}%` }}>{fmt(t)}</span>
            ))}
            {barLevels.map((v) => <i key={'lv' + v} className="kv-hmlevel" style={{ bottom: `${((v - cmin) / span) * 100}%` }} />)}
          </div>
          {title && label && <span className="kv-hmbarlabel"><MathInline text={label} size={11.5} vertical /></span>}
        </div>
      </div>
      {nf > 1 && (
        <div className="kv-anim">
          <button className="kv-animbtn" onClick={togglePlay}
            aria-label={playing ? 'Pause' : 'Play'}>{playing ? '⏸' : '▶'}</button>
          <input type="range" className="kv-animscrub" min={0} max={nf - 1} value={fi}
            onChange={(e) => { setPlaying(false); setFi(+e.target.value); }}
            aria-label="Frame" />
          <span className="kv-animcount">{fi + 1}/{nf}</span>
        </div>
      )}
      <div className="kv-hmmeta">{nx} × {ny} grid{nf > 1 ? ` · ${nf} frames` : ''}</div>
      <style>{`
        .kv-hmrow { display: flex; gap: 10px; align-items: stretch; justify-content: center; }
        .kv-hmmain { min-width: 0; display: flex; flex-direction: column; align-items: center; }
        .kv-hmcanvas { display: block; max-width: 100%; border-radius: 4px; }
        .kv-hmbar { position: relative; display: flex; gap: 6px; align-items: stretch; padding-top: 6px; }
        .kv-hmgrad { width: 12px; border-radius: 3px; }
        .kv-hmlevel { position: absolute; left: -18px; width: 12px; height: 0; border-top: 1.5px solid var(--text); opacity: 0.7; }
        .kv-hmticks { position: relative; width: 34px; }
        .kv-hmticks span { position: absolute; left: 0; transform: translateY(50%); font: 11px var(--font-sans); font-variant-numeric: tabular-nums; color: var(--text-faint); white-space: nowrap; }
        .kv-hmbarlabel { writing-mode: vertical-rl; text-align: center; font: 500 11.5px var(--font-sans); color: var(--text-muted); }
        .kv-hmmeta { font: 11px var(--font-sans); color: var(--text-faint); margin-top: 4px; text-align: center; }
        .kv-hmloading { font: 12px var(--font-mono, monospace); color: var(--text-muted);
          border: 1px dashed var(--border); border-radius: 8px; padding: 2rem; max-width: 560px; text-align: center; }
        .kv-hmylabel { font: 500 11.5px var(--font-sans); color: var(--text-muted);
          writing-mode: vertical-rl; transform: rotate(180deg); text-align: center; align-self: center; }
        .kv-hmxlabel { font: 500 11.5px var(--font-sans); color: var(--text-muted); text-align: center; margin-top: 4px; }
        .kv-anim { display: flex; align-items: center; gap: 0.55rem; margin: 0.45rem auto 0; max-width: 430px; }
        .kv-animbtn { font: 12px var(--font-mono, monospace); width: 28px; height: 24px; line-height: 1;
          border: 1px solid var(--border); border-radius: 6px; background: transparent;
          color: var(--accent); cursor: pointer; }
        .kv-animbtn:hover { background: color-mix(in srgb, var(--accent) 12%, transparent); }
        .kv-animscrub { flex: 1; accent-color: var(--accent); }
        .kv-animcount { font: 11px var(--font-mono, monospace); color: var(--text-muted); min-width: 3.2em; text-align: right; }
      `}</style>
    </figure>
  );
}

// plot3d: a curve in 3-space — drag to rotate
export function Curve3D({ chart, title }) {
  const [view, setView] = useState({ yaw: -0.65, pitch: 0.42 });
  const drag = useRef(null);
  const { xs, ys, zs, label } = chart;
  const W = 420, Hh = 320;
  // equal aspect: one scale for all three axes, so geometry is not distorted —
  // a flat curve LOOKS flat, a globule looks like a globule
  const rng = (a) => { const f = a.filter(Number.isFinite); return f.length ? { lo: Math.min(...f), hi: Math.max(...f) } : { lo: 0, hi: 1 }; };
  const RX = rng(xs), RY = rng(ys), RZ = rng(zs);
  const half = Math.max(RX.hi - RX.lo, RY.hi - RY.lo, RZ.hi - RZ.lo, 1e-12) / 2;
  const norm = (a, r) => ({ m: a.map((v) => (v - (r.lo + r.hi) / 2) / half), lo: r.lo, hi: r.hi });
  const X = norm(xs, RX), Y = norm(ys, RY), Z = norm(zs, RZ);
  const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw);
  const cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
  const proj = (x, y, z) => {
    // yaw about the vertical, then pitch; y is "up"
    const x1 = x * cy + z * sy;
    const z1 = -x * sy + z * cy;
    const y2 = y * cp - z1 * sp;
    const z2 = y * sp + z1 * cp;
    return { px: W / 2 + x1 * (W * 0.32), py: Hh / 2 - y2 * (Hh * 0.36), d: z2 };
  };
  // axes box corners and edges
  const C = [-1, 1];
  const edges = [];
  for (const a of C) for (const b of C) edges.push([[-1, a, b], [1, a, b]], [[a, -1, b], [a, 1, b]], [[a, b, -1], [a, b, 1]]);
  const pts = xs.map((_, i) => proj(X.m[i], Z.m[i], Y.m[i])); // z is "up" on screen: map data z->screen y
  let path = '';
  let pen = false;
  pts.forEach((p) => {
    if (!Number.isFinite(p.px) || !Number.isFinite(p.py)) { pen = false; return; }
    path += `${pen ? 'L' : 'M'}${p.px.toFixed(1)},${p.py.toFixed(1)} `;
    pen = true;
  });
  const onDown = (e) => {
    e.preventDefault();
    drag.current = { x: e.clientX, y: e.clientY, yaw: view.yaw, pitch: view.pitch };
    const move = (ev) => {
      const d = drag.current;
      if (!d) return;
      setView({
        yaw: d.yaw + (ev.clientX - d.x) * 0.01,
        pitch: Math.max(-1.4, Math.min(1.4, d.pitch + (ev.clientY - d.y) * 0.01)),
      });
    };
    const up = () => {
      drag.current = null;
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };
  const fmt = (v) => (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(1) : String(Math.round(v * 100) / 100));
  return (
    <figure className="kv-fig">
      {(title || label) && <figcaption className="kv-figtitle"><MathInline text={title || label} size={13} /></figcaption>}
      <svg viewBox={`0 0 ${W} ${Hh}`} className="kv-c3svg" onPointerDown={onDown} role="img" aria-label={label || '3D curve'}>
        {edges.map(([a, b], i) => {
          const p = proj(a[0], a[2], a[1]);
          const q = proj(b[0], b[2], b[1]);
          return <line key={i} x1={p.px} y1={p.py} x2={q.px} y2={q.py} className="kv-c3edge" />;
        })}
        <path d={path} className="kv-c3line" fill="none" />
        {pts.length <= 120 && pts.map((p, i) => Number.isFinite(p.px) && Number.isFinite(p.py) && <circle key={i} cx={p.px} cy={p.py} r={2.1} className="kv-c3pt" opacity={0.45 + 0.55 * ((p.d + 1.6) / 3.2)} />)}
      </svg>
      <div className="kv-hmmeta">
        drag to rotate · x [{fmt(X.lo)}, {fmt(X.hi)}] · y [{fmt(Y.lo)}, {fmt(Y.hi)}] · z [{fmt(Z.lo)}, {fmt(Z.hi)}]
      </div>
      <style>{`
        .kv-c3svg { width: 100%; max-width: 480px; cursor: grab; touch-action: none; }
        .kv-c3svg:active { cursor: grabbing; }
        .kv-c3edge { stroke: var(--border); stroke-width: 1; }
        .kv-c3line { stroke: var(--accent); stroke-width: 1.8; }
        .kv-c3pt { fill: var(--accent); }
      `}</style>
    </figure>
  );
}

// scatter: unordered points (spectra in the complex plane, samples). Each
// series carries its own x — nothing is shared or sorted. `opts` carries the
// publication framing: hard axis windows (points outside are clipped, exactly
// like matplotlib set_xlim), axis labels, and per-series open/filled markers —
// so a reproduced figure can wear the paper's own frame.
// Drawn at the width it is shown, up to SCATTER_W, as LineChart is: a viewBox
// scaled down to a phone's card shrank its 10.5px tick labels to 8px.
const SCATTER_W = 430;
const SCATTER_MIN_W = 260;
// A title too long for the chart's width (a phone, a feed thumbnail) goes on
// two lines, broken at the space nearest its middle, so neither end is cut off.
// The width is estimated at 0.56 em a character for 13 px sans.
const TITLE_PX = 13;
const TITLE_EM_PER_CHAR = 0.56;
const TITLE_LINE = 15;
export function titleLines(title, width) {
  if (!title || hasMath(title) || title.length * TITLE_PX * TITLE_EM_PER_CHAR <= width - 2 * TITLE_PX) return title ? [title] : [];
  const mid = title.length / 2;
  let cut = -1;
  for (let i = 0; i < title.length; i++) if (title[i] === ' ' && (cut < 0 || Math.abs(i - mid) < Math.abs(cut - mid))) cut = i;
  return cut < 0 ? [title] : [title.slice(0, cut), title.slice(cut + 1)];
}
export function ScatterPlot({ chart: given, title, opts = {} }) {
  const wrapRef = useRef(null);
  const [shown, setShown] = useState(SCATTER_W);
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return undefined;
    const measure = () => {
      const w = Math.round(el.clientWidth);
      if (w > 0) setShown((prev) => (Math.abs(prev - w) > 1 ? w : prev));
    };
    measure();
    if (typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const W = Math.max(SCATTER_MIN_W, Math.min(SCATTER_W, shown));
  let H = Math.round(350 * (W / SCATTER_W));
  // the clip's id is unique on the page: two cells can each hold a chart keyed sc-0
  const clipId = `vc-clip-${useId().replace(/:/g, '')}`;
  // in a grid (lib/figureGrid.js): which tick labels a neighbor panel already shows
  const inner = (given.fig && given.fig.inner) || {};
  // yright("label", "axis"): those series read against a linear y axis on the right
  const yright = given.fig && given.fig.yright;
  const rightSet = new Set((yright && yright.series) || []);
  const hasRight = given.series.some((s) => rightSet.has(s.name));
  const tLines = titleLines(title, W);
  const tExtra = Math.max(0, tLines.length - 1) * TITLE_LINE;
  H += tExtra;
  const P = { l: inner.y ? 14 : 64, r: hasRight || (given.fig && given.fig.roomRight) ? 64 : 14, t: title ? 30 + tExtra : 16, b: inner.x ? 12 : opts.xlabel ? 46 : 36 };
  // Everything below is in t-space (lib/axisScale.js): the value on a linear
  // axis, log10 of it on a log axis (xscale/yscale), where a value <= 0 is a gap.
  const xLog = opts.xscale === 'log';
  const yLog = opts.yscale === 'log';
  const tx = scaleOf(opts.xscale);
  const ty = scaleOf(opts.yscale);
  const chart = { ...given, series: given.series.map((s) => {
    const r = rightSet.has(s.name);
    const f = r ? (v) => v : ty;
    // fill_between and rangebars carry the two ends of each interval
    const ends = s.interval ? { lo: s.lo.map(f), hi: s.hi.map(f) } : {};
    return { ...s, ...ends, right: r, xs: s.xs.map(tx), ys: s.ys.map(f) };
  }) };
  const hideX = xLog ? given.series.reduce((n, s) => n + s.xs.filter((v) => Number.isFinite(v) && v <= 0).length, 0) : 0;
  const hideY = yLog ? given.series.reduce((n, s) => n + (rightSet.has(s.name) ? 0 : (s.interval ? [...s.lo, ...s.hi] : s.ys).filter((v) => Number.isFinite(v) && v <= 0).length), 0) : 0;
  const hidden = [hideX && `${hideX} x value${hideX > 1 ? 's' : ''} ≤ 0`, hideY && `${hideY} y value${hideY > 1 ? 's' : ''} ≤ 0`].filter(Boolean);
  const winT = (l, f) => (l && Number.isFinite(f(l[0])) && Number.isFinite(f(l[1])) ? [f(l[0]), f(l[1])] : null);
  const xlimT = winT(opts.xlim, tx);
  const ylimT = winT(opts.ylim, ty);
  // the program's marks on this figure (animParse.js parseMark), and their extent
  const marks = [...((given.fig && given.fig.marks) || []), ...(given.marks || [])]
    .map((k) => markToT(k, tx, ty))
    .filter((k) => Object.values(k).every((v) => typeof v !== 'number' || Number.isFinite(v)));
  const markX = marks.flatMap((k) => (k.kind === 'vline' || k.kind === 'xtick' ? [k.x] : k.kind === 'label' ? [k.x] : k.kind === 'rect' ? [k.x - k.w / 2, k.x + k.w / 2] : k.kind === 'segment' ? [k.x1, k.x2] : []));
  const markY = marks.flatMap((k) => (k.kind === 'hline' || k.kind === 'ytick' ? [k.y] : k.kind === 'label' ? [k.y] : k.kind === 'rect' ? [k.y - k.h / 2, k.y + k.h / 2] : k.kind === 'segment' ? [k.y1, k.y2] : []));
  // the extent of what is drawn: a nan or inf is a gap, not a coordinate
  const allx = [...chart.series.flatMap((s) => s.xs), ...markX].filter(Number.isFinite);
  const yvals = (s) => (s.interval ? [...s.lo, ...s.hi] : s.ys);
  const ally = [...chart.series.filter((s) => !s.right).flatMap(yvals), ...markY].filter(Number.isFinite);
  const ally2 = chart.series.filter((s) => s.right).flatMap(yvals).filter(Number.isFinite);
  if (!allx.length) allx.push(0, 1);
  if (!ally.length) ally.push(0, 1);
  let xlo, xhi, lo, hi;
  if (xlimT) { [xlo, xhi] = xlimT; }
  else {
    xlo = allx.reduce((a, b) => Math.min(a, b)); xhi = allx.reduce((a, b) => Math.max(a, b));
    const xpad = (xhi - xlo || Math.abs(xhi) || 1) * 0.06;
    xlo -= xpad; xhi += xpad;
  }
  if (ylimT) { [lo, hi] = ylimT; }
  else {
    lo = ally.reduce((a, b) => Math.min(a, b)); hi = ally.reduce((a, b) => Math.max(a, b));
    const ypad = (hi - lo || Math.abs(hi) || 1) * 0.06;
    lo -= ypad; hi += ypad;
  }
  // aspect("equal"): a unit as long across as up (lib/axisScale.js equalAspect)
  if (opts.aspect === 'equal') {
    const e = equalAspect(xlo, xhi, lo, hi, W - P.l - P.r, 160, 440);
    ({ xlo, xhi } = e);
    [lo, hi] = [e.ylo, e.yhi];
    H = Math.round(P.t + P.b + e.plotH);
  }
  let lo2 = 0;
  let hi2 = 1;
  if (ally2.length) {
    lo2 = Math.min(...ally2); hi2 = Math.max(...ally2);
    const pad2 = (hi2 - lo2 || Math.abs(hi2) || 1) * 0.06;
    lo2 -= pad2; hi2 += pad2;
  }
  // to 1/1000 px: the build and the browser may differ in a float's last digit
  // (a log10, a pow), and a hydrated chart must print the same attributes
  const X = (x) => px3(P.l + ((x - xlo) / (xhi - xlo || 1)) * (W - P.l - P.r));
  const Y = (v) => px3(P.t + (1 - (v - lo) / (hi - lo || 1)) * (H - P.t - P.b));
  const Y2 = (v) => P.t + (1 - (v - lo2) / (hi2 - lo2 || 1)) * (H - P.t - P.b);
  const inWin = (x, y) => x >= xlo && x <= xhi && y >= lo && y <= hi; // false for nan
  const inWin2 = (x, y) => x >= xlo && x <= xhi && y >= lo2 && y <= hi2;
  const niceTicks = (a, b, n) => {
    const span = b - a || 1;
    const step0 = span / Math.max(1, n);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const norm = step0 / mag;
    // round to the NEAREST of 1-2-5-10, matplotlib-style, not the next above
    const step = (norm > 7 ? 10 : norm > 3 ? 5 : norm > 1.4 ? 2 : 1) * mag;
    const t0 = Math.ceil(a / step) * step;
    const out = [];
    for (let t = t0; t <= b + step * 1e-9; t += step) out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
    return out;
  };
  const fmt = (v) => {
    if (v === 0) return '0';
    const a = Math.abs(v);
    if (a >= 1e5 || a < 1e-3) return v.toExponential(1).replace('e+', 'e');
    return String(+v.toPrecision(4));
  };
  // ticks as { t, label }: a log axis gets decades (lib/axisScale.js logTicks)
  const autoTicks = (log, a, b, n) => (log ? logTicks(a, b, n).ticks : niceTicks(a, b, n).map((t) => ({ t, label: fmt(t) })));
  const placedY = marks.filter((k) => k.kind === 'ytick' && Number.isFinite(k.y));
  const yticks = placedY.length ? placedY.map((k) => ({ t: k.y, label: k.text })) : autoTicks(yLog, lo, hi, 4);
  const placed = marks.filter((k) => k.kind === 'xtick' && Number.isFinite(k.x));
  const xticks = placed.length ? placed.map((k) => ({ t: k.x, label: k.text })) : autoTicks(xLog, xlo, xhi, 5);
  // tick labels that would not fit, laid out as LineChart does (lib/axisScale.js):
  // the left margin as wide as the program's y tick labels, a row per y tick,
  // and x tick labels turned (categories) or thinned (numbers)
  const tickW = (s) => (hasMath(s) ? chartLabel(s, 10.5).w : String(s).length * 6.3);
  if (!inner.y && placedY.length) P.l = Math.max(P.l, 12 + Math.max(...yticks.map((k) => tickW(k.label))));
  if (opts.aspect !== 'equal' && placedY.length > 1) {
    const ys = placedY.map((k) => k.y).sort((a, b) => a - b);
    let gap = Infinity;
    for (let i = 1; i < ys.length; i++) { const d = ys[i] - ys[i - 1]; if (d > 0 && d < gap) gap = d; }
    const ph = H - P.t - P.b;
    if (Number.isFinite(gap)) H += Math.max(0, Math.min(Math.round((YTICK_ROW * Math.abs(hi - lo)) / gap), MAX_PLOT_H) - ph);
  }
  const xtW = xticks.map((k) => tickW(k.label));
  const xLay = xTickLayout(xticks.map((k) => X(k.t)), xtW, placed.length > 0);
  if (xLay.turn && !inner.x) {
    const over = xtW.reduce((m, w, i) => (xLay.show[i] ? Math.max(m, turnedReach(w) - (X(xticks[i].t) - 4)) : m), 0);
    if (over > 0) P.l += over;
    const drop = turnedDrop(xtW.filter((_, i) => xLay.show[i]));
    P.b += drop;
    H += drop;
  }
  // The legend is HTML above the plot, not a box inside it. A spectrum puts
  // points wherever it likes, including the top right corner, and a legend
  // drawn over them hides data -- which is the same reason LineChart keeps
  // its own legend outside.
  // one legend entry per label, drawn like its first series
  // (drawn like the label's first series that is not an interval, if it has one)
  const firstOf = (name) => { const j = chart.series.findIndex((t) => t.name === name && !t.interval); return j >= 0 ? j : chart.series.findIndex((t) => t.name === name); };
  const named = chart.series.map((s, k) => ({ s, k })).filter(({ s, k }) => s.name && firstOf(s.name) === k);
  // Each series' drawing (lib/seriesStyle.jsx): the program's style() for its
  // label; else a scatter color name ("blue", "blue o" for rings); else its
  // palette slot. Points are small dots and a plot() here is a solid line,
  // unless the style says otherwise.
  const styles = (given.fig && given.fig.styles) || {};
  const slots = paletteSlots(chart.series, styles);
  const draws = chart.series.map((s, k) => {
    const st = styles[s.name] || {};
    const open = s.open ?? (opts.open ? !!opts.open[k] : false);
    const d = drawStyle({ ...st, line: st.line || (s.line ? 'solid' : 'noline') }, slots[k], false);
    if (!st.color && SCATTER_NAMED[s.color]) d.color = SCATTER_NAMED[s.color];
    if (!st.marker) d.marker = s.line ? null : open ? 'ring' : 'dot';
    return d;
  });
  return (
    <div className="vc-scatter" ref={wrapRef}>
    <svg viewBox={`0 0 ${W} ${H}`} className="kcell-plot" role="img"
      aria-label={title || `scatter of ${chart.series.map((s) => s.name).join(', ')}`}>
      <style>{`
        .kp-ax { stroke: var(--border-strong, var(--border)); stroke-width: 1; }
        .kp-tick, .kp-legend { font: 10.5px var(--font-mono); fill: var(--text-muted); }
        .kp-legend { font-weight: 700; }
        .kp-title { font: 600 13px var(--font-sans); fill: var(--text); }
        .kp-grid { stroke: var(--border); opacity: 0.55; }
        .kp-axlabel { font: italic 600 12px var(--font-sans); fill: var(--text-muted); }
        .kp-legendbox { fill: var(--bg-card); stroke: var(--border); }
      `}</style>
      {tLines.map((t, i) => <SvgText key={i} text={t} size={TITLE_PX} color="var(--text)" x={W / 2} y={16 + i * TITLE_LINE} className="kp-title" textAnchor="middle" />)}
      <clipPath id={clipId}><rect x={P.l} y={P.t} width={W - P.l - P.r} height={H - P.t - P.b} /></clipPath>
      {yticks.map(({ t, label }) => (
        <g key={'y' + t}>
          <line x1={P.l} y1={Y(t)} x2={W - P.r} y2={Y(t)} className="kp-grid" />
          {!inner.y && <SvgText text={label} size={10.5} x={P.l - 7} y={Y(t) + 3.5} className="kp-tick" textAnchor="end" />}
        </g>
      ))}
      {xticks.map(({ t, label }, i) => (
        <g key={'x' + t}>
          <line x1={X(t)} y1={H - P.b} x2={X(t)} y2={H - P.b + 4} className="kp-ax" />
          {!inner.x && xLay.show[i] && (xLay.turn
            ? <SvgText text={label} size={10.5} x={X(t) + TURN_DX} y={H - P.b + TURN_DY} className="kp-tick" textAnchor="end" rotate={-TICK_TURN_DEG} />
            : <SvgText text={label} size={10.5} x={X(t)} y={H - P.b + 16} className="kp-tick" textAnchor="middle" />)}
        </g>
      ))}
      {marks.map((k, i) => (k.kind === 'rect' ? (
        <rect key={`mb${i}`} className="vc-mark-band" x={X(k.x - k.w / 2)} y={Y(k.y + k.h / 2)} width={Math.abs(X(k.x + k.w / 2) - X(k.x - k.w / 2))} height={Math.abs(Y(k.y - k.h / 2) - Y(k.y + k.h / 2))} />
      ) : null))}
      {marks.map((k, i) => {
        if (k.kind === 'hline') return <line key={`m${i}`} className="vc-mark" x1={P.l} x2={W - P.r} y1={Y(k.y)} y2={Y(k.y)} />;
        if (k.kind === 'vline') return <line key={`m${i}`} className="vc-mark" x1={X(k.x)} x2={X(k.x)} y1={P.t} y2={H - P.b} />;
        if (k.kind === 'segment') return <line key={`m${i}`} className="vc-mark" x1={X(k.x1)} y1={Y(k.y1)} x2={X(k.x2)} y2={Y(k.y2)} />;
        if (k.kind === 'label' && hasMath(k.text)) {
          const l = chartLabel(k.text, 12);
          return <SvgText key={`m${i}`} className="vc-mark-label" text={k.text} size={12} x={X(k.x)} y={Y(k.y) + (l.h - l.d) / 2} textAnchor="middle" />;
        }
        if (k.kind === 'label') return <text key={`m${i}`} className="vc-mark-label" x={X(k.x)} y={Y(k.y)} textAnchor="middle" dominantBaseline="middle">{k.text}</text>;
        return null;
      })}
      <line x1={P.l} y1={P.t} x2={P.l} y2={H - P.b} className="kp-ax" />
      <line x1={P.l} y1={H - P.b} x2={W - P.r} y2={H - P.b} className="kp-ax" />
      {hasRight && autoTicks(false, lo2, hi2, 4).map(({ t, label }) => (
        <g key={'y2' + t}>
          <line x1={W - P.r} y1={Y2(t)} x2={W - P.r + 4} y2={Y2(t)} className="kp-ax" />
          <text x={W - P.r + 7} y={Y2(t) + 3.5} className="kp-tick" textAnchor="start">{label}</text>
        </g>
      ))}
      {hasRight && <line x1={W - P.r} y1={P.t} x2={W - P.r} y2={H - P.b} className="kp-ax" />}
      {hasRight && yright.label && (
        <SvgText text={yright.label} size={12} x={W - 10} y={P.t + (H - P.t - P.b) / 2} className="kp-axlabel" textAnchor="middle" rotate={90} />
      )}
      {opts.xlabel && !inner.x && (
        <SvgText text={opts.xlabel} size={12} x={P.l + (W - P.l - P.r) / 2} y={H - 6} className="kp-axlabel" textAnchor="middle" />
      )}
      {opts.ylabel && !inner.y && (
        <SvgText text={opts.ylabel} size={12} x={14} y={P.t + (H - P.t - P.b) / 2} className="kp-axlabel" textAnchor="middle" rotate={-90} />
      )}
      {chart.series.map((s, k) => {
        const d = draws[k];
        // a yright series reads against the right axis
        const Yk = s.right ? Y2 : Y;
        const inK = s.right ? inWin2 : inWin;
        const fin2 = (i) => Number.isFinite(s.xs[i]) && Number.isFinite(s.lo[i]) && Number.isFinite(s.hi[i]);
        if (s.interval === 'fill') {
          // the band from lo to hi; a gap splits it
          const runs = [];
          let run = [];
          s.xs.forEach((_, i) => { if (fin2(i)) run.push(i); else { if (run.length) runs.push(run); run = []; } });
          if (run.length) runs.push(run);
          const pt = (i, v) => `${X(s.xs[i]).toFixed(1)} ${Math.max(-1e4, Math.min(1e4, Yk(v))).toFixed(1)}`;
          const dd = runs.map((r) => `M${r.map((i) => pt(i, s.hi[i])).join('L')}L${r.slice().reverse().map((i) => pt(i, s.lo[i])).join('L')}Z`).join('');
          return <path key={k} d={dd} fill={d.color} fillOpacity="0.2" stroke="none" clipPath={`url(#${clipId})`} />;
        }
        if (s.interval === 'err') {
          // a capped bar from lo to hi at each x
          let dd = '';
          s.xs.forEach((_, i) => {
            if (!fin2(i)) return;
            const x = X(s.xs[i]);
            const [a, b] = [Yk(s.lo[i]), Yk(s.hi[i])].map((v) => Math.max(-1e4, Math.min(1e4, v)).toFixed(1));
            dd += `M${x.toFixed(1)} ${a}V${b}M${(x - 3.5).toFixed(1)} ${a}H${(x + 3.5).toFixed(1)}M${(x - 3.5).toFixed(1)} ${b}H${(x + 3.5).toFixed(1)}`;
          });
          return <path key={k} d={dd} stroke={d.color} strokeWidth="1.3" fill="none" clipPath={`url(#${clipId})`} />;
        }
        const path = () => {
          // a non-finite point lifts the pen: the line has a gap there
          let p = '';
          let pen = false;
          s.xs.forEach((x, i) => {
            const y = s.ys[i];
            if (!Number.isFinite(x) || !Number.isFinite(y)) { pen = false; return; }
            p += `${pen ? 'L' : 'M'}${X(x).toFixed(1)} ${Math.max(-1e4, Math.min(1e4, Yk(y))).toFixed(1)}`;
            pen = true;
          });
          return p;
        };
        return (
          <g key={k}>
            {d.line && <path d={path()} className={d.primary ? 'vc-primary' : undefined} fill="none" stroke={d.color} strokeWidth={d.width} strokeDasharray={d.dash} strokeLinejoin="round" clipPath={`url(#${clipId})`} />}
            {d.marker && s.xs.map((x, i) => inK(x, s.ys[i]) && (d.marker === 'dot'
              ? <circle key={i} cx={X(x).toFixed(1)} cy={Yk(s.ys[i]).toFixed(1)} r={1.5} fill={d.color} />
              : <Marker key={i} kind={d.marker} x={X(x)} y={Yk(s.ys[i])} color={d.color} r={d.marker === 'ring' ? 2.1 : 2.2} />
            ))}
          </g>
        );
      })}
    </svg>
    {named.length > 0 && (
      <div className="vc-legend vc-legend-dots">
        {named.map(({ s, k }) => (
          <span key={s.name + '-l'}>
            <Swatch d={draws[k].marker === 'dot' ? { ...draws[k], marker: 'circle' } : draws[k]} interval={s.interval || null} />
            <MathInline text={s.name} size={12} />{s.right && <em className="vc-right-note"> (right axis)</em>}
          </span>
        ))}
      </div>
    )}
    {/* points a log axis cannot draw (y <= 0) are left out quietly: a note about it read as machinery */}
    </div>
  );
}

// quiver: a vector field as arrows in a framed, tick-labeled plane. The
// figure() window is the physical extent; aspect is preserved when given.
export function QuiverPlot({ chart, title, opts = {} }) {
  const keep = chart.xs.map((_, i) => [chart.xs[i], chart.ys[i], chart.us[i], chart.vs[i]].every(Number.isFinite));
  const [xs, ys, us, vs] = [chart.xs, chart.ys, chart.us, chart.vs].map((a) => a.filter((_, i) => keep[i]));
  const { label } = chart;
  let xlo, xhi, ylo, yhi;
  if (opts.xlim) [xlo, xhi] = opts.xlim;
  else { xlo = Math.min(...xs); xhi = Math.max(...xs); const p = (xhi - xlo || 1) * 0.06; xlo -= p; xhi += p; }
  if (opts.ylim) [ylo, yhi] = opts.ylim;
  else { ylo = Math.min(...ys); yhi = Math.max(...ys); const p = (yhi - ylo || 1) * 0.06; ylo -= p; yhi += p; }
  const P = { l: 52, r: 10, t: title ? 30 : 12, b: 40 };
  const xspan = xhi - xlo || 1, yspan = yhi - ylo || 1;
  let pw = 360, ph = Math.round(pw * yspan / xspan);
  if (ph > 520) { ph = 520; pw = Math.round(ph * xspan / yspan); }
  const W = pw + P.l + P.r, H = ph + P.t + P.b;
  const X = (x) => px3(P.l + ((x - xlo) / xspan) * pw);
  const Y = (y) => px3(P.t + ((yhi - y) / yspan) * ph);
  // arrow scale: the largest arrow spans ~1.6 typical point spacings
  let vmax = 1e-30;
  for (let i = 0; i < us.length; i++) vmax = Math.max(vmax, Math.hypot(us[i], vs[i]));
  const spacing = Math.sqrt((xspan * yspan) / Math.max(1, xs.length));
  const s = (1.6 * spacing) / vmax;
  const arrows = xs.map((x, i) => {
    const x1 = X(x), y1 = Y(ys[i]);
    const x2 = X(x + us[i] * s), y2 = Y(ys[i] + vs[i] * s);
    const ang = Math.atan2(y2 - y1, x2 - x1);
    const hl = 3.2;
    return { x1, y1, x2, y2,
      hx1: x2 - hl * Math.cos(ang - 0.45), hy1: y2 - hl * Math.sin(ang - 0.45),
      hx2: x2 - hl * Math.cos(ang + 0.45), hy2: y2 - hl * Math.sin(ang + 0.45) };
  });
  const fmtv = (v) => (Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(0) : String(+v.toPrecision(4)));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="kcell-plot" style={{ maxWidth: W }} role="img" aria-label={title || label}>
      <style>{`
        .kq-ax { stroke: var(--border-strong, var(--border)); stroke-width: 1; fill: none; }
        .kq-tick { font: 10.5px var(--font-mono); fill: var(--text-muted); }
        .kq-title { font: 600 13px var(--font-sans); fill: var(--text); }
        .kq-axlabel { font: italic 600 12px var(--font-sans); fill: var(--text-muted); }
        .kq-arrow { stroke: var(--accent); stroke-width: 1.1; fill: none; stroke-linecap: round; }
      `}</style>
      {title && <text x={W / 2} y={16} className="kq-title" textAnchor="middle">{title}</text>}
      <rect x={P.l} y={P.t} width={pw} height={ph} className="kq-ax" />
      {heatTicks(xlo, xhi, 5).map((t) => (
        <g key={'x' + t}>
          <line x1={X(t)} y1={P.t + ph} x2={X(t)} y2={P.t + ph + 4} className="kq-ax" />
          <text x={X(t)} y={P.t + ph + 15} className="kq-tick" textAnchor="middle">{fmtv(t)}</text>
        </g>
      ))}
      {heatTicks(ylo, yhi, 5).map((t) => (
        <g key={'y' + t}>
          <line x1={P.l - 4} y1={Y(t)} x2={P.l} y2={Y(t)} className="kq-ax" />
          <text x={P.l - 6} y={Y(t) + 3.5} className="kq-tick" textAnchor="end">{fmtv(t)}</text>
        </g>
      ))}
      {opts.xlabel && <text x={P.l + pw / 2} y={H - 6} className="kq-axlabel" textAnchor="middle">{opts.xlabel}</text>}
      {opts.ylabel && (
        <text x={12} y={P.t + ph / 2} className="kq-axlabel" textAnchor="middle"
          transform={`rotate(-90 12 ${P.t + ph / 2})`}>{opts.ylabel}</text>
      )}
      {arrows.map((a, i) => (
        <path key={i} className="kq-arrow"
          d={`M${a.x1.toFixed(1)} ${a.y1.toFixed(1)} L${a.x2.toFixed(1)} ${a.y2.toFixed(1)} M${a.hx1.toFixed(1)} ${a.hy1.toFixed(1)} L${a.x2.toFixed(1)} ${a.y2.toFixed(1)} L${a.hx2.toFixed(1)} ${a.hy2.toFixed(1)}`} />
      ))}
    </svg>
  );
}

// ---- interactive 3D scene viewer ---------------------
// Renders a scene chart: grouped wireframes and point markers from the Viv
// scene protocol. Drag = orbit (yaw about the scene's z axis, pitch about the
// screen x axis), wheel = zoom, double-click = reset, slider = cutaway (clips
// geometry nearest the viewer, exposing a cross-section of anything). Groups
// toggle from the legend; per-vertex scalars colormap along a shared scale.
const SCENE_CMAP = [
  [68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37],
];
function sceneColorFor(t) {
  const x = Math.max(0, Math.min(1, t)) * (SCENE_CMAP.length - 1);
  const i = Math.min(SCENE_CMAP.length - 2, Math.floor(x));
  const f = x - i;
  const c = SCENE_CMAP[i].map((a, k) => Math.round(a + f * (SCENE_CMAP[i + 1][k] - a)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
// pure projection helper (exported for tests): yaw about z, pitch about x
export function sceneProject(p, cx, cy, cz, yaw, pitch) {
  const x = p[0] - cx, y = p[1] - cy, z = p[2] - cz;
  const cy1 = Math.cos(yaw), sy1 = Math.sin(yaw);
  const x1 = x * cy1 - y * sy1, y1 = x * sy1 + y * cy1;
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const y2 = y1 * cp - z * sp, z2 = y1 * sp + z * cp;
  return [x1, y2, z2]; // screen-x, depth (into screen), screen-up
}
export function SceneView({ chart, title }) {
  const canvasRef = useRef(null);
  const [yaw, setYaw] = useState(0.6);
  const [pitch, setPitch] = useState(-1.05);
  const [zoom, setZoom] = useState(1.0);
  const [cut, setCut] = useState(0);
  const [hidden, setHidden] = useState(() => new Set());
  const drag = useRef(null);

  // flatten geometry once: segments + points + bbox + scalar range
  const geo = useMemo(() => {
    const segs = [];
    const pts = [];
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    let vmin = Infinity, vmax = -Infinity;
    chart.groups.forEach((g, gi) => {
      const see = (x, y, z) => {
        if (x < lo[0]) lo[0] = x; if (x > hi[0]) hi[0] = x;
        if (y < lo[1]) lo[1] = y; if (y > hi[1]) hi[1] = y;
        if (z < lo[2]) lo[2] = z; if (z > hi[2]) hi[2] = z;
      };
      for (const w of g.wires) {
        for (let i = 0; i < w.xs.length; i++) see(w.xs[i], w.ys[i], w.zs[i]);
        if (w.vs) for (const v of w.vs) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
        for (let i = 0; i + 1 < w.xs.length; i++) {
          segs.push({
            a: [w.xs[i], w.ys[i], w.zs[i]], b: [w.xs[i + 1], w.ys[i + 1], w.zs[i + 1]],
            gi, v: w.vs ? 0.5 * (w.vs[i] + w.vs[i + 1]) : null,
          });
        }
      }
      for (const p of g.verts) {
        for (let i = 0; i < p.xs.length; i++) {
          see(p.xs[i], p.ys[i], p.zs[i]);
          if (p.vs) { const v = p.vs[i]; if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
          pts.push({ a: [p.xs[i], p.ys[i], p.zs[i]], gi, v: p.vs ? p.vs[i] : null });
        }
      }
    });
    const c = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])];
    const r = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-12) * 0.5;
    return { segs, pts, c, r, vmin, vmax, hasVals: vmax > vmin || Number.isFinite(vmin) };
  }, [chart]);

  const W = 680, H = 460;
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
    canvas.width = W * dpr; canvas.height = H * dpr;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const css = getComputedStyle(canvas);
    const mut = css.getPropertyValue('--text-muted').trim() || '#888';
    const { segs, pts, c, r, vmin, vmax } = geo;
    const s = (Math.min(W, H) * 0.42 * zoom) / r;
    const P = (p) => sceneProject(p, c[0], c[1], c[2], yaw, pitch);
    // depth range this frame, for the cutaway threshold
    let dlo = Infinity, dhi = -Infinity;
    const items = [];
    for (const sg of segs) {
      if (hidden.has(chart.groups[sg.gi].name)) continue;
      const A = P(sg.a), B = P(sg.b);
      const d = 0.5 * (A[1] + B[1]);
      if (d < dlo) dlo = d; if (d > dhi) dhi = d;
      items.push({ t: 0, A, B, d, gi: sg.gi, v: sg.v });
    }
    for (const pt of pts) {
      if (hidden.has(chart.groups[pt.gi].name)) continue;
      const A = P(pt.a);
      if (A[1] < dlo) dlo = A[1]; if (A[1] > dhi) dhi = A[1];
      items.push({ t: 1, A, d: A[1], gi: pt.gi, v: pt.v });
    }
    const thresh = dlo + cut * (dhi - dlo) * 0.98;
    items.sort((a, b) => b.d - a.d); // far first (painter's)
    const X = (p) => W / 2 + s * p[0];
    const Y = (p) => H / 2 - s * p[2];
    for (const it of items) {
      if (it.d < thresh) continue; // cutaway: drop the near side
      const depthT = (it.d - dlo) / (dhi - dlo || 1);
      const alpha = 0.45 + 0.55 * (1 - depthT);
      const col = it.v != null && vmax > vmin
        ? sceneColorFor((it.v - vmin) / (vmax - vmin))
        : SERIES_COLORS[it.gi % SERIES_COLORS.length];
      ctx.globalAlpha = alpha;
      if (it.t === 0) {
        ctx.strokeStyle = col;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.moveTo(X(it.A), Y(it.A));
        ctx.lineTo(X(it.B), Y(it.B));
        ctx.stroke();
      } else {
        ctx.fillStyle = col;
        ctx.beginPath();
        ctx.arc(X(it.A), Y(it.A), 3, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = mut;
    ctx.font = '11px monospace';
    ctx.fillText('drag: rotate · wheel: zoom · double-click: reset', 10, H - 10);
  }, [geo, yaw, pitch, zoom, cut, hidden, chart]);

  // wheel zoom needs a non-passive listener to preventDefault
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (e) => {
      e.preventDefault();
      setZoom((z) => Math.max(0.2, Math.min(8, z * Math.exp(-e.deltaY * 0.0012))));
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, []);

  const onDown = (e) => {
    drag.current = { x: e.clientX, y: e.clientY, yaw, pitch };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onMove = (e) => {
    if (!drag.current) return;
    setYaw(drag.current.yaw + (e.clientX - drag.current.x) * 0.008);
    setPitch(Math.max(-Math.PI, Math.min(0, drag.current.pitch + (e.clientY - drag.current.y) * 0.008)));
  };
  const onUp = () => { drag.current = null; };
  const reset = () => { setYaw(0.6); setPitch(-1.05); setZoom(1); setCut(0); };
  const anyVals = Number.isFinite(geo.vmin) && geo.vmax > geo.vmin;

  return (
    <div className="kcell-scene" style={{ maxWidth: W, width: '100%' }}>
      {(chart.title || title) && (
        <div style={{ font: '600 13px var(--font-sans)', textAlign: 'center', marginBottom: 4 }}>
          {chart.title || title}
        </div>
      )}
      <canvas ref={canvasRef}
        style={{ width: '100%', aspectRatio: `${W} / ${H}`, cursor: 'grab', touchAction: 'none',
          border: '1px solid var(--border)', borderRadius: 6 }}
        onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onDoubleClick={reset} />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 14px', alignItems: 'center',
        font: '11.5px var(--font-mono)', color: 'var(--text-muted)', marginTop: 6 }}>
        {chart.groups.map((g, gi) => (
          <button key={g.name} type="button"
            onClick={() => setHidden((h) => {
              const n = new Set(h);
              if (n.has(g.name)) n.delete(g.name); else n.add(g.name);
              return n;
            })}
            style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0,
              font: 'inherit', color: 'inherit', opacity: hidden.has(g.name) ? 0.35 : 1,
              display: 'inline-flex', alignItems: 'center', gap: 4 }}>
            <span style={{ width: 10, height: 10, borderRadius: 2, display: 'inline-block',
              background: SERIES_COLORS[gi % SERIES_COLORS.length] }} />
            {g.name}
          </button>
        ))}
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6, marginLeft: 'auto' }}>
          cutaway
          <input type="range" min="0" max="1" step="0.01" value={cut}
            onChange={(e) => setCut(Number(e.target.value))} style={{ width: 110 }} />
        </label>
      </div>
      {anyVals && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4,
          font: '11px var(--font-mono)', color: 'var(--text-muted)' }}>
          <span>{geo.vmin.toPrecision(3)}</span>
          <span style={{ flex: '0 1 160px', height: 8, borderRadius: 4,
            background: `linear-gradient(90deg, ${SCENE_CMAP.map((c, i) =>
              `rgb(${c[0]},${c[1]},${c[2]}) ${(100 * i) / (SCENE_CMAP.length - 1)}%`).join(', ')})` }} />
          <span>{geo.vmax.toPrecision(3)}</span>
        </div>
      )}
    </div>
  );
}

// A grid of figures (grid("title", rows, cols)): its panels in rows and
// columns, each drawn by Chart like any figure, with the shared ranges and the
// labels a neighbor shows worked out by lib/figureGrid.js. When the page is too
// narrow for the columns (under about 200 px a panel, 300 for a heatmap with
// axes), the panels stack in one
// column, in the program's order, each keeping its own labels.
export function GridChart({ chart }) {
  const ref = useRef(null);
  const [width, setWidth] = useState(720);
  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const measure = () => { const w = Math.round(el.clientWidth); if (w > 0) setWidth((p) => (Math.abs(p - w) > 1 ? w : p)); };
    measure();
    if (typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  // a heatmap with axes needs room for its ticks and its colorbar beside the picture
  const least = chart.panels.some((p) => p.some((c) => c.kind === 'heatmap' && c.fig && c.fig.xlim && c.fig.ylim)) ? 300 : 200;
  const stacked = chart.cols > 1 && width < chart.cols * least;
  const panels = useMemo(() => layoutPanels(chart, stacked), [chart, stacked]);
  const cols = stacked ? 1 : chart.cols;
  return (
    <figure className="vc vc-gridfig" ref={ref}>
      {chart.title && <figcaption className="vc-title vc-grid-title">{chart.title}</figcaption>}
      <div className="vc-panels" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
        {panels.map((p, k) => (
          <div className="vc-panel" key={k}>{p.map((c, i) => <Chart key={c.key || i} chart={c} />)}</div>
        ))}
      </div>
    </figure>
  );
}

export function Chart({ chart }) {
  // Title, axis labels, limits and style come from the program's own figure()
  // record and from nowhere else: this component takes no presentation props,
  // so a page cannot decorate a chart the program did not describe.
  const fig = (chart && chart.fig) || {};
  const title = fig.title || undefined;
  const opts = { xlabel: fig.xlabel || undefined, ylabel: fig.ylabel || undefined, xlim: fig.xlim, ylim: fig.ylim, style: fig.style, xscale: fig.xscale, yscale: fig.yscale, aspect: fig.aspect, origin: fig.origin, colormap: fig.colormap, clim: fig.clim, levels: fig.levels };
  if (chart && chart.kind === 'diagram') return <DiagramView chart={chart} />;
  if (chart && chart.kind === 'grid') return <GridChart chart={chart} />;
  if (chart && chart.kind === 'anim') return <AnimPlayer chart={chart} />;
  if (chart && chart.kind === 'heatmap') return <Heatmap chart={chart} title={title} opts={opts || {}} />;
  if (chart && chart.kind === 'scene') return <SceneView chart={chart} title={title} />;
  if (chart && chart.kind === 'curve3d') return <Curve3D chart={chart} title={title} />;
  if (chart && chart.kind === 'scatter') return <ScatterPlot chart={chart} title={title} opts={opts || {}} />;
  if (chart && chart.kind === 'quiver') return <QuiverPlot chart={chart} title={title} opts={opts || {}} />;
  return <LineChart chart={chart} />;
}

// The cell editor: the highlighted source in a <pre>, under a textarea with
// transparent text that owns the caret and the editing. It grows to fit its
// code. Shared by the tour's cells (VivCell) and the notebook view.
//   code, onChange(text), onKeyDown(event), label   as for a textarea
//   taRef                                           the textarea, for focus
export function CellEditor({ code, onChange, onKeyDown, label, taRef = null, readOnly = false }) {
  const ownRef = useRef(null);
  const ta = taRef || ownRef;
  const preRef = useRef(null);
  const html = useMemo(() => highlight(code) + '\n', [code]);
  useEffect(() => {
    const el = ta.current;
    if (el) { el.style.height = 'auto'; el.style.height = el.scrollHeight + 2 + 'px'; }
  }, [code, ta]);
  const syncScroll = useCallback(() => {
    const el = ta.current, pre = preRef.current;
    if (el && pre) { pre.scrollLeft = el.scrollLeft; pre.scrollTop = el.scrollTop; }
  }, [ta]);
  return (
    <div className="kcell-editor">
      <pre ref={preRef} className="kcell-hl" aria-hidden="true"
           dangerouslySetInnerHTML={{ __html: html }} />
      <textarea
        ref={ta}
        className="kcell-src"
        value={code}
        spellCheck="false"
        readOnly={readOnly}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={onKeyDown}
        onScroll={syncScroll}
        aria-label={label}
      />
      <style>{`
        /* Editor: highlighted <pre> behind a transparent-text <textarea>; the
           textarea owns editing + caret, the pre owns the colors. Metrics on
           the two MUST match exactly or the caret drifts. */
        .kcell-editor { position: relative; }
        .kcell-hl, .kcell-src {
          margin: 0; padding: 10px 12px; font: 12.5px/1.55 var(--font-mono);
          white-space: pre; overflow-x: auto; tab-size: 4;
        }
        .kcell-hl { pointer-events: none; color: var(--text); min-height: 100%; }
        .kcell-src {
          position: absolute; inset: 0; width: 100%; height: 100%;
          display: block; border: none; outline: none; resize: none;
          background: transparent; color: transparent; caret-color: var(--text);
        }
        /* keep selected text transparent too, or the browser paints the
           textarea's text on top of the highlighted <pre> (double vision). */
        .kcell-src::selection { background: color-mix(in srgb, var(--accent) 30%, transparent); color: transparent; }
        .kcell-src::-moz-selection { background: color-mix(in srgb, var(--accent) 30%, transparent); color: transparent; }

      `}</style>
    </div>
  );
}

// standalone: the cell runs ONLY its own source (no notebook prefix, no
// registry) — the mode doc-example cells use, where every snippet is
// independent and doctest-enforced.
/** Output as the doctests compare it: the graph/trace/steps/story records
 *  dropped, the ends trimmed. */
export const comparableOutput = (out) => String(out || '').split('\n')
  .filter((l) => !/^__(graph|trace|steps|story)__/.test(l))
  .map((l) => l.replace(/\s+$/, ''))
  .join('\n').trim();

/** The run's output beside the expected output, after Run: a mark saying
 *  whether they are identical, and the first line where they differ marked in
 *  both columns. */
export function ExpectedCompare({ actual, expected, expectedLabel = 'expected' }) {
  const a = comparableOutput(actual).split('\n');
  const e = comparableOutput(expected).split('\n');
  let first = -1;
  for (let i = 0; i < Math.max(a.length, e.length); i++) {
    if (a[i] !== e[i]) { first = i; break; }
  }
  const col = (lines, title) => (
    <div className="kcmp-col">
      <div className="kcmp-title">{title}</div>
      <pre className="kcmp-pre">{lines.map((l, i) => (
        <span key={i} className={i === first ? 'kcmp-line kcmp-diff' : 'kcmp-line'}>{l || ' '}{'\n'}</span>
      ))}{first >= lines.length && <span className="kcmp-line kcmp-diff kcmp-missing">(no line {first + 1}){'\n'}</span>}</pre>
    </div>
  );
  return (
    <div className={`kcmp ${first < 0 ? 'kcmp-same' : 'kcmp-differs'}`}>
      <div className="kcmp-mark" role="status">
        {first < 0
          ? <><b>✓ identical</b> to the {expectedLabel} output ({a.length} line{a.length === 1 ? '' : 's'})</>
          : <><b>✗ differs</b> from the {expectedLabel} output, first at line {first + 1}</>}
      </div>
      <div className="kcmp-cols">
        {col(a, 'this run')}
        {col(e, expectedLabel)}
      </div>
      <style>{`
        .kcmp { border-top: 1px dashed var(--border); }
        .kcmp-mark { padding: 6px 12px; font: 12px var(--font-sans); color: var(--text-muted); }
        .kcmp-same .kcmp-mark b { color: var(--ok, #15803d); }
        .kcmp-differs .kcmp-mark b { color: var(--bad, #b91c1c); }
        .kcmp-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 0; }
        .kcmp-col + .kcmp-col { border-left: 1px dashed var(--border); }
        .kcmp-title { padding: 3px 12px; font: 600 11px var(--font-mono); letter-spacing: 0.05em; text-transform: uppercase; color: var(--text-muted); }
        .kcmp-pre { margin: 0; padding: 4px 12px 9px; font: 12px/1.5 var(--font-mono); color: var(--text); white-space: pre-wrap; overflow-wrap: anywhere; max-height: 340px; overflow-y: auto; background: color-mix(in srgb, var(--text) 3%, var(--bg)); }
        .kcmp-diff { background: color-mix(in srgb, #dc2626 18%, transparent); display: inline-block; width: 100%; }
        .kcmp-missing { color: var(--text-muted); font-style: italic; }
        @media (max-width: 640px) { .kcmp-cols { grid-template-columns: 1fr; } .kcmp-col + .kcmp-col { border-left: none; border-top: 1px dashed var(--border); } }
      `}</style>
    </div>
  );
}

// expected: the output the cell should print; after Run it shows beside the
//           run's output with a mark (identical / differs at line N).
// recorded: { output, note } -- a run recorded elsewhere, shown under the code
//           before any run here, labeled with `note`; a run here replaces it.
// runnable: false for a program the tab cannot hold (no Run button).
// tag:      the label in the cell's head (default "Viv [n]").
export default function VivCell({ cached, cachedOutput = null, cachedEngine = '', cachedSeconds = 0, idx, src, graphOpen = false, liveSwap = null, standalone = false, viz = null, expected = null, recorded = null, runnable = true, tag = null }) {
  const [code, setCode] = useState(src);
  const [blocks, setBlocks] = useState(null); // ordered text/narrate output
  const [rawOut, setRawOut] = useState(''); // own output text, for a `viz`
  const [charts, setCharts] = useState([]);
  const [graph, setGraph] = useState(null);
  const [trace, setTrace] = useState([]);
  const [state, setState] = useState('idle'); // idle | running | done | error
  const [ms, setMs] = useState(0);
  const [ownText, setOwnText] = useState(''); // the run's whole output, for `expected`

  // Register (and keep registered) this cell's current source.
  useEffect(() => { if (!standalone) registry.set(idx, code); }, [idx, code, standalone]);

  const VizComp = viz ? VIZ[viz] : null;

  // compiled-twin state, declared before run() so its deps can reference it:
  // the offline precomputed result is held SEPARATELY from any live run, and
  // `view` picks which one the cell displays. When the cached output arrives
  // as a PROP (server-side read), it is parsed synchronously so the charts
  // render statically — no fetch, no flash.
  const [cacheMeta, setCacheMeta] = useState(() =>
    (cachedOutput != null ? { engine: cachedEngine, seconds: cachedSeconds } : null));
  const [cachedRes, setCachedRes] = useState(() => {
    if (cachedOutput == null) return null;
    const { blocks: bl, charts: ch, graph: g, trace: tr } = parseOwnOutput(cachedOutput);
    return { blocks: bl.length ? bl : [{ type: 'text', s: '(no output)' }], charts: ch, graph: g, trace: tr };
  });
  const [view, setView] = useState('offline'); // 'offline' | 'live'

  const run = useCallback(async () => {
    // In the precomputed view with the untouched production code, "running"
    // the cell replays the offline result: the tab cannot hold the production
    // size, and the cache IS this exact program's output. Edited cells (and
    // the browser-sized view) run live as usual.
    if (cachedRes && view === 'offline' && code === src) {
      setState('running');
      await new Promise((r) => setTimeout(r, 200));
      setState('idle');
      return;
    }
    setState('running');
    try {
      const ex = await loadWasm();
      const prefix = standalone ? '' : [...registry.entries()]
        .filter(([i]) => i < idx)
        .sort(([a], [b]) => a - b)
        .map(([, s]) => s)
        .join('\n');
      const cellSrc = await withModules(standalone ? code : (registry.get(idx) ?? code));
      const t0 = performance.now();
      let own;
      if (ex.viv_run_cell) {
        own = callWasmCell(ex, prefix, cellSrc);
      } else {
        const program = prefix ? `${prefix}\n${SENTINEL_STMT}\n${cellSrc}` : cellSrc;
        const raw = joinKept(callWasm(ex, ex.viv_run_keep || ex.viv_run, program));
        const cut = raw.lastIndexOf(SENTINEL);
        own = cut >= 0 ? raw.slice(cut + SENTINEL.length).replace(/^\n/, '') : raw;
      }
      setMs(performance.now() - t0);
      // a diagnostic is a LINE that starts with `error`; a program that shows a
      // variable called `shape_error` has not failed
      const isErr = /^error\b/m.test(own);
      const { blocks: bl, charts: ch, graph: g, trace: tr } = parseOwnOutput(own);
      setBlocks(bl.length ? bl : [{ type: 'text', s: '(no output)' }]);
      setRawOut(isErr ? '' : own);
      setOwnText(own);
      setCharts(isErr ? [] : ch);
      setGraph(isErr ? null : g);
      setTrace(isErr ? [] : tr);
      setState(isErr ? 'error' : 'done');
      setView('live'); // you just ran it — show your run; the switch flips back
    } catch (e) {
      setBlocks([{ type: 'text', s: String(e) }]);
      setRawOut('');
      setOwnText(String(e));
      setCharts([]);
      setGraph(null);
      setTrace([]);
      setState('error');
      setView('live');
    }
  }, [idx, code, cachedRes, view, src]);

  const onKey = useCallback((e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); run(); }
  }, [run]);

  // register with the page-level Run-all bar
  useEffect(() => {
    if (!standalone) runners.set(idx, run);
    return () => { runners.delete(idx); };
  }, [idx, run, standalone]);

  // compiled-twin cache fetch: the offline precomputed result (C backend,
  // full problem size), parsed once through the same output pipeline.
  useEffect(() => {
    if (!cached) return;
    let alive = true;
    fetch(cached)
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => {
        if (!alive || !j || !j.cells || !j.cells[idx]) return;
        const { blocks: bl, charts: ch, graph: g, trace: tr } = parseOwnOutput(j.cells[idx].output);
        setCachedRes({
          blocks: bl.length ? bl : [{ type: 'text', s: '(no output)' }],
          charts: ch, graph: g, trace: tr,
        });
        setCacheMeta({ engine: j.engine, seconds: j.seconds });
      })
      .catch(() => {});
    return () => { alive = false; };
  }, [cached, idx]);

  // Flipping the view also flips the CODE between its production knob and the
  // browser-sized one (the notebook's declared liveSwap), so the source shown
  // always matches the results shown. A cell the user has edited away from
  // either canonical form is left alone.
  const applyView = useCallback((v) => {
    setView(v);
    if (liveSwap) {
      const swaps = Array.isArray(liveSwap) ? liveSwap : [liveSwap];
      setCode((c) => {
        let out = c;
        for (const sw of swaps) {
          if (v === 'live' && out.includes(sw.from)) out = out.replace(sw.from, sw.to);
          if (v === 'offline' && out.includes(sw.to)) out = out.replace(sw.to, sw.from);
        }
        return out;
      });
    }
  }, [liveSwap]);

  // the toolbar's global offline/live switch drives every cell at once
  useEffect(() => {
    if (!cached) return;
    viewSetters.set(idx, applyView);
    return () => { viewSetters.delete(idx); };
  }, [cached, idx, applyView]);

  const hasLive = blocks !== null;
  const showOffline = cachedRes && view === 'offline';
  const disp = showOffline
    ? cachedRes
    : hasLive
      ? { blocks, charts, graph, trace }
      : cachedRes
        ? { blocks: [{ type: 'text', s: '(not run yet: press ▶ Run, or switch back to ⚡ precomputed)' }], charts: [], graph: null, trace: [] }
        : recorded && recorded.output != null
          ? (() => { const r = parseOwnOutput(recorded.output); return { blocks: r.blocks.length ? r.blocks : [{ type: 'text', s: '(no output)' }], charts: r.charts, graph: null, trace: [] }; })()
          : { blocks, charts, graph, trace };
  const showingRecorded = !hasLive && !showOffline && recorded && recorded.output != null;
  const compare = expected != null && hasLive && !showOffline;

  return (
    <div className={`kcell kcell-${state}`}>
      <div className="kcell-head">
        <span className="kcell-tag">{tag || `Viv [${idx + 1}]`}</span>
        {runnable && (
          <button className="kcell-run" onClick={run} disabled={state === 'running'}>
            {state === 'running' ? 'running…' : hasLive ? '▶ Run again' : '▶ Run'}
          </button>
        )}
        {showingRecorded && recorded.note && <span className="kcell-ms kcell-rectag">{recorded.note}</span>}
        {state === 'done' && !showOffline && <span className="kcell-ms">{ms.toFixed(0)} ms{standalone ? '' : ` · ran cells 1–${idx + 1}`}</span>}
        {state === 'error' && !showOffline && <span className="kcell-ms kcell-errtag">error</span>}
        {cachedRes && (
          <span className="kcell-viewsw" role="group" aria-label="Which result to show">
            <button className={showOffline ? 'kcell-viewon' : ''} title={cacheMeta ? cacheMeta.engine : ''}
              onClick={() => applyView('offline')}>⚡ precomputed</button>
            <button className={!showOffline ? 'kcell-viewon' : ''}
              title="Switches the problem size to the smaller value the browser can run"
              onClick={() => applyView('live')}>▶ run in browser</button>
          </span>
        )}
        {showOffline && cacheMeta && (
          <span className="kcell-ms kcell-cachetag" title={cacheMeta.engine}>
            {cacheMeta.engine.split(' ')[0]} · precomputed in {cacheMeta.seconds} s
          </span>
        )}
      </div>
      <CellEditor code={code} onChange={setCode} onKeyDown={onKey} label={`Viv code cell ${idx + 1}`} />
      {compare && <ExpectedCompare actual={ownText} expected={expected} expectedLabel={recorded && expected === recorded.output ? 'recorded' : 'expected'} />}
      {!compare && disp.blocks !== null && disp.blocks !== undefined && (
        <div className={`kcell-outwrap${showingRecorded ? ' kcell-recorded' : ''}`}>
          {disp.blocks.map((b, i) =>
            b.type === 'narrate'
              ? <div key={i} className="kcell-narrate">▸ {b.s}</div>
              : <pre key={i} className="kcell-out">{b.s}</pre>
          )}
        </div>
      )}
      {(disp.charts || []).map((c, i) => <Chart key={(showOffline ? 'o' : 'l') + i} chart={c} />)}
      {VizComp && (rawOut || showOffline) && <VizComp text={rawOut || cachedOutput || ''} />}
      {disp.graph && disp.graph.edges.length > 0 && <CallGraph graph={disp.graph} trace={disp.trace} defaultOpen={graphOpen} />}
      <style>{`
        .kcell { margin: 1.1rem 0 1.6rem; border: 1px solid var(--border); border-radius: 10px; overflow: hidden; background: var(--bg-card); }
        .kcell-error { border-color: #b91c1c66; }
        .kcell-head { display: flex; align-items: center; gap: 0.7rem; padding: 6px 10px; border-bottom: 1px solid var(--border); background: color-mix(in srgb, var(--accent) 4%, var(--bg-card)); }
        .kcell-tag { font: 600 11px var(--font-mono); color: var(--text-muted); letter-spacing: 0.03em; }
        .kcell-run { font: 600 12px var(--font-sans); padding: 3px 12px; border-radius: 6px; border: 1px solid var(--accent); background: var(--accent); color: var(--on-accent, #fff); cursor: pointer; }
        .kcell-run:disabled { opacity: 0.6; cursor: default; }
        .kcell-ms { font: 11px var(--font-mono); color: var(--text-muted); }
        .kcell-errtag { color: #b91c1c; font-weight: 700; }
        .kcell-cachetag { color: var(--accent); font-weight: 700; }
        .kcell-rectag { font-style: italic; }

        /* offline/live result switch: which of the two retained results shows */
        .kcell-viewsw { display: inline-flex; border: 1px solid var(--border); border-radius: 7px; overflow: hidden; }
        .kcell-viewsw button { font: 600 11px var(--font-mono); padding: 3px 10px; border: none;
          background: transparent; color: var(--text-muted); cursor: pointer; }
        .kcell-viewsw button + button { border-left: 1px solid var(--border); }
        .kcell-viewsw .kcell-viewon { background: color-mix(in srgb, var(--accent) 14%, transparent);
          color: var(--accent); }

        /* narration: the program announcing its own flow */
        .kcell-narrate { font: italic 600 12px var(--font-sans); color: var(--accent); padding: 4px 12px; border-top: 1px dashed var(--border); }
        .kcell-narrate + .kcell-narrate { border-top: none; padding-top: 0; }

        /* the live call graph */

        .kcell-out { margin: 0; padding: 9px 12px; border-top: 1px dashed var(--border); background: color-mix(in srgb, var(--text) 3%, var(--bg)); font: 12px/1.5 var(--font-mono); color: var(--text); white-space: pre-wrap; max-height: 340px; overflow-y: auto; }
        .kcell-plot { display: block; width: min(540px, 100%); margin: 8px auto 10px; }
      `}</style>
    </div>
  );
}
