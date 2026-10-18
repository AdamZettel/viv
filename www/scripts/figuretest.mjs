// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// How program output becomes charts (VivCell.jsx parseOwnOutput): figure() is
// the only grouping, and nothing on a chart is invented.
//
//   node scripts/figuretest.mjs            the checks below, and the corpus
//   node scripts/figuretest.mjs --write    (re)write the corpus's expected parses
//
// The corpus is lang/viv/tests/figures/: record streams (NAME.out) and the
// chart each must parse to (NAME.json), in the canonical form.
// The Jupyter parser is held to the same files
// (tools/jupyter/tests/test_figures_corpus.py), and the engines must print the
// probes' records byte for byte (lang/viv/tests/figures.rs).
import { build } from 'esbuild';
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const bundled = await build({ entryPoints: [join(here, '../src/components/VivCell.jsx')], bundle: true, format: 'esm', platform: 'node', write: false, jsx: 'automatic', loader: { '.js': 'jsx', '.css': 'empty' }, logLevel: 'silent' });
const { parseOwnOutput } = await import('data:text/javascript;base64,' + Buffer.from(bundled.outputFiles[0].text).toString('base64'));

let fails = 0;
let checks = 0;
const ok = (cond, what) => { checks++; if (!cond) { fails++; console.log(`FAIL ${what}`); } };
const charts = (lines) => parseOwnOutput(lines.join('\n')).charts;

// no figure open: every record is its own chart, whatever its x
{
  const c = charts(['plot: a ; [0, 1, 2] ; [1, 2, 3]', 'plot: b ; [0, 1, 2] ; [3, 2, 1]', 'scatter: p ;  ; [0, 1] ; [1, 1]', 'scatter: q ;  ; [0, 1] ; [2, 2]']);
  ok(c.length === 4, `no figure: four records are four charts (got ${c.length})`);
}
// one figure: every plot, scatter and bars joins its chart, until the next figure
{
  const c = charts(['figure: t ; x ; y', 'plot: curve ; [0, 1, 2] ; [1, 2, 3]', 'show done', 'scatter: point ;  ; [1] ; [2]', 'plot: other x ; [0, 0.5] ; [0, 1]', 'figure: u ; x ; y', 'plot: next ; [0, 1] ; [1, 0]']);
  ok(c.length === 2, `two figures are two charts (got ${c.length})`);
  ok(c[0].series.length === 3 && c[0].fig.title === 't', 'the first figure holds its plot, its scatter and its second plot');
  ok(c[1].fig.title === 'u' && c[1].series.length === 1, 'the second figure starts a new chart even on the same x');
}
// plots that share an x inside a figure stay one x-based chart; bars join on the same x
{
  const c = charts(['figure: h ; x ; count', 'bars: counts ; [0, 1, 2] ; [3, 5, 2]', 'plot: fit ; [0, 1, 2] ; [3.2, 4.8, 2.1]']);
  ok(c.length === 1 && c[0].series.length === 2 && c[0].series[0].bars, 'a histogram and a fitted curve in one figure are one chart');
}
// marks attach to the open figure, before or after its plots, and never make a chart
{
  const c = charts(['figure: bands ; k ; E', 'xtick: L ; 0', 'plot: band ; [0, 1] ; [0, 1]', 'hline: zero ; 0', 'label: z ; valence band top ; 0.5 0', 'rect: gap ; 0.5 0.25 1 0.5']);
  ok(c.length === 1, `marks do not make charts (got ${c.length})`);
  const marks = [...(c[0].fig.marks || []), ...(c[0].marks || [])];
  ok(marks.map((m) => m.kind).join(',') === 'xtick,hline,label,rect', `the marks, in order (got ${marks.map((m) => m.kind).join(',')})`);
  ok(marks.find((m) => m.kind === 'label').text === 'valence band top', "a label's text is the program's");
}
// animation records with no figure open stay an animation
{
  const c = charts(['stage: s ; 0 1 0 1', 'hline: h ; 0.5', 'frame: 1']);
  ok(c.length === 1 && c[0].kind === 'anim', 'hline inside a stage is part of the animation');
}
// plot2d: frames of one movie only inside one figure
{
  const z = '[1, 2, 3, 4]';
  const bare = charts([`plot2d: f ; 2 2 ; ${z}`, `plot2d: f ; 2 2 ; ${z}`]);
  ok(bare.length === 2, `no figure: two plot2d calls are two heatmaps (got ${bare.length})`);
  const movie = charts(['figure: field ; x ; y', `plot2d: f ; 2 2 ; ${z}`, 'show t', `plot2d: f ; 2 2 ; ${z}`]);
  ok(movie.length === 1 && movie[0].frames.length === 2, 'in one figure, same-label plot2d calls are frames');
}
// scatter markers: filled unless the program asks for open circles
{
  const c = charts(['figure: s ; x ; y', 'scatter: a ;  ; [0] ; [0]', 'scatter: b ;  ; [1] ; [1]', 'scatter: c ; blue o ; [2] ; [2]']);
  ok(c[0].series.map((s) => s.open).join(',') === 'false,false,true', 'only an "o" marker is open');
}

// log axes (lib/axisScale.js): decades labeled 10^k, nonpositive values are gaps
{
  const { logTicks, scaleOf } = await import('../src/lib/axisScale.js');
  const t = logTicks(-12, 0, 7);
  ok(t.ticks.map((k) => k.label).join(' ') === '10⁻¹² 10⁻¹⁰ 10⁻⁸ 10⁻⁶ 10⁻⁴ 10⁻² 10⁰', `twelve decades, every second one labeled (got ${t.ticks.map((k) => k.label).join(' ')})`);
  ok(t.minor.length === 0, 'no minor ticks when decades are skipped');
  const d = logTicks(-3, -1, 5);
  ok(d.ticks.length === 3 && d.minor.length === 16, `three decades labeled, 2..9 minor between (got ${d.ticks.length}, ${d.minor.length})`);
  const narrow = logTicks(Math.log10(2), Math.log10(8), 5);
  ok(narrow.ticks.map((k) => k.label).join(',') === '2,3,4,5,6,7,8', `under a decade: plain numbers (got ${narrow.ticks.map((k) => k.label).join(',')})`);
  const one = logTicks(Math.log10(0.0125), Math.log10(0.4), 5);
  ok(one.ticks.map((k) => k.label).join(',') === '0.02,0.05,0.1,0.2', `a decade and a half: 1, 2, 5 (got ${one.ticks.map((k) => k.label).join(',')})`);
  const f = scaleOf('log');
  ok(Number.isNaN(f(0)) && Number.isNaN(f(-1)) && f(100) === 2, 'zero and negatives are gaps on a log axis');
  const c = charts(['figure: g ; t ; N', 'plot: n ; [0, 1] ; [1, 10]', 'yscale: log', 'figure: h ; t ; N', 'plot: m ; [0, 1] ; [1, 10]']);
  ok(c[0].fig.yscale === 'log' && !c[1].fig.yscale, 'a scale after the plots sets its own figure, not the next');
}

// series style (lib/seriesStyle.jsx): palette order by first appearance; a slot a style claims is skipped
{
  const b = await build({ entryPoints: [join(here, '../src/lib/seriesStyle.jsx')], bundle: true, format: 'esm', platform: 'node', write: false, jsx: 'automatic', logLevel: 'silent' });
  const { paletteSlots, drawStyle } = await import('data:text/javascript;base64,' + Buffer.from(b.outputFiles[0].text).toString('base64'));
  const series = [{ name: 'exact' }, { name: 'computed' }, { name: 'exact' }, { name: '' }];
  ok(paletteSlots(series).join(',') === '0,1,0,2', `labels share a slot (got ${paletteSlots(series).join(',')})`);
  ok(paletteSlots(series, { exact: { color: 's2' } }).join(',') === '-1,0,-1,2', `a claimed slot is skipped (got ${paletteSlots(series, { exact: { color: 's2' } }).join(',')})`);
  const d = drawStyle({ line: 'dash', color: 's2', marker: 'ring', width: 'thick' }, 0, false);
  ok(d.color === 'var(--s2)' && d.dash && d.width === 2.8 && d.marker === 'ring' && d.line, 'every word of a style is drawn');
  const plain = drawStyle(undefined, 7, true);
  ok(plain.color === 'var(--s2)' && !plain.dash && plain.width === 1.8 && plain.marker === 'circle', 'no style: palette, solid, 1.8 px, dots when sparse');
}

// aspect("equal") (lib/axisScale.js equalAspect): a unit as long across as up
{
  const { equalAspect } = await import('../src/lib/axisScale.js');
  const round = equalAspect(-1, 1, -1, 1, 400, 160, 520);
  ok(round.plotH === 400 && round.xlo === -1 && round.yhi === 1, 'a unit circle in a 400 px plot is 400 px tall, ranges untouched');
  const tall = equalAspect(0, 1, 0, 4, 400, 160, 520);
  ok(tall.plotH === 520 && Math.abs((tall.xhi - tall.xlo) / 400 - 4 / 520) < 1e-12 && Math.abs(tall.xlo + tall.xhi - 1) < 1e-12, 'too tall: held at the limit, x widened about its center');
  const flat = equalAspect(0, 10, 0, 1, 400, 160, 520);
  ok(flat.plotH === 160 && Math.abs((flat.yhi - flat.ylo) - 4) < 1e-12, 'too flat: held at the least height, y widened');
  const c = charts(['figure: o ; x ; y', 'aspect: equal', 'plot: a ; [0, 1] ; [1, 0]']);
  ok(c[0].fig.aspect === 'equal', 'aspect is a setting of its figure');
}

// marks on a heatmap (lib/axisScale.js heatExtent): in the window's units, or in cells
{
  const { heatExtent } = await import('../src/lib/axisScale.js');
  const near = (a, b) => Math.abs(a[0] - b[0]) < 1e-12 && Math.abs(a[1] - b[1]) < 1e-12;
  const win = heatExtent(2, 3, { xlim: [0, 3], ylim: [0, 2], origin: 'lower' });
  ok(near(win.place(0, 0), [0, 1]) && near(win.place(3, 2), [1, 0]), 'with a window: (xlo, ylo) is the bottom left, (xhi, yhi) the top right');
  ok(near(heatExtent(2, 3, { xlim: [0, 3], ylim: [0, 2] }).place(0, 0), [0, 1]), 'with a window, origin does not move a mark: ylo stays at the bottom');
  const upper = heatExtent(2, 3, {});
  ok(near(upper.place(0, 0), [1 / 6, 1 / 4]) && near(upper.place(2, 1), [5 / 6, 3 / 4]), 'no window: the center of cell (row 0, column 0) is at the top left');
  const lower = heatExtent(2, 3, { origin: 'lower' });
  ok(near(lower.place(0, 0), [1 / 6, 3 / 4]) && near(lower.place(0, 1), [1 / 6, 1 / 4]), 'no window, origin("lower"): row 0 is at the bottom');
  const c = charts(['figure: f ; x ; y ; 0 3 0 2', 'hline: before ; 1', 'plot2d: u ; 2 3 ; [0, 1, 2, 3, 4, 5]', 'segment: after ; 0 0 3 2']);
  ok(c.length === 1 && c[0].kind === 'heatmap', 'marks on a heatmap make no chart of their own');
  const hm = [...(c[0].fig.marks || []), ...(c[0].marks || [])];
  ok(hm.map((m) => m.kind).join(',') === 'hline,segment', `a heatmap keeps the marks before and after its plot2d (got ${hm.map((m) => m.kind).join(',')})`);
}

// grid("t", rows, cols): the next rows * cols figures are its panels; share and
// yright (lib/figureGrid.js layoutPanels)
{
  const c = charts(['grid: g ; 2 1', 'share: x', 'figure: a ;  ; y', 'plot: p ; [0, 1, 2] ; [1, 2, 3]', 'hline: h ; 2',
    'figure: b ; t ; z', 'plot: q ; [0, 5] ; [4, 1]', 'plot: e ; [0, 5] ; [40, 10]', 'yright: e ; joules', 'figure: c ; x ; y', 'plot: r ; [0, 1] ; [0, 1]']);
  ok(c.length === 2 && c[0].kind === 'grid' && c[0].panels.length === 2, `a 2 x 1 grid holds two panels, and the third figure stands alone (got ${c.length} charts)`);
  ok(c[0].panels[0][0].marks.length === 1 && c[0].share === 'x', 'a mark goes to its panel; share is the grid\'s');
  ok(c[0].panels[1][0].fig.yright.series.join() === 'e' && c[0].panels[1][0].fig.yright.label === 'joules', 'yright names the series and the right axis');
  const { layoutPanels, sharedRange } = await import('../src/lib/figureGrid.js');
  const laid = layoutPanels(c[0]);
  ok(laid[0][0].fig.xlim.join() === '0,5' && laid[1][0].fig.xlim.join() === '0,5', `shared x: the union of the column (got ${laid[0][0].fig.xlim})`);
  ok(laid[0][0].fig.inner.x && !laid[1][0].fig.inner.x, 'only the lowest panel of a column labels the shared x axis');
  ok(!layoutPanels(c[0], true)[0][0].fig.inner.x, 'stacked on a narrow page, every panel keeps its labels');
  ok(laid[0][0].fig.roomRight === true && laid[1][0].fig.roomRight === true, 'a column with a right axis keeps room for it in every panel, so the plots line up');
  const r = sharedRange([laid[1][0]], 'y');
  ok(Math.abs(r[0] - (1 - 0.21)) < 1e-12 && Math.abs(r[1] - (4 + 0.21)) < 1e-12, `a shared y range leaves out the right-axis series and is padded 7% (got ${r})`);
  ok(charts(['grid: g ; 0 2', 'figure: a ; x ; y', 'plot: p ; [0, 1] ; [1, 2]'])[0].kind !== 'grid', 'a grid of no rows is no grid');
}

// ---- the corpus -------------------------------------------------------------
// The canonical form: what a chart IS,
// without the parser's bookkeeping (keys, which component draws it). A number
// that is not finite is null.
const CORPUS = join(here, '../../lang/viv/tests/figures');
const num = (v) => (Number.isFinite(v) ? v : null);
const nums = (a) => a.map(num);
function canonMark(m) {
  const out = {};
  for (const [k, v] of Object.entries(m)) out[k] = typeof v === 'number' ? num(v) : v;
  return out;
}
function canonFig(f) {
  if (!f) return null;
  const out = { title: f.title, xlabel: f.xlabel, ylabel: f.ylabel };
  if (f.xlim) out.xlim = nums(f.xlim);
  if (f.ylim) out.ylim = nums(f.ylim);
  if (f.style) out.style = f.style;
  if (f.xscale) out.xscale = f.xscale;
  if (f.yscale) out.yscale = f.yscale;
  if (f.aspect) out.aspect = f.aspect;
  if (f.yright) out.yright = f.yright;
  if (f.styles) out.styles = f.styles;
  if (f.origin) out.origin = f.origin;
  if (f.colormap) out.colormap = f.colormap;
  if (f.clim) out.clim = nums(f.clim);
  if (f.levels) out.levels = nums(f.levels);
  return out;
}
export function canon(charts) {
  return charts.map((c) => {
    if (c.kind === 'grid') return { kind: 'grid', title: c.title, rows: c.rows, cols: c.cols, share: c.share, panels: c.panels.map(canon) };
    if (c.kind === 'anim') return { kind: 'anim', records: c.lines.length };
    if (c.kind === 'diagram') return { kind: 'diagram', fig: canonFig(c.fig), marks: c.marks.map((m) => (m.kind === 'arrow' ? { ...m, x: nums(m.x), y: nums(m.y) } : canonMark(m))) };
    if (c.kind === 'scene' || c.kind === 'quiver' || c.kind === 'curve3d') return { kind: c.kind };
    const marks = [...((c.fig && c.fig.marks) || []), ...(c.marks || [])].map(canonMark);
    if (c.kind === 'heatmap') {
      return { kind: 'heatmap', fig: canonFig(c.fig), label: c.label, nx: c.nx, ny: c.ny, frames: c.frames.map(nums), marks };
    }
    const series = c.series.map((s) => {
      if (s.interval) return { name: s.name, type: s.interval === 'fill' ? 'fill' : 'rangebars', x: nums(c.x ? c.x.values : s.xs), lo: nums(s.lo), hi: nums(s.hi) };
      const type = s.bars ? 'bars' : c.x || s.line ? 'line' : 'points';
      const out = { name: s.name, type, x: nums(c.x ? c.x.values : s.xs), y: nums(c.x ? s.values : s.ys) };
      if (type === 'points') { out.color = s.color; out.open = !!s.open; }
      return out;
    });
    return { kind: 'xy', fig: canonFig(c.fig), series, marks };
  });
}
// arrays of numbers on one line, so a diff of an expected parse reads
const pretty = (v) => JSON.stringify(v, null, 2).replace(/\[\s*([-\d.e+,\snul]*?)\s*\]/g, (m, inner) => `[${inner.split(/,\s*/).join(', ')}]`) + '\n';
const write = process.argv.includes('--write');
for (const f of readdirSync(CORPUS).filter((n) => n.endsWith('.out')).sort()) {
  const text = readFileSync(join(CORPUS, f), 'utf8');
  const got = pretty(canon(parseOwnOutput(text).charts));
  const jf = join(CORPUS, f.replace(/\.out$/, '.json'));
  if (write) { writeFileSync(jf, got); continue; }
  let want = null;
  try { want = readFileSync(jf, 'utf8'); } catch { /* missing */ }
  ok(want !== null && JSON.stringify(JSON.parse(want)) === JSON.stringify(JSON.parse(got)), `corpus ${f}: the parse differs from ${f.replace(/\.out$/, '.json')}\n${got}`);
}

console.log(`figuretest: ${checks - fails} of ${checks} checks pass`);
process.exit(fails ? 1 : 0);
