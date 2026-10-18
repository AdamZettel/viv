// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// What a model gets back from running a program: the lines the program meant
// for a reader, its tests, and a sentence per figure instead of the figure's
// data.
//
// Plot records are most of a numerical program's output (96.6-99.5% of the
// bytes in the four programs measured), and a
// model cannot look at them. It can use the shape: how many points, over what
// range, and what the y values do. So each record becomes one line, and the
// numbers the program printed with `show` and `print` come back as written.
import { FIGURE_LINE } from '../../www/src/lib/postCheck.js';

// records the site draws that FIGURE_LINE (the feed's figure-size rule) does not count
const SCENE_LINE = /^(scene|wire|verts):/;
// records that set something about the open figure rather than draw
const FIGURE_SETTINGS = new Set(['xscale', 'yscale', 'aspect', 'yright', 'style', 'origin', 'colormap', 'clim', 'levels']);
const INSTRUMENT = /^__[a-z]+__/; // the interpreter's call graph and trace
const MAX_SHOWN_LINES = 200;
const MAX_SHOWN_BYTES = 16 * 1024;

const fmt = (x) => (Number.isFinite(x) ? Number(x.toPrecision(4)).toString() : String(x));

/** The numbers of a `[a, b, c]` field; `nan` and `inf` read as themselves. */
function numbers(field) {
  const m = field.match(/^\s*\[(.*)\]\s*$/s);
  if (!m) return null;
  if (!m[1].trim()) return [];
  return m[1].split(',').map((t) => Number(t.trim().replace(/^(-?)inf$/, '$1Infinity')));
}

function range(xs) {
  let lo = Infinity;
  let hi = -Infinity;
  let bad = 0;
  for (const x of xs) {
    if (Number.isNaN(x)) { bad++; continue; }
    if (x < lo) lo = x;
    if (x > hi) hi = x;
  }
  return { lo, hi, bad };
}

/** One sentence about how a series' y values move: rising, falling, or where the extremes are. */
function trend(ys) {
  if (ys.length < 3) return '';
  let up = true;
  let down = true;
  for (let i = 1; i < ys.length; i++) {
    if (ys[i] < ys[i - 1]) up = false;
    if (ys[i] > ys[i - 1]) down = false;
  }
  if (up) return 'rising throughout';
  if (down) return 'falling throughout';
  const { lo, hi } = range(ys);
  return `first ${fmt(ys[0])}, last ${fmt(ys[ys.length - 1])}, min at index ${ys.indexOf(lo)}, max at index ${ys.indexOf(hi)}`;
}

function describeSeries(kind, fields) {
  const label = (fields[0] || '').trim() || '(no label)';
  const arrays = fields.slice(1).map(numbers).filter(Boolean);
  if (kind === 'plot2d') {
    // plot2d: label ; rows cols ; [data]
    const [rows, cols] = (fields[1] || '').trim().split(/\s+/).map(Number);
    const data = numbers(fields[2] || '') || [];
    const r = range(data);
    return `${kind} "${label}": a ${rows} x ${cols} field, values from ${fmt(r.lo)} to ${fmt(r.hi)}${r.bad ? `, ${r.bad} NaN` : ''}`;
  }
  if (!arrays.length) return `${kind} "${label}"`;
  const n = arrays[0].length;
  const names = kind === 'quiver' ? ['x', 'y', 'u', 'v'] : kind === 'fill_between' || kind === 'rangebars' ? ['x', 'lo', 'hi'] : ['x', 'y', 'z'];
  const parts = arrays.map((a, i) => {
    const r = range(a);
    return `${names[i] || `field ${i + 1}`} ${fmt(r.lo)}..${fmt(r.hi)}${r.bad ? ` (${r.bad} NaN)` : ''}`;
  });
  const y = arrays[1];
  const shape = y && (kind === 'plot' || kind === 'scatter' || kind === 'bars') ? `; y ${trend(y)}` : '';
  return `${kind} "${label}": ${n} points, ${parts.join(', ')}${shape}`;
}

/**
 * The run's output, summarized.
 * @returns {{ error: string|null, shown: string[], shownTruncated: number, tests: {name, passed, detail}[], figures: string[], animation: string|null }}
 */
export function summarize(out) {
  const lines = String(out || '').split('\n');
  const shown = [];
  let shownBytes = 0;
  let shownTruncated = 0;
  const tests = [];
  const figures = [];
  let current = null; // the figure the series lines belong to
  const anim = {};
  let stage = null;
  let error = null;
  for (const raw of lines) {
    const line = raw.replace(/\s+$/, '');
    if (!line || INSTRUMENT.test(line)) continue;
    if (/^error\b/.test(line)) { error = error || line; continue; }
    const t = line.match(/^(PASS|FAIL) (.*?)(?: -- (.*))?$/);
    if (t) { tests.push({ name: t[2], passed: t[1] === 'PASS', detail: t[3] || '' }); continue; }
    if (/^\d+ assertions? FAILED$/.test(line)) continue; // the summary count; the FAIL lines say it
    if (FIGURE_LINE.test(line) || SCENE_LINE.test(line)) {
      const kind = line.slice(0, line.indexOf(':'));
      const fields = line.slice(kind.length + 1).split(' ; ');
      if (kind === 'figure' || kind === 'scene') {
        const [title, xl, yl, lims] = fields.map((f) => f.trim());
        current = { head: kind === 'figure' ? `figure "${title}" (x: ${xl || '?'}, y: ${yl || '?'}${lims ? `, window ${lims}` : ''})` : `3D scene "${title}"`, series: [] };
        figures.push(current);
      } else if (['plot', 'scatter', 'bars', 'fill_between', 'rangebars', 'plot2d', 'plot3d', 'quiver', 'wire', 'verts'].includes(kind)) {
        if (!current) { current = { head: 'a figure with no title', series: [] }; figures.push(current); }
        current.series.push(describeSeries(kind, fields));
      } else if (kind === 'grid' || kind === 'share') {
        // a grid of figures: the figures after it are its panels
        if (kind === 'grid') { current = { head: `a grid "${fields[0].trim()}" of ${(fields[1] || '').trim().replace(' ', ' x ')} panels (the figures below)`, series: [] }; figures.push(current); }
        else if (current) current.head += ` [share ${fields[0].trim()}]`;
      } else if ((kind === 'box' || kind === 'arrow' || kind === 'note') && current) {
        // a diagram's marks: counted, not listed
        current.marks = current.marks || { box: 0, arrow: 0, note: 0 };
        current.marks[kind] += 1;
      } else if (FIGURE_SETTINGS.has(kind) && current) {
        // a setting of the open figure: part of its description
        current.head += ` [${kind} ${fields.map((f) => f.trim()).join(' ')}]`;
      } else {
        // an animation record: count them by kind, and keep the stage and the captions
        anim[kind] = (anim[kind] || 0) + 1;
        if (kind === 'stage') stage = fields[0].trim();
      }
      continue;
    }
    if (shown.length >= MAX_SHOWN_LINES || shownBytes + line.length > MAX_SHOWN_BYTES) { shownTruncated++; continue; }
    shown.push(line);
    shownBytes += line.length + 1;
  }
  const animation = Object.keys(anim).length
    ? `an animation${stage ? ` "${stage}"` : ''}: ${Object.entries(anim).map(([k, n]) => `${n} ${k}`).join(', ')} records`
    : null;
  return { error, shown, shownTruncated, tests, figures: figures.map((f) => [f.head + (f.marks ? ` [a diagram: ${f.marks.box} boxes, ${f.marks.arrow} arrows, ${f.marks.note} notes]` : ''), ...f.series.map((s) => `  - ${s}`)].join('\n')), animation };
}

/** The summary as the text a model reads. */
export function summaryText(s, { seconds = null, engine = 'the interpreter (viv.wasm)' } = {}) {
  const out = [];
  if (s.error) out.push(`The program stopped with an error:\n${s.error}`);
  const passed = s.tests.filter((t) => t.passed).length;
  if (s.tests.length) {
    out.push(`Tests: ${passed} of ${s.tests.length} passed.`);
    out.push(s.tests.map((t) => `${t.passed ? 'PASS' : 'FAIL'} ${t.name}${t.detail ? ` -- ${t.detail}` : ''}`).join('\n'));
  } else if (!s.error) out.push('No test ran. A program checks its claims with `test "..." { assert_near(...) }` blocks.');
  if (s.shown.length) out.push(`Printed:\n${s.shown.join('\n')}${s.shownTruncated ? `\n(${s.shownTruncated} more lines not shown)` : ''}`);
  if (s.figures.length) out.push(`Figures (described; the data points are not included):\n${s.figures.join('\n')}`);
  if (s.animation) out.push(`Also ${s.animation}.`);
  out.push(`Ran in ${engine}${seconds !== null ? ` in ${seconds.toFixed(2)} s` : ''}.`);
  return out.join('\n\n');
}
