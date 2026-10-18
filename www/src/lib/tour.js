// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The walkthrough of a recorded run on the explorer's program graph
// (components/ProgramGraph.jsx, GraphPanel.jsx): the stops a reader is
// taken through, in the order the run reached them, each with the box it
// stands at, the lines to mark and what to say. Pure functions, no DOM: the
// page and scripts/graph-browsertest.mjs share them.
//
// The stops come from the recording's tree (lib/explore.js, buildTree), the
// same one the Timeline steps through:
//
//   intro     what the program is for: its intent, else its header comment
//   setup     the used modules' top-level lines (constants, tables), as one stop
//   lines     a run of top-level statements of the program, or of a call's
//             lines outside its steps that set a number
//   call      a recorded call: the def's `what`
//   step      a step: its text, the numbers its lines set, its holds, and the
//             defs it calls that the walk does not enter, with their counts
//   loop      a loop: how many passes, the watched values across them
//   pass      one pass: its watched values; the first pass (and any pass the
//             reader opens) is then walked step by step, the others are one
//             stop each, so a loop of 14 passes is 14 stops, not 14 x 9
//   end       what the run printed, the loops' pass counts, the holds checked
//
// Defs the story does not record (the integral kernels, called tens of
// thousands of times) are never stops: a step names the busiest of them, and
// the reader can open one in the panel.
import { labelOf, iterValues } from './explore.js';

const MAIN = '<main>';

/** Scalars (numbers and booleans) among a set of recorded values. */
export function scalars(values) {
  const out = {};
  for (const [k, v] of Object.entries(values || {})) {
    if (typeof v === 'number' || typeof v === 'boolean') out[k] = v;
    else if (typeof v === 'string' && /^-?(nan|inf)$/.test(v)) out[k] = v;
  }
  return out;
}

/** A value as the narration shows it: 7 significant digits, small and large ones with an exponent, a real minus sign. */
export function fmtValue(v) {
  if (typeof v === 'boolean') return String(v);
  if (typeof v === 'string') return v;
  if (typeof v !== 'number' || !Number.isFinite(v)) return String(v);
  if (v === 0) return '0';
  if (Number.isInteger(v) && Math.abs(v) < 1e9) return v.toLocaleString('en-US').replace('-', '−');
  const a = Math.abs(v);
  let s;
  if (a >= 1e-3 && a < 1e7) s = String(Number(v.toPrecision(7)));
  else s = v.toExponential(2).replace(/\.?0+e/, 'e').replace('e+', 'e');
  return s.replace(/^-/, '−').replace('e-', 'e−');
}

/** An array value's shape in words: `13 × 13`, or null. */
export function shapeOf(v) {
  return v && typeof v === 'object' && Array.isArray(v.shape) ? v.shape.join(' × ') : null;
}

/** The names a def's debug lens watches per iteration, in order. */
function watchedPerPass(ix, def) {
  const d = (ix.bundle.debug || []).find((x) => x.def === def && x.kind === 'watch' && x.at === 'iteration');
  if (!d) return [];
  // the bundle names a watch's items, computed ones (`name = expr`) included
  return d.names || d.text.replace(/^watch\s+per\s+iteration\s+/, '').split(/\s*,\s*/).filter(Boolean);
}

/** The values a pass ends with: the names watched per iteration when the debug lens has them, else its scalars. */
export function passValues(ix, it) {
  const names = watchedPerPass(ix, it.def);
  const w = it.watch.find((x) => x.at === 'iteration');
  const all = { ...scalars(iterValues(it)), ...(w ? scalars(w.values) : {}) };
  if (names.length) return names.filter((n) => n in all).map((n) => [n, all[n]]);
  return Object.entries(all).slice(0, 5);
}

/** For a loop: each watched name's value across its passes, for the sparklines. */
export function loopSeries(ix, loop) {
  const names = watchedPerPass(ix, loop.def);
  const rows = loop.children.map((it) => Object.fromEntries(passValues(ix, it)));
  const keys = names.length ? names : [...new Set(rows.flatMap((r) => Object.keys(r)))].slice(0, 4);
  const out = [];
  for (const k of keys) {
    const pts = rows.map((r, j) => ({ x: j + 1, y: typeof r[k] === 'number' ? r[k] : null })).filter((p) => p.y !== null && Number.isFinite(p.y));
    if (pts.length >= 2 && new Set(pts.map((p) => p.y)).size > 1) out.push({ name: k, pts });
  }
  return out;
}

/**
 * A short name for a variable from its glossary line: the words before the
 * first comma, colon, semicolon or parenthesis outside math and code, at
 * most 64 characters; '' when there is none.
 */
export function shortGloss(ix, def, name) {
  const g = ((ix.bundle.intent || {})[def] || {}).glossary || [];
  const hit = g.find(([n]) => n === name);
  if (!hit) return '';
  const t = hit[1];
  let math = false;
  let code = false;
  let end = t.length;
  for (let i = 0; i < t.length; i++) {
    const c = t[i];
    if (c === '\\') {
      i++;
      continue;
    }
    if (c === '$' && !code) math = !math;
    else if (c === '`' && !math) code = !code;
    else if (!math && !code && (',;:('.includes(c) || (c === '.' && t[i + 1] === ' '))) {
      end = i;
      break;
    }
  }
  const out = t.slice(0, end).trim().replace(/\.$/, '');
  return out.length <= 64 && !math && !code ? out : '';
}

/** The first paragraph of the program's own header comment, or of its first `###` block. */
export function programPurpose(ix) {
  const it = (ix.bundle.intent || {})[MAIN] || {};
  if (it.what) return it.what;
  const P = ix.programLines;
  const root = ix.toModule(1)?.module;
  const para = [];
  let fence = false;
  for (let x = 1; x <= P.length && ix.toModule(x)?.module === root; x++) {
    const t = P[x - 1];
    if (/^\s*###\s*$/.test(t)) {
      if (fence && para.length) break;
      fence = !fence;
      continue;
    }
    if (fence) {
      // a post's header lines (`title: ...`) are not its prose
      if (/^[a-z][\w-]*:\s/i.test(t) && !para.length) continue;
      if (!t.trim()) {
        if (para.length) break;
        continue;
      }
      para.push(t.trim());
      continue;
    }
    const m = /^#(?![|:!@~#[\]])\s?(.*)$/.exec(t);
    if (!m) {
      if (para.length) break;
      continue;
    }
    if (!m[1].trim()) {
      if (para.length) break;
      continue;
    }
    para.push(m[1].trim());
  }
  return para.join(' ');
}

/** The graph's node for a def's step, if the graph has it. */
function stepNode(ix, def, id) {
  return ix.nodes.get(`${def}/${id}`) || null;
}

/** The lines a stop marks in its def's code: { line, end } in program lines, or null. */
function markOf(ix, n) {
  if (n.kind === 'step') {
    const g = stepNode(ix, n.def, n.id);
    return g ? { line: g.line, end: g.end } : null;
  }
  if (n.kind === 'loop' || n.kind === 'iter') {
    // the whole loop for the loop; its first line for a pass
    const loop = n.kind === 'loop' ? n : n.parent;
    const g = loop.graphId && ix.nodes.get(loop.graphId);
    if (g) return { line: g.line, end: n.kind === 'loop' ? g.end : g.line };
    // a loop the graph has no box for: the lines its first pass recorded in its def
    const xs = [];
    const go = (m) => {
      for (const c of m.children) {
        if (c.kind === 'line' && c.def === loop.def) xs.push(c.line);
        else if (c.kind !== 'call') go(c);
      }
    };
    if (loop.children[0]) go(loop.children[0]);
    if (xs.length) return { line: Math.min(...xs), end: n.kind === 'loop' ? Math.max(...xs) : Math.min(...xs) };
    // nothing recorded inside: the def's first loop line
    const d = ix.nodes.get(loop.def);
    for (let x = d ? d.line : 0; d && x <= d.end; x++) if (/^\s*(for|while)\b/.test(ix.programLines[x - 1] || '')) return { line: x, end: x };
    return null;
  }
  if (n.kind === 'call') {
    const g = ix.nodes.get(n.def);
    return g && g.line > 0 ? { line: g.line, end: g.line } : null;
  }
  return null;
}

/** The graph box a node stands at: its step, its loop, or its def. */
function boxOf(ix, n) {
  if (n.kind === 'step') return stepNode(ix, n.def, n.id) ? `${n.def}/${n.id}` : n.def;
  if (n.kind === 'loop') return n.graphId || n.def;
  if (n.kind === 'iter') return n.parent.graphId || n.def;
  if (n.kind === 'group') return MAIN;
  return n.def;
}

/** The numbers a node's own lines set (not inside a nested step, loop or call), last value of each. */
function valuesSet(n) {
  const out = new Map();
  for (const c of n.children) if (c.kind === 'line') for (const [k, v] of Object.entries(c.values || {})) out.set(k, v);
  for (const w of n.watch) for (const [k, v] of Object.entries(w.values || {})) out.set(k, v);
  return out;
}

/** Every hold checked in a subtree: { ok, fail, sentences: Map(sentence -> { ok, fail }) }. */
export function holdsIn(n) {
  const out = { ok: 0, fail: 0, sentences: new Map() };
  const go = (m) => {
    for (const h of m.holds) {
      h.ok ? out.ok++ : out.fail++;
      const s = out.sentences.get(h.sentence) || { ok: 0, fail: 0 };
      h.ok ? s.ok++ : s.fail++;
      out.sentences.set(h.sentence, s);
    }
    for (const c of m.children) go(c);
  };
  go(n);
  return out;
}

/**
 * The stops of a walkthrough. `open` is a Set of pass keys the reader asked
 * to walk step by step (beyond each loop's first pass).
 */
export function tourStops(tree, ix, open = new Set()) {
  const out = [];
  const add = (s) => out.push({ k: out.length, ...s });
  add({ kind: 'intro', node: tree, box: MAIN, def: MAIN, mark: null, title: 'What the program does' });
  const seenLoops = new Set();
  // `fold`: the node's own lines are part of its stop (a step's, a pass's)
  const walk = (n, depth, fold) => {
    let run = [];
    const lineStop = (ls) => {
      const nums = new Map();
      for (const l of ls) for (const [k, v] of Object.entries(scalars(l.values))) nums.set(k, v);
      // the program's own statements are the story's spine; elsewhere, a
      // run of lines is a stop only when it sets a number
      if (n.def !== MAIN && !nums.size) return;
      const lines = ls.map((l) => l.line);
      add({ kind: 'lines', node: ls[0], nodes: ls, box: n.def, def: n.def, mark: { line: Math.min(...lines), end: Math.max(...lines) }, values: [...nums], title: n.def === MAIN ? 'The program' : `In ${n.def}`, depth });
    };
    const flush = () => {
      if (run.length && !fold) lineStop(run);
      run = [];
    };
    // a top-level statement's line is recorded when it ends, after the calls
    // it made; the walk shows the statement first, then its calls
    const early = new Set();
    const kids = n.children;
    for (let j = 0; j < kids.length; j++) {
      const c = kids[j];
      if (c.kind === 'line') {
        if (!early.has(c)) run.push(c);
        continue;
      }
      flush();
      if (n.def === MAIN && !fold && c.kind === 'call') {
        let k = j + 1;
        while (k < kids.length && kids[k].kind === 'call') k++;
        const stmt = kids[k];
        if (stmt && stmt.kind === 'line' && !early.has(stmt)) {
          early.add(stmt);
          lineStop([stmt]);
        }
      }
      if (c.kind === 'group') {
        add({ kind: 'setup', node: c, box: MAIN, def: MAIN, mark: null, title: labelOf(c, ix), depth });
      } else if (c.kind === 'call') {
        add({ kind: 'call', node: c, box: c.def, def: c.def, mark: markOf(ix, c), title: `${c.def} is called`, depth });
        walk(c, depth + 1, false);
      } else if (c.kind === 'step') {
        add({ kind: 'step', node: c, box: boxOf(ix, c), def: c.def, mark: markOf(ix, c), values: [...valuesSet(c)], title: `${c.def} / ${c.id}`, depth });
        walk(c, depth + 1, true);
      } else if (c.kind === 'loop') {
        // a loop met again (its def called again) is one stop: its passes were walked the first time
        const seenKey = `${c.def}|${c.graphId || ''}`;
        const again = seenLoops.has(seenKey);
        seenLoops.add(seenKey);
        add({ kind: 'loop', node: c, box: boxOf(ix, c), def: c.def, mark: markOf(ix, c), title: labelOf(c, ix), depth, again });
        if (again) continue;
        for (const it of c.children) {
          const walked = it.n === 1 || open.has(it.key);
          add({ kind: 'pass', node: it, box: boxOf(ix, it), def: it.def, mark: markOf(ix, it), values: passValues(ix, it), title: `Pass ${it.n} of ${c.children.length}`, depth: depth + 1, walked });
          if (walked) walk(it, depth + 2, true);
        }
      }
    }
    flush();
  };
  walk(tree, 0, false);
  add({ kind: 'end', node: tree, box: MAIN, def: MAIN, mark: null, title: 'The end of the run' });
  return out;
}

/**
 * The defs a step calls, directly or below, that the walk does not stop in,
 * busiest first: [{ id, count }] with each def's calls in the whole run.
 * `recorded` is the set of defs the story records calls of.
 */
export function hotBelow(model, stepId, recorded, max = 3) {
  const g = model.byId.get(stepId);
  if (!g || !(g.calls || []).length) return [];
  const total = (d) => (model.callers.get(d) || []).reduce((s, c) => s + (model.counts.get(`${c}>${d}`) || 0), 0);
  const seen = new Set();
  const queue = [...g.calls];
  while (queue.length) {
    const d = queue.shift();
    if (seen.has(d) || recorded.has(d)) continue;
    seen.add(d);
    for (const c of model.callees.get(d) || []) queue.push(c);
  }
  return [...seen].map((id) => ({ id, count: total(id) })).filter((x) => x.count > 1).sort((a, b) => b.count - a.count).slice(0, max);
}

/** The defs the recording has call nodes for. */
export function recordedDefs(tree) {
  const s = new Set();
  const go = (n) => {
    if (n.kind === 'call') s.add(n.def);
    for (const c of n.children) go(c);
  };
  go(tree);
  return s;
}
