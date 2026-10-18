// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The walk through a run (/explore/walk/<slug>, components/ZoomWalk.jsx): the
// run's call graph, with a camera that flies to the def that is executing,
// and the source beside it at reading size. Pure functions, no DOM;
// scripts/zoomwalk-test.mjs checks them.
//
//   traceSource(bundle)            the program with every def traced (see below)
//   layoutWorld(ix, tree)          the call graph: one box per def, in layers by call depth
//   codeFiles(ix)                  the program as files in reading order, for the code pane
//   buildFrames(tree, ix, opts)    the run as the camera's stops, loops and repeated calls collapsed
//   targetRect(frame, world)       the part of the graph a frame is about
//   fitView(rect, vw, vh)          the view [cx, cy, w] that shows a rect
//   interpolateZoom(a, b)          van Wijk and Nuij's smooth zoom-and-pan path between two views
//   nextOver / nextOut             step over and step out on the frame list
//
// Trace source. The explorer's recording (lang/viv/src/graph.rs,
// `viv_explore_run`) has call and ret events only for defs with steps. The
// walk needs every call, so it adds one line to the program's `debug * { }`
// block before recording: `trace <every def>`. That is the debug lens's own
// `trace` item (the debug lens), so the interpreter and
// the compiled C record the same events and the program prints what it
// always printed. The story keeps its first 100,000 events (STORY_CAP in
// lang/viv/src/interp.rs); the walk keeps at most FRAME_CAP stops.
import { MATH_SPLIT } from './richText.js';
import { VIV_BUILTINS } from './highlight.js';

export const FRAME_CAP = 20000;
// A traced run with more events than TRACE_LIMIT is recorded again without
// tracing the defs it called more than HOT_CALLS times (silicon's plane-wave
// kernels run millions of times): see hotDefs.
export const TRACE_LIMIT = 150000;
export const HOT_CALLS = 400;

/** The defs called more than HOT_CALLS times in the kept events, without steps of their own. */
export function hotDefs(events, ix) {
  const n = new Map();
  for (const e of events) if (e[0] === 'call') n.set(e[1], (n.get(e[1]) || 0) + 1);
  return new Set([...n.entries()].filter(([d, k]) => k > HOT_CALLS && !ix.nodes.get(d)?.stepped).map(([d]) => d));
}

/**
 * The defs a `parallel for` reaches: the defs called in its body, and the
 * defs they call. Tracing one would make the passes of the loop write the
 * same story, which the compiler refuses, so the walk leaves them untraced
 * (the line recording skips them the same way, `parallel_reached` in
 * lang/viv/src/debug_lens.rs).
 */
export function parallelReached(bundle) {
  const names = new Set(bundle.nodes.filter((n) => n.kind === 'def').map((n) => n.id));
  const lines = bundle.program.split('\n');
  const todo = [];
  lines.forEach((l, k) => {
    if (!/^\s*parallel\s+for\b/.test(l)) return;
    // the loop's body: from its line to the brace that closes it
    let depth = 0;
    for (let j = k; j < lines.length; j++) {
      const s = lines[j].replace(/"(?:[^"\\]|\\.)*"/g, '""').replace(/#.*$/, '');
      for (const m of s.matchAll(/([A-Za-z_]\w*)\s*\(/g)) if (names.has(m[1])) todo.push(m[1]);
      for (const ch of s) depth += ch === '{' ? 1 : ch === '}' ? -1 : 0;
      if (depth <= 0 && j > k) break;
      if (depth <= 0 && s.includes('}')) break;
    }
  });
  const callees = new Map();
  for (const [from, to] of bundle.calls || []) {
    if (!callees.has(from)) callees.set(from, []);
    callees.get(from).push(to);
  }
  const out = new Set();
  while (todo.length) {
    const d = todo.pop();
    if (out.has(d)) continue;
    out.add(d);
    for (const c of callees.get(d) || []) todo.push(c);
  }
  return out;
}

/** The program with a `trace` line naming every def, so each call records `call` and `ret`. */
export function traceSource(bundle, also = new Set()) {
  const skip = new Set([...parallelReached(bundle), ...also]);
  const defs = bundle.nodes.filter((n) => n.kind === 'def' && n.id !== '<main>' && /^[A-Za-z_]\w*$/.test(n.id) && !skip.has(n.id)).map((n) => n.id);
  if (!defs.length) return bundle.program;
  const line = `  trace ${defs.join(', ')}\n`;
  if (/^debug \* \{\n/m.test(bundle.program)) return bundle.program.replace(/^debug \* \{\n/m, (m) => m + line);
  return `${bundle.program.replace(/\n*$/, '\n')}debug * {\n${line}}\n`;
}

// ---------------------------------------------------------------------------
// The call graph
// ---------------------------------------------------------------------------

// world units: a def box is DEF_W x DEF_H; layers are columns COL_GAP apart
export const DEF_W = 240;
export const DEF_H = 74;
const COL_GAP = 120;
const ROW_GAP = 26;

/** Every call node of the tree, in the order the run made them. */
function callsInOrder(tree) {
  const out = [];
  const go = (n) => {
    if (n.kind === 'call') out.push(n);
    for (const c of n.children) go(c);
  };
  go(tree);
  return out;
}

/** The call node a node runs in. */
function callOf(n) {
  let up = n.parent;
  while (up && up.kind !== 'call') up = up.parent;
  return up;
}

/** A def's signature as the reader sees it: from the intent lens, or its name and parameters. */
export function sigOf(ix, def) {
  if (def === '<main>') return '';
  const s = ix.bundle.intent?.[def]?.sig;
  if (s) return s;
  const g = defNode(ix, def);
  return `${def}(${(g?.params || []).join(', ')})`;
}

const TEX = {
  lVert: '‖', rVert: '‖', Vert: '‖', top: 'ᵀ', lambda: 'λ', Delta: 'Δ', delta: 'δ', rho: 'ρ', sigma: 'σ', Sigma: 'Σ',
  alpha: 'α', beta: 'β', gamma: 'γ', epsilon: 'ε', varepsilon: 'ε', mu: 'μ', nu: 'ν', pi: 'π', phi: 'φ', psi: 'ψ',
  Psi: 'Ψ', omega: 'ω', Omega: 'Ω', theta: 'θ', chi: 'χ', tau: 'τ', kappa: 'κ', nabla: '∇', partial: '∂', le: '≤',
  ge: '≥', leq: '≤', geq: '≥', approx: '≈', times: '×', cdot: '·', to: '→', infty: '∞', sum: 'Σ', int: '∫', sqrt: '√',
};

/** The first sentence of a def's intent, as plain text (TeX turned into the symbols it stands for), or ''. */
export function intentLine(ix, def) {
  const what = ix.bundle.intent?.[def]?.what || '';
  if (!what) return '';
  const plain = what
    .replace(/\[test "[^"]*"\]/g, '')
    .replace(/\$([^$]*)\$/g, (_, t) => t.replace(/\\([A-Za-z]+)/g, (m, w) => TEX[w] ?? '').replace(/\\[,;!]/g, ' ').replace(/\^\{?(\d+)\}?/g, (_, n) => n.replace(/\d/g, (c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[c])).replace(/[{}]/g, ''))
    .replace(/`/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  const m = plain.match(/^.*?[.!?](?=\s|$)/);
  return (m ? m[0] : plain).trim();
}

/** The same first sentence with its TeX kept, for a reader with KaTeX
 *  (lib/richText.js renderProse): a sentence ends outside the math and the
 *  code, so `$f(x) = 0.5$` does not end one. '' when the def has no intent. */
export function intentLineTex(ix, def) {
  const what = (ix.bundle.intent?.[def]?.what || '').replace(/\[test "[^"]*"\]/g, '').replace(/\s+/g, ' ').trim();
  if (!what) return '';
  let out = '';
  for (const part of what.split(MATH_SPLIT)) {
    if (/^[$`]/.test(part) && MATH_SPLIT.test(part)) {
      out += part;
      continue;
    }
    const m = part.match(/^.*?[.!?](?=\s|$)/);
    if (m) return (out + m[0]).trim();
    out += part;
  }
  return out.trim();
}

/**
 * The call graph of the run: one box per def the run entered, laid out in
 * layers by call depth (a def's layer is the fewest calls it takes to reach
 * it from the top level), callers left of their callees. Within a layer the
 * boxes are ordered to keep call edges short (the barycenter heuristic of
 * Sugiyama, Tagawa and Toda, 1981: a few sweeps that put each box at the mean
 * position of its neighbors in the layer before). The module a def comes from
 * is a label and a tint on its box, nothing more.
 */
export function layoutWorld(ix, tree) {
  const calls = callsInOrder(tree);
  const order = [];
  const seen = new Set();
  for (const c of calls) {
    if (!seen.has(c.def)) {
      seen.add(c.def);
      order.push(c.def);
    }
  }
  const edgeCount = new Map();
  for (const c of calls) {
    const up = callOf(c);
    if (!up) continue;
    const k = `${up.def}\u0000${c.def}`;
    edgeCount.set(k, (edgeCount.get(k) || 0) + 1);
  }
  const edges = [...edgeCount.entries()].map(([k, n]) => {
    const [from, to] = k.split('\u0000');
    return { from, to, n };
  });
  const callCount = new Map();
  for (const c of calls) callCount.set(c.def, (callCount.get(c.def) || 0) + 1);

  // layers: breadth first from the top level
  const out = new Map();
  const inn = new Map();
  for (const e of edges) {
    if (e.from === e.to) continue;
    if (!out.has(e.from)) out.set(e.from, []);
    out.get(e.from).push(e.to);
    if (!inn.has(e.to)) inn.set(e.to, []);
    inn.get(e.to).push(e.from);
  }
  const layerOf = new Map([[order[0], 0]]);
  const queue = [order[0]];
  while (queue.length) {
    const d = queue.shift();
    for (const c of out.get(d) || []) {
      if (!layerOf.has(c)) {
        layerOf.set(c, layerOf.get(d) + 1);
        queue.push(c);
      }
    }
  }
  for (const d of order) if (!layerOf.has(d)) layerOf.set(d, 1);
  const nLayers = Math.max(...layerOf.values()) + 1;
  const layers = Array.from({ length: nLayers }, () => []);
  for (const d of order) layers[layerOf.get(d)].push(d);

  // order within layers: barycenter sweeps, down then up
  const pos = new Map();
  const index = () => layers.forEach((l) => l.forEach((d, k) => pos.set(d, k)));
  index();
  const bary = (d, nbrs, layer) => {
    const ps = (nbrs.get(d) || []).filter((x) => layerOf.get(x) === layer).map((x) => pos.get(x));
    return ps.length ? ps.reduce((a, b) => a + b, 0) / ps.length : pos.get(d);
  };
  for (let sweep = 0; sweep < 6; sweep++) {
    const down = sweep % 2 === 0;
    const ls = down ? range(1, nLayers - 1) : range(0, nLayers - 2).reverse();
    for (const L of ls) {
      const nb = down ? inn : out;
      const other = down ? L - 1 : L + 1;
      const key = new Map(layers[L].map((d) => [d, bary(d, nb, other)]));
      layers[L].sort((a, b) => key.get(a) - key.get(b) || pos.get(a) - pos.get(b));
      index();
    }
  }

  // modules, numbered in the order the run first reached them, for the tint
  const moduleOf = (def) => {
    const g = defNode(ix, def);
    const m = def === '<main>' || !g || !(g.line > 0) ? null : ix.toModule(g.line);
    return m ? m.module : ix.bundle.modules[0];
  };
  const modules = [];
  const defs = new Map();
  layers.forEach((l, L) => {
    const top = -((l.length - 1) * (DEF_H + ROW_GAP)) / 2;
    l.forEach((def, k) => {
      const module = moduleOf(def);
      if (!modules.includes(module)) modules.push(module);
      defs.set(def, {
        def, slot: defs.size, layer: L, module, tint: modules.indexOf(module) % 6, calls: callCount.get(def) || 0,
        sig: sigOf(ix, def), intent: intentLine(ix, def), tex: intentLineTex(ix, def),
        x: L * (DEF_W + COL_GAP), y: top + k * (DEF_H + ROW_GAP), w: DEF_W, h: DEF_H,
      });
    });
  });
  const ds = [...defs.values()];
  const x0 = Math.min(...ds.map((d) => d.x));
  const y0 = Math.min(...ds.map((d) => d.y));
  const bounds = { x: x0, y: y0, w: Math.max(...ds.map((d) => d.x + d.w)) - x0, h: Math.max(...ds.map((d) => d.y + d.h)) - y0 };
  return { defs, edges, bounds, layers, modules };
}

function range(a, b) {
  const out = [];
  for (let x = a; x <= b; x++) out.push(x);
  return out;
}

/** A call edge: a cubic from the caller's right side to the callee's left, or a loop for a def that calls its own layer or one before it. */
export function edgePath(a, b) {
  const y1 = a.y + a.h / 2;
  const y2 = b.y + b.h / 2;
  if (a === b) {
    const x = a.x + a.w;
    return `M ${x} ${a.y + a.h * 0.3} C ${x + 50} ${a.y - 10}, ${x + 50} ${a.y + a.h + 10}, ${x} ${a.y + a.h * 0.7}`;
  }
  if (b.layer > a.layer) {
    const x1 = a.x + a.w;
    const x2 = b.x;
    const bend = Math.max(40, (x2 - x1) * 0.5);
    return `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`;
  }
  // backwards or within a layer: out of the caller's bottom, into the callee's top
  const x1 = a.x + a.w / 2;
  const x2 = b.x + b.w / 2;
  const below = b.y >= a.y;
  const ya = below ? a.y + a.h : a.y;
  const yb = below ? b.y : b.y + b.h;
  const s = below ? 1 : -1;
  return `M ${x1} ${ya} C ${x1 + 30} ${ya + s * 40}, ${x2 + 30} ${yb - s * 40}, ${x2} ${yb}`;
}

// ---------------------------------------------------------------------------
// The code pane
// ---------------------------------------------------------------------------

/**
 * The program as files in reading order: for each module, its lines (as
 * program line numbers) sorted by their line in the module. The program the
 * explorer records is the modules spliced together; the pane shows each one
 * as the file it is.
 */
const SPLICE_BANNER = /^# ---- (#! use \S+|end of \S+) ----\s*$/;

export function codeFiles(ix) {
  const byMod = new Map();
  let rootLine = 0; // the last line of the opened file seen so far
  for (let x = 1; x <= ix.programLines.length; x++) {
    // the banners the splicer writes around a used module are not in any
    // file; the one that opens a module stands where the file's import line
    // was (the spliced program no longer has it), so the opened file keeps a
    // row there that names the module (`use`)
    const banner = (ix.programLines[x - 1] || '').match(SPLICE_BANNER);
    if (banner) {
      const use = banner[1].match(/^#! use (\S+)/);
      if (use && rootLine > 0) {
        if (!byMod.has(0)) byMod.set(0, []);
        byMod.get(0).push({ x, n: rootLine + 1, use: use[1] });
      }
      continue;
    }
    const m = ix.toModule(x);
    const k = m ? m.moduleIndex : 0;
    if (!byMod.has(k)) byMod.set(k, []);
    byMod.get(k).push({ x, n: m ? m.line : x });
    if (k === 0) rootLine = m ? m.line : x;
  }
  const files = [...byMod.entries()].map(([k, rows]) => {
    rows.sort((a, b) => a.n - b.n);
    while (rows.length && !(ix.programLines[rows[rows.length - 1].x - 1] || '').trim()) rows.pop();
    return { index: k, module: ix.bundle.modules[k] || ix.bundle.modules[0], rows };
  });
  // the root module (the program the reader opened) first, then the others in the order the program uses them
  files.sort((a, b) => (a.index === 0 ? -1 : b.index === 0 ? 1 : a.rows[0].x - b.rows[0].x));
  const fileOf = new Map();
  files.forEach((f, k) => f.rows.forEach((r) => fileOf.set(r.x, k)));
  return { files, fileOf };
}

/** The def a program line is in ('<main>' outside every def). */
export function defAt(ix, x) {
  if (!ix._defRanges) ix._defRanges = ix.bundle.nodes.filter((n) => n.kind === 'def' && n.line > 0).map((n) => [n.line, n.end, n.id]);
  let best = null;
  for (const [a, b, id] of ix._defRanges) if (x >= a && x <= b && (!best || b - a < best[1] - best[0])) best = [a, b, id];
  return best ? best[2] : '<main>';
}

/** The first line of a def in the program, or null for the top level. */
export function defLine(ix, def) {
  const g = defNode(ix, def);
  return g && g.line > 0 ? g.line : null;
}

/** A def's node in the bundle (a record shares its name with its constructor's def; the def is the one wanted). */
export function defNode(ix, def) {
  const g = ix.nodes.get(def);
  if (g && g.kind === 'def') return g;
  return ix.bundle.nodes.find((n) => n.kind === 'def' && n.id === def) || g;
}

/** The step headings of the program: program line of a `step` block -> { gid, name, text }. */
export function stepHeadings(ix) {
  const out = new Map();
  for (const n of ix.bundle.nodes) {
    if (n.kind !== 'step') continue;
    out.set(n.line, { gid: n.id, name: n.step, text: ix.bundle.steps?.[n.id] || '' });
  }
  return out;
}

/** The defs a line of source calls, from the given set, in the order they appear. */
export function calleesOn(text, defs) {
  const s = text.replace(/"(?:[^"\\]|\\.)*"/g, '""').replace(/#.*$/, '');
  const out = [];
  for (const m of s.matchAll(/([A-Za-z_]\w*)\s*\(/g)) if (defs.has(m[1]) && !out.includes(m[1])) out.push(m[1]);
  if (/^\s*def\s/.test(s)) out.shift();
  return out;
}

/**
 * The line where `caller` calls `callee`: the recorded one when the run
 * recorded lines in the caller, else the first line of the caller's source
 * that calls it, else the caller's head.
 */
export function callSite(ix, caller, callee, recorded = null) {
  if (recorded) return recorded;
  const g = defNode(ix, caller);
  const [a, b] = g && g.line > 0 ? [g.line, g.end] : [1, ix.programLines.length];
  const want = new Set([callee]);
  for (let x = a; x <= b; x++) {
    if (caller === '<main>' && defAt(ix, x) !== '<main>') continue;
    if (calleesOn(ix.programLines[x - 1] || '', want).length) return x;
  }
  return g && g.line > 0 ? g.line : null;
}

/** Where the code pane should be for a frame: the def it shows and the line it marks (null at a call: the def's head). */
export function codeTarget(frame, ix) {
  if (frame.kind === 'start' || frame.kind === 'end' || frame.kind === 'group') return { def: '<main>', line: null };
  return { def: frame.def, line: lineOf(frame, ix) };
}

// ---------------------------------------------------------------------------
// Frames: the run as the camera's stops
// ---------------------------------------------------------------------------

/**
 * The run as an ordered list of frames. A frame is one thing the camera
 * shows: a call (entering a def), a step, a line, a loop (all its passes), a
 * pass, the return to the caller, the module setup, and the end.
 *
 *   loops: 'one'     each iteration loop is walked once (its first pass), with
 *                    a counter of how many passes it made
 *          'notable' the first pass and the passes worth reading (explore.js)
 *          'every'   every pass
 *
 * Calls of the same def made one after another from one place (a loop without
 * steps calling enorm per column) are one visit with a count. Each frame has
 * `depth`, the number of calls open around it (the root is 1); a call frame
 * is inside its callee, a ret frame back in the caller.
 */
export function buildFrames(tree, ix, opts = {}) {
  const loops = opts.loops || 'one';
  const cap = opts.cap || FRAME_CAP;
  const frames = [];
  let truncated = false;
  const lastLine = new Map(); // call node key -> last line seen in it
  const viaOf = new Map(); // call node key -> { caller, line }: where the call came from
  const push = (f) => {
    if (frames.length >= cap) {
      truncated = true;
      return false;
    }
    frames.push(f);
    return true;
  };
  const stepGid = (n) => {
    for (let m = n; m; m = m.parent) {
      if (m.kind === 'step') return `${m.def}/${m.id}`;
      if (m.kind === 'call') return null;
    }
    return null;
  };
  const passOf = (n) => {
    for (let m = n; m; m = m.parent) {
      if (m.kind === 'iter') return { n: m.n, total: m.parent.children.length, loop: m.parent.graphId || null };
      if (m.kind === 'call') return null;
    }
    return null;
  };
  const visitKids = (n, stack) => {
    const kids = n.children;
    for (let j = 0; j < kids.length; j++) {
      const c = kids[j];
      if (c.kind === 'call') {
        let k = j;
        while (k + 1 < kids.length && kids[k + 1].kind === 'call' && kids[k + 1].def === c.def) k++;
        // a line's event comes after its statement ran (debug_lens.rs), so the
        // line a call was made from is the line event that follows the calls
        // of that statement, when there is one
        let m = k + 1;
        while (m < kids.length && kids[m].kind === 'call') m++;
        const site = m < kids.length && kids[m].kind === 'line' ? kids[m].line : null;
        if (!visit(c, stack, k - j + 1, site)) return false;
        j = k;
      } else if (!visit(c, stack, 1)) return false;
    }
    return true;
  };
  const visit = (n, stack, count, site = null) => {
    const call = stack[stack.length - 1];
    const base = { def: call.def, depth: stack.length, stack, node: n, step: stepGid(n), pass: passOf(n), via: viaOf.get(call.key) || null };
    if (n.kind === 'call') {
      const from = site ?? lastLine.get(call.key) ?? null;
      const inner = [...stack, n];
      const via = { caller: call.def, line: from, count };
      viaOf.set(n.key, via);
      if (!push({ ...base, kind: 'call', def: n.def, depth: inner.length, stack: inner, via, count, step: null, pass: null })) return false;
      if (!visitKids(n, inner)) return false;
      return push({ ...base, kind: 'ret', from: n.def, line: from, count, back: via });
    }
    if (n.kind === 'step') {
      if (!push({ ...base, kind: 'step', gid: `${n.def}/${n.id}` })) return false;
      return visitKids(n, stack);
    }
    if (n.kind === 'line') {
      lastLine.set(call.key, n.line);
      return push({ ...base, kind: 'line', line: n.line, values: n.values });
    }
    if (n.kind === 'loop') {
      const total = n.children.length;
      if (!push({ ...base, kind: 'loop', gid: n.graphId || null, total })) return false;
      const pick = n.children.filter((it) => loops === 'every' || it.n === 1 || (loops === 'notable' && n.notable && n.notable.has(it.n)));
      for (const it of pick) {
        if (!push({ ...base, kind: 'iter', node: it, gid: n.graphId || null, pass: { n: it.n, total, loop: n.graphId || null }, shown: pick.length })) return false;
        if (!visitKids(it, stack)) return false;
      }
      return true;
    }
    if (n.kind === 'iter') return visitKids(n, stack);
    if (n.kind === 'group') return push({ ...base, kind: 'group', modules: n.modules || [] });
    return true;
  };
  push({ kind: 'start', def: tree.def, depth: 1, stack: [tree], node: tree, step: null, pass: null });
  visitKids(tree, [tree]);
  if (!truncated) frames.push({ kind: 'end', def: tree.def, depth: 0, stack: [tree], node: tree, step: null, pass: null });
  return { frames, truncated, via: viaOf };
}

/** Step over: the next frame that is not inside a call made from here. */
export function nextOver(frames, i) {
  const d = frames[i].kind === 'call' ? frames[i].depth : frames[i].depth;
  for (let j = i + 1; j < frames.length; j++) if (frames[j].depth <= d) return j;
  return frames.length - 1;
}

/** Step out: the next frame back in the caller. */
export function nextOut(frames, i) {
  const d = frames[i].depth;
  for (let j = i + 1; j < frames.length; j++) if (frames[j].depth < d) return j;
  return frames.length - 1;
}

/** The line a frame highlights, as a program line, or null. */
export function lineOf(frame, ix) {
  if (frame.kind === 'line' || frame.kind === 'ret') return frame.line ?? null;
  if ((frame.kind === 'step' || frame.kind === 'loop' || frame.kind === 'iter') && frame.gid) {
    const g = ix.nodes.get(frame.gid);
    return g ? g.line : null;
  }
  return null;
}

// ---------------------------------------------------------------------------
// The camera
// ---------------------------------------------------------------------------

/** The whole graph, with a margin. */
export function wholeRect(world) {
  const b = world.bounds;
  return { x: b.x - 30, y: b.y - 30, w: b.w + 60, h: b.h + 60 };
}

/**
 * The rect of the graph a frame is about: the box of the def that runs, with
 * room around it for its neighbors' edges. On a narrow screen (vw < 480 px)
 * the room is less, so the box's text stays readable.
 */
export function targetRect(frame, world, ix, vw = 1000) {
  if (frame.kind === 'start' || frame.kind === 'end') return wholeRect(world);
  return defRect(world.defs.get(frame.def), world, vw);
}

/** The rect that frames one def's box. */
export function defRect(d, world, vw = 1000) {
  if (!d) return wholeRect(world);
  const padX = vw < 480 ? 24 : d.w * 0.55;
  const padY = vw < 480 ? 24 : 70;
  return { x: d.x - padX, y: d.y - padY, w: d.w + 2 * padX, h: d.h + 2 * padY };
}

/** The view [cx, cy, w] (center, and view width in world units) that fits a rect in a vw x vh viewport. */
export function fitView(rect, vw, vh, margin = 0.08) {
  const w = Math.max(rect.w, (rect.h * vw) / vh) * (1 + 2 * margin);
  return [rect.x + rect.w / 2, rect.y + rect.h / 2, w];
}

/**
 * Smooth and efficient zooming and panning (J. J. van Wijk and W. A. A. Nuij,
 * IEEE InfoVis 2003): the path between views a = [ux0, uy0, w0] and
 * b = [ux1, uy1, w1] that zooms out, pans and zooms in, minimizing perceived
 * motion. rho trades zoom against pan (sqrt(2), the paper's choice and
 * d3-zoom's). Returns t -> view for t in [0, 1], with `S`, the path's length
 * in the paper's units: a flight should take time proportional to S.
 */
export function interpolateZoom(a, b, rho = Math.SQRT2) {
  const [ux0, uy0, w0] = a;
  const [ux1, uy1, w1] = b;
  const dx = ux1 - ux0;
  const dy = uy1 - uy0;
  const d2 = dx * dx + dy * dy;
  const rho2 = rho * rho;
  const rho4 = rho2 * rho2;
  let f;
  let S;
  if (d2 < 1e-12) {
    // a pure zoom
    S = Math.log(w1 / w0) / rho;
    f = (t) => [ux0 + t * dx, uy0 + t * dy, w0 * Math.exp(rho * t * S)];
  } else {
    const d1 = Math.sqrt(d2);
    const b0 = (w1 * w1 - w0 * w0 + rho4 * d2) / (2 * w0 * rho2 * d1);
    const b1 = (w1 * w1 - w0 * w0 - rho4 * d2) / (2 * w1 * rho2 * d1);
    const r0 = Math.log(Math.sqrt(b0 * b0 + 1) - b0);
    const r1 = Math.log(Math.sqrt(b1 * b1 + 1) - b1);
    S = (r1 - r0) / rho;
    const c0 = Math.cosh(r0);
    const s0 = Math.sinh(r0);
    f = (t) => {
      if (t >= 1) return [ux1, uy1, w1];
      const s = t * S;
      const u = (w0 / (rho2 * d1)) * (c0 * Math.tanh(rho * s + r0) - s0);
      return [ux0 + u * dx, uy0 + u * dy, (w0 * c0) / Math.cosh(rho * s + r0)];
    };
  }
  f.S = Math.abs(S);
  return f;
}

/** Ease in and out (cubic), as d3's transitions do by default. */
export function ease(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

/** How long a flight takes, in ms at speed 1: proportional to the path length, within bounds. */
export function flightMs(S) {
  return Math.max(260, Math.min(1500, 520 * S));
}

/** How long the camera rests on a frame while playing, in ms at speed 1. */
export function dwellMs(frame) {
  return { start: 900, end: 900, call: 450, ret: 250, step: 650, line: 380, loop: 800, iter: 500, group: 600 }[frame.kind] || 400;
}

// ---------------------------------------------------------------------------
// The run's call graph in the atlas's layout
// ---------------------------------------------------------------------------

const NOT_BUILTIN = new Set(['if', 'for', 'while', 'return', 'def', 'match', 'and', 'or', 'not', 'in', 'step', 'parallel', 'else']);

/**
 * The builtins a def's lines call, with counts: names the compiler's builtin
 * list holds, called on a code line of the def. The top level is the root
 * file's lines outside every def; the modules it uses are not its lines, and a
 * `###` doc block or a comment is prose, not code.
 */
export function builtinsOf(ix, def) {
  const g = def === '<main>' ? null : defNode(ix, def);
  if (!ix._allDefs) ix._allDefs = new Set(ix.bundle.nodes.filter((n) => n.kind === 'def').map((n) => n.id));
  if (!ix._proseLines) {
    ix._proseLines = new Set();
    let inDoc = false;
    ix.programLines.forEach((t, k) => {
      const fence = /^\s*###\s*$/.test(t || '');
      if (fence || inDoc) ix._proseLines.add(k + 1);
      if (fence) inDoc = !inDoc;
    });
  }
  const out = {};
  const rows = g && g.line > 0
    ? range(g.line + 1, g.end)
    : range(1, ix.programLines.length).filter((x) => defAt(ix, x) === '<main>' && (ix.toModule(x)?.moduleIndex ?? 0) === 0);
  for (const x of rows) {
    if (ix._proseLines.has(x)) continue;
    const s = (ix.programLines[x - 1] || '').replace(/"(?:[^"\\]|\\.)*"/g, '""').replace(/#.*$/, '');
    for (const m of s.matchAll(/(?<![\w.])([A-Za-z_][\w.]*)\s*\(/g)) {
      const w = m[1];
      if (NOT_BUILTIN.has(w) || ix._allDefs.has(w) || !VIV_BUILTINS.has(w)) continue;
      out[w] = (out[w] || 0) + 1;
    }
  }
  return out;
}

/**
 * The run's call graph as a code map (the shape lib/codemap.js lays out, the
 * atlas's): a def per box the run entered, its module, its call distance
 * from the top level (the walk's layer), its lines and the calls the run made
 * from it. `id` of the top level is '<main>', as in the walk.
 */
export function walkMap(ix, world) {
  const modules = world.modules.map((m) => ({ path: `${m}.viv` }));
  const out = new Map();
  for (const e of world.edges) {
    if (!out.has(e.from)) out.set(e.from, []);
    out.get(e.from).push([e.to, callSite(ix, e.from, e.to), 0, false, e.n]);
  }
  const defs = [...world.defs.values()].map((d) => {
    const g = d.def === '<main>' ? null : defNode(ix, d.def);
    const line = g && g.line > 0 ? g.line : 0;
    return {
      id: d.def, main: d.def === '<main>', module: Math.max(0, world.modules.indexOf(d.module)), dist: d.layer,
      line, end: g && g.end > 0 ? g.end : line, loops: [], calls: out.get(d.def) || [],
      intent: d.intent, tex: d.tex, doc: '', params: g?.params || [], count: d.calls,
    };
  });
  return { entry: '<main>', modules, defs };
}
