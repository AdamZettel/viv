// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// What the explorer's program graph shows, and the Viv program that draws it.
//
// The graph (lang/viv/src/graph.rs, in the bundle) has every def the program
// text defines: for the water SCF, 1,033, most of them other methods, other
// bases and reference data. The run's call graph (the interpreter's
// `__graph__` line: each caller and callee with the number of calls) says
// which of them ran. The picture has one box per def that ran, never one per
// call, and opens from a default depth: the defs with steps, the defs on a
// call path to one, and what those call. A reader opens a box to see what it
// calls and closes it again.
//
// This module decides only WHICH defs are boxes. The drawing is made by Viv:
// `drawSource` writes the graph of the shown defs as a `ProgramGraph` literal,
// with a `GraphView` (which boxes hide callees, how many times each call ran,
// where the boxes of the previous drawing were), and a call of lib/graphdraw's
// `draw_view`; viv.wasm runs it (in the page's worker, or at stage time in
// scripts/explore-build.mjs) and the page renders the figure records.
//
// Used by scripts/explore-build.mjs (node) and components/ProgramGraph.jsx.

export const MAIN = '<main>';
/** A program whose run calls at most this many defs opens with all of them. */
export const OPEN_ALL_BELOW = 32;

import { importsOf, pathsOf, stripImports } from './uses.js';

/** The run's calls from the interpreter's output: [[from, to, count], ...], or null. */
export function callsOf(output) {
  const line = String(output || '').split('\n').find((l) => l.startsWith('__graph__ '));
  if (!line) return null;
  const g = JSON.parse(line.slice('__graph__ '.length));
  return g.edges.map((e) => [e.from, e.to, e.count]);
}

/**
 * The model the page works with: the defs, their steps, and the call graph of
 * the run (or, with no run, the calls the text makes from `<main>`).
 */
export function graphModel(bundle) {
  const nodes = bundle.nodes || [];
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const defs = new Map(nodes.filter((n) => n.kind === 'def').map((n) => [n.id, n]));
  const inner = new Map();
  for (const n of nodes) {
    if (n.kind !== 'step' && n.kind !== 'loop') continue;
    if (!inner.has(n.def)) inner.set(n.def, []);
    inner.get(n.def).push(n);
  }
  const counts = new Map();
  const callees = new Map();
  const callers = new Map();
  const link = (a, b) => {
    if (!callees.has(a)) callees.set(a, []);
    if (!callees.get(a).includes(b)) callees.get(a).push(b);
    if (!callers.has(b)) callers.set(b, []);
    if (!callers.get(b).includes(a)) callers.get(b).push(a);
  };
  const ranCalls = bundle.calls || null;
  // the multi-paths (from the bundle's
  // `multipaths`): a stage with several implementations of one intent is one
  // node between the def that uses it (its host) and its members, `mp:<stage>@<host>`.
  // The registry and the drivers the production path passes through to reach
  // the default (`through`) are folded into it: the picture shows the stage
  // and its members, not the sweep machinery.
  const stages = (bundle.multipaths && bundle.multipaths.stages) || [];
  const mps = new Map();
  if (ranCalls) {
    const machinery = new Set();
    for (const s of stages) {
      if (!s.multi || !(s.through || []).length) continue;
      for (const t of s.through) machinery.add(t);
      if (s.registry) machinery.add(s.registry);
    }
    const ranFrom = new Set(ranCalls.map((e) => e[0]));
    // what the machinery called: the members it ran, and how often
    const viaMachinery = new Map();
    const direct = new Map(ranCalls.map(([a, b, c]) => [`${a}>${b}`, c]));
    const folded = new Set();
    for (const s of stages) {
      if (!s.multi) continue;
      for (const h of s.hosts || []) {
        if (!defs.has(h.def) || !ranFrom.has(h.def)) continue;
        const id = `mp:${s.id}@${h.def}`;
        const members = new Map();
        for (const m of s.members) {
          if ((s.through || []).length) {
            const c = ranCalls.filter(([a, b]) => b === m && machinery.has(a)).reduce((t, e) => t + e[2], 0);
            if (c) members.set(m, c);
          } else if (direct.has(`${h.def}>${m}`)) {
            members.set(m, direct.get(`${h.def}>${m}`));
            folded.add(`${h.def}>${m}`);
          }
        }
        if (!members.size && (s.through || []).length && !ranCalls.some(([a, b]) => a === h.def && b === s.through[0])) continue;
        mps.set(id, { id, stage: s, host: h.def, hostStep: h.step || null, members, arch: bundle.multipaths.arch });
      }
    }
    for (const [a, b, c] of ranCalls) {
      if (!defs.has(a) || !defs.has(b)) continue;
      if (machinery.has(a) || machinery.has(b) || folded.has(`${a}>${b}`)) {
        if (machinery.has(a) && !machinery.has(b)) viaMachinery.set(b, (viaMachinery.get(b) || 0) + c);
        continue;
      }
      counts.set(`${a}>${b}`, c);
      link(a, b);
    }
    for (const mp of mps.values()) {
      counts.set(`${mp.host}>${mp.id}`, 1);
      link(mp.host, mp.id);
      for (const [m, c] of mp.members) {
        counts.set(`${mp.id}>${m}`, c);
        link(mp.id, m);
      }
    }
  } else {
    for (const e of bundle.edges || []) {
      if (e.kind !== 'call') continue;
      const from = byId.get(e.from);
      if (from && defs.has(from.def) && defs.has(e.to)) link(from.def, e.to);
    }
  }
  // the defs the run reached, nearest first
  const ran = [];
  const depth = new Map();
  if (defs.has(MAIN)) {
    depth.set(MAIN, 0);
    ran.push(MAIN);
    for (let i = 0; i < ran.length; i++) {
      for (const c of callees.get(ran[i]) || []) {
        if (!depth.has(c)) {
          depth.set(c, depth.get(ran[i]) + 1);
          ran.push(c);
        }
      }
    }
  }
  const ranSet = new Set(ran);
  // the defs with steps that ran, and every def on a call path to one
  const lead = new Set([MAIN]);
  const stepped = ran.filter((d) => (inner.get(d) || []).some((n) => n.kind === 'step'));
  const stack = [...stepped];
  while (stack.length) {
    const d = stack.pop();
    if (lead.has(d) && d !== MAIN) continue;
    lead.add(d);
    for (const p of callers.get(d) || []) if (ranSet.has(p) && !lead.has(p)) stack.push(p);
  }
  // the defs folded into a multi-path (its drivers and registry, and what only
  // they called): each to the multi-paths that stand for it in the picture,
  // so a stop of the walkthrough at one of them shows its multi-path
  const foldedTo = new Map();
  for (const mp of mps.values()) {
    for (const t of [...(mp.stage.through || []), mp.stage.registry].filter(Boolean)) {
      if (!foldedTo.has(t)) foldedTo.set(t, []);
      foldedTo.get(t).push(mp.id);
    }
  }
  if (ranCalls && foldedTo.size) {
    const origCallers = new Map();
    for (const [a, b] of ranCalls) {
      if (!origCallers.has(b)) origCallers.set(b, []);
      origCallers.get(b).push(a);
    }
    for (const d of origCallers.keys()) {
      if (ranSet.has(d) || foldedTo.has(d) || !defs.has(d)) continue;
      const seen = new Set([d]);
      const queue = [...origCallers.get(d)];
      for (let i = 0; i < queue.length; i++) {
        const c = queue[i];
        if (seen.has(c)) continue;
        seen.add(c);
        if (foldedTo.has(c)) {
          foldedTo.set(d, foldedTo.get(c));
          break;
        }
        queue.push(...(origCallers.get(c) || []));
      }
    }
  }
  // who calls whom in the text, for the defs the run did not reach
  const notRun = [...defs.keys()].filter((d) => !ranSet.has(d) && d !== MAIN);
  return { bundle, byId, defs, inner, counts, callees, callers, ran, ranSet, depth, lead, stepped, notRun, fromRun: !!ranCalls, mps, stages, foldedTo };
}

/**
 * The box that shows `id` in the picture: itself, or for a def folded into
 * multi-paths, the one whose host is nearest on `node`'s chain of calls (a
 * walkthrough stop's node; its `parent`s carry `def`), else the first.
 */
export function boxFor(model, id, node = null) {
  const d = String(id).split('/')[0];
  if (model.ranSet.has(d) || !model.foldedTo || !model.foldedTo.has(d)) return id;
  const cands = model.foldedTo.get(d);
  for (let n = node && node.parent; n; n = n.parent) {
    const hit = cands.find((c) => model.mps.get(c) && model.mps.get(c).host === n.def);
    if (hit) return hit;
  }
  return cands[0];
}

/** Whether an id is a multi-path's node or one of its marks (`mp:eri@ao_eri`, `mp:eri@ao_eri/fork`). */
export const isMp = (id) => String(id || '').startsWith('mp:');
/** The multi-path node an id belongs to. */
export const mpOf = (id) => String(id).split('/')[0];

/** What a stage's members must agree on, in a few words: `within 1e-12`. */
export function contractWords(s) {
  if (s.tolerance && Number.isFinite(s.tolerance.value)) return `within ${Number(s.tolerance.value).toExponential(0).replace('e+0', '')}`;
  return String(s.contract || '').replace(/\s*\[(test|program) "[^"]*"\]/g, '').trim();
}

/** How many of a stage's members are independent, and how many of those were seen agreeing. */
export function robustWords(s) {
  const ag = s.agreeing_families;
  return `${s.independent} independent of ${s.members.length}, ${ag === null || ag === undefined ? 'agreement not measured' : `${ag} seen agreeing`}`;
}

/** The defs whose callees are open when the page loads. */
export function defaultOpen(model) {
  if (model.ran.length <= OPEN_ALL_BELOW) return new Set(model.ran);
  return new Set(model.lead);
}

/** The defs drawn as boxes: from `<main>`, the callees of every open def drawn. */
export function shownOf(model, open) {
  const shown = new Set();
  if (!model.defs.has(MAIN)) return shown;
  const queue = [MAIN];
  shown.add(MAIN);
  for (let i = 0; i < queue.length; i++) {
    const d = queue[i];
    if (!open.has(d)) continue;
    for (const c of model.callees.get(d) || []) {
      if (!shown.has(c)) {
        shown.add(c);
        queue.push(c);
      }
    }
  }
  return shown;
}

/** The callees of `d` the picture does not show. */
export function hiddenCallees(model, shown, d) {
  return (model.callees.get(d) || []).filter((c) => !shown.has(c));
}

/** Whether closing the open def `d` would take a box off the picture. */
export function closes(model, open, shown, d) {
  if (!open.has(d) || !(model.callees.get(d) || []).length) return false;
  const without = new Set(open);
  without.delete(d);
  return shownOf(model, without).size < shown.size;
}

/**
 * `open` with what must be opened for `id` (a def, or a step's def) to be
 * drawn: each def on the shortest call path from `<main>` to it. Returns the
 * same set when `id` is shown already, or null when the run never reached it.
 */
export function revealed(model, open, id) {
  const def = String(id).split('/')[0];
  if (!model.ranSet.has(def)) return null;
  const shown = shownOf(model, open);
  if (shown.has(def)) return open;
  // the path by the shallowest callers
  const path = [];
  let d = def;
  while (d !== MAIN) {
    const ps = (model.callers.get(d) || []).filter((p) => model.depth.has(p) && model.depth.get(p) < model.depth.get(d));
    if (!ps.length) return null;
    ps.sort((a, b) => model.depth.get(a) - model.depth.get(b));
    // prefer a caller already drawn
    d = ps.find((p) => shown.has(p)) || ps[0];
    path.push(d);
    if (shown.has(d) && open.has(d)) break;
  }
  const next = new Set(open);
  for (const p of path) next.add(p);
  return next;
}

// ---- the Viv program ----------------------------------------------------------------

const GREEK = { alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ε', varepsilon: 'ε', eta: 'η', theta: 'θ', lambda: 'λ', mu: 'μ', nu: 'ν', pi: 'π', rho: 'ρ', sigma: 'σ', tau: 'τ', phi: 'φ', varphi: 'φ', chi: 'χ', psi: 'ψ', omega: 'ω', Gamma: 'Γ', Delta: 'Δ', Theta: 'Θ', Lambda: 'Λ', Pi: 'Π', Sigma: 'Σ', Phi: 'Φ', Psi: 'Ψ', Omega: 'Ω' };
const SIGNS = { top: 'ᵀ', le: '≤', leq: '≤', ge: '≥', geq: '≥', ne: '≠', neq: '≠', approx: '≈', cdot: '·', times: '×', to: '→', infty: '∞', sum: 'Σ', partial: '∂', nabla: '∇', sqrt: '√', pm: '±', mp: '∓', in: '∈', ldots: '…', dots: '…', cdots: '⋯', min: 'min', max: 'max', log: 'log', exp: 'exp' };

/**
 * A step's text for the picture, which draws plain text: its inline math
 * (`$\phi(\alpha)$`) with the dollar signs gone and the common commands as
 * the characters they stand for.
 */
export function plainText(t) {
  return String(t || '').replace(/\$([^$]*)\$/g, (_, m) =>
    m
      .replace(/\\(left|right)\b/g, '')
      .replace(/\\\|/g, '‖')
      .replace(/\^\{?\\top\}?/g, 'ᵀ')
      .replace(/\\([A-Za-z]+)/g, (w, name) => GREEK[name] || SIGNS[name] || name)
      .replace(/\\[,;! ]/g, ' ')
      .replace(/[{}]/g, ''));
}

const vivStr = (s) =>
  '"' +
  String(s == null ? '' : s).replace(/[\\"\n\r\t]|[\u0000-\u001f\u007f]/g, (c) => {
    if (c === '\\') return '\\\\';
    if (c === '"') return '\\"';
    if (c === '\n') return '\\n';
    if (c === '\r') return '\\r';
    if (c === '\t') return '\\t';
    return `\\u{${c.charCodeAt(0).toString(16)}}`;
  }) +
  '"';
const strs = (xs) => (xs.length ? `[${xs.map(vivStr).join(', ')}]` : 'split("")');
const num = (x) => (Number.isInteger(x) ? String(x) : Number(x).toFixed(3).replace(/\.?0+$/, ''));
const nums = (xs) => (xs.length ? `[${xs.map(num).join(', ')}]` : 'zeros(0)');

/**
 * The Viv text that draws what `open` shows: the lines to append to the
 * drawing library (the bundle's `drawLib`). `hints` are the def boxes of the
 * drawing on the page ({ id, x, y }: left side and top, down from the top);
 * `way` is its direction ('across', 'down'), kept so a redrawing does not turn.
 * Returns { src, shown, nodes } (nodes: how many the literal holds).
 */
export function drawSource(model, open, { hints = [], way = '', title = '' } = {}) {
  const { bundle, defs, inner } = model;
  const shown = shownOf(model, open);
  // the nodes: each drawn def with its steps and loops, then each def they
  // call that is not drawn (named at the foot of its caller's box)
  const hidden = new Set();
  for (const d of shown) for (const c of hiddenCallees(model, shown, d)) hidden.add(c);
  const keep = [];
  for (const n of bundle.nodes) {
    if (n.kind === 'def' && (shown.has(n.id) || hidden.has(n.id))) keep.push(n);
    else if ((n.kind === 'step' || n.kind === 'loop') && shown.has(n.def)) keep.push(n);
  }
  const index = new Map(keep.map((n, k) => [n.id, k]));
  const ownerOf = (n) => (n.kind === 'def' ? n.id : n.def);
  // the edges: calls that ran (from the step that makes them, when the text
  // says which) and the values steps hand each other
  const ek = [];
  const ef = [];
  const et = [];
  const ev = [];
  const ec = [];
  const counts = [];
  const pairs = new Set();
  for (const e of bundle.edges) {
    const a = index.get(e.from);
    const b = index.get(e.to);
    if (a === undefined || b === undefined) continue;
    const from = keep[a];
    if (e.kind === 'call') {
      const pair = `${ownerOf(from)}>${e.to}`;
      if (!shown.has(ownerOf(from)) || !model.callees.get(ownerOf(from))?.includes(e.to)) continue;
      pairs.add(pair);
      counts.push(model.counts.get(pair) || 0);
    } else {
      if (!shown.has(ownerOf(from)) || !shown.has(ownerOf(keep[b]))) continue;
      counts.push(0);
    }
    ek.push(e.kind);
    ef.push(a);
    et.push(b);
    ev.push((e.vars || []).join(' '));
    ec.push(e.carried ? 1 : 0);
  }
  // a call the run made that the text does not show (through a function
  // value, say) leaves from its caller's header
  for (const d of shown) {
    for (const c of model.callees.get(d) || []) {
      const pair = `${d}>${c}`;
      if (pairs.has(pair) || !index.has(c) || !index.has(d)) continue;
      ek.push('call');
      ef.push(index.get(d));
      et.push(index.get(c));
      ev.push('');
      ec.push(0);
      counts.push(model.counts.get(pair) || 0);
    }
  }
  const stepText = (n) => plainText(n.text || (bundle.steps && bundle.steps[n.id]) || '');
  const g = [
    ['name', vivStr(bundle.name || title)],
    ['ids', strs(keep.map((n) => n.id))],
    ['kinds', strs(keep.map((n) => n.kind))],
    ['labels', strs(keep.map((n) => (n.kind === 'step' ? n.step : n.kind === 'loop' ? n.header : n.label || n.id)))],
    ['texts', strs(keep.map((n) => (n.kind === 'step' ? stepText(n) : '')))],
    ['parents', nums(keep.map((n) => (n.parent ? index.get(n.parent) ?? -1 : -1)))],
    ['owners', nums(keep.map((n) => index.get(ownerOf(n)) ?? -1))],
    ['lines', nums(keep.map((n) => n.line || 0))],
    ['ends', nums(keep.map((n) => n.end || 0))],
    ['params', strs(keep.map((n) => (n.params || []).join(' ')))],
    ['reads', strs(keep.map((n) => (n.kind === 'step' ? (n.reads || []).join(' ') : '')))],
    ['writes', strs(keep.map((n) => (n.writes || []).join(' ')))],
    ['globals', strs(keep.map((n) => (n.globals || []).join(' ')))],
    ['calls', strs(keep.map((n) => (n.calls || []).join(' ')))],
    ['edge_kinds', strs(ek)],
    ['edge_from', nums(ef)],
    ['edge_to', nums(et)],
    ['edge_vars', strs(ev)],
    ['edge_carried', nums(ec)],
  ];
  const isDef = (n) => n.kind === 'def';
  const hintIds = hints.filter((h) => shown.has(h.id));
  const v = [
    ['shown', nums(keep.map((n) => (isDef(n) && shown.has(n.id) ? 1 : 0)))],
    ['open', nums(keep.map((n) => (isDef(n) && shown.has(n.id) && closes(model, open, shown, n.id) ? 1 : 0)))],
    ['more', nums(keep.map((n) => (isDef(n) && shown.has(n.id) ? hiddenCallees(model, shown, n.id).length : 0)))],
    ['counts', nums(counts)],
    ['hint_ids', strs(hintIds.map((h) => h.id))],
    ['hint_x', nums(hintIds.map((h) => h.x))],
    ['hint_y', nums(hintIds.map((h) => h.y))],
    ['dir', way === 'down' ? '2' : way === 'across' ? '1' : '0'],
  ];
  // the multi-paths whose host is drawn: closed (one box) or open (a fork, a
  // path per member, a join), as `open` says
  const mpk = [...(model.mps || new Map()).values()].filter((mp) => shown.has(mp.id) && shown.has(mp.host) && index.has(mp.host));
  const mpv = [
    ['ids', strs(mpk.map((mp) => mp.stage.id))],
    ['titles', strs(mpk.map((mp) => plainText(mp.stage.title)))],
    ['contracts', strs(mpk.map((mp) => contractWords(mp.stage)))],
    ['notes', strs(mpk.map((mp) => robustWords(mp.stage)))],
    ['hosts', nums(mpk.map((mp) => index.get(mp.host)))],
    ['host_steps', nums(mpk.map((mp) => (mp.hostStep && index.has(mp.hostStep) ? index.get(mp.hostStep) : -1)))],
    ['members', strs(mpk.map((mp) => mp.stage.members.join(' ')))],
    ['defaults', strs(mpk.map((mp) => mp.stage.default))],
    ['ran', strs(mpk.map((mp) => [...mp.members.keys()].join(' ')))],
    ['open', nums(mpk.map((mp) => (open.has(mp.id) ? 1 : 0)))],
  ];
  const lit = (type, fields) => `${type} {\n${fields.map(([k, x]) => `    ${k}: ${x}`).join(',\n')}\n}`;
  const call = mpk.length
    ? `draw_view_mp(${lit('ProgramGraph', g)}, ${lit('GraphView', v)}, ${lit('MultiPaths', mpv)}, ${vivStr(title || bundle.name || '')})`
    : `draw_view(${lit('ProgramGraph', g)}, ${lit('GraphView', v)}, ${vivStr(title || bundle.name || '')})`;
  const src = `\n# ---- the view the page asked for\n${call}\n`;
  return { src, shown, nodes: keep.length };
}

/** The def boxes of a drawing's records, as hints: { id, x, y }. */
export function hintsOf(records) {
  const out = [];
  for (const line of String(records || '').split('\n')) {
    if (!line.startsWith('box: ')) continue;
    const f = line.slice(5).split(' ; ');
    if (f[0].includes('/')) continue;
    const v = f[2].trim().split(/\s+/).map(Number);
    out.push({ id: f[0], x: Math.min(v[0], v[2]), y: -Math.max(v[1], v[3]) });
  }
  return out;
}

/** The direction a drawing's defs run: its figure's x label. */
export function wayOf(records) {
  const m = String(records || '').match(/^figure: .*? ; (across|down) ;/m);
  return m ? m[1] : '';
}

/** The drawing's own lines of an interpreter's output (not `__graph__` and the like). */
export function drawingOf(output) {
  return String(output || '')
    .split('\n')
    .filter((l) => l && !l.startsWith('__'))
    .map((l) => `${l}\n`)
    .join('');
}

/**
 * The drawing library as one text: lib/graphdraw with the modules it uses
 * spliced in (as an import does), its test blocks left out. `read(name)`
 * returns a module's text.
 */
export function drawLibrary(read) {
  const seen = new Set();
  const parts = [];
  const tryRead = (name) => { try { return read(name); } catch { return null; } };
  const add = (name) => {
    if (seen.has(name)) return;
    seen.add(name);
    const src = read(name);
    for (const d of importsOf(src)) {
      for (const p of pathsOf(d)) {
        if (tryRead(p) !== null) { add(p); break; }
      }
    }
    parts.push(`# ---- #! use ${name}\n${withoutTests(stripImports(src))}`);
  };
  add('lib/graphdraw');
  return parts.join('\n');
}

/** A module's text without its `test "..." { ... }` blocks. */
function withoutTests(src) {
  const out = [];
  let skipping = false;
  for (const line of src.split('\n')) {
    if (!skipping && /^test "/.test(line)) {
      skipping = !/^}\s*$/.test(line);
      continue;
    }
    if (skipping) {
      if (/^}\s*$/.test(line)) skipping = false;
      continue;
    }
    out.push(line);
  }
  return out.join('\n');
}
