// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The explorer's model of one recorded run (sections 9
// and 11). Pure functions, no DOM: scripts/explore-browsertest.mjs and the
// page share them.
//
//   parseRecording(out)            the run's output -> { events, total, printed, ... }
//   buildTree(events, bundle)      events -> a tree of calls, steps, loops, iterations, lines
//   stopsOf(tree, view)            the positions the reader moves through, given what is open
//   valuesAt(stop, events)         what the watched variables hold at a position
//
// A position is a "stop": the start of a call, a step, a loop (all of its
// passes at once), one iteration, or one recorded line. Loops are closed by
// default, so one move goes past a loop however many times it ran; opening
// one shows its first pass line by line and then only the passes where
// something changed.

/** The run's output, split: the story's events, what the program printed, the counts. */
export function parseRecording(out) {
  if (out.startsWith('error:')) return { error: out.slice(6).trim() };
  const lines = out.split('\n');
  const storyLine = lines.find((l) => l.startsWith('__story__ '));
  if (!storyLine) return { error: 'the run printed no story' };
  const story = JSON.parse(storyLine.slice('__story__ '.length));
  const ex = lines.find((l) => l.startsWith('__explore__ '));
  const counts = ex ? JSON.parse(ex.slice('__explore__ '.length)) : { fails: 0, holds: 0 };
  const printed = lines.filter((l) => !l.startsWith('__')).join('\n').replace(/\n+$/, '');
  return { events: story.events, total: story.total, printed, fails: counts.fails, holdFails: counts.holds, bytes: storyLine.length };
}

/** A number as a person reads it: ten significant digits, trailing zeros off (the story's own rule). */
export function numText(x) {
  if (typeof x === 'string') return x;
  if (typeof x === 'boolean') return String(x);
  if (!Number.isFinite(x)) return String(x);
  if (x === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(x)));
  const trim = (s) => (s.includes('.') ? s.replace(/0+$/, '').replace(/\.$/, '') : s);
  if (e < -4 || e >= 10) {
    const [m, ex] = x.toExponential(9).split('e');
    const n = Number(ex);
    return `${trim(m)}e${n < 0 ? '-' : '+'}${String(Math.abs(n)).padStart(2, '0')}`;
  }
  return trim(x.toFixed(Math.max(0, 9 - e)));
}

/** Program line -> { module, line } via the bundle's runs. */
export function lineMapper(bundle) {
  const runs = bundle.lineMap;
  return (x) => {
    let lo = 0;
    let hi = runs.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const [start, mod, mline, count] = runs[mid];
      if (x < start) hi = mid - 1;
      else if (x >= start + count) lo = mid + 1;
      else return { module: bundle.modules[mod], moduleIndex: mod, line: mline + (x - start) };
    }
    return null;
  };
}

/** Everything the page looks up in the bundle, indexed once. */
export function indexBundle(bundle) {
  const nodes = new Map(bundle.nodes.map((n) => [n.id, n]));
  const kids = new Map();
  for (const n of bundle.nodes) {
    if (!n.parent) continue;
    if (!kids.has(n.parent)) kids.set(n.parent, []);
    kids.get(n.parent).push(n);
  }
  const rows = new Map();
  for (const r of bundle.lines) rows.set(`${r.def}:${r.line}`, r);
  const details = new Map();
  for (const d of bundle.details || []) {
    if (!details.has(d.def)) details.set(d.def, []);
    details.get(d.def).push(d);
  }
  const lineDefs = new Set(bundle.lineDefs || []);
  const debugAt = new Map();
  for (const d of bundle.debug || []) {
    const k = `${d.def}/${d.at}`;
    if (!debugAt.has(k)) debugAt.set(k, []);
    debugAt.get(k).push(d);
  }
  const programLines = bundle.program.split('\n');
  // the digits lens (viv digits), when the bundle has one:
  // its lines by `file.viv:line`
  const digitLines = new Map(((bundle.digits && bundle.digits.lines) || []).map((l) => [l.at, l]));
  return { bundle, nodes, kids, rows, details, lineDefs, debugAt, programLines, toModule: lineMapper(bundle), digitLines };
}

// ---------------------------------------------------------------------------
// The tree
// ---------------------------------------------------------------------------

/**
 * The run as a tree. Node kinds: `call` (the root is the call of `<main>`),
 * `step`, `loop` (the passes of one iteration loop), `iter`, `line`, and
 * `group` (top-level lines the used modules ran while they were set up). Each
 * node has `key` (unique), `i` (the event it starts at) and `end`.
 */
export function buildTree(events, ix) {
  let key = 0;
  const mk = (o, parent) => ({ key: key++, children: [], watch: [], holds: [], branches: [], parent, ...o });
  const root = mk({ kind: 'call', def: '<main>', i: 0 }, null);
  const stack = [root];
  const top = () => stack[stack.length - 1];
  const closeTo = (pred, i) => {
    for (let k = stack.length - 1; k > 0; k--) {
      if (pred(stack[k])) {
        for (let j = stack.length - 1; j >= k; j--) stack[j].end = i;
        stack.length = k;
        return;
      }
    }
  };
  events.forEach((e, i) => {
    const [kind, def] = e;
    // a def the intent explains line by line without steps records no call
    // events: its lines stand in an implicit call, closed by the next event
    // from anywhere else
    while (stack.length > 1 && top().implicit && def !== top().def && kind !== 'call') {
      top().end = i;
      stack.pop();
    }
    if (kind === 'watch' && ix && ix.lineDefs.has(def) && e[2].startsWith('line:') && callOf(top()).def !== def) {
      const n = mk({ kind: 'call', def, i, implicit: true }, top());
      top().children.push(n);
      stack.push(n);
    }
    const p = top();
    if (kind === 'call') {
      const n = mk({ kind: 'call', def, i }, p);
      p.children.push(n);
      stack.push(n);
    } else if (kind === 'ret') {
      closeTo((n) => n.kind === 'call' && n.def === def, i);
    } else if (kind === 'enter') {
      const n = mk({ kind: 'step', def, id: e[2], depth: e[3], i }, p);
      p.children.push(n);
      stack.push(n);
    } else if (kind === 'exit') {
      closeTo((n) => n.kind === 'step' && n.def === def && n.id === e[2], i);
    } else if (kind === 'iter') {
      const nIter = Number(e[2]);
      let loop = p.children[p.children.length - 1];
      if (!(loop && loop.kind === 'loop' && loop.def === def && nIter === loop.children.length + 1 && nIter > 1)) {
        loop = mk({ kind: 'loop', def, i }, p);
        p.children.push(loop);
      }
      const it = mk({ kind: 'iter', def, n: nIter, i }, loop);
      loop.children.push(it);
      stack.push(it);
    } else if (kind === 'iter_end') {
      closeTo((n) => n.kind === 'iter' && n.def === def, i);
    } else if (kind === 'watch') {
      const at = e[2];
      if (at.startsWith('line:')) {
        p.children.push(mk({ kind: 'line', def, line: Number(at.slice(5)), values: e[3] || {}, i, end: i }, p));
      } else if (at.startsWith('br:')) {
        const [, l, b] = at.split(':');
        p.branches.push({ line: Number(l), taken: b === '1', i });
      } else {
        p.watch.push({ at, values: e[3] || {}, i });
      }
    } else if (kind === 'hold') {
      p.holds.push({ at: e[2], ok: e[3] === true, sentence: e[4], worst: e[5], i });
    }
  });
  for (let j = stack.length - 1; j >= 0; j--) if (stack[j].end === undefined) stack[j].end = events.length;
  const fix = (n) => {
    if (n.kind === 'loop') n.end = n.children.length ? n.children[n.children.length - 1].end : n.i;
    for (const c of n.children) fix(c);
  };
  fix(root);
  if (ix) {
    groupModuleLines(root, ix);
    for (const l of allOf(root, 'loop')) {
      l.graphId = loopGraphId(l, ix);
      l.notable = notableIterations(l, ix);
    }
  }
  return root;
}

/** Every node of a kind, in order. */
export function allOf(root, kind) {
  const out = [];
  const go = (n) => {
    if (n.kind === kind) out.push(n);
    for (const c of n.children) go(c);
  };
  go(root);
  return out;
}

/** Runs of top-level lines that belong to used modules (their constants and tables) become one group each. */
function groupModuleLines(n, ix) {
  if (n.kind === 'call' && n.def === '<main>') {
    const out = [];
    for (const c of n.children) {
      const m = c.kind === 'line' ? ix.toModule(c.line) : null;
      const prev = out[out.length - 1];
      if (m && m.moduleIndex !== 0) {
        if (prev && prev.kind === 'group') {
          prev.children.push(c);
          if (!prev.modules.includes(m.module)) prev.modules.push(m.module);
          c.parent = prev;
          prev.end = c.i;
          continue;
        }
        const g = { key: `g${c.key}`, kind: 'group', def: '<main>', modules: [m.module], i: c.i, end: c.i, children: [c], watch: [], holds: [], branches: [], parent: n };
        c.parent = g;
        out.push(g);
        continue;
      }
      out.push(c);
    }
    n.children = out;
  }
  for (const c of n.children) groupModuleLines(c, ix);
}

/** The graph's box for a recorded loop: the outermost loop around its first step, or its def's first iteration loop. */
function loopGraphId(loop, ix) {
  const first = (n) => {
    for (const c of n.children) {
      if (c.kind === 'step' && c.def === loop.def) return c;
      const d = c.kind === 'call' ? null : first(c);
      if (d) return d;
    }
    return null;
  };
  const s = first(loop);
  if (s) {
    let g = ix.nodes.get(`${s.def}/${s.id}`);
    let found = null;
    while (g && g.parent && g.def === loop.def) {
      if (g.kind === 'loop') found = g.id;
      g = ix.nodes.get(g.parent);
    }
    if (found) return found;
  }
  const row = ix.bundle.lines.find((r) => r.def === loop.def && r.kind === 'iterloop');
  return row ? `${loop.def}/loop@${row.line}` : null;
}

/**
 * The path of an iteration at the loop's own level: which of its steps ran and
 * which way each of their branches went. Nested loops and calls are left out,
 * since how often an inner loop runs, and the paths inside it, change from pass
 * to pass for ordinary reasons (lmpar's iteration count, a retried trial step).
 */
/** The condition written on a branch's line, e.g. `ratio <= LM_RATIO_SHRINK`, or null. */
function conditionAt(line, ix) {
  const src = ix?.programLines?.[line - 1];
  const m = src && src.match(/^\s*(?:\}\s*else\s+)?(?:if|while)\s+(.*?)\s*\{?\s*(?:#.*)?$/);
  return m ? m[1] : null;
}

function pathParts(it, ix) {
  const parts = new Map();
  const go = (n, where) => {
    for (const b of n.branches) {
      const cond = conditionAt(b.line, ix);
      const what = cond ? `\`${cond}\` was ${b.taken ? 'true' : 'false'}` : `a branch went the ${b.taken ? 'true' : 'false'} way`;
      parts.set(`b${b.line}${b.taken ? 't' : 'f'}`, `${what} in ${where}`);
    }
    for (const c of n.children) {
      if (c.kind === 'step') {
        parts.set(`s${c.id}`, `step ${c.id} ran`);
        go(c, `step ${c.id}`);
      }
    }
  };
  go(it, 'the loop body');
  return parts;
}

function signature(parts) {
  return [...parts.keys()].sort().join(' ');
}

function failedHolds(n) {
  const out = [];
  const go = (m) => {
    for (const h of m.holds) if (!h.ok) out.push(h);
    for (const c of m.children) go(c);
  };
  go(n);
  return out;
}

/** The last value of each watched name in an iteration (numbers and booleans). */
export function iterValues(it) {
  const v = {};
  const go = (n) => {
    const evs = [...n.watch.map((w) => ({ i: w.i, values: w.values })), ...n.children.filter((c) => c.kind === 'line').map((c) => ({ i: c.i, values: c.values }))].sort((a, b) => a.i - b.i);
    for (const w of evs) for (const [k, x] of Object.entries(w.values)) if (typeof x !== 'object') v[k] = x;
    for (const c of n.children) if (c.kind === 'step' || c.kind === 'iter') go(c);
  };
  go(it);
  return v;
}

/**
 * The passes of a loop worth showing, each with why: the first, the last,
 * one where a hold failed, the first pass to take a path no earlier pass took
 * (a branch the other way, a step skipped or added, at the loop's own level),
 * and one where a watched true/false value flipped. A path seen before is not
 * news, so a loop that alternates between two paths shows each once.
 */
export function notableIterations(loop, ix) {
  const out = new Map();
  const note = (it, why) => {
    if (!out.has(it.n)) out.set(it.n, []);
    if (!out.get(it.n).includes(why)) out.get(it.n).push(why);
  };
  const its = loop.children;
  if (!its.length) return out;
  note(its[0], 'the first pass');
  const seen = new Set();
  const seenParts = new Set();
  let prevVals = null;
  for (const it of its) {
    const parts = pathParts(it, ix);
    const sig = signature(parts);
    if (seen.size && !seen.has(sig)) {
      // name what is new; a path that only drops something seen before is new as a combination
      const fresh = [...parts].filter(([k]) => !seenParts.has(k)).map(([, text]) => text);
      note(it, fresh.length ? `first time: ${[...new Set(fresh)].join('; ')}` : 'a combination of branches no earlier pass took');
    }
    seen.add(sig);
    for (const k of parts.keys()) seenParts.add(k);
    if (failedHolds(it).length) note(it, 'a hold failed');
    const vals = iterValues(it);
    if (prevVals) for (const [k, x] of Object.entries(vals)) if (typeof x === 'boolean' && prevVals[k] !== undefined && prevVals[k] !== x) note(it, `${k} became ${x}`);
    prevVals = vals;
  }
  note(its[its.length - 1], 'the last pass');
  return out;
}

/** Which calls to record line by line in a second pass, by def (beyond the first): the last, and any where a hold failed. */
export function secondPassCalls(root) {
  const by = new Map();
  const go = (n) => {
    if (n.kind === 'call' && n.parent && !n.implicit) {
      if (!by.has(n.def)) by.set(n.def, []);
      by.get(n.def).push(n);
    }
    for (const c of n.children) go(c);
  };
  go(root);
  const out = {};
  for (const [d, calls] of by) {
    if (calls.length < 2) continue;
    const pick = new Set([calls.length]);
    calls.forEach((c, k) => {
      if (failedHolds(c).length) pick.add(k + 1);
    });
    out[d] = [...pick].filter((k) => k > 1).sort((a, b) => a - b).slice(0, 4);
  }
  return out;
}

/** Which iterations to record line by line in a second pass, by def (beyond the first). */
export function secondPassIters(root) {
  const by = {};
  for (const loop of allOf(root, 'loop')) {
    for (const n of loop.notable.keys()) {
      if (n <= 1) continue;
      (by[loop.def] ||= new Set()).add(n);
    }
  }
  const out = {};
  // at most six passes per def: the last, and the earliest of the others
  for (const [d, s] of Object.entries(by)) {
    const ns = [...s].sort((a, b) => a - b);
    const last = ns[ns.length - 1];
    out[d] = [...new Set([...ns.slice(0, 5), last])];
  }
  return out;
}

/** Whether a node's lines were recorded (it holds a line, directly or in a step). */
export function hasLines(n) {
  return n.children.some((c) => c.kind === 'line' || ((c.kind === 'step' || c.kind === 'loop' || c.kind === 'iter') && hasLines(c)));
}

// ---------------------------------------------------------------------------
// Positions
// ---------------------------------------------------------------------------

/**
 * The stops, in order, given what the reader opened and closed.
 * view = { open: Set of keys, closed: Set of keys, shown: Set of iteration keys }.
 * Calls and steps are open unless closed; loops, groups and iterations other
 * than the first are closed unless opened. An open loop shows its first pass,
 * its notable passes and any the reader asked for.
 */
export function stopsOf(root, view) {
  const out = [];
  const isOpen = (n) => {
    if (view.closed.has(n.key)) return false;
    if (view.open.has(n.key)) return true;
    if (n.kind === 'call' || n.kind === 'step') return true;
    if (n.kind === 'iter') return n.n === 1;
    return false;
  };
  const go = (n, depth) => {
    out.push({ node: n, depth });
    if (n.kind === 'line') return;
    if (!isOpen(n)) return;
    if (n.kind === 'loop') {
      for (const it of n.children) {
        if (it.n === 1 || n.notable.has(it.n) || view.shown.has(it.key)) go(it, depth + 1);
      }
      return;
    }
    for (const c of n.children) go(c, depth + 1);
  };
  go(root, 0);
  return out;
}

/** The ancestors of a node, outermost first, the node last. */
export function pathOf(n) {
  const out = [];
  for (let m = n; m; m = m.parent) out.unshift(m);
  return out;
}

/** The innermost call around a node (the root for top-level code). */
export function callOf(n) {
  for (let m = n; m; m = m.parent) if (m.kind === 'call') return m;
  return null;
}

/** The innermost step around a node, or null. */
export function stepOf(n) {
  for (let m = n; m; m = m.parent) {
    if (m.kind === 'step') return m;
    if (m.kind === 'call') return null;
  }
  return null;
}

/** The index of a node's stop, or of its nearest ancestor that has one. */
export function indexOfNode(stops, node) {
  for (let m = node; m; m = m.parent) {
    const k = stops.findIndex((s) => s.node === m);
    if (k >= 0) return k;
  }
  return 0;
}

/** The view that makes a node a stop: its loops opened, its iteration shown. */
export function reveal(view, node) {
  const open = new Set(view.open);
  const closed = new Set(view.closed);
  const shown = new Set(view.shown);
  for (let m = node.parent; m; m = m.parent) {
    closed.delete(m.key);
    if (m.kind === 'loop' || m.kind === 'group') open.add(m.key);
    if (m.kind === 'iter') {
      shown.add(m.key);
      if (m.n !== 1) open.add(m.key);
    }
  }
  if (node.kind === 'iter') shown.add(node.key);
  return { open, closed, shown };
}

/** The graph node a tree node stands at: its step, its loop's box, or its def. */
export function graphIdOf(n, ix) {
  if (!n) return null;
  if (n.kind === 'step') return `${n.def}/${n.id}`;
  if (n.kind === 'loop') return n.graphId || n.def;
  if (n.kind === 'iter') return n.parent.graphId || n.def;
  if (n.kind === 'line') {
    const row = ix.rows.get(`${n.def}:${n.line}`);
    if (row && row.step) return `${n.def}/${row.step}`;
    const up = n.parent;
    return up && up.kind !== 'call' && up.kind !== 'group' ? graphIdOf(up, ix) : n.def;
  }
  if (n.kind === 'group') return '<main>';
  return n.def;
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

/**
 * What each watched variable of the current call holds at a stop: the last
 * value recorded for it, from the call's start up to and including the stop's
 * own event. `changed` are the names the stop's own line assigned.
 */
export function valuesAt(node, events) {
  const call = callOf(node) || node;
  const def = call.def;
  const upTo = node.kind === 'line' ? node.i : node.kind === 'loop' || node.kind === 'iter' || node.kind === 'step' ? node.end : node.i;
  const vars = new Map();
  let depth = 0;
  for (let i = call.i; i <= Math.min(upTo, events.length - 1); i++) {
    const e = events[i];
    if (i > call.i && e[0] === 'call' && e[1] === def) depth++;
    if (e[0] === 'ret' && e[1] === def && depth > 0) depth--;
    if (depth > 0 || e[0] !== 'watch' || e[1] !== def || e[2].startsWith('br:')) continue;
    for (const [k, v] of Object.entries(e[3] || {})) vars.set(k, { value: v, i, at: e[2] });
  }
  const changed = node.kind === 'line' ? Object.keys(node.values) : [];
  return { vars, changed };
}

/** A scalar's history: per pass of the innermost loop around the node, or along the call. */
export function historyOf(name, node, events) {
  let loop = null;
  for (let m = node; m; m = m.parent) {
    if (m.kind === 'loop') {
      loop = m;
      break;
    }
    if (m.kind === 'call') break;
  }
  if (node.kind === 'loop') loop = node;
  if (loop) {
    const pts = [];
    for (const it of loop.children) {
      const v = iterValues(it)[name];
      if (typeof v === 'number' || (typeof v === 'string' && /nan|inf/.test(v))) pts.push({ x: it.n, y: Number(v), key: it.key });
    }
    const cur = node.kind === 'iter' ? node.n : (() => { for (let m = node; m; m = m.parent) if (m.kind === 'iter' && m.parent === loop) return m.n; return null; })();
    return { by: 'pass', pts, cur };
  }
  const call = callOf(node);
  const pts = [];
  for (let i = call.i; i < Math.min(call.end, events.length); i++) {
    const e = events[i];
    if (e[0] === 'watch' && e[1] === call.def && e[3] && typeof e[3][name] === 'number') pts.push({ x: pts.length + 1, y: e[3][name], i });
  }
  const cur = pts.filter((p) => p.i <= node.i).length || null;
  return { by: 'assignment', pts, cur };
}

/**
 * What the page calls a def: its name, or for a gradient def the compiler
 * generated for `value_and_grad` (lang/viv/src/grad.rs), "gradient of f".
 */
export function defLabel(ix, def) {
  const n = ix && ix.nodes && ix.nodes.get(def);
  return (n && n.kind === 'def' && n.label) || def;
}

/** A short label for a stop. */
export function labelOf(n, ix) {
  if (n.kind === 'call') return n.def === '<main>' && !n.parent ? 'the start of the run' : n.implicit ? `in ${defLabel(ix, n.def)} (its first call, line by line)` : `call ${defLabel(ix, n.def)}`;
  if (n.kind === 'step') return ix.bundle.steps[`${n.def}/${n.id}`] || n.id;
  if (n.kind === 'loop') {
    const g = n.graphId && ix.nodes.get(n.graphId);
    return `${g ? g.header : 'loop'}: ${n.children.length} ${n.children.length === 1 ? 'pass' : 'passes'}`;
  }
  if (n.kind === 'iter') return `pass ${n.n} of ${n.parent.children.length}`;
  if (n.kind === 'group') return `setting up ${n.modules.length === 1 ? n.modules[0] : `${n.modules.length} modules`}: ${n.children.length} top-level ${n.children.length === 1 ? 'line' : 'lines'}`;
  if (n.kind === 'line') {
    const m = ix.toModule(n.line);
    return m ? `${m.module}.viv line ${m.line}` : `line ${n.line}`;
  }
  return n.kind;
}

/**
 * For a small run, the second pass records everything: every pass of every
 * iteration loop, every call of every def recorded line by line, and every
 * pass of the loops nested in them (lmder's trial steps). The Read view's
 * pass selector then has values for each pass and trial. Returns null when
 * the first recording is too big for that (a long SCF), and the second pass
 * records only the passes worth reading (secondPassIters, secondPassCalls).
 */
export const WHOLE_RUN_EVENTS = 12000;
export const WHOLE_RUN_PASSES = 40;
export function wholeRunRequest(root, events) {
  if (events > WHOLE_RUN_EVENTS) return null;
  const passes = {};
  for (const loop of allOf(root, 'loop')) passes[loop.def] = Math.max(passes[loop.def] || 0, loop.children.length);
  const calls = {};
  const go = (n) => {
    if (n.kind === 'call' && n.parent && !n.implicit) calls[n.def] = (calls[n.def] || 0) + 1;
    for (const c of n.children) go(c);
  };
  go(root);
  if (Object.values(passes).some((k) => k > WHOLE_RUN_PASSES)) return null;
  const upto = (k) => Array.from({ length: Math.max(0, Math.min(k, WHOLE_RUN_PASSES) - 1) }, (_, j) => j + 2);
  const iters = {};
  for (const [d, k] of Object.entries(passes)) if (k > 1) iters[d] = upto(k);
  const callsOut = {};
  for (const [d, k] of Object.entries(calls)) if (k > 1) callsOut[d] = upto(k);
  const every = [...new Set([...Object.keys(passes), ...Object.keys(calls)])];
  return { iters, calls: callsOut, every };
}

/** The digits lens's entry for a line of a module (`{module, line}`), or null. */
export function digitsLine(ix, m) {
  if (!m || !ix.digitLines || !ix.digitLines.size) return null;
  const key = `${m.module}.viv:${m.line}`;
  if (ix.digitLines.has(key)) return ix.digitLines.get(key);
  for (const [at, l] of ix.digitLines) if (at.endsWith(`/${key}`) || key.endsWith(`/${at}`)) return l;
  return null;
}

/** A printed number split where its stable digits stop: [stable, rest], as
 * `viv digits` writes it with a `|` (17 significant digits). */
export function splitDigits(value, stable) {
  const s = Number(value).toExponential(16);
  const [mant, exp] = s.split('e');
  const neg = mant.startsWith('-');
  const ds = mant.replace(/[^0-9]/g, '');
  const keep = stable >= 15.95 ? ds.length : Math.min(ds.length, Math.floor(stable));
  const put = (from, to) => {
    let o = '';
    for (let i = from; i < to; i++) o += (i === 1 ? '.' : '') + ds[i];
    return o;
  };
  const tail = `e${Number(exp)}`;
  const head = put(0, keep);
  return [(neg ? '-' : '') + (keep === ds.length ? head.replace(/0+$/, '').replace(/\.$/, '') : head), put(keep, ds.length).replace(/0+$/, '') + tail];
}
