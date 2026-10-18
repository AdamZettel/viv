// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The reading view of the explorer: the
// program read like a book, in the order its calls reach each def, one unit at
// a time. Pure functions, no DOM: the page (components/ReadView.jsx) and
// scripts/read-browsertest.mjs share them.
//
//   buildReading(ix)               the bundle -> { sections, units }
//   unitRun(unit, ix, run)         what the recorded run says about a unit
//   fillRun(text, unit, ix, run)   a `run` line with its {name} placeholders filled
//
// A unit is one scroll segment: a def's signature and card, then each of its
// steps in source order, then each line anchor inside a step. A def's section
// begins right after the unit whose lines first call it, so lmpar's section
// starts at lmder's lm_parameter step. Each unit reveals the def's lines up to
// its own last line; the code a reader has seen stays on screen.
import { valuesAt, allOf } from './explore.js';

const MAIN = '<main>';

/** Whether a bundle has intent enough for the reading view to be the default. */
export function hasIntent(bundle) {
  return Object.entries(bundle.intent || {}).some(([d, v]) => d !== MAIN && v.entry && v.what);
}

/** A def the reading view gives a section: the top level, and defs with an .intent entry or steps. */
function explained(ix, def) {
  if (def === MAIN) return true;
  const it = ix.bundle.intent[def];
  const node = ix.nodes.get(def);
  return !!(it && (it.entry || (node && node.stepped)));
}

/** The statement table by def, sorted by line. */
function rowsByDef(bundle) {
  const m = new Map();
  for (const r of bundle.tex || []) {
    if (!m.has(r.def)) m.set(r.def, []);
    m.get(r.def).push(r);
  }
  for (const rs of m.values()) rs.sort((a, b) => a.line - b.line);
  return m;
}

const isBrace = (s) => /^\s*[{}]*\s*$/.test(s || '');

/** A row that only initializes: `x = 0`, `ok = false`, `w = zeros(n)`. */
export function isTrivial(r) {
  return r.kind === 'assign' && /(=|\\gets) (-?\d+(\.\d+)?|\\text\{(true|false)\}|\\mathbf\{0\}_\{[^}]*\})$/.test(r.tex);
}

/**
 * The top level, as the page shows it: the code lines of the root module and
 * of any used module whose top level holds the program's steps (a deck such as
 * dft/pwdft.viv), without the defs, which get sections of their own, and
 * without the test blocks; and the prose of the `###` blocks, in order.
 */
export function mainPieces(ix) {
  const P = ix.programLines;
  const inDef = new Set();
  for (const n of ix.bundle.nodes) if (n.kind === 'def' && n.line > 0) for (let x = n.line; x <= n.end; x++) inDef.add(x);
  const mods = new Set([0]);
  for (const n of ix.bundle.nodes) if (n.kind === 'step' && n.def === MAIN) mods.add(ix.toModule(n.line)?.moduleIndex ?? 0);
  const pieces = [];
  let prose = null;
  let header = true;
  let lastMod = 0;
  for (let x = 1; x <= P.length; x++) {
    const m = ix.toModule(x);
    if (!m || !mods.has(m.moduleIndex) || inDef.has(x)) continue;
    if (m.moduleIndex !== lastMod) {
      header = true;
      lastMod = m.moduleIndex;
    }
    const t = P[x - 1];
    if (/^###\s*$/.test(t)) {
      if (prose) {
        pieces.push({ kind: 'prose', text: cleanProse(prose.lines, prose.header) });
        prose = null;
      } else {
        prose = { lines: [], header };
      }
      header = false;
      continue;
    }
    if (prose) {
      prose.lines.push(t);
      continue;
    }
    // a test block runs to its closing brace
    if (/^test\s+"/.test(t)) {
      let depth = 0;
      for (; x <= P.length; x++) {
        for (const c of P[x - 1]) {
          if (c === '{') depth++;
          else if (c === '}') depth--;
        }
        if (depth <= 0 && P[x - 1].includes('}')) break;
      }
      continue;
    }
    pieces.push({ kind: 'line', line: x });
  }
  return pieces;
}

/** A post's `###` block as prose: the header's `key: value` lines dropped (a key can be two words, `scale large:`). */
function cleanProse(lines, header) {
  let ls = lines;
  if (header) {
    const k = ls.findIndex((l) => !l.trim());
    if (k >= 0 && ls.slice(0, k).every((l) => /^[a-z][\w-]*( [\w-]+)*:/i.test(l.trim()))) ls = ls.slice(k + 1);
  }
  return ls.join('\n').trim();
}

/** The graph's steps of a def, by line. */
function stepsOf(ix, def) {
  return ix.bundle.nodes.filter((n) => n.kind === 'step' && n.def === def).sort((a, b) => a.line - b.line);
}

/** A def's units, before reading order: its card, its steps, its line anchors. */
function defUnits(ix, def, rows) {
  const node = ix.nodes.get(def);
  const lo = node.line;
  let hi = node.end;
  // a def's span can end in blank lines; they are not part of what is read
  while (hi > lo && !(ix.programLines[hi - 1] || '').trim()) hi--;
  const steps = stepsOf(ix, def);
  const dets = ix.details.get(def) || [];
  const anchors = dets.filter((d) => d.label === '=' && d.line !== null && d.line >= lo && d.line <= hi);
  const cuts = [];
  for (const s of steps) cuts.push({ kind: 'step', line: s.line, end: s.end, step: s });
  for (const a of anchors) {
    const enc = a.step ? steps.find((s) => s.step === a.step) : null;
    const row = rows.find((r) => r.line === a.line);
    cuts.push({ kind: 'anchor', line: a.line, end: enc ? enc.end : hi, rowEnd: row ? row.end : a.line, anchor: a, stepId: a.step || null });
  }
  cuts.sort((a, b) => a.line - b.line || (a.kind === 'step' ? -1 : 1));
  const units = [{ kind: 'intro', line: lo, end: hi }, ...cuts];
  let prevEnd = lo - 1;
  units.forEach((u, k) => {
    const next = units[k + 1] ? units[k + 1].line : hi + 1;
    // an anchor's own lines are its statement; a step's run to its end or its first child
    const focusEnd = u.kind === 'anchor' ? Math.min(u.rowEnd, next - 1) : Math.max(u.line, Math.min(u.end, next - 1));
    const from = prevEnd + 1;
    const to = k === units.length - 1 ? hi : focusEnd;
    u.reveal = range(from, to);
    // the unit's own lines: from its first line that is not a lone brace
    const own = range(u.kind === 'intro' ? lo : u.kind === 'anchor' ? u.line : from, focusEnd);
    u.focus = own.filter((x) => !isBrace(ix.programLines[x - 1]) || x === u.line);
    prevEnd = to;
  });
  return { lines: range(lo, hi), units };
}

function range(a, b) {
  const out = [];
  for (let x = a; x <= b; x++) out.push(x);
  return out;
}

/**
 * The top level's units: a prose block and the code under it, each of the
 * top level's steps, and its line anchors; the code alone when there is none
 * of these.
 */
function mainUnits(ix, rows) {
  const pieces = mainPieces(ix);
  const lines = pieces.filter((p) => p.kind === 'line').map((p) => p.line);
  const steps = stepsOf(ix, MAIN);
  const stepAt = new Map(steps.map((s) => [s.line, s]));
  const anchorAt = new Map((ix.details.get(MAIN) || []).filter((d) => d.label === '=' && d.line !== null).map((d) => [d.line, d]));
  const units = [];
  let cur = null;
  const open = (u) => {
    cur = { reveal: [], focus: [], ...u };
    units.push(cur);
  };
  for (const p of pieces) {
    if (p.kind === 'prose') {
      open({ kind: units.length ? 'prose' : 'intro', prose: p.text });
      continue;
    }
    const x = p.line;
    const st = stepAt.get(x);
    const an = anchorAt.get(x);
    if (st) open({ kind: 'step', step: st, line: x, end: st.end });
    else if (an) open({ kind: 'anchor', anchor: an, stepId: an.step || null, line: x, end: (rows.find((r) => r.line === x) || { end: x }).end });
    else if (!cur) open({ kind: 'intro', prose: '' });
    cur.reveal.push(x);
  }
  units.forEach((u, k) => {
    // blank lines at the start of a piece of code are not part of it
    while (u.kind !== 'step' && u.reveal.length && !(ix.programLines[u.reveal[0] - 1] || '').trim()) u.reveal.shift();
    const next = units[k + 1] && units[k + 1].reveal.length ? units[k + 1].reveal[0] : Infinity;
    const own = (x) => (u.kind === 'step' ? x >= u.line && x <= Math.min(u.end, next - 1) : u.kind === 'anchor' ? x >= u.line && x <= u.end : true);
    u.focus = u.reveal.filter((x) => own(x) && !isBrace(ix.programLines[x - 1]) && (ix.programLines[x - 1] || '').trim());
    if (u.kind === 'step' && !u.focus.includes(u.line)) u.focus.unshift(u.line);
    u.line = u.reveal[0] || 0;
  });
  const kept = new Set(units.flatMap((u) => u.reveal));
  return { lines: lines.filter((x) => kept.has(x)), units };
}

/**
 * Split a unit that is not a step after a statement that calls an explained
 * def (the first time), so the callee's section comes right after the call.
 */
function splitAtCalls(ix, def, units, rows, isExplained) {
  const out = [];
  for (const u of units) {
    if (u.kind === 'step' || !u.reveal.length) {
      out.push(u);
      continue;
    }
    let rest = u;
    let guard = 0;
    while (guard++ < 50) {
      const inFocus = rows.filter((r) => rest.focus.includes(r.line));
      const cut = inFocus.find((r) => r.calls.some((c) => c !== def && isExplained(c)) && rest.reveal.indexOf(r.end) < rest.reveal.length - 1 && rest.reveal.includes(r.end));
      if (!cut) break;
      const at = rest.reveal.indexOf(cut.end) + 1;
      const head = { ...rest, reveal: rest.reveal.slice(0, at), focus: rest.focus.filter((x) => x <= cut.end) };
      const tail = { kind: 'cont', reveal: rest.reveal.slice(at), focus: rest.focus.filter((x) => x > cut.end), line: rest.reveal[at] };
      out.push(head);
      rest = tail;
    }
    out.push(rest);
  }
  return out.filter((u) => u.kind !== 'cont' || u.focus.length);
}

/**
 * The program as a book. Sections are defs in the order the reader meets
 * them, each unit a scroll segment.
 */
export function buildReading(ix) {
  const rows = rowsByDef(ix.bundle);
  const isExplained = (d) => explained(ix, d);
  const sections = [];
  const units = [];
  const visited = new Set();
  const glossSeen = new Map();

  const visit = (def, path, from) => {
    if (visited.has(def) || (def !== MAIN && !ix.nodes.get(def))) return;
    visited.add(def);
    const drows = rows.get(def) || [];
    const base = def === MAIN ? mainUnits(ix, drows) : defUnits(ix, def, drows);
    const us = splitAtCalls(ix, def, base.units, drows, isExplained);
    const section = { i: sections.length, def, path: [...path, def], lines: base.lines, units: [], from, depth: path.length };
    sections.push(section);
    const intent = ix.bundle.intent[def] || {};
    const gloss = new Map((intent.glossary || []).map(([k, t]) => [k, t]));
    const symbols = intent.symbols || {};
    if (!glossSeen.has(def)) glossSeen.set(def, new Set());
    const seen = glossSeen.get(def);
    const stepCount = us.filter((u) => u.kind === 'step').length;
    let stepNo = 0;
    for (const u of us) {
      if (u.kind === 'step') stepNo++;
      const focusSet = new Set(u.focus);
      const urows = drows.filter((r) => focusSet.has(r.line));
      const stepId = u.kind === 'step' ? u.step.step : u.kind === 'anchor' ? u.stepId || (u.anchor && u.anchor.step) || null : null;
      const dets = (ix.details.get(def) || []).filter((d) => {
        if (d.label === '=') return false;
        if (u.kind === 'anchor') return d.under === u.anchor.anchor && (d.step || null) === (u.anchor.step || null);
        if (u.kind === 'step') return d.step === u.step.step && !d.under;
        if (u.kind === 'intro') return !d.step && !d.under;
        return false;
      });
      // symbols and glossary lines, at the first unit that uses the name
      const used = new Set();
      if (u.kind === 'intro' && def !== MAIN) for (const p of ix.nodes.get(def)?.params || []) used.add(p);
      // a bare initialization (x = 0) is not where a name is met
      for (const r of urows) if (!isTrivial(r)) for (const n of r.names) used.add(n);
      const fresh = [];
      for (const n of used) {
        if (seen.has(n) || !gloss.has(n)) continue;
        seen.add(n);
        fresh.push({ name: n, symbol: symbols[n] || null, text: gloss.get(n) });
      }
      const stepPath = [];
      if (stepId) {
        for (let g = ix.nodes.get(`${def}/${stepId}`); g && g.kind !== 'def'; g = ix.nodes.get(g.parent)) if (g.kind === 'step') stepPath.unshift(g.step);
      }
      const unit = {
        i: units.length,
        key: `${def}|${u.kind}|${u.kind === 'step' ? u.step.step : u.kind === 'anchor' ? u.anchor.anchor : u.line}|${units.length}`,
        def,
        section: section.i,
        kind: u.kind,
        stepId,
        stepPath,
        stepNo: u.kind === 'step' ? stepNo : null,
        stepCount,
        anchor: u.anchor || null,
        prose: u.prose || '',
        reveal: u.reveal,
        focus: u.focus,
        rows: urows,
        details: dets,
        gloss: fresh,
        symbols,
        calls: [],
      };
      section.units.push(unit.i);
      units.push(unit);
      // the defs this unit's lines call: their sections come next
      const called = [];
      for (const r of urows) {
        const src = ix.programLines[r.line - 1] || '';
        // a def handed over as a value (the function lmder calls) before the def that is called
        const byValue = r.calls.filter((c) => !new RegExp(`\\b${c}\\s*\\(`).test(src));
        for (const c of [...byValue, ...r.calls]) if (!called.includes(c)) called.push(c);
      }
      if (u.kind === 'step') for (const c of u.step.calls || []) if (!called.includes(c)) called.push(c);
      for (const c of called) {
        if (c === def || visited.has(c) || !isExplained(c)) continue;
        unit.calls.push(c);
        visit(c, section.path, unit.i);
      }
    }
  };
  visit(MAIN, [], null);
  return { sections, units };
}

// ---------------------------------------------------------------------------
// The recorded run, per unit
// ---------------------------------------------------------------------------

/** Where each unit ran: its occurrences in the recording's tree. */
export function indexRun(tree, events) {
  const steps = new Map();
  const calls = new Map();
  const lines = new Map();
  const go = (n) => {
    if (n.kind === 'step') {
      const k = `${n.def}/${n.id}`;
      if (!steps.has(k)) steps.set(k, []);
      steps.get(k).push(n);
    } else if (n.kind === 'call') {
      if (!calls.has(n.def)) calls.set(n.def, []);
      calls.get(n.def).push(n);
    } else if (n.kind === 'line') {
      const k = `${n.def}:${n.line}`;
      if (!lines.has(k)) lines.set(k, []);
      lines.get(k).push(n);
    }
    for (const c of n.children) go(c);
  };
  go(tree);
  // every hold, by def, place and sentence, and by sentence alone
  const holds = new Map();
  const bySentence = new Map();
  for (const e of events) {
    if (e[0] !== 'hold') continue;
    const k = `${e[1]}|${e[2]}|${e[4]}`;
    for (const [m, key] of [[holds, k], [bySentence, e[4]]]) {
      if (!m.has(key)) m.set(key, { def: e[1], at: e[2], sentence: e[4], ok: 0, fail: 0 });
      const h = m.get(key);
      if (e[3] === true) h.ok++;
      else h.fail++;
    }
  }
  return { steps, calls, lines, holds, bySentence, events };
}

/** The tree node of a unit's k-th run (from 1), or null. */
export function occurrence(unit, rx, k = 1) {
  if (!rx) return null;
  if (unit.kind === 'step') return (rx.steps.get(`${unit.def}/${unit.stepId}`) || [])[k - 1] || null;
  if (unit.kind === 'anchor' && unit.anchor && unit.anchor.line !== null) {
    const l = (rx.lines.get(`${unit.def}:${unit.anchor.line}`) || [])[k - 1];
    if (l) return l;
    if (unit.stepId) return (rx.steps.get(`${unit.def}/${unit.stepId}`) || [])[k - 1] || null;
  }
  if (unit.def === MAIN) {
    // the top level: its last recorded line among the unit's lines
    const ls = unit.focus.map((x) => (rx.lines.get(`${MAIN}:${x}`) || [])[0]).filter(Boolean);
    return ls.length ? ls[ls.length - 1] : null;
  }
  return (rx.calls.get(unit.def) || [])[k - 1] || null;
}

/** The values the run held at the end of a unit's k-th run: name -> value. */
export function valuesOf(unit, rx, k = 1) {
  const n = occurrence(unit, rx, k);
  if (!n) return null;
  const at = n.kind === 'call' ? { ...n, kind: 'step' } : n;
  const { vars } = valuesAt(at, rx.events);
  const out = new Map();
  for (const [name, v] of vars) out.set(name, v.value);
  if (n.kind === 'line') for (const [name, v] of Object.entries(n.values || {})) out.set(name, v);
  return out;
}

/** How often a unit ran. */
export function runCount(unit, rx) {
  if (!rx) return 0;
  if (unit.kind === 'step') return (rx.steps.get(`${unit.def}/${unit.stepId}`) || []).length;
  if (unit.kind === 'intro' && unit.def !== MAIN) return (rx.calls.get(unit.def) || []).length;
  return 0;
}

/** The holds checked at a unit (a step's `hold at`, or a def's per-iteration holds on its card). */
export function holdsAt(unit, rx) {
  if (!rx) return [];
  const at = unit.kind === 'step' ? unit.stepId : unit.kind === 'intro' ? 'iteration' : null;
  if (!at) return [];
  return [...rx.holds.values()].filter((h) => h.def === unit.def && h.at === at);
}

/** A value as a sentence shows it: four significant digits unless asked for more. */
export function fmtValue(v, digits = 4) {
  if (v === null || v === undefined) return null;
  if (typeof v === 'boolean') return String(v);
  if (typeof v === 'string') return v;
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) return String(v);
    if (v === 0) return '0';
    const a = Math.abs(v);
    if (a >= 1e-4 && a < 1e7) return String(Number(v.toPrecision(digits)));
    const [m, e] = v.toExponential(digits - 1).split('e');
    return `${Number(m)}e${Number(e)}`;
  }
  if (typeof v === 'object' && v.values) {
    const vals = v.values.map(Number);
    const shape = v.shape || [vals.length];
    if (shape.length <= 1 || (shape.length === 2 && shape[0] === 1)) {
      if (vals.length > 8) return `${vals.length} values from ${fmtValue(Math.min(...vals), digits)} to ${fmtValue(Math.max(...vals), digits)}`;
      return `(${vals.map((x) => fmtValue(x, digits)).join(', ')})`;
    }
    const [r, c] = shape.slice(-2);
    if (shape.length === 2 && r <= 4 && c <= 4) {
      const rowsOut = [];
      for (let i = 0; i < r; i++) rowsOut.push(`[${vals.slice(i * c, (i + 1) * c).map((x) => fmtValue(x, digits)).join(', ')}]`);
      return `[${rowsOut.join(', ')}]`;
    }
    return `a ${shape.join(' × ')} array`;
  }
  if (typeof v === 'object' && v.shape) return `a ${v.shape.join(' × ')} array (norm ${fmtValue(v.norm, digits)})`;
  return null;
}

/**
 * A `run` or `predict` line with its placeholders filled from the recording:
 * `{name}` is the value at the end of the unit's first run, `{name#k}` at the
 * end of its k-th, `{name:p}` with p significant digits. Returns the text
 * split into pieces: strings, and { value } or { missing } for each
 * placeholder, so the page can mark what it filled in. Math spans are left
 * alone: braces there are LaTeX.
 */
export function fillRun(text, unit, rx) {
  const pieces = [];
  const parts = String(text).split(/(\$\$[^$]+\$\$|\$[^$]+\$)/);
  const cache = new Map();
  for (const part of parts) {
    if (part.startsWith('$')) {
      pieces.push(part);
      continue;
    }
    let last = 0;
    const re = /\{([A-Za-z_]\w*)(?:#(\d+))?(?::(\d+))?\}/g;
    let m;
    while ((m = re.exec(part))) {
      pieces.push(part.slice(last, m.index));
      const k = Number(m[2] || 1);
      if (!cache.has(k)) cache.set(k, valuesOf(unit, rx, k));
      const vals = cache.get(k);
      const v = vals ? fmtValue(vals.get(m[1]), Number(m[3] || 4)) : null;
      pieces.push(v === null ? { missing: m[0] } : { value: v, name: m[1] });
      last = m.index + m[0].length;
    }
    pieces.push(part.slice(last));
  }
  return pieces;
}

/** An equation line: its LaTeX (dollars and citations off) and what it cites. */
export function equationParts(text) {
  const holds = [...String(text).matchAll(/\[hold "([^"]*)"\]/g)].map((m) => m[1]);
  const tests = [...String(text).matchAll(/\[test "([^"]*)"\]/g)].map((m) => m[1]);
  const body = String(text).replace(/\[(hold|test) "[^"]*"\]/g, '').trim();
  const m = body.match(/^\$\$([\s\S]*)\$\$\s*$/);
  return { tex: (m ? m[1] : body.replace(/\$/g, '')).trim(), holds, tests };
}

/** The variables a unit's lines assign whose values the run recorded, for the values list. */
export function unitValues(unit, rx, max = 6) {
  const vals = valuesOf(unit, rx, 1);
  if (!vals) return [];
  const loopVars = new Set(unit.rows.filter((r) => r.kind === 'for').flatMap((r) => r.writes));
  const out = [];
  for (const r of unit.rows) {
    for (const w of r.writes) {
      if (loopVars.has(w) || out.some((o) => o.name === w)) continue;
      const v = vals.get(w);
      if (v === undefined) continue;
      const f = fmtValue(v);
      if (f === null) continue;
      out.push({ name: w, value: f, symbol: unit.symbols[w] || null });
      if (out.length >= max) return out;
    }
  }
  return out;
}

export { allOf };

// ---------------------------------------------------------------------------
// The annotated source: every def's code with its docs beside it
// ---------------------------------------------------------------------------
//
// Read v2: the code is
// the spine, and each line carries notes beside it: the doc of every variable
// whose glossary line starts to hold there (a variable's own doc at its first
// real use, a reuse site's doc where its meaning changes), a step's text and
// its why, how and equations at the step's first line, and a line anchor's
// explanation at its statement. Two orders read the same notes: All the code
// (every file, top to bottom, as written) and a core path (the stops of the
// program's `path` block, in its order).

/** Lines of a def as the page shows them: from `def` to the closing brace, trailing blanks off. */
function defLines(ix, def) {
  const node = ix.nodes.get(def);
  if (!node) return [];
  const lo = node.line;
  let hi = node.end;
  while (hi > lo && !(ix.programLines[hi - 1] || '').trim()) hi--;
  return range(lo, hi);
}

/** The line table row whose event stands for a line: the line's own, or the loop without steps it sits in. */
export function eventLine(ix, def, line) {
  if (ix.rows.has(`${def}:${line}`)) return line;
  let best = null;
  for (const r of ix.bundle.lines) {
    if (r.def !== def || r.kind !== 'loop' || r.line > line || r.end < line) continue;
    if (!best || r.line < best.line) best = r;
  }
  return best ? best.line : null;
}

/**
 * Every note of a def, by program line. A note is
 *   { kind: 'doc', name, key, symbol, text, reuse }         a variable's doc
 *   { kind: 'step', stepId, text, details, depth }          a step begins
 *   { kind: 'anchor', anchor, text, details }               one statement explained
 *   { kind: 'def', details }                                the def's own equations
 */
export function defNotes(ix, def) {
  const notes = new Map();
  const add = (line, n) => {
    if (line === null || line === undefined) return;
    if (!notes.has(line)) notes.set(line, []);
    notes.get(line).push(n);
  };
  const intent = ix.bundle.intent[def] || {};
  const dets = ix.details.get(def) || [];
  const node = ix.nodes.get(def);
  // the def's own equations and top-level details, at its first line
  const top = dets.filter((d) => !d.step && !d.under && d.label !== '=');
  if (top.length && node) add(node.line, { kind: 'def', details: top });
  // steps
  for (const s of ix.bundle.nodes) {
    if (s.kind !== 'step' || s.def !== def) continue;
    let depth = 0;
    for (let g = ix.nodes.get(s.parent); g && g.kind !== 'def'; g = ix.nodes.get(g.parent)) if (g.kind === 'step') depth++;
    const own = dets.filter((d) => d.step === s.step && !d.under && d.label !== '=');
    add(s.line, { kind: 'step', stepId: s.step, text: ix.bundle.steps[`${def}/${s.step}`] || s.text || '', details: own, depth });
  }
  // line anchors, with the details indented under them
  for (const a of dets.filter((d) => d.label === '=' && d.line !== null)) {
    const under = dets.filter((d) => d.under === a.anchor && (d.step || null) === (a.step || null) && d.label !== '=');
    add(a.line, { kind: 'anchor', anchor: a.anchor, step: a.step || null, text: a.text, details: under });
  }
  // variable docs
  for (const d of intent.docs || []) {
    if (d.line === null) continue;
    add(d.line, { kind: 'doc', name: d.name, key: d.key, symbol: d.symbol, text: d.text, reuse: d.key.includes('@') });
  }
  // a variable's own doc and a later doc of it at the same line: the later
  // one says what it holds there, and is enough
  for (const [line, ns] of notes) {
    const reused = new Set(ns.filter((n) => n.kind === 'doc' && n.reuse).map((n) => n.name));
    if (reused.size) notes.set(line, ns.filter((n) => !(n.kind === 'doc' && !n.reuse && reused.has(n.name))));
  }
  // at one line: the def's equations, then steps (outer first), then anchors, then docs
  const order = { def: 0, step: 1, anchor: 2, doc: 3 };
  for (const ns of notes.values()) ns.sort((a, b) => order[a.kind] - order[b.kind] || (a.depth || 0) - (b.depth || 0));
  return notes;
}

/**
 * A run of lines cut into chunks: a chunk starts at each line that has notes,
 * and holds the lines up to the next such line. The notes of a chunk sit
 * beside its first line.
 */
export function chunksOf(lines, notes) {
  const out = [];
  let cur = null;
  for (const x of lines) {
    const ns = notes.get(x);
    if (!cur || (ns && ns.length)) {
      cur = { line: x, lines: [], notes: ns || [] };
      out.push(cur);
    }
    cur.lines.push(x);
  }
  return out;
}

/** The first `def ... {` line's module and file of a def or the top level. */
function moduleOf(ix, def) {
  if (def === MAIN) return ix.bundle.name;
  return (ix.bundle.intent[def] || {}).module || ix.toModule(ix.nodes.get(def)?.line || 1)?.module || null;
}

/**
 * The program as files, each file as written: its top-level code (without its
 * `###` header, test blocks and defs) and its defs in source order. A def with
 * an intent entry or steps is annotated; the others (test helpers, modules
 * without an .intent file) are listed, their code behind a toggle.
 */
export function buildFiles(ix) {
  const mods = ix.bundle.modules;
  const P = ix.programLines;
  const byMod = mods.map((name, k) => ({ name, index: k, defs: [], top: [], lines: 0, annotated: 0 }));
  const inDef = new Set();
  const defsAt = [];
  for (const n of ix.bundle.nodes) {
    if (n.kind !== 'def' || !(n.line > 0) || n.id === MAIN) continue;
    for (let x = n.line; x <= n.end; x++) inDef.add(x);
    defsAt.push(n);
  }
  defsAt.sort((a, b) => a.line - b.line);
  for (const n of defsAt) {
    const m = ix.toModule(n.line);
    if (!m) continue;
    const it = ix.bundle.intent[n.id];
    const annotated = !!(it && (it.entry || n.stepped)) && !isTestHelper(ix, n);
    // the comment lines right above a def are its own
    const above = [];
    for (let x = n.line - 1; x > 0; x--) {
      const t = (P[x - 1] || '').trim();
      if (!t.startsWith('#') || /^#\s*-{3,}/.test(t) || /^#!/.test(t) || inDef.has(x)) break;
      above.unshift(x);
    }
    for (const x of above) inDef.add(x);
    byMod[m.moduleIndex].defs.push({ def: n.id, line: n.line, annotated, test: isTestHelper(ix, n), lines: defLines(ix, n.id), comment: above.filter((x) => (P[x - 1] || '').trim() !== '#~ test') });
    if (annotated) byMod[m.moduleIndex].annotated++;
  }
  // top-level lines per module: skip the ### header, tests and defs
  let header = new Array(mods.length).fill(false);
  let seenCode = new Array(mods.length).fill(false);
  for (let x = 1; x <= P.length; x++) {
    const m = ix.toModule(x);
    if (!m || inDef.has(x)) continue;
    const k = m.moduleIndex;
    const t = P[x - 1];
    if (/^###\s*$/.test(t)) {
      header[k] = !header[k];
      continue;
    }
    if (header[k]) continue;
    if (/^test\s+"/.test(t)) {
      let depth = 0;
      for (; x <= P.length; x++) {
        for (const c of P[x - 1]) {
          if (c === '{') depth++;
          else if (c === '}') depth--;
        }
        if (depth <= 0 && P[x - 1].includes('}')) break;
      }
      continue;
    }
    if (/^#!\s*use\b/.test(t) || /^#~/.test(t)) continue;
    if (!seenCode[k] && !t.trim()) continue;
    seenCode[k] = true;
    byMod[k].top.push(x);
  }
  for (const f of byMod) {
    // blank lines at the end of the top level, and a trailing ruler comment, are not code
    while (f.top.length && !(P[f.top[f.top.length - 1] - 1] || '').trim()) f.top.pop();
    while (f.top.length && /^#\s*-{3,}/.test(P[f.top[f.top.length - 1] - 1] || '')) f.top.pop();
    f.lines = f.top.length + f.defs.reduce((s, d) => s + d.lines.length, 0);
  }
  return byMod;
}

/** A def the module marks for its tests (`#~ test` on the line above it). */
function isTestHelper(ix, n) {
  for (let x = n.line - 1; x > 0 && x > n.line - 4; x--) {
    const t = (ix.programLines[x - 1] || '').trim();
    if (t === '#~ test') return true;
    if (t && !t.startsWith('#')) return false;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Passes: which pass (and trial) of the run the values come from
// ---------------------------------------------------------------------------

/**
 * The run's passes: the iteration loop with the most passes (lmder's, one per
 * Jacobian), and inside each pass its trials, when the loop body holds another
 * loop with steps (lmder's trial steps): each trial begins where that loop's
 * first step begins. Returns null when the run has no loop.
 */
export function passModel(ix, tree, rx) {
  if (!tree) return null;
  // the outermost loop that goes round more than once (lmder's, not lmpar's
  // inside it; an SCF loop, not a loop inside one of its passes)
  const depth = (n) => {
    let k = 0;
    for (let m = n.parent; m; m = m.parent) k++;
    return k;
  };
  const loops = allOf(tree, 'loop').filter((l) => l.children.length > 1);
  if (!loops.length) return null;
  let main = loops[0];
  for (const l of loops) {
    const d = depth(l) - depth(main);
    if (d < 0 || (d === 0 && l.children.length > main.children.length)) main = l;
  }
  const def = main.def;
  // the step that opens each trial: the first step, by line, inside a loop
  // of the graph nested in the main loop's box
  let marker = null;
  const box = main.graphId;
  if (box) {
    const inner = ix.bundle.nodes.filter((g) => g.kind === 'loop' && g.def === def && g.id !== box && isUnder(ix, g, box));
    for (const g of inner) {
      const first = ix.bundle.nodes.filter((s) => s.kind === 'step' && s.def === def && isUnder(ix, s, g.id)).sort((a, b) => a.line - b.line)[0];
      if (first && (!marker || first.line < marker.line)) marker = first;
    }
  }
  const occ = marker ? rx.steps.get(`${def}/${marker.step}`) || [] : [];
  const passes = main.children.map((it) => {
    const starts = occ.filter((s) => s.i >= it.i && s.i <= it.end);
    const trials = starts.map((s, k) => ({ t: k + 1, i: s.i, end: k + 1 < starts.length ? starts[k + 1].i - 1 : it.end }));
    return { n: it.n, i: it.i, end: it.end, trials };
  });
  return { def, loopLabel: labelFor(ix, main), marker: marker ? marker.step : null, passes, start: main.i, end: main.end };
}

function isUnder(ix, g, id) {
  for (let p = ix.nodes.get(g.parent); p; p = ix.nodes.get(p.parent)) if (p.id === id) return true;
  return false;
}

function labelFor(ix, loop) {
  const g = loop.graphId && ix.nodes.get(loop.graphId);
  return g ? g.header : 'loop';
}

/** Where an event falls: { pass, trial } (0 when it is before or after the passes). */
export function locate(pm, i) {
  if (!pm) return { pass: 0, trial: 0 };
  for (const p of pm.passes) {
    if (i < p.i || i > p.end) continue;
    const t = p.trials.find((tr) => i >= tr.i && i <= tr.end);
    return { pass: p.n, trial: t ? t.t : 0, trials: p.trials.length };
  }
  return { pass: 0, trial: 0, before: i < pm.start };
}

/** A pass label as the page prints it. */
export function passLabel(loc, pm) {
  if (!pm) return 'this run';
  if (!loc.pass) return loc.before ? 'before the first pass' : 'after the last pass';
  if (loc.trial && loc.trials > 1) return `pass ${loc.pass} · trial ${loc.trial}`;
  return `pass ${loc.pass}`;
}

/** The recorded line events of each def, in order: def -> [line node]. */
export function linesByDef(rx) {
  if (!rx) return new Map();
  if (rx.byDef) return rx.byDef;
  const m = new Map();
  for (const ns of rx.lines.values()) {
    for (const n of ns) {
      if (!m.has(n.def)) m.set(n.def, []);
      m.get(n.def).push(n);
    }
  }
  for (const ns of m.values()) ns.sort((a, b) => a.i - b.i);
  rx.byDef = m;
  return m;
}

/**
 * The value a variable had where its doc starts to hold, in the chosen pass
 * and trial (sel = { pass, trial }). The window is the trial's events, else
 * the pass's, else the whole run: the first of them in which the doc's
 * statement ran. A doc at one statement (a variable's own doc) reads that
 * statement's event: what it assigned, or else what the variable held
 * there. A doc for a step (`temp@shrink`, `lo..hi` its lines) reads the last
 * value the step assigned. Returns { value, loc }, loc being where the event
 * really was, or null.
 */
export function valueAt(ix, rx, pm, sel, def, line, name, stepEnd = null) {
  if (!rx) return null;
  const evs = linesByDef(rx).get(def) || [];
  if (!evs.length) return null;
  // a line inside a loop without steps is recorded at the loop's line
  const at = eventLine(ix, def, line);
  const windows = [];
  const pass = pm && sel ? pm.passes.find((p) => p.n === sel.pass) : null;
  if (pass) {
    const tr = pass.trials.find((t) => t.t === sel.trial);
    if (tr) windows.push([tr.i, tr.end]);
    windows.push([pass.i, pass.end]);
  }
  windows.push([0, Infinity]);
  for (const [a, b] of windows) {
    let site = null;
    let last = null;
    for (const n of evs) {
      if (n.i < a) continue;
      if (n.i > b) break;
      const inStep = stepEnd !== null && n.line >= Math.min(line, at ?? line) && n.line <= stepEnd;
      // the step's first run in the window, and no later one
      if (site && stepEnd !== null && !inStep && n.def === site.def) break;
      if (!site && (n.line === at || inStep)) site = n;
      if (inStep && n.values && n.values[name] !== undefined) last = n;
      if (stepEnd === null && site) break;
    }
    if (!site) continue;
    if (last) return { value: last.values[name], loc: locate(pm, last.i) };
    let v = site.values ? site.values[name] : undefined;
    if (v === undefined) v = valuesAt(site, rx.events).vars.get(name)?.value;
    if (v !== undefined) return { value: v, loc: locate(pm, site.i) };
  }
  return null;
}

// ---------------------------------------------------------------------------
// The core path
// ---------------------------------------------------------------------------

/**
 * A `path` block of the bundle as stops the page shows, in the path's order.
 * Each stop has the program lines it shows (a step's lines, one statement's,
 * or the def's signature for a def), the step's own details, the path's
 * details, and what the reading skips between it and the stop before, in the
 * order the calls reach it.
 */
export function buildPath(ix, path, book) {
  const order = book ? book.units : buildReading(ix).units;
  const unitIndex = (st) => {
    if (st.step) return order.findIndex((u) => u.def === st.def && u.kind === 'step' && u.stepId === st.step);
    return order.findIndex((u) => u.def === st.def && u.kind === 'intro');
  };
  const stops = [];
  let prevAt = -1;
  const stopAt = new Set(path.stops.map(unitIndex));
  // a unit whose lines a stop shows (a step inside a stop's step) is not skipped
  for (const st of path.stops) {
    const node = st.step ? ix.nodes.get(`${st.def}/${st.step}`) : null;
    if (!node) continue;
    order.forEach((u, j) => {
      const g = u.kind === 'step' ? ix.nodes.get(`${u.def}/${u.stepId}`) : null;
      if (g && u.def === st.def && g.line >= node.line && g.end <= node.end) stopAt.add(j);
    });
  }
  path.stops.forEach((st, k) => {
    const node = st.step ? ix.nodes.get(`${st.def}/${st.step}`) : ix.nodes.get(st.def);
    let lines = [];
    if (st.anchor && st.line) {
      const row = (ix.bundle.tex || []).find((r) => r.def === st.def && r.line === st.line);
      lines = range(st.line, row ? row.end : st.line);
    } else if (st.step && node) {
      lines = range(node.line, node.end);
    } else if (node) {
      lines = [node.line];
    }
    const at = unitIndex(st);
    const skipped = [];
    if (at > prevAt + 1 && prevAt >= -1) {
      for (let j = prevAt + 1; j < at; j++) {
        const u = order[j];
        // a unit a later stop shows is not skipped
        if (u.kind === 'cont' || u.kind === 'prose' || stopAt.has(j)) continue;
        skipped.push(u);
      }
    }
    if (at >= 0) prevAt = Math.max(prevAt, at);
    const dets = (ix.details.get(st.def) || []).filter((d) => (st.step ? d.step === st.step : !d.step) && !d.under && d.label !== '=');
    stops.push({ i: k, stop: st, def: st.def, stepId: st.step || null, anchor: st.anchor || null, lines, node, details: dets, pathDetails: st.details, skipped, unitAt: at });
  });
  // the rest of the program after the last stop
  const tail = [];
  for (let j = prevAt + 1; j < order.length; j++) if (order[j].kind !== 'cont' && order[j].kind !== 'prose' && !stopAt.has(j)) tail.push(order[j]);
  return { id: path.id, title: path.title, stops, tail };
}

/** Skipped units, grouped by def in order, as one line of text: `lmder: setup, check_input; qrfac (all of it)`. */
export function skippedText(units, ix, stopModules = null) {
  const groups = [];
  for (const u of units) {
    const mod = moduleOf(ix, u.def);
    if (stopModules && u.def !== MAIN && !stopModules.has(mod)) {
      const last = groups[groups.length - 1];
      if (!last || last.module !== mod) groups.push({ module: mod, def: null, parts: [] });
      continue;
    }
    const last = groups[groups.length - 1];
    if (!last || last.def !== u.def) groups.push({ def: u.def, parts: [], whole: false });
    const g = groups[groups.length - 1];
    if (u.kind === 'intro') g.card = true;
    else if (u.kind === 'step') g.parts.push(u.stepId);
    else if (u.kind === 'anchor') g.parts.push(`= ${u.anchor.anchor}`);
  }
  return groups.map((g) => {
    if (g.module) return `${g.module}.viv`;
    const name = g.def === MAIN ? 'the program' : g.def;
    const all = ix.bundle.nodes.filter((n) => n.kind === 'step' && n.def === g.def).length;
    if (g.card && (g.parts.length === 0 || g.parts.length >= all)) return `${name} (all of it)`;
    return g.parts.length ? `${name}: ${g.parts.join(', ')}` : name;
  });
}

/** The variables a def has (its parameters and every variable it assigns) with no doc: what `documents every variable` forbids. */
export function undocumented(ix, def) {
  const it = ix.bundle.intent[def] || {};
  const have = new Set((it.docs || []).map((d) => d.name));
  return (it.vars || []).filter((v) => !have.has(v));
}
