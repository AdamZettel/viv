// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// A ladder as a page sees it, from the report `viv ladder` (or viv_ladder in
// the browser) produces. ONE copy, used by the site build for /ladders and for
// ladder posts, by the server when a ladder post is published, and by the
// composer's preview.
//
// The report is the compiler's: per rung, the program at that rung, what the
// rung added, edited and carried (with content addresses), its tests, and every
// rule it broke. This module adds what a reader wants beside that -- each
// function's one-line `what`, which lines of an edited function are new, the
// rung's note split into its lead and its paragraphs -- and nothing the
// compiler did not establish.
//
// `docOf(program)` is the host's `viv doc` (parsed JSON), or null.

/** The rung's note as paragraphs: a bare `#|` line separates them. */
export function noteParagraphs(note) {
  const paras = [];
  let cur = [];
  for (const l of String(note || '').split('\n')) {
    if (!l.trim()) { if (cur.length) paras.push(cur.join(' ')); cur = []; } else cur.push(l.trim());
  }
  if (cur.length) paras.push(cur.join(' '));
  return paras;
}

/** The rung's body with its leading note taken off: the code a reader sees. */
export function codeOf(text) {
  const lines = String(text || '').split('\n');
  let i = 0;
  while (i < lines.length && lines[i].trim().startsWith('#|')) i += 1;
  return lines.slice(i).join('\n').replace(/^\n+|\n+$/g, '');
}

const defCode = (program, fn) => (fn ? program.split('\n').slice(fn.rawLine - 1, fn.rawEnd).join('\n') : '');
const whatOf = (fn, code) => {
  if (!fn) return '';
  if (fn.what) return fn.what;
  const docstring = (code.match(/^def [^\n]*\{\n\s*"([^"]*)"/) || [])[1];
  return docstring ? docstring.replace(/\s+/g, ' ').trim() : '';
};

/**
 * The view: { rungs: [...], top: { functions, tests, lines }, broken: [...] }.
 * Each rung: { n, title, why, notes, code, marks, added, edited, carried, tests, lines, output, broken }
 * where `marks` are the 0-based lines of `code` that an edited function did
 * not have on the rung below, and each ledger entry carries its `what`.
 */
export function ladderView(report, docOf = () => null) {
  const rungs = [];
  let prevDoc = null;
  let prevProgram = '';
  const since = {}; // name -> rung that last set its address
  for (const r of report.rungs) {
    const doc = docOf(r.program) || { functions: [] };
    const fnOf = (name) => doc.functions.find((f) => f.name === name);
    const code = codeOf(r.text);
    const marks = [];
    const ledger = (list, extra) => list.map((e) => {
      const fn = fnOf(e.name);
      const c = defCode(r.program, fn);
      return { ...e, what: whatOf(fn, c), ...(extra ? extra(e, fn, c) : {}) };
    });
    const edited = ledger(r.edited, (e, fn, c) => {
      // which lines of the new body the old body did not have
      const before = new Set(defCode(prevProgram, prevDoc && prevDoc.functions.find((f) => f.name === e.name)).split('\n').map((l) => l.trim()));
      const at = code.indexOf(c.split('\n')[0]);
      if (at >= 0) {
        const first = code.slice(0, at).split('\n').length - 1;
        c.split('\n').forEach((l, k) => { if (l.trim() && !before.has(l.trim())) marks.push(first + k); });
      }
      since[e.name] = r.n;
      return {};
    });
    const added = ledger(r.added, (e) => { since[e.name] = r.n; return {}; });
    const carried = ledger(r.carried, (e) => ({ since: since[e.name] || 1 }));
    const paras = noteParagraphs(r.note);
    rungs.push({
      n: r.n, title: r.title, why: paras[0] || '', notes: paras.slice(1),
      code, marks, added, edited, carried,
      tests: r.tests.map((name) => ({ name, isNew: r.newTests.includes(name) })),
      lines: r.lines, output: r.output, broken: r.broken,
    });
    prevDoc = doc;
    prevProgram = r.program;
  }
  const last = rungs[rungs.length - 1];
  const lastReport = report.rungs[report.rungs.length - 1];
  const top = last ? { functions: last.added.length + last.edited.length + last.carried.length, tests: last.tests.length, lines: lastReport.lines } : { functions: 0, tests: 0, lines: 0 };
  return { rungs, top, broken: rungs.flatMap((r) => r.broken.map((b) => `rung ${r.n}: ${b}`)) };
}
