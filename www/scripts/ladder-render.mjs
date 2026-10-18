// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// A ladder rendered for a page, at build time: viv_ladder's report shaped by
// ladderView.js, with the prose through KaTeX and each rung's program staged
// for the workbench. Shared by ladder-build (the /ladders pages) and feed-build
// (ladder posts).
import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { prose } from './prose.mjs';
import { ladderView, codeOf } from '../src/lib/ladderView.js';

// ---- the whole program at each rung, as a reader follows it ------------------
//
// The compiler's program at rung k puts a redefined def in the rung that
// redefined it. A reader following the program wants the def where it first
// appeared, with its new body, so the program grows in place from rung to
// rung. Defs are hoisted, so the two orders run the same; the build checks
// that by running both and comparing what they print.

/** A rung body as segments: { def: name, lines } for each top-level def, { lines } between. */
function segments(code) {
  const out = [];
  let plain = [];
  const lines = code.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^def (\w+)\s*\(/);
    if (!m) { plain.push(lines[i]); continue; }
    if (plain.length) out.push({ lines: plain });
    plain = [];
    let j = i;
    while (j < lines.length && lines[j] !== '}') j++;
    out.push({ def: m[1], lines: lines.slice(i, j + 1) });
    i = j;
  }
  if (plain.length) out.push({ lines: plain });
  return out;
}

const SPDX = /^# SPDX-/;

/** The program at each rung, defs in place: [{ lines: [{ text, def }] }]. */
export function inPlacePrograms(report) {
  const bodies = report.rungs.map((r) => segments(codeOf(r.text)));
  const base = String(report.base || '').split('\n').filter((l) => !l.startsWith('#!'));
  const tail = String(report.tail || '').split('\n').filter((l) => !SPDX.test(l));
  return report.rungs.map((r, k) => {
    const first = {};
    const latest = {};
    for (let j = 0; j <= k; j++) {
      for (const seg of bodies[j]) {
        if (!seg.def) continue;
        if (!(seg.def in first)) first[seg.def] = j;
        latest[seg.def] = seg.lines;
      }
    }
    const out = base.map((text) => ({ text, def: null }));
    for (let j = 0; j <= k; j++) {
      out.push({ text: '', def: null }, { text: `# rung ${j + 1}. ${report.rungs[j].title}`, def: null }, { text: '', def: null });
      for (const seg of bodies[j]) {
        if (!seg.def) out.push(...seg.lines.map((text) => ({ text, def: null })));
        else if (first[seg.def] === j) out.push(...latest[seg.def].map((text) => ({ text, def: seg.def })));
      }
    }
    out.push(...tail.map((text) => ({ text, def: null })));
    // one blank line at most between pieces, none at the ends
    const lines = out.filter((l, i) => l.text.trim() || (i > 0 && out[i - 1].text.trim()));
    while (lines.length && !lines[0].text.trim()) lines.shift();
    while (lines.length && !lines[lines.length - 1].text.trim()) lines.pop();
    return { lines };
  });
}

/** For each line of `b`, whether it is also in `a`, by longest common subsequence. */
export function keptLines(a, b) {
  const n = a.length;
  const m = b.length;
  const L = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      L[i][j] = a[i] === b[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
    }
  }
  const kept = new Array(m).fill(false);
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { kept[j] = true; i++; j++; } else if (L[i + 1][j] >= L[i][j + 1]) i++; else j++;
  }
  return kept;
}

/** What a run printed, less the PASS lines (the tests are listed on their own). */
const printedOf = (output) => String(output || '').split('\n').filter((l) => l.trim() && !/^(PASS |__)/.test(l));

/** `ex` is the instantiated viv.wasm's exports. */
export function ladderRenderer(ex) {
  const put = (text) => {
    const enc = new TextEncoder().encode(text);
    const ptr = ex.viv_alloc(enc.length);
    new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
    return [ptr, enc.length];
  };
  const take = (packed) => new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
  const call1 = (fn, text) => take(fn(...put(text)));
  const run = (text) => call1(ex.viv_run, text);
  const docOf = (program) => { try { return JSON.parse(call1(ex.viv_doc, program)); } catch { return null; } };

  /** { errors, view, report, seconds }. With `stageDir`, writes rungN.viv there and sets each rung's `file` under `openBase`. */
  return function renderLadder(src, { stageDir = null, openBase = null } = {}) {
    const t0 = performance.now();
    const reportText = call1(ex.viv_ladder, src);
    const seconds = (performance.now() - t0) / 1000;
    if (reportText.startsWith('error')) return { errors: [reportText], view: null, report: null, seconds };
    const report = JSON.parse(reportText);
    const view = ladderView(report, docOf);
    const errors = view.broken.slice();
    const programs = inPlacePrograms(report);
    for (const [k, r] of view.rungs.entries()) {
      const prog = programs[k].lines;
      const text = prog.map((l) => l.text).join('\n') + '\n';
      // the reader's order must print what the compiler's order printed
      const lines = (o) => o.split('\n').filter((l) => l.trim() && !l.startsWith('__'));
      const outA = lines(report.rungs[k].output);
      const outB = lines(run(text));
      if (outA.join('\n') !== outB.join('\n')) {
        errors.push(`rung ${r.n}: the program with each def in its first place prints differently from the compiler's program`);
      }
      const prevProg = k ? programs[k - 1].lines.map((l) => l.text) : [];
      const kept = keptLines(prevProg, prog.map((l) => l.text));
      const edited = new Set(r.edited.map((e) => e.name));
      r.program = text.replace(/\n$/, '');
      const kinds = prog.map((l, i) => (kept[i] || !l.text.trim() ? 's' : l.def && edited.has(l.def) ? 'e' : 'a'));
      // a blank line inside a run of new lines is part of the run
      for (let i = 0; i < kinds.length; i++) {
        if (prog[i].text.trim()) continue;
        let a = i - 1;
        let b = i + 1;
        while (a >= 0 && !prog[a].text.trim()) a--;
        while (b < prog.length && !prog[b].text.trim()) b++;
        if (a >= 0 && b < prog.length && kinds[a] !== 's' && kinds[a] === kinds[b]) kinds[i] = kinds[a];
      }
      r.lineKinds = kinds;
      const printed = printedOf(report.rungs[k].output);
      const printedBefore = k ? printedOf(report.rungs[k - 1].output) : [];
      const keptOut = keptLines(printedBefore, printed);
      r.printed = printed.map((t, i) => ({ text: t, isNew: !keptOut[i] }));
      r.titleHtml = prose(r.title);
      r.whyHtml = prose(r.why);
      r.notesHtml = r.notes.map(prose);
      r.seconds = Math.round(seconds / view.rungs.length * 100) / 100;
      if (stageDir) {
        mkdirSync(stageDir, { recursive: true });
        writeFileSync(join(stageDir, `rung${r.n}.viv`), text);
        r.file = `${openBase}/rung${r.n}.viv`;
      }
      delete r.output;
    }
    return { errors, view, report, seconds };
  };
}
