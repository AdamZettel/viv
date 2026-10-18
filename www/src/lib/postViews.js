// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The views of a post's file (components/PostViews.jsx), all from ONE parse:
// splitNotebook (src/lib/notebook.js) cuts the file into header, prose and code
// cells, and everything here reads those cells. Nothing is a second copy of the
// file, so the views cannot drift from it or from each other.
//
//   article    prose as prose; each code cell folded to a one-line summary,
//              its tests as a list of names, its output under it
//   notebook   prose and editable code cells, each with its output
//   code       the program alone: prose cells and `#|` doc blocks folded to
//              one line each; code, ordinary comments and tests as written
//   source     the file as stored, byte for byte
import { docHeader } from './notebook.js';

export const VIEWS = [
  { id: 'article', label: 'Article', hint: 'the prose, with figures and results; code folded' },
  { id: 'notebook', label: 'Notebook', hint: 'prose and code cells, each cell runs' },
  { id: 'code', label: 'Code', hint: 'the program alone, prose folded' },
  { id: 'source', label: 'Source', hint: 'the .viv file exactly as stored' },
];
export const isView = (v) => VIEWS.some((x) => x.id === v);

/**
 * Blank out string literals and `#` comments on a line, keeping its length,
 * so braces and keywords inside them are not counted.
 */
function codeOnly(line) {
  let out = '';
  let inStr = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inStr) {
      if (ch === '\\') { out += '  '; i++; continue; }
      if (ch === '"') inStr = false;
      out += ' ';
    } else if (ch === '"') { inStr = true; out += ' '; } else if (ch === '#') { break; } else out += ch;
  }
  return out;
}

/** The test names in a piece of code, in order: `test "name" {`. */
export function testNames(code) {
  const names = [];
  for (const m of String(code).matchAll(/^[ \t]*test[ \t]+"((?:[^"\\]|\\.)*)"/gm)) names.push(m[1].replace(/\\(.)/g, '$1'));
  return names;
}

/** Names of the functions a piece of code defines, in order. */
export function defNames(code) {
  return [...String(code).matchAll(/^[ \t]*def[ \t]+([A-Za-z_]\w*)/gm)].map((m) => m[1]);
}

/**
 * A code cell cut at its top-level `rung "title" { ... }` blocks:
 * [{ kind: 'code', text } | { kind: 'rung', title, note, body, text }], in
 * order; joining the texts gives the code back. `note` is the `#|` lines
 * that open the rung (its why and notes), `body` the rest of its code.
 */
export function rungSegments(code) {
  const lines = String(code).split('\n');
  const segs = [];
  let buf = [];
  const flush = () => { if (buf.length) segs.push({ kind: 'code', text: buf.join('\n') }); buf = []; };
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^rung[ \t]+"((?:[^"\\]|\\.)*)"[ \t]*\{/);
    if (!m) { buf.push(lines[i]); continue; }
    let depth = 0;
    let j = i;
    for (; j < lines.length; j++) {
      for (const ch of codeOnly(lines[j])) { if (ch === '{') depth++; else if (ch === '}') depth--; }
      if (depth <= 0 && j >= i) break;
    }
    flush();
    const inner = lines.slice(i + 1, j);
    let k = 0;
    const note = [];
    while (k < inner.length && /^\s*#\|/.test(inner[k])) { note.push(inner[k].replace(/^\s*#\|\s?/, '')); k++; }
    const body = inner.slice(k).join('\n').replace(/^\s*\n/, '').replace(/\s+$/, '');
    segs.push({ kind: 'rung', title: m[1].replace(/\\(.)/g, '$1'), note: note.join('\n'), body: dedent(body), text: lines.slice(i, j + 1).join('\n') });
    i = j;
  }
  flush();
  return segs;
}

function dedent(text) {
  const ls = text.split('\n');
  const ind = Math.min(...ls.filter((l) => l.trim()).map((l) => l.match(/^[ \t]*/)[0].length));
  return Number.isFinite(ind) && ind > 0 ? ls.map((l) => l.slice(Math.min(ind, l.match(/^[ \t]*/)[0].length))).join('\n') : text;
}

/** What a folded code cell says about itself: lines, the defs, the tests. */
export function codeSummary(code) {
  const text = String(code).replace(/\n+$/, '');
  return { lines: text ? text.split('\n').length : 0, defs: defNames(text), tests: testNames(text) };
}

/** "code: 14 lines · def th_model, th_fcn · 2 tests", as a plain string. */
export function summaryText(s, { withTests = true } = {}) {
  const parts = [`${s.lines} line${s.lines === 1 ? '' : 's'}`];
  if (s.defs.length) parts.push(`def ${s.defs.slice(0, 3).join(', ')}${s.defs.length > 3 ? ` +${s.defs.length - 3}` : ''}`);
  if (withTests && s.tests.length) parts.push(`${s.tests.length} test${s.tests.length === 1 ? '' : 's'}`);
  return parts.join(' · ');
}

/**
 * Test results out of a run's text: { name: { ok, why } } from its PASS and
 * FAIL lines (`PASS name`, `FAIL name -- why`).
 */
export function testResults(out) {
  const res = {};
  for (const l of String(out || '').split('\n')) {
    let m = l.match(/^PASS (.*)$/);
    if (m) { res[m[1].trim()] = { ok: true }; continue; }
    m = l.match(/^FAIL (.*?)(?: -- (.*))?$/);
    if (m) res[m[1].trim()] = { ok: false, why: m[2] || '' };
  }
  return res;
}

/** A prose cell as one line, for the Code view: its heading, or its first words. */
export function proseLabel(body) {
  const lines = String(body).split('\n').map((l) => l.trim()).filter(Boolean);
  const h = lines.find((l) => /^#{1,4}\s/.test(l));
  const first = (h ? h.replace(/^#+\s*/, '') : lines[0] || '').replace(/\$[^$]*\$/g, '…').replace(/[`*_]/g, '');
  return first.length > 72 ? `${first.slice(0, 70).replace(/\s+\S*$/, '')}…` : first;
}

/**
 * The Code view: the file as a list of pieces, each either code to show as it
 * is, or a fold (the header, a prose cell, a `#|` block, a `#!` doc header)
 * shown as one line that opens to its text.
 *   { kind: 'code', text, line }
 *   { kind: 'fold', what: 'header' | 'prose' | 'doc', label, text, line, lines }
 * Joining every piece's text gives the file back, less the newlines between
 * pieces (each piece is whole lines).
 */
export function codePieces(cells) {
  const out = [];
  const pushCode = (text, line) => {
    if (!text) return;
    const last = out[out.length - 1];
    if (last && last.kind === 'code') last.text += `\n${text}`;
    else out.push({ kind: 'code', text, line });
  };
  for (const c of cells) {
    const body = c.text.replace(/\n$/, '');
    const n = body.split('\n').length;
    if (c.kind === 'header') {
      out.push({ kind: 'fold', what: 'header', label: `the post's header: ${c.fields.post || 'fields and prose'}`, text: body, line: c.start, lines: n });
    } else if (c.kind === 'prose') {
      out.push({ kind: 'fold', what: 'prose', label: proseLabel(c.body) || 'prose', text: body, line: c.start, lines: n });
    } else {
      if (c.empty) continue;
      const code = c.code;
      const head = docHeader(code);
      let at = c.codeStart;
      if (head.doc) {
        const hl = head.doc.replace(/\n+$/, '').split('\n');
        out.push({ kind: 'fold', what: 'doc', label: head.fields[0] ? head.fields[0].join(' ') : 'file header', text: hl.join('\n'), line: at, lines: hl.length });
        at += head.doc.split('\n').length - 1;
      }
      const lines = head.body.replace(/\n+$/, '').split('\n');
      let buf = [];
      let from = at;
      for (let i = 0; i < lines.length; i++) {
        if (/^\s*#\|/.test(lines[i])) {
          let j = i;
          while (j < lines.length && /^\s*#\|/.test(lines[j])) j++;
          pushCode(buf.join('\n'), from);
          buf = [];
          const block = lines.slice(i, j);
          const first = block.map((l) => l.replace(/^\s*#\|\s?/, '')).find((l) => l.trim()) || '';
          out.push({ kind: 'fold', what: 'doc', label: first.length > 72 ? `${first.slice(0, 70).replace(/\s+\S*$/, '')}…` : first, text: block.join('\n'), line: at + i, lines: block.length, indent: block[0].match(/^\s*/)[0] });
          i = j - 1;
          from = at + j;
        } else buf.push(lines[i]);
      }
      pushCode(buf.join('\n').replace(/^\n+/, ''), from);
    }
  }
  return out;
}

/** The program as the Code view shows it unfolded: code, comments and tests; no prose, no `#|`. */
export const programText = (pieces) => `${pieces.filter((p) => p.kind === 'code').map((p) => p.text.replace(/^\n+|\n+$/g, '')).join('\n\n')}\n`;

/** Where the header's prose puts the figure: an article's `[figure]` paragraph. */
export function splitAtFigure(html) {
  const parts = String(html || '').split(/<p>\s*\[figure\]\s*<\/p>/);
  return parts.length > 1 ? [parts[0], parts.slice(1).join('')] : [parts[0], null];
}

/** The view to open: ?view=, then #view=, then what this reader chose last, then Article. */
export function initialView(loc, stored) {
  try {
    const q = new URLSearchParams(loc.search).get('view');
    if (isView(q)) return q;
    const h = (loc.hash.match(/(?:^#|&)view=(\w+)/) || [])[1];
    if (isView(h)) return h;
  } catch { /* no location */ }
  return isView(stored) ? stored : 'article';
}
