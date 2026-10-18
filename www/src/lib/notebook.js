// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// A Viv notebook is an ordinary .viv file, cut into cells at its prose fences.
//
// The lexer (lang/viv/src/lexer.rs) treats a line that is exactly `###` as
// opening a block of text that runs to the next `###` line, or to the end of
// the file. The compiler skips the block. This module reads the same file as
// cells:
//
//   header  the first fenced block, when it opens the file (only blank lines
//           before it): a post's `key: value` fields, a blank line, its prose
//   prose   every other fenced block: Markdown, $math$ through KaTeX
//   code    the program text between two fences
//
// Cells run in order, and running cell N means running cells 1..N; the whole
// file still runs as one program in both engines, so a notebook is never a
// second copy of anything.
//
// Every cell keeps its exact text, so joining the texts gives the file back
// byte for byte (joinCells), and the 1-based line range it came from.

import { HEADER_FIELD } from './postCheck.js';

// The lexer's two tests, reproduced exactly. An opening fence may be indented
// (only whitespace before the `#`); a closing one may not. Both may carry
// trailing whitespace.
const opensFence = (line) => /^\s*###\s*$/.test(line);
const closesFence = (line) => line.trimEnd() === '###';

/** `key: value` lines, then the prose after them (see src/lib/postCheck.js). */
function headerParts(body) {
  const lines = body.split('\n');
  const fields = {};
  let i = 0;
  for (; i < lines.length; i++) {
    const m = lines[i].match(HEADER_FIELD);
    if (!m) break;
    fields[m[1].toLowerCase()] = m[2].trim();
  }
  return { fields, prose: lines.slice(i).join('\n').replace(/^\n+|\n+$/g, '') };
}

/** A code cell's text as lead (blank lines before), code, trail (the newline
 *  after the last line of code, and blank lines after it). */
function codeParts(text) {
  const lead = (text.match(/^(?:[ \t]*\n)*/) || [''])[0];
  const rest = text.slice(lead.length);
  const trail = (rest.match(/(?:\n[ \t]*)*\n?$/) || [''])[0];
  return { lead, code: rest.slice(0, rest.length - trail.length), trail };
}

/**
 * Split a .viv source into cells.
 *
 * Each cell is { kind, text, start, end } with `start`..`end` its 1-based,
 * inclusive line range in the file, plus:
 *   header  { fields, prose, body }
 *   prose   { body }                         the Markdown between the fences
 *   code    { lead, code, trail, codeStart } code without its surrounding
 *           blank lines, and the file line its first line of code is on;
 *           `empty` when the cell is only blank lines
 *
 * A file with no fences is one code cell.
 */
export function splitNotebook(src) {
  const text = String(src ?? '');
  // lines with their newline kept, so slices rejoin to the exact bytes
  const lines = text.match(/[^\n]*\n|[^\n]+$/g) || [];
  const bare = (i) => lines[i].replace(/\n$/, '');
  const cells = [];
  let codeFrom = 0; // first line of the code cell being gathered
  const pushCode = (to) => {
    if (to <= codeFrom) return;
    const t = lines.slice(codeFrom, to).join('');
    const parts = codeParts(t);
    const leadLines = (parts.lead.match(/\n/g) || []).length;
    cells.push({ kind: 'code', text: t, start: codeFrom + 1, end: to, ...parts, codeStart: codeFrom + 1 + leadLines, empty: parts.code.trim() === '' });
  };
  let i = 0;
  while (i < lines.length) {
    if (!opensFence(bare(i))) { i++; continue; }
    // a fenced block from line i to its closing fence (or the end of the file)
    let j = i + 1;
    while (j < lines.length && !closesFence(bare(j))) j++;
    const closed = j < lines.length;
    const last = closed ? j : lines.length - 1;
    const body = lines.slice(i + 1, closed ? j : lines.length).join('').replace(/\n$/, '');
    // the header: the file's first fence, with only blank lines before it
    const isHeader = cells.length === 0 && lines.slice(0, i).every((l) => l.trim() === '');
    // blank lines before the header belong to it, so no empty code cell leads the file
    if (!isHeader) pushCode(i);
    const from = isHeader ? 0 : i;
    const cell = { kind: isHeader ? 'header' : 'prose', text: lines.slice(from, last + 1).join(''), start: from + 1, end: last + 1, body };
    if (isHeader) Object.assign(cell, headerParts(body));
    cells.push(cell);
    i = last + 1;
    codeFrom = i;
  }
  pushCode(lines.length);
  return cells;
}

/** The file the cells came from, byte for byte. */
export const joinCells = (cells) => cells.map((c) => c.text).join('');

/** A code cell with its code replaced; the blank lines around it are kept. */
export function withCode(cell, code) {
  const c = String(code).replace(/\n+$/, '');
  const { lead, trail } = cell;
  return { ...cell, code: c, text: lead + c + (trail || (c ? '\n' : '')), empty: c.trim() === '' };
}

/** Whether a file is more than one cell of code: it has prose between its parts. */
export const isNotebook = (cells) => cells.some((c) => c.kind === 'prose');

/**
 * What to send the interpreter for each code cell: `units[k]` is the file text
 * from just after code cell k-1 through the end of code cell k, with one
 * trailing newline taken off. Prose between them rides along (the lexer skips
 * it), so a unit's line numbers are the file's, shifted by where it starts.
 *
 * The units are built so that `units.slice(0, k).join('\n')` is exactly the
 * file up to the end of code cell k-1: the interpreter's incremental session
 * (interp::run_cell) keys its state on the hash of prefix + "\n" + cell, and
 * this is what makes running cell k after cell k-1 continue that session
 * instead of rebuilding it.
 *
 * Returns [{ index, text, line }]: the cell's index in `cells`, the unit's
 * text, and the file line the unit starts on.
 */
export function runUnits(cells) {
  const units = [];
  let from = 0;
  cells.forEach((c, k) => {
    if (c.kind !== 'code' || c.empty) return;
    const text = cells.slice(from, k + 1).map((x) => x.text).join('');
    units.push({ index: k, text: text.replace(/\n$/, ''), line: cells[from].start });
    from = k + 1;
  });
  return units;
}

/**
 * Output for display: the lines a run prints, capped. Returns { shown, hidden }
 * where `hidden` counts the lines left out; figure records never count against
 * the cap (the figure is drawn, not printed).
 */
export function capLines(text, max = 60) {
  const lines = String(text).split('\n');
  if (lines.length <= max) return { shown: text, hidden: 0 };
  return { shown: lines.slice(0, max).join('\n'), hidden: lines.length - max };
}

// Directives, not description: a block that opens with one of these stays code.
const DIRECTIVES = new Set(['use', 'lang', 'solves', 'channel']);

/**
 * A code cell that opens with a file header, a run of `#!` lines such as the
 * PDE solvers' (`#! pde`, `#! equation`, `#! | prose`), split into that
 * header and the code under it. The page renders the header as text with its
 * math, and the editor holds the code; `doc + body` is the cell's code again,
 * byte for byte. Returns { doc: '', body: code } when there is no header.
 *
 * `fields` is [[key, value]] in file order; `prose` is the `#! |` lines,
 * joined as Markdown.
 */
export function docHeader(code) {
  const lines = String(code).match(/[^\n]*\n|[^\n]+$/g) || [];
  let n = 0;
  while (n < lines.length && /^#!(\s|$)/.test(lines[n])) n++;
  const first = n ? (lines[0].match(/^#!\s*(\S+)/) || [])[1] : null;
  if (!n || !first || DIRECTIVES.has(first)) return { doc: '', body: String(code), fields: [], prose: '' };
  // blank lines after the header go with it, so the editor opens on code
  let m = n;
  while (m < lines.length && lines[m].trim() === '') m++;
  const fields = [];
  const prose = [];
  for (const l of lines.slice(0, n)) {
    const t = l.replace(/\n$/, '').replace(/^#!\s?/, '');
    if (t === '|' || t.startsWith('| ')) prose.push(t.slice(2));
    else {
      const k = t.match(/^(\S+)\s*(.*)$/);
      if (k) fields.push([k[1], k[2]]);
    }
  }
  return { doc: lines.slice(0, m).join(''), body: lines.slice(m).join(''), fields, prose: prose.join('\n'), lines: m };
}
