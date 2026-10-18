// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The code the graph's panel (components/GraphPanel.jsx) shows for a def, a
// step or the top level: the program's statements, without the prose that the
// same source file carries for its post and its lenses. Pure functions, no
// DOM: the page and scripts/graph-browsertest.mjs share them.
//
// Each line of the program text gets a kind:
//
//   code     a statement, or part of one (an end-of-line comment stays on it)
//   blank    an empty line
//   post     a `###` block of a post or notebook, the fences included
//   doc      an annotation comment: `#|` `#||` `#|||` (intent), `#:` (a
//            variable's doc), `#~` `#[` `#]` `#@` (lens marks), `#!` (a header
//            directive such as `#! post`)
//   note     a comment on a line of its own that is kept: one line, at most
//            NOTE_CHARS characters after the `#`, not next to another comment
//   comment  every other comment line: a paragraph of two or more `#` lines,
//            or one longer line
//   test     a top-level `test "..." { ... }` block
//
// The code-only view keeps `code` and `note` lines, and one blank line where
// the source has one or more between them.

/** The longest one-line comment the code-only view keeps, in characters after the `#`. */
export const NOTE_CHARS = 80;

const FENCE = /^\s*###\s*$/;
const DOC = /^\s*#(?:\|+|:|~|\[|\]|@|!)/;
const COMMENT = /^\s*#/;
const TEST = /^test\s+"/;

/**
 * The kind of each line of `lines` (index x - 1 for line x). `moduleOf(x)`,
 * when given, names the module a line belongs to: a `###` block never runs on
 * into the next module, even when one is left open.
 */
export function lineKinds(lines, moduleOf = null) {
  const kinds = new Array(lines.length);
  let fence = false;
  let mod = null;
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i] || '';
    if (moduleOf) {
      const m = moduleOf(i + 1);
      if (m !== mod) {
        mod = m;
        fence = false;
      }
    }
    if (FENCE.test(t)) {
      kinds[i] = 'post';
      fence = !fence;
      continue;
    }
    if (fence) {
      kinds[i] = 'post';
      continue;
    }
    if (TEST.test(t)) {
      // a test block runs to its closing brace
      let depth = 0;
      let j = i;
      for (; j < lines.length; j++) {
        for (const c of lines[j] || '') {
          if (c === '{') depth++;
          else if (c === '}') depth--;
        }
        kinds[j] = 'test';
        if (depth <= 0 && (lines[j] || '').includes('}')) break;
      }
      i = j;
      continue;
    }
    if (!t.trim()) kinds[i] = 'blank';
    else if (DOC.test(t)) kinds[i] = 'doc';
    else if (COMMENT.test(t)) kinds[i] = 'comment';
    else kinds[i] = 'code';
  }
  // a comment line alone between code or blank lines, and short, is a note
  for (let i = 0; i < lines.length; i++) {
    if (kinds[i] !== 'comment') continue;
    let j = i;
    while (j + 1 < lines.length && (kinds[j + 1] === 'comment' || kinds[j + 1] === 'doc')) j++;
    const one = j === i && kinds[i - 1] !== 'doc';
    if (one && lines[i].trim().replace(/^#\s?/, '').length <= NOTE_CHARS) kinds[i] = 'note';
    i = j;
  }
  return kinds;
}

const cache = new WeakMap();
/** The kinds of the lines of an explorer index's program, computed once. */
export function kindsOf(ix) {
  let k = cache.get(ix);
  if (!k) {
    k = lineKinds(ix.programLines, (x) => ix.toModule(x)?.module ?? null);
    cache.set(ix, k);
  }
  return k;
}

/**
 * The lines of `xs` (program line numbers, ascending) the code-only view
 * shows: code and notes, and one blank line where the source has only blank
 * lines between two shown lines (where it left something out, the panel's
 * gap mark separates them instead).
 */
export function codeOnly(kinds, xs) {
  const out = [];
  let blank = 0;
  let skipped = false;
  let prev = 0;
  for (const x of xs) {
    if (prev && x !== prev + 1) skipped = true;
    prev = x;
    const k = kinds[x - 1];
    if (k === 'code' || k === 'note') {
      if (blank && !skipped && out.length) out.push(blank);
      blank = 0;
      skipped = false;
      out.push(x);
    } else if (k === 'blank') {
      if (!blank) blank = x;
    } else skipped = true;
  }
  return out;
}

/**
 * The comment lines right above a def's first line (its `#|` intent, its
 * `#@` stamp, a plain comment on it), up to the first line that is not a
 * comment: [first, last], or null.
 */
export function docAbove(lines, defLine) {
  let x = defLine - 1;
  while (x >= 1 && COMMENT.test(lines[x - 1] || '') && !FENCE.test(lines[x - 1] || '')) x--;
  return x + 1 < defLine ? [x + 1, defLine - 1] : null;
}

/** A def's inline intent, `#|` what, `#||` how, `#|||` why, read from the comments above it. */
export function inlineIntent(lines, defLine) {
  const r = docAbove(lines, defLine);
  const out = { what: '', how: '', why: '' };
  if (!r) return out;
  const key = ['', 'what', 'how', 'why'];
  for (let x = r[0]; x <= r[1]; x++) {
    const m = /^\s*#(\|{1,3})(?!\|)\s?(.*)$/.exec(lines[x - 1] || '');
    if (!m) continue;
    const k = key[m[1].length];
    out[k] = out[k] ? `${out[k]} ${m[2].trim()}` : m[2].trim();
  }
  return out;
}

/**
 * The first sentence of a prose text and the rest: a period, question or
 * exclamation mark followed by a space and a capital, a backtick, a `$` or an
 * opening parenthesis, outside `$...$` math and backtick code.
 */
export function firstSentence(text) {
  const s = String(text || '').trim();
  let math = false;
  let code = false;
  for (let i = 0; i < s.length - 2; i++) {
    const c = s[i];
    if (c === '\\') {
      i++;
      continue;
    }
    if (c === '`' && !math) code = !code;
    else if (c === '$' && !code) math = !math;
    else if (!math && !code && '.?!'.includes(c) && s[i + 1] === ' ' && /[A-Z`$(]/.test(s[i + 2])) {
      // not an abbreviation such as "e.g. " or "i.e. "
      if (c === '.' && /(?:\b[a-z]\.[a-z]|\bet al|\bvs|\bcf)$/i.test(s.slice(Math.max(0, i - 6), i))) continue;
      return [s.slice(0, i + 1), s.slice(i + 2).trim()];
    }
  }
  return [s, ''];
}
