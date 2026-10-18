// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The `#!` header of a teaching program, and its prose -> html.
//
// Ladders and PDE solvers carry their own page in their source:
//
//     #! <key> <one line>            metadata: `#! goal ...`, `#! method ...`
//     #! | prose, with $math$, `code` and *emphasis*; a bare `#! |` ends a paragraph
//
// Math goes through KaTeX HERE, at build time, and a formula KaTeX cannot parse
// fails the build -- so no page ships a formula that renders as red source.
import katex from 'katex';
import { MATH_SPLIT, sciNotation, strongAcross } from '../src/lib/richText.js';

export const esc = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

// the same inline rules as src/lib/richText.js: **strong**, *emphasis*, and
// [words](url) for a site path, an anchor or an http(s) address
const text = (s) => sciNotation(esc(s)).replace(/ -- /g, ' — ')
  .replace(/\*\*([^*\s][^*]*?)\*\*/g, '<strong>$1</strong>')
  .replace(/\*([^*\s][^*]*?)\*/g, '<em>$1</em>')
  .replace(/\[([^\]\n]+)\]\(((?:https?:\/\/|\/|#)[^\s)"]*)\)/g, '<a href="$2">$1</a>');

// `\$` is a literal dollar sign -- money, not mathematics. It is taken out of
// the text before the math is found, so `\$1000` cannot open a formula, and put
// back afterwards: as a dollar sign in prose and in `code`, and as LaTeX's own
// `\$` inside a formula.
const DOLLAR = '\u0000D\u0000'; // no `$` in the sentinel: the splitter would find it
const guard = (src) => String(src).replace(/\\\$/g, DOLLAR);
const unguard = (s, as = '$') => s.split(DOLLAR).join(as);

/** Prose -> html: `$...$` inline and `$$...$$` displayed, both through KaTeX;
 *  `code` as code, *x* as emphasis, `\$` a dollar sign, the rest escaped. */
export const prose = (src) => {
  const parts = guard(src).split(MATH_SPLIT); // Pandoc's rule for `$` (src/lib/richText.js)
  // punctuation right after inline math stays on its line (see src/lib/richText.js)
  const glue = parts.map((p, i) => (/^\$[^$]/.test(p) ? ((parts[i + 1] || '').match(/^[.,;:!?)\]]+/) || [''])[0] : ''));
  return strongAcross(parts, parts.map((part, i) => {
    if (i > 0 && glue[i - 1]) part = part.slice(glue[i - 1].length);
    const html = prosePart(part);
    return glue[i] ? `<span class="math-glue" style="white-space:nowrap">${html}${esc(glue[i])}</span>` : html;
  })).join('');
};

function prosePart(part) {
  if (part.startsWith('$$') && part.endsWith('$$') && part.length > 4) return katex.renderToString(unguard(part.slice(2, -2).trim(), '\\$'), { throwOnError: true, output: 'html', displayMode: true });
  if (part.startsWith('$') && part.length > 2 && (part.match(MATH_SPLIT) || [])[0] === part) return katex.renderToString(unguard(part.slice(1, -1), '\\$'), { throwOnError: true, output: 'html' });
  if (part.startsWith('`') && part.endsWith('`') && part.length > 2) return `<code>${esc(unguard(part.slice(1, -1)))}</code>`;
  return unguard(text(part));
}

/** A display equation. */
export const equation = (tex) => katex.renderToString(unguard(guard(tex), '\\$'), { throwOnError: true, output: 'html', displayMode: true });

/** The value of `#! <key> ...`, or ''. */
export const meta = (src, key) => (src.match(new RegExp(`^#! ${key}\\s+(.+)$`, 'm')) || [])[1]?.trim() || '';

/** The `#! |` paragraphs, as raw text. */
export function notesOf(src) {
  const paras = [];
  let cur = [];
  const flush = () => { if (cur.length) paras.push(cur.join(' ')); cur = []; };
  for (const l of src.split('\n')) {
    const m = l.match(/^#! \|\s?(.*)$/);
    if (!m) continue;
    if (!m[1].trim()) flush(); else cur.push(m[1]);
  }
  flush();
  return paras;
}

/** The program with its `#!` header taken off: what is left is what runs. */
export const withoutHeader = (src) => src.split('\n').filter((l) => !l.startsWith('#!')).join('\n').replace(/^\n+/, '');
