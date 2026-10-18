// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Prose from a post's header, rendered in the browser: $math$ through KaTeX,
// `code` as code, *emphasis*, the rest escaped. The same grammar the build's
// scripts/prose.mjs renders at build time, so a post reads the same whether it
// came from the repo or from the server.
const esc = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

// A number in programmer's e-notation left in prose or a label (4.3e7,
// 2.061e-9) set as a power of ten: 4.3 × 10<sup>7</sup>. Only a number standing
// alone; a name (SLy5, x1e3) or anything in `code` or $math$ is left as it is.
const SCI = /(?<![\w.])(-?\d+(?:\.\d+)?)[eE]([-+]?)0*(\d+)(?![\w.])/g;
export const sciNotation = (html) => String(html).replace(SCI, (_, m, sign, e) => `${m === '1' ? '' : `${m} × `}10<sup>${sign === '-' ? '\u2212' : ''}${e}</sup>`);

/** TeX in a sentence, as plain text for a tooltip, a page title or a feed:
 *  `$4.3 \times 10^{7}$` reads 4.3 × 10^7, `\kappa` reads κ, `$` goes. */
const GREEK = { alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ε', varepsilon: 'ε', kappa: 'κ', eta: 'η', xi: 'ξ', chi: 'χ', phi: 'φ', varphi: 'φ', Gamma: 'Γ', Lambda: 'Λ', Sigma: 'Σ', Phi: 'Φ', Psi: 'Ψ', Pi: 'Π', Theta: 'Θ', partial: '∂', nabla: '∇', hbar: 'ℏ', sum: '∑', prod: '∏', int: '∫', neq: '≠', ne: '≠', sim: '∼', propto: '∝', in: '∈', ll: '≪', gg: '≫', lesssim: '≲', gtrsim: '≳', rightarrow: '→', mid: '|', quad: ' ', qquad: ' ', lambda: 'λ', mu: 'μ', nu: 'ν', omega: 'ω', pi: 'π', rho: 'ρ', sigma: 'σ', tau: 'τ', psi: 'ψ', zeta: 'ζ', theta: 'θ', Delta: 'Δ', Omega: 'Ω', infty: '∞', times: '×', cdot: '·', le: '≤', leq: '≤', ge: '≥', geq: '≥', approx: '≈', pm: '±', top: 'ᵀ', ell: 'ℓ', ldots: '…', dots: '…', to: '→' };
// `\tfrac14`, `\frac12x`, `\sqrt3`: TeX's one-token arguments, braced, so
// the braced rules below read them
const BARE = (t) => t
  .replace(/\\([td]?frac)\s*([A-Za-z0-9])\s*([A-Za-z0-9])/g, '\\$1{$2}{$3}')
  .replace(/\\([td]?frac)\s*(\{[^{}]*\})\s*([A-Za-z0-9])/g, '\\$1$2{$3}')
  .replace(/\\([td]?frac)\s*([A-Za-z0-9])\s*(\{[^{}]*\})/g, '\\$1{$2}$3')
  .replace(/\\sqrt\s*(\\[A-Za-z]+|[A-Za-z0-9])/g, '\\sqrt{$1}');
export function texToPlain(src) {
  return String(src || '').replace(/\\\$/g, '\u0000').replace(/\$([^$]+)\$/g, (_, t) => BARE(t)
    .replace(/\\left|\\right|\\big|\\Big|\\bigg/g, '')
    .replace(/\\[td]?frac\{([^{}]*)\}\{([^{}]*)\}/g, '$1/$2')
    .replace(/\\sqrt\{([^{}]*)\}/g, '√$1')
    .replace(/\\(?:lVert|rVert|Vert)/g, '‖').replace(/\\[lr]vert/g, '|')
    .replace(/\\(?:text|mathrm|mathbf|operatorname)\{([^{}]*)\}/g, '$1')
    .replace(/\\([A-Za-z]+)/g, (m, w) => (w in GREEK ? GREEK[w] : w))
    .replace(/\\[,;:! ]/g, ' ').replace(/\{,\}/g, ',').replace(/\\%/g, '%')
    .replace(/\^\{([^{}]*)\}/g, '^$1').replace(/_\{([^{}]*)\}/g, '_$1')
    .replace(/[{}]/g, '').replace(/\s+/g, ' ').trim()).replace(/\u0000/g, '$');
}
/** TeX as plain ASCII-leaning text, for a feed reader that shows no math:
 *  `10^{-14}` reads 10^-14, `\sqrt{2}` sqrt(2), `N^{-1/2}` N^(-1/2),
 *  `e_{k+1}` e_(k+1), `\frac{4}{3}n^3` (4/3)n^3, `Q^\top` Q^T. */
export function texToAscii(tex) {
  const group = (g) => (/^[A-Za-z0-9.]+$/.test(g) || /^-?\d+(?:\.\d+)?$/.test(g) ? g : `(${g})`);
  let t = BARE(String(tex || ''));
  // innermost braces first, so nested forms unwind
  for (let i = 0; i < 6; i++) {
    t = t
      .replace(/\\[td]?frac\{([^{}]*)\}\{([^{}]*)\}/g, (_, a, b) => `${group(a)}/${group(b)}`)
      .replace(/\\sqrt\{([^{}]*)\}/g, 'sqrt($1)')
      .replace(/\\(?:text|mathrm|mathbf|mathit|operatorname|mathsf|textrm)\{([^{}]*)\}/g, '$1')
      .replace(/\\left|\\right|\\big|\\Big|\\bigg/g, '')
      .replace(/\^\{([^{}]*)\}/g, (_, g) => `^${group(g.trim())}`)
      .replace(/_\{([^{}]*)\}/g, (_, g) => `_${group(g.trim())}`);
  }
  return t
    .replace(/\^\\top\b|\^\\intercal\b|\^T\b/g, '^T')
    .replace(/\\(?:lVert|rVert|Vert)/g, '||').replace(/\\[lr]vert/g, '|')
    .replace(/\\([A-Za-z]+)/g, (m, w) => (w in GREEK ? GREEK[w] : w))
    .replace(/\\[,;:! ]/g, ' ').replace(/\{,\}/g, ',').replace(/\\%/g, '%').replace(/\\([{}$&#_])/g, '$1')
    .replace(/[{}]/g, '').replace(/\s+/g, ' ').trim();
}

// **strong** before *emphasis*, or the emphasis rule would eat the inner pair.
// A link is [words](url) to a site path, an anchor or an http(s) address;
// anything else (a `javascript:` url) stays text.
const text = (s) => sciNotation(esc(s)).replace(/ -- /g, ' — ')
  .replace(/\*\*([^*\s][^*]*?)\*\*/g, '<strong>$1</strong>')
  .replace(/\*([^*\s][^*]*?)\*/g, '<em>$1</em>')
  .replace(/\[([^\]\n]+)\]\(((?:https?:\/\/|\/|#)[^\s)"]*)\)/g, '<a href="$2">$1</a>');

/** An indented block: a paragraph every line of which starts with four spaces
 *  (or a tab), Markdown's code block. A table of numbers or a formula set out
 *  in columns is written so, and it keeps its lines and columns. */
export const isIndentedBlock = (p) => {
  const ls = String(p || '').split('\n').filter((l) => l.trim());
  return ls.length > 0 && ls.every((l) => /^( {4}|\t)/.test(l));
};
export function indentedBlockHtml(p) {
  const ls = String(p).replace(/\t/g, '    ').split('\n');
  while (ls.length && !ls[ls.length - 1].trim()) ls.pop();
  const cut = Math.min(...ls.filter((l) => l.trim()).map((l) => l.match(/^ */)[0].length));
  return `<pre class="prose-pre"><code>${esc(ls.map((l) => l.slice(cut)).join('\n'))}</code></pre>`;
}

/** Plain text whose only markup is Markdown's code spans: `x` -> <code>x</code>,
 *  the rest escaped. For the doc and reference notes, which are written so. */
export function codeSpans(src) {
  return String(src || '').split(/(`[^`]+`)/).map((p) => (p.length > 2 && p.startsWith('`') && p.endsWith('`') ? `<code>${esc(p.slice(1, -1))}</code>` : esc(p))).join('');
}

let katexPromise = null;
export function loadKatex() {
  if (!katexPromise) katexPromise = import('katex').then((m) => m.default || m).catch(() => null);
  return katexPromise;
}

// `\$` is a literal dollar sign; see scripts/prose.mjs, which does the same at
// build time. It is guarded before the math is found and restored after.
const DOLLAR = '\u0000D\u0000'; // no `$` in the sentinel: the splitter would find it
const guard = (src) => String(src || '').replace(/\\\$/g, DOLLAR);
const unguard = (s, as = '$') => s.split(DOLLAR).join(as);

// Where the math is, by Pandoc's rule for `$`: `$$...$$` is displayed; an
// inline `$...$` opens on a `$` with no space after it and closes on a `$`
// with no space before it and no digit after it. So "costs $5 and $10" and
// "$ 5" stay text, and a formula never swallows the prose between two prices.
// Code spans are found in the same pass, so a `$` inside `code` is code.
// scripts/prose.mjs splits on the same expression.
export const MATH_SPLIT = /(\$\$[^$]+\$\$|\$(?=[^\s$])[^$]*?[^\s$]\$(?!\d)|\$[^\s$]\$(?!\d)|`[^`]+`)/;

/** Render with a KaTeX instance (or null: math is shown as its source). Synchronous. */
export function renderProse(src, katex) {
  const parts = guard(src).split(MATH_SPLIT);
  // punctuation right after inline math stays on the math's line: KaTeX ends in
  // an inline-block, and a browser may break after one, leaving a lone period
  const PUNCT = /^[.,;:!?)\]]+/;
  const glue = parts.map((part, i) => (/^\$[^$]/.test(part) && part.endsWith('$') && katex ? ((parts[i + 1] || '').match(PUNCT) || [''])[0] : ''));
  return strongAcross(parts, parts.map((part, i) => {
    if (i > 0 && glue[i - 1]) part = part.slice(glue[i - 1].length);
    const html = renderPart(part, katex);
    return glue[i] ? `<span class="math-glue" style="white-space:nowrap">${html}${esc(glue[i])}</span>` : html;
  })).join('');
}

/** `**bold with `code` in it**`: the text rule pairs `**` inside one stretch of
 *  text, so a pair with a code span or a formula between its halves is left
 *  as two lone `**`. They are paired here, across the stretches, in order:
 *  one that opens before a non-space with one that closes after a non-space. */
export function strongAcross(parts, html) {
  const isText = (p) => !(p.length > 2 && ((p.startsWith('`') && p.endsWith('`')) || (p.startsWith('$') && p.endsWith('$'))));
  const at = [];
  html.forEach((h, i) => {
    if (!isText(parts[i])) return;
    for (const m of h.matchAll(/\*\*/g)) at.push([i, m.index]);
  });
  if (at.length < 2) return html;
  const out = html.slice();
  const edits = new Map();
  for (let k = 0; k + 1 < at.length; k += 2) {
    const [oi, oj] = at[k];
    const [ci, cj] = at[k + 1];
    if (/\s/.test(out[oi][oj + 2] || ' ') || /\s/.test(out[ci][cj - 1] || ' ')) continue;
    (edits.get(oi) || edits.set(oi, []).get(oi)).push([oj, '<strong>']);
    (edits.get(ci) || edits.set(ci, []).get(ci)).push([cj, '</strong>']);
  }
  for (const [i, es] of edits) {
    es.sort((a, b) => b[0] - a[0]);
    for (const [j, tag] of es) out[i] = out[i].slice(0, j) + tag + out[i].slice(j + 2);
  }
  return out;
}

function renderPart(part, katex) {
    if (part.startsWith('$$') && part.endsWith('$$') && part.length > 4) {
      if (!katex) return `<span class="tex-src">${esc(unguard(part.slice(2, -2)))}</span>`;
      try { return katex.renderToString(unguard(part.slice(2, -2).trim(), '\\$'), { throwOnError: false, output: 'html', displayMode: true }); } catch { return esc(unguard(part)); }
    }
    if (part.startsWith('$') && part.endsWith('$') && part.length > 2 && MATH_SPLIT.test(part) && part.match(MATH_SPLIT)[0] === part) {
      if (!katex) return `<span class="tex-src">${esc(unguard(part.slice(1, -1)))}</span>`;
      try { return katex.renderToString(unguard(part.slice(1, -1), '\\$'), { throwOnError: false, output: 'html' }); } catch { return esc(unguard(part)); }
    }
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2) return `<code>${esc(unguard(part.slice(1, -1)))}</code>`;
    return unguard(text(part));
}

/** One of a post's paragraphs: an indented block as it is written, anything
 *  else as prose. */
export const noteHtml = (p, katex) => (isIndentedBlock(p) ? indentedBlockHtml(p) : renderProse(p, katex));

/** The async form: loads KaTeX when the text has math. */
export async function prose(src) {
  const katex = /\$[^$]+\$/.test(src || '') ? await loadKatex() : null;
  return renderProse(src, katex);
}

// ---- block-level Markdown, for a notebook's prose cells ----------------------
// Paragraphs, `#`/`##`/`###` headings, `-`/`*`/`1.` lists, ``` fenced code,
// four-space indented blocks, and
// `$$ ... $$` display math standing on its own lines; everything inline goes
// through renderProse above. Headings start at h2: the page's title is the h1.
const slugify = (s) => s.toLowerCase().replace(/<[^>]+>/g, '').replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

/** Markdown -> html, synchronously, with a KaTeX instance (or null). */
export function renderMarkdown(src, katex) {
  const lines = String(src || '').replace(/\r/g, '').split('\n');
  const out = [];
  let i = 0;
  const inline = (s) => renderProse(s, katex);
  const LIST = /^\s{0,3}([-*]|\d+[.)])\s+(.*)$/;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    // display math that opens with `$$` on its own line and runs over several
    // lines (an aligned block): taken whole, up to the line that closes it, so
    // a line inside it that starts with `-` or `+` is not read as a list
    if (/^\s*\$\$/.test(line) && (line.match(/\$\$/g) || []).length === 1) {
      const block = [line.trim()];
      i++;
      while (i < lines.length && !/\$\$/.test(lines[i])) block.push(lines[i++]);
      if (i < lines.length) block.push(lines[i++].trim());
      out.push(inline(block.join('\n')));
      continue;
    }
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      const level = Math.min(Math.max(h[1].length, 2), 4);
      const body = inline(h[2].trim());
      out.push(`<h${level} id="${slugify(h[2])}">${body}</h${level}>`);
      i++;
      continue;
    }
    if (/^\s*```/.test(line)) {
      const code = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i++]);
      i++; // the closing fence
      out.push(`<pre class="nb-pre"><code>${esc(code.join('\n'))}</code></pre>`);
      continue;
    }
    // an indented block (four spaces): a table or a formula set out in
    // columns, kept as it is written
    if (/^( {4}|\t)\s*\S/.test(line)) {
      const block = [];
      while (i < lines.length && (/^( {4}|\t)/.test(lines[i]) || (!lines[i].trim() && /^( {4}|\t)\s*\S/.test(lines[i + 1] || '')))) block.push(lines[i++]);
      out.push(indentedBlockHtml(block.join('\n')).replace('class="prose-pre"', 'class="nb-pre prose-pre"'));
      continue;
    }
    // the mark where a post's article view places its figure; a notebook
    // shows each figure under the cell that draws it
    if (line.trim() === '[figure]' && !(lines[i + 1] || '').trim()) { i++; continue; }
    const li = line.match(LIST);
    if (li) {
      const ordered = /\d/.test(li[1]);
      const items = [];
      while (i < lines.length) {
        const m = lines[i].match(LIST);
        if (m && /\d/.test(m[1]) === ordered) { items.push(m[2]); i++; continue; }
        // an indented line continues the item above it
        if (items.length && /^\s{2,}\S/.test(lines[i]) && !LIST.test(lines[i])) { items[items.length - 1] += ' ' + lines[i].trim(); i++; continue; }
        break;
      }
      const tag = ordered ? 'ol' : 'ul';
      out.push(`<${tag}>${items.map((t) => `<li>${inline(t)}</li>`).join('')}</${tag}>`);
      continue;
    }
    // a paragraph: up to a blank line, a heading, a list or a code fence
    const para = [];
    while (i < lines.length && lines[i].trim() && !/^#{1,4}\s/.test(lines[i]) && !/^\s*```/.test(lines[i]) && !(para.length && LIST.test(lines[i]))) para.push(lines[i++].trim());
    const joined = para.join('\n');
    // display math alone in its paragraph is not wrapped in <p>
    if (/^\$\$[^$]+\$\$$/.test(joined)) out.push(inline(joined));
    else out.push(`<p>${inline(joined)}</p>`);
  }
  return out.join('\n');
}

/** The async form: loads KaTeX when the text has math. */
export async function markdown(src) {
  const katex = /\$[^$]+\$/.test(src || '') ? await loadKatex() : null;
  return renderMarkdown(src, katex);
}
