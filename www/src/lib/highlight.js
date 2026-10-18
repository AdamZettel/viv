// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The site's one syntax highlighter.
//
// Plain JavaScript with no dependencies, so the SAME tokenizer runs when Astro
// builds a static page and in the reader's tab, where the front page recolors
// the C it emits on every keystroke. Every code block on the site goes through
// here and comes out in the same token classes (tk-*), whose colors are six
// variables in viv.css.
//
// It is a tokenizer, not a parser: comments, strings, numbers, words. A word is
// a keyword, a type, a builtin, a function (it is followed by an open paren), or
// nothing. That is all a reader's eye needs, and it never changes the text --
// highlight(src) with the tags stripped is src, character for character, which
// the editable cells rely on.
//
// Viv's builtin list is not typed here. It is builtins.json, which
// scripts/probe-builtins.mjs MEASURES by running each one through the compiler.
// Its keywords are vivKeywords.json, the compiler's own list (features.rs
// KEYWORDS, written by scripts/features-build.mjs).
import builtins from '../data/builtins.json' with { type: 'json' };
import vivKeywords from '../data/vivKeywords.json' with { type: 'json' };

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const span = (cls, s) => `<span class="tk-${cls}">${esc(s)}</span>`;
// a token over several lines, one span per line, so a caller can split the
// highlighted text on newlines (line-numbered views) without open spans
const spanLines = (cls, s) => s.split('\n').map((t) => (t ? span(cls, t) : '')).join('\n');
const set = (words) => new Set(words.split(/\s+/).filter(Boolean));

export const VIV_BUILTINS = new Set(
  builtins.groups.flatMap((g) => g.items.map((i) => i.name.split(/[ (]/)[0])).concat(['figure', 'plot', 'assert_near', 'assert_true', 'assert_gt', 'assert_lt', 'assert_allclose', 'vline', 'fade_out', 'apart', 'min', 'bars', 'histogram', 'circle', 'rect', 'segment', 'poly', 'frame']));

// The parser's soft keywords that are also ordinary names in a program
// (`step`, `start`, `key`, `at` ...): a keyword only inside one construct, so
// they are left uncolored. `Int` and `Str` are colored as types.
const VIV_SOFT_NAMES = set('by reversed step debug placeholder phrase starting with watch hold trace within per iteration at key sep start _');
const VIV_TYPE_WORDS = set('Int Str');

/** The words colored as Viv keywords, here and in the Monaco editors. */
export const VIV_KEYWORDS = vivKeywords.keywords.filter((w) => !VIV_SOFT_NAMES.has(w) && !VIV_TYPE_WORDS.has(w));

const LANGS = {
  viv: {
    line: '#',
    fence: true, // `###` ... `###` is a block of text, not program
    doc: /^#(?:\|+|:|~|\[|\])/, // annotation comments: #| #|| #: #~ #[ #]
    strings: '"',
    triple: true, // """ ... """ spans lines (the lexer's triple-quoted string)
    keywords: new Set(VIV_KEYWORDS),
    types: new Set(['Num', 'Arr', 'CArr', 'Bool', 'Cplx', ...VIV_TYPE_WORDS]),
    builtins: VIV_BUILTINS,
    number: /^\d+\.?\d*(?:[eE][+-]?\d+)?i?/,
  },
  c: {
    line: '//', block: ['/*', '*/'], strings: '"\'', preproc: true,
    keywords: set('if else for while do return break continue switch case default goto sizeof typedef struct union enum static const extern inline volatile register restrict'),
    types: set('void int double float char long short unsigned signed size_t complex bool viv_arr viv_fn viv_clos1 FILE'),
    number: /^(?:0x[0-9a-fA-F]+|\d+\.?\d*(?:[eE][+-]?\d+)?)[fFuUlL]*/,
  },
  rust: {
    line: '//', block: ['/*', '*/'], strings: '"',
    keywords: set('fn let mut pub use mod impl trait struct enum match if else for while loop in return break continue where as const static ref move self Self crate super dyn unsafe async await type'),
    types: set('u8 u16 u32 u64 usize i8 i16 i32 i64 isize f32 f64 bool str String Vec Option Result Box Rc RefCell HashMap Some None Ok Err'),
    number: /^\d[\d_]*\.?[\d_]*(?:[eE][+-]?\d+)?(?:_?[fiu]\d+|usize)?/,
  },
  js: {
    line: '//', block: ['/*', '*/'], strings: '"\'`',
    keywords: set('const let var function return if else for while do break continue new class extends import export from default await async try catch finally throw typeof instanceof of in this null undefined true false'),
    types: set('WebAssembly Uint8Array Float64Array TextEncoder TextDecoder Promise Map Set Array Object Number String Math JSON console document window fs'),
    number: /^(?:0x[0-9a-fA-F]+|\d+\.?\d*(?:[eE][+-]?\d+)?)n?/,
  },
  python: {
    line: '#', strings: '"\'', triple: true,
    keywords: set('def return if elif else for while in not and or import from as class with try except finally raise lambda yield pass break continue None True False is'),
    types: set('int float str list dict tuple set np'),
    number: /^\d+\.?\d*(?:[eE][+-]?\d+)?j?/,
  },
  wat: {
    line: ';', strings: '"',
    keywords: set('module func param result local global memory export import call return if else end loop block br br_if'),
    types: set('i32 i64 f32 f64'),
    number: /^-?\d+\.?\d*/,
    wordRe: /^[a-zA-Z_$][\w$.]*/,
  },
};
LANGS.ts = LANGS.js;
LANGS.javascript = LANGS.js;
LANGS.py = LANGS.python;

function code(src, L) {
  let out = '';
  let i = 0;
  const n = src.length;
  const wordRe = L.wordRe || /^[a-zA-Z_][a-zA-Z0-9_]*/;
  while (i < n) {
    const rest = src.slice(i);
    let m;
    // Viv's fenced text block: `###` on its own line, to the next one. It is
    // prose (a post's header and notes), so it is dimmed like a comment rather
    // than tokenized as code.
    if (L.fence && rest.startsWith('###') && (i === 0 || src[i - 1] === '\n') && /^###[ \t]*(\n|$)/.test(rest)) {
      const close = rest.slice(3).search(/\n###[ \t]*(\n|$)/);
      const stop = close < 0 ? n : i + 3 + close + rest.slice(3 + close + 1).indexOf('\n') + 2;
      const block = src.slice(i, close < 0 ? n : stop);
      out += spanLines('com', block);
      i += block.length;
      continue;
    }
    if (L.block && rest.startsWith(L.block[0])) {
      const end = src.indexOf(L.block[1], i + L.block[0].length);
      const stop = end < 0 ? n : end + L.block[1].length;
      out += spanLines('com', src.slice(i, stop));
      i = stop;
    } else if (L.preproc && rest[0] === '#' && /^[ \t]*$/.test(src.slice(src.lastIndexOf('\n', i - 1) + 1, i))) {
      m = rest.match(/^#[^\n]*/);
      out += span('pre', m[0]);
      i += m[0].length;
    } else if (rest.startsWith(L.line)) {
      m = rest.match(/^[^\n]*/);
      out += span(L.doc && L.doc.test(m[0]) ? 'doc' : 'com', m[0]);
      i += m[0].length;
    } else if (L.triple && (rest.startsWith('"""') || (L.strings.includes("'") && rest.startsWith("'''")))) {
      // a triple-quoted string runs over lines to the same three quotes
      const q3 = rest.slice(0, 3);
      let j = 3;
      while (j < rest.length && !rest.startsWith(q3, j)) j += rest[j] === '\\' ? 2 : 1;
      const lit = rest.slice(0, Math.min(j + 3, rest.length));
      out += spanLines('str', lit);
      i += lit.length;
    } else if (L.strings.includes(rest[0])) {
      const q = rest[0];
      let j = 1;
      while (j < rest.length && rest[j] !== q && (q === '`' || rest[j] !== '\n')) j += rest[j] === '\\' ? 2 : 1;
      const lit = rest.slice(0, Math.min(j + 1, rest.length));
      out += spanLines('str', lit);
      i += lit.length;
    } else if ((m = rest.match(L.number)) && m[0] && !/[a-zA-Z_]/.test(src[i - 1] || '')) {
      out += span('num', m[0]);
      i += m[0].length;
    } else if ((m = rest.match(wordRe))) {
      const w = m[0];
      const after = src.slice(i + w.length).match(/^\s*\(/);
      if (L.keywords.has(w)) out += span('kw', w);
      else if (L.types && L.types.has(w)) out += span('type', w);
      else if (L.builtins && L.builtins.has(w) && after) out += span('fn', w);
      else if (after) out += span('call', w);
      else out += esc(w);
      i += w.length;
    } else if ((m = rest.match(/^(?:->|=>|[<>=!+\-*\/@%&|^~?:]+)/))) {
      out += span('op', m[0]);
      i += m[0].length;
    } else {
      out += esc(rest[0]);
      i += 1;
    }
  }
  return out;
}

// One shell command: the program, its flags, strings and a trailing comment.
function shellCommand(cmd) {
  let out = '';
  let first = true;
  for (const part of cmd.split(/(\s+|"[^"]*"|'[^']*')/)) {
    if (!part) continue;
    if (/^\s+$/.test(part)) out += part;
    else if (/^["']/.test(part)) out += span('str', part);
    else if (part.startsWith('#')) { out += span('com', cmd.slice(cmd.indexOf(part))); break; }
    else if (first) { out += span('fn', part); first = false; }
    else if (/^--?[a-zA-Z]/.test(part)) out += span('flag', part);
    else if (/^(&&|\|\||\||>|>>|<|;|\\)$/.test(part)) { out += span('op', part); first = /^(&&|\|\||\||;)$/.test(part); }
    else out += esc(part);
  }
  return out;
}

// `sh`: every line is a command. `console`: a session -- lines that start with
// `$ ` are commands, the rest is what the tool printed, and stays as printed.
function shell(src, session) {
  let continued = false;
  return src.split('\n').map((line) => {
    const prompt = line.match(/^(\s*\$ )(.*)$/);
    const isCmd = session ? prompt || continued : true;
    if (!isCmd) return line.trim().startsWith('#') && !session ? span('com', line) : `<span class="tk-out">${esc(line)}</span>`;
    const body = prompt ? prompt[2] : line;
    const html = (prompt ? span('prompt', prompt[1]) : '') + (body.trim().startsWith('#') ? span('com', body) : shellCommand(body));
    continued = /\\\s*$/.test(line);
    return html;
  }).join('\n');
}

function html(src) {
  return esc(src)
    .replace(/(&lt;!--[\s\S]*?--&gt;)/g, '<span class="tk-com">$1</span>')
    .replace(/(&lt;\/?)([a-zA-Z][\w-]*)((?:\s+[\w:-]+(?:=(?:"[^"]*"|'[^']*'|[^\s&]+))?)*)\s*(\/?&gt;)/g, (all, open, tag, attrs, close) =>
      `<span class="tk-op">${open}</span><span class="tk-kw">${tag}</span>${attrs.replace(/([\w:-]+)(=)("[^"]*"|'[^']*'|[^\s&]+)?/g,
        (a, name, eq, val) => `<span class="tk-type">${name}</span>${eq}${val ? `<span class="tk-str">${val}</span>` : ''}`)}<span class="tk-op">${close}</span>`);
}

function json(src) {
  return esc(src)
    .replace(/("(?:[^"\\]|\\.)*")(\s*:)?/g, (all, s, colon) => (colon ? `<span class="tk-type">${s}</span>${colon}` : `<span class="tk-str">${s}</span>`))
    .replace(/\b(-?\d+\.?\d*(?:[eE][+-]?\d+)?|true|false|null)\b(?![^<]*<\/span>)/g, '<span class="tk-num">$1</span>');
}

/** Highlight `src` as `lang`. Unknown or missing languages come back escaped and otherwise untouched. */
export function highlight(src, lang = '') {
  const l = String(lang || '').toLowerCase();
  if (l === 'sh' || l === 'bash' || l === 'shell' || l === 'zsh') return shell(src, false);
  if (l === 'console' || l === 'shellsession' || l === 'session') return shell(src, true);
  if (l === 'html' || l === 'xml') return html(src);
  if (l === 'json') return json(src);
  if (LANGS[l]) return code(src, LANGS[l]);
  return esc(src);
}

/** For a fence with no language: say what it looks like, or '' when unsure. Conservative on purpose. */
export function guessLanguage(src) {
  if (/^\s*\$ /m.test(src)) return 'console';
  if (/#include|^\s*(static|typedef|double|int|void)\b[^\n]*[;{]\s*$/m.test(src)) return 'c';
  if (/\b(fn|impl|pub|let mut|enum|struct)\b[^\n]*[{;(]/.test(src) && /;\s*$|\{\s*$/m.test(src) && !/^\s*def /m.test(src)) return 'rust';
  if (/^\s*(def \w+\(|show |narrate |test "|rung "|record \w+ \{|for \w+ to |#\||#:|#\[)/m.test(src)) return 'viv';
  return '';
}

export const LANGUAGES = [...Object.keys(LANGS), 'sh', 'console', 'html', 'json'];
