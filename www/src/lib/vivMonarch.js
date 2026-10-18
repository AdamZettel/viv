// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The Monaco grammar for Viv, shared by the workbench (components/VivIde.jsx)
// and the composer's editor (components/VivEditor.jsx). Whichever registers
// `viv` first wins, so both register this one.
//
// A `###` line on its own opens a fenced text block and the next one closes it
// (a post's header and its prose, as lib/highlight.js reads them). The block is
// prose, so it is not tokenized as code: its lines are `prose`, its markdown
// headings `prose.heading`, its `$...$` math `prose.math`, and the fences
// themselves `comment`.
import { VIV_KEYWORDS } from './highlight.js';

export const VIV_TYPES = ['Num', 'Arr', 'CArr', 'Bool', 'Cplx'];

/** The Monarch tokenizer; `builtins` are colored as `type.identifier`. */
export function vivMonarch(builtins = []) {
  return {
    keywords: VIV_KEYWORDS,
    builtins,
    types: VIV_TYPES,
    tokenizer: {
      root: [
        [/^###\s*$/, { token: 'comment', next: '@fence' }],
        [/#[|:~[\]].*$/, 'comment.doc'],
        [/#.*$/, 'comment'],
        [/"[^"]*"?/, 'string'],
        [/[a-zA-Z_][a-zA-Z0-9_]*/, {
          cases: { '@keywords': 'keyword', '@types': 'type.identifier', '@builtins': 'type.identifier', '@default': 'identifier' },
        }],
        [/\d+\.\d*([eE][+-]?\d+)?i?/, 'number.float'],
        [/\d+i?/, 'number'],
        [/[{}()[\]]/, '@brackets'],
        [/[<>=!+\-*/%@]+/, 'operator'],
      ],
      fence: [
        [/^###\s*$/, { token: 'comment', next: '@pop' }],
        [/^#{1,6}\s.*$/, 'prose.heading'],
        [/\$[^$]*\$/, 'prose.math'],
        [/[^$]+/, 'prose'],
        [/\$/, 'prose'],
      ],
    },
  };
}

/** Theme rules for the prose tokens, given the colors of plain text and of muted text. */
export const proseRules = (text, muted) => [
  { token: 'prose', foreground: muted },
  { token: 'prose.heading', foreground: text, fontStyle: 'bold' },
  { token: 'prose.math', foreground: muted },
];
