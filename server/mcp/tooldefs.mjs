// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The five tools as MCP lists them: names, descriptions, input schemas. Shared
// by the feed's server and the Netlify Function (netlify.mjs), which
// differ only in how long a run may take, so the definitions take that number.
// This file imports nothing that needs SQLite or the feed's database.
import { SECTIONS } from './reference.mjs';
// the pack's size, measured when the site builds it (www/scripts/llms-build.mjs)
import PACK from '../../www/src/data/promptPack.json' with { type: 'json' };

export const KINDS = ['post', 'program', 'module', 'builtin', 'example'];

const READ_ONLY = { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false };

/** The tool list, for a server whose runs stop after `runSeconds`. */
export function toolDefs({ runSeconds = 15 } = {}) {
  return [
    {
      name: 'run_viv',
      title: 'Run a Viv program',
      description: `Run a Viv program in the Viv interpreter (the same viv.wasm the site runs in the browser), with a ${runSeconds} s time limit and a memory limit. Returns the lines the program printed with show and print, every test as PASS or FAIL with the failing assertion, and a one-line description of each figure instead of its data points. A program can load modules with a \`#! use lib/<name>\` or \`#! use examples/<name>\` line. Read viv_reference first if you have not written Viv before: it is not Python.`,
      inputSchema: {
        type: 'object',
        properties: { source: { type: 'string', description: 'The whole program, as a .viv file' } },
        required: ['source'],
        additionalProperties: false,
      },
      annotations: { ...READ_ONLY, title: 'Run a Viv program' },
    },
    {
      name: 'check_post',
      title: 'Check a post against the posting rules',
      description: 'Run a post (a .viv file with a fenced ### header) and apply the rules the site applies before publishing: title, date and tags in the header, at least one test and none failing, a figure for a demo or an article, 10 s or less. If the header names a parent with `from:` and `edge:`, the claim is checked too, by running both programs. Returns the errors that would block publishing (none means it would be accepted), the tests, and the program address. Publishes nothing.',
      inputSchema: {
        type: 'object',
        properties: { source: { type: 'string', description: 'The whole post: the ### header block, then the program' } },
        required: ['source'],
        additionalProperties: false,
      },
      annotations: { ...READ_ONLY, title: 'Check a post' },
    },
    {
      name: 'search',
      title: 'Search posts, programs, modules and docs',
      description: `Full-text search over published posts, the example programs, the lib/ modules (load one with \`import <name>\`, lib/ left out and \`/\` written \`.\`), the builtins and the documented examples. Words are matched as prefixes; every word must match, or, when nothing matches all of them, any word. Kinds: ${KINDS.join(', ')}. Solution posts are not searchable.`,
      inputSchema: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Words to look for, e.g. "symmetric eigenvalues" or "landau damping"' },
          kind: { type: 'string', enum: KINDS, description: 'Only this kind of result' },
          limit: { type: 'integer', minimum: 1, maximum: 25, description: 'At most this many results (default 10)' },
        },
        required: ['query'],
        additionalProperties: false,
      },
      annotations: { ...READ_ONLY, title: 'Search' },
    },
    {
      name: 'get_post',
      title: 'Get a published post',
      description: 'A published post by its slug: the full source (header included), the header fields, the tests that passed when the server ran it, what it printed, its lane (ai, draft, reviewed or published) and its lineage (the versions of its family and the checked edges between them). The source of a solution post is not returned, and a problem lists only how many solutions it has.',
      inputSchema: {
        type: 'object',
        properties: { slug: { type: 'string', description: 'The post slug, as in https://viv-lang.org/p/<slug>' } },
        required: ['slug'],
        additionalProperties: false,
      },
      annotations: { ...READ_ONLY, title: 'Get a post' },
    },
    {
      name: 'viv_reference',
      title: 'The Viv language reference',
      description: `How to write Viv, from the prompt pack the site builds (every example in it ran). With no arguments: the grammar, the Python habits that fail and what to write instead, and the constructs the interpreter runs but the C compiler refuses (about ${PACK.coreKTokens}k tokens). Sections: ${Object.keys(SECTIONS).join(', ')}, or all (about ${PACK.kTokens}k tokens). With topic: the documented examples on that topic, with their output.`,
      inputSchema: {
        type: 'object',
        properties: {
          section: { type: 'string', enum: ['core', ...Object.keys(SECTIONS), 'all'], description: 'Which part (default core)' },
          topic: { type: 'string', description: 'An example topic, e.g. "Arrays", "Complex arrays & the FFT", "Stencils and PDE steps"' },
        },
        additionalProperties: false,
      },
      annotations: { ...READ_ONLY, title: 'Viv reference' },
    },
  ];
}
