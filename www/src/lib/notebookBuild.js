// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Build-time side of the notebook view: which files get a /nb page, and each
// page's prose rendered ahead of time (KaTeX runs here, so the math is in the
// HTML and the reader's tab does not load KaTeX for it).
//
// A /nb path is a file's repository name: lib/hf/rhf, examples/tov,
// pde/heat_1d, and posts/<slug> for a post.
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, relative } from 'node:path';
import katex from 'katex';
import { splitNotebook } from './notebook.js';
import { renderMarkdown } from './richText.js';

// the site is built from www/, and the repository is its parent
const ROOT = join(process.cwd(), '..');
const DIRS = [
  { head: 'posts', dir: 'posts', title: 'Posts' },
  { head: 'examples', dir: 'lang/viv/examples', title: 'Example programs' },
  { head: 'pde', dir: 'pde', title: 'PDE solvers' },
  { head: 'lib', dir: 'lib', title: 'Modules (lib/)' },
];
// A file bigger than this is fetched by the page instead of written into it:
// lib/hfref holds reference tables of several hundred kilobytes.
export const EMBED_BYTES = 96 * 1024;

const walk = (d) => readdirSync(d).sort().flatMap((f) => {
  const p = join(d, f);
  return statSync(p).isDirectory() ? walk(p) : p.endsWith('.viv') ? [p] : [];
});

/** Every file with a notebook page: { path, head, file, url, workbench, bytes }. */
export function notebookFiles() {
  const out = [];
  for (const { head, dir } of DIRS) {
    const abs = join(ROOT, dir);
    if (!existsSync(abs)) continue;
    for (const f of walk(abs)) {
      const rel = relative(abs, f).split('\\').join('/');
      const bytes = statSync(f).size;
      if (head === 'posts') {
        const m = rel.match(/^(\d{4}-\d{2}-\d{2})-([a-z0-9-]+)\.viv$/);
        if (!m) continue;
        out.push({ path: `posts/${m[2]}`, head, file: f, url: `/programs/posts/${rel}`, workbench: `/workbench?open=posts/${rel}`, bytes });
      } else {
        out.push({ path: `${head}/${rel.replace(/\.viv$/, '')}`, head, file: f, url: `/programs/${head}/${rel}`, workbench: `/workbench?open=${head}/${rel}`, bytes });
      }
    }
  }
  return out;
}

export const notebookGroups = DIRS.map(({ head, title }) => ({ head, title }));

/** { [cell index]: html } for the header's prose and every prose cell. */
export function proseHtmlOf(src) {
  const html = {};
  splitNotebook(src).forEach((c, i) => {
    if (c.kind === 'prose') html[i] = renderMarkdown(c.body, katex);
    if (c.kind === 'header' && c.prose) html[i] = renderMarkdown(c.prose, katex);
  });
  return html;
}

/** Why the interpreter cannot run this file, or ''. Records are compiled-C only
 *  (the tour's section 11 says so), and the interpreter stops at the first one. */
export function expectedError(src) {
  const m = src.match(/^\s*record\s+([A-Z]\w*)\s*\{/m);
  return m
    ? `This program declares a record type (${m[1]}), and records run only in compiled C for now, so the Run buttons here are off. Open it in the workbench and use its C tab to compile it.`
    : '';
}
