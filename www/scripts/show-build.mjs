// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// show-build -- the showcase pages (/show/<slug>), from the programs in
// src/data/showcases.mjs.
//
// Each program carries its own page in its `###` header: `post:` is the title,
// `equation:` the display equation, `summary:` the short answer, and the
// paragraphs after the fields are the problem. This script RUNS each one
// through viv.wasm -- the interpreter a reader's tab runs -- and refuses to
// build if a test fails, a program has no tests, or a program draws no figure
// or animation. What a page shows (the numbers, the figure, the list of passing
// tests, the run time) is that run's output; none of it is typed here.
//
// Runs are kept in www/.show-cache by the hash of the expanded program and of
// viv.wasm, so a page is run again only when its program or the interpreter
// changes.
//
// Writes www/src/data/show.json, and stages for the browser
//   public/programs/show/<slug>.viv      the program as written (the workbench opens it)
//   public/programs/show/<slug>.run.viv  the program with its imports spliced in (the Run button runs it)
//   public/programs/show/<slug>.fig.txt  the build-time run's figure records and shown values (the scenes draw them)
import { readFileSync, writeFileSync, readdirSync, mkdirSync, existsSync } from 'node:fs';
import { dirname, join, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import katex from 'katex';
import { highlight } from '../src/lib/highlight.js';
import { renderMarkdown, texToPlain } from '../src/lib/richText.js';
import { isAnimRecord } from '../src/lib/animParse.js';
import { FIGURE_LINE, HEADER_FIELD } from '../src/lib/postCheck.js';
import { expandUses } from '../src/lib/uses.js';
import { SHOWCASES } from '../src/data/showcases.mjs';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const STAGE = join(WWW, 'public/programs/show');
const CACHE = join(WWW, '.show-cache');
mkdirSync(STAGE, { recursive: true });
mkdirSync(CACHE, { recursive: true });

// modules, resolved as scripts/feed-build.mjs resolves a post's
const DIRS = { dft: 'dft', pde: 'pde', lib: 'lib', examples: 'lang/viv/examples' };
const POSTS = join(ROOT, 'posts');
const postFiles = Object.fromEntries(readdirSync(POSTS).filter((f) => f.endsWith('.viv')).map((f) => [f.replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.viv$/, ''), f]));
const resolve = async (name) => {
  if (name.startsWith('post:')) {
    const f = postFiles[name.slice(5)];
    if (!f) throw new Error(`module \`${name}\`: no post by that slug`);
    return readFileSync(join(POSTS, f), 'utf8');
  }
  const [head, ...rest] = name.split('/');
  const file = DIRS[head] && rest.length ? join(ROOT, DIRS[head], `${rest.join('/')}.viv`) : null;
  if (!file || !existsSync(file)) throw new Error(`module \`${name}\`: no module by that name`);
  return readFileSync(file, 'utf8');
};

const wasmBytes = readFileSync(join(WWW, 'public/viv/viv.wasm'));
const wasmHash = createHash('sha256').update(wasmBytes).digest('hex');
let ex = null;
async function interpreter() {
  if (!ex) ex = (await WebAssembly.instantiate(wasmBytes, {})).instance.exports;
  return ex;
}
function call(e, fn, text) {
  const enc = new TextEncoder().encode(text);
  const ptr = e.viv_alloc(enc.length);
  new Uint8Array(e.memory.buffer, ptr, enc.length).set(enc);
  const packed = fn(ptr, enc.length);
  return new TextDecoder().decode(new Uint8Array(e.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
}

/** The `###` header: its fields, and the prose after them. */
function header(src) {
  const lines = src.split('\n');
  const a = lines.findIndex((l) => l.trim() === '###');
  const b = a < 0 ? -1 : lines.findIndex((l, i) => i > a && l.trim() === '###');
  if (a < 0 || b < 0) return { fields: {}, body: '' };
  const fields = {};
  let i = a + 1;
  let key = null;
  for (; i < b && lines[i].trim(); i++) {
    const m = lines[i].match(HEADER_FIELD);
    if (m) {
      key = m[1];
      fields[key] = m[2].trim();
    } else if (key) fields[key] += ` ${lines[i].trim()}`;
  }
  return { fields, body: lines.slice(i, b).join('\n').trim() };
}

/** The program as the page shows it: no header block, no SPDX lines at the end. */
function bodyOf(src) {
  const lines = src.split('\n');
  const a = lines.findIndex((l) => l.trim() === '###');
  const b = a < 0 ? -1 : lines.findIndex((l, i) => i > a && l.trim() === '###');
  const rest = b >= 0 ? lines.slice(b + 1) : lines;
  return rest.filter((l) => !/^#\s*SPDX-/.test(l)).join('\n').trim() + '\n';
}

// programs the home page runs that are not showcase pages (their pages are elsewhere)
const HOME_RUNS = { 'pde-kdv_soliton': 'pde/kdv_soliton.viv' };

const tex = (t) => katex.renderToString(t, { displayMode: true, throwOnError: true });

const errors = [];
const pages = [];
for (const s of SHOWCASES) {
  const path = join(ROOT, s.file);
  if (!existsSync(path)) {
    errors.push(`${s.slug}: no file ${s.file}`);
    continue;
  }
  const src = readFileSync(path, 'utf8');
  const { fields, body } = header(src);
  for (const k of ['post', 'equation', 'summary']) if (!fields[k]) errors.push(`${s.slug}: no \`${k}:\` in the header of ${s.file}`);
  let expanded;
  try {
    expanded = await expandUses(src, resolve);
  } catch (e) {
    errors.push(`${s.slug}: ${e.message}`);
    continue;
  }
  writeFileSync(join(STAGE, `${s.slug}.viv`), src);
  writeFileSync(join(STAGE, `${s.slug}.run.viv`), expanded);

  const key = createHash('sha256').update(wasmHash).update('\0').update(expanded).digest('hex').slice(0, 32);
  const cached = join(CACHE, `${key}.json`);
  let run;
  if (existsSync(cached)) run = JSON.parse(readFileSync(cached, 'utf8'));
  else {
    const e = await interpreter();
    const t0 = performance.now();
    const out = call(e, e.viv_run, expanded);
    run = { out, seconds: Math.round(((performance.now() - t0) / 1000) * 100) / 100 };
    writeFileSync(cached, JSON.stringify(run));
  }
  const lines = run.out.split('\n');
  const tests = lines.filter((l) => l.startsWith('PASS ')).map((l) => l.slice(5));
  const failed = lines.filter((l) => l.startsWith('FAIL ') || l.startsWith('error'));
  if (failed.length) errors.push(`${s.slug}: ${failed[0]}`);
  if (!tests.length) errors.push(`${s.slug}: no tests ran`);
  const figure = lines.filter((l) => FIGURE_LINE.test(l) || isAnimRecord(l)).join('\n');
  if (!figure) errors.push(`${s.slug}: draws no figure or animation`);
  // the figure records, then the shown values (`name: value`) a scene's readout uses
  const shownLines = lines.filter((l) => !FIGURE_LINE.test(l) && !isAnimRecord(l) && /^[A-Za-z_]\w*: -?[\d.]/.test(l));
  writeFileSync(join(STAGE, `${s.slug}.fig.txt`), [figure, ...shownLines].join('\n'));
  const animated = lines.some((l) => l.startsWith('stage:'));
  const shown = lines
    .filter((l) => !isAnimRecord(l))
    .map((l) => l.match(/^([A-Za-z_]\w*): (-?[\d.]+(?:e[+-]?\d+)?)$/))
    .filter(Boolean)
    .map((m) => ({ name: m[1], value: m[2] }));

  let equationHtml = '';
  let summaryHtml = '';
  let bodyHtml = '';
  try {
    equationHtml = tex(fields.equation || 'x');
    summaryHtml = renderMarkdown(fields.summary || '', katex);
    bodyHtml = renderMarkdown(body, katex);
  } catch (e) {
    errors.push(`${s.slug}: ${e.message}`);
  }
  const program = bodyOf(src);
  const post = s.file.startsWith('posts/') ? basename(s.file).replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.viv$/, '') : null;
  pages.push({
    ...s,
    title: fields.post,
    summaryPlain: texToPlain(fields.summary || '').replace(/`/g, ''),
    equationHtml, summaryHtml, bodyHtml,
    program,
    programHtml: highlight(program.replace(/\n$/, ''), 'viv'),
    lines: program.split('\n').length - 1,
    post,
    tests, figure, animated, shown,
    seconds: run.seconds,
  });
  console.log(`show-build: ${s.slug.padEnd(20)} ${String(tests.length).padStart(2)} tests pass   ${run.seconds.toFixed(2)} s`);
}

// the other programs the home page's scenes run: staged the same way, with
// their build-time figure records, so a scene draws at once
for (const [name, file] of Object.entries(HOME_RUNS)) {
  try {
    const expanded = await expandUses(readFileSync(join(ROOT, file), 'utf8'), resolve);
    writeFileSync(join(STAGE, `${name}.run.viv`), expanded);
    const key = createHash('sha256').update(wasmHash).update('\0').update(expanded).digest('hex').slice(0, 32);
    const cached = join(CACHE, `${key}.json`);
    let run;
    if (existsSync(cached)) run = JSON.parse(readFileSync(cached, 'utf8'));
    else {
      const e = await interpreter();
      const t0 = performance.now();
      const out = call(e, e.viv_run, expanded);
      run = { out, seconds: Math.round(((performance.now() - t0) / 1000) * 100) / 100 };
      writeFileSync(cached, JSON.stringify(run));
    }
    const lines = run.out.split('\n');
    const failed = lines.filter((l) => l.startsWith('FAIL ') || l.startsWith('error'));
    if (failed.length) errors.push(`${name}: ${failed[0]}`);
    const shown = lines.filter((l) => !FIGURE_LINE.test(l) && !isAnimRecord(l) && /^[A-Za-z_]\w*: -?[\d.]/.test(l));
    writeFileSync(join(STAGE, `${name}.fig.txt`), [...lines.filter((l) => FIGURE_LINE.test(l) || isAnimRecord(l)), ...shown].join('\n'));
  } catch (e) {
    errors.push(`${name}: ${e.message}`);
  }
}

if (errors.length) {
  console.error(`show-build: ${errors.length} problem${errors.length > 1 ? 's' : ''}:\n  ${errors.join('\n  ')}`);
  process.exit(1);
}
writeFileSync(join(WWW, 'src/data/show.json'), JSON.stringify({ pages }, null, 1) + '\n');
console.log(`show-build: ${pages.length} showcase pages, ${pages.reduce((n, p) => n + p.tests.length, 0)} tests, every one passing`);
