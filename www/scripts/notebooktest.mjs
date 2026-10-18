// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Tests for the notebook splitter (src/lib/notebook.js).
//
//     npm run notebooktest
//
// 1. Small cases, each with the cells it must produce.
// 2. Every .viv in the repository splits and joins back to itself byte for
//    byte, and its code cells' line ranges point at the lines they hold.
// 3. For every file with prose cells: running the cells one after another
//    through the interpreter's incremental session (viv_run_cell, as the
//    notebook view does) prints, cell by cell, exactly what running the whole
//    file prints -- so each cell's output is the part of the program's output
//    that cell produced, and nothing is lost or repeated. It also checks that
//    each cell after the first ran alone rather than rebuilding the ones above.
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, dirname, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { splitNotebook, joinCells, runUnits, withCode, isNotebook, docHeader } from '../src/lib/notebook.js';
import { expandUses } from '../src/lib/uses.js';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
let fails = 0;
let checks = 0;
const ok = (cond, what) => {
  checks++;
  if (!cond) { fails++; console.log(`FAIL ${what}`); }
};
const kinds = (cells) => cells.map((c) => (c.kind === 'code' && c.empty ? 'blank' : c.kind)).join(' ');

// ---- 1. small cases ---------------------------------------------------------
{
  const src = 'x = 1.0\nshow x\n';
  const c = splitNotebook(src);
  ok(kinds(c) === 'code', 'no fences: one code cell');
  ok(c[0].code === 'x = 1.0\nshow x' && c[0].start === 1 && c[0].end === 2, 'no fences: the code and its lines');
  ok(!isNotebook(c), 'no fences: not a notebook');
}
{
  const src = '###\npost: A title\ndate: 2026-01-01\n\nThe lede.\n###\n\na = 2.0\nshow a\n\n###\n## Next\n\nProse, with $x^2$.\n###\n\nb = a * a\nshow b\n';
  const c = splitNotebook(src);
  ok(kinds(c) === 'header code prose code', `header, code, prose, code (got ${kinds(c)})`);
  ok(c[0].fields.post === 'A title' && c[0].prose === 'The lede.', 'the header: fields, then prose');
  ok(c[1].code === 'a = 2.0\nshow a' && c[1].codeStart === 8, 'the first code cell starts on line 8');
  ok(c[2].body === '## Next\n\nProse, with $x^2$.' && c[2].start === 11 && c[2].end === 15, 'the prose cell, lines 11-15');
  ok(c[3].code === 'b = a * a\nshow b' && c[3].codeStart === 17, 'the second code cell starts on line 17');
  ok(isNotebook(c), 'a notebook');
  const u = runUnits(c);
  ok(u.length === 2 && u.map((x) => x.text).join('\n') + '\n' === src, 'the run units join to the file');
  ok(u[1].line === 11, 'the second unit starts at the prose above its code');
  const edited = c.map((x, i) => (i === 1 ? withCode(x, 'a = 3.0\nshow a') : x));
  ok(joinCells(edited) === src.replace('a = 2.0', 'a = 3.0'), 'an edit keeps the blank lines around the code');
}
{
  // blank lines before the header belong to it; a fence that is never closed
  // runs to the end of the file, as it does in the lexer
  const src = '\n\n###\npost: T\n###\nx = 1.0\n###\nunclosed prose\nmore';
  const c = splitNotebook(src);
  ok(kinds(c) === 'header code prose', `leading blanks and an unclosed fence (got ${kinds(c)})`);
  ok(c[0].start === 1 && c[2].body === 'unclosed prose\nmore', 'the unclosed block holds the rest');
}
{
  // a fence opened in the middle of a file is prose, not a header
  const src = 'x = 1.0\n###\nnot a header\n###\ny = 2.0';
  ok(kinds(splitNotebook(src)) === 'code prose code', 'a fence after code is a prose cell');
}
{
  // the lexer's rules: an opening fence may be indented, a closing one may
  // not; `### Title` is not a fence at all
  const src = 'a = 1.0\n  ###\nprose\n  ###\nstill prose\n###\n### Not a fence\nb = 1.0';
  const c = splitNotebook(src);
  ok(kinds(c) === 'code prose code', `indented fences (got ${kinds(c)})`);
  ok(c[1].body === 'prose\n  ###\nstill prose', 'an indented line does not close a block');
  ok(c[2].code === '### Not a fence\nb = 1.0', '`### Title` is a comment line');
}
{
  // two prose blocks in a row: the blank code cell between them is kept for
  // the round trip, marked empty, and gets no run unit
  const src = 'a = 1.0\n###\none\n###\n\n###\ntwo\n###\nb = 2.0\n';
  const c = splitNotebook(src);
  ok(kinds(c) === 'code prose blank prose code', `an empty code cell (got ${kinds(c)})`);
  ok(joinCells(c) === src && runUnits(c).length === 2, 'no unit for the empty cell');
  ok(runUnits(c).map((x) => x.text).join('\n') + '\n' === src, 'the units still join to the file');
}
{
  ok(joinCells(splitNotebook('')) === '' && splitNotebook('').length === 0, 'an empty file');
  ok(joinCells(splitNotebook('x\r\ny\r\n')) === 'x\r\ny\r\n', 'CRLF line ends survive');
}

// ---- 2. every file in the repository ------------------------------------------
const walk = (d) => readdirSync(d).flatMap((f) => {
  const p = join(d, f);
  if (f === 'target' || f === 'node_modules' || f.startsWith('.')) return [];
  return statSync(p).isDirectory() ? walk(p) : p.endsWith('.viv') ? [p] : [];
});
const files = ['posts', 'lib', 'pde', 'dft', 'lang/viv/examples', 'ladders'].map((d) => join(ROOT, d)).filter(existsSync).flatMap(walk);
const notebooks = [];
let headers = 0;
for (const f of files) {
  const src = readFileSync(f, 'utf8');
  const cells = splitNotebook(src);
  const name = relative(ROOT, f);
  ok(joinCells(cells) === src, `${name}: the cells join back to the file`);
  const lines = src.split('\n');
  for (const c of cells) {
    if (c.kind !== 'code' || c.empty) continue;
    const first = c.code.split('\n')[0];
    ok(lines[c.codeStart - 1] === first, `${name}: a code cell's first line is line ${c.codeStart}`);
    const h = docHeader(c.code);
    ok(h.doc + h.body === c.code, `${name}: a cell's header and code join back to the cell`);
    if (h.doc) headers++;
    // a notebook's code comments show as source, so math in them shows as TeX;
    // the prose cells are where math renders
    if (isNotebook(cells)) {
      const comments = c.code.split('\n').filter((l) => /^\s*#(?!!)/.test(l));
      const tex = comments.filter((l) => /\$[^$]+\$/.test(l));
      // math split over two comment lines shows as TeX too: an odd number of $ on a line opens it
      const split = comments.filter((l) => ((l.match(/\$/g) || []).length % 2 === 1));
      ok(!tex.length && !split.length, `${name}: no $math$ in a notebook's code comments (${tex[0] || split[0] || ''})`);
    }
  }
  const units = runUnits(cells);
  if (units.length) ok(units.map((u) => u.text).join('\n') === joinCells(cells.slice(0, units[units.length - 1].index + 1)).replace(/\n$/, ''), `${name}: the run units join to the file`);
  if (isNotebook(cells)) notebooks.push({ name, src, cells, units });
}
ok(headers >= 15, `the PDE solvers' #! headers are found and rendered as text (found ${headers})`);
{
  const h = docHeader('#! pde Heat\n#! equation u_t = u_{xx}\n#! | A rod, $\\sin(\\pi x)$.\n#! |\n#! | Second.\n\npi = 3.0\n');
  ok(h.fields.length === 2 && h.fields[1][0] === 'equation' && h.prose === 'A rod, $\\sin(\\pi x)$.\n\nSecond.' && h.body === 'pi = 3.0\n', 'docHeader: fields, prose, and the code under them');
  ok(docHeader('import nn\nx = 1.0').doc === '', 'docHeader: an import line stays code');
}

// ---- 3. cell outputs are the program's output, cell by cell -------------------
const wasm = readFileSync(join(WWW, 'public/viv/viv.wasm'));
const { instance } = await WebAssembly.instantiate(wasm, {});
const ex = instance.exports;
const read = (packed) => new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
function run(text) {
  const b = new TextEncoder().encode(text);
  const ptr = ex.viv_alloc(b.length);
  new Uint8Array(ex.memory.buffer, ptr, b.length).set(b);
  return read(ex.viv_run(ptr, b.length));
}
function runCell(prefix, cell) {
  const enc = new TextEncoder();
  const pb = enc.encode(prefix);
  const cb = enc.encode(cell);
  const ptr = ex.viv_alloc(pb.length + cb.length);
  const mem = new Uint8Array(ex.memory.buffer, ptr, pb.length + cb.length);
  mem.set(pb, 0);
  mem.set(cb, pb.length);
  const t0 = performance.now();
  const out = read(ex.viv_run_cell(ptr, pb.length + cb.length, pb.length));
  return { out, ms: performance.now() - t0 };
}
// the program's output without the call graph and its trace, which describe
// a whole run and so are not a cell's to show
const program = (out) => out.split('\n').filter((l) => !/^__(graph|trace|steps|story)__ /.test(l)).join('\n').replace(/\n+$/, '');
const resolve = async (name) => {
  if (name.startsWith('post:')) {
    const f = readdirSync(join(ROOT, 'posts')).find((x) => x.replace(/^\d{4}-\d{2}-\d{2}-/, '') === `${name.slice(5)}.viv`);
    return readFileSync(join(ROOT, 'posts', f), 'utf8');
  }
  const [head, ...rest] = name.split('/');
  const dir = { dft: 'dft', pde: 'pde', lib: 'lib', examples: 'lang/viv/examples' }[head];
  return readFileSync(join(ROOT, dir, `${rest.join('/')}.viv`), 'utf8');
};
for (const nb of notebooks) {
  const seen = new Set();
  const texts = [];
  for (const u of nb.units) texts.push(await expandUses(u.text, resolve, [], seen));
  const whole = program(run(texts.join('\n')));
  const parts = [];
  let alone = 0;
  for (let j = 0; j < texts.length; j++) {
    const { out } = runCell(texts.slice(0, j).join('\n'), texts[j]);
    parts.push(program(out));
    // a cell that continued the session runs only itself: running it again
    // with the same prefix (which rebuilds the prefix first) prints the same
    if (j > 0 && program(runCell(texts.slice(0, j).join('\n'), texts[j]).out) === parts[j]) alone++;
  }
  const joined = parts.filter((p) => p !== '').join('\n');
  ok(joined === whole, `${nb.name}: the cells' outputs, in order, are the whole program's output`);
  if (joined !== whole) {
    const a = joined.split('\n');
    const b = whole.split('\n');
    const at = a.findIndex((l, i) => l !== b[i]);
    console.log(`     first difference at output line ${at + 1}:\n       cells: ${a[at]}\n       whole: ${b[at]}`);
  }
  ok(alone === texts.length - 1, `${nb.name}: a re-run of each cell prints what its first run did`);
  console.log(`notebooktest: ${nb.name}: ${nb.cells.filter((c) => c.kind === 'prose').length} prose cells, ${texts.length} code cells, output attributed cell by cell`);
}

// rand() carries from cell to cell: a two-cell notebook drawing in each cell
// prints what the whole program prints, whether the second cell continues the
// session or rebuilds the first (the MINSTD state is part of the session)
{
  const src = '# two cells that draw\nsrand(2026.0)\nfirst = rand()\nshow first\n\n###\nThe second cell draws the next number.\n###\n\nsecond = rand()\nshow second\n';
  const units = runUnits(splitNotebook(src)).map((u) => u.text);
  ok(units.length === 2, `the rand notebook has two code cells (got ${units.length})`);
  const whole = program(run(units.join('\n')));
  const a = program(runCell('', units[0]).out);
  const b = program(runCell(units[0], units[1]).out);
  ok([a, b].join('\n') === whole, `two cells drawing rand() print what the whole program prints (cells: ${a} / ${b}; whole: ${whole.replace(/\n/g, ' / ')})`);
  runCell('', 'x_other = 0.0');
  ok(program(runCell(units[0], units[1]).out) === b, 'the second cell draws the same number when it rebuilds the first');
  const cold = program(run('second = rand()\nshow second'));
  ok(b !== cold, 'and it is not the number a fresh program draws first');
}

// the incremental session: time a Run all against rebuilding every prefix
if (notebooks.length) {
  const nb = notebooks[0];
  const texts = nb.units.map((u) => u.text);
  let chained = 0;
  runCell('', 'x_warm = 0.0');
  for (let j = 0; j < texts.length; j++) chained += runCell(texts.slice(0, j).join('\n'), texts[j]).ms;
  // the same Run all with the session reset before each cell, so every cell
  // rebuilds the cells above it
  let cold = 0;
  for (let j = 0; j < texts.length; j++) {
    runCell('', 'x_reset = 0.0');
    cold += runCell(texts.slice(0, j).join('\n'), texts[j]).ms;
  }
  console.log(`notebooktest: ${nb.name}: Run all took ${chained.toFixed(1)} ms through the session, ${cold.toFixed(1)} ms rebuilding the cells above each one`);
}

console.log(`notebooktest: ${files.length} files split and rejoined, ${notebooks.length} notebook${notebooks.length === 1 ? '' : 's'} run cell by cell; ${checks - fails} of ${checks} checks pass`);
process.exit(fails ? 1 : 0);
