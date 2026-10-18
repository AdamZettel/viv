// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// pde-build -- the "PDE solvers" section, generated from the solvers.
//
// Every page under /pde is one program in pde/<name>.viv, and the program
// carries its own page in its `#!` header:
//
//     #! pde       <title>
//     #! equation  <LaTeX, one line>
//     #! kind      parabolic | hyperbolic | elliptic | dispersive | nonlinear parabolic
//     #! method    <one line>
//     #! checks    <one line: what the tests hold it to>
//     #! | prose ...
//
// This script RUNS each one through viv.wasm -- the same interpreter a reader's
// tab runs -- and refuses to build if any test fails, any program has no tests,
// or any program draws no figure. What the page shows (the numbers, the figure,
// the list of passing tests) is that run's output; none of it is typed here.
// (`pde/check.sh` is the other half: the same programs through the C backend,
// required to print the same thing line for line.)
//
// Emits www/src/data/pde.json and stages the sources for the workbench.
import { readFileSync, writeFileSync, readdirSync, mkdirSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { highlight } from '../src/lib/highlight.js';
import { prose, equation, meta, notesOf, withoutHeader } from './prose.mjs';
import { isAnimRecord } from '../src/lib/animParse.js';
import { FIGURE_LINE } from '../src/lib/postCheck.js';
import { expandUses } from '../src/lib/uses.js';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const SRC = join(ROOT, 'pde');
const STAGE = join(WWW, 'public/programs/pde');

// The reading order. A solver missing from its list is appended to its group, so
// adding a program never needs an edit here; an unknown `#! kind` is an error.
const GROUPS = [
  { kind: 'parabolic', title: 'Diffusion (parabolic)',
    blurb: 'Smoothing, irreversible, and stiff: an explicit step is limited by $\\Delta t \\le h^2/2$, so the working methods are implicit and the work is a tridiagonal solve.',
    order: ['heat_1d', 'heat_2d_adi', 'black_scholes'] },
  { kind: 'hyperbolic', title: 'Waves and transport (hyperbolic)',
    blurb: 'Information moves at a finite speed along characteristics. Nothing smooths the solution, so the scheme\'s own dissipation and dispersion are what you see, and shocks need conservation.',
    order: ['advection_1d', 'wave_1d', 'burgers_shock', 'euler_sod'] },
  { kind: 'elliptic', title: 'Steady states and modes (elliptic)',
    blurb: 'No time to march through: every unknown depends on every other, and the work is a large sparse linear system or its eigenvalues.',
    order: ['laplace_2d', 'poisson_2d', 'drum_modes'] },
  { kind: 'dispersive', title: 'Waves whose speed depends on wavelength (dispersive)',
    blurb: 'The linear part is exact in Fourier space for any time step, so these are split-step spectral codes built on `fft`.',
    order: ['schrodinger_1d', 'kdv_soliton', 'nls_soliton'] },
  { kind: 'nonlinear parabolic', title: 'A nonlinearity against diffusion (nonlinear parabolic)',
    blurb: 'Steepening or growth on one side, diffusion on the other, and a front whose width and speed are set by the balance.',
    order: ['burgers_viscous', 'fisher_kpp'] },
];

const { instance } = await WebAssembly.instantiate(readFileSync(join(WWW, 'public/viv/viv.wasm')), {});
const ex = instance.exports;
function call(fn, text) {
  const enc = new TextEncoder().encode(text);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const packed = fn(ptr, enc.length);
  return new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
}

const errors = [];
const solvers = [];
mkdirSync(STAGE, { recursive: true });
for (const file of readdirSync(SRC).filter((f) => f.endsWith('.viv')).sort()) {
  const slug = file.replace(/\.viv$/, '');
  const src = readFileSync(join(SRC, file), 'utf8');
  writeFileSync(join(STAGE, file), src);
  const kind = meta(src, 'kind');
  for (const key of ['pde', 'equation', 'kind', 'method', 'checks']) if (!meta(src, key)) errors.push(`${slug}: no \`#! ${key}\` line`);
  if (kind && !GROUPS.some((g) => g.kind === kind)) errors.push(`${slug}: unknown kind "${kind}"`);

  const t0 = performance.now();
  const out = call(ex.viv_run, src);
  const seconds = (performance.now() - t0) / 1000;
  const lines = out.split('\n');
  const tests = lines.filter((l) => l.startsWith('PASS ')).map((l) => l.slice(5));
  const failed = lines.filter((l) => l.startsWith('FAIL ') || l.startsWith('error'));
  if (failed.length) errors.push(`${slug}: ${failed[0]}`);
  if (!tests.length) errors.push(`${slug}: no tests ran`);
  if (!lines.some((l) => l.startsWith('figure:'))) errors.push(`${slug}: draws no figure`);

  // `show x` prints "x: value"; a scalar one is a measured number for the page
  const shown = lines.filter((l) => !isAnimRecord(l)).map((l) => l.match(/^([A-Za-z_]\w*): (-?[\d.]+(?:e[+-]?\d+)?|-?inf|nan)$/)).filter(Boolean).map((m) => ({ name: m[1], value: m[2] }));
  // the figure records only: what the page's chart island is handed
  // every figure record, marks and settings included: one list, postCheck's
  const figure = lines.filter((l) => FIGURE_LINE.test(l)).join('\n');

  const doc = JSON.parse(call(ex.viv_doc, src));
  const body = withoutHeader(src);
  let eqHtml = '';
  let notes = [];
  try {
    eqHtml = equation(meta(src, 'equation') || 'x');
    notes = notesOf(src).map(prose);
  } catch (e) {
    errors.push(`${slug}: ${e.message}`);
  }
  solvers.push({
    slug, file, kind,
    title: meta(src, 'pde'),
    equationHtml: eqHtml,
    methodHtml: prose(meta(src, 'method')),
    checksHtml: prose(meta(src, 'checks')),
    notes,
    source: body,
    sourceHtml: highlight(body, 'viv'),
    lines: body.replace(/\n$/, '').split('\n').length,
    functions: doc.functions.map((f) => ({ name: f.name, what: f.what || '' })),
    shown, tests, figure,
    seconds: Math.round(seconds * 100) / 100,
  });
  console.log(`pde-build: ${slug.padEnd(16)} ${String(tests.length).padStart(2)} tests pass   ${seconds.toFixed(2)} s`);
}

const groups = GROUPS.map((g) => {
  const mine = solvers.filter((s) => s.kind === g.kind).map((s) => s.slug);
  for (const s of g.order) if (!mine.includes(s)) errors.push(`group "${g.kind}" lists ${s}, which does not exist or has another kind`);
  const slugs = [...g.order.filter((s) => mine.includes(s)), ...mine.filter((s) => !g.order.includes(s))];
  return { kind: g.kind, title: g.title, blurbHtml: prose(g.blurb), slugs };
});

if (errors.length) {
  console.error(`pde-build: ${errors.length} problem${errors.length > 1 ? 's' : ''}:\n  ${errors.join('\n  ')}`);
  process.exit(1);
}
// reading order, flattened: what prev/next on a page follow
const order = groups.flatMap((g) => g.slugs);
solvers.sort((a, b) => order.indexOf(a.slug) - order.indexOf(b.slug));
mkdirSync(join(WWW, 'src/data'), { recursive: true });
writeFileSync(join(WWW, 'src/data/pde.json'), JSON.stringify({ groups, solvers }, null, 1));
console.log(`pde-build: ${solvers.length} solvers, ${solvers.reduce((n, s) => n + s.tests.length, 0)} tests, every one passing`);
