// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Stage the repo's own material into the site.
//
// The rule this enforces: the website never keeps its own copy of a Viv
// program. Everything under /programs is generated from lang/viv/ at build
// time, so the docs cannot describe a compiler that isn't next to them in the
// repo.
//
// Writes:
//   www/public/programs/examples/*.viv   served to the IDE and terminal
//   www/src/data/programs.json           the gallery index (title, blurb, stats)
//   www/src/data/facts.json              measured counts the pages quote

import { readdirSync, readFileSync, writeFileSync, mkdirSync, rmSync, statSync } from 'node:fs';
import { join, basename, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const EX = join(ROOT, 'lang/viv/examples');
const STD = join(ROOT, 'lang/viv/std');
const OUT_PROGRAMS = join(WWW, 'public/programs/examples');
const OUT_DATA = join(WWW, 'src/data');

// ---- how the gallery groups the decks -------------------------------------
// Ordered: the first matching group wins, so put specific before general.
const GROUPS = [
  {
    slug: 'quantum-chemistry',
    title: 'Quantum chemistry',
    blurb:
      'Hartree–Fock from the Gaussian integrals up, with the published energies to compare against.',
    match: /^(h2_sto3g|scf_|hchain_|geomopt|casida)/,
  },
  {
    slug: 'plasma',
    title: 'Plasma physics',
    blurb:
      'Kinetic and fluid plasma: Landau damping, drift-wave turbulence, tokamak equilibrium, transport coefficients.',
    match: /^(vlasov_|two_stream|oneil_|spitzer_|hasegawa_|grad_shafranov|debye_)/,
  },
  {
    slug: 'nuclear-astro',
    title: 'Nuclear & astrophysics',
    blurb: 'Neutron-star structure and nuclear level densities by exact counting.',
    match: /^(tov|level_density)/,
  },
  {
    slug: 'numerics',
    title: 'Numerical methods',
    blurb: 'Finite elements, molecular dynamics, a spectral Poisson solve, and the records refactor of the FEM solver.',
    match: /^(fem_|md_|fft_)/,
  },
  {
    slug: 'differentiation',
    title: 'Automatic differentiation',
    blurb:
      'Forward mode by dual numbers, reverse mode by tape, and deriv() — AD as a compiler transform.',
    match: /^autodiff_/,
  },
  {
    slug: 'symbolic',
    title: 'Symbolic computation & discovery',
    blurb:
      'A computer-algebra system on flat arrays, symbolic regression, Meijer-G metamodeling, symmetry discovery.',
    match: /^(mini_cas|symbolic_|meijer_|nn_)/,
  },
  {
    slug: 'language',
    title: 'Language features',
    blurb: 'The minimal demos: arithmetic and functions; control flow and recursion.',
    match: /^(area|control)/,
  },
];

// Results worth printing next to a program. Every figure here is copied from
// lang/viv/examples/README.md, which records what each one was checked against.
const VALIDATED = {
  'h2_sto3g.viv': ['−1.11671 Eₕ', 'Szabo & Ostlund'],
  'scf_mcmurchie.viv': ['−1.11671 Eₕ', 'general integral engine'],
  'scf_water.viv': ['−74.94208 Eₕ', 'Crawford −74.9420787, ~1 µEₕ'],
  'scf_water_631gd.viv': ['−76.0192 Eₕ', '6-31G*, 19 basis functions'],
  'scf_sto3g.viv': ['12 molecules + MP2', 'published HF/STO-3G'],
  'hchain_sto3g.viv': ['H₈ −4.06498, H₂₀ −4.79091', 'C++ interpreter, exact'],
  'geomopt.viv': ['H₂ Rₑ = 1.346 a₀', 'HF/STO-3G equilibrium'],
  'geomopt_water.viv': ['r 1.870 a₀, ∠ 100.0°', 'HF/STO-3G water'],
  'fem_poisson.viv': ['error ratio → 4', 'O(h²), the P1 rate'],
  'fem_poisson_records.viv': ['identical to the above', 'records are zero-cost'],
  'md_polymer.viv': ['energy conserved ~0.5%', 'R_g 11.5 → 6.0'],
  'fft_poisson.viv': ['error 1.7e-16', 'exact solution — spectral, to rounding'],
  'autodiff_forward.viv': ["f′(2) = 9.926737", 'dual numbers, exact'],
  'autodiff_reverse.viv': ['∇g matches forward', 'one backward pass'],
  'autodiff_deriv.viv': ['f′(2) = 9.926737', 'AD as an IR transform'],
  'tov.viv': ['0.7087 M☉ maximum', 'Oppenheimer–Volkoff 1939'],
  'level_density.viv': ['p(40) = 37338', 'exact count vs Hardy–Ramanujan'],
  'mini_cas.viv': ['11 asserts pass', 'symbolic diff, simplify to fixpoint'],
  'symbolic_regression.viv': ['4 / 4 recovered exactly', 'AI-Feynman inner loop'],
  'meijer_g.viv': ['log ✓ rational ✓ arctan ✗', 'identification by snapping'],
  'nn_symmetry.viv': ['exact after the substitution', 'the oracle finds x−y'],
  'spitzer_harm.viv': ['within 2% of NRL', 'Spitzer–Härm 1953'],
  'vlasov_landau.viv': ['γ matches the dispersion root', 'Landau damping'],
  'two_stream_pic.viv': ['growth rate vs theory', 'particle-in-cell'],
  'oneil_trapping.viv': ["O'Neil trapping regime", 'nonlinear saturation'],
  'hasegawa_wakatani.viv': ['enstrophy + energy balance', 'drift-wave turbulence'],
  'grad_shafranov.viv': ['relaxed equilibrium', 'tokamak flux surfaces'],
  'debye_scf.viv': ['self-consistent screening', 'Debye length recovered'],
  'casida_tdhf.viv': ['ω_TDA 0.356462 / ω_RPA 0.354778', 'Casida TDHF, water'],
};

function firstBlurb(src) {
  // The header comment block is the program's own abstract. Take its first
  // sentence, unwrapped, with the leading '# ' stripped.
  const lines = [];
  for (const line of src.split('\n')) {
    if (line.startsWith('#')) lines.push(line.replace(/^#+\s?/, ''));
    else if (lines.length) break;
    else if (line.trim() === '') continue;
    else break;
  }
  const text = lines.join(' ').replace(/\s+/g, ' ').trim();
  const stop = text.search(/\.\s|\.$/);
  return (stop > 0 ? text.slice(0, stop + 1) : text).trim();
}

const titleCase = (name) =>
  basename(name, '.viv')
    .replace(/_/g, ' ')
    .replace(/\b(sto3g|631gd|scf|tov|fem|md|cas|nn|mp2|hf)\b/gi, (m) => m.toUpperCase());

function stageProgram(dir, file) {
  const src = readFileSync(join(dir, file), 'utf8');
  const lines = src.split('\n');
  const group = GROUPS.find((g) => g.match.test(file))?.slug ?? 'numerics';
  const [result, reference] = VALIDATED[file] ?? [null, null];
  return {
    file,
    title: titleCase(file),
    blurb: firstBlurb(src),
    lines: lines.length,
    defs: (src.match(/^def\s+\w+/gm) || []).length,
    tests: (src.match(/^\s*test\s+"/gm) || []).length,
    asserts: (src.match(/assert_\w+\(/g) || []).length,
    plots: (src.match(/\b(plot|plot2d|plot3d|scatter|quiver|figure|scene)\(/g) || []).length,
    group,
    result,
    reference,
  };
}

rmSync(OUT_PROGRAMS, { recursive: true, force: true });
mkdirSync(OUT_PROGRAMS, { recursive: true });

const programs = [];
for (const file of readdirSync(EX).filter((f) => f.endsWith('.viv')).sort()) {
  writeFileSync(join(OUT_PROGRAMS, file), readFileSync(join(EX, file)));
  programs.push(stageProgram(EX, file));
}
// the contracts prelude ships too — `viv --use std/contracts.viv` needs it
let stdFiles = [];
try {
  stdFiles = readdirSync(STD).filter((f) => f.endsWith('.viv'));
  for (const f of stdFiles) writeFileSync(join(OUT_PROGRAMS, f), readFileSync(join(STD, f)));
} catch {}

const groups = GROUPS.filter((g) => programs.some((p) => p.group === g.slug)).map((g) => ({
  slug: g.slug,
  title: g.title,
  blurb: g.blurb,
}));
writeFileSync(
  join(OUT_DATA, 'programs.json'),
  JSON.stringify({ groups, programs, std: stdFiles }, null, 2)
);

// ---- measured facts the pages quote --------------------------------------
// Anything the site states as a count is counted here, so a page can never
// drift from the repo. No hand-typed inventory numbers in the .astro files.
const rsLines = (p) => readFileSync(join(ROOT, p), 'utf8').split('\n').length;
const files = {
  'lang/viv/src/lexer.rs': 'Lexer',
  'lang/viv/src/parser.rs': 'Parser',
  'lang/viv/src/ast.rs': 'AST',
  'lang/viv/src/lower.rs': 'Lowering + C emission rules',
  'lang/viv/src/defun.rs': 'Defunctionalization (higher-order → specialized)',
  'lang/viv/src/interp.rs': 'Tree-walking interpreter',
  'lang/viv/src/wasm_backend.rs': 'Direct wasm backend',
  'lang/viv/src/lib.rs': 'Library surface + the wasm ABI',
  'lang/viv/src/main.rs': 'CLI',
  'crates/knotkit/src/ir.rs': 'Kit: content-addressed IR store',
  'crates/knotkit/src/backend.rs': 'Kit: C backend + C AST + DebugMap',
  'crates/knotkit/src/diag.rs': 'Kit: diagnostics as data',
  'crates/knotkit/src/span.rs': 'Kit: spans + source map',
  'crates/knotkit/src/io.rs': 'Kit: injected I/O',
};
const inventory = Object.entries(files).map(([path, what]) => ({
  path,
  what,
  lines: rsLines(path),
}));

const countTests = (dir) => {
  let n = 0;
  const walk = (d) => {
    for (const e of readdirSync(d, { withFileTypes: true })) {
      const p = join(d, e.name);
      if (e.isDirectory()) {
        if (e.name !== 'target') walk(p);
      } else if (e.name.endsWith('.rs')) {
        n += (readFileSync(p, 'utf8').match(/#\[test\]/g) || []).length;
      }
    }
  };
  walk(dir);
  return n;
};

const docCorpus = JSON.parse(readFileSync(join(OUT_DATA, 'vivDocExamples.json'), 'utf8'));
let wasmBytes = null;
try {
  wasmBytes = statSync(join(WWW, 'public/viv/viv.wasm')).size;
} catch {}

const facts = {
  inventory,
  totalRustLines: inventory.reduce((s, r) => s + r.lines, 0),
  rustTests: countTests(join(ROOT, 'lang')) + countTests(join(ROOT, 'crates')),
  programs: programs.length,
  programLines: programs.reduce((s, p) => s + p.lines, 0),
  programAsserts: programs.reduce((s, p) => s + p.asserts, 0),
  docTopics: docCorpus.topics.length,
  docExamples: docCorpus.examples.length,
  siblings: readdirSync(join(ROOT, 'lang')).filter((d) =>
    statSync(join(ROOT, 'lang', d)).isDirectory()
  ),
  wasmBytes,
  generated: new Date().toISOString().slice(0, 10),
};
writeFileSync(join(OUT_DATA, 'facts.json'), JSON.stringify(facts, null, 2));

console.log(
  `staged ${programs.length} programs (${facts.programLines} lines, ${facts.programAsserts} asserts), ` +
    `${facts.rustTests} rust tests, ` +
    `${facts.totalRustLines} lines of compiler` +
    (wasmBytes ? `, viv.wasm ${wasmBytes.toLocaleString()} bytes` : ', viv.wasm MISSING')
);
