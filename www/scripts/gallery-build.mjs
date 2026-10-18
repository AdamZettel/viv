// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// gallery-build -- src/data/vivgallery.json: the entries of the Viv side's
// gallery (/gallery), each a Viv program with the first figure its own run
// drew, filed under the topics of /vipyr/gallery (TOPICS in src/lib/site.ts).
//
//     node scripts/gallery-build.mjs           write the file (needs the viv CLI)
//     node scripts/gallery-build.mjs --check   exit 1 when the file is stale
//
// The entries come from two places. The Viv programs the site already shows
// (plane-wave DFT, the PDE solvers, the example programs with a reference
// value), and the Viv programs vipyr's gallery calls: ENTRIES below names, for
// each /vipyr/gallery entry, the Viv program behind it.
//
// A figure is drawn only from figure records a Viv program printed: a post's
// recorded run (feed.json), a PDE solver's (pde.json), or the run of the
// entry's `draw` program (a small program beside the library that calls it and
// draws vipyr's figure in Viv), drawn by `viv plot` as /vipyr/gallery's
// figures are. An entry whose program draws nothing gets
// no figure; an example program shows the result it is checked on instead.
// The VIV environment variable names the compiler; otherwise the repository's
// target/release/viv, then `viv` on the PATH.
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { FIGURE_LINE } from '../src/lib/postCheck.js';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const OUT = join(WWW, 'src/data/vivgallery.json');
// the records each `draw` program printed, keyed by its expanded source, so a
// build reruns only the programs (or modules) that changed
const RUNS = join(WWW, 'src/data/galleryruns.json');
const data = (f) => JSON.parse(readFileSync(join(WWW, 'src/data', f), 'utf8'));
import { REPO_PUBLIC, GITHUB as REPO } from '../src/lib/repo.js';
const GITHUB = `${REPO}/blob/main`;
const FIG_WIDTH = 640;

const feed = data('feed.json');
const pde = data('pde.json');
const programs = data('programs.json');
const explore = data('explore.json');
const gallery = data('gallery.json');

// the roots the workbench can open a file from (/programs/<root>/...), and the
// ones with a notebook page (/nb/<root>/...); see feed-build.mjs and notebookBuild.js
// (repository path prefix -> served prefix)
const HOSTED = {
  'posts/': 'posts/', 'lib/': 'lib/', 'pde/': 'pde/', 'dft/': 'dft/', 'lang/viv/examples/': 'examples/',
  'catalog/': 'catalog/', 'ports/': 'ports/', 'stat/': 'stat/', 'study/': 'study/', 'examples/': 'examples/',
  'gallery/': 'gallery/',
};
const NOTEBOOK = { 'lib/': 'lib/', 'pde/': 'pde/', 'lang/viv/examples/': 'examples/' };

// ---- the Viv programs behind vipyr's gallery -------------------------------------
// vipyr: the /vipyr/gallery entries that call this program. file: the Viv
// program (repository path). post / explore: the Viv side's pages that run it,
// when there are any; the figure is the post's. draw: a program beside the
// library that calls it and draws vipyr's figure in Viv; its figure comes first,
// and the workbench link opens it.
const ENTRIES = [
  { id: 'silicon-bands', topic: 'science', section: 'solids', title: 'The band structure of silicon',
    summary: 'A plane-wave DFT code computes silicon\'s Kohn-Sham bands along L, Gamma and X, in the browser tab.',
    file: 'dft/si_check.viv', href: '/dft', post: 'silicon-bands-from-the-crystal-up', explore: 'silicon-dense',
    vipyr: ['plot-silicon-bands'] },
  { id: 'landau-damping', topic: 'science', section: 'plasma', title: 'Landau damping from the plasma dispersion function',
    summary: 'The roots of the kinetic dielectric function give the damping rate and frequency of a Langmuir wave at each wavelength.',
    file: 'lib/zfunc.viv', post: 'landau-damping-from-the-dispersion-relation',
    vipyr: ['plot-landau-damping', 'plot-plasma-dispersion-function'] },
  { id: 'thurber-fit', topic: 'math', section: 'fitting', title: 'Thurber\'s electron mobility fitted to NIST\'s certified values',
    summary: 'MINPACK\'s Levenberg-Marquardt, ported to Viv, fits a rational model to NIST\'s Thurber data.',
    file: 'lib/minpack/lmder.viv', post: 'fitting-electron-mobility-with-minpack', explore: 'minpack-thurber',
    vipyr: ['plot-nist-thurber-fit'] },
  { id: 'cartpole-lqr', topic: 'control', section: 'linear', title: 'Balancing a cart-pole with LQR',
    summary: 'The Riccati equation gives the feedback gains, and the simulated pole comes back upright from 25 degrees.',
    file: 'lib/are.viv', post: 'lqr-balances-the-cart-pole',
    vipyr: ['plot-cartpole-lqr'] },
  { id: 'stability-map', topic: 'control', section: 'linear', title: 'Lyapunov equations for the cart-pole\'s feedback gains',
    summary: 'The Lyapunov solver behind vipyr\'s stability map of the cart-pole, on the Schur form.',
    file: 'lib/lyap.viv', vipyr: ['plot-stability-map'] },
  { id: 'four-bit-adder', topic: 'hardware', section: 'logic', title: 'A four-bit adder from NAND gates',
    summary: 'Gates built from NAND, a full adder from gates, and four of them in a row, checked on every input.',
    file: 'lib/gates.viv', post: 'building-a-four-bit-adder-out-of-nand-gates',
    vipyr: ['plot-four-bit-adder'] },
  { id: 'oxygen16-skyrme', topic: 'science', section: 'nuclear', title: 'Oxygen-16 from a Skyrme energy functional',
    summary: 'Skyrme Hartree-Fock with SLy5 on a radial mesh: the binding energy, the charge radius and the densities.',
    file: 'lib/skyrme/skyrmehf.viv', post: 'skyrme-hartree-fock-of-oxygen-16', explore: 'skyrme-o16',
    vipyr: ['plot-oxygen16-skyrme'] },
  { id: 'scf-mixers', topic: 'science', section: 'nuclear', title: 'Density mixers for a self-consistent field',
    summary: 'Linear, Broyden and Pulay mixing, the library the Skyrme code iterates calcium-40 to self-consistency with.',
    file: 'lib/scf/mix.viv', vipyr: ['plot-scf-mixers'] },
  { id: 'sd-shell-model', topic: 'science', section: 'nuclear', title: 'The nuclear shell model in the sd shell',
    summary: 'A configuration-interaction shell model with dense and Lanczos eigensolvers, for neon-20 and magnesium-24.',
    file: 'lib/nuclear/shell/shellmodel.viv', vipyr: ['plot-sd-shell-model', 'plot-sd-shell-eigensolvers'] },
  { id: 'water-pyscf', topic: 'science', section: 'qchem', title: 'Water by restricted Hartree-Fock, ported from PySCF',
    summary: 'The port of PySCF\'s restricted Hartree-Fock that vipyr calls, run on water in the 6-31G basis.',
    file: 'study/ports/pyscf/vivscf.viv', explore: 'pyscf-port-water',
    vipyr: ['plot-water-hartree-fock', 'plot-water-orbital-levels'] },
  { id: 'boys-function', topic: 'science', section: 'qchem', title: 'The Boys function',
    summary: 'The integral behind every Gaussian-basis electron repulsion integral, by series, recursion and asymptotics.',
    file: 'lib/hf/boys.viv', draw: 'lib/hf/gallery/boys.viv', vipyr: ['plot-boys-function'] },
  { id: 'dftatom', topic: 'science', section: 'solids', title: 'Atoms from hydrogen to uranium',
    summary: 'The Viv port of dftatom: radial Kohn-Sham equations for one atom, compared with NIST\'s LDA tables.',
    file: 'ports/dftatom/atom.viv', vipyr: ['plot-atoms-against-nist'] },
  { id: 'ising', topic: 'science', section: 'statmech', title: 'The Ising model by Monte Carlo',
    summary: 'Metropolis sampling of the two-dimensional Ising model on either side of its critical temperature.',
    file: 'stat/ising/ising_mc.viv', vipyr: ['plot-ising-phase-transition'] },
  { id: 'lj-nvt', topic: 'science', section: 'statmech', title: 'A Lennard-Jones fluid at constant temperature',
    summary: 'Molecular dynamics with a thermostat, ported from Allen and Tildesley\'s Fortran, against the fluid\'s equation of state.',
    file: 'ports/liquids/md_nvt.viv', vipyr: ['plot-lj-liquid-structure'] },
  { id: 'lj-nve', topic: 'science', section: 'statmech', title: 'A Lennard-Jones liquid at constant energy',
    summary: 'Velocity Verlet molecular dynamics, ported from Allen and Tildesley: energy conservation and diffusion.',
    file: 'ports/liquids/md_nve.viv', vipyr: ['plot-lj-dynamics'] },
  { id: 'feature-catalog', topic: 'data', section: 'features', title: 'Sessions from a transaction log',
    summary: 'Per-customer sessions with state, computed one event at a time; transaction velocity and delinquency sit beside it.',
    file: 'catalog/sessions.viv', vipyr: ['plot-feature-catalog'] },
  { id: 'special-functions', topic: 'stats', section: 'dist', title: 'The special functions under the distributions',
    summary: 'The normal, incomplete gamma and incomplete beta functions that the distributions are computed from.',
    file: 'lib/stats/special.viv', draw: 'lib/stats/gallery_special.viv', vipyr: ['plot-special-function-accuracy'] },
  { id: 'linear-models', topic: 'stats', section: 'regression', title: 'Least squares that keeps its digits',
    summary: 'Ordinary least squares by QR, the code vipyr checks against NIST\'s certified regressions.',
    file: 'lib/stats/linmod.viv', draw: 'lib/stats/gallery_longley.viv', vipyr: ['plot-nist-linear-regression'] },
  { id: 'glm', topic: 'stats', section: 'regression', title: 'Generalized linear models',
    summary: 'Iteratively reweighted least squares for GLMs, compared with R\'s glm.',
    file: 'lib/stats/glm.viv', draw: 'lib/stats/gallery_glm.viv', vipyr: ['plot-glm-probit'] },
  { id: 'tiny-gpt', topic: 'ml', section: 'nn', title: 'A small GPT trained on one paragraph',
    summary: 'nanoGPT\'s layout written in Viv, trained in the browser tab and stepped through call by call.',
    file: 'lib/gpt/story_gpt.viv', explore: 'tiny-gpt', vipyr: ['plot-tiny-gpt'] },
  { id: 'binomial-options', topic: 'finance', section: 'options', title: 'Options on a binomial tree',
    summary: 'Cox-Ross-Rubinstein trees for European and American options, against the Black-Scholes formula.',
    file: 'examples/options/crr.viv', draw: 'examples/options/gallery_smile.viv', vipyr: ['plot-option-pricing'] },
  { id: 'heston', topic: 'finance', section: 'options', title: 'The Heston model by Fourier cosine series',
    summary: 'European prices under stochastic volatility by the COS method, and the smile they imply.',
    file: 'examples/options/heston.viv', draw: 'examples/options/gallery_heston.viv', vipyr: ['plot-heston-smile'] },
  { id: 'american-put', topic: 'finance', section: 'options', title: 'An American put by Crank-Nicolson',
    summary: 'The early-exercise boundary from a finite-difference solve, checked against a binomial tree.',
    file: 'examples/options/cnpde.viv', draw: 'examples/options/gallery_american.viv', vipyr: ['plot-american-put'] },
  { id: 'ode-solve', topic: 'math', section: 'ode', title: 'Stiff and nonstiff ODE solvers',
    summary: 'Dormand-Prince, Radau and BDF behind one solve call, for Robertson\'s stiff kinetics and batches of epidemics.',
    file: 'lib/ode/solve.viv', draw: 'lib/ode/gallery_robertson.viv', vipyr: ['plot-ode-robertson', 'plot-ode-sir-batch'] },
  { id: 'randla', topic: 'math', section: 'linalg', title: 'Randomized trace estimation and low-rank approximation',
    summary: 'Hutch++ for the trace of a matrix seen only through products, and randomly pivoted Cholesky for a kernel matrix.',
    file: 'lib/randla/api.viv', draw: 'lib/randla/gallery_trace.viv', vipyr: ['plot-randomized-trace-and-lowrank'] },
  { id: 'digits', topic: 'math', section: 'accuracy', title: 'Which digits of a result are real',
    summary: 'Cancellation, Kahan summation and a Hilbert solve, the programs vipyr runs under the digits lens.',
    file: 'examples/digits/hilbert.viv', vipyr: ['plot-digits-lens'] },
  { id: 'rv32', topic: 'hardware', section: 'processors', title: 'Fibonacci on a RISC-V processor',
    summary: 'A C program compiled, assembled and run on an RV32IM simulator, each step written in Viv.',
    file: 'gallery/hardware/cc_fib.viv', vipyr: ['plot-fib-on-rv32'] },
  { id: 'power-flow', topic: 'power', section: 'flow', title: 'AC power flow on MATPOWER cases',
    summary: 'Newton-Raphson and fast-decoupled power flow with a sparse LU, from the 9-bus case to 9241 buses.',
    file: 'lib/power/api.viv', vipyr: ['plot-case118-voltages', 'plot-pf-convergence'] },
  { id: 'interconnection', topic: 'power', section: 'flow', title: 'An interconnection study',
    summary: 'A new generator added to the IEEE 118-bus system, with every single-line outage checked.',
    file: 'lib/power/study.viv', vipyr: ['plot-interconnection-study'] },
  { id: 'pv-curve', topic: 'power', section: 'stability', title: 'A PV curve by continuation power flow',
    summary: 'The voltage at a bus as the transfer into it grows, traced around the nose of the curve.',
    file: 'lib/power/cpf.viv', vipyr: ['plot-pv-curve'] },
];
// vipyr gallery entries with no Viv program in the repository to list:
// plot-sklearn-to-c writes its Viv program (loans.viv) at run time.
const NOT_LISTED = {
  'plot-sklearn-to-c': 'its Viv program is generated when the script runs',
  'plot-gguf-quantization': 'lib/gguf reads a model file of a megabyte or more; its Viv programs are lib/gguf/tests',
};

// ---- the site's own Viv programs (what /applications listed) ------------------------
// PDE solvers file under math, except the one that prices options.
const PDE_PLACE = { black_scholes: ['finance', 'options'] };
// example program groups (stage.mjs GROUPS) to a topic and section
const EXAMPLE_PLACE = {
  'quantum-chemistry': ['science', 'qchem'],
  plasma: ['science', 'plasma'],
  'nuclear-astro': ['science', 'nuclear'],
  numerics: ['math', 'pde'],
  differentiation: ['math', 'autodiff'],
  symbolic: ['math', 'symbolic'],
};
const EXAMPLE_FILE_PLACE = { 'md_polymer.viv': ['science', 'statmech'] };
// card titles: a program's own first line is a description, too long for a card
const EXAMPLE_TITLES = {
  'autodiff_deriv.viv': 'The exact derivative as a compiler transform',
  'autodiff_forward.viv': 'Forward-mode differentiation with dual numbers',
  'autodiff_reverse.viv': 'Reverse-mode differentiation on a tape',
  'debye_scf.viv': 'Debye screening from the nonlinear Poisson-Boltzmann equation',
  'fem_poisson.viv': 'The Poisson equation by finite elements',
  'fem_poisson_records.viv': 'The same finite-element solver written with records',
  'fft_poisson.viv': 'The Poisson equation solved spectrally',
  'geomopt.viv': 'The bond length of H2 by geometry optimization',
  'geomopt_water.viv': 'The geometry of water by optimization',
  'grad_shafranov.viv': 'A tokamak equilibrium from the Grad-Shafranov equation',
  'h2_sto3g.viv': 'Hartree-Fock for the hydrogen molecule',
  'hasegawa_wakatani.viv': 'Drift-wave turbulence in the Hasegawa-Wakatani model',
  'hchain_sto3g.viv': 'Hartree-Fock for a chain of hydrogen atoms',
  'level_density.viv': 'Nuclear level density by counting states',
  'md_polymer.viv': 'Molecular dynamics of a bead-spring polymer',
  'meijer_g.viv': 'Identifying a function through the Meijer G-function',
  'mini_cas.viv': 'A small computer-algebra system on flat arrays',
  'nn_symmetry.viv': 'Finding a symmetry with a neural network',
  'oneil_trapping.viv': 'Particle trapping after Landau damping',
  'scf_mcmurchie.viv': 'Hartree-Fock with a McMurchie-Davidson integral engine',
  'scf_sto3g.viv': 'Hartree-Fock and MP2 for small molecules in STO-3G',
  'scf_water.viv': 'Hartree-Fock for water in STO-3G',
  'scf_water_631gd.viv': 'Hartree-Fock for water in 6-31G*',
  'spitzer_harm.viv': 'The Spitzer-Härm conductivity of a plasma',
  'symbolic_regression.viv': 'Recovering a formula from data',
  'tov.viv': 'The maximum mass of a free-neutron star',
  'two_stream_pic.viv': 'The two-stream instability in a particle-in-cell code',
  'vlasov_landau.viv': 'Landau damping in a Vlasov-Poisson solver',
};
const PDE_TITLES = {
  burgers_shock: "Burgers' equation with a shock and a rarefaction",
  drum_modes: 'The modes of a drum',
};
// the card text keeps to commas: no dashes between clauses
const plain = (s) => String(s).replace(/\s+[—–]\s+/g, ', ');

// ---- links --------------------------------------------------------------------------
const served = (roots, f) => {
  const r = Object.keys(roots).find((k) => f.startsWith(k));
  return r ? roots[r] + f.slice(r.length) : null;
};
const hosted = (f) => !!served(HOSTED, f);
const workbench = (f) => (hosted(f) ? `/workbench?open=${served(HOSTED, f)}` : null);
const notebook = (f) => (served(NOTEBOOK, f) ? `/nb/${served(NOTEBOOK, f).replace(/\.viv$/, '')}` : null);
// a link into the repository only while it is public (src/lib/repo.js)
const source = (f) => (REPO_PUBLIC ? `${GITHUB}/${f}` : null);

// ---- figures ------------------------------------------------------------------------
const vivBin = () => {
  if (process.env.VIV) return process.env.VIV;
  const local = join(ROOT, 'target/release/viv');
  return existsSync(local) ? local : 'viv';
};
const hash = (s) => createHash('sha256').update(s).digest('hex').slice(0, 16);
// the first chart a run drew, as SVG; null when the records hold no chart `viv plot` draws
function draw(records, check) {
  if (!records || !/^(figure|grid): /m.test(records)) return null;
  if (check) return 'unchecked';
  const r = spawnSync(vivBin(), ['plot', '-', '--chart', '1', '--width', String(FIG_WIDTH)], { input: records, encoding: 'utf8', maxBuffer: 1 << 26 });
  if (r.status !== 0 || !r.stdout.startsWith('<svg')) {
    console.error(`gallery-build: viv plot failed: ${(r.stderr || r.error || '').toString().slice(0, 300)}`);
    process.exit(1);
  }
  return r.stdout;
}

// the figure records of a `draw` program: run by the CLI (cwd the repository),
// or taken from galleryruns.json when its expanded source has not changed
const runs = existsSync(RUNS) ? JSON.parse(readFileSync(RUNS, 'utf8')) : {};
function drawRun(file, check, problems) {
  const ex = spawnSync(vivBin(), ['expand', file], { cwd: ROOT, encoding: 'utf8', maxBuffer: 1 << 26 });
  if (ex.status !== 0) {
    problems.push(`${file}: viv expand failed: ${(ex.stderr || '').slice(0, 300)}`);
    return '';
  }
  const key = hash(ex.stdout);
  if (runs[file] && runs[file].key === key) return runs[file].records;
  if (check) {
    problems.push(`${file} changed since its run was recorded (node scripts/gallery-build.mjs reruns it)`);
    return '';
  }
  const t0 = performance.now();
  const r = spawnSync('nice', [vivBin(), 'run', file], { cwd: ROOT, encoding: 'utf8', maxBuffer: 1 << 26 });
  const seconds = Math.round((performance.now() - t0) / 10) / 100;
  const lines = (r.stdout || '').split('\n');
  const failed = lines.find((l) => l.startsWith('FAIL ') || l.startsWith('error'));
  if (r.status !== 0 || failed) {
    problems.push(`${file}: ${failed || (r.stderr || '').slice(0, 300)}`);
    return '';
  }
  const records = lines.filter((l) => FIGURE_LINE.test(l)).join('\n');
  runs[file] = { key, seconds, records };
  console.log(`gallery-build: ran ${file} in ${seconds} s`);
  return records;
}

function build(check) {
  const posts = new Map(feed.posts.map((p) => [p.slug, p]));
  const explores = new Map(explore.programs.map((p) => [p.slug, p]));
  const vipyrSlugs = new Map(gallery.entries.map((e) => [e.slug, e]));
  const entries = [];
  const problems = [];

  // the programs vipyr's gallery calls
  for (const e of ENTRIES) {
    const post = e.post ? posts.get(e.post) : null;
    if (e.post && !post) problems.push(`${e.id}: no post ${e.post}`);
    if (e.explore && !explores.has(e.explore)) problems.push(`${e.id}: no explored program ${e.explore}`);
    if (!existsSync(join(ROOT, e.file))) problems.push(`${e.id}: ${e.file} is missing`);
    for (const v of e.vipyr) if (!vipyrSlugs.has(v)) problems.push(`${e.id}: no /vipyr/gallery/${v}`);
    if (e.draw && !existsSync(join(ROOT, e.draw))) problems.push(`${e.id}: ${e.draw} is missing`);
    const drawn = e.draw && existsSync(join(ROOT, e.draw)) ? drawRun(e.draw, check, problems) : '';
    const records = drawn || (post ? post.figure : '');
    const runFile = e.draw || (post ? `posts/${post.file}` : e.file);
    entries.push({
      id: e.id,
      topic: e.topic,
      section: e.section,
      title: e.title,
      summary: e.summary,
      file: e.file,
      href: e.href || (post ? `/p/${post.slug}` : e.explore ? `/explore/${e.explore}` : (e.draw && workbench(e.draw)) || notebook(e.file) || workbench(e.file) || source(e.file)),
      post: post ? `/p/${post.slug}` : null,
      explore: e.explore ? `/explore/${e.explore}` : null,
      workbench: workbench(runFile),
      source: hosted(runFile) ? null : source(runFile),
      vipyr: e.vipyr.map((s) => ({ href: `/vipyr/gallery/${s}`, title: vipyrSlugs.get(s)?.title || s })),
      figFrom: records && /^(figure|grid): /m.test(records) ? (drawn ? e.draw : `posts/${post.file}`) : null,
      figHash: records ? hash(records) : null,
      fig: draw(records, check),
    });
  }
  const covered = new Set(ENTRIES.flatMap((e) => e.vipyr));
  for (const s of vipyrSlugs.keys()) {
    if (!covered.has(s) && !NOT_LISTED[s]) problems.push(`/vipyr/gallery/${s} has no Viv program in ENTRIES (or NOT_LISTED)`);
  }

  // the PDE solvers
  for (const s of pde.solvers) {
    const [topic, section] = PDE_PLACE[s.slug] || ['math', 'pde'];
    entries.push({
      id: `pde-${s.slug}`,
      topic,
      section,
      title: PDE_TITLES[s.slug] || s.title,
      summary: null,
      // the solver's first note, already rendered (its math is KaTeX HTML)
      summaryHtml: (s.notes || [])[0] || null,
      file: `pde/${s.file}`,
      href: `/pde/${s.slug}`,
      post: null,
      explore: null,
      workbench: workbench(`pde/${s.file}`),
      source: null,
      vipyr: [],
      figFrom: s.figure ? `pde/${s.file}` : null,
      figHash: s.figure ? hash(s.figure) : null,
      fig: draw(s.figure, check),
    });
  }

  // the example programs that carry a reference value: the card shows the result
  for (const p of programs.programs.filter((x) => x.reference && x.result)) {
    const place = EXAMPLE_FILE_PLACE[p.file] || EXAMPLE_PLACE[p.group];
    if (!place) {
      problems.push(`examples/${p.file}: group ${p.group} has no place in EXAMPLE_PLACE`);
      continue;
    }
    if (!EXAMPLE_TITLES[p.file]) {
      problems.push(`examples/${p.file}: no card title in EXAMPLE_TITLES`);
      continue;
    }
    entries.push({
      id: `example-${p.file.replace(/\.viv$/, '')}`,
      topic: place[0],
      section: place[1],
      title: EXAMPLE_TITLES[p.file],
      summary: null,
      file: `lang/viv/examples/${p.file}`,
      href: notebook(`lang/viv/examples/${p.file}`),
      post: null,
      explore: null,
      workbench: workbench(`lang/viv/examples/${p.file}`),
      source: null,
      vipyr: [],
      result: { value: plain(p.result), reference: plain(p.reference) },
      figFrom: null,
      figHash: null,
      fig: null,
    });
  }

  if (problems.length) {
    console.error(`gallery-build:\n  ${problems.join('\n  ')}`);
    process.exit(1);
  }
  return entries;
}

if (process.argv.includes('--check')) {
  const want = build(true);
  const have = existsSync(OUT) ? JSON.parse(readFileSync(OUT, 'utf8')).entries : [];
  const strip = (es) => JSON.stringify(es.map(({ fig, ...rest }) => ({ ...rest, drawn: !!fig })));
  if (strip(want.map((e) => ({ ...e, fig: e.fig ? 1 : null }))) !== strip(have)) {
    console.error('gallery-build: src/data/vivgallery.json is stale (node scripts/gallery-build.mjs rewrites it)');
    process.exit(1);
  }
  console.log(`gallery-build: src/data/vivgallery.json is current (${have.length} entries)`);
} else {
  const entries = build(false);
  const used = new Set(ENTRIES.map((e) => e.draw).filter(Boolean));
  writeFileSync(RUNS, JSON.stringify(Object.fromEntries(Object.entries(runs).filter(([f]) => used.has(f)).sort()), null, 1) + '\n');
  writeFileSync(OUT, JSON.stringify({ notListed: NOT_LISTED, entries }, null, 1) + '\n');
  const drawn = entries.filter((e) => e.fig).length;
  console.log(`gallery-build: ${entries.length} entries, ${drawn} with a figure from their own run`);
}
