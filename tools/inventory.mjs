#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The inventory: every page, program, function, package, verb, document and
// test suite in the repository, as one list.
//
//     node tools/inventory.mjs                 write inventory.json and INVENTORY.md
//     node tools/inventory.mjs ls [kind] [--json] [--grep text] [--urls]
//                                              print the list (one asset per line)
//     node tools/inventory.mjs kinds           the kinds and how many of each
//
// Each asset is { kind, id, title, source: [paths], url, checked, status }.
// `url` is the site address (null when the asset has no page), `checked` is
// the reference it is held to when the site's data names one, and `status`
// says what state it is in (a post's lane, a program's test count).
//
// It reads the site's generated data (www/src/data/*.json and *.mjs) rather
// than deriving the same lists again, and walks the source directories for
// the rest; the gate's checks come from gate.CHECKS itself (python3). The
// output is sorted and carries no timestamps. inventory.json and INVENTORY.md
// are build products, not committed (.gitignore): `ls` and `kinds` build the
// list fresh on every call, and www's `npm run stage` writes both files for
// /inventory. No dependencies beyond node and python3.
//
// Paths that publish/exclude.txt leaves out of the public repository are left
// out here too, so /inventory lists only what a reader of the repository can
// open. In a checkout without publish/ (the public one) nothing is filtered.
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, relative, dirname, basename } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { execFileSync } from 'node:child_process';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const DATA = join(ROOT, 'www/src/data');
const OUT_JSON = join(ROOT, 'inventory.json');
const OUT_MD = join(ROOT, 'INVENTORY.md');

// The kinds, in the order the listing and the page show them, with a line
// that says what each one is.
export const KINDS = [
  ['dir', 'A directory of the repository'],
  ['page', 'A page of the site with no data behind its route'],
  ['post', 'A post on /posts: a program whose tests passed when the site was built'],
  ['claim', 'A claim: a bounded statement, the Viv program that checks it and its vipyr twin'],
  ['channel', 'A channel: a syllabus of posts'],
  ['show', 'A showcase page: one program, its reference and its run'],
  ['scene', 'A scene on the home page or the gallery'],
  ['gallery', 'An entry of the Viv gallery'],
  ['vipyr-gallery', 'A worked Python example in the vipyr gallery'],
  ['explore', 'A program in the explorer, stepped line by line'],
  ['ladder', 'A ladder: a program built up one complete program per step'],
  ['pde', 'A PDE solver page'],
  ['engine', 'An engine: a program that reads input files and runs in the browser'],
  ['compare', 'A Python and Viv pair with the same output'],
  ['doc', 'A topic of the language docs, with runnable examples'],
  ['dft-source', 'A source file of the plane-wave DFT code, as a page'],
  ['vipyr-family', 'A family of vipyr functions'],
  ['vipyr', 'A vipyr API function'],
  ['atlas', 'A code atlas: the call graph of a vipyr function'],
  ['program', 'A .viv program'],
  ['package', 'A vipyr extra as its own Python distribution (packages/)'],
  ['port', 'A port of a published code to Viv, with its upstream license (ports/)'],
  ['cli', 'A verb of the viv command'],
  ['feature', 'An entry of the language feature catalog (lang/viv/features, `viv features`)'],
  ['test', 'A test suite'],
  ['gate', 'A check of the gate (tools/gate/gate.py)'],
];
const KIND_ORDER = Object.fromEntries(KINDS.map(([k], i) => [k, i]));

// the /nb pages (www/src/lib/notebookBuild.js): repository directory -> address head
const NB_DIRS = [['posts', 'posts'], ['lang/viv/examples', 'examples'], ['pde', 'pde'], ['lib', 'lib']];
// directories the map breaks down one level further
const DEEP_DIRS = ['lang', 'lib', 'www', 'tools', 'python', 'study'];
const SKIP_DIRS = new Set(['node_modules', 'target', '.git', '__pycache__', 'dist', '.astro', '.venv']);

// What each directory holds, one line each. A directory missing here falls
// back to its README's first sentence; AGENTS.md's map is written from these.
export const DIR_NOTES = {
  'LICENSES/': 'The license texts REUSE.toml points to',
  'bench/': 'Timing programs for the Hartree-Fock integrals',
  'bugs/': 'The bug ledger (ledger.jsonl) and the test of each fixed bug',
  'catalog/': 'Per-entity features with state (sessions, velocity, delinquency), each a Viv function called on a DataFrame',
  'channels/': 'The syllabus of each posts channel, one Markdown file per channel',
  'claims/': 'Claims: a bounded statement, the program that checks it and its vipyr twin, by field',
  'claude/': 'The Claude Code plugin for vipyr (skill and hooks)',
  'crates/': 'Rust crates shared by the languages',
  'crates/knotkit/': 'knotkit, the kit for content-addressed, multi-target compilers that Viv is built on',
  'dft/': 'Plane-wave Kohn-Sham DFT in Viv, with its numpy reference and the published values',
  'docs/': 'The tutorial',
  'engines/': 'Engines: Viv programs with a declared interface that read input files (POSCAR, XYZ) and run as wasm',
  'examples/': 'Worked examples for Python users: data science, deploying scikit-learn, digits, options, CUDA',
  'gallery/': 'The vipyr gallery scripts, by topic',
  'kernels/': 'Kernel libraries exported for Python with a manifest (k1)',
  'ladders/': 'Ladder programs: one complete, tested program per step',
  'lang/': 'The languages built on knotkit',
  'lang/oa/': 'A small operator-algebra language, a second probe of the kit',
  'lang/skein/': 'Skein, a small teaching Scheme compiled to C through knotkit',
  'lang/viv/': 'The Viv compiler: lexer, parser, types, interpreter, C and wasm backends, the viv command',
  'lang/weft/': 'Weft, a weak-form language that compiles to Viv',
  'lib/': "Viv's library: the modules programs import",
  'lib/algo/': 'Algorithms with a spec of what each promises (algo.spec)',
  'lib/basis/': 'Gaussian basis sets as data',
  'lib/cas/': 'A computer algebra system, and Viv code generated from its expressions',
  'lib/cc/': 'A C compiler written in Viv, to RV32IM assembly',
  'lib/eigen/': 'Eigensolvers',
  'lib/gguf/': 'A reader for GGUF model files and their quantized tensors',
  'lib/gpt/': 'A small GPT: the forward pass, the gradient and training',
  'lib/hf/': 'Hartree-Fock: integrals, SCF, RHF, UHF and ROHF, checked against PySCF',
  'lib/hfref/': 'Reference tables for lib/hf (Boys function, Cartesian to spherical)',
  'lib/minpack/': "A port of MINPACK's lmder (Levenberg-Marquardt) and its helpers",
  'lib/nprandom/': "NumPy 1.24.4's random generators, draw for draw",
  'lib/nuclear/': 'Nuclear structure: the shell model and its interactions',
  'lib/ode/': 'ODE integrators, stiff (BDF) and non-stiff',
  'lib/power/': 'Power-system studies: load flow, distribution factors, transfer limits',
  'lib/randla/': 'Randomized linear algebra: sketches, range finders, trace estimates',
  'lib/rv32/': 'An RV32IM assembler and simulator',
  'lib/scf/': 'The SCF loop shared by the mean-field codes',
  'lib/skyrme/': 'Skyrme Hartree-Fock of closed-shell nuclei',
  'lib/stats/': 'Statistics: descriptive, distributions, GLMs, special functions',
  'packages/': 'The vipyr extras as Python distributions (chem, hardware, nuclear, plasma, symreg, xc)',
  'pde/': 'One program per partial differential equation, each with its exact checks',
  'plasma/': 'Kinetic plasma programs: Vlasov-Poisson and Landau damping',
  'ports/': 'Ports of published codes with their upstream license and comparison (dftatom, liquids)',
  'posts/': 'The posts: each a .viv program with a header, published when its tests pass',
  'python/': 'vipyr, Viv as a Python package, and its tests',
  'python/tests/': "vipyr's tests: what the package promises",
  'python/vipyr/': 'The vipyr package: loading .viv files, the import hook, frames, magics, translate, compile',
  'scf/': 'The programs that check lib/hf as a whole against PySCF',
  'server/': 'The MCP endpoint (server/mcp), served at /mcp by a Netlify Function',
  'show/': 'Programs behind the showcase pages',
  'stat/': 'Statistical-mechanics programs exported for Python (Ising)',
  'study/': 'Studies and measurements: comparisons with Python, port ledgers, memory checks',
  'study/benchmarks/': 'Viv versions of programs from published benchmark sets',
  'study/compare/': 'Python and Viv pairs that print the same output, by topic',
  'study/idioms/': 'Probes of how idiomatic Python maps to Viv',
  'study/memcheck/': 'Programs written to break the memory model',
  'study/ports/': 'Port ledgers: what each port needed that Viv lacked',
  'study/release/': 'The release audit',
  'study/vivkernels/': 'Prototypes for the kernel export',
  'tests/': 'Test inputs outside the cargo crates (C programs for lib/cc)',
  'third_party/': 'Vendored third-party code (SQLite)',
  'tools/': 'Scripts: the gate, merge, checks, generators and harnesses',
  'tools/algo/': 'The lib/algo spec checker',
  'tools/arrays/': 'Array cases compared against NumPy',
  'tools/bytes/': 'Byte and float-format cases',
  'tools/cas/': 'Generators that use lib/cas',
  'tools/cc/': 'The C corpus check for lib/cc',
  'tools/claims/': 'The claims runner',
  'tools/export/': 'Checks of exported kernels',
  'tools/fuzz/': 'vivsmith, the program fuzzer',
  'tools/gate/': 'The gate: every check a change must pass (gate.py), its cache and the site check',
  'tools/gguf/': 'lib/gguf against gguf-py',
  'tools/graph/': 'Graph drawing',
  'tools/imports/': 'The import resolver study',
  'tools/ints/': 'Integer cases compared against Python and NumPy',
  'tools/jupyter/': 'A Jupyter kernel for Viv and the notebook converter',
  'tools/libm-sweep/': "Viv's libm against V8's Math and the C library",
  'tools/licenses/': 'License headers and the REUSE check',
  'tools/llm-pack-trial/': 'A trial of the prompt pack with a model',
  'tools/maps/': 'Timings of maps in every engine',
  'tools/mathtext/': 'Glyph metrics for math in figure labels',
  'tools/memcheck/': 'Sanitizer and leak runs of the C backend',
  'tools/phrases/': 'Cases for phrases',
  'tools/print-fmt-sweep/': 'Number formatting against Python',
  'tools/py2viv/': 'py2viv, the strict Python to Viv translator',
  'tools/release/': 'Release scripts',
  'tools/rv32/': 'riscv-tests for the RV32IM simulator',
  'tools/steps/': 'The rewrite into steps',
  'tools/strings/': 'String timings',
  'tools/surface/': 'Surface-syntax cases',
  'tools/vivdb/': 'vivdb, the program database',
  'tools/vivxc/': 'The exchange-correlation catalog',
  'tools/vmath/': 'Checks of the math functions at their worst cases',
  'tools/wheels/': 'Builds of the vipyr wheels',
  'www/': 'viv-lang.org, the site (Astro); the compiler runs in the browser as wasm',
  'www/public/': 'Static files: the wasm compiler, workers, staged programs',
  'www/scripts/': 'Site build steps (stage) and checks (check)',
  'www/src/': 'Pages, components, layouts and the generated data the pages read',
  'xc/': 'Exchange-correlation functionals derived from Libxc 6.1.0',
};

const rel = (p) => relative(ROOT, p).split('\\').join('/');
const exists = (p) => existsSync(join(ROOT, p));
const read = (p) => readFileSync(join(ROOT, p), 'utf8');
const json = (name) => JSON.parse(readFileSync(join(DATA, name), 'utf8'));
const isDir = (p) => { try { return statSync(p).isDirectory(); } catch { return false; } };
const plain = (s) => (s == null ? '' : String(s))
  .replace(/<[^>]*>/g, '').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
  .replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/\s+/g, ' ').trim();
// Cut to n characters, never inside a $...$ formula: a cut that would leave
// one open moves back to before that formula's opening `$`.
const clip = (s, n = 160) => {
  const t = plain(s);
  if (t.length <= n) return t;
  let cut = t.slice(0, n - 1);
  if ((cut.match(/(?<!\\)\$/g) || []).length % 2) cut = cut.slice(0, cut.lastIndexOf('$'));
  return `${cut.trimEnd().replace(/[,;:(]+$/, '').trimEnd()}…`;
};

// The paths publish/exclude.txt leaves out of the public repository
// (publish/select.mjs), or none when this checkout has no publish/.
const EXCLUDED = await (async () => {
  const sel = join(ROOT, 'publish', 'select.mjs');
  if (!existsSync(sel) || !existsSync(join(ROOT, 'publish', 'exclude.txt'))) return () => false;
  const { excluder } = await import(pathToFileURL(sel).href);
  return excluder();
})();

// The files git knows: tracked, plus untracked ones not ignored, less the
// excluded ones. Build products and caches stay out of the inventory, so two
// checkouts of one commit give the same list. Without git, every file counts.
const KNOWN = (() => {
  try {
    const out = execFileSync('git', ['ls-files', '-z', '--cached', '--others', '--exclude-standard'],
      { cwd: ROOT, encoding: 'utf8', maxBuffer: 1 << 28, stdio: ['ignore', 'pipe', 'ignore'] });
    const files = new Set(out.split('\0').filter(Boolean).filter((f) => !EXCLUDED(f)));
    const dirs = new Set();
    for (const f of files) for (let i = f.indexOf('/'); i >= 0; i = f.indexOf('/', i + 1)) dirs.add(f.slice(0, i));
    return { files, dirs };
  } catch { return null; }
})();
const known = (p, dir = false) => !KNOWN || (dir ? KNOWN.dirs : KNOWN.files).has(rel(p));

function walk(dir, pred) {
  if (!isDir(dir)) return [];
  return readdirSync(dir).sort().flatMap((f) => {
    if (SKIP_DIRS.has(f)) return [];
    const p = join(dir, f);
    if (isDir(p)) return known(p, true) ? walk(p, pred) : [];
    return known(p) && pred(p) ? [p] : [];
  });
}

function asset(kind, id, title, source, url = null, checked = null, status = null) {
  const all = (Array.isArray(source) ? source : [source]).filter(Boolean);
  const src = all.filter((p) => p.endsWith('/') || !EXCLUDED(p));
  if (all.length && !src.length) return { gone: true };
  return { kind, id: String(id), title: clip(title) || String(id), source: [...new Set(src)].sort(),
    url: url || null, checked: checked ? clip(checked, 240) : null, status: status || null };
}

// ---------------------------------------------------------------------------
// the repository map

/** A directory's one-line description: its README's first sentence, when it has one. */
function readmeLine(dir) {
  for (const f of ['README.md', 'README', 'readme.md']) {
    const p = join(dir, f);
    if (!existsSync(p)) continue;
    const lines = readFileSync(p, 'utf8').split('\n');
    let head = '';
    for (const l of lines) {
      const t = l.trim();
      if (!t || t.startsWith('<!--') || t.startsWith('```') || t.startsWith('|') || t.startsWith('![')) continue;
      if (t.startsWith('#')) { head ||= t.replace(/^#+\s*/, ''); if (head) break; continue; }
    }
    return clip(head, 140);
  }
  return '';
}

function dirs() {
  const out = [];
  const top = readdirSync(ROOT).sort().filter((f) => !f.startsWith('.') && !SKIP_DIRS.has(f) && isDir(join(ROOT, f)) && known(join(ROOT, f), true));
  for (const d of top) {
    out.push(asset('dir', `${d}/`, DIR_NOTES[`${d}/`] || readmeLine(join(ROOT, d)) || d, `${d}/`));
    if (!DEEP_DIRS.includes(d)) continue;
    for (const s of readdirSync(join(ROOT, d)).sort()) {
      const p = join(ROOT, d, s);
      if (s.startsWith('.') || SKIP_DIRS.has(s) || !isDir(p) || !known(p, true)) continue;
      const id = `${d}/${s}/`;
      out.push(asset('dir', id, DIR_NOTES[id] || readmeLine(p) || id, id));
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// site pages

function staticPages() {
  const base = join(ROOT, 'www/src/pages');
  return walk(base, (p) => p.endsWith('.astro') && !rel(p).includes('['))
    .map((p) => {
      const r = relative(base, p).split('\\').join('/').replace(/\.astro$/, '');
      const url = r === 'index' ? '/' : `/${r.replace(/\/index$/, '')}`;
      const src = readFileSync(p, 'utf8');
      const m = src.match(/<(?:Base|DocPage)[^>]*?\btitle=["']([^"']+)["']/s) || src.match(/\btitle=["']([^"']+)["']/);
      return asset('page', url, m ? m[1] : url, rel(p), url, null, 'live');
    });
}

async function sitePages() {
  const out = [];
  const feed = json('feed.json');
  for (const p of feed.posts) {
    const tests = (p.tests || []).length;
    const status = [p.kind, p.lane && `lane ${p.lane}`, `${tests} test${tests === 1 ? '' : 's'}`].filter(Boolean).join(', ');
    const cites = (p.cites || []).length ? `claims: ${p.cites.join(', ')}` : null;
    const file = p.file.includes('/') ? p.file : `posts/${p.file}`;   // a claim's file is under claims/
    out.push(asset('post', p.slug, p.title || p.titleText, file, `/p/${p.slug}`, cites, status));
  }
  for (const c of json('claims.json').claims) {
    const cited = (c.citedBy || []).length;
    out.push(asset('claim', c.slug, c.title, [c.file, c.twin], `/p/${c.slug}`, c.reference,
      [c.status, c.lane && `lane ${c.lane}`, c.topic, cited ? `cited by ${cited} posts` : null].filter(Boolean).join(', ')));
  }
  for (const c of feed.channels) {
    out.push(asset('channel', c.slug, c.title, `channels/${c.file}`, `/c/${c.slug}`, null,
      `${(c.posts || []).length} posts`));
  }
  for (const s of json('show.json').pages) {
    out.push(asset('show', s.slug, s.title, s.file, `/show/${s.slug}`, s.reference, `${(s.tests || []).length} tests`));
  }
  const { SCENES } = await import(pathToFileURL(join(DATA, 'scenes.mjs')).href);
  for (const s of SCENES) {
    const id = s.slug || s.id;
    out.push(asset('scene', id, s.title, 'www/src/data/scenes.mjs', s.href, s.ref,
      `${s.kind} scene, ${s.home ? 'on the home page' : 'in the gallery'}`));
  }
  for (const e of json('vivgallery.json').entries) {
    out.push(asset('gallery', e.id, e.title, [e.file, e.figFrom].filter(Boolean), e.href || e.post || e.explore,
      null, `${e.topic}/${e.section}`));
  }
  for (const e of json('gallery.json').entries) {
    out.push(asset('vipyr-gallery', e.slug, e.title, e.file, `/vipyr/gallery/${e.slug}`, null, `${e.topic}/${e.section}`));
  }
  for (const p of json('explore.json').programs) {
    out.push(asset('explore', p.slug, p.title, p.file, `/explore/${p.slug}`, null,
      `${p.defs} defs, ${p.steps} steps; walk at /explore/walk/${p.slug}`));
  }
  for (const l of json('ladders.json').ladders) {
    out.push(asset('ladder', l.name, l.title, `ladders/${l.file}`, `/ladders/${l.name}`, null, `${l.rungs.length} rungs`));
  }
  for (const s of json('pde.json').solvers) {
    out.push(asset('pde', s.slug, s.title, `pde/${s.file}`, `/pde/${s.slug}`, s.checksHtml,
      `${s.kind}, ${(s.tests || []).length} tests`));
  }
  for (const e of json('engines.json').engines) {
    const val = (e.validated || []).map((v) => v.file);
    out.push(asset('engine', e.name, e.title, [e.file, ...val], `/engines/${e.name}`,
      val.length ? `validated by ${val.join(', ')}` : null, `${e.lines} lines`));
  }
  for (const t of json('compare.json').topics) {
    for (const p of t.pairs) {
      const base = `study/compare/${p.topic}/${p.name}`;
      out.push(asset('compare', `${p.topic}/${p.name}`, p.title, [`${base}.py`, `${base}.viv`].filter(exists),
        `/compare/${p.topic}/${p.name}`, 'the Python program\'s output', `output ${p.output}`));
    }
  }
  for (const t of json('vivDocExamples.json').topics) {
    out.push(asset('doc', t.slug, t.title, 'www/src/data/vivDocExamples.json', `/docs/${t.slug}`, null, 'live'));
  }
  for (const s of json('dftSource.json').sources) {
    // a file outside dft/ (lib/lobpcg.viv) carries its repository path, as on /dft/source
    out.push(asset('dft-source', s.slug, s.title, s.path || `dft/${s.file}`, `/dft/source/${s.slug}`, null, s.group));
  }
  return out;
}

function vipyrAssets() {
  const v = json('vipyr.json');
  const out = [];
  const families = new Set(Object.values(v.functions).map((f) => f.family));
  for (const f of v.fields) {
    if (!families.has(f.id)) continue;
    out.push(asset('vipyr-family', f.id, f.title, 'www/src/data/vipyr.json', `/vipyr/${f.id}`, null, f.dist));
  }
  for (const q of Object.keys(v.functions).sort()) {
    const f = v.functions[q];
    const [family, name] = q.split('.');
    const ref = f.reference && typeof f.reference === 'object'
      ? f.reference.program || f.reference.compare || Object.values(f.reference)[0] : f.reference;
    out.push(asset('vipyr', `vp.${q}`, f.summary, f.sources || [], `/vipyr/${family}/${name}`, ref,
      [f.kind, f.dist].filter(Boolean).join(', ')));
  }
  const cm = json('codemap.json');
  for (const fn of Object.keys(cm.functions).sort()) {
    const e = cm.functions[fn];
    const m = cm.maps[e.map];
    out.push(asset('atlas', fn, e.title || (v.functions[fn] && `vp.${fn}`) || fn, m ? m.file : [], `/vipyr/map/${fn}`, null,
      m ? `${m.defs} defs in ${m.modules} modules` : null));
  }
  return out;
}

// ---------------------------------------------------------------------------
// programs

function programTitle(src) {
  const lines = src.split('\n');
  if (lines[0].trim() === '###') {
    const m = src.match(/^(?:post|title):\s*(.+)$/m);
    if (m) return m[1];
  }
  for (const l of lines) {
    const t = l.trim();
    if (!t) continue;
    if (/SPDX|Copyright/.test(t)) continue;
    const m = t.match(/^#[|:!]?\s*(.+)/);
    if (m && !/^#{3}$/.test(t)) return m[1].replace(/^\|+\s*/, '');
    if (!t.startsWith('#')) break;
  }
  return '';
}

function programs() {
  const refs = {};
  for (const p of json('programs.json').programs) if (p.reference) refs[`lang/viv/examples/${p.file}`] = p.reference;
  for (const s of json('show.json').pages) if (s.reference) refs[s.file] = s.reference;
  const nb = {};
  for (const [dir, head] of NB_DIRS) {
    for (const f of walk(join(ROOT, dir), (p) => p.endsWith('.viv'))) {
      const r = relative(join(ROOT, dir), f).split('\\').join('/');
      if (head === 'posts') {
        const m = r.match(/^(\d{4}-\d{2}-\d{2})-([a-z0-9-]+)\.viv$/);
        if (m) nb[rel(f)] = `/nb/posts/${m[2]}`;
      } else nb[rel(f)] = `/nb/${head}/${r.replace(/\.viv$/, '')}`;
    }
  }
  // every .viv file git knows, wherever it is
  return walk(ROOT, (p) => p.endsWith('.viv')).map((f) => {
    const r = rel(f);
    const src = readFileSync(f, 'utf8');
    const tests = (src.match(/^\s*test\s+["'(]/gm) || []).length;
    return asset('program', r, programTitle(src) || basename(r, '.viv'), r, nb[r], refs[r],
      tests ? `${tests} test${tests === 1 ? '' : 's'}` : 'no tests');
  });
}

// ---------------------------------------------------------------------------
// CLI verbs, tests, gate checks

/** A pyproject.toml's name and description. */
function pyproject(dir) {
  const p = join(dir, 'pyproject.toml');
  if (!existsSync(p)) return {};
  const t = readFileSync(p, 'utf8');
  return { name: (t.match(/^name\s*=\s*"([^"]+)"/m) || [])[1], description: (t.match(/^description\s*=\s*"([^"]+)"/m) || [])[1] };
}

/** packages/ and ports/: one record per distribution directory. */
function distributions(kind, top) {
  const base = join(ROOT, top);
  if (!isDir(base)) return [];
  return readdirSync(base).sort()
    .filter((d) => !d.startsWith('_') && !d.startsWith('.') && isDir(join(base, d)) && known(join(base, d), true))
    .map((d) => {
      const dir = join(base, d);
      const files = walk(dir, () => true).map(rel);
      const { name, description } = pyproject(dir);
      const vivs = files.filter((f) => f.endsWith('.viv')).length;
      const cmp = files.find((f) => basename(f) === 'COMPARISON.md');
      return asset(kind, `${top}/${d}/`, description || readmeLine(dir) || d, [`${top}/${d}/`, ...files], null,
        cmp ? `compared with upstream in ${cmp}` : null,
        [name && `pip name ${name}`, `${vivs} .viv files`].filter(Boolean).join(', '));
    });
}

/** The feature catalog: each `== id` block of lang/viv/features/*.txt. */
function features() {
  const dir = 'lang/viv/features';
  if (!exists(dir)) return [];
  let shown = new Set();
  try { shown = new Set(json('features.json').areas.map((a) => a.key)); } catch { /* no staged data */ }
  const out = [];
  for (const f of readdirSync(join(ROOT, dir)).sort().filter((x) => x.endsWith('.txt'))) {
    const area = f.replace(/\.txt$/, '');
    const blocks = read(`${dir}/${f}`).split(/^== /m).slice(1);
    for (const b of blocks) {
      const id = b.split('\n')[0].trim();
      const field = (k) => (b.match(new RegExp(`^${k}:\\s*(.+)$`, 'm')) || [])[1];
      const ex = field('example');
      const exFile = ex && `${dir}/examples/${ex}.viv`;
      out.push(asset('feature', id, field('title') || id, [`${dir}/${f}`, exFile && exists(exFile) ? exFile : null],
        shown.has(area) ? `/reference#features-${area}` : null, field('doc'), area));
    }
  }
  return out;
}

function cliVerbs() {
  const agent = read('lang/viv/src/agent.rs');
  const main = read('lang/viv/src/main.rs');
  const listed = new Map();
  const re = /Cmd \{ name: "([^"]+)", group: "([^"]+)", usage: "((?:[^"\\]|\\.)*)", what: "((?:[^"\\]|\\.)*)"/g;
  for (const m of agent.matchAll(re)) listed.set(m[1], { group: m[2], usage: m[3], what: m[4] });
  const dispatched = new Set([...main.matchAll(/args\[1\] == "([a-z][a-z0-9-]*)"/g)].map((m) => m[1]));
  const out = [];
  for (const [name, c] of listed) {
    out.push(asset('cli', `viv ${name}`, c.what, 'lang/viv/src/agent.rs', null, null, `viv help (${c.group}); ${c.usage.replace(/\\"/g, '"')}`));
  }
  for (const name of [...dispatched].sort()) {
    if (listed.has(name)) continue;
    out.push(asset('cli', `viv ${name}`, `viv ${name}`, 'lang/viv/src/main.rs', null, null, 'not in viv help'));
  }
  return out;
}

function tests() {
  const out = [];
  const add = (files, how) => {
    for (const f of files) out.push(asset('test', rel(f), how(f).title, rel(f), null, null, how(f).run));
  };
  add(walk(join(ROOT, 'lang/viv/tests'), (p) => p.endsWith('.rs') && dirname(p) === join(ROOT, 'lang/viv/tests')),
    (f) => ({ title: `cargo test of viv: ${basename(f, '.rs')}`, run: `cargo test -p viv --test ${basename(f, '.rs')}` }));
  for (const c of readdirSync(join(ROOT, 'crates')).sort()) {
    add(walk(join(ROOT, 'crates', c, 'tests'), (p) => p.endsWith('.rs')),
      (f) => ({ title: `cargo test of ${c}: ${basename(f, '.rs')}`, run: `cargo test -p ${c} --test ${basename(f, '.rs')}` }));
  }
  const py = (dir) => walk(join(ROOT, dir), (p) => /\/test_[^/]+\.py$/.test(p));
  for (const d of ['python/tests', 'packages', 'tools', 'xc', 'catalog']) {
    add(py(d), (f) => ({ title: `pytest: ${basename(f, '.py').replace(/^test_/, '')}`, run: `python3 -m pytest -q ${rel(f)}` }));
  }
  const pkg = JSON.parse(read('www/package.json')).scripts;
  for (const name of Object.keys(pkg).sort()) {
    if (!/test$|:check$|^check$/.test(name) || name === 'check') continue;
    out.push(asset('test', `www:${name}`, `site check ${name}`, 'www/package.json', null, null, `cd www && npm run ${name}`));
  }
  if (exists('tests/cc')) {
    out.push(asset('test', 'tests/cc/', 'C programs the C front end compiles and runs', 'tests/cc/', null, null, 'tools/gate.sh quick'));
  }
  return out;
}

// The checks as gate.py builds them (literal Check(...) entries and the ones
// three() makes alike), read from gate.CHECKS rather than parsed from source.
const GATE_DUMP = `import json, sys
sys.path.insert(0, sys.argv[1])
import gate
print(json.dumps([{"name": c.name, "tier": c.tier, "only": sorted(c.only) if c.only else None,
                   "what": c.what, "cmd": c.cmd} for c in gate.CHECKS]))`;

function gateChecks() {
  const out = execFileSync('python3', ['-c', GATE_DUMP, join(ROOT, 'tools/gate')],
    { cwd: ROOT, encoding: 'utf8', maxBuffer: 1 << 26, stdio: ['ignore', 'pipe', 'inherit'] });
  return JSON.parse(out).map((c) => asset('gate', c.name, c.what || c.name, 'tools/gate/gate.py', null, null,
    c.only ? `tier ${c.tier}, runs in ${c.only.join(', ')}` : `tier ${c.tier}`));
}

// ---------------------------------------------------------------------------

export async function build() {
  const all = [
    ...dirs(), ...staticPages(), ...(await sitePages()), ...vipyrAssets(), ...programs(),
    ...distributions('package', 'packages'), ...distributions('port', 'ports'),
    ...cliVerbs(), ...features(), ...tests(), ...gateChecks(),
  ];
  // one asset per kind and id: a later duplicate is dropped; an asset whose
  // every source file is excluded from the public repository is dropped too
  const key = new Set();
  const items = all.filter((a) => !a.gone).filter((a) => { const k = `${a.kind}\t${a.id}`; if (key.has(k)) return false; key.add(k); return true; });
  items.sort((a, b) => (KIND_ORDER[a.kind] - KIND_ORDER[b.kind]) || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  const counts = Object.fromEntries(KINDS.map(([k]) => [k, items.filter((a) => a.kind === k).length]));
  return {
    format: 1,
    about: 'Every asset of the Viv repository. Written by tools/inventory.mjs; do not edit by hand.',
    kinds: KINDS.map(([k, what]) => ({ kind: k, what, count: counts[k] })),
    total: items.length,
    items,
  };
}

const mdCell = (s) => (s == null ? '' : String(s)).replace(/\|/g, '\\|');

function markdown(inv) {
  const lines = [
    '<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->',
    '<!-- SPDX-FileCopyrightText: 2026 The Viv authors -->',
    '<!-- Written by tools/inventory.mjs from inventory.json; do not edit by hand. -->',
    '# Inventory',
    '',
    `Every asset of the repository, ${inv.total} in all. The same list is in`,
    'inventory.json, on the site at /inventory, and in the terminal:',
    '',
    '```sh',
    'node tools/inventory.mjs ls                 # everything, one line each',
    'node tools/inventory.mjs ls post            # one kind',
    'node tools/inventory.mjs ls --grep pyscf    # assets whose fields mention a word',
    'node tools/inventory.mjs ls program --json  # the records, as JSON lines',
    '```',
    '',
    '| kind | count | what |',
    '|---|---:|---|',
    ...inv.kinds.map((k) => `| [${k.kind}](#${k.kind}) | ${k.count} | ${k.what} |`),
    '',
  ];
  for (const k of inv.kinds) {
    if (!k.count) continue;
    lines.push(`## ${k.kind}`, '', `${k.what}. ${k.count} in all.`, '', '| id | title | source | url | checked against | status |', '|---|---|---|---|---|---|');
    for (const a of inv.items.filter((x) => x.kind === k.kind)) {
      const src = a.source.length > 3 ? `${a.source.slice(0, 3).join(', ')} and ${a.source.length - 3} more` : a.source.join(', ');
      lines.push(`| ${mdCell(a.id)} | ${mdCell(a.title === a.id ? '' : a.title)} | ${mdCell(src)} | ${mdCell(a.url || '')} | ${mdCell(a.checked || '')} | ${mdCell(a.status || '')} |`);
    }
    lines.push('');
  }
  return lines.join('\n');
}

function ls(inv, args) {
  const asJson = args.includes('--json');
  const gi = args.indexOf('--grep');
  const grep = gi >= 0 ? (args[gi + 1] || '').toLowerCase() : null;
  const words = args.filter((a, i) => !a.startsWith('--') && !(gi >= 0 && i === gi + 1));
  const kind = words[0] || null;
  if (kind && !(kind in KIND_ORDER)) {
    console.error(`inventory: no kind ${kind}; the kinds are ${KINDS.map(([k]) => k).join(', ')}`);
    return 2;
  }
  let items = inv.items.filter((a) => !kind || a.kind === kind);
  if (grep) items = items.filter((a) => JSON.stringify(a).toLowerCase().includes(grep));
  if (asJson) {
    for (const a of items) console.log(JSON.stringify(a));
    return 0;
  }
  const w = Math.min(48, Math.max(4, ...items.map((a) => a.id.length)));
  for (const a of items) {
    const where = a.url || a.source[0] || '';
    const label = a.title !== a.id ? `  ${a.title}` : '';
    console.log(`${kind ? '' : `${a.kind.padEnd(13)} `}${a.id.padEnd(w)}  ${where}${label}`);
  }
  if (process.stdout.isTTY) console.error(`${items.length} assets`);
  return 0;
}

async function main(argv) {
  const inv = await build();
  const js = `${JSON.stringify(inv, null, 1)}\n`;
  const md = `${markdown(inv)}\n`;
  if (argv[0] === 'ls') return ls(inv, argv.slice(1));
  if (argv[0] === 'kinds') {
    for (const k of inv.kinds) console.log(`${k.kind.padEnd(14)} ${String(k.count).padStart(5)}  ${k.what}`);
    console.log(`${'total'.padEnd(14)} ${String(inv.total).padStart(5)}`);
    return 0;
  }
  writeFileSync(OUT_JSON, js);
  writeFileSync(OUT_MD, md);
  console.log(`inventory: ${inv.total} assets -> inventory.json, INVENTORY.md`);
  return 0;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main(process.argv.slice(2)).then((c) => process.exit(c), (e) => { console.error(e); process.exit(1); });
}
