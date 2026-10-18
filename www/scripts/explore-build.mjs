// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// explore-build -- the bundles /explore loads, one per program in
// src/data/explorables.mjs.
//
// For each program this reads the file, every module its imports reach
// and each module's .intent and .debug files, and hands them to viv.wasm's
// `viv_explore_bundle` (lang/viv/src/graph.rs): the same code the compiler's
// `viv graph --bundle` runs. The bundle holds the program as the explorer
// runs it, its graph (defs, steps, loops, each step's interface), the line
// table, the map back to each module's lines, and the intent and debug lenses.
// The reader's tab runs the program and
// records the story; here it is run once, plainly, for its call graph (which
// defs ran and how often each called which), and the picture of the program
// is drawn from it by lib/graphdraw.viv in viv.wasm (src/lib/graphView.js
// says which defs are boxes; the page redraws with the same code).
//
// Writes www/public/explore/<slug>.json (and <slug>.calls.json, the run's
// call counts alone) and www/src/data/explore.json (the
// index: title, blurb, and counts the pages quote); the call graphs are kept
// in www/.explore-cache by the program's hash. Fails the build if a
// program's bundle cannot be made, or if any of its math does not typeset:
// every `$...$` in the intent, every glossary symbol and every line's LaTeX
// (lang/viv/src/tex.rs) goes through the site's KaTeX with errors on. `viv
// lens check` checks the same math in Rust, for what KaTeX would refuse; this
// is the final word.
import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import katex from 'katex';
import { createHash } from 'node:crypto';
import { EXPLORABLES } from '../src/data/explorables.mjs';
import { callsOf, graphModel, defaultOpen, drawSource, drawingOf, drawLibrary } from '../src/lib/graphView.js';
import { importsOf } from '../src/lib/uses.js';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const OUT = join(WWW, 'public/explore');
mkdirSync(OUT, { recursive: true });
// the runs' call graphs, by program hash
const CACHE = join(WWW, '.explore-cache');
mkdirSync(CACHE, { recursive: true });

const wasm = join(WWW, 'public/viv/viv.wasm');
if (!existsSync(wasm)) {
  console.error('explore-build: no public/viv/viv.wasm (run www/build-wasm.sh)');
  process.exit(1);
}
const { instance } = await WebAssembly.instantiate(readFileSync(wasm), {});
const ex = instance.exports;
if (!ex.viv_explore_bundle) {
  console.error('explore-build: this viv.wasm has no viv_explore_bundle (rebuild it with www/build-wasm.sh)');
  process.exit(1);
}
const call = (fn, text) => {
  const enc = new TextEncoder().encode(text);
  const p = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, p, enc.length).set(enc);
  const packed = fn(p, enc.length);
  return new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
};

/** A module by its repository name, with its lens files when it has them. */
function mod(name) {
  const o = { name, src: readFileSync(join(ROOT, `${name}.viv`), 'utf8') };
  for (const k of ['intent', 'debug']) {
    const p = join(ROOT, `${name}.${k}`);
    if (existsSync(p)) o[k] = readFileSync(p, 'utf8');
  }
  return o;
}

/** The math of a bundle that does not typeset: [where, what KaTeX said]. */
function badMath(b) {
  const out = [];
  const tex = (where, t, display) => {
    try {
      katex.renderToString(t, { throwOnError: true, displayMode: display, strict: 'ignore' });
    } catch (err) {
      out.push([where, `${t.slice(0, 80)}: ${String(err.message || err).split('\n')[0]}`]);
    }
  };
  const spans = (where, text) => {
    const t = String(text || '').replace(/\\\$/g, '');
    for (const m of t.matchAll(/\$\$([^$]+)\$\$|\$([^$]+)\$/g)) tex(where, m[1] || m[2], !!m[1]);
  };
  for (const [def, it] of Object.entries(b.intent || {})) {
    for (const k of ['what', 'how', 'why']) spans(`${def} ${k}`, it[k]);
    for (const [name, t] of it.glossary || []) spans(`${def}:${name}`, t);
    for (const [name, sym] of Object.entries(it.symbols || {})) tex(`${def}:${name} symbol`, sym, false);
    // the variable docs, each reuse site's included
    for (const d of it.docs || []) {
      spans(`${def}:${d.key}`, d.text);
      if (d.symbol) tex(`${def}:${d.key} symbol`, d.symbol, false);
    }
  }
  for (const p of b.paths || []) {
    spans(`path ${p.id}`, p.title);
    for (const st of p.stops) {
      spans(`path ${p.id} ${st.target}`, st.text);
      for (const d of st.details) spans(`path ${p.id} ${st.target} ${d.label}`, d.text.replace(/\[(hold|test) "[^"]*"\]/g, ''));
    }
  }
  for (const [k, t] of Object.entries(b.steps || {})) spans(`step ${k}`, t);
  for (const d of b.details || []) spans(`${d.def}${d.step ? '/' + d.step : ''} ${d.label}`, d.text.replace(/\[(hold|test) "[^"]*"\]/g, ''));
  for (const [m, t] of Object.entries(b.moduleText || {})) spans(`module ${m}`, t);
  for (const r of b.tex || []) tex(`${r.def} line ${r.line}`, r.tex, false);
  return out;
}

// The architecture lenses: each `.arch` file,
// with the directories its `covers` line names, and the members each of its
// multi-paths was last seen agreeing with (arch_ratchet.json, which `viv arch
// check --update --verify` writes). A program under a lens's directories gets
// its multi-paths in its bundle.
const ARCHES = [];
(function findArches(dir) {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    if (e.name.startsWith('.') || ['node_modules', 'www', 'target', 'dist', 'public'].includes(e.name) || e.name.startsWith('target')) continue;
    const p = join(dir, e.name);
    if (e.isDirectory()) findArches(p);
    else if (e.name.endsWith('.arch')) {
      const text = readFileSync(p, 'utf8');
      const covers = ((text.match(/^\s+covers\s+(.*)$/m) || [])[1] || '').split(/\s+/).filter(Boolean);
      ARCHES.push({ path: p.slice(ROOT.length + 1), text, covers });
    }
  }
})(ROOT);
const RATCHET = existsSync(join(ROOT, 'arch_ratchet.json')) ? JSON.parse(readFileSync(join(ROOT, 'arch_ratchet.json'), 'utf8')) : {};
const archFor = (file) => ARCHES.find((a) => a.covers.some((c) => file.startsWith(`${c}/`)));

// The modules a lens's stages are defined in that a program does not load:
// the members off the production path, the registries and the intent defs,
// with the modules they use. The bundle measures the stages with them, as
// `viv arch report` does with every covered module, so a multi-path whose
// other members the program never loads still shows them, their
// independence and its intent.
// The modules a module's import lines name, by their repository names, found
// as the compiler finds them (lang/viv/src/imports.rs, `candidates`): beside
// the module, under lib/, under the repository root.
function depsOf(name, src) {
  const dir = name.includes('/') ? name.slice(0, name.lastIndexOf('/')) : '';
  const out = [];
  for (const d of importsOf(src)) {
    const p = d.replace(/\./g, '/');
    const cands = p.startsWith('lib/') ? [p] : [dir && `${dir}/${p}`, `lib/${p}`, p].filter(Boolean);
    const hit = cands.find((c) => existsSync(join(ROOT, `${c}.viv`)));
    if (hit && !out.includes(hit)) out.push(hit);
  }
  return out;
}

function archModules(arch, have) {
  const names = new Set();
  for (const m of arch.text.matchAll(/^\s+(members|intent|registry|default)\s+(.*)$/gm)) for (const n of m[2].split(/\s+/)) if (n) names.add(n);
  for (const m of arch.text.matchAll(/^\s+contract\s+([A-Za-z_]\w*)\s+within\b/gm)) names.add(m[1]);
  const owner = new Map();
  const scan = (dir) => {
    if (!existsSync(dir)) return;
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const p = join(dir, e.name);
      if (e.isDirectory()) scan(p);
      else if (e.name.endsWith('.viv')) {
        const name = p.slice(ROOT.length + 1).replace(/\.viv$/, '');
        for (const m of readFileSync(p, 'utf8').matchAll(/^def\s+([A-Za-z_]\w*)\s*\(/gm)) if (names.has(m[1]) && !owner.has(m[1])) owner.set(m[1], name);
      }
    }
  };
  for (const c of arch.covers) scan(join(ROOT, c));
  const out = [];
  const seen = new Set(have);
  const add = (name) => {
    if (seen.has(name)) return;
    seen.add(name);
    const o = mod(name);
    out.push(o);
    for (const d of depsOf(name, o.src)) add(d);
  };
  for (const name of new Set(owner.values())) add(name);
  return out;
}

// lib/graphdraw with the modules it uses, as the page runs it
const drawLib = drawLibrary((name) => readFileSync(join(ROOT, `${name}.viv`), 'utf8'));
const index = [];
let failed = 0;
for (const e of EXPLORABLES) {
  const root = mod(e.file.replace(/\.viv$/, ''));
  const used = [];
  const seen = new Set();
  const walk = (name, src) => {
    for (const d of depsOf(name, src)) {
      if (seen.has(d)) continue;
      seen.add(d);
      const o = mod(d);
      used.push(o);
      walk(d, o.src);
    }
  };
  walk(root.name, root.src);
  const arch = archFor(e.file);
  const request = arch ? { root, used, arch: { path: arch.path, text: arch.text, modules: archModules(arch, [root.name, ...used.map((u) => u.name)]) }, agreeing: (RATCHET[arch.path] || {}).agreeing || {} } : { root, used };
  const text = call(ex.viv_explore_bundle, JSON.stringify(request));
  if (text.startsWith('error')) {
    console.error(`explore-build: ${e.file}: ${text}`);
    failed++;
    continue;
  }
  const b = JSON.parse(text);
  const bad = badMath(b);
  if (bad.length) {
    for (const [where, why] of bad.slice(0, 20)) console.error(`explore-build: ${e.file}: math that KaTeX cannot typeset, at ${where}: ${why}`);
    failed++;
    continue;
  }
  // the run's call graph: which defs ran, and how many times each called
  // which (the interpreter's `__graph__` line). The picture has a box for
  // each def that ran, not for every def the text defines. Kept in
  // .explore-cache by the program's hash: the run is the same every time.
  const hash = createHash('sha256').update(b.program).digest('hex').slice(0, 16);
  const cached = join(CACHE, `${e.slug}-${hash}.json`);
  let calls = null;
  let runMs = 0;
  if (existsSync(cached)) calls = JSON.parse(readFileSync(cached, 'utf8'));
  else {
    const t0 = performance.now();
    const out = call(ex.viv_run_release, b.program);
    runMs = performance.now() - t0;
    calls = callsOf(out);
    if (!calls) {
      console.error(`explore-build: ${e.file}: the run printed no call graph: ${out.slice(0, 300)}`);
      failed++;
      continue;
    }
    writeFileSync(cached, JSON.stringify(calls));
  }
  b.calls = calls;
  // the whole program as a picture, drawn by lib/graphdraw.viv's draw_view in
  // viv.wasm: the page runs the same library and the same code
  // (src/lib/graphView.js) when the reader opens or closes a box
  const model = graphModel(b);
  const t1 = performance.now();
  const { src: viewSrc, shown } = drawSource(model, defaultOpen(model), { title: b.name });
  const out = call(ex.viv_run_release, drawLib + viewSrc);
  const drawMs = performance.now() - t1;
  const drawing = drawingOf(out);
  const boxes = (drawing.match(/^box: /gm) || []).length;
  if (!drawing.startsWith('figure: ') || !boxes) {
    console.error(`explore-build: ${e.file}: the program's drawing failed: ${out.slice(0, 300)}`);
    failed++;
    continue;
  }
  // the digits lens, when `viv digits <file> -o <file>.digits.json` wrote one
  // beside the program
  const digitsFile = join(ROOT, e.file.replace(/\.viv$/, '.digits.json'));
  const digits = existsSync(digitsFile) ? `,"digits":${JSON.stringify(JSON.parse(readFileSync(digitsFile, 'utf8')))}` : '';
  const withDrawing = `${text.trimEnd().replace(/}$/, '')},"calls":${JSON.stringify(calls)},"drawLib":${JSON.stringify(drawLib)},"drawing":${JSON.stringify(drawing)}${digits}}`;
  writeFileSync(join(OUT, `${e.slug}.json`), withDrawing);
  // the run's call counts alone, for the vipyr code maps (/vipyr/map/<function>,
  // components/CodeMap.jsx), which show them on the defs of the same library
  writeFileSync(join(OUT, `${e.slug}.calls.json`), JSON.stringify({ program: e.file, calls }));
  const count = (k) => b.nodes.filter((n) => n.kind === k).length;
  const stepped = b.nodes.filter((n) => n.kind === 'def' && n.stepped).map((n) => n.id);
  index.push({
    slug: e.slug,
    file: e.file,
    title: e.title,
    blurb: e.blurb,
    posts: e.posts || [],
    defs: count('def'),
    steps: count('step'),
    loops: count('loop'),
    stepped,
    modules: b.modules.length,
    bytes: withDrawing.length,
    drawn: boxes,
    ran: model.ran.length,
    shown: shown.size,
  });
  const mpNote = b.multipaths ? `; ${b.multipaths.stages.filter((x) => x.multi).length} multi-paths from ${b.multipaths.arch}, ${model.mps.size} used by the run` : '';
  console.log(`explore-build: ${e.slug} (${e.file})${mpNote}: ${count('def')} defs, ${count('step')} steps in ${stepped.length} of them, ${b.modules.length} modules, ${Object.values(b.intent || {}).reduce((s, it) => s + (it.docs || []).length, 0)} variable docs, ${(b.paths || []).map((p) => `a ${p.stops.length}-stop path`).join(', ') || 'no path'}; ${model.ran.length} defs ran${runMs ? ` (run in ${(runMs / 1000).toFixed(1)} s)` : ''}, ${shown.size} drawn at first as ${boxes} boxes in ${Math.round(drawMs)} ms; ${(withDrawing.length / 1e6).toFixed(1)} MB`);
}
writeFileSync(join(WWW, 'src/data/explore.json'), JSON.stringify({ programs: index }, null, 1) + '\n');
if (failed) process.exit(1);
