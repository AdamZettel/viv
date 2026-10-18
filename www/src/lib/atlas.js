// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The atlas of any program on the site, made in the reader's browser (/atlas,
// components/Explorer.jsx and ZoomWalk.jsx with a `load` in place of a slug).
// The eight programs of /explore have their bundles made at build time
// (scripts/explore-build.mjs); this does the same steps here, for everything
// else:
//
//   1. the modules: the compiler (viv_expand, lib/uses.js) names each module
//      the program's imports reach; each is fetched from the staged tree under
//      /programs, which the workbench reads too. /programs/atlas-index.json
//      lists what is staged, so a candidate path that is not there is never
//      fetched.
//   2. the bundle: viv_explore_bundle (lang/viv/src/graph.rs), in a worker
//      (public/atlas-worker.js), from the program and those module texts.
//   3. the run's call graph: the program run once, plainly, for which defs ran
//      and how often each called which. A program that reads input files, or
//      that runs past RUN_BUDGET_MS, is not run: its atlas is the static one,
//      the calls the text makes, and the page says why.
//   4. the picture: lib/graphdraw.viv's draw_view in viv.wasm, as the build
//      draws it (lib/graphView.js).
import { expandUses } from './uses.js';
import { callsOf, graphModel, defaultOpen, drawSource, drawingOf, drawLibrary } from './graphView.js';
import { importsOf, pathsOf, stagedPrograms } from './uses.js';
import { vivWorker } from './vivWasm.js';
export { readDraft } from './atlasLink.js';

/** The longest the call-graph run may take before the atlas is shown without a run. */
export const RUN_BUDGET_MS = 20000;
/** The longest the explorer's recording may take before it gives up and keeps the atlas. */
export const RECORD_BUDGET_MS = 60000;
/** How many times a module the bundle still asks for is fetched and the bundle made again. */
const MAX_REFETCH = 16;
/** The builtins that read files: a program that calls one needs input this site does not have. */
const READS_FILES = /\bread_(text|lines|bytes)\s*\(/;

/** The staged programs, by path without `.viv` (the workbench reads the same list). */
const stagedIndex = () => stagedPrograms().then((s) => s || new Set());

/**
 * The resolver for a program named `name` (its staged path without `.viv`,
 * or `scratch` for text with no file): `./x` is beside it, `post:<slug>` is a
 * post, anything else a staged path.
 */
function resolverFor(name) {
  const dir = name.includes('/') ? name.slice(0, name.lastIndexOf('/')) : '';
  const cache = new Map();
  const fetchText = (path) => {
    if (!cache.has(path)) cache.set(path, fetch(`/programs/${path}.viv`).then((r) => (r.ok ? r.text() : null)).catch(() => null));
    return cache.get(path);
  };
  return async (want) => {
    const staged = await stagedIndex();
    let path = want;
    if (want.startsWith('post:')) {
      const posts = await fetch('/programs/posts/index.json').then((r) => (r.ok ? r.json() : {})).catch(() => ({}));
      const file = posts[want.slice(5)];
      if (!file) return null;
      path = `posts/${file.replace(/\.viv$/, '')}`;
    } else if (want.startsWith('./')) {
      path = dir ? `${dir}/${want.slice(2)}` : want.slice(2);
    }
    if (!staged.has(path)) return null;
    return fetchText(path);
  };
}

function worker() {
  return vivWorker('/atlas-worker.js');
}

/** One request to an atlas worker; rejects after `budget` ms (the caller ends the worker). */
function ask(w, op, text, budget = 0) {
  return new Promise((resolve, reject) => {
    const id = Math.random().toString(36).slice(2);
    let timer = null;
    const on = (e) => {
      if (e.data.id !== id) return;
      w.removeEventListener('message', on);
      if (timer) clearTimeout(timer);
      if (e.data.error) reject(new Error(e.data.error));
      else resolve(e.data);
    };
    w.addEventListener('message', on);
    if (budget) {
      timer = setTimeout(() => {
        w.removeEventListener('message', on);
        const err = new Error('budget');
        err.budget = true;
        reject(err);
      }, budget);
    }
    w.postMessage({ id, op, text });
  });
}

/** lib/graphdraw and the modules it imports, fetched, then spliced as the build splices them. */
async function drawLib(resolve) {
  const texts = new Map();
  const get = async (name) => {
    if (texts.has(name)) return;
    const t = await resolve(name);
    texts.set(name, t);
    if (t === null) return;
    for (const d of importsOf(t)) {
      for (const p of pathsOf(d)) {
        await get(p);
        if (texts.get(p) !== null) break;
      }
    }
  };
  await get('lib/graphdraw');
  return drawLibrary((name) => {
    const t = texts.get(name);
    if (t === null || t === undefined) throw new Error(`no module ${name}`);
    return t;
  });
}

/**
 * The atlas of a program: its explorer bundle with the run's call counts and
 * the first picture. `{ name, src }` is the program; `onPhase(text)` hears
 * what is happening. Returns `{ bundle, ran, note }`: `ran` is false when the
 * program was not run, and `note` then says why.
 */
export async function buildAtlas({ name, src, onPhase = () => {} }) {
  const resolve = resolverFor(name);
  onPhase('Finding the modules the program imports');
  const modules = {};
  await expandUses(src, resolve, [], new Set(), modules);
  const used = [];
  const have = new Set();
  for (const [k, t] of Object.entries(modules)) {
    const n = k.replace(/^\//, '');
    if (t === null || have.has(n)) continue;
    have.add(n);
    used.push({ name: n, src: t });
  }
  const w = worker();
  try {
    onPhase(`Making the bundle from the program and ${used.length} ${used.length === 1 ? 'module' : 'modules'}`);
    let text = '';
    for (let round = 0; ; round += 1) {
      text = (await ask(w, 'bundle', JSON.stringify({ root: { name, src }, used }))).out;
      // a module only the bundle's expansion reaches (a conditional import):
      // fetch the first candidate it looked for that is staged, and again
      const m = text.startsWith('error') && text.match(/looked for ([^)]*)\)/);
      if (!m || round >= MAX_REFETCH) break;
      let got = false;
      for (const c of m[1].split(', ')) {
        const n = c.replace(/\.viv$/, '');
        if (have.has(n)) continue;
        const t = await resolve(n);
        if (t === null) continue;
        have.add(n);
        used.push({ name: n, src: t });
        got = true;
        break;
      }
      if (!got) break;
    }
    if (text.startsWith('error')) throw new Error(text.replace(/^error:\s*/, ''));
    const bundle = JSON.parse(text);

    let ran = false;
    let note = null;
    if (READS_FILES.test(bundle.program)) {
      note = 'The program reads input files that the site does not have, so the atlas shows the calls its text makes without running it.';
    } else {
      onPhase('Running the program once for its call graph');
      const runner = worker();
      try {
        const r = await ask(runner, 'run', bundle.program, RUN_BUDGET_MS);
        const calls = callsOf(r.out);
        if (calls) {
          bundle.calls = calls;
          ran = true;
        } else if (!r.out.startsWith('error')) {
          // the interpreter prints a call graph only when a def was called: a
          // run that finished without calling one has an empty graph
          bundle.calls = [];
          ran = true;
        } else {
          const why = r.out.replace(/^error:\s*/, '').split('\n')[0].slice(0, 200);
          note = `The run stopped (${why}), so the atlas shows the calls the program's text makes.`;
        }
      } catch (err) {
        // a trap inside viv.wasm (stack overflow, out of memory) ends the run,
        // not the atlas: the bundle is already made
        if (err.budget) {
          note = `The run was stopped after ${RUN_BUDGET_MS / 1000} s, and the atlas shows the calls the program's text makes.`;
        } else {
          const why = String(err.message || err).split('\n')[0].slice(0, 200);
          note = `The run stopped (${why}), so the atlas shows the calls the program's text makes.`;
        }
      } finally {
        runner.terminate();
      }
    }

    onPhase('Drawing the atlas');
    const lib = await drawLib(resolve);
    const model = graphModel(bundle);
    const { src: viewSrc } = drawSource(model, defaultOpen(model), { title: bundle.name });
    const drawn = await ask(w, 'run', lib + viewSrc);
    const drawing = drawingOf(drawn.out);
    if (drawing.startsWith('figure: ')) {
      bundle.drawLib = lib;
      bundle.drawing = drawing;
    }
    return { bundle, ran, note };
  } finally {
    w.terminate();
  }
}
