// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// dev-assets -- before `npm run dev`: put back the generated, git-ignored
// files the pages load, so a fresh clone's workbench and /explore work.
//
//     node scripts/dev-assets.mjs        (astro.config.mjs runs it when the dev server starts)
//
// Cheap ones are staged here when missing or stale; the expensive ones are
// built only when what they need is present, and otherwise named in one line
// with the command that builds them. It never fails the dev server: a page
// whose asset is missing says so itself.
//
//   public/viv/viv.wasm      the browser compiler        npm run wasm (cargo, minutes)
//   public/embed/viv.wasm    the embed's copy            copied from public/viv
//   public/monaco/           the workbench's editor      scripts/stage-monaco.mjs
//   public/programs/         the IDE's example files     scripts/stage.mjs
//   public/llms*.txt         the language for LLMs       scripts/llms-build.mjs
//   public/explore/*.json    /explore's bundles          scripts/explore-build.mjs (needs viv.wasm)
//   public/dft/, engines/    the in-tab DFT and engines  scripts/dft-precompute.mjs, engine-build.mjs
//
// `npm run stage` regenerates everything (the build runs it); `npm run dev:full`
// is that followed by the dev server.
import { copyFileSync, existsSync, mkdirSync, openSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { spawn, spawnSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const P = (p) => join(WWW, p);
const mtime = (p) => (existsSync(p) ? statSync(p).mtimeMs : 0);
const missing = [];

/** The newest mtime of the files under dir whose names end in ext. */
function newest(dir, ext) {
  let t = 0;
  if (!existsSync(dir)) return t;
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) {
      if (e.name !== 'target' && e.name !== 'node_modules') t = Math.max(t, newest(p, ext));
    } else if (e.name.endsWith(ext)) t = Math.max(t, statSync(p).mtimeMs);
  }
  return t;
}

function run(script, why) {
  console.log(`dev-assets: ${why}: node scripts/${script}`);
  const r = spawnSync(process.execPath, [P(`scripts/${script}`)], { cwd: WWW, stdio: 'inherit' });
  if (r.status !== 0) missing.push(`scripts/${script} failed (exit ${r.status}); run it by hand to see why`);
  return r.status === 0;
}

// ---- the browser compiler ---------------------------------------------------------
const wasm = P('public/viv/viv.wasm');
const hasWasm = existsSync(wasm);
if (!hasWasm) {
  missing.push('public/viv/viv.wasm (the in-browser compiler): build it with `npm run wasm` (needs cargo and the wasm32 target)');
} else {
  const src = Math.max(newest(join(ROOT, 'lang/viv/src'), '.rs'), newest(join(ROOT, 'knotkit'), '.rs'));
  if (src > mtime(wasm))
    console.log('dev-assets: public/viv/viv.wasm is older than the compiler sources; `npm run wasm` rebuilds it');
  const embed = P('public/embed/viv.wasm');
  if (mtime(embed) < mtime(wasm)) {
    mkdirSync(dirname(embed), { recursive: true });
    copyFileSync(wasm, embed);
    console.log('dev-assets: copied viv.wasm to public/embed/');
  }
}

// ---- Monaco, for the workbench --------------------------------------------------
const monacoPkg = P('node_modules/monaco-editor/package.json');
if (!existsSync(monacoPkg)) missing.push('node_modules/monaco-editor: run `npm install`');
else if (!existsSync(P('public/monaco/vs/loader.js')) || mtime(monacoPkg) > mtime(P('public/monaco/vs/loader.js')))
  run('stage-monaco.mjs', 'public/monaco is missing or older than node_modules');

// ---- the example programs and the LLM text --------------------------------------
const examples = join(ROOT, 'lang/viv/examples');
if (hasWasm) {
  if (!existsSync(P('public/programs/examples')) || newest(examples, '.viv') > mtime(P('public/programs/examples')))
    run('stage.mjs', 'public/programs is missing or older than lang/viv/examples');
  if (!existsSync(P('public/llms.txt'))) run('llms-build.mjs', 'public/llms.txt is missing');
} else if (!existsSync(P('public/programs/examples'))) {
  missing.push('public/programs (the IDE\'s examples): `node scripts/stage.mjs` once viv.wasm is built');
}

// ---- /explore's bundles -----------------------------------------------------------
// Missing bundles, and bundles older than the compiler or the programs they
// record, are rebuilt in the background: explore-build takes from seconds to a
// minute, and the dev server should not wait for it. A page whose bundle is
// still being built says so until the build finishes.
const explore = JSON.parse(readFileSync(P('src/data/explore.json'), 'utf8'));
const progs = Math.max(newest(join(ROOT, 'lib'), '.viv'), ...explore.programs.map((p) => newest(join(ROOT, dirname(p.file)), '.viv')));
const stale = explore.programs
  .filter((p) => {
    const b = mtime(P(`public/explore/${p.slug}.json`));
    return !b || b < mtime(wasm) || b < progs;
  })
  .map((p) => p.slug);
if (stale.length) {
  // one build at a time: a dev server restarted while the last build still runs
  // would otherwise start a second one writing the same files
  const pidFile = P('.explore-build.pid');
  const running = (() => {
    try {
      process.kill(Number(readFileSync(pidFile, 'utf8')), 0);
      return true;
    } catch {
      return false;
    }
  })();
  if (hasWasm && running) console.log('dev-assets: an /explore bundle build is already running; log in www/.explore-build.log');
  else if (hasWasm) {
    const log = P('.explore-build.log');
    const out = openSync(log, 'w');
    const child = spawn(process.execPath, [P('scripts/explore-build.mjs')], { cwd: WWW, detached: true, stdio: ['ignore', out, out] });
    child.unref();
    writeFileSync(pidFile, String(child.pid));
    console.log(`dev-assets: rebuilding ${stale.length} of ${explore.programs.length} /explore bundles in the background (${stale.join(', ')}); log in www/.explore-build.log`);
  } else
    missing.push(
      `public/explore/*.json (${stale.length} bundles for /explore): \`node scripts/explore-build.mjs\` once viv.wasm is built`,
    );
}

// ---- the in-tab DFT and the engines: precompiled, slow; named, not built ----------
// the showcase programs the Run buttons and the home page's demo scroll run
if (hasWasm && !existsSync(P('public/programs/show'))) run('show-build.mjs', 'public/programs/show is missing');

if (!existsSync(P('public/dft')) || !readdirSync(P('public/dft')).some((f) => f.endsWith('.wasm')))
  missing.push('public/dft (the in-tab DFT on /dft): `node scripts/dft-precompute.mjs`');
if (!existsSync(P('public/engines'))) missing.push('public/engines (the /engines runs): `node scripts/engine-build.mjs`');

if (missing.length) {
  console.log('dev-assets: not staged, so the pages that load them will say so:');
  for (const m of missing) console.log('  - ' + m);
} else console.log('dev-assets: every generated asset is in place');
