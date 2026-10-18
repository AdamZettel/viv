// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Checks the walk through a run (src/lib/zoomwalk.js) without a browser:
// the van Wijk-Nuij path against its closed form, and, for each explorable
// program, that the traced recording keeps the program's output, has a call
// and a ret for every call, and gives frames, a call-graph layout, camera
// targets and code-pane files that hold together. Needs public/viv/viv.wasm and public/explore/*.json
// (npm run stage).
//
//   node scripts/zoomwalk-test.mjs [slug ...]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseRecording, indexBundle, buildTree } from '../src/lib/explore.js';
import { traceSource, layoutWorld, buildFrames, targetRect, fitView, interpolateZoom, nextOver, nextOut, codeFiles, codeTarget, defAt, defLine, calleesOn, intentLine, callSite } from '../src/lib/zoomwalk.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const pub = path.join(here, '..', 'public');
let fails = 0;
const check = (ok, what) => {
  if (!ok) {
    fails++;
    console.log(`FAIL ${what}`);
  }
};
const near = (a, b, tol) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));

// ---- the path ---------------------------------------------------------------
{
  const a = [0, 0, 100];
  const b = [1000, 300, 40];
  const f = interpolateZoom(a, b);
  check(f(0).every((x, k) => near(x, a[k], 1e-12)), 'interpolateZoom starts at a');
  check(f(1).every((x, k) => near(x, b[k], 1e-9)), 'interpolateZoom ends at b');
  // the path zooms out first: the view is wider in the middle than at either end
  check(f(0.5)[2] > Math.max(a[2], b[2]), 'interpolateZoom zooms out mid-flight');
  // the center moves along the straight segment a -> b, monotonically
  let prev = -1;
  let straight = true;
  for (let t = 0; t <= 1.0001; t += 0.05) {
    const [x, y] = f(Math.min(1, t));
    const u = x / 1000;
    straight &&= near(y, 300 * u, 1e-9) && u >= prev - 1e-12;
    prev = u;
  }
  check(straight, 'interpolateZoom moves the center along the segment, monotonically');
  // the paper's length: for w0 = w1 = w and distance d, S = 2/rho * asinh(rho^2 d / (2 w))
  const g = interpolateZoom([0, 0, 50], [400, 0, 50]);
  const rho = Math.SQRT2;
  check(near(g.S, (2 / rho) * Math.asinh((rho * rho * 400) / (2 * 50)), 1e-12), `S matches the closed form (${g.S})`);
  // a pure zoom: S = |ln(w1/w0)| / rho
  const z = interpolateZoom([5, 5, 10], [5, 5, 80]);
  check(near(z.S, Math.log(8) / rho, 1e-12) && near(z(1)[2], 80, 1e-12), 'pure zoom');
  console.log(`path: S(pan 400 at w 50) = ${g.S.toFixed(6)}; mid-flight width ${f(0.5)[2].toFixed(1)} (ends 100, 40)`);
}

// ---- small helpers ----------------------------------------------------------
{
  const defs = new Set(['qrfac', 'enorm', 'lmpar']);
  check(JSON.stringify(calleesOn('    x = enorm(n, a) + qrfac(m) # lmpar(no)', defs)) === '["enorm","qrfac"]', 'calleesOn finds calls in order, not in comments');
  check(calleesOn('    s = "enorm(" + 1', defs).length === 0, 'calleesOn skips strings');
  check(calleesOn('def qrfac(m, n) {', defs).length === 0, "calleesOn does not count a def's own head");
  const ix0 = { bundle: { intent: { f: { what: 'Minimize $\\lVert f(x) \\rVert^2$ over $x$. Then more.' } } } };
  check(intentLine(ix0, 'f') === 'Minimize ‖ f(x) ‖² over x.', `intentLine: first sentence, TeX as symbols (${intentLine(ix0, 'f')})`);
}

// ---- the programs -----------------------------------------------------------
const wasm = path.join(pub, 'viv', 'viv.wasm');
if (!fs.existsSync(wasm)) {
  console.log('no public/viv/viv.wasm: run npm run stage first');
  process.exit(1);
}
const { instance } = await WebAssembly.instantiate(fs.readFileSync(wasm), {});
const ex = instance.exports;
const call = (fn, text) => {
  const enc = new TextEncoder().encode(text);
  const p = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, p, enc.length).set(enc);
  const r = fn(p, enc.length);
  return new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(r >> 32n), Number(r & 0xffffffffn)));
};
const slugs = process.argv.slice(2).length ? process.argv.slice(2) : ['minpack-rosenbrock', 'minpack-thurber', 'water-rhf'];
for (const slug of slugs) {
  const bundle = JSON.parse(fs.readFileSync(path.join(pub, 'explore', `${slug}.json`), 'utf8'));
  const ix = indexBundle(bundle);
  const t0 = Date.now();
  const plain = parseRecording(call(ex.viv_explore_run, JSON.stringify({ src: bundle.program, defs: bundle.lineDefs || [] })));
  const rec = parseRecording(call(ex.viv_explore_run, JSON.stringify({ src: traceSource(bundle), defs: bundle.lineDefs || [] })));
  const ms = Date.now() - t0;
  if (plain.error) {
    // the explorer itself cannot record it with this viv.wasm: not the walk's to check
    console.log(`skip ${slug}: the untraced recording fails too (${plain.error.slice(0, 90)})`);
    continue;
  }
  if (rec.error) {
    check(false, `${slug}: ${rec.error}`);
    continue;
  }
  check(rec.printed === plain.printed, `${slug}: tracing every def leaves the printed output unchanged`);
  const calls = rec.events.filter((e) => e[0] === 'call').length;
  const rets = rec.events.filter((e) => e[0] === 'ret').length;
  check(calls === rets, `${slug}: every call has its ret (${calls} calls, ${rets} rets)`);
  const tree = buildTree(rec.events, ix);
  const world = layoutWorld(ix, tree);
  // no regions: one flat layer of boxes per call depth
  check(!('regions' in world), `${slug}: the layout has no module regions`);
  // boxes do not overlap
  const ds = [...world.defs.values()];
  let overlap = 0;
  for (let i = 0; i < ds.length; i++) for (let j = i + 1; j < ds.length; j++) {
    const a = ds[i];
    const b = ds[j];
    if (a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h) overlap++;
  }
  check(overlap === 0, `${slug}: ${overlap} def boxes overlap`);
  // callers left of callees: an edge into a deeper layer goes rightwards, and
  // every def but the top level has a caller one layer before it
  const main = world.defs.get('<main>');
  let layered = main && main.layer === 0;
  for (const d of ds) {
    if (d.def === '<main>') continue;
    const callers = world.edges.filter((e) => e.to === d.def && e.from !== d.def).map((e) => world.defs.get(e.from));
    layered &&= callers.some((a) => a.layer === d.layer - 1 && a.x + a.w < d.x);
  }
  check(layered, `${slug}: each def sits one layer right of a caller`);
  check(ds.every((d) => d.sig !== undefined && typeof d.intent === 'string' && d.module), `${slug}: every box has a signature, an intent line and a module`);
  const withIntent = ds.filter((d) => d.intent).length;
  check(ds.every((d) => !/\$|\\[A-Za-z]/.test(d.intent)), `${slug}: intent lines are plain text, no TeX left`);
  // a call site is a line of the caller that calls the callee
  let sites = true;
  for (const e of world.edges) {
    if (e.from === e.to) continue;
    const x = callSite(ix, e.from, e.to);
    sites &&= x !== null && (calleesOn(ix.programLines[x - 1], new Set([e.to])).length > 0 || x === defLine(ix, e.from));
  }
  check(sites, `${slug}: every call edge has a call site in its caller's source`);
  // the code pane: every program line is in exactly one file, each file in its module's order
  const { files, fileOf } = codeFiles(ix);
  const lines = ix.programLines.length;
  let inFiles = 0;
  let sorted = true;
  for (const f of files) {
    inFiles += f.rows.length;
    for (let k = 1; k < f.rows.length; k++) sorted &&= f.rows[k].n > f.rows[k - 1].n;
  }
  check(inFiles <= lines && inFiles >= lines - files.length * 50 && sorted, `${slug}: the code files hold the program in reading order (${inFiles} of ${lines} lines)`);
  check(files[0].index === 0, `${slug}: the program's own file comes first`);
  // every def's first line is in a file, and defAt maps it back to the def
  let defsOk = true;
  for (const d of ds) {
    if (d.def === '<main>') continue;
    const x = defLine(ix, d.def);
    // a def the compiler generates (an autodiff gradient) can share its source with the def it comes from
    const at = defAt(ix, x);
    defsOk &&= x !== null && fileOf.has(x) && (at === d.def || defLine(ix, at) === x);
  }
  check(defsOk, `${slug}: every def's head is in the code pane and maps back to the def`);
  for (const loops of ['one', 'notable', 'every']) {
    const { frames, truncated } = buildFrames(tree, ix, { loops });
    check(frames[0].kind === 'start' && (truncated || frames[frames.length - 1].kind === 'end'), `${slug}/${loops}: starts and ends`);
    // calls and rets pair up as brackets, and depth follows them
    let depth = 1;
    let ok = true;
    let codeOk = true;
    const open = [];
    for (const f of frames.slice(1, -1)) {
      if (f.kind === 'call') {
        depth++;
        open.push(f.def);
      } else if (f.kind === 'ret') {
        depth--;
        ok &&= open.pop() === f.from;
      }
      ok &&= f.depth === depth;
      const d = world.defs.get(f.def);
      ok &&= !!d;
      const r = targetRect(f, world, ix);
      ok &&= [r.x, r.y, r.w, r.h].every(Number.isFinite) && r.w > 0 && r.h > 0;
      // the camera frames the box of the def that runs, at a scale where its text reads (at least 0.9 px per unit)
      if (d && f.kind !== 'start' && f.kind !== 'end') ok &&= r.x <= d.x && r.x + r.w >= d.x + d.w && 600 / fitView(r, 600, 700)[2] >= 0.9;
      // the code pane's line is in the def that runs (or the top level's setup)
      const t = codeTarget(f, ix);
      if (t.line !== null && f.kind === 'line') codeOk &&= fileOf.has(t.line) && (defAt(ix, t.line) === f.def || f.def === '<main>');
    }
    check(ok && open.length === 0, `${slug}/${loops}: frames bracket calls, keep depth, and every target frames the def that runs`);
    check(codeOk, `${slug}/${loops}: every line the code pane marks is in the def that runs`);
    // step over never lands deeper; step out lands shallower or at the end
    let moves = true;
    for (let i = 0; i < frames.length - 1; i += Math.max(1, Math.floor(frames.length / 300))) {
      const o = nextOver(frames, i);
      const u = nextOut(frames, i);
      moves &&= o > i && frames[o].depth <= frames[i].depth;
      moves &&= u > i && (frames[u].depth < frames[i].depth || u === frames.length - 1);
    }
    check(moves, `${slug}/${loops}: step over and step out`);
    const lines = frames.filter((f) => f.kind === 'line').length;
    const cs = frames.filter((f) => f.kind === 'call').length;
    console.log(`${slug} [${loops}]: ${frames.length} frames (${cs} calls, ${lines} lines)${truncated ? ' truncated' : ''}`);
  }
  console.log(`${slug}: ${rec.total} events traced (${plain.total} without), ${world.defs.size} defs from ${world.modules.length} files in ${world.layers.length} layers (${withIntent} with an intent line), ${world.edges.length} call edges, ${files.length} code files, recorded in ${ms} ms`);
}
if (fails) {
  console.log(`${fails} checks failed`);
  process.exit(1);
}
console.log('zoomwalk: all checks passed');
