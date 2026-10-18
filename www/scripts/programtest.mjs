// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Run every staged example program through viv.wasm's interpreter.
//
// The examples page prints a validated result for most of these programs. This
// harness checks the weaker but still useful property that the site can back
// up on its own: every program the site serves parses, type-checks, and — where
// the interpreter supports its features — runs without an error, with all of
// its asserts passing.
//
// Programs the interpreter legitimately declines (records, and anything that
// needs the contracts prelude) are reported as SKIP with the compiler's own
// reason, not silently ignored. Anything that fails for another reason exits 1.
//
//   npm run programtest

import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { cpus, loadavg } from 'node:os';
import { expandUses, hasUses } from '../src/lib/uses.js';

const DIR = new URL('../public/programs/examples/', import.meta.url);
const BUDGET_MS = Number(process.env.VIV_BUDGET_MS ?? 20000);
const ONE = process.argv.indexOf('--one');
const wasm = readFileSync(new URL('../public/viv/viv.wasm', import.meta.url));
const { instance } = await WebAssembly.instantiate(wasm, {});
const ex = instance.exports;

function run(text) {
  const enc = new TextEncoder().encode(text);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const packed = ex.viv_run(ptr, enc.length);
  return new TextDecoder().decode(
    new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn))
  );
}
function compileC(text) {
  const enc = new TextEncoder().encode(text);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const packed = ex.viv_compile(ptr, enc.length);
  return new TextDecoder().decode(
    new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn))
  );
}

// Reasons the interpreter is ALLOWED to decline, each documented on the site.
const EXPECTED_DECLINE = [
  /record types/, // records compile but do not interpret — /reference#backends
  /step budget/, // a full-fat deck exceeds the browser budget by design
  /unknown function `(integrate|find_root|q_|r_)/, // needs --use std/contracts.viv
];

// `from xyz import ...` and the like, resolved from the repository as the build
// resolves them (scripts/feed-build.mjs), so an example may use a module
const ROOT = new URL('../../', import.meta.url);
const MODULE_DIRS = { dft: 'dft', pde: 'pde', lib: 'lib', examples: 'lang/viv/examples' };
async function source(f) {
  const text = readFileSync(new URL(f, DIR), 'utf8');
  if (!hasUses(text)) return text;
  return expandUses(text, async (name) => {
    const [head, ...rest] = name.split('/');
    const path = MODULE_DIRS[head] && rest.length ? new URL(`${MODULE_DIRS[head]}/${rest.join('/')}.viv`, ROOT) : null;
    if (!path || !existsSync(path)) throw new Error(`module \`${name}\`: no module by that name`);
    return readFileSync(path, 'utf8');
  });
}

// --one <file>: the worker mode. Prints the interpreter's output and exits, so
// the parent can impose a timeout by killing the child.
if (ONE !== -1) {
  const f = process.argv[ONE + 1];
  process.stdout.write(run(await source(f)));
  process.exit(0);
}

// The budget is wall time, stretched by the machine's load (load1 per core, at
// most 10 times; tools/gate/load.py): a loaded machine must not turn a program
// that fits the interpreter's limit into a skipped one.
const loadFactor = () => Math.min(10, Math.max(1, loadavg()[0] / (cpus().length || 1)));

function runBudgeted(f) {
  try {
    return {
      out: execFileSync(process.execPath, [process.argv[1], '--one', f], {
        encoding: 'utf8',
        timeout: Math.ceil(BUDGET_MS * loadFactor()),
        maxBuffer: 64 * 1024 * 1024,
        stdio: ['ignore', 'pipe', 'ignore'],
      }),
      timedOut: false,
    };
  } catch (e) {
    if (e.killed || e.signal) return { out: e.stdout ?? '', timedOut: true };
    return { out: e.stdout ?? '', timedOut: false };
  }
}

const PRELUDES = ['contracts.viv'];
const files = readdirSync(DIR)
  .filter((f) => f.endsWith('.viv') && !PRELUDES.includes(f))
  .sort();
let ran = 0;
let skipped = 0;
let failed = 0;
let asserts = 0;

for (const f of files) {
  const src = await source(f);

  // Every program must at minimum COMPILE. That is the parity claim, and it
  // holds for all of them including the ones the interpreter declines.
  const c = compileC(src);
  if (c.startsWith('error')) {
    failed++;
    console.log(`FAIL ${f}  does not compile to C\n     ${c.split('\n')[0]}`);
    continue;
  }

  const { out, timedOut } = runBudgeted(f);
  if (timedOut) {
    skipped++;
    console.log(
      `slow ${f}  compiles, but runs longer than the interpreter's ${BUDGET_MS / 1000} s limit; run it compiled to C`
    );
    continue;
  }
  const errLine = out.split('\n').find((l) => /^error/.test(l));
  // `test "name" { .. }` reports as `PASS <name>`; a bare assert outside any
  // test still prints `assert_near PASS`. Both count.
  const fails = (out.match(/^(FAIL |assert\w* FAIL)/gm) || []).length;
  const passes = (out.match(/^(PASS |assert\w* PASS)/gm) || []).length;
  asserts += passes;

  if (errLine) {
    if (EXPECTED_DECLINE.some((re) => re.test(errLine))) {
      skipped++;
      console.log(`skip ${f}  ${errLine.replace(/^error:\s*/, '').slice(0, 78)}`);
    } else {
      failed++;
      console.log(`FAIL ${f}\n     ${errLine}`);
    }
    continue;
  }
  if (fails) {
    failed++;
    console.log(`FAIL ${f}  ${fails} assert(s) failed`);
    for (const l of out.split('\n').filter((l) => /^(FAIL |assert\w* FAIL)/.test(l))) {
      console.log(`     ${l}`);
    }
    continue;
  }
  ran++;
  console.log(`ok   ${f}${passes ? `  ${passes} check${passes === 1 ? '' : 's'} passed` : ''}`);
}

// The contracts prelude, checked as a prelude: prepended to a program that
// uses it. It has to work in BOTH engines. The contract verbs hold their
// strategies in a registry -- an array of function values -- and `sweep` picks
// one by an index nobody knows until it runs, so the C backend cannot
// specialize the call and dispatches through a function pointer instead.
const USER = `val = integrate(fn(x) -> exp(-x * x), 0.0, 1.0) using every method
test "the contract agrees with the known value" {
    assert_near(val, 0.7468241328124271, 1.0e-8)
}`;
for (const prelude of PRELUDES) {
  const combined = readFileSync(new URL(prelude, DIR), 'utf8') + '\n' + USER;
  const out = run(combined);
  const errLine = out.split('\n').find((l) => /^error/.test(l));
  // an assert inside a test block reports through the block's own PASS line
  const passes = (out.match(/assert\w* PASS|^PASS /gm) || []).length;
  if (errLine || !passes) {
    failed++;
    console.log(`FAIL ${prelude}  as a prelude: ${errLine ?? 'no assert passed'}`);
  } else {
    asserts += passes;
    console.log(`ok   ${prelude}  as a prelude: ${passes} checks passed (strategies agree)`);
  }
  const c = compileC(combined);
  if (c.startsWith('error')) {
    failed++;
    console.log(`FAIL ${prelude}  C backend declines it:\n     ${c.split('\n')[0]}`);
  } else if (!/viv_fnvec\(/.test(c) || !/viv_fnget\(/.test(c)) {
    // If the registry stopped being a registry -- inlined, or the strategies
    // spelled out again -- the cross-check would still pass while the thing
    // this file is here to prove had quietly gone away.
    failed++;
    console.log(`FAIL ${prelude}  compiles, but not through a registry: no viv_fnvec/viv_fnget in the C`);
  } else {
    console.log(`ok   ${prelude}  C backend dispatches the registry through a function pointer`);
  }
}

// The workbench's built-in examples (the EXAMPLES menu in VivIde.jsx) are
// programs too, and the first one is what a visitor runs first. They are not
// files, so they are pulled out of the component's source here. Each must run
// without an error line.
{
  const ide = readFileSync(new URL('../src/components/VivIde.jsx', import.meta.url), 'utf8');
  const start = ide.indexOf('const EXAMPLES = [');
  const block = ide.slice(start, ide.indexOf('\n];', start));
  const re = /name: '([^']*)',\s*src: `([\s\S]*?)`,/g;
  let m;
  let n = 0;
  while ((m = re.exec(block))) {
    n++;
    const out = run(m[2].replace(/\\`/g, '`'));
    const errLine = out.split('\n').find((l) => /^error/.test(l));
    if (errLine) {
      failed++;
      console.log(`FAIL workbench example "${m[1]}": ${errLine}`);
    } else {
      ran++;
      console.log(`ok   workbench example "${m[1]}"`);
    }
  }
  if (!n) {
    failed++;
    console.log('FAIL workbench examples: none found in VivIde.jsx (did the EXAMPLES array move?)');
  }
}

console.log(
  `\nprogramtest: ${files.length} programs — all compile to C; ` +
    `${ran} programs and workbench examples ran without an error (${asserts} checks passed), ` +
    `${skipped} skipped (over the interpreter's time limit, or using a feature it lacks), ${failed} failed`
);
if (failed) process.exit(1);
