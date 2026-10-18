// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Regenerate the /internals page's artifacts by running the compiler.
//
// The internals page claims to show one program through every representation
// the compiler has. That claim is only worth making if the dumps are real, so
// this script produces them: it shells out to the `viv` binary for tokens,
// AST, IR, C99 and the wasm module, and to `cc` for the native run. The output
// is committed as src/data/trace.json, because the site build (Netlify) has
// neither Rust nor a C compiler.
//
// Run it after any change to the frontend or a backend:  npm run trace

import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, readFileSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);

// Small enough to trace by eye; big enough to exercise a definition, a call,
// a counted loop, mutation of a global, and output.
const PROGRAM = `def sq(x) {
    return x * x
}

total = 0.0
for i to 10 {
    total = total + sq(i)
}
show total`;

const tmp = mkdtempSync(join(tmpdir(), 'viv-trace-'));
const src = join(tmp, 'sumsq.viv');
writeFileSync(src, PROGRAM + '\n');

const viv = (...args) =>
  execFileSync('cargo', ['run', '-q', '-p', 'viv', '--', ...args], {
    cwd: ROOT,
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
  });

console.log('building the compiler…');
execFileSync('cargo', ['build', '-q', '--release', '-p', 'viv'], { cwd: ROOT, stdio: 'inherit' });

const tokens = viv(src, '--emit', 'tokens').trim();
const ast = viv(src, '--emit', 'ast').trim();
const ir = viv(src, '--emit', 'ir').trim();

const cPath = join(tmp, 'sumsq.c');
const cNote = viv(src, '-o', cPath).trim();
const c = readFileSync(cPath, 'utf8').trim();

const wasmPath = join(tmp, 'sumsq.wasm');
const wasmNote = viv(src, '-o', wasmPath).trim();
const wasmSize = statSync(wasmPath).size;
const wasmBuf = readFileSync(wasmPath);
const wasmHex = [...wasmBuf].map((b) => b.toString(16).padStart(2, '0')).join(' ');

// Walk the module's section headers so the page's byte budget is measured, not
// asserted. A wasm section is one id byte + a LEB128 payload length.
const SECTION_NAMES = {
  0: 'custom', 1: 'type', 2: 'import', 3: 'function', 4: 'table', 5: 'memory',
  6: 'global', 7: 'export', 8: 'start', 9: 'element', 10: 'code', 11: 'data',
  12: 'data count',
};
function wasmSections(buf) {
  const out = [{ name: 'header', bytes: 8, note: '\\0asm magic + version 1' }];
  let p = 8;
  while (p < buf.length) {
    const id = buf[p];
    let n = 0, shift = 0, q = p + 1;
    for (;;) {
      const b = buf[q++];
      n |= (b & 0x7f) << shift;
      shift += 7;
      if (!(b & 0x80)) break;
    }
    const header = q - p;
    out.push({
      name: SECTION_NAMES[id] ?? `id ${id}`,
      bytes: header + n,
      count: n > 0 ? buf[q] : 0,
    });
    p = q + n;
  }
  return out;
}
const sections = wasmSections(wasmBuf);

// Compile and run the C, so the "285" on the page is a number this machine
// actually printed rather than a number someone typed.
let native = null;
try {
  const bin = join(tmp, 'sumsq');
  execFileSync('cc', [cPath, '-O2', '-lm', '-o', bin], { stdio: 'pipe' });
  native = execFileSync(bin, { encoding: 'utf8' }).trim();
} catch (e) {
  native = `(no C compiler available: ${e.message.split('\n')[0]})`;
}

// And the interpreter, for the three-way agreement.
const interp = viv('run', '--quiet', src).trim();

// Run the wasm module the direct backend emitted, against the runtime in
// viv.wasm (the imports it asks for), to show the third executor reaching the
// same number.
let wasmRun = null;
try {
  const { loadViv, runDirect } = await import('../src/lib/vivDirect.js');
  const ex = await loadViv(readFileSync(join(WWW, 'public/viv/viv.wasm')));
  const r = await runDirect(ex, readFileSync(wasmPath));
  wasmRun = r.error ? `error: ${r.error}` : r.output.trim();
} catch (e) {
  wasmRun = `(could not instantiate: ${e.message.split('\n')[0]})`;
}

// The annotations design page shows `viv doc` / `whatis` / `lsfunc` / `src`
// output on a real annotated program, so capture it here rather than pasting
// it into the page. lang/viv/examples/annotated.viv compiles and runs.
const ANN = join(ROOT, 'lang/viv/examples/annotated.viv');
const annotations = {
  source: readFileSync(ANN, 'utf8').trim(),
  doc: viv('doc', ANN).trim(),
  card: viv('doc', ANN, 'prim_overlap').trim(),
  whatis: viv('whatis', ANN, 'Nb').trim(),
  lsfunc: viv('lsfunc', ANN).trim(),
  lsvar: viv('lsvar', ANN).trim(),
  srcListing: viv('src', ANN, 'prim_overlap').trim(),
  run: viv('run', '--quiet', ANN).trim(),
};

// The version-control design page shows `viv hash` and `viv diff` on a real
// pair of files: v2 renames a def (body untouched), reformats another and adds
// a comment to it, and changes the top-level code. Written here, hashed here.
const V1 = `def sq(x) {
    return x * x
}

def cube(x) {
    return x * x * x
}

total = 0.0
for i to 10 {
    total = total + sq(i)
}
show total`;
const V2 = `# renamed sq -> square; nothing else about it changed
def square(x) {
    return x * x
}

def cube(x) {
    # a comment, and different whitespace below


    return x  *  x  *  x
}

total = 0.0
for i to 10 {
    total = total + square(i)
}
show total`;
const v1p = join(tmp, 'v1.viv');
const v2p = join(tmp, 'v2.viv');
writeFileSync(v1p, V1 + '\n');
writeFileSync(v2p, V2 + '\n');
const vcs = {
  v1: V1,
  v2: V2,
  hash1: viv('hash', v1p).trim(),
  hash2: viv('hash', v2p).trim(),
  diff: viv('diff', v1p, v2p).trim(),
};

// The CLI echoes the absolute path it wrote to, and `tmp` is under the user's
// home. This output is SERVED, so scrub the temp directory to a neutral name
// before it is committed — the same discipline build-wasm.sh applies with
// --remap-path-prefix.
const scrub = (s) => s.replaceAll(tmp + '/', '').replaceAll(tmp, '.');

const out = {
  program: PROGRAM,
  tokens,
  tokenCount: tokens.split(/\s+/).filter(Boolean).length,
  ast,
  astLines: ast.split('\n').length,
  ir,
  irNodes: new Set(ir.match(/^\s*([0-9a-f]{8})/gm)?.map((s) => s.trim())).size,
  c,
  cLines: c.split('\n').length,
  cNote: scrub(cNote),
  native: scrub(native),
  interp,
  wasmNote: scrub(wasmNote),
  wasmSize,
  wasmHex,
  sections,
  annotations,
  vcs,
  wasmRun,
  generated: new Date().toISOString().slice(0, 10),
};
const json = JSON.stringify(out, null, 2);
if (json.includes(process.env.HOME) || json.includes('/var/folders') || json.includes(tmp)) {
  console.error('FAIL: a local path leaked into trace.json');
  process.exit(1);
}
writeFileSync(join(WWW, 'src/data/trace.json'), json);

console.log(`tokens: ${out.tokenCount}`);
console.log(`ast:    ${out.astLines} lines`);
console.log(`ir:     ${out.irNodes} unique nodes`);
console.log(`c:      ${out.cLines} lines  -> native says: ${native}`);
console.log(`wasm:   ${wasmSize} bytes    -> wasm says:   ${wasmRun}`);
console.log(`interp:                       -> interp says: ${interp}`);
