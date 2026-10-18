#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Compile an engine to one WebAssembly module that runs ANY input deck.
//
//     node engines/build-wasm.mjs            build engines/wasm/<engine>.wasm and the manifest (needs emcc)
//     node engines/build-wasm.mjs --check    is the committed module built from today's sources?
//
// dft/build-wasm.sh compiles one module per material, the deck baked in. An
// engine takes the reader's crystal, so its module must read the deck when it
// runs. The program (engines/dft-bands.viv) is compiled with silicon's deck in
// place, and the generated C gets one addition: right after the deck's last
// statement (the marker `DECK_LOADED = 1.0`), a call that reads the reader's
// deck from standard input and overwrites every deck variable by name. With
// nothing on standard input the module computes silicon.
//
// Why not the language's own checkpoint(), which loads globals by name the
// same way? It snapshots every global, and this program has complex arrays,
// which checkpoints do not support yet (lang/viv/src/lower.rs). The loader
// below reads the same byte layout, restricted to the deck's variables, all of
// which are numbers or real arrays.
//
// Like dft/wasm/, the module is committed, because the site's host has no
// emscripten; the manifest records the SHA-256 of the C it came from, and
// --check regenerates that C and compares.
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, writeFileSync, existsSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir, homedir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = dirname(HERE);
const VIV = join(ROOT, 'target/release/viv');
const OUT = join(HERE, 'wasm');
const MANIFEST = join(OUT, 'manifest.json');

// the engines that run compiled, and the deck variables each loads
const ENGINES = {
  'dft-bands': ['NSPEC', 'ZION', 'RLOC', 'CLOC', 'NPRJ', 'RPRJ', 'HPRJ', 'ALAT', 'A', 'NAT', 'ATSPEC', 'ATFRAC',
    'ECUT', 'NBANDS', 'SMEAR', 'NKMESH', 'MIX', 'SCF_TOL', 'MAXIT', 'NITER', 'NITER_BANDS', 'EIG_TOL', 'NCORNER', 'CORNER', 'NSEG',
    'HAS_REF', 'REF_ETOT', 'REF_NB', 'REFPATH'],
};
const MARKER = 'DECK_LOADED = 1.0;';

/** The loader, in C, for these variables. `types` maps a name to 'double' or 'viv_arr'. */
function loader(names, types) {
  const cases = names.map((n) => (types[n] === 'double'
    ? `        else if (strcmp(nm, "${n}") == 0) { if (tag != 0) viv_deck_die("${n} is not a number"); viv_deck_take(&${n}, 8); }`
    : `        else if (strcmp(nm, "${n}") == 0) { if (tag != 1) viv_deck_die("${n} is not an array"); viv_deck_arr(&${n}); }`)).join('\n');
  return `
/* ---- added by engines/build-wasm.mjs: the reader's deck, from standard input ----
   Layout (the compiler's checkpoint format): "VCK2", then per variable a u32
   name length, the name, a tag byte, and tag 0: one f64, or tag 1: i32 rank,
   i32 dims[8], i32 size, size f64s. Empty input: keep the compiled-in deck. */
static unsigned char* viv_deck_buf = 0;
static size_t viv_deck_len = 0, viv_deck_at = 0;
static void viv_deck_die(const char* m) { fprintf(stderr, "deck: %s\\n", m); printf("error: deck: %s\\n", m); exit(2); }
static void viv_deck_take(void* dst, size_t n) {
    if (viv_deck_at + n > viv_deck_len) viv_deck_die("the deck ends early");
    memcpy(dst, viv_deck_buf + viv_deck_at, n);
    viv_deck_at += n;
}
static void viv_deck_arr(viv_arr* a) {
    viv_deck_take(&a->rank, 4);
    viv_deck_take(a->dims, 32);
    viv_deck_take(&a->size, 4);
    if (a->size < 0) viv_deck_die("negative array size");
    a->data = (double*)malloc((size_t)(a->size > 0 ? a->size : 1) * 8);
    if (!a->data) viv_deck_die("out of memory");
    viv_deck_take(a->data, (size_t)a->size * 8);
}
static void viv_engine_load_deck(void) {
    size_t cap = 1 << 16;
    viv_deck_buf = (unsigned char*)malloc(cap);
    size_t n;
    while ((n = fread(viv_deck_buf + viv_deck_len, 1, cap - viv_deck_len, stdin)) > 0) {
        viv_deck_len += n;
        if (viv_deck_len == cap) { cap *= 2; viv_deck_buf = (unsigned char*)realloc(viv_deck_buf, cap); }
    }
    if (viv_deck_len == 0) return;
    if (viv_deck_len < 4 || memcmp(viv_deck_buf, "VCK2", 4) != 0) viv_deck_die("not a deck (no VCK2 header)");
    viv_deck_at = 4;
    while (viv_deck_at < viv_deck_len) {
        unsigned int l; char nm[64]; unsigned char tag;
        viv_deck_take(&l, 4);
        if (l >= sizeof nm) viv_deck_die("a variable name is too long");
        viv_deck_take(nm, l); nm[l] = 0;
        viv_deck_take(&tag, 1);
        if (0) {}
${cases}
        else viv_deck_die("a variable this engine does not read");
    }
    printf("DECK: loaded %zu bytes\\n", viv_deck_len);
}
`;
}

function genC(engine, tmp) {
  const cfile = join(tmp, `${engine}.c`);
  execFileSync(VIV, [join(HERE, `${engine}.viv`), '-o', cfile], { cwd: ROOT, stdio: ['ignore', 'ignore', 'inherit'] });
  let c = readFileSync(cfile, 'utf8');
  // the manifest hashes this C, so it must be the same on every machine: no
  // path of this checkout in it (the compiler names a program by its file
  // name in the error locations it writes)
  if (c.includes(ROOT)) throw new Error(`${engine}: the generated C names a local path; its hash would depend on the checkout`);
  const names = ENGINES[engine];
  const types = {};
  for (const n of names) {
    const m = c.match(new RegExp(`^(double|viv_arr) ${n};$`, 'm'));
    if (!m) throw new Error(`${engine}: the generated C declares no global ${n}`);
    types[n] = m[1];
  }
  const hits = c.split(MARKER).length - 1;
  if (hits !== 1) throw new Error(`${engine}: expected the marker "${MARKER}" once in the generated C, found it ${hits} times`);
  c = c.replace(MARKER, `${MARKER} viv_engine_load_deck();`);
  c = c.replace(/\nint main\(void\) \{/, `${loader(names, types)}\nint main(void) {`);
  if (!/#include <string.h>/.test(c)) c = `#include <string.h>\n${c}`;
  writeFileSync(cfile, c);
  return cfile;
}
const sha = (f) => createHash('sha256').update(readFileSync(f)).digest('hex');

const tmp = mkdtempSync(join(tmpdir(), 'viv-engines-'));
try {
  const check = process.argv.includes('--check');
  const manifest = existsSync(MANIFEST) ? JSON.parse(readFileSync(MANIFEST, 'utf8')) : { modules: {} };
  let status = 0;
  const built = {};
  for (const engine of Object.keys(ENGINES)) {
    const cfile = genC(engine, tmp);
    const hash = sha(cfile);
    const wasm = join(OUT, `${engine}.wasm`);
    if (check) {
      const ok = manifest.modules[engine] && manifest.modules[engine].c_sha256 === hash && existsSync(wasm);
      console.log(ok ? `ok    ${engine}.wasm is built from the current sources` : `STALE ${engine}.wasm -- the C it would be compiled from has changed; run node engines/build-wasm.mjs`);
      if (!ok) status = 1;
      continue;
    }
    mkdirSync(OUT, { recursive: true });
    // -DVIV_RELEASE: a served module is a release build, without the
    // compiled-in array bounds checks and their table of source lines
    execFileSync('emcc', ['-O2', '-DVIV_RELEASE', cfile, '-o', wasm, '-sSTANDALONE_WASM=1', '-sALLOW_MEMORY_GROWTH=1', '-sINITIAL_MEMORY=33554432', '-sSTACK_SIZE=1048576',
      `-ffile-prefix-map=${tmp}=.`], { stdio: 'inherit' });
    const bytes = readFileSync(wasm);
    if (bytes.includes(Buffer.from(homedir()))) throw new Error(`${engine}.wasm: a local path leaked into the module`);
    built[engine] = { c_sha256: hash, bytes: bytes.length };
    console.log(`built ${engine}.wasm (${bytes.length} bytes)`);
  }
  if (!check) {
    const emcc = execFileSync('emcc', ['--version']).toString().split('\n')[0].replace(/.*\) ([0-9.]+).*/, '$1');
    writeFileSync(MANIFEST, `${JSON.stringify({ _: 'written by engines/build-wasm.mjs; do not edit', emcc, flags: '-O2 -DVIV_RELEASE -sSTANDALONE_WASM=1 -sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=1048576', modules: built }, null, 2)}\n`);
  }
  process.exitCode = status;
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
