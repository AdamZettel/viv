#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Race one program four ways: the interpreter compiled to wasm (the Run
# button), the direct wasm backend, Viv -> C -> emcc -O2 -> wasm, and Viv ->
# C -> native (`viv build`, the exact build). Every way is timed around the program's run alone --
# no process start, no module load or compile -- and the best of RUNS (default
# 3) is reported, with what each printed checked against the others.
#
#     tools/race-engines.sh bench/jacobi_stencil.viv bench/matmul_chain.viv
#
# Needs node, a viv.wasm with the runtime exports (www/build-wasm.sh), a C
# compiler, and emcc (skipped when absent). The wasm ways run in node, which
# is V8, as Chrome does. Build modes match tools/both-engines.sh: a debug
# build everywhere (contracts sweep, the debug lens records).
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
viv="${VIV:-$root/target/release/viv}"
runs=${RUNS:-3}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
EMCC=$(command -v emcc || ls "$HOME"/emsdk/upstream/emscripten/emcc 2>/dev/null | head -1 || true)

printf '%-26s %12s %12s %12s %12s   %s\n' program interp-wasm direct-wasm emcc-O2 native agree
for src in "$@"; do
    name=$(basename "$src" .viv)
    "$viv" "$src" -o "$tmp/$name.wasm" > /dev/null
    "$viv" "$src" -o "$tmp/$name.c" > /dev/null
    "$viv" build "$src" -o "$tmp/$name" > /dev/null
    if [ -n "$EMCC" ]; then
        # main is called by hand, so the timer brackets it and nothing else
        "$EMCC" -O2 "$tmp/$name.c" -o "$tmp/$name.emcc.mjs" -sMODULARIZE=1 -sEXPORT_ES6=1 -sINVOKE_RUN=0 \
            -sEXIT_RUNTIME=0 -sALLOW_MEMORY_GROWTH=1 -sENVIRONMENT=node -sEXPORTED_RUNTIME_METHODS=callMain 2> /dev/null
    fi
    ROOT="$root" TMP="$tmp" NAME="$name" SRC="$src" RUNS="$runs" HAVE_EMCC="${EMCC:+1}" node --input-type=module - <<'JS'
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { performance } from 'node:perf_hooks';
const { ROOT, TMP, NAME, SRC, RUNS, HAVE_EMCC } = process.env;
const { loadViv, runDirect } = await import(`${ROOT}/www/src/lib/vivDirect.js`);
const { expandUses } = await import(`${ROOT}/www/src/lib/uses.js`);
const runs = Number(RUNS);
const best = async (f) => { let t = Infinity, out = ''; for (let i = 0; i < runs; i++) { const r = await f(); t = Math.min(t, r.t); out = r.out; } return { t, out }; };
const clean = (s) => s.replace(/^__(graph|trace)__.*\n/gm, '');
const vivBytes = readFileSync(process.env.VIV_WASM || `${ROOT}/www/public/viv/viv.wasm`);

// 1. the interpreter in viv.wasm, on the source with its modules expanded
const DIRS = { dft: 'dft', pde: 'pde', lib: 'lib', examples: 'lang/viv/examples' };
const text = await expandUses(readFileSync(SRC, 'utf8'), async (m) => {
  const [head, ...rest] = m.split('/');
  return readFileSync(`${ROOT}/${DIRS[head]}/${rest.join('/')}.viv`, 'utf8');
});
const interp = await best(async () => {
  const ex = await loadViv(vivBytes);
  const enc = new TextEncoder().encode(text);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const t0 = performance.now();
  const packed = ex.viv_run(ptr, enc.length);
  const t = (performance.now() - t0) / 1000;
  return { t, out: clean(new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)))) };
});
// 2. the direct backend's module
const modBytes = readFileSync(`${TMP}/${NAME}.wasm`);
const direct = await best(async () => {
  const ex = await loadViv(vivBytes);
  const r = await runDirect(ex, modBytes);
  return { t: r.seconds, out: r.error ? `error: ${r.error}` : r.output };
});
// 3. emcc -O2
let emcc = null;
if (HAVE_EMCC) {
  const factory = (await import(`${TMP}/${NAME}.emcc.mjs`)).default;
  emcc = await best(async () => {
    let out = '';
    const m = await factory({ print: (s) => { out += s + '\n'; }, printErr: () => {} });
    const t0 = performance.now();
    m.callMain([]);
    return { t: (performance.now() - t0) / 1000, out };
  });
}
// 4. native, timed from here around the process (its start is ~1 ms)
const native = await best(async () => {
  const t0 = performance.now();
  let out;
  try { out = execFileSync(`${TMP}/${NAME}`, { encoding: 'utf8' }); } catch (e) { out = e.stdout; }
  return { t: (performance.now() - t0) / 1000, out };
});
const strip = (s) => s.split('\n').filter((l) => !/assertions? FAILED$/.test(l)).join('\n');
const agree = [direct, emcc, native].filter(Boolean).map((r) => (strip(r.out) === strip(interp.out) ? 'same' : 'DIFFERS'));
const f = (r) => (r ? `${r.t.toFixed(3)} s` : '-');
console.log(`${NAME.padEnd(26)} ${f(interp).padStart(12)} ${f(direct).padStart(12)} ${f(emcc).padStart(12)} ${f(native).padStart(12)}   ${agree.join(' ')}`);
JS
done
