// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Running posts for the claims database: the interpreter (viv.wasm, what a
// reader's tab runs) and, when asked, the native engine (Viv -> C -> cc).
//
// The interpreter's output is taken from what feed-build already recorded
// (public/programs/posts/<file>.cells.json) when the staged source is the
// post as it is now; otherwise the post is run here, cell by cell the way
// feed-build and the notebook run it, and the result cached under
// www/.claims-cache/ by the hash of the source. The native run is cached the
// same way.
import { readFileSync, writeFileSync, existsSync, mkdirSync, mkdtempSync, rmSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join, dirname } from 'node:path';
import { tmpdir } from 'node:os';
import { spawnSync } from 'node:child_process';
import { expandUses } from '../../src/lib/uses.js';
import { splitNotebook, runUnits } from '../../src/lib/notebook.js';

const sha = (s) => createHash('sha256').update(s).digest('hex').slice(0, 16);

export function runner(WWW) {
  const ROOT = dirname(WWW);
  const SRC = join(ROOT, 'posts');
  const STAGE = join(WWW, 'public/programs/posts');
  const CACHE = join(WWW, '.claims-cache');
  const DIRS = { dft: 'dft', pde: 'pde', lib: 'lib', examples: 'lang/viv/examples' };
  const postFiles = Object.fromEntries(readdirSync(SRC).filter((f) => f.endsWith('.viv')).map((f) => [f.replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.viv$/, ''), f]));
  const resolve = async (name) => {
    if (name.startsWith('post:')) return readFileSync(join(SRC, postFiles[name.slice(5)]), 'utf8');
    const [head, ...rest] = name.split('/');
    const file = DIRS[head] && rest.length ? join(ROOT, DIRS[head], `${rest.join('/')}.viv`) : null;
    if (!file || !existsSync(file)) throw new Error(`module \`${name}\`: no module by that name`);
    return readFileSync(file, 'utf8');
  };
  let ex = null;
  async function wasm() {
    if (ex) return ex;
    const path = join(WWW, 'public/viv/viv.wasm');
    if (!existsSync(path)) throw new Error('www/public/viv/viv.wasm is missing: build it with www/build-wasm.sh');
    ex = (await WebAssembly.instantiate(readFileSync(path), {})).instance.exports;
    return ex;
  }
  const dec = (packed, e) => new TextDecoder().decode(new Uint8Array(e.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
  function callCell(e, prefix, cell) {
    const enc = new TextEncoder();
    const pb = enc.encode(prefix);
    const cb = enc.encode(cell);
    const ptr = e.viv_alloc(pb.length + cb.length);
    const mem = new Uint8Array(e.memory.buffer, ptr, pb.length + cb.length);
    mem.set(pb, 0);
    mem.set(cb, pb.length);
    return dec(e.viv_run_cell(ptr, pb.length + cb.length, pb.length), e);
  }

  /** { out, cells: { [cell index]: output }, from } for a post's source. */
  async function interp(file, src) {
    const staged = join(STAGE, file);
    const cellsFile = join(STAGE, file.replace(/\.viv$/, '.cells.json'));
    if (existsSync(staged) && existsSync(cellsFile) && readFileSync(staged, 'utf8') === src) {
      const cells = JSON.parse(readFileSync(cellsFile, 'utf8'));
      return { cells, out: Object.keys(cells).sort((a, b) => a - b).map((k) => cells[k]).join('\n'), from: 'feed-build' };
    }
    const key = sha(src);
    const cached = join(CACHE, 'interp', `${key}.json`);
    if (existsSync(cached)) return { ...JSON.parse(readFileSync(cached, 'utf8')), from: 'cache' };
    const e = await wasm();
    const cellsList = splitNotebook(src);
    const units = runUnits(cellsList);
    const cells = {};
    const seen = new Set();
    const texts = [];
    for (const u of units) texts.push(await expandUses(u.text, resolve, [], seen));
    texts.forEach((t, j) => { cells[units[j].index] = callCell(e, texts.slice(0, j).join('\n'), t); });
    const res = { cells, out: Object.keys(cells).sort((a, b) => a - b).map((k) => cells[k]).join('\n') };
    mkdirSync(join(CACHE, 'interp'), { recursive: true });
    writeFileSync(cached, JSON.stringify(res));
    return { ...res, from: 'run' };
  }

  /** The native engine's PASS/FAIL lines, or null when there is no compiler. */
  function native(file, src) {
    const viv = process.env.VIV || join(ROOT, 'target/release/viv');
    if (!existsSync(viv)) return null;
    const key = sha(src);
    const cached = join(CACHE, 'native', `${key}.json`);
    if (existsSync(cached)) return JSON.parse(readFileSync(cached, 'utf8'));
    const tmp = mkdtempSync(join(tmpdir(), 'claims-'));
    let res;
    try {
      const path = join(SRC, file);
      const c = join(tmp, 'p.c');
      const bin = join(tmp, 'p');
      const r1 = spawnSync(viv, [path, '-o', c], { encoding: 'utf8', timeout: 120000 });
      if (r1.status !== 0) res = { ok: false, error: `does not compile: ${(r1.stderr || '').split('\n')[0]}`, pass: [], fail: [] };
      else {
        const r2 = spawnSync(process.env.CC || 'cc', ['-O2', '-o', bin, c, '-lm'], { encoding: 'utf8', timeout: 300000 });
        if (r2.status !== 0) res = { ok: false, error: 'does not build (cc)', pass: [], fail: [] };
        else {
          const r3 = spawnSync(bin, [], { encoding: 'utf8', timeout: 300000, maxBuffer: 1 << 28 });
          const out = `${r3.stdout || ''}${r3.stderr || ''}`;
          const lines = out.split('\n');
          res = { ok: true, pass: lines.filter((l) => l.startsWith('PASS ')).map((l) => l.slice(5)), fail: lines.filter((l) => l.startsWith('FAIL ')).map((l) => l.slice(5)) };
        }
      }
    } finally { rmSync(tmp, { recursive: true, force: true }); }
    mkdirSync(join(CACHE, 'native'), { recursive: true });
    writeFileSync(cached, JSON.stringify(res));
    return res;
  }
  return { interp, native, postFiles };
}
