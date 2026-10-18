// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// run-wasm -- run a module from the direct wasm backend, the way a tab does.
//
//     target/release/viv prog.viv -o prog.wasm
//     node tools/run-wasm.mjs prog.wasm
//     node tools/run-wasm.mjs --time prog.wasm     (seconds on stderr)
//     node tools/run-wasm.mjs --profile prog.wasm  (the builtins it called through
//                                                  the runtime's bridge, on stderr)
//     node tools/run-wasm.mjs --memory prog.wasm   (the size of the wasm memory
//                                                  after the run, in bytes, on stderr)
//     node tools/run-wasm.mjs --interp prog.viv    (the interpreter in viv.wasm:
//                                                  the Run button's engine)
//     node tools/run-wasm.mjs --file=data.xyz=path/to/it.xyz prog.wasm
//                                                 (a file the program's read_text
//                                                  finds as data.xyz: in a browser
//                                                  a program reads only the files
//                                                  attached to it)
//
// The module imports its runtime from viv.wasm (www/public/viv/viv.wasm, or
// VIV_WASM=<path>), which must be a build with the `vrt_*` exports
// (www/build-wasm.sh). What it prints, and how it ends, follow `viv run`:
// stdout is the program's output; an error prints `error: <message>` to
// stderr instead of the output, and exits 1; failed assertions and holds are
// counted on stderr and exit 1. So `viv run` and this can be diffed directly
// (tools/three-engines.sh does, as engine 3).
//
// The exit status is set, never forced with process.exit: on macOS node
// writes to a pipe asynchronously, and exiting at once cuts a long output
// short (it did: 75 lines of the molecular-dynamics post's 124).
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadViv, runDirect } from '../www/src/lib/vivDirect.js';
import { expandUses } from '../www/src/lib/uses.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const args = process.argv.slice(2);
const time = args.includes('--time');
const file = args.find((a) => !a.startsWith('--'));
if (!file) {
  console.error('usage: node tools/run-wasm.mjs [--time] [--profile] <program.wasm> | --interp <program.viv>');
  process.exitCode = 2;
} else {
  const viv = await loadViv(readFileSync(process.env.VIV_WASM || join(ROOT, 'www/public/viv/viv.wasm')));
  for (const a of args.filter((x) => x.startsWith('--file='))) {
    const [name, path] = a.slice('--file='.length).split('=');
    putFile(viv, name, readFileSync(path ?? name));
  }
  process.exitCode = args.includes('--interp') ? await interp(viv) : await direct(viv);
}

/** A file in viv.wasm's table of files, for read_text. */
function putFile(viv, name, bytes) {
  if (!viv.vrt_file_put) throw new Error('this viv.wasm has no file table; rebuild it with www/build-wasm.sh');
  const put = (b) => {
    const p = viv.viv_alloc(b.length);
    new Uint8Array(viv.memory.buffer, p, b.length).set(b);
    return p;
  };
  const nb = new TextEncoder().encode(name);
  const np = put(nb);
  const dp = put(bytes);
  viv.vrt_file_put(np, nb.length, dp, bytes.length);
}

/** The module in `file`, against viv.wasm's runtime. */
async function direct(viv) {
  if (!viv.vrt_init) {
    console.error('run-wasm: this viv.wasm has no runtime exports; rebuild it with www/build-wasm.sh');
    return 2;
  }
  const r = await runDirect(viv, readFileSync(file));
  if (args.includes('--profile')) {
    // what went through the bridge to the interpreter's builtins
    const packed = viv.vrt_profile();
    const json = new TextDecoder().decode(new Uint8Array(viv.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
    for (const [name, [calls, copied]] of Object.entries(JSON.parse(json))) {
      console.error(`${name.padEnd(20)} ${String(calls).padStart(10)} calls ${String(copied).padStart(14)} elements copied`);
    }
  }
  if (r.error) {
    // what the program printed before it stopped, as `viv run` and the C engine print it
    process.stdout.write(r.partial);
    console.error(`error: ${r.error}`);
    // 4: a run-time error, as `viv run` and the C engine exit
    return 4;
  }
  process.stdout.write(r.output);
  if (time) console.error(`${r.seconds.toFixed(3)} s`);
  if (args.includes('--memory')) console.error(`memory ${viv.memory.buffer.byteLength}`);
  if (r.holds > 0) console.error(`${r.holds} hold${r.holds === 1 ? '' : 's'} FAILED`);
  if (r.fails > 0) console.error(`${r.fails} assertion${r.fails === 1 ? '' : 's'} FAILED`);
  return r.fails > 0 || r.holds > 0 ? 1 : 0;
}

/** The source in `file`, imports expanded, through the interpreter in viv.wasm. */
async function interp(viv) {
  // modules resolve as the feed build resolves them (tools/hf_wasm_time.mjs)
  const DIRS = { dft: 'dft', pde: 'pde', lib: 'lib', examples: 'lang/viv/examples' };
  const resolve = async (name) => {
    if (name.startsWith('post:')) {
      const slug = name.slice(5);
      const hit = readdirSync(join(ROOT, 'posts')).find((f) => f.replace(/^\d{4}-\d{2}-\d{2}-/, '') === `${slug}.viv`);
      if (!hit) throw new Error(`module \`${name}\`: no post by that slug in posts/`);
      return readFileSync(join(ROOT, 'posts', hit), 'utf8');
    }
    const [head, ...rest] = name.split('/');
    const f = DIRS[head] && rest.length ? join(ROOT, DIRS[head], `${rest.join('/')}.viv`) : null;
    if (!f || !existsSync(f)) throw new Error(`module \`${name}\`: no module by that name`);
    return readFileSync(f, 'utf8');
  };
  const src = await expandUses(readFileSync(file, 'utf8'), resolve);
  const enc = new TextEncoder().encode(src);
  const ptr = viv.viv_alloc(enc.length);
  new Uint8Array(viv.memory.buffer, ptr, enc.length).set(enc);
  const t0 = performance.now();
  const packed = (viv.viv_run_keep || viv.viv_run)(ptr, enc.length);
  const secs = (performance.now() - t0) / 1000;
  const out = new TextDecoder().decode(new Uint8Array(viv.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
  const stop = out.indexOf('\0');
  if (stop >= 0 || out.startsWith('error: ')) {
    if (stop > 0) process.stdout.write(out.slice(0, stop).replace(/^__(graph|trace)__.*\n/gm, ''));
    console.error(out.slice(stop + 1).trimEnd());
    return 4;
  }
  process.stdout.write(out.replace(/^__(graph|trace)__.*\n/gm, ''));
  if (time) console.error(`${secs.toFixed(3)} s`);
  return out.split('\n').some((l) => l.startsWith('FAIL ')) ? 1 : 0;
}
