// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Imports: `import nn`, `import hf.scf as
// scf`, `from nn import rmsnorm, softmax_rows`. The compiler resolves them
// (lang/viv/src/imports.rs): it finds each module, splices it in once, keeps
// each module's names its own and renames what collides. The page runs the
// same code through the compiler's wasm build (`viv_expand`), so the site and
// the CLI cannot disagree; this file only fetches what the compiler asks for.
//
// The exchange: the compiler is handed the program and the module texts
// fetched so far, by the names it asked for; it answers with the expanded
// program, or with the names it still needs (an import names a module that
// imports others, which only its text says), and the host fetches those and
// asks again. A name it asks for is a candidate path: `./a/b` beside the
// program, `/lib/a/b` and `/a/b` under the repository; a candidate that is
// not there is answered with null.
//
// Where the text comes from is the host's business, so `resolve(name)` is
// given: the site fetches a staged program, the server reads the repository
// or its database, the build reads the repository. `resolve` is called with
// a repository path (`lib/hf/scf`, `dft/kmesh`, `examples/tov` for
// lang/viv/examples, `post:<slug>`), or `./name` for a module beside the
// program, and throws (or returns null) when there is no such module.
import { vivExports } from './vivWasm.js';
import { PUBLIC_SITE } from './repo.js';

/** Whether a text imports anything: an import statement, or a `#! use` line (which the compiler refuses, naming the import to write). */
export const hasUses = (src) => /^(import\s+[A-Za-z_]|from\s+[A-Za-z_][\w.]*\s+import\b|#! use )/m.test(src);

/** A module's own text, with its top-level `test` blocks taken out. */
export function withoutTests(text) {
  const out = [];
  let depth = 0;
  let inTest = false;
  for (const line of text.split('\n')) {
    if (!inTest && /^\s*test\s/.test(line) && line.trimEnd().endsWith('{')) {
      inTest = true;
      depth = 0;
    }
    if (inTest) {
      let quoted = false;
      for (let i = 0; i < line.length; i += 1) {
        const ch = line[i];
        if (quoted) {
          if (ch === '"' && line[i - 1] !== '\\') quoted = false;
          continue;
        }
        if (ch === '"') quoted = true;
        else if (ch === '#') break;
        else if (ch === '{') depth += 1;
        else if (ch === '}') depth -= 1;
      }
      if (depth <= 0) inTest = false;
      continue;
    }
    out.push(line);
  }
  return out.join('\n');
}

/**
 * The modules a text's import lines name, as dotted names (`hf.scf`), for
 * tools that list a file's dependencies without expanding it. `from a import
 * b` lists `a` and `a.b` (`b` may be a submodule); a name that is no module
 * is for the caller to drop.
 */
export function importsOf(src) {
  const out = [];
  let fence = false;
  const lines = src.split('\n');
  for (let i = 0; i < lines.length; i += 1) {
    const l = lines[i];
    if (l.trim() === '###') { fence = !fence; continue; }
    if (fence) continue;
    let m = l.match(/^import\s+([^#]+?)\s*(#.*)?$/);
    if (m) {
      for (const part of m[1].split(',')) out.push(part.trim().split(/\s+/)[0]);
      continue;
    }
    m = l.match(/^from\s+([A-Za-z_][\w.]*)\s+import\s+([^#]*?)\s*(#.*)?$/);
    if (m) {
      let list = m[2];
      if (list.startsWith('(') && !list.includes(')')) {
        while (i + 1 < lines.length && !lines[i].includes(')')) { i += 1; list += ` ${lines[i].replace(/#.*$/, '')}`; }
      }
      out.push(m[1]);
      for (const n of list.replace(/[()]/g, '').split(',')) {
        const w = n.trim().split(/\s+/)[0];
        if (w) out.push(`${m[1]}.${w}`);
      }
    }
  }
  return out;
}

/** A text with its import statements blanked (a parenthesized list's lines too), for a tool that splices modules itself. */
export function stripImports(src) {
  const lines = src.split('\n');
  let fence = false;
  for (let i = 0; i < lines.length; i += 1) {
    if (lines[i].trim() === '###') { fence = !fence; continue; }
    if (fence || !/^(import\s+[A-Za-z_]|from\s+[A-Za-z_][\w.]*\s+import\b)/.test(lines[i])) continue;
    const open = lines[i].includes('(') && !lines[i].includes(')');
    lines[i] = '';
    while (open && i + 1 < lines.length) {
      i += 1;
      const done = lines[i].includes(')');
      lines[i] = '';
      if (done) break;
    }
  }
  return lines.join('\n');
}

/** A dotted module name's repository paths, in the compiler's order (lib/ first): `hf.scf` is lib/hf/scf, then hf/scf. */
export const pathsOf = (dotted) => {
  const p = dotted.replace(/\./g, '/');
  return p.startsWith('lib/') || p.startsWith('examples/') ? [p] : [`lib/${p}`, p];
};

// ---- the compiler ---------------------------------------------------------------

let compilerPromise = null;

/** Hand uses.js a compiler already instantiated (a page that has one); otherwise it loads viv.wasm itself. */
export function setCompiler(exports) {
  compilerPromise = Promise.resolve(exports);
}

function compiler() {
  if (!compilerPromise) {
    compilerPromise = (async () => {
      if (typeof window === 'undefined') {
        const { readFileSync } = await import('node:fs');
        const bytes = readFileSync(new URL('../../public/viv/viv.wasm', import.meta.url));
        return (await WebAssembly.instantiate(bytes, {})).instance.exports;
      }
      // the page's one compiled viv.wasm (src/lib/vivWasm.js)
      return vivExports();
    })();
    compilerPromise.catch(() => {
      compilerPromise = null;
    });
  }
  return compilerPromise;
}

function call(ex, text) {
  const enc = new TextEncoder().encode(text);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const packed = ex.viv_expand(ptr, enc.length);
  const p = Number(packed >> 32n);
  const n = Number(packed & 0xffffffffn);
  return new TextDecoder().decode(new Uint8Array(ex.memory.buffer, p, n));
}

/** A module's text by the name the compiler asked for, or null. */
async function fetchModule(resolve, name) {
  const plain = name.startsWith('/') ? name.slice(1) : name;
  try {
    const t = await resolve(plain);
    return typeof t === 'string' ? t : null;
  } catch {
    return null;
  }
}

/**
 * A program with its imports expanded, by the compiler. `seen` (a Set of
 * module names) is shared by texts that run in one session, a notebook's
 * cells: a module an earlier cell spliced in is not spliced again.
 * `modules` (an object) caches fetched texts across calls.
 */
export async function expandUses(src, resolve, _stack = [], seen = new Set(), modules = {}) {
  if (!hasUses(src) && !/^#! if-(none-)?used /m.test(src)) return src;
  const ex = await compiler();
  for (let round = 0; round < 64; round += 1) {
    const ans = JSON.parse(call(ex, JSON.stringify({ src, modules, seen: [...seen] })));
    if (ans.need) {
      for (const n of ans.need) modules[n] = await fetchModule(resolve, n);
      continue;
    }
    if (ans.error) throw new Error(ans.error);
    for (const u of ans.units || []) seen.add(u);
    return ans.text;
  }
  throw new Error('the imports did not settle: a module keeps asking for more');
}

/**
 * The modules a program's expansion reads, each once, with its text: what the
 * explorer bundle and the lenses carry. `name` is the module's name as the
 * compiler found it (`lib/hf/scf`, `dft/kmesh`, `./seq` beside the program).
 */
export async function moduleClosure(src, resolve) {
  const modules = {};
  const seen = new Set();
  await expandUses(src, resolve, [], seen, modules);
  const out = [];
  for (const u of seen) {
    const key = Object.keys(modules).find((k) => (k.startsWith('/') ? k.slice(1) : k) === u && modules[k] !== null);
    if (key !== undefined) out.push({ name: u, text: modules[key] });
  }
  return out;
}

let stagedPromise = null;
/**
 * What is staged under /programs, by path without `.viv` (/programs/atlas-index.json,
 * made by src/pages/programs/atlas-index.json.js), or null when the index cannot
 * be read. The compiler names several candidate paths for a module; with this
 * list only the one that is there is fetched.
 */
export function stagedPrograms() {
  if (!stagedPromise) {
    stagedPromise = fetch('/programs/atlas-index.json')
      .then((r) => (r.ok ? r.json() : null))
      .then((xs) => (Array.isArray(xs) ? new Set(xs) : null))
      .catch(() => null);
  }
  return stagedPromise;
}

/**
 * The browser's resolver: staged programs under /programs, and posts through
 * the server or the build's index (the public build has no server and reads
 * the index only). `base` is the program's own staged path
 * without `.viv` (`claims/math/qr_orthogonality`), so a module beside it
 * (`./qrlib`) is found; text with no file has no modules beside it.
 */
export function browserResolver(base = null) {
  const dir = base && base.includes('/') ? base.slice(0, base.lastIndexOf('/')) : '';
  const cache = new Map();
  const text = async (url) => {
    if (!cache.has(url)) cache.set(url, fetch(url).then((r) => (r.ok ? r.text() : null)).catch(() => null));
    return cache.get(url);
  };
  return async (name) => {
    if (name.startsWith('post:')) {
      const slug = name.slice(5);
      const viaApi = PUBLIC_SITE ? null : await text(`/api/posts/${encodeURIComponent(slug)}`);
      if (viaApi) { try { const j = JSON.parse(viaApi); if (j.post && j.post.source) return j.post.source; } catch { /* not the api */ } }
      const index = await text('/programs/posts/index.json');
      const file = index ? JSON.parse(index)[slug] : null;
      const t = file ? await text(`/programs/posts/${file}`) : null;
      if (t === null) throw new Error(`no post \`${slug}\``);
      return t;
    }
    let path = name;
    if (name.startsWith('./')) {
      if (!base) return null;
      path = dir ? `${dir}/${name.slice(2)}` : name.slice(2);
    }
    const staged = await stagedPrograms();
    if (staged && !staged.has(path)) return null;
    const t = await text(`/programs/${path}.viv`);
    if (t === null) throw new Error(`no module \`${name}\``);
    return t;
  };
}
