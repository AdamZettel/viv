// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The data files a program reads, attached before it runs in the browser. A
// browser has no file system: viv.wasm keeps a table of files (vrt_file_put),
// and read_text, read_lines and read_bytes read
// from it. A program names its files by repository path, whole
// ("lib/power/cases/case118.m") or as the start of one
// ("lib/stats/data/" + name + ".dat", f"lib/power/cases/{name}.m"); every
// staged data file (/programs/data-index.json) that a string in the program
// begins is attached under that path.

let indexPromise = null;
function dataIndex() {
  if (!indexPromise) {
    indexPromise = fetch('/programs/data-index.json')
      .then((r) => (r.ok ? r.json() : []))
      .catch(() => []);
  }
  return indexPromise;
}

/** The staged data files a program's text names, by path. */
export async function filesNamedBy(src) {
  if (!/\bread_(text|lines|bytes)\s*\(/.test(src)) return [];
  const index = await dataIndex();
  if (!index.length) return [];
  // the text of each string literal up to its first f-string field
  const starts = [...src.matchAll(/"([^"\n{]*)/g)].map((m) => m[1]).filter((t) => t.includes('/'));
  return index.filter((p) => starts.some((t) => p.startsWith(t)));
}

const attached = new WeakMap(); // exports -> Set of paths already in its table

/** Put the files `src` names into this viv.wasm instance's table (each once). */
export async function attachFiles(ex, src) {
  if (!ex || !ex.vrt_file_put) return;
  const want = await filesNamedBy(src);
  if (!attached.has(ex)) attached.set(ex, new Set());
  const have = attached.get(ex);
  for (const path of want) {
    if (have.has(path)) continue;
    const r = await fetch(`/programs/${path}`).catch(() => null);
    if (!r || !r.ok) continue;
    const data = new Uint8Array(await r.arrayBuffer());
    const name = new TextEncoder().encode(path);
    const np = ex.viv_alloc(name.length);
    new Uint8Array(ex.memory.buffer, np, name.length).set(name);
    const dp = ex.viv_alloc(data.length);
    new Uint8Array(ex.memory.buffer, dp, data.length).set(data);
    ex.vrt_file_put(np, name.length, dp, data.length);
    have.add(path);
  }
}
