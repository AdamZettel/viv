// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The page's one copy of viv.wasm. Every component and every worker gets the
// compiler from here, so a page view fetches it at most once, and only when a
// reader asks for a run (or opens something that cannot be shown without one).
//
//   VIV_WASM_URL      where the compiler is served. The build puts a copy at a
//                     content-hashed address under /_astro/ (astro.config.mjs,
//                     vivWasmAddress), cached forever; dev and node use
//                     /viv/viv.wasm.
//   vivModule()       the compiled WebAssembly.Module, fetched and compiled once
//   vivExports()      a new instance's exports (its own memory)
//   vivWorker(path)   a Worker that is handed the compiled module instead of
//                     fetching viv.wasm itself (the workers in public/ read it)
//   withVivModule(m)  a worker message with the module attached, for the workers
//                     that need the compiler for some jobs only
/* global __VIV_WASM_URL__ */
export const VIV_WASM_URL = typeof __VIV_WASM_URL__ === 'string' ? __VIV_WASM_URL__ : '/viv/viv.wasm';

let modulePromise = null;

/** The compiled compiler, shared by the whole page. */
export function vivModule() {
  if (!modulePromise) {
    modulePromise = (async () => {
      const res = await fetch(VIV_WASM_URL);
      if (!res.ok) throw new Error(`could not load the compiler (${res.status})`);
      if (WebAssembly.compileStreaming && /application\/wasm/.test(res.headers.get('content-type') || '')) {
        return WebAssembly.compileStreaming(res);
      }
      return WebAssembly.compile(await res.arrayBuffer());
    })();
    // a failed load (offline) is tried again on the next run
    modulePromise.catch(() => {
      modulePromise = null;
    });
  }
  return modulePromise;
}

/** A fresh instance of the compiler: its exports. */
export async function vivExports() {
  const instance = await WebAssembly.instantiate(await vivModule(), {});
  return instance.exports;
}

/** `msg` with the compiled module attached as `vivModule`, once it is compiled. */
export async function withVivModule(msg) {
  return { ...msg, vivModule: await vivModule() };
}

/** Post `msg` to `w`; when the module cannot be posted (an old browser), without it. */
export function postVivMessage(w, msg) {
  try {
    w.postMessage(msg);
  } catch {
    w.postMessage({ ...msg, vivModule: null });
  }
}

/**
 * A worker from public/ that takes the page's compiled module as its first
 * message ({vivModule}) instead of fetching viv.wasm. When the module cannot be
 * had, the worker is told so ({vivModule: null}) and fetches VIV_WASM_URL.
 */
export function vivWorker(path, options) {
  const w = new Worker(`${path}?module=1&wasm=${encodeURIComponent(VIV_WASM_URL)}`, options);
  vivModule().then(
    (m) => postVivMessage(w, { vivModule: m }),
    () => w.postMessage({ vivModule: null }),
  );
  return w;
}
