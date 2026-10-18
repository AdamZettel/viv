// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Runs a showcase program in the reader's tab, off the page's thread: one
// worker (public/viv-worker.js) for the whole page, so viv.wasm is fetched
// and compiled once however many demos run. The program is the one
// scripts/show-build.mjs staged with its imports spliced in.
import { vivWorker } from './vivWasm.js';

let worker = null;
let serial = 0;
const waiting = new Map();

function theWorker() {
  if (!worker) {
    worker = vivWorker('/viv-worker.js');
    worker.onmessage = (e) => {
      const w = waiting.get(e.data.id);
      if (!w) return;
      waiting.delete(e.data.id);
      if (e.data.error) w.reject(new Error(e.data.error));
      else w.resolve({ out: e.data.out, seconds: e.data.seconds });
    };
  }
  return worker;
}

const sources = new Map();
async function sourceOf(slug) {
  if (!sources.has(slug)) {
    sources.set(
      slug,
      fetch(`/programs/show/${slug}.run.viv`).then((r) => {
        if (!r.ok) throw new Error(`the program is not staged (${r.status})`);
        return r.text();
      }),
    );
  }
  return sources.get(slug);
}

/** Run the showcase program `slug`; resolves to {out, seconds}. Runs queue in the one worker. */
export async function runShowcase(slug) {
  const src = await sourceOf(slug);
  const id = `show-${++serial}`;
  return new Promise((resolve, reject) => {
    waiting.set(id, { resolve, reject });
    theWorker().postMessage({ id, src });
  });
}

/** The test lines of a run: [{name, pass, why}]. */
export function testsOf(out) {
  return String(out || '')
    .split('\n')
    .filter((l) => l.startsWith('PASS ') || l.startsWith('FAIL '))
    .map((l) => {
      const pass = l.startsWith('PASS ');
      const rest = l.slice(5);
      const cut = pass ? -1 : rest.indexOf(' -- ');
      return { name: cut >= 0 ? rest.slice(0, cut) : rest, pass, why: cut >= 0 ? rest.slice(cut + 4) : '' };
    });
}
