// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Stage Monaco's AMD bundle into public/monaco/vs.
//
// The workbench self-hosts the editor rather than pulling it from a CDN:
// VivIde does `loader.config({ paths: { vs: '/monaco/vs' } })`, so the whole
// site keeps working offline and nothing it serves depends on a third party
// staying up. Copied from node_modules at build time instead of committed, so
// the version is whatever package.json pins and 13 MB stays out of git.
//
// Only what the workbench uses: the editor core, the base workers, and the
// Viv-adjacent language grammars (the C tab shows generated C). The other ~90
// languages are skipped, which is most of the weight.

import { cp, mkdir, rm, readdir, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const SRC = join(WWW, 'node_modules/monaco-editor/min/vs');
const OUT = join(WWW, 'public/monaco/vs');

if (!existsSync(SRC)) {
  console.error(`monaco not installed: ${SRC} is missing — run npm install`);
  process.exit(1);
}

// The grammars worth shipping. Viv's own language is registered in JS by
// VivIde (registerViv), so it needs no grammar file; `c` and `cpp` back the
// C tab.
const KEEP_LANGS = new Set(['c', 'cpp', 'rust', 'python', 'javascript', 'typescript']);

await rm(OUT, { recursive: true, force: true });
await mkdir(OUT, { recursive: true });

await cp(join(SRC, 'loader.js'), join(OUT, 'loader.js'));
await cp(join(SRC, 'editor'), join(OUT, 'editor'), { recursive: true });
await cp(join(SRC, 'base'), join(OUT, 'base'), { recursive: true });

// basic-languages: keep a handful, drop the rest.
const langs = await readdir(join(SRC, 'basic-languages'));
let kept = 0;
for (const l of langs) {
  if (!KEEP_LANGS.has(l)) continue;
  await cp(join(SRC, 'basic-languages', l), join(OUT, 'basic-languages', l), { recursive: true });
  kept++;
}

async function dirSize(p) {
  let total = 0;
  for (const e of await readdir(p, { withFileTypes: true })) {
    const f = join(p, e.name);
    total += e.isDirectory() ? await dirSize(f) : (await stat(f)).size;
  }
  return total;
}
const mb = ((await dirSize(OUT)) / 1024 / 1024).toFixed(1);
console.log(`staged monaco -> public/monaco/vs (${mb} MB, ${kept} grammars)`);
