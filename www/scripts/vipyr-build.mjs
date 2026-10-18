// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// vipyr-build -- src/data/vipyr.json, the data of the /vipyr pages.
//
//     node scripts/vipyr-build.mjs
//
// The file is written by `python -m vipyr.docs` (python/vipyr/docs.py) from
// the programs' manifests and the wrappers `viv export` writes from them, so
// it needs Python with NumPy and a native `viv` ($VIV or target/release/viv).
// Where either is missing (the Netlify build has only the wasm compiler) the
// committed file is used as it is; python/tests/test_vipyr_programs.py fails when the
// committed file is stale.
import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const OUT = join(WWW, 'src/data/vipyr.json');
const viv = process.env.VIV || join(ROOT, 'target/release/viv');

let why = null;
if (!existsSync(viv)) why = `no native viv at ${viv}`;
else {
  const r = spawnSync('python3', ['-m', 'vipyr.docs'], { cwd: join(ROOT, 'python'), encoding: 'utf8', env: { ...process.env, VIV: viv } });
  if (r.error || r.status !== 0) why = `python3 -m vipyr.docs failed: ${(r.error && r.error.message) || r.stderr.trim().split('\n').pop()}`;
}
if (why) {
  if (!existsSync(OUT)) {
    console.error(`vipyr-build: ${why}, and there is no committed ${OUT}`);
    process.exit(1);
  }
  console.log(`vipyr-build: ${why}; using the committed src/data/vipyr.json`);
} else console.log('vipyr-build: wrote src/data/vipyr.json');
