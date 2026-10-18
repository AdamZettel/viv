// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// codemap-check -- every call line of every code map (public/codemap/*.json,
// /vipyr/map/<function>) stays clear of the tiles it does not connect.
//
//     node scripts/codemap-check.mjs
//
// For each map, laid out with every module open and again with the modules a
// big map starts with folded (components/CodeMap.jsx, FOLD_ABOVE), it routes
// every call one by one (src/lib/codemap.js, makeRouter) and every trunk
// between two modules, and counts the lines that pass over or touch a tile
// other than their own caller and callee. Any such line fails the check: it
// would look like a call the code does not make.
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { layoutMap, makeRouter, tileHits, trunks } from '../src/lib/codemap.js';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const DIR = join(WWW, 'public/codemap');
const FOLD_ABOVE = 80;
let bad = 0;
let lines = 0;
const t0 = performance.now();
for (const f of readdirSync(DIR).filter((n) => n.endsWith('.json')).sort()) {
  const map = JSON.parse(readFileSync(join(DIR, f), 'utf8'));
  const folds = [new Set()];
  if (map.defs.length > FOLD_ABOVE) {
    const open = new Set(map.defs.filter((d) => d.dist <= 1).map((d) => d.module));
    folds.push(new Set(map.modules.map((_, k) => k).filter((k) => !open.has(k))));
  }
  for (const folded of folds) {
    const lay = layoutMap(map, folded);
    const router = makeRouter(lay);
    const ls = [];
    for (const [a, b] of lay.edges) {
      if (a === b) continue;
      const pts = router.call(a, b);
      if (pts) ls.push({ pts, ends: [a, b] });
    }
    for (const t of trunks(lay, router)) ls.push({ pts: t.pts, ends: [] });
    const { hits, bad: which } = tileHits(lay, ls);
    lines += ls.length;
    const tag = `${f.replace(/\.json$/, '')}${folded.size ? ' (folded)' : ''}`;
    if (hits) {
      bad += hits;
      console.error(`codemap-check: ${tag}: ${hits} of ${ls.length} lines cross a tile, e.g. ${which.slice(0, 3).map(([e, t]) => `${e} over ${t}`).join('; ')}`);
    } else console.log(`codemap-check: ${tag}: ${ls.length} lines, none crosses a tile`);
  }
}
console.log(`codemap-check: ${lines} lines in ${Math.round(performance.now() - t0)} ms, ${bad} crossing a tile`);
if (bad) process.exit(1);
