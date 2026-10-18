// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// /programs/atlas-index.json: every Viv file staged under public/programs, by
// its path there without `.viv` (`lib/hf/scf`, `examples/tov`). The atlas
// (src/lib/atlas.js) asks it before fetching a module the compiler names, so
// a candidate path that is not staged is not fetched at all.
import { readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';

export function GET() {
  const root = join(process.cwd(), 'public/programs');
  const out = [];
  const walk = (dir, rel) => {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const r = rel ? `${rel}/${e.name}` : e.name;
      if (e.isDirectory()) walk(join(dir, e.name), r);
      else if (e.name.endsWith('.viv')) out.push(r.slice(0, -'.viv'.length));
    }
  };
  if (existsSync(root)) walk(root, '');
  out.sort();
  return new Response(JSON.stringify(out), { headers: { 'Content-Type': 'application/json' } });
}
