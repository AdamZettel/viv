// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// /programs/data-index.json: the data files staged under public/programs (every
// file that is not a program or a build record), by their path there, which is
// their repository path (`lib/power/cases/case118.m`). The workbench reads it to
// attach the files a program names before it runs (src/lib/programFiles.js).
import { readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const DATA = /\.(m|raw|dat)$/;

export function GET() {
  const root = join(process.cwd(), 'public/programs');
  const out = [];
  const walk = (dir, rel) => {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const r = rel ? `${rel}/${e.name}` : e.name;
      if (e.isDirectory()) walk(join(dir, e.name), r);
      else if (DATA.test(e.name)) out.push(r);
    }
  };
  if (existsSync(root)) walk(root, '');
  out.sort();
  return new Response(JSON.stringify(out), { headers: { 'Content-Type': 'application/json' } });
}
