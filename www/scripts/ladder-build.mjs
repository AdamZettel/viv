// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// ladder-build -- the /ladders pages, from programs written as ladders.
//
// A ladder is ONE Viv program whose rungs are a language construct:
//
//     #! ladder <Title>
//     #! | intro prose (optional)
//     rung "One Gaussian, and the overlap of two" {
//         #| why this is the natural first step, then paragraphs of notes
//         #|
//         #| with $math$ and `code`; a bare `#|` line ends a paragraph
//         def prim_overlap(...) { ... }
//         test "a normalized Gaussian overlaps itself exactly once" { ... }
//     }
//     rung "Kinetic energy integrals" {
//         def contracted_one_elec(...) { ... }     # redefining a def EDITS it
//         ...
//     }
//
// `viv run` runs the top rung; `viv ladder` runs every rung and reports, and
// that report is what this script renders. The rules -- every test passes, a
// test once written stays, every rung adds a test and adds something, an
// untouched function keeps its content address and an edit changes it -- live
// in the compiler (lang/viv/src/ladder.rs), not here. The build fails on any
// broken rule, so a page never describes a ladder the compiler rejected.
//
// Emits www/src/data/ladders.json and stages, for the workbench, the ladder
// file and the program at each rung.
import { readFileSync, writeFileSync, readdirSync, mkdirSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { prose, meta, notesOf } from './prose.mjs';
import { ladderRenderer } from './ladder-render.mjs';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const SRC = join(ROOT, 'ladders');
const STAGE = join(WWW, 'public/programs/ladders');

const { instance } = await WebAssembly.instantiate(readFileSync(join(WWW, 'public/viv/viv.wasm')), {});
const renderLadder = ladderRenderer(instance.exports);

const errors = [];
const ladders = [];
if (existsSync(SRC)) {
  for (const file of readdirSync(SRC).filter((f) => f.endsWith('.viv')).sort()) {
    const name = file.replace(/\.viv$/, '');
    const src = readFileSync(join(SRC, file), 'utf8');
    mkdirSync(join(STAGE, name), { recursive: true });
    writeFileSync(join(STAGE, file), src);
    const { errors: errs, view } = renderLadder(src, { stageDir: join(STAGE, name), openBase: `ladders/${name}` });
    if (errs.length) { errors.push(...errs.map((e) => `${name}: ${e}`)); continue; }
    // a number the intro quotes is one a test checks: the intro cannot drift from the ladder
    const intro = notesOf(src);
    const testNames = view.rungs[view.rungs.length - 1].tests.map((t) => t.name).join(' | ');
    for (const num of intro.join(' ').replace(/\u2212/g, '-').match(/-?\d+\.\d+/g) || []) {
      if (!new RegExp(`(^|[^\\d.])${num.replace(/[.-]/g, '\\$&')}(?![\\d])`).test(testNames)) {
        errors.push(`${name}: the intro quotes ${num}, which no test on the top rung names`);
      }
    }
    ladders.push({
      name, file, title: meta(src, 'ladder') || name, titleHtml: prose(meta(src, 'ladder') || name),
      intro: intro.map(prose),
      rungs: view.rungs, top: view.top,
    });
    console.log(`ladder-build: ${name}: ${view.rungs.length} rungs, ${view.top.tests} tests and ${view.top.functions} functions at the top; ` +
      `${view.rungs.reduce((s, r) => s + r.carried.length, 0)} carried-forward addresses verified`);
  }
}
if (errors.length) {
  console.error(`ladder-build: ${errors.length} rule${errors.length > 1 ? 's' : ''} broken:\n  ${errors.join('\n  ')}`);
  process.exit(1);
}
mkdirSync(join(WWW, 'src/data'), { recursive: true });
writeFileSync(join(WWW, 'src/data/ladders.json'), JSON.stringify({ ladders }, null, 1));
