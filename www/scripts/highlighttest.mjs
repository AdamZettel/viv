// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The highlighter must never change the text it colors: strip the tags it adds,
// undo the escaping, and the source has to come back character for character.
// (The editable cells lay a highlighted copy UNDER a transparent textarea; one
// dropped or doubled character and the caret drifts off the glyphs.)
//
// Checked over every Viv program in the repo, the DFT sources, the C the
// compiler emitted for the internals page, and a sample of each other language.
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { highlight, LANGUAGES } from '../src/lib/highlight.js';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const strip = (h) => h.replace(/<\/?span[^>]*>/g, '').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');

const cases = [];
for (const dir of ['lang/viv/examples', 'lang/viv/std', 'dft']) {
  for (const f of readdirSync(join(ROOT, dir)).filter((f) => f.endsWith('.viv'))) cases.push([`${dir}/${f}`, 'viv', readFileSync(join(ROOT, dir, f), 'utf8')]);
}
const trace = JSON.parse(readFileSync(join(WWW, 'src/data/trace.json'), 'utf8'));
cases.push(['trace.c', 'c', trace.c]);
cases.push(['backend.rs', 'rust', readFileSync(join(ROOT, 'crates/knotkit/src/backend.rs'), 'utf8')]);
cases.push(['highlight.js', 'js', readFileSync(join(WWW, 'src/lib/highlight.js'), 'utf8')]);
cases.push(['make_deck.py', 'python', readFileSync(join(ROOT, 'dft/ref/make_deck.py'), 'utf8')]);
cases.push(['run.sh', 'sh', readFileSync(join(ROOT, 'dft/run.sh'), 'utf8')]);
cases.push(['session', 'console', '$ viv run a.viv --use "b c.viv" | head -3   # comment\nPASS x < y && z\n$ cc -O2 \\\n    a.c -lm\n']);
cases.push(['html', 'html', '<!-- c --><script src="https://x/y.js" defer></script>\n<viv-cell height=300 data-x=\'1\'>a < b && c</viv-cell>\n']);
cases.push(['json', 'json', '{ "a": [1, 2.5e-3, true, null], "b<c": "x & y" }']);
cases.push(['unknown', '', 'a < b && "c"']);

let bad = 0;
const tokens = {};
for (const [name, lang, src] of cases) {
  const h = highlight(src, lang);
  if (strip(h) !== src) {
    bad++;
    const back = strip(h);
    let k = 0;
    while (k < src.length && src[k] === back[k]) k++;
    console.log(`FAIL ${name} (${lang}): text changed at offset ${k}: ${JSON.stringify(src.slice(k - 20, k + 20))} -> ${JSON.stringify(back.slice(k - 20, k + 20))}`);
  }
  for (const m of h.matchAll(/class="tk-(\w+)"/g)) tokens[m[1]] = (tokens[m[1]] || 0) + 1;
}
console.log(`highlighttest: ${cases.length} sources in ${LANGUAGES.length} languages, ${bad} changed; tokens: ${Object.entries(tokens).map(([k, v]) => `${k} ${v}`).join(', ')}`);
process.exit(bad ? 1 : 0);
