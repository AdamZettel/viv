// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Build /llms.txt, /llms-full.txt and /viv-prompt-pack.md into public/.
//
//   node scripts/llms-build.mjs          build, after checking what the pack claims
//
// Before writing anything it runs, through the shipped interpreter
// (public/viv/viv.wasm):
//   - every documented example, whose output must still be the recorded one
//     (the pack quotes that output as what Viv prints);
//   - every Python habit, which must fail, and its fix, which must not; the
//     habit's error message goes into the table as measured;
//   - the example post, which must pass the feed's rules (src/lib/postCheck.js).
// Any failure stops the build with exit 1: the pack never teaches something
// the interpreter does not do.
import { readFileSync, writeFileSync, mkdirSync, existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { spawnSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadSources, buildLlms, HABITS, TRAPS, EXAMPLE_POST } from './llms-pack.mjs';
import { judge } from '../src/lib/postCheck.js';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const { instance } = await WebAssembly.instantiate(readFileSync(join(WWW, 'public/viv/viv.wasm')), {});
const ex = instance.exports;
function run(text) {
  const enc = new TextEncoder().encode(text);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const packed = ex.viv_run(ptr, enc.length);
  return new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
}
// the interpreter's call-graph instrumentation is not program output
const clean = (out) => out.split('\n').filter((l) => !/^__(graph|trace|steps|story)__/.test(l)).join('\n').trim();

const problems = [];
const src = loadSources();

for (const e of src.docs.examples) {
  const out = clean(run(e.code));
  if (e.expectError !== undefined) {
    if (!out.startsWith('error') || !out.includes(e.expectError)) problems.push(`example "${e.title}": expected an error containing "${e.expectError}", got: ${out.split('\n')[0]}`);
  } else if (out !== e.expect) problems.push(`example "${e.title}": the output changed; run npm run doctest`);
}

const habits = HABITS.map((h) => {
  const bad = clean(run(h.python));
  const good = clean(run(h.viv));
  const says = bad.split('\n').find((l) => l.startsWith('error')) || '';
  if (!says) problems.push(`habit "${h.habit}": \`${h.python}\` ran without an error; the table would be wrong`);
  if (/^error/m.test(good)) problems.push(`habit "${h.habit}": the fix \`${h.viv}\` fails: ${good}`);
  return { ...h, says };
});

// The traps: the interpreter runs each, the compiler (when built) refuses it.
// Without a built compiler the C half is not checked and the pack leaves the
// compiler's message out rather than quote one nobody ran.
const VIV = join(WWW, '..', 'target', 'release', 'viv');
const haveViv = existsSync(VIV);
const scratch = mkdtempSync(join(tmpdir(), 'llms-'));
const traps = TRAPS.map((t) => {
  const interp = clean(run(t.code));
  if (/^error/m.test(interp)) problems.push(`trap "${t.what}": the interpreter refuses it (${interp.split('\n')[0]}); it is not a trap any more`);
  const fixed = clean(run(t.fix));
  if (/^error/m.test(fixed)) problems.push(`trap "${t.what}": the fix fails: ${fixed}`);
  if (!haveViv) return t;
  const file = join(scratch, 'trap.viv');
  writeFileSync(file, t.code);
  const c = spawnSync(VIV, [file, '-o', join(scratch, 'trap.c')], { encoding: 'utf8' });
  const said = (c.stderr || '').split('\n').find((l) => l.startsWith('error')) || '';
  if (c.status === 0 || !said.includes(t.cSays)) problems.push(`trap "${t.what}": expected the compiler to refuse it with "${t.cSays}", got exit ${c.status}: ${said}`);
  return { ...t, cSaid: said.replace(/^error: /, '') };
});
rmSync(scratch, { recursive: true, force: true });

const post = judge(EXAMPLE_POST, run(EXAMPLE_POST), null);
for (const e of post.errors) problems.push(`the example post is refused: ${e}`);

if (problems.length) {
  console.error(`llms: ${problems.length} problem(s); nothing written`);
  for (const p of problems) console.error(`  ${p}`);
  process.exit(1);
}

const { index, full, pack } = buildLlms(src, { habits, traps });
const pub = join(WWW, 'public');
mkdirSync(pub, { recursive: true });
writeFileSync(join(pub, 'llms.txt'), index);
writeFileSync(join(pub, 'llms-full.txt'), full);
writeFileSync(join(pub, 'viv-prompt-pack.md'), pack);

// The pack's size in tokens, for the pages and the MCP tool that quote it
// (agents.astro, server/mcp/tooldefs.mjs): estimated from its bytes at
// BYTES_PER_TOKEN, what cl100k_base measured on this pack (100,537 bytes, 33,054
// tokens, 2026-10-16), so the figure is the same on every machine, with or
// without tiktoken. `core` is what viv_reference returns with no arguments.
const BYTES_PER_TOKEN = 3.04;
const kTokens = (s) => Math.max(1, Math.round(Buffer.byteLength(s) / BYTES_PER_TOKEN / 1000));
const packParts = pack.split(/^(?=## \d+\. )/m);
const CORE_HEADINGS = ['Grammar and types', 'Python habits that do not work', 'The interpreter runs these; the C compiler refuses them'];
const core = [packParts[0], ...packParts.filter((p) => CORE_HEADINGS.some((h) => p.slice(0, p.indexOf('\n')).endsWith(h)))].join('');
writeFileSync(join(WWW, 'src/data/promptPack.json'), JSON.stringify({ bytes: Buffer.byteLength(pack), kTokens: kTokens(pack), coreKTokens: kTokens(core) }, null, 1) + '\n');

// Token counts, when Python's tiktoken is installed: o200k_base (GPT-4o) if
// this tiktoken has it, else cl100k_base (GPT-4). Claude's tokenizer is not
// public; Anthropic's pricing page says Claude 4.7 and later produce about 30%
// more tokens for the same text, so the Claude figure printed is x 1.3, an
// estimate.
const count = spawnSync('python3', ['-c', 'import sys,tiktoken; n="o200k_base" if "o200k_base" in tiktoken.list_encoding_names() else "cl100k_base"; e=tiktoken.get_encoding(n); print(n, " ".join(str(len(e.encode(open(p).read()))) for p in sys.argv[1:]))',
  join(pub, 'llms.txt'), join(pub, 'viv-prompt-pack.md'), join(pub, 'llms-full.txt')], { encoding: 'utf8' });
const [encoding, ...counts] = count.status === 0 ? count.stdout.trim().split(' ') : [];
const toks = encoding ? counts.map(Number) : null;
const kb = (s) => `${(Buffer.byteLength(s) / 1024).toFixed(1)} KB`;
const t = (i) => (toks ? `, ${toks[i]} ${encoding} tokens (~${Math.round(toks[i] * 1.3)} Claude, estimated)` : '');
console.log(`llms: ${src.docs.examples.length} examples, ${habits.length} habits and ${traps.length} traps checked${haveViv ? '' : ' (the traps\' C half skipped: no target/release/viv)'}, the example post passes`);
console.log(`  public/llms.txt            ${kb(index)}${t(0)}`);
console.log(`  public/viv-prompt-pack.md  ${kb(pack)}${t(1)}`);
console.log(`  public/llms-full.txt       ${kb(full)}${t(2)}`);
