// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// compare-build -- the /compare pages: the same program in Python and in Viv,
// side by side, from study/compare.
//
// A pair is study/compare/<topic>/<name>.{py,viv,md,gen.json}, and may have a
// third program, <name>.phrased.viv: the Viv written with phrases
// which must have the same regions and print
// what <name>.viv prints. Both programs
// are cut into regions by `# @@ <title>` marker lines, and the page lines the
// two up region by region; <name>.md holds the page's title, what the outputs
// are expected to do, and the notes. <name>.gen.json holds what each program
// printed and what py2viv made of the Python; study/compare/check.sh writes it
// (it runs Python, which this build does not need) and fails when it is stale.
//
// This build checks what it can without Python:
//   - the two programs have the same region titles, in the same order;
//   - the Viv program, run by the interpreter in viv.wasm (the engine the page's
//     Run button uses), prints exactly the output recorded in gen.json;
//   - the recorded outputs agree after the lines the notes say to drop.
// Any failure stops the build, naming the pair.
//
// Writes www/src/data/compare.json and stages each Viv program at
// public/programs/compare/<topic>/<name>.viv for the workbench.
import { readFileSync, writeFileSync, readdirSync, mkdirSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { highlight } from '../src/lib/highlight.js';
import { expandUses, hasUses } from '../src/lib/uses.js';
import { prose } from './prose.mjs';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const SRC = join(ROOT, 'study/compare');
const STAGE = join(WWW, 'public/programs/compare');
const MARK = /^\s*# @@ (.+?)\s*$/;

const { instance } = await WebAssembly.instantiate(readFileSync(join(WWW, 'public/viv/viv.wasm')), {});
const ex = instance.exports;

/** The interpreter in viv.wasm, as the Run button calls it; files it writes stay in its table. */
function runInterp(src) {
  const enc = new TextEncoder().encode(src);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const packed = ex.viv_run(ptr, enc.length);
  const out = new TextDecoder().decode(new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
  return out.replace(/^__(graph|trace|steps|story)__.*\n/gm, '');
}

// modules resolve as tools/run-wasm.mjs resolves them
const DIRS = { dft: 'dft', pde: 'pde', lib: 'lib', examples: 'lang/viv/examples' };
const resolve = async (name) => {
  const [head, ...rest] = name.split('/');
  const f = DIRS[head] && rest.length ? join(ROOT, DIRS[head], `${rest.join('/')}.viv`) : null;
  if (!f || !existsSync(f)) throw new Error(`module \`${name}\`: no module by that name`);
  return readFileSync(f, 'utf8');
};

/** Highlighted html cut into lines, each line's open spans closed at its end and reopened on the next. */
function htmlLines(src, lang) {
  const html = highlight(src, lang);
  const lines = [];
  let cur = '';
  const open = [];
  for (const tok of html.split(/(<span [^>]*>|<\/span>|\n)/)) {
    if (!tok) continue;
    if (tok === '\n') {
      lines.push(cur + '</span>'.repeat(open.length));
      cur = open.join('');
    } else if (tok.startsWith('<span ')) {
      open.push(tok);
      cur += tok;
    } else if (tok === '</span>') {
      open.pop();
      cur += tok;
    } else {
      cur += tok;
    }
  }
  lines.push(cur + '</span>'.repeat(open.length));
  return lines;
}

/**
 * A program cut at its marker lines: [{ title, lines: [{ n, html }] }], the
 * first with title null (what comes before the first marker). Line numbers are
 * the file's; the marker lines themselves are left out (the page shows the
 * title between the rows), as are the SPDX license lines, and blank lines at
 * either end of a region too.
 */
function regionsOf(src, lang) {
  const text = src.replace(/\n$/, '');
  const html = htmlLines(text, lang);
  const regions = [{ title: null, lines: [] }];
  text.split('\n').forEach((line, i) => {
    const m = line.match(MARK);
    if (/^\s*#\s*SPDX-/.test(line)) return;   // the license lines (LICENSING.md) are not shown
    if (m) regions.push({ title: m[1], lines: [] });
    else regions[regions.length - 1].lines.push({ n: i + 1, html: html[i], blank: !line.trim() });
  });
  for (const r of regions) {
    while (r.lines.length && r.lines[0].blank) r.lines.shift();
    while (r.lines.length && r.lines[r.lines.length - 1].blank) r.lines.pop();
    r.lines = r.lines.map(({ n, html: h }) => ({ n, html: h }));
  }
  return regions;
}

function frontMatter(text, where) {
  const lines = text.split('\n');
  if (lines[0].trim() !== '---') throw new Error(`${where}: the notes must start with a --- line`);
  const meta = {};
  for (let i = 1; i < lines.length; i += 1) {
    if (lines[i].trim() === '---') return { meta, body: lines.slice(i + 1).join('\n').trim() };
    const k = lines[i].indexOf(':');
    meta[lines[i].slice(0, k).trim()] = lines[i].slice(k + 1).trim();
  }
  throw new Error(`${where}: the front matter has no closing --- line`);
}

/**
 * The notes, as blocks: { kind: 'li' | 'h' | 'table', html }. A bullet starts
 * with `- ` and continues on indented lines; `### ` is a heading; a run of
 * lines starting with `|` is a table (its first row the header, a `|---|` row
 * skipped); `%%` lines are left out (study/compare/measure.py's markers).
 */
function notesHtml(body) {
  const blocks = [];
  let table = null;
  const cells = (line) => line.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
  for (const line of body.split('\n')) {
    if (line.startsWith('%%')) continue;
    if (line.startsWith('|')) {
      if (!table) blocks.push((table = { kind: 'table', rows: [] }));
      if (!/^\|[\s|:-]+\|$/.test(line.trim())) table.rows.push(cells(line));
      continue;
    }
    table = null;
    const last = blocks[blocks.length - 1];
    if (!line.trim()) {
      if (last?.kind === 'p') last.done = true;
    } else if (line.startsWith('### ')) blocks.push({ kind: 'h', src: line.slice(4) });
    else if (line.startsWith('- ')) blocks.push({ kind: 'li', src: line.slice(2) });
    else if (/^\s/.test(line) && last?.kind === 'li') last.src += ' ' + line.trim();
    else if (last?.kind === 'p' && !last.done) last.src += ' ' + line.trim();
    else blocks.push({ kind: 'p', src: line.trim() });
  }
  return blocks.map((b) => {
    if (b.kind !== 'table') return { kind: b.kind, html: prose(b.src) };
    const [head, ...rows] = b.rows;
    const tr = (r, tag) => `<tr>${r.map((c) => `<${tag}>${prose(c)}</${tag}>`).join('')}</tr>`;
    return { kind: 'table', html: `<table><thead>${tr(head, 'th')}</thead><tbody>${rows.map((r) => tr(r, 'td')).join('')}</tbody></table>` };
  });
}

/** An output as lines, each marked with why it is left out of the comparison, if it is. */
function outputLines(text, dropRe, keepPass) {
  const lines = text.replace(/\n$/, '').split('\n');
  return lines.map((t) => ({
    t,
    drop: !keepPass && t.startsWith('PASS ') ? 'test' : dropRe && dropRe.test(t) ? 'stated' : null,
  }));
}

const errors = [];
const topics = JSON.parse(readFileSync(join(SRC, 'topics.json'), 'utf8'));
const out = { topics: [] };
let pairCount = 0;
for (const topic of topics) {
  const dir = join(SRC, topic.dir);
  const names = readdirSync(dir).filter((f) => f.endsWith('.py')).map((f) => f.slice(0, -3)).sort();
  const pairs = [];
  for (const name of names) {
    const where = `study/compare/${topic.dir}/${name}`;
    const base = join(dir, name);
    const need = ['.viv', '.md', '.gen.json'].filter((ext) => !existsSync(base + ext));
    if (need.length) { errors.push(`${where}: no ${need.join(', ')} (run study/compare/check.sh --update)`); continue; }
    const py = readFileSync(base + '.py', 'utf8');
    const viv = readFileSync(base + '.viv', 'utf8');
    const gen = JSON.parse(readFileSync(base + '.gen.json', 'utf8'));
    const { meta, body } = frontMatter(readFileSync(base + '.md', 'utf8'), where);
    const pyRegions = regionsOf(py, 'python');
    const vivRegions = regionsOf(viv, 'viv');
    const pyTitles = pyRegions.slice(1).map((r) => r.title);
    const vivTitles = vivRegions.slice(1).map((r) => r.title);
    if (JSON.stringify(pyTitles) !== JSON.stringify(vivTitles)) {
      errors.push(`${where}: the regions differ\n  python: ${pyTitles.join(' | ')}\n  viv:    ${vivTitles.join(' | ')}`);
      continue;
    }
    if (pyRegions[0].lines.length || vivRegions[0].lines.length) {
      errors.push(`${where}: code before the first \`# @@\` marker`);
      continue;
    }

    // the Viv program, run here, prints what check.sh recorded
    let expanded;
    try {
      expanded = await expandUses(viv, resolve);
    } catch (e) {
      errors.push(`${where}: ${e.message}`);
      continue;
    }
    const ran = runInterp(expanded);
    if (ran !== gen.viv) {
      errors.push(`${where}: viv.wasm's interpreter prints something other than the recorded output (run study/compare/check.sh --update)\n` +
        `  recorded: ${JSON.stringify(gen.viv.slice(0, 200))}\n  ran:      ${JSON.stringify(ran.slice(0, 200))}`);
      continue;
    }
    // the Viv written with phrases, when the pair has it: the same regions,
    // the same output
    let phr = null;
    if (existsSync(base + '.phrased.viv')) {
      const ptext = readFileSync(base + '.phrased.viv', 'utf8');
      const pr = regionsOf(ptext, 'viv');
      const titles = pr.slice(1).map((r) => r.title);
      if (JSON.stringify(titles) !== JSON.stringify(pyTitles)) {
        errors.push(`${where}.phrased.viv: the regions differ from the pair's\n  phrased: ${titles.join(' | ')}`);
        continue;
      }
      let pexp;
      try {
        pexp = await expandUses(ptext, resolve);
      } catch (e) {
        errors.push(`${where}.phrased.viv: ${e.message}`);
        continue;
      }
      if (runInterp(pexp) !== gen.viv) {
        errors.push(`${where}.phrased.viv: viv.wasm's interpreter prints something other than ${name}.viv's recorded output`);
        continue;
      }
      mkdirSync(join(STAGE, topic.dir), { recursive: true });
      writeFileSync(join(STAGE, topic.dir, `${name}.phrased.viv`), ptext);
      phr = { regions: pr.slice(1).map((r) => r.lines), lines: ptext.replace(/\n$/, '').split('\n').length };
    }
    const keepPass = meta.keep_pass === 'yes';
    const pyDrop = meta.py_drop ? new RegExp(meta.py_drop) : null;
    const vivDrop = meta.viv_drop ? new RegExp(meta.viv_drop) : null;
    const pyOut = outputLines(gen.python, pyDrop, true);
    const vivOut = outputLines(gen.viv, vivDrop, keepPass);
    const kept = (ls) => ls.filter((l) => !l.drop).map((l) => l.t).join('\n');
    if (kept(pyOut) !== kept(vivOut)) {
      errors.push(`${where}: the recorded outputs differ after the stated drops`);
      continue;
    }

    // py2viv's translation, lined up with the Python where its markers allow
    let p2v = { ok: gen.py2viv.ok };
    if (gen.py2viv.ok) {
      const tr = regionsOf(gen.py2viv.text, 'viv');
      const titles = tr.slice(1).map((r) => r.title);
      p2v = {
        ok: true,
        aligned: JSON.stringify(titles) === JSON.stringify(pyTitles),
        helpers: tr[0].lines,
        regions: tr.slice(1).map((r) => r.lines),
        all: htmlLines(gen.py2viv.text.replace(/\n$/, ''), 'viv').map((html, i) => ({ n: i + 1, html })),
        sameOutput: gen.py2viv.same_output,
        output: gen.py2viv.output,
      };
    } else {
      p2v = { ok: false, refusal: gen.py2viv.text.replace(/\n$/, '') };
    }

    let svg = null;
    let png = null;
    if (gen.figure_svg && existsSync(join(dir, gen.figure_svg))) svg = readFileSync(join(dir, gen.figure_svg), 'utf8');
    if (gen.figure_png && existsSync(join(dir, gen.figure_png))) png = 'data:image/png;base64,' + readFileSync(join(dir, gen.figure_png)).toString('base64');

    mkdirSync(join(STAGE, topic.dir), { recursive: true });
    writeFileSync(join(STAGE, topic.dir, `${name}.viv`), viv);
    pairs.push({
      topic: topic.dir, name, title: meta.title, order: Number(meta.order),
      output: meta.output, reason: meta.reason || '', reasonHtml: prose(meta.reason || ''), keepPass,
      notes: notesHtml(body),
      regions: pyTitles.map((title, i) => ({ title, py: pyRegions[i + 1].lines, viv: vivRegions[i + 1].lines })),
      vivSource: viv, usesModules: hasUses(viv),
      pyLines: py.replace(/\n$/, '').split('\n').length, vivLines: viv.replace(/\n$/, '').split('\n').length,
      outputs: { python: pyOut, viv: vivOut },
      p2v, phr, svg, png,
    });
    pairCount += 1;
  }
  pairs.sort((a, b) => a.order - b.order || a.name.localeCompare(b.name));
  const orders = pairs.map((p) => p.order);
  if (new Set(orders).size !== orders.length) errors.push(`study/compare/${topic.dir}: two pairs have the same order`);
  out.topics.push({ ...topic, pairs });
}

if (errors.length) {
  console.error('compare-build: FAILED\n  ' + errors.join('\n  '));
  process.exit(1);
}
writeFileSync(join(WWW, 'src/data/compare.json'), JSON.stringify(out) + '\n');
const translated = out.topics.flatMap((t) => t.pairs).filter((p) => p.p2v.ok).length;
console.log(`compare-build: ${pairCount} pairs in ${out.topics.length} topics; every Viv program reprints its recorded output in viv.wasm; py2viv translates ${translated}`);
