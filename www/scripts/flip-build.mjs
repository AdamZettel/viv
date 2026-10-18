// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// flip-build -- src/data/flip.json: for each page of one side of the site, the
// pages of the other side that show the same code. The top bar's flip control
// (layouts/Base.astro) reads it.
//
//     node scripts/flip-build.mjs           write the file
//     node scripts/flip-build.mjs --check   exit 1 when the file is stale
//
// A pair is made only from a file both pages are about:
//
//   vipyr function (/vipyr/<family>/<name>, and its map /vipyr/map/<q>)
//     Its files are the Viv sources its reference page lists and the file its
//     code map starts in. It pairs with a post, an explored program, a DFT
//     source page or an engine when that page's own .viv file is one of them,
//     or imports one of them directly; with the explored program its
//     registry entry names; and with a post its reference text names.
//   gallery entry (/vipyr/gallery/<slug>)
//     (a showcase page, /show/<slug>, counts as the Viv page of its program and
//     of the /gallery entry src/data/showcases.mjs names for it)
//     It pairs with a post, explored program or DFT source page whose .viv
//     file its text or code names, and with the Viv pages of the program
//     /gallery lists for it (src/data/vivgallery.json, scripts/gallery-build.mjs).
//
// The Viv side's pages get the same pairs the other way round. A page with no
// pair gets no entry, and the control does not appear on it.
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { importsOf } from '../src/lib/uses.js';

const WWW = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WWW);
const OUT = join(WWW, 'src/data/flip.json');
const data = (f) => JSON.parse(readFileSync(join(WWW, 'src/data', f), 'utf8'));

const vipyr = data('vipyr.json');
const codemap = data('codemap.json');
const gallery = data('gallery.json');
const feed = data('feed.json');
const explore = data('explore.json');
const dftSource = data('dftSource.json');
const engines = data('engines.json');
const vivGallery = data('vivgallery.json');
const show = data('show.json');

/** The repository files a module text imports directly (as explore-build resolves them). */
function importsFiles(file, src) {
  const dir = file.includes('/') ? file.slice(0, file.lastIndexOf('/')) : '';
  const out = [];
  for (const d of importsOf(src)) {
    const p = d.replace(/\./g, '/');
    const cands = p.startsWith('lib/') ? [p] : [dir && `${dir}/${p}`, `lib/${p}`, p].filter(Boolean);
    const hit = cands.find((c) => existsSync(join(ROOT, `${c}.viv`)));
    if (hit) out.push(`${hit}.viv`);
  }
  return out;
}
const read = (f) => (existsSync(join(ROOT, f)) ? readFileSync(join(ROOT, f), 'utf8') : '');

// ---- the Viv side's pages that are about one file: [href, label, file, imports]
const vivPages = [];
for (const p of feed.posts) {
  const file = `posts/${p.file}`;
  vivPages.push({ href: `/p/${p.slug}`, label: `Post: ${p.title}`, file, imports: importsFiles(file, p.source || read(file)) });
}
for (const p of explore.programs) {
  vivPages.push({ href: `/explore/${p.slug}`, label: `Explore: ${p.title}`, file: p.file, imports: importsFiles(p.file, read(p.file)), slug: p.slug });
}
for (const s of show.pages) {
  vivPages.push({ href: `/show/${s.slug}`, label: s.title, file: s.file, imports: importsFiles(s.file, read(s.file)) });
}
for (const s of dftSource.sources) {
  const file = `dft/${s.file}`;
  vivPages.push({ href: `/dft/source/${s.slug}`, label: `DFT source: ${s.title}`, file, imports: [] });
}
for (const e of engines.engines) {
  vivPages.push({ href: `/engines/${e.name}`, label: `Engine: ${e.title}`, file: e.file, imports: e.program ? [`${e.program}.viv`] : [] });
}

const pairs = new Map();
const add = (a, b) => {
  if (!pairs.has(a.href)) pairs.set(a.href, new Map());
  pairs.get(a.href).set(b.href, b);
  if (!pairs.has(b.href)) pairs.set(b.href, new Map());
  pairs.get(b.href).set(a.href, a);
};

// ---- vipyr functions
for (const [q, f] of Object.entries(vipyr.functions)) {
  if (f.kind === 'coming') continue;
  const cm = codemap.functions[q];
  const files = new Set([...(f.sources || []).filter((s) => s.endsWith('.viv')), ...(cm ? [codemap.maps[cm.map].file] : [])]);
  const text = JSON.stringify([f.checked, f.see_also, f.notes_md, f.description, f.references]);
  const named = new Set(text.match(/posts\/[\w.-]+\.viv/g) || []);
  const page = { href: `/vipyr/${f.family}/${f.name}`, label: `vp.${q}` };
  const mapPage = cm ? { href: `/vipyr/map/${q}`, label: `Map of vp.${q}` } : null;
  for (const v of vivPages) {
    const hit = files.has(v.file) || v.imports.some((i) => files.has(i)) || (v.slug && v.slug === f.explore) || named.has(v.file);
    if (!hit) continue;
    add(page, v);
    if (mapPage) {
      if (!pairs.has(mapPage.href)) pairs.set(mapPage.href, new Map());
      pairs.get(mapPage.href).set(v.href, v);
    }
  }
}

// ---- gallery entries
for (const e of gallery.entries) {
  const text = [e.intro, ...e.cells.map((c) => `${c.md || ''}\n${c.code || ''}`)].join('\n');
  const named = new Set(text.match(/[\w./-]+\.viv\b/g) || []);
  const page = { href: `/vipyr/gallery/${e.slug}`, label: `Gallery: ${e.title}` };
  for (const v of vivPages) if (named.has(v.file)) add(page, v);
}
// the Viv program /gallery lists for a vipyr example: its own page and its post or run
for (const e of vivGallery.entries) {
  const label = e.post ? null : `Viv program: ${e.file}`;
  const sp = show.pages.find((p) => p.gallery === e.id);
  const pages = [e.href, e.post, e.explore, sp ? `/show/${sp.slug}` : null].filter((h) => h && h.startsWith('/'));
  for (const v of e.vipyr) {
    for (const href of new Set(pages)) {
      const known = vivPages.find((p) => p.href === href);
      add({ href: v.href, label: `Gallery: ${v.title}` }, { href, label: known ? known.label : href === '/dft' ? 'Plane-wave DFT' : label });
    }
  }
}

const out = {};
for (const href of [...pairs.keys()].sort()) {
  out[href] = [...pairs.get(href).values()].map(({ href: h, label }) => ({ href: h, label })).sort((a, b) => a.label.localeCompare(b.label));
}
const text = JSON.stringify({ pages: out }, null, 1) + '\n';
if (process.argv.includes('--check')) {
  if (!existsSync(OUT) || readFileSync(OUT, 'utf8') !== text) {
    console.error('flip-build: src/data/flip.json is stale (node scripts/flip-build.mjs rewrites it)');
    process.exit(1);
  }
  console.log(`flip-build: src/data/flip.json is current (${Object.keys(out).length} pages with a counterpart)`);
} else {
  writeFileSync(OUT, text);
  console.log(`flip-build: ${Object.keys(out).length} pages with a counterpart`);
}
