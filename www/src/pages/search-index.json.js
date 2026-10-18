// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// /search-index.json: what the search box in the top bar searches. One entry
// per page: the vipyr reference and gallery, the Viv docs topics, posts,
// ladders, walk-throughs, and the pages named in src/lib/site.ts. Built with
// the site, so a page added there is searchable without another step.
import vipyr from '../data/vipyr.json';
import gallery from '../data/gallery.json';
import { texToPlain } from '../lib/richText.js';
import docs from '../data/vivDocExamples.json';
import feed from '../data/feed.json';
import ladders from '../data/ladders.json';
import explore from '../data/explore.json';
import { SIDES, TOPICS } from '../lib/site';

export function GET() {
  const pages = [];
  const add = (title, url, kind, text = '', boost = 0) => pages.push({ title, url, kind, text, boost });
  for (const s of SIDES) {
    for (const l of [...s.pages, ...(s.more || [])]) add(l.label, l.href, s.label, l.blurb || '', 3);
  }
  add('Home', '/', 'viv-lang.org', 'vipyr and the Viv language', 1);
  for (const f of vipyr.fields) {
    const t = TOPICS.find((t) => t.families.includes(f.id));
    add(`vp.${f.id}`, `/vipyr/${f.id}`, 'vipyr family', `${f.title}. ${f.about}${t ? ' ' + t.title + '.' : ''}`, 2);
  }
  for (const fn of Object.values(vipyr.functions)) {
    add(`vp.${fn.qualname}`, `/vipyr/${fn.family}/${fn.name}`, 'vipyr API', fn.summary || '', 2);
  }
  for (const e of gallery.entries) add(e.title, `/vipyr/gallery/${e.slug}`, 'Gallery', texToPlain(e.summary || ''), 2);
  for (const t of docs.topics) add(t.title, `/docs/${t.slug}`, 'Viv docs', t.intro || '');
  for (const p of feed.posts) add(p.titleText || p.title, `/p/${p.slug}`, 'Post', (p.tags || []).join(' '));
  for (const l of ladders.ladders) add(l.title, `/ladders/${l.name}`, 'Ladder', (typeof l.intro === 'string' ? l.intro : '').replace(/<[^>]+>/g, ''));
  for (const p of explore.programs) add(p.title, `/explore/${p.slug}`, 'Explore', p.blurb || '');
  return new Response(JSON.stringify({ pages }), { headers: { 'Content-Type': 'application/json' } });
}
