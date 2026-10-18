// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Channels: where a post goes, and where a course lives.
//
// A channel is a feed of the posts that name it (`#! channel algebra-2`) and,
// above that feed, a SYLLABUS: sections of posts in the order a learner should
// take them. The syllabus is a Markdown file -- channels/<slug>.md in the
// repository, or the text a member writes on the site -- in this shape:
//
//     # Algebra 2
//     One paragraph: the summary, shown in the rail and on the card.
//     More paragraphs: the intro, shown at the top of the channel's page.
//
//     ## Quadratics
//     - the-quadratic-formula-derived — a note after the slug, optional
//     - solve-the-quadratic
//
// ONE copy of the parser, for the site build, the server and the composer.

export const CHANNEL_SLUG_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

/** { title, summary, intro: [paragraphs], sections: [{ title, items: [{ slug, note }] }], errors } */
export function parseChannel(text) {
  const errors = [];
  let title = '';
  const paras = [];
  const sections = [];
  let cur = [];
  const flush = () => { if (cur.length) paras.push(cur.join(' ')); cur = []; };
  for (const raw of String(text || '').split('\n')) {
    const line = raw.trimEnd();
    if (/^# /.test(line)) { if (title) errors.push('two `# ` titles; a channel has one'); title = line.slice(2).trim(); continue; }
    if (/^## /.test(line)) { flush(); sections.push({ title: line.slice(3).trim(), items: [] }); continue; }
    const item = line.match(/^- ([a-z0-9-]+)\s*(?:[—–-]+\s*(.*))?$/);
    if (item) {
      if (!sections.length) errors.push(`\`${line}\`: a post goes under a \`## \` section`);
      else if (!CHANNEL_SLUG_RE.test(item[1])) errors.push(`\`${item[1]}\` is not a post's slug`);
      else sections[sections.length - 1].items.push({ slug: item[1], note: (item[2] || '').trim() });
      continue;
    }
    if (line.startsWith('- ')) { errors.push(`\`${line}\`: an item is \`- <slug> — <note>\``); continue; }
    if (!line.trim()) { flush(); continue; }
    if (sections.length) { errors.push(`\`${line.slice(0, 40)}\`: prose goes before the first section`); continue; }
    cur.push(line.trim());
  }
  flush();
  if (!title) errors.push('no `# Title` line');
  return { title, summary: paras[0] || '', intro: paras.slice(1), sections, errors };
}

/** Every slug a syllabus lists, in order. */
export const syllabusSlugs = (channel) => channel.sections.flatMap((s) => s.items.map((i) => i.slug));

/** The Markdown back from the parts (what the site stores and the editor shows). */
export function channelText({ title, summary, intro = [], sections = [] }) {
  const out = [`# ${title}`, ''];
  if (summary) out.push(summary, '');
  for (const p of intro) out.push(p, '');
  for (const s of sections) {
    out.push(`## ${s.title}`);
    for (const i of s.items) out.push(`- ${i.slug}${i.note ? ` — ${i.note}` : ''}`);
    out.push('');
  }
  return out.join('\n').replace(/\n+$/, '\n');
}
