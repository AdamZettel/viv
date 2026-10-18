// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The Atom feed of /posts: one entry per post, newest first, from the same
// feed.json. It stays at /feed.xml, the address subscribers already have, and
// its <id> stays the old one: an Atom id is permanent, and a new one would make
// a reader treat it as a different subscription.
import feed from '../data/feed.json';

const SITE = 'https://viv-lang.org';
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

export function GET() {
  const { author, posts } = feed;
  const updated = posts.length ? `${posts[0].date}T12:00:00Z` : '1970-01-01T00:00:00Z';
  const entries = posts.map((p) => `  <entry>
    <title>${esc(p.titleText || p.title)}</title>
    <link href="${SITE}/p/${p.slug}"/>
    <id>${SITE}/p/${p.slug}</id>
    <published>${p.date}T12:00:00Z</published>
    <updated>${p.date}T12:00:00Z</updated>
    <summary>${esc(p.summaryText || p.title)}</summary>
${p.tags.map((t) => `    <category term="${esc(t)}"/>`).join('\n')}
  </entry>`).join('\n');
  const xml = `<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Viv posts</title>
  <subtitle>Computational demos that run in your browser and check themselves</subtitle>
  <link href="${SITE}/posts"/>
  <link rel="self" href="${SITE}/feed.xml"/>
  <id>${SITE}/feed</id>
  <updated>${updated}</updated>
  <author><name>${esc(author.name)}</name><uri>${esc(author.url)}</uri></author>
${entries}
</feed>
`;
  return new Response(xml, { headers: { 'Content-Type': 'application/atom+xml; charset=utf-8' } });
}
