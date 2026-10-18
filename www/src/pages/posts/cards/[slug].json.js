// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// One post's card record (src/lib/feedCards.js), for the cards /posts loads
// as the reader scrolls.
import feed from '../../../data/feed.json';
import { cardProps } from '../../../lib/feedCards.js';

export function getStaticPaths() {
  return feed.posts.map((p) => ({ params: { slug: p.slug }, props: { p } }));
}

export function GET({ props }) {
  return new Response(JSON.stringify(cardProps(props.p)), { headers: { 'Content-Type': 'application/json' } });
}
