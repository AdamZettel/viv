// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// What a post's card needs, and the cards a page loads as they come into view.
//
// A card is FeedPost.jsx with the post's build record minus its program text
// (fetched when the reader asks for it). A claim's card shows its title, line
// and figure; its statement, scope, reference and table are its own page's.
// /posts renders the first cards in the page and the rest as short stubs that
// become cards when they scroll near: every card's props written into the
// HTML made the page 9 MB. The stubs' records are /posts/cards/<slug>.json.

const claimCard = ({ statement, statementHtml, scope, scopeHtml, reference, referenceHtml, scopeFirstHtml, referenceFirstHtml, table, whyHtml, preview, pyPreview, agreement, ...keep }) => keep;

/** The props of a post's card. */
export const cardProps = ({ source, shown, ...rest }) => (rest.claim ? { ...rest, claim: claimCard(rest.claim) } : rest);

/** A post's date as its card shows it; UTC, so the build and the browser agree. */
export const dayLabel = (iso) =>
  new Date(iso + 'T12:00:00Z').toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC' });

/** In the browser: turn every `.feed-lazy[data-slug]` into its card when it comes near the viewport. */
export function mountLazyCards(author) {
  const stubs = [...document.querySelectorAll('.feed-lazy[data-slug]')];
  if (!stubs.length) return;
  let mods = null;
  const load = () =>
    (mods ||= Promise.all([import('react'), import('react-dom/client'), import('../components/FeedPost.jsx')]));
  const mount = async (el) => {
    const slug = el.dataset.slug;
    delete el.dataset.slug;
    try {
      const [rec, [React, { createRoot }, { default: FeedPost }]] = await Promise.all([
        fetch(`/posts/cards/${slug}.json`).then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status))))),
        load(),
      ]);
      const host = document.createElement('div');
      createRoot(host).render(React.createElement(FeedPost, { post: rec, author, dateLabel: dayLabel(rec.date) }));
      el.replaceChildren(host);
      el.classList.remove('feed-lazy');
    } catch {
      // the stub stays: its title still links to the post's page
    }
  };
  if (!('IntersectionObserver' in window)) {
    stubs.forEach(mount);
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        io.unobserve(e.target);
        mount(e.target);
      }
    },
    { rootMargin: '800px 0px' },
  );
  for (const el of stubs) io.observe(el);
}
