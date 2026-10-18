// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
import { useEffect, useState } from 'react';
import Reviews from './Reviews.jsx';
import { api, hasServer } from '../lib/api.js';

// A post that the build made a page for is also in the server's database once
// one is running (cli.mjs import). This island adds what only the server has --
// reviews, and forks made on the site -- under the static page, and renders
// nothing when there is no server.
export default function PostExtras({ slug, heading = true }) {
  const [state, setState] = useState(null); // { session, currency, forks }
  useEffect(() => {
    let alive = true;
    (async () => {
      const health = await hasServer();
      if (!health || !alive) return;
      const [me, post] = await Promise.all([api.me().catch(() => ({ user: null })), api.post(slug).catch(() => null)]);
      if (!post || !alive) return;
      const f = await api.forks(slug).catch(() => ({ forks: [] }));
      if (alive) setState({ session: me.user, currency: health.currency, forks: f.forks, mine: post.post.mine, post: post.post });
    })();
    return () => { alive = false; };
  }, [slug]);
  if (!state) return null;
  return (
    <div className="post-extras">
      {heading && <h2 id="reviews">Reviews</h2>}
      <div className="ap-reviews"><Reviews slug={slug} session={state.session} lastRun={null} currency={state.currency} /></div>
      {state.forks.length > 0 && (
        <>
          {heading && <h2 id="forks">Forks on the site</h2>}
          <ul className="ap-forks">
            {state.forks.map((f) => (
              <li key={f.slug}><a href={`/p/${f.slug}`}>{f.title}</a><span>@{f.author.handle} · {f.date}{f.agent ? ' · written with an agent' : ''}</span></li>
            ))}
          </ul>
        </>
      )}
      {state.mine && <p className="run-actions"><a href={`/compose?edit=${encodeURIComponent(slug)}`}>Edit this post on the site</a></p>}
    </div>
  );
}
