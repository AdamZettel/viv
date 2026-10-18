// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
import { useEffect, useState } from 'react';
import FeedPost from './FeedPost.jsx';
import LadderRungs, { LadderRules } from './LadderRungs.jsx';
import { api, hasServer } from '../lib/api.js';

// A post alone on its page, for every kind but the article (ArticlePage.jsx):
// the feed's card, then what the kind adds -- a ladder's rungs, a problem's
// solutions -- then the numbers the run printed, the forks, and the address.
// The same component serves the static page (feed.json) and a live one from
// a server; `live` says which.
//
// `forks` and `solutions` are [{slug, title, date (a label), author, agent}].
//
// A post the build made a page for is served statically even when a server is
// running, so this page asks the server what IT knows about the reader -- have
// they solved this problem, may they see its answers -- and prefers that.

export default function PostPage({ post, author, dateLabel, forks = [], solutions = null, session = null, currency = null, live = false, children = null, onReveal = null, onAround = null, bare = false }) {
  const kind = post.kind || 'demo';
  const [fromServer, setFromServer] = useState(null);
  const [member, setMember] = useState(null);
  useEffect(() => {
    if (live || kind !== 'problem') return undefined;
    let alive = true;
    (async () => {
      if (!(await hasServer()) || !alive) return;
      const [p, me] = await Promise.all([api.post(post.slug).catch(() => null), api.me().catch(() => ({ user: null }))]);
      if (!alive) return;
      if (p) setFromServer(p.post);
      setMember(me.user);
    })();
    return () => { alive = false; };
  }, [live, kind, post.slug]);
  // what the reader may see: the server's answer when there is one; otherwise
  // nothing, with the build's list behind a disclosure they have to open
  const known = solutions || post.solutions || [];
  const sols = fromServer ? fromServer.solutions : live ? known : [];
  const hidden = fromServer ? fromServer.hiddenSolutions : live ? post.hiddenSolutions || 0 : known.length;
  const solvedIt = !!(fromServer && fromServer.attempt && fromServer.attempt.solved);
  const reveal = onReveal || (fromServer && hidden > 0
    ? async () => { try { const { post: p } = await api.post(post.slug, true); setFromServer(p); } catch { /* leave it shut */ } }
    : null);
  const openHref = (r) => (r.file ? `/workbench?open=${r.file}` : null);
  return (
    <div className="feed-grid feed-one">
      <div className="feed-main">
        {post.status && post.status !== 'published' && <p className="feed-filter">Not posted yet: only you can see it. <a href={`/compose?edit=${encodeURIComponent(post.slug)}`}>Edit</a></p>}
        {!bare && <p className="feed-crumb"><a href="/posts">← Posts</a></p>}
        {!bare && <div className="feed-item"><FeedPost post={fromServer ? { ...post, attempt: fromServer.attempt, solvedBy: fromServer.solvedBy, solutions: fromServer.solutions, hiddenSolutions: fromServer.hiddenSolutions } : post} author={author} dateLabel={dateLabel} session={session || member} currency={currency} live={live} canSave={live || !!fromServer} onAround={onAround || ((a) => setFromServer((old) => (old ? { ...old, ...a } : old)))} solved={solvedIt} page /></div>}

        {!bare && kind === 'ladder' && post.ladder && (
          <section className="post-rungs" id="rungs">
            <h2>The rungs</h2>
            <p className="post-rungs-lede">
              One program, {post.ladder.rungs.length} rungs. The program at rung <em>k</em> is everything up to
              and including rung <em>k</em>; every rung runs, and every test from the rungs below runs again on
              each. At the top: {post.ladder.top.functions} functions,
              {' '}{post.ladder.top.tests} tests, {post.ladder.top.lines} lines.
            </p>
            <LadderRungs rungs={post.ladder.rungs} openHref={openHref} showBroken={false} />
            <h3 id="rules">The rules every ladder must pass</h3>
            <LadderRules />
          </section>
        )}

        {bare && kind === 'ladder' && post.ladder && (
          <section className="post-forks" id="rules">
            <h2>The rules every ladder must pass</h2>
            <LadderRules />
          </section>
        )}

        {!bare && post.shown && post.shown.length > 0 && (
          <section className="post-forks">
            <h2>Numbers the program printed</h2>
            <dl className="post-shown">
              {post.shown.map((v) => (
                <div key={v.name}>
                  <dt>{v.name}</dt>
                  <dd>{v.value}</dd>
                </div>
              ))}
            </dl>
          </section>
        )}

        {kind === 'problem' && (
          <section className="post-forks" id="solutions">
            <h2>Solutions <span style={{ marginLeft: 8, color: 'var(--heading)' }}>{sols.length + hidden}</span></h2>
            {sols.length + hidden === 0 && (
              <p className="ac-note">
                No solutions posted yet. Press <strong>Solve it</strong> on the card to edit the program there.
                {live && ' When you are signed in, your work is saved to your account; it is posted only if you choose to post it.'}
              </p>
            )}
            {hidden > 0 && (
              <p className="ac-note post-hidden">
                {hidden} solution{hidden === 1 ? ' is' : 's are'} posted. {hidden === 1 ? 'It is' : 'They are'} hidden until you have solved it yourself.
                {reveal && <button type="button" className="fp-savework" onClick={reveal}>Show {hidden === 1 ? 'it' : 'them'} anyway</button>}
              </p>
            )}
            {!reveal && hidden > 0 && known.length > 0 && (
              <details className="post-answers">
                <summary>Read {hidden === 1 ? 'it' : 'them'} anyway</summary>
                <ul>
                  {known.map((f) => (
                    <li key={f.slug}><a href={`/p/${f.slug}`}>{f.title}</a><span>@{f.author.handle} · {f.date}{f.agent ? ' · written with an agent' : ''}</span></li>
                  ))}
                </ul>
              </details>
            )}
            {sols.length > 0 && (
              <ul>
                {sols.map((f) => (
                  <li key={f.slug}><a href={`/p/${f.slug}`}>{f.title}</a><span>@{f.author.handle} · {f.date}{f.agent ? ' · written with an agent' : ''}</span></li>
                ))}
              </ul>
            )}
          </section>
        )}

        {forks.length > 0 && (
          <section className="post-forks" id="forks">
            <h2>Forks <span style={{ marginLeft: 8, color: 'var(--heading)' }}>{forks.length}</span></h2>
            <ul>
              {forks.map((f) => (
                <li key={f.slug}><a href={`/p/${f.slug}`}>{f.title}</a><span>@{f.author.handle} · {f.date}{f.agent ? ' · written with an agent' : ''}</span></li>
              ))}
            </ul>
          </section>
        )}

        {children}

        <p className="post-address" id="address">
          {live ? 'Checked by the server when it was posted' : 'Checked when the site was built'}: {post.tests.length} of {post.tests.length + (post.failed || []).length} tests passed{(post.failed || []).length ? ` (${post.failed.length} left to solve)` : ''}, in {post.seconds < 0.1 ? 'under 0.1' : post.seconds} s.
          {post.file && <> <a href={`/programs/posts/${post.file}`} download>Download {post.file}</a></>}
        </p>
      </div>
    </div>
  );
}
