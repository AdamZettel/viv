// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
import { useCallback, useEffect, useMemo, useState } from 'react';
import { api, hasServer, dayLabel } from '../lib/api.js';
import { loadKatex, renderProse, noteHtml } from '../lib/richText.js';
import { parseChannel, channelText } from '../lib/channels.js';
import FeedPost, { Avatar } from './FeedPost.jsx';

// A channel: its syllabus, then its feed. The syllabus is sections of posts
// in the order a learner should take them (channels/<slug>.md in the
// repository, or what the channel's owner wrote on the site); the feed is
// every post that names the channel, newest first.
//
// Static: `channel` comes from feed.json (sections carry `post` cards) and
// `posts` is the channel's feed as the build saw it. Live: everything comes
// from the server, and a member can follow the channel, see which problems
// they have solved, and -- if it is theirs -- edit the syllabus in place.

function Editor({ slug, text: initial, onSaved, onCancel, creating = false }) {
  const [text, setText] = useState(initial || '# \n\nOne paragraph: what the channel is for.\n\n## First section\n- a-post-slug — a note\n');
  const [slugField, setSlugField] = useState(slug || '');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const parsed = useMemo(() => parseChannel(text), [text]);
  const save = async () => {
    setBusy(true); setErr(null);
    try {
      const r = creating ? await api.createChannel(slugField.trim(), text) : await api.updateChannel(slug, text);
      onSaved(r.channel);
    } catch (e) { setErr(e.errors && e.errors.length ? e.errors : [e.message]); } finally { setBusy(false); }
  };
  return (
    <section className="ch-editor">
      {creating ? <h1>Start a channel</h1> : <h2>Edit the syllabus</h2>}
      <p className="ch-editor-help">
        Write Markdown in this order: a <code># Title</code> line, one paragraph of summary, any further introduction,
        then <code>## Section</code> headings, each followed by lines like <code>- post-slug — a note</code>. List only posts
        whose header names this channel (<code>channel: {creating ? (slugField || 'slug') : slug}</code>); posts in the channel that you do not list appear after the listed ones.
      </p>
      {creating && (
        <label className="ch-field"><span>Short name <small>used in the channel's address, /c/&lt;name&gt;: lowercase words joined by hyphens</small></span>
          <input value={slugField} onChange={(e) => setSlugField(e.target.value.toLowerCase())} placeholder="algebra-2" /></label>
      )}
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={16} spellCheck="false" aria-label="The syllabus" />
      {parsed.errors.length > 0 && <ul className="ch-errors">{parsed.errors.map((e, i) => <li key={i}>{e}</li>)}</ul>}
      {err && <ul className="ch-errors">{err.map((e, i) => <li key={i}>{e}</li>)}</ul>}
      <p className="ch-editor-actions">
        <button type="button" className="cp-primary" disabled={busy || parsed.errors.length > 0 || (creating && !slugField.trim())} onClick={save}>{busy ? 'Saving…' : creating ? 'Create the channel' : 'Save the syllabus'}</button>
        {onCancel && <button type="button" onClick={onCancel}>Cancel</button>}
        <span className="ch-editor-count">{parsed.sections.length} section{parsed.sections.length === 1 ? '' : 's'} · {parsed.sections.reduce((n, s) => n + s.items.length, 0) } posts listed</span>
      </p>
    </section>
  );
}

// The channel as a FEED: the same three columns as /feed, the posts as cards in
// the syllabus's order with the section titles as thin dividers between them,
// then the channel's other posts. What the channel is for sits in the right
// rail, beside Follow. (No function props: the static page hands this
// component to the browser as JSON.)
export function ChannelBody({ channel, posts, channels = [], author = null, session = null, currency = null, live = false, onFollow = null, onEdit = null }) {
  // a channel the build made a page for is served statically even with a server
  // running: ask it what this member has solved, and follow from there
  const [fromServer, setFromServer] = useState(null);
  const [member, setMember] = useState(null);
  useEffect(() => {
    if (live) return undefined;
    let alive = true;
    (async () => {
      if (!(await hasServer()) || !alive) return;
      const [c, me] = await Promise.all([api.channel(channel.slug).catch(() => null), api.me().catch(() => ({ user: null }))]);
      if (!alive) return;
      if (c) setFromServer(c.channel);
      setMember(me.user);
    })();
    return () => { alive = false; };
  }, [live, channel.slug]);
  const who2 = session || member;
  const solved = (fromServer ? fromServer.solved : channel.solved) || [];
  const bySlug = new Map(posts.map((p) => [p.slug, p]));
  const listed = new Set(channel.sections.flatMap((s) => s.items.map((i) => i.slug)));
  const more = posts.filter((p) => !listed.has(p.slug));
  const nProblems = posts.filter((p) => p.kind === 'problem').length;
  const nSolved = posts.filter((p) => p.kind === 'problem' && solved.includes(p.slug)).length;
  const who = who2 || author;
  const card = (p) => (
    <div className="feed-item" key={p.slug}>
      <FeedPost post={fromServer && fromServer.attempts ? { ...p, attempt: fromServer.attempts[p.slug] || null } : p} author={author} dateLabel={dayLabel(p.date)} session={who2} currency={currency} live={live} canSave={live || !!fromServer} solved={solved.includes(p.slug)} />
    </div>
  );
  return (
    <div className="feed-grid">
      <aside className="feed-rail feed-left" aria-label="You and the channels">
        {who && (
          <section className="rail-card profile">
            <Avatar author={who} size={44} />
            <h2><a href={session ? '/account' : who.url || '#'} rel="noopener">{who.name}</a></h2>
            <p className="profile-handle">@{who.handle}</p>
            {who.tagline && <p className="profile-tagline">{who.tagline}</p>}
          </section>
        )}
        <section className="rail-card">
          <h2>Channels</h2>
          <ul className="channels">
            <li><a href="/posts">Everything</a></li>
            {channels.map((c) => (
              <li key={c.slug}><a href={`/c/${c.slug}`} aria-current={c.slug === channel.slug ? 'true' : undefined} title={c.summary}>{c.title}</a><span>{c.posts}</span></li>
            ))}
          </ul>
        </section>
      </aside>

      <div className="feed-main">
        <h1 className="sr-only">{channel.title}</h1>
        {posts.length === 0 && <p className="feed-filter">Nothing posted here yet.</p>}
        {channel.sections.map((s, i) => {
          const here = s.items.map((item) => bySlug.get(item.slug)).filter(Boolean);
          if (!here.length) return null;
          return (
            <div className="ch-section" key={i} id={s.topic || undefined}>
              <p className="ch-divider"><span>{s.title}</span></p>
              {here.map(card)}
            </div>
          );
        })}
        {more.length > 0 && (
          <div className="ch-section">
            {channel.sections.length > 0 && <p className="ch-divider"><span>Also in {channel.title}</span></p>}
            {more.map(card)}
          </div>
        )}
      </div>

      <aside className="feed-rail feed-right" aria-label="About the channel">
        <section className="rail-card ch-about">
          <h2>{channel.title}</h2>
          {channel.summaryHtml !== undefined ? <p dangerouslySetInnerHTML={{ __html: channel.summaryHtml }} /> : <p>{channel.summary}</p>}
          {(channel.introHtml || channel.intro || []).map((t, i) => (channel.introHtml ? <p key={i} dangerouslySetInnerHTML={{ __html: t }} /> : <p key={i}>{t}</p>))}
          <p className="rail-small">
            {posts.length} post{posts.length === 1 ? '' : 's'}
            {nProblems > 0 && <> · {nProblems} problem{nProblems === 1 ? '' : 's'}{who2 ? `, ${nSolved} solved by you` : ''}</>}
            {channel.owner && <> · started by @{channel.owner.handle}</>}
          </p>
          <p className="ch-actions">
            {(live || fromServer) && (onFollow || fromServer) && (
              <button type="button" className={`ch-follow${(fromServer ? fromServer.following : channel.following) ? ' on' : ''}`}
                onClick={onFollow || (async () => {
                  if (!who2) { location.href = `/account?next=/c/${channel.slug}`; return; }
                  const on = !(fromServer && fromServer.following);
                  try { await api.follow(channel.slug, on); setFromServer((c) => ({ ...c, following: on })); } catch { /* the rail says the truth next load */ }
                })}
                aria-pressed={(fromServer ? fromServer.following : channel.following)}>
                {(fromServer ? fromServer.following : channel.following) ? '✓ Following' : 'Follow'}
              </button>
            )}
            {live && channel.mine && onEdit && <button type="button" className="cp-link" onClick={onEdit}>Edit the syllabus</button>}
          </p>
        </section>
        <section className="rail-card">
          <h2>How to read it</h2>
          <p>
            Read the posts from top to bottom.
            {nProblems > 0 && <> On a <strong>problem</strong>, press <strong>Solve it</strong> to open its program in the card, fill in the functions, and press Run.</>}
            {/* a sign-in exists only where a server answers (not the public build) */}
            {nProblems > 0 && (live || fromServer) && ' When you are signed in, your work is saved to your account, and nobody else sees it unless you post it.'}
          </p>
        </section>
      </aside>
    </div>
  );
}

/** The live page: /c/<slug> (rewritten to /channel?s=<slug>), or /channel?new=1 to start one. */
export default function ChannelLive() {
  const [state, setState] = useState('loading');
  const [channel, setChannel] = useState(null);
  const [channels, setChannels] = useState([]);
  const [posts, setPosts] = useState([]);
  const [session, setSession] = useState(null);
  const [currency, setCurrency] = useState(null);
  const [editing, setEditing] = useState(false);
  const [err, setErr] = useState('');
  const params = typeof location !== 'undefined' ? new URLSearchParams(location.search) : new URLSearchParams();
  // /c/<slug>, or /channel?s=<slug>. Opened bare (/channel/), there is no
  // channel to ask the server for.
  const cm = typeof location !== 'undefined' ? location.pathname.match(/^\/c\/([^/]+)\/?$/) : null;
  const slug = params.get('s') || (cm ? cm[1] : '');
  const creating = params.get('new') === '1';

  const load = useCallback(async () => {
    const [{ channel: c }, { posts: raw }, all] = await Promise.all([api.channel(slug), api.posts({ channel: slug, lane: 'all', limit: 50 }), api.channels().catch(() => ({ channels: [] }))]);
    const katex = /\$/.test([c.summary, ...c.intro, ...raw.map((p) => p.summary + p.notes.join(''))].join('')) ? await loadKatex() : null;
    setChannel({ ...c, summaryHtml: renderProse(c.summary, katex), introHtml: c.intro.map((t) => renderProse(t, katex)) });
    setChannels(all.channels);
    setPosts(raw.map((p) => ({ ...p, summaryHtml: renderProse(p.summary, katex), notes: p.notes.map((n) => noteHtml(n, katex)) })));
    document.title = `${c.title} · Viv posts`;
  }, [slug]);

  useEffect(() => {
    (async () => {
      const health = await hasServer();
      if (!health) { setState('none'); return; }
      setCurrency(health.currency);
      const me = await api.me().catch(() => ({ user: null }));
      setSession(me.user);
      if (creating) { setState('new'); return; }
      if (!slug) { setErr('This address does not name a channel. A channel\'s address looks like /c/<name>'); setState('error'); return; }
      await load();
      setState('ok');
    })().catch((e) => { setErr(e.message); setState('error'); });
  }, [creating, load]);

  const follow = async () => {
    if (!session) { location.href = '/account?next=' + encodeURIComponent(location.pathname); return; }
    try { await api.follow(slug, !channel.following); setChannel((c) => ({ ...c, following: !c.following })); } catch (e) { setErr(e.message); }
  };

  if (state === 'loading') return <p className="ac-note">Loading…</p>;
  if (state === 'none') return <p className="ac-note">Channels started on the site are stored by the site's server, and this copy of the site runs without one. <a href="/posts">Back to Posts</a>.</p>;
  if (state === 'error') return <p className="ac-note">{err}. <a href="/posts">Back to Posts</a>.</p>;
  if (state === 'new') {
    if (!session) return <p className="ac-note"><a href="/account?next=/channel?new=1">Sign in</a> to start a channel.</p>;
    return <div className="ch"><Editor creating onSaved={(c) => { location.href = `/c/${c.slug}`; }} /></div>;
  }
  if (editing) return <div className="ch"><Editor slug={slug} text={channel.text} onCancel={() => setEditing(false)} onSaved={async () => { setEditing(false); await load(); }} /></div>;
  return <ChannelBody channel={channel} posts={posts} channels={channels} session={session} currency={currency} live onFollow={follow} onEdit={() => setEditing(true)} />;
}
