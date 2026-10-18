// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
import { useEffect, useMemo, useState } from 'react';
import VivCell, { parseOwnOutput, Chart } from './VivCell.jsx';
import { Avatar, ForkMark, TexLine } from './FeedPost.jsx';
import Reviews from './Reviews.jsx';
import { articleBlocks, readMinutes } from '../lib/article.js';
import { withoutHeader } from '../lib/postCheck.js';
import { renderProse, loadKatex } from '../lib/richText.js';

// An article on its own page: the long form the site's PDE and ladder pages
// use. The prose is the body, with `## ` headings and the figure where the
// author put `[figure]`; then what the run established; then the program,
// editable, with its functions; then reviews and forks.
//
// `post.notes` are rendered paragraphs (html). `live` means a server is behind
// the page: reviews, forks and Edit. Everything shown comes from the post's
// header and its run; the page adds layout and nothing else.

const fmtMs = (ms) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);

export default function ArticlePage({ post, author: siteAuthor, dateLabel, session = null, currency = null, live = false, forks = [] }) {
  const author = post.author || siteAuthor;
  const blocks = useMemo(() => articleBlocks(post.notes), [post.notes]);
  const charts = useMemo(() => parseOwnOutput(post.figure).charts, [post.figure]);
  const [source, setSource] = useState(post.source || null);
  const [showProgram, setShowProgram] = useState(false);
  const [katex, setKatex] = useState(null);
  useEffect(() => { if ((post.functions || []).some((f) => /\$[^$]+\$/.test(f.what || ''))) loadKatex().then(setKatex); }, [post.functions]);
  useEffect(() => {
    if (source !== null || !post.file) return;
    fetch(`/programs/posts/${post.file}`).then((r) => r.text()).then((t) => setSource(withoutHeader(t))).catch(() => setSource(''));
  }, [source, post.file]);
  const mine = !!(live && session && post.author && session.handle === post.author.handle);
  const forkHref = `/compose?fork=${encodeURIComponent(post.slug)}${post.file ? `&file=${encodeURIComponent(post.file)}` : ''}`;
  const media = <figure className="ap-media">{charts.map((c, i) => <Chart key={i} chart={c} />)}</figure>;

  return (
    <article className="ap">
      <p className="eyebrow ap-eyebrow">
        <a href="/posts">Posts</a> · Article
        {post.parent && <> · <a className="fp-fork" href={`/p/${post.parent.slug}`}><ForkMark /> fork of {post.parent.author ? `@${post.parent.author.handle}` : post.parent.slug}</a></>}
        {post.agent && <> · written with an agent</>}
      </p>
      <h1 className="ap-title">{post.title}</h1>
      <div className="ap-byline">
        <Avatar author={author} size={36} />
        <div>
          <span className="ap-name"><a href={author.url || '#'} rel="noopener">{author.name}</a> <span className="ap-handle">@{author.handle}</span></span>
          <span className="ap-when"><time dateTime={post.date}>{dateLabel}</time> · {readMinutes(post.summaryHtml, post.notes)} min read · <a className="ap-address" href={`/p/${post.slug}`} title="the program's content address: a hash of its code after parsing, so comments and spacing do not change it">#{post.address}</a></span>
        </div>
      </div>
      {post.summaryHtml && <p className="lede ap-lede" dangerouslySetInnerHTML={{ __html: post.summaryHtml }} />}

      <div className="ap-body">
        {blocks.map((b, i) => {
          if (b.type === 'figure') return <div key={i}>{media}</div>;
          if (b.type === 'pre') return <div key={i} dangerouslySetInnerHTML={{ __html: b.html }} />;
          if (b.type === 'h2') return <h2 key={i} dangerouslySetInnerHTML={{ __html: b.html }} />;
          return <p key={i} dangerouslySetInnerHTML={{ __html: b.html }} />;
        })}
      </div>

      <h2 id="run">What the run shows</h2>
      <div className="run-result">
        <div className="run-main">
          <ul className="run-tests">
            {post.tests.map((t) => <li key={t}><span aria-hidden="true">✓</span><TexLine text={t} /></li>)}
          </ul>
          <p className="run-prov">
            {live ? 'Run by the server when it was posted' : 'Computed when the site was built'}, with the same interpreter your browser runs:
            {' '}{post.tests.length} of {post.tests.length} tests passed, in {post.seconds < 0.1 ? 'under 0.1 s' : fmtMs(post.seconds * 1000)}.
          </p>
        </div>
        {post.shown && post.shown.length > 0 && (
          <dl className="run-shown">
            {post.shown.map((v) => (
              <div key={v.name} style={{ display: 'contents' }}><dt>{v.name}</dt><dd>{v.value}</dd></div>
            ))}
          </dl>
        )}
      </div>

      {/* The program stays out of the reading. Someone learning the physics or the
          math should not have to scroll past code to finish the article; the
          program is one click away for anyone who wants it, and the list of
          checked claims above says what it established without showing it. */}
      <details className="ap-program" onToggle={(e) => setShowProgram(e.currentTarget.open)}>
        <summary>The program behind this article <small>({post.lines} lines of Viv, editable, with its tests)</small></summary>
        <p>
          {post.functions && post.functions.length ? `${post.functions.length === 1 ? 'One function' : `${post.functions.length} functions`}. ` : ''}Change a
          number and press Run (or ⌘/Ctrl + Enter). The tests run too, and one fails if your edit breaks what the article claims.
        </p>
        {post.functions && post.functions.length > 0 && (
          <dl className="pde-fns">
            {post.functions.map((f) => (
              <div key={f.name} style={{ display: 'contents' }}><dt><code>{f.name}</code></dt><dd dangerouslySetInnerHTML={{ __html: renderProse(f.what, katex) }} /></div>
            ))}
          </dl>
        )}
        <div className="cellhost">{showProgram && source !== null && <VivCell idx={0} src={source} standalone />}</div>
        <p className="run-actions">
          <a href={forkHref}>Fork this post</a>
          {mine && <a href={`/compose?edit=${encodeURIComponent(post.slug)}`}>Edit</a>}
          {post.file ? <a href={`/programs/posts/${post.file}`} download>Download {post.file}</a> : <a href={`/workbench`}>Open the workbench</a>}
        </p>
      </details>

      {live && (
        <>
          <h2 id="reviews">Reviews</h2>
          <div className="ap-reviews"><Reviews slug={post.slug} session={session} lastRun={null} currency={currency} /></div>
        </>
      )}
      {(forks.length > 0) && (
        <>
          <h2 id="forks">Forks</h2>
          <ul className="ap-forks">
            {forks.map((f) => (
              <li key={f.slug}><a href={`/p/${f.slug}`}>{f.title}</a><span>@{f.author.handle} · {f.date}{f.agent ? ' · written with an agent' : ''}</span></li>
            ))}
          </ul>
        </>
      )}
    </article>
  );
}
