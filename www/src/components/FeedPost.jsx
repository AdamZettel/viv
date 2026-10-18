// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
import { splitNotebook, isNotebook } from '../lib/notebook.js';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { loadWasm, parseOwnOutput, Chart, withModules } from './VivCell.jsx';
import { highlight } from '../lib/highlight.js';
import { runProgram } from '../lib/runProgram.js';
import { attachFiles } from '../lib/programFiles.js';
import { pathsOf, stagedPrograms } from '../lib/uses.js';
import Reviews, { reviewsLabel } from './Reviews.jsx';
import ClaimPost from './ClaimPost.jsx';
import FigureBox from './FigureBox.jsx';
import { readMinutes } from '../lib/article.js';
import { api } from '../lib/api.js';
import { PUBLIC_SITE } from '../lib/repo.js';

// One post in the feed. A post is a RECORD before it is a message: the card
// opens with what kind of thing it is and its content address, then who wrote
// it, then the claim. A demo is whole on the card: claim, figure, notes behind
// a link. An article's card is its FRONT -- title, lede, figure, reading time
// -- and the article itself is its page (ArticlePage.jsx). Under the media a
// verification strip says what the run established, and a toolbar does things
// with the program.
//
//   Run    runs the post in THIS tab with the same interpreter the build used,
//          and redraws the figure from that run.
//   Code   opens the program inside the card, editable; Run then runs the edit.
//   Fork   takes the post to the composer as the start of your own.
//
// `post.author` (a server post) wins over the `author` prop (the build's one
// author). `session` is the signed-in member or null; `live` means a server
// is behind the page (reviews, forks, edit).

/** The fork mark: a branch, in the site's green, wherever a post is a fork of another. */
/** The mark a solution carries: a check in a ring. */
export function SolveMark() {
  return (
    <svg className="fp-forkmark" viewBox="0 0 14 14" width="13" height="13" aria-hidden="true">
      <circle cx="7" cy="7" r="5.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <path d="M4.3 7.2l1.9 1.9 3.6-3.9" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function ForkMark() {
  return (
    <svg className="fp-forkmark" viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="4" cy="3" r="1.8" /><circle cx="12" cy="3" r="1.8" /><circle cx="8" cy="13" r="1.8" />
      <path d="M4 4.8v1.4a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2V4.8M8 8.2v3" />
    </svg>
  );
}

export function Avatar({ author, size = 36 }) {
  if (author.avatar) return <img className="fp-avatar" src={author.avatar} alt="" width={size} height={size} />;
  const initials = (author.name || '?').split(/\s+/).map((w) => w[0]).slice(0, 2).join('').toUpperCase();
  return <span className="fp-avatar fp-monogram" style={{ width: size, height: size, fontSize: Math.round(size * 0.38) }} aria-hidden="true">{initials}</span>;
}

import { withoutHeader as stripHeader, LANE_LABELS, LANE_NOTES } from '../lib/postCheck.js';
import { renderProse, loadKatex, texToPlain } from '../lib/richText.js';
import { explorableForPost } from '../data/explorables.mjs';
import { lookInsideLinks } from '../lib/atlasLink.js';
import LookInside from './LookInside.jsx';
const fmtMs = (ms) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);

/** "+2 functions · edits slope · +1 test", from a rung as ladderView.js shapes it. */
function rungMeta(r) {
  const fresh = r.tests.filter((t) => t.isNew).length;
  return [
    r.added.length ? `+${r.added.length} function${r.added.length === 1 ? '' : 's'}` : '',
    r.edited.length ? `edits ${r.edited.map((e) => e.name).join(', ')}` : '',
    fresh ? `+${fresh} test${fresh === 1 ? '' : 's'}` : '',
  ].filter(Boolean).join(' · ');
}

// `live` says the page is served BY the server; `canSave` says a server is
// merely reachable, which is all that saving a member's own work needs -- a
// built page has live = false and canSave = true once it has found one.
// A claim (kind `claim`) has its own card: ClaimPost.jsx.
export default function FeedPost(props) {
  if (props.post && props.post.kind === 'claim') return <ClaimPost post={props.post} author={props.post.author || props.author} dateLabel={props.dateLabel} page={props.page} full={props.full || null} Avatar={Avatar} boxes={props.boxes || null} />;
  return <PostCard {...props} />;
}

/** A line that may hold $math$ (a test's name, a rung's title): plain text as
 *  the server rendered it, typeset once KaTeX has loaded after hydration. */
export function TexLine({ text }) {
  const has = /\$/.test(text || '');
  const [katex, setKatex] = useState(null);
  useEffect(() => { if (has) loadKatex().then((k) => k && setKatex(() => k)); }, [has]);
  if (!has) return text;
  return katex ? <span dangerouslySetInnerHTML={{ __html: renderProse(text, katex) }} /> : texToPlain(text);
}

function PostCard({ post: given, author: siteAuthor, dateLabel, session = null, currency = null, onWallet = null, onAround = null, live = false, canSave = null, page = false, solved = false, boxes = null }) {
  const [post, setPost] = useState(given);
  useEffect(() => { setPost(given); setAttempt(given.attempt || null); }, [given]);
  const author = post.author || siteAuthor;
  const kind = post.kind || 'demo';
  // the post's atlas, and its explorer when the build recorded a run (src/data/explorables.mjs)
  const explorable = live ? null : explorableForPost(post.slug);
  const inside = explorable ? lookInsideLinks({ slug: explorable.slug }) : !live && post.file ? lookInsideLinks({ open: `posts/${post.file}` }) : null;
  const keeps = canSave === null ? live : canSave;
  const cardRef = useRef(null);
  const built = useMemo(() => parseOwnOutput(post.figure).charts, [post.figure]);
  const [charts, setCharts] = useState(built);
  const [runs, setRuns] = useState(0);
  const [state, setState] = useState('idle'); // idle | running | done | failed | error
  const [report, setReport] = useState(null); // { pass, fail, ms, failing[], message }
  // a post with prose cells between its code opens as a notebook on its page;
  // the build says so (post.notebook), and a post from the server is checked here
  const notebook = !!post.notebook || (typeof post.source === 'string' && isNotebook(splitNotebook(post.source)));
  const [notesOpen, setNotesOpen] = useState(kind === 'article' || page);
  // one panel below the toolbar at a time: opening one closes the other, and the
  // opened panel is scrolled into view (a panel that opens below a 440px code
  // panel, out of sight, looks like a button that did nothing)
  const [panel, setPanel] = useState(page ? 'reviews' : null); // null | 'code' | 'reviews'
  const codeOpen = panel === 'code';
  const reviewsOpen = panel === 'reviews';
  const showPanel = useCallback((name) => {
    setPanel((p) => (p === name ? null : name));
    requestAnimationFrame(() => {
      const el = cardRef.current && cardRef.current.querySelector(name === 'code' ? '.fp-code' : '.rv');
      if (el) { const r = el.getBoundingClientRect(); if (r.top < 0 || r.bottom > innerHeight) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }
    });
  }, []);
  // a problem: the member's own work on it, private to them until they post it
  const [attempt, setAttempt] = useState(given.attempt || null); // { solved }
  const [saving, setSaving] = useState(false);
  const [katex, setKatex] = useState(null);
  useEffect(() => { if (post.equation && !post.equationHtml) loadKatex().then(setKatex); }, [post.equation, post.equationHtml]);
  const [voting, setVoting] = useState(false);
  const [liking, setLiking] = useState(false);
  const like = useCallback(async () => {
    if (!session) { location.href = '/account?next=' + encodeURIComponent(location.pathname); return; }
    setLiking(true);
    try {
      const r = await api.like(post.slug, !(post.likes && post.likes.mine));
      setPost((p) => ({ ...p, likes: r.likes }));
    } catch (e) { setReport({ message: e.message }); setState('error'); } finally { setLiking(false); }
  }, [session, post.slug, post.likes]);
  // which run the figure shows: the browser scale, or a run the author recorded elsewhere
  const [scale, setScale] = useState('browser');
  const recorded = (post.recorded || []).find((r) => r.scale === scale) || null;
  const recordedCharts = useMemo(() => (recorded ? parseOwnOutput(recorded.figure).charts : null), [recorded]);
  const vote = useCallback(async (v) => {
    if (!session) { location.href = '/account?next=' + encodeURIComponent(location.pathname); return; }
    setVoting(true);
    try {
      const r = await api.vote(post.slug, post.votes && post.votes.mine === v ? 0 : v);
      setPost((p) => ({ ...p, lane: r.lane, rating: r.rating, votes: r.votes, reviews: r.reviews, forks: r.forks }));
    } catch (e) { setReport({ message: e.message }); setState('error'); } finally { setVoting(false); }
  }, [session, post.slug, post.votes]);
  const [lastRun, setLastRun] = useState(null);
  const [original, setOriginal] = useState(null);
  const [code, setCode] = useState(null);
  const [copied, setCopied] = useState(false);
  const taRef = useRef(null);

  const fetchSource = useCallback(async () => {
    if (original !== null) return code;
    if (post.source) { setOriginal(post.source); setCode(post.source); return post.source; }
    const r = await fetch(`/programs/posts/${post.file}`);
    if (!r.ok) throw new Error(`could not fetch the program (HTTP ${r.status})`);
    const src = stripHeader(await r.text());
    setOriginal(src);
    setCode(src);
    return src;
  }, [original, code, post.file, post.source]);

  // Save the member's work on a problem: private, theirs, and never a post until
  // they say so. The server runs it to say whether the problem's tests passed.
  const keepWork = useCallback(async (src) => {
    if (!session) { location.href = '/account?next=' + encodeURIComponent(location.pathname); return; }
    setSaving(true);
    try {
      const r = await api.saveAttempt(post.slug, src ?? code ?? '');
      setAttempt(r.attempt);
      setPost((p) => ({ ...p, solutions: r.solutions, hiddenSolutions: r.hiddenSolutions, solvedBy: r.solvedBy }));
      // the page around the card shows the same answers; it should open too
      if (onAround) onAround({ solutions: r.solutions, hiddenSolutions: r.hiddenSolutions, solvedBy: r.solvedBy, attempt: r.attempt });
    } catch (e) { setReport({ message: e.message }); setState('error'); } finally { setSaving(false); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session, post.slug, code, onAround]);

  const run = useCallback(async () => {
    setState('running');
    try {
      const [ex, src] = await Promise.all([loadWasm(), fetchSource().then((t) => withModules(t, post.file ? `posts/${post.file.replace(/\.viv$/, '')}` : null))]);
      await attachFiles(ex, src);
      await new Promise((r) => setTimeout(r, 30)); // let "Running" paint before the interpreter takes the thread
      const t0 = performance.now();
      const out = await runProgram(ex, src);
      const ms = performance.now() - t0;
      const lines = out.split('\n');
      const diagnostic = lines.find((l) => /^error\b/.test(l));
      if (diagnostic) { setReport({ message: diagnostic }); setState('error'); return; }
      const pass = lines.filter((l) => l.startsWith('PASS ')).length;
      const failing = lines.filter((l) => l.startsWith('FAIL ')).map((l) => l.slice(5).split(' -- ')[0]);
      const drawn = parseOwnOutput(out).charts;
      if (drawn.length) { setCharts(drawn); setRuns((n) => n + 1); }
      setReport({ pass, fail: failing.length, failing, ms });
      setState(failing.length ? 'failed' : 'done');
      if (!(code !== null && original !== null && code !== original)) setLastRun({ pass, fail: failing.length, ms });
      // a problem worked on in the card: the edit is the member's own work. Signed
      // in, a passing one is kept on the server, privately; otherwise this tab keeps it.
      if (kind === 'problem' && code !== null && original !== null && code !== original) {
        try { sessionStorage.setItem(`viv.solve.${post.slug}`, code); } catch { /* the composer then starts from the problem */ }
        if (keeps && session && !failing.length && pass > 0) keepWork(code);
      }
    } catch (e) {
      setReport({ message: String(e.message || e) });
      setState('error');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fetchSource, code, original, kind, post.slug, post.file, keepWork, live, session]);

  // Open the program to work on: where the work was left off (the saved attempt,
  // or this tab's copy), or the problem as posted when there is none.
  const resume = useCallback(async () => {
    let src = null;
    if (keeps && session) { try { src = (await api.attempt(post.slug)).attempt?.source || null; } catch { src = null; } }
    if (src === null) { try { src = sessionStorage.getItem(`viv.solve.${post.slug}`); } catch { src = null; } }
    try { await fetchSource(); } catch (e) { setReport({ message: String(e.message || e) }); setState('error'); return; }
    if (src !== null) setCode(src);
    showPanel('code');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [keeps, session, post.slug, fetchSource, showPanel]);

  const toggleCode = useCallback(async () => {
    if (!codeOpen) {
      try { await fetchSource(); } catch (e) { setReport({ message: String(e.message || e) }); setState('error'); return; }
    }
    showPanel('code');
  }, [codeOpen, fetchSource, showPanel]);

  const copyLink = useCallback(async () => {
    try { await navigator.clipboard.writeText(`${window.location.origin}/p/${post.slug}`); } catch { /* the link is on the date too */ }
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  }, [post.slug]);

  const html = useMemo(() => (code === null ? '' : highlight(code, 'viv') + '\n'), [code]);
  useEffect(() => {
    const ta = taRef.current;
    if (ta) { ta.style.height = 'auto'; ta.style.height = `${ta.scrollHeight + 2}px`; }
  }, [code, codeOpen]);
  const edited = code !== null && original !== null && code !== original;
  const mine = !!(live && session && post.author && session.handle === post.author.handle);
  const forkHref = `/compose?fork=${encodeURIComponent(post.slug)}${post.file ? `&file=${encodeURIComponent(post.file)}` : ''}`;
  const unit = (currency && currency.unit) || 'cr';

  const media = (
    <div className={`fp-media${state === 'running' && !recorded ? ' fp-busy' : ''}`}>
      {post.recorded && post.recorded.length > 0 && (
        <div className="fp-scales" role="tablist" aria-label="Which run">
          <button type="button" role="tab" aria-selected={!recorded} onClick={() => setScale('browser')}>
            run in your browser
          </button>
          {post.recorded.map((r) => (
            <button key={r.scale} type="button" role="tab" aria-selected={scale === r.scale} onClick={() => setScale(r.scale)} title={Object.entries(r.sets).map(([k, v]) => `${k} = ${v}`).join(', ')}>
              {r.scale} · recorded by the author
            </button>
          ))}
        </div>
      )}
      {recorded && (
        <p className="fp-recorded-note">
          <b>Recorded by the author, not run here.</b> {Object.entries(recorded.sets).map(([k, v]) => `${k} = ${v}`).join(' · ')}{recorded.note ? ` · ${recorded.note}` : ''}
          {' · '}<code>viv run{Object.entries(recorded.sets).map(([k, v]) => ` --set ${k}=${v}`).join('')} …</code>
        </p>
      )}
      {boxes
        ? boxes.map((b, i) => <FigureBox key={i} box={b} />)
        : (recordedCharts || charts).map((c, i) => <Chart key={`${recorded ? recorded.scale : runs}-${i}`} chart={c} />)}
    </div>
  );
  // the one formula the post is about: rendered at build for a committed post,
  // here for one the server serves
  // (Until KaTeX has loaded -- or if it fails to -- show nothing rather than the
  // LaTeX source: a formula in raw TeX reads as a broken page.)
  const equationHtml = post.equationHtml || (post.equation && katex ? renderProse(`$$${post.equation}$$`, katex) : '');
  // an import splices another file in; say which, and link to the staged file
  // it names (the compiler's candidates, lib/ first, looked up in the staged
  // index), or name it without a link when it is a folder of modules
  const uses = [...new Set((code || post.source || '').split('\n').map((l) => (l.match(/^(?:from\s+([A-Za-z_][\w.]*)\s+import\b|import\s+([A-Za-z_][\w.]*))/) || []).slice(1).find(Boolean)).filter(Boolean))];
  const [staged, setStaged] = useState(null);
  useEffect(() => {
    if (!uses.length || staged) return;
    let live = true;
    stagedPrograms().then((s) => { if (live) setStaged(s || new Set()); });
    return () => { live = false; };
  }, [uses.length, staged]);
  const useHref = (d) => {
    const path = staged && pathsOf(d).find((p) => staged.has(p));
    return path ? `/workbench?open=${path}.viv` : null;
  };
  const hasMedia = charts.length > 0 || (post.recorded && post.recorded.length > 0) || state === 'running';
  const solvedHere = kind === 'problem' && state === 'done' && edited;
  const solvedBefore = !!(attempt && attempt.solved);
  // A problem's tests ARE its questions, so they stay on screen: a run marks
  // them off rather than replacing them with a list of what went wrong.
  const questions = (() => {
    if (kind !== 'problem') return [];
    const open = (post.failed || []).map((f) => f.split(' -- ')[0]);
    const names = [...open, ...post.tests.filter((t) => !open.includes(t))];
    if ((state === 'done' || state === 'failed') && report && report.failing) {
      const failing = new Set(report.failing);
      return names.map((name) => ({ name, ok: !failing.has(name) }));
    }
    return names.map((name) => ({ name, ok: !open.includes(name) }));
  })();
  const hasSaved = !!attempt;
  const toSolve = post.failed || [];
  const notes = post.notes.length > 0 && (
    <div className="fp-notes">
      {post.notes.map((n, i) => (n.startsWith('<pre') ? <div key={i} dangerouslySetInnerHTML={{ __html: n }} /> : <p key={i} dangerouslySetInnerHTML={{ __html: n }} />))}
    </div>
  );

  return (
    <article className={`fp fp-${kind} fp-in-${post.lane || 'draft'}`} id={post.slug} ref={cardRef}>
      <div className="fp-record">
        <span className={`fp-kind fp-kind-${kind}`}>{kind}</span>
        {post.lane && <span className={`fp-lane fp-lane-${post.lane}`} title={LANE_NOTES[post.lane]}>{LANE_LABELS[post.lane] || post.lane}</span>}
        {post.parent && (
          <a className="fp-fork" href={`/p/${post.parent.slug}`} title={post.parent.title ? `forked from "${post.parent.title}"` : 'forked from another post'}>
            <ForkMark /> fork of {post.parent.author ? `@${post.parent.author.handle}` : post.parent.slug}
          </a>
        )}
        {kind === 'solution' && post.solves && (
          <a className="fp-fork fp-solves" href={`/p/${post.solves}`} title={post.problem && post.problem.title ? `solves "${post.problem.title}"` : 'solves a problem post'}>
            <SolveMark /> solves {post.problem && post.problem.title ? post.problem.title : post.solves}
          </a>
        )}
        {post.channel && <a className="fp-channel" href={`/c/${post.channel}`} title="the channel it is posted to">{post.channelTitle || post.channel}</a>}
        {(solved || solvedBefore) && <span className="fp-fork fp-solves" title="you have solved this problem; your solution stays private until you post it"><SolveMark /> solved by you</span>}
        {post.agent && <span className="fp-agent" title={`Written with an AI agent (${post.agent})`}>written with an agent</span>}
        <a className="fp-address" href={`/p/${post.slug}`} title="the program's content address: a hash of its code after parsing, so comments and spacing do not change it">#{post.address}</a>
      </div>

      <header className="fp-head">
        <a href={author.url || '#'} rel="noopener" className="fp-avatar-link" tabIndex={-1} aria-hidden="true"><Avatar author={author} /></a>
        <div className="fp-who">
          <span className="fp-name"><a href={author.url || '#'} rel="noopener">{author.name}</a></span>
          <span className="fp-handle">@{author.handle}</span>
          <span className="fp-dot" aria-hidden="true">·</span>
          <a className="fp-when" href={`/p/${post.slug}`}><time dateTime={post.date}>{dateLabel}</time></a>
          {kind === 'article' && <span className="fp-read">{readMinutes(post.summaryHtml, post.notes)} min read</span>}
        </div>
      </header>

      <div className="fp-text">
        <h2>{page ? post.title : <a href={`/p/${post.slug}`}>{post.title}</a>}</h2>
        {equationHtml && <div className="fp-equation" dangerouslySetInnerHTML={{ __html: equationHtml }} />}
        {post.summaryHtml && <p className="fp-summary" dangerouslySetInnerHTML={{ __html: post.summaryHtml }} />}
      </div>

      {hasMedia && media}

      {kind === 'ladder' && post.ladder && (
        <ol className="fp-rungs">
          {post.ladder.rungs.map((r) => (
            <li key={r.n}>
              <span className="fp-rung-n">{r.n}</span>
              {page ? <a href={`#rung-${r.n}`}><TexLine text={r.title} /></a> : <a href={`/p/${post.slug}#rung-${r.n}`}><TexLine text={r.title} /></a>}
              <span className="fp-rung-meta">{rungMeta(r)}</span>
            </li>
          ))}
        </ol>
      )}

      {questions.length > 0 && (
        <ul className="fp-tosolve" aria-label="What this problem asks">
          {questions.map((q) => (
            <li key={q.name} className={q.ok ? 'ok' : ''}><span aria-hidden="true">{q.ok ? '✓' : '✗'}</span><TexLine text={q.name} /></li>
          ))}
        </ul>
      )}

      {kind === 'article' && !page && (
        <p className="fp-readmore">
          <span>{post.notes.length} paragraph{post.notes.length === 1 ? '' : 's'}{post.functions && post.functions.length ? ` · ${post.functions.length} function${post.functions.length === 1 ? '' : 's'}` : ''} · {readMinutes(post.summaryHtml, post.notes)} min</span>
        </p>
      )}

      {kind === 'demo' && post.notes.length > 0 && (
        <div className="fp-notes-wrap">
          <button type="button" className="fp-notes-toggle" onClick={() => setNotesOpen((o) => !o)} aria-expanded={notesOpen}>
            {notesOpen ? 'Hide the notes' : `Notes · ${post.notes.length} paragraph${post.notes.length === 1 ? '' : 's'}`}
          </button>
          {notesOpen && notes}
        </div>
      )}

      <div className="fp-strip">
        <span className={`fp-verdict fp-${recorded ? 'idle' : state}`} role="status">
          {recorded && <><b>✓ {recorded.tests.length} test{recorded.tests.length === 1 ? '' : 's'} pass</b><i>in the author's recorded run · not run by the server</i></>}
          {!recorded && state === 'idle' && kind === 'problem' && solvedBefore && <><b>✓ you solved this</b><i>your solution is saved in your account · press Open your work to see it</i></>}
          {!recorded && state === 'idle' && kind === 'problem' && !solvedBefore && <><b>✗ {toSolve.length} test{toSolve.length === 1 ? '' : 's'} to solve</b><i>{hasSaved ? 'you have work saved on it · ' : ''}{post.tests.length ? `${post.tests.length} pass already · ` : ''}the unfinished functions run, and the failing tests show what they must compute</i></>}
          {!recorded && state === 'idle' && kind === 'solution' && <><b>✓ {post.tests.length} test{post.tests.length === 1 ? '' : 's'} pass</b><i>{post.problem ? `the problem's tests, unchanged, among them · ` : ''}{live ? 'checked by the server when posted' : 'checked when the site was built'}</i></>}
          {!recorded && state === 'idle' && kind === 'ladder' && post.ladder && <><b>✓ {post.tests.length} test{post.tests.length === 1 ? '' : 's'} pass</b><i>on {post.ladder.rungs.length} rungs, each rerunning every earlier test · {live ? 'checked by the server when posted' : 'checked when the site was built'}</i></>}
          {!recorded && state === 'idle' && !['problem', 'solution', 'ladder'].includes(kind) && <><b>✓ {post.tests.length} test{post.tests.length === 1 ? '' : 's'} pass</b><i>{live ? 'checked by the server when posted' : 'checked when the site was built'}</i></>}
          {!recorded && state === 'running' && <b>running in your browser…</b>}
          {!recorded && state === 'done' && solvedHere && <><b>✓ solved: {report.pass} of {report.pass} tests pass</b><i>{fmtMs(report.ms)} in your browser · {keeps && session ? 'saved to your account (only you can see it)' : 'your edit'}</i></>}
          {!recorded && state === 'done' && !solvedHere && <><b>✓ {report.pass} of {report.pass} tests pass</b><i>{fmtMs(report.ms)} in your browser{edited ? ' · your edit' : ''}</i></>}
          {!recorded && state === 'failed' && kind === 'problem' && <><b>✗ {report.fail} of {report.pass + report.fail} tests to solve</b><i>{fmtMs(report.ms)}{edited ? ' · your edit' : ' · as posted'}</i></>}
          {!recorded && state === 'failed' && kind !== 'problem' && <><b>✗ {report.fail} of {report.pass + report.fail} tests fail</b><i>{fmtMs(report.ms)}{edited ? ' · your edit' : ''}</i></>}
          {!recorded && state === 'error' && <b className="fp-diag">{report.message}</b>}
        </span>
        <span className="fp-strip-right">
          {live && post.votes && !mine && (
            <span className="fp-votes" title={`rating ${post.rating}: up-votes and reviews that confirm the result raise it; down-votes and reviews that find a problem lower it`}>
              <button type="button" className={`fp-vote${post.votes.mine === 1 ? ' on' : ''}`} disabled={voting} onClick={() => vote(1)} aria-label="Vote up: the post's claim is right" aria-pressed={post.votes.mine === 1}>▲ {post.votes.up}</button>
              <button type="button" className={`fp-vote fp-vote-down${post.votes.mine === -1 ? ' on' : ''}`} disabled={voting} onClick={() => vote(-1)} aria-label="Vote down: the post's claim is wrong" aria-pressed={post.votes.mine === -1}>▼ {post.votes.down}</button>
              <span className="fp-rating">{post.rating}</span>
            </span>
          )}
          {live && post.votes && mine && <span className="fp-rating" title="up-votes, down-votes, rating">▲ {post.votes.up} ▼ {post.votes.down} · {post.rating}</span>}
          {live && post.reviews && (
            <button type="button" className="fp-chip" onClick={() => showPanel('reviews')}>
              {reviewsLabel(post.reviews, unit)}
            </button>
          )}
          {post.forks > 0 && <a className="fp-chip" href={`/p/${post.slug}#forks`}>{post.forks} fork{post.forks === 1 ? '' : 's'}</a>}
          {kind === 'problem' && <a className="fp-chip" href={`/p/${post.slug}#solutions`} title={post.hiddenSolutions ? 'the answers are hidden until you have solved it' : undefined}>
            {(() => {
              const n = (post.solutions ? post.solutions.length : 0) + (post.hiddenSolutions || 0);
              if (!n) return 'no solutions yet';
              return `${n} solution${n === 1 ? '' : 's'}${post.hiddenSolutions ? ' · hidden' : ''}`;
            })()}
          </a>}
          {kind === 'problem' && live && post.solvedBy > 0 && <span className="fp-chip fp-chip-quiet" title="people whose own solution passes its tests">{post.solvedBy} solved it</span>}
          {post.tags.map((t) => <a key={t} className="fp-tag" href={`/posts#tag=${encodeURIComponent(t)}`}>#{t.replace(/\s+/g, '-')}</a>)}
        </span>
      </div>
      {state === 'failed' && kind !== 'problem' && (
        <ul className="fp-failing">
          {report.failing.slice(0, 4).map((t) => <li key={t}>✗ <TexLine text={t} /></li>)}
        </ul>
      )}

      <div className="fp-tools">
        {!page && (
          <a href={`/p/${post.slug}`} className={kind === 'problem' ? 'fp-read fp-read-quiet' : 'fp-read'} title={notebook ? 'the post as an article: prose, figures and results, with the code folded; also as a notebook, the program, or the file' : 'the post on its own page: the prose and results, the program, or the file'}>
            {kind === 'article' ? 'Read the article' : kind === 'ladder' ? 'Read the ladder' : 'Read'}
          </a>
        )}
        {kind === 'problem' && !solvedHere && (
          <button type="button" className="fp-read fp-read-button" onClick={resume}>{hasSaved ? 'Open your work' : 'Solve it'}</button>
        )}
        {kind === 'problem' && solvedHere && !PUBLIC_SITE && (
          <a href={`/compose?solve=${encodeURIComponent(post.slug)}${post.file ? `&file=${encodeURIComponent(post.file)}` : ''}`} className="fp-read">Post it publicly</a>
        )}
        <button type="button" onClick={() => { setScale('browser'); run(); }} disabled={state === 'running'} className={!page ? 'fp-run fp-run-quiet' : 'fp-run'} title={post.recorded && post.recorded.length ? 'runs the program at the size written for the browser' : kind === 'ladder' ? 'runs the top rung, which is the whole program' : undefined}>
          <svg viewBox="0 0 24 24" width="12" height="12" fill="currentColor" aria-hidden="true"><path d="M7 4.5v15l12-7.5z" /></svg>
          {state === 'running' ? 'Running' : runs ? 'Run again' : 'Run'}
        </button>
        <button type="button" onClick={toggleCode} aria-expanded={codeOpen}>{codeOpen ? 'Hide code' : 'Show code'}</button>
        {live && (
          <button type="button" onClick={() => showPanel('reviews')} aria-expanded={reviewsOpen}>
            {post.reviews && post.reviews.n ? `Reviews · ${post.reviews.n}` : 'Review'}
          </button>
        )}
        {!PUBLIC_SITE && <a href={forkHref}>Fork</a>}
        {mine ? <a href={`/compose?edit=${encodeURIComponent(post.slug)}`}>Edit</a> : (
          !live && <a href={`/workbench?open=posts/${post.file}`}>Open in the workbench</a>
        )}
        {live && post.likes && (
          <button type="button" onClick={like} disabled={liking} className={`fp-like${post.likes.mine ? ' on' : ''}`} aria-pressed={post.likes.mine} aria-label={post.likes.mine ? 'Unlike' : 'Like'}>
            <svg viewBox="0 0 24 24" width="14" height="14" fill={post.likes.mine ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2" strokeLinejoin="round" aria-hidden="true"><path d="M12 20.5s-7.5-4.6-9.3-9.2C1.4 8 3.6 4.5 7.2 4.5c2 0 3.5 1.1 4.8 2.7 1.3-1.6 2.8-2.7 4.8-2.7 3.6 0 5.8 3.5 4.5 6.8C19.5 15.9 12 20.5 12 20.5z" /></svg>
            {post.likes.n || ''}
          </button>
        )}
        <button type="button" onClick={copyLink} className="fp-tool-quiet">{copied ? 'Link copied' : 'Copy link'}</button>
      </div>
      {inside && <LookInside links={inside} compact />}

      {codeOpen && code !== null && (
        <div className="fp-code">
          <div className="fp-code-head">
            <span>{(post.file || `${post.slug}.viv`).replace(/^\d{4}-\d{2}-\d{2}-/, '')} · {code.split('\n').length} lines · editable</span>
            {notebook && !page && <a href={`/p/${post.slug}?view=notebook`} title="the same file with its prose rendered between the code cells">Read it as a notebook</a>}
            {edited && <button type="button" onClick={() => setCode(original)}>Reset</button>}
          </div>
          <div className="fp-code-scroll">
            <div className="fp-editor">
              <pre aria-hidden="true" dangerouslySetInnerHTML={{ __html: html }} />
              <textarea ref={taRef} value={code} spellCheck="false" aria-label={`Source of ${post.title}`}
                onChange={(e) => setCode(e.target.value)}
                onKeyDown={(e) => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); run(); } }} />
            </div>
          </div>
          {uses.length > 0 && (
            <div className="fp-uses">
              Uses {uses.map((u, i) => (
                <span key={u}>{i ? ', ' : ''}{useHref(u) ? <a href={useHref(u)}>{u}</a> : <code>{u}</code>}</span>
              ))}: {uses.length === 1 ? 'its code is inserted where its' : 'each one\'s code is inserted where its'} import line is, and Run runs the combined program.
            </div>
          )}
          <div className="fp-code-foot">
            {kind === 'problem' ? (
              <>
                Fill in the functions and press Run (⌘/Ctrl + Enter).
                {keeps && session && <> Your work is saved to your account, where only you can see it. Posting it is a separate step.</>}
                {keeps && !session && <> <a href={`/account?next=/p/${post.slug}`}>Sign in</a> to save your work to your account; without that, it is lost when you close this page.</>}
                {keeps && session && edited && <button type="button" className="fp-savework" disabled={saving} onClick={() => keepWork()}>{saving ? 'Saving…' : 'Save my work'}</button>}
              </>
            ) : 'Change a number and press Run (⌘/Ctrl + Enter). The tests run too.'}
          </div>
        </div>
      )}
      {live && reviewsOpen && (
        <Reviews slug={post.slug} session={session} lastRun={lastRun} currency={currency} onWallet={onWallet} />
      )}
    </article>
  );
}
