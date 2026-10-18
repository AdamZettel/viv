// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Chart, parseOwnOutput } from './VivCell.jsx';
import { useCellRunner, ProseCell, CodeCell, OutText, headerMarkdown } from './Notebook.jsx';
import { splitNotebook, docHeader, joinCells } from '../lib/notebook.js';
import { VIEWS, initialView, codeSummary, summaryText, testNames, testResults, rungSegments, codePieces, programText, splitAtFigure } from '../lib/postViews.js';
import { highlight } from '../lib/highlight.js';
import { lookInsideLinks } from '../lib/atlasLink.js';
import LookInside from './LookInside.jsx';
import { renderProse, loadKatex } from '../lib/richText.js';
import '../styles/notebook.css';
import '../styles/postviews.css';

// A post's page: one file, four ways to read it (src/lib/postViews.js).
//
//   Article    the prose, with each cell's figures and printed results where
//              the cell makes them; code folded to one line that opens in
//              place; tests as a list of names that pass or fail. Run all
//              runs the cells here, and the outputs replace the build's.
//   Notebook   the prose and the code cells, editable, each with its Run.
//   Code       the program alone, prose and `#|` docs folded to a line each.
//   Source     the file as stored.
//
// Every view reads the same cells (splitNotebook), and the Article and
// Notebook views share one runner, so an edit in the notebook shows in the
// article and a run in either lands in both.
//
// props:
//   src          the file, as stored
//   post         the card from feed.json (title, kind, tests, failed, ladder, ...)
//   author       { name, handle }
//   dateLabel    the date as the page shows it
//   proseHtml    { [cell index]: html } rendered at build time
//   cellOut      { [cell index]: what that cell printed in the build's run }
//   expectError  why the interpreter cannot run this file, or ''
//   explain      { slug, title } when the explorer can open this program

const STORE = 'viv.postView';
// Math in code: the `$...$` in a comment of highlighted code, typeset by KaTeX,
// so a formula in a `#|` note reads as math and not as TeX. Only comment spans
// are touched; without KaTeX the html is returned as it came.
// Test names and comments also carry prices ("$1000 ... is $1628.89"), so text is
// typeset only when every `$...$` in it reads as math by Pandoc's rule: no space
// just inside either dollar, and no digit right after the closing one.
const TEX_SPAN = /\$\$[^$]+\$\$|\$[^$]+\$/g;
const looksLikeMath = (raw) => {
  const s = String(raw).replace(/\\\$/g, '');
  const spans = [...s.matchAll(TEX_SPAN)];
  if (!spans.length || s.replace(TEX_SPAN, '').includes('$')) return false;
  return spans.every((m) => !/^\$+\s|\s\$+$/.test(m[0]) && !/\d/.test(s[m.index + m[0].length] || ''));
};
const textHtml = (s, katex) => (katex && looksLikeMath(s) ? renderProse(s, katex) : s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])));
const unesc = (s) => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
function mathInCode(html, katex) {
  if (!katex || !html.includes('$')) return html;
  return html.replace(/(<span class="tk-(?:doc|com)">)([^<]*)(<\/span>)/g, (all, open, body, close) => (
    looksLikeMath(unesc(body)) ? open + renderProse(unesc(body), katex) + close : all));
}
const fmtSecs = (s) => (s < 0.1 ? `${Math.max(1, Math.round(s * 1000))} ms` : `${s.toFixed(1)} s`);

function ViewSwitch({ view, setView }) {
  const refs = useRef({});
  const onKey = (e) => {
    const i = VIEWS.findIndex((v) => v.id === view);
    let j = null;
    if (e.key === 'ArrowRight') j = (i + 1) % VIEWS.length;
    else if (e.key === 'ArrowLeft') j = (i + VIEWS.length - 1) % VIEWS.length;
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = VIEWS.length - 1;
    if (j === null) return;
    e.preventDefault();
    setView(VIEWS[j].id);
    refs.current[VIEWS[j].id]?.focus();
  };
  return (
    <div className="pv-switch" role="tablist" aria-label="How to read this post" onKeyDown={onKey}>
      {VIEWS.map((v) => (
        <button key={v.id} ref={(el) => { refs.current[v.id] = el; }} type="button" role="tab" id={`pv-tab-${v.id}`}
          aria-selected={view === v.id} aria-controls="pv-panel" tabIndex={view === v.id ? 0 : -1} title={v.hint}
          onClick={() => setView(v.id)}>
          {v.label}
        </button>
      ))}
    </div>
  );
}

/** A list of test names, each marked: passed, failed, or not run. */
function Tests({ names, status, label = 'Tests', katex = null }) {
  if (!names.length) return null;
  const n = names.filter((t) => status[t]?.ok).length;
  const bad = names.filter((t) => status[t] && !status[t].ok).length;
  return (
    <div className="pv-tests">
      <p className="pv-tests-head">{label} <span>{bad ? `${bad} of ${names.length} fail` : n === names.length ? `${n} of ${names.length} pass` : `${names.length}`}</span></p>
      <ul>
        {names.map((t) => {
          const s = status[t];
          const cls = !s ? 'idle' : s.ok ? 'pass' : 'fail';
          return (
            <li key={t} className={`pv-t pv-t-${cls}`} data-test={cls}>
              <span className="pv-t-mark" aria-hidden="true">{!s ? '·' : s.ok ? '✓' : '✗'}</span>
              <span><span dangerouslySetInnerHTML={{ __html: textHtml(t, katex) }} />{s && !s.ok && s.why ? <small> {s.why}</small> : null}</span>
              <span className="sr-only">{!s ? ' (not run)' : s.ok ? ' (passes)' : ' (fails)'}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/** Code folded to one line that opens in place. */
function Fold({ code, summary, label = 'Code', first, katex = null }) {
  const [open, setOpen] = useState(false);
  const html = useMemo(() => (open ? mathInCode(highlight(code, 'viv'), katex) : ''), [open, code, katex]);
  return (
    <div className={`pv-fold${open ? ' open' : ''}`}>
      <button type="button" className="pv-fold-btn" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <span className="pv-fold-tri" aria-hidden="true">{open ? '▾' : '▸'}</span>
        <span className="pv-fold-what">{label}</span>
        <span className="pv-fold-sum">{summary}</span>
        <span className="pv-fold-act">{open ? 'Hide' : 'Show'}</span>
      </button>
      {open && (
        <pre className="pv-pre" tabIndex={0} aria-label={`${label}${first ? `, from line ${first}` : ''}`}><code dangerouslySetInnerHTML={{ __html: html }} /></pre>
      )}
    </div>
  );
}

/** What a cell printed, less its PASS and FAIL lines (the test list shows those). */
function Printed({ parsed, error, skipFigures = false }) {
  const blocks = parsed.blocks
    .map((b) => (b.type === 'narrate' ? b : { ...b, s: b.s.split('\n').filter((l) => !/^(PASS|FAIL) /.test(l)).join('\n') }))
    .filter((b) => b.type === 'narrate' || b.s.trim());
  if (!blocks.length && (skipFigures || !parsed.charts.length)) return null;
  return (
    <div className="pv-out">
      {blocks.length > 0 && <p className="pv-out-head">{error ? 'Error' : 'Printed'}</p>}
      {blocks.length > 0 && <OutText blocks={blocks} error={error} />}
      {!error && !skipFigures && parsed.charts.map((c, i) => <figure className="pv-fig" key={i}><Chart chart={c} /></figure>)}
    </div>
  );
}

// ---- Article -------------------------------------------------------------------

function useKatex(need) {
  const [k, setK] = useState(null);
  useEffect(() => { if (need) loadKatex().then(setK); }, [need]);
  return k;
}

/** One code cell in the article: its output, its tests, its code folded. */
function ArticleCode({ cell, idx, number, out, live, ladder, problem, katex, figureElsewhere }) {
  const head = useMemo(() => docHeader(cell.code), [cell.code]);
  const headMd = useMemo(() => (head.doc ? headerMarkdown(head) : ''), [head]);
  const code = head.body.replace(/\n+$/, '');
  const first = cell.codeStart + (head.lines || 0);
  const parsed = out ? out.parsed : null;
  const status = out ? testResults(out.text) : {};
  const segs = useMemo(() => (ladder ? rungSegments(code) : null), [ladder, code]);
  const hasRungs = segs && segs.some((s) => s.kind === 'rung');
  const printed = parsed && (
    <Printed parsed={parsed} error={out.error} skipFigures={figureElsewhere} />
  );
  if (hasRungs) {
    let n = 0;
    return (
      <section className="pv-cell" data-cell={number}>
        {headMd && <ProseCell body={headMd} />}
        {segs.map((s, i) => {
          if (s.kind === 'code') {
            const t = s.text.replace(/^\n+|\n+$/g, '');
            if (!t.trim()) return null;
            return <Fold key={i} code={t} summary={summaryText(codeSummary(t))} label="Set-up" katex={katex} />;
          }
          n += 1;
          const r = ladder.rungs[n - 1] || {};
          const paras = String(s.note).split(/\n\s*\n/).map((p) => p.replace(/\s*\n\s*/g, ' ').trim()).filter(Boolean);
          const whyHtml = r.whyHtml !== undefined && r.title === s.title ? [r.whyHtml, ...(r.notesHtml || [])] : paras.map((p) => renderProse(p, katex));
          const names = testNames(s.body);
          const sum = codeSummary(s.body);
          return (
            <section className="pv-rung" id={`rung-${n}`} key={i}>
              <h2><span className="pv-rung-n" aria-hidden="true">{n}</span><span dangerouslySetInnerHTML={{ __html: r.titleHtml && r.title === s.title ? r.titleHtml : renderProse(s.title, katex) }} /></h2>
              {whyHtml.map((h, k) => <p key={k} className={k ? 'pv-rung-note' : 'pv-rung-why'} dangerouslySetInnerHTML={{ __html: h }} />)}
              {(r.added?.length > 0 || r.edited?.length > 0) && (
                <p className="pv-rung-ledger">
                  {r.added?.length > 0 && <>new: {r.added.map((f, k) => <Fragment key={f.name}>{k ? ', ' : ''}<code>{f.name}</code></Fragment>)}</>}
                  {r.added?.length > 0 && r.edited?.length > 0 && ' · '}
                  {r.edited?.length > 0 && <>edited: {r.edited.map((f, k) => <Fragment key={f.name}>{k ? ', ' : ''}<code>{f.name}</code></Fragment>)}</>}
                </p>
              )}
              <Tests names={names} status={status} label={`Rung ${n} tests`} katex={katex} />
              <Fold code={s.body} summary={summaryText(sum, { withTests: false })} label={`Rung ${n} code`} katex={katex} />
            </section>
          );
        })}
        {printed}
      </section>
    );
  }
  const sum = codeSummary(code);
  return (
    <section className="pv-cell" data-cell={number}>
      {headMd && <div className="pv-dochead"><ProseCell body={headMd} /></div>}
      {printed}
      {!problem && <Tests names={sum.tests} status={status} katex={katex} />}
      {code.trim() && <Fold code={code} summary={summaryText(sum)} label="Code" first={first} katex={katex} />}
      {live && <span className="sr-only">This output is from your run.</span>}
    </section>
  );
}

/** A problem's statement: the functions to write, and the tests to make pass. */
function Task({ cells, status, katex, onSolve, slug }) {
  const defs = [];
  for (const c of cells) {
    if (c.kind !== 'code') continue;
    const lines = c.code.split('\n');
    lines.forEach((l, i) => {
      const m = l.match(/^def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)/);
      if (!m) return;
      const doc = [];
      for (let k = i - 1; k >= 0 && /^\s*#\|/.test(lines[k]); k--) doc.unshift(lines[k].replace(/^\s*#\|\s?/, ''));
      defs.push({ name: m[1], args: m[2], doc: doc.join(' ') });
    });
  }
  const names = cells.filter((c) => c.kind === 'code').flatMap((c) => testNames(c.code));
  return (
    <section className="pv-task" aria-labelledby="pv-task-h">
      <h2 id="pv-task-h">Your task</h2>
      {defs.length > 0 && (
        <>
          <p className="pv-task-lede">Write {defs.length === 1 ? 'this function' : `these ${defs.length} functions`}:</p>
          <dl className="pv-task-defs">
            {defs.map((d) => (
              <div key={d.name}><dt><code>{d.name}({d.args})</code></dt>{d.doc && <dd dangerouslySetInnerHTML={{ __html: renderProse(d.doc, katex) }} />}</div>
            ))}
          </dl>
        </>
      )}
      <Tests names={names} status={status} label="Make these tests pass" katex={katex} />
      <p className="pv-task-act">
        <button type="button" className="nb-runall" onClick={onSolve}>Solve it in the notebook</button>
      </p>
    </section>
  );
}

function ArticleView({ cells, proseHtml, outs, post, katex, onSolve, liveAny }) {
  const headerIdx = cells.findIndex((c) => c.kind === 'header');
  const [before, after] = splitAtFigure(headerIdx >= 0 ? proseHtml[headerIdx] : '');
  // an article's `[figure]`: the first cell that draws puts its figures there
  const figIdx = after !== null ? cells.findIndex((c, i) => c.kind === 'code' && outs[i]?.parsed?.charts.length) : -1;
  const problem = post.kind === 'problem';
  const allStatus = useMemo(() => Object.assign({}, ...Object.values(outs).map((o) => testResults(o.text))), [outs]);
  let number = 0;
  return (
    <div className="pv-article">
      {before && <div className="nb-prose pv-prose" dangerouslySetInnerHTML={{ __html: before }} />}
      {figIdx >= 0 && (
        <figure className="pv-fig pv-fig-lead">{outs[figIdx].parsed.charts.map((c, i) => <Chart key={i} chart={c} />)}</figure>
      )}
      {after && <div className="nb-prose pv-prose" dangerouslySetInnerHTML={{ __html: after }} />}
      {/* a problem reads problem-first: its statement, then the task */}
      {problem && <Task cells={cells} status={allStatus} katex={katex} onSolve={onSolve} slug={post.slug} />}
      {cells.map((c, i) => {
        if (c.kind === 'header') return null;
        if (c.kind === 'prose') return <div key={i} className="nb-prose pv-prose" dangerouslySetInnerHTML={{ __html: proseHtml[i] || '' }} />;
        if (c.empty) return null;
        number += 1;
        return (
          <ArticleCode key={i} cell={c} idx={i} number={number} out={outs[i]} live={liveAny} katex={katex}
            ladder={post.kind === 'ladder' ? post.ladder || { rungs: [] } : null} problem={problem} figureElsewhere={i === figIdx} />
        );
      })}
    </div>
  );
}

// ---- Code and Source -----------------------------------------------------------

function CodePiece({ p, katex }) {
  const [open, setOpen] = useState(false);
  if (p.kind === 'code') {
    return <pre className="pv-code-run" data-line={p.line}><code dangerouslySetInnerHTML={{ __html: mathInCode(highlight(p.text, 'viv'), katex) }} /></pre>;
  }
  const what = p.what === 'header' ? '###' : p.what === 'prose' ? '### prose' : '#|';
  return (
    <div className={`pv-cfold pv-cfold-${p.what}`}>
      <button type="button" aria-expanded={open} onClick={() => setOpen((o) => !o)} style={p.indent ? { paddingLeft: `calc(10px + ${p.indent.length}ch)` } : undefined}>
        <span aria-hidden="true">{open ? '▾' : '▸'}</span> <span className="pv-cfold-what">{what}</span> <span className="pv-cfold-label">{p.label}</span> <span className="pv-cfold-n">{p.lines} line{p.lines === 1 ? '' : 's'}</span>
      </button>
      {open && (p.what === 'doc' && katex
        ? <pre className="pv-cfold-text" dangerouslySetInnerHTML={{ __html: renderProse(p.text, katex) }} />
        : <pre className="pv-cfold-text">{p.text}</pre>)}
    </div>
  );
}

function useCopy() {
  const [done, setDone] = useState('');
  const copy = useCallback(async (text, what) => {
    try { await navigator.clipboard.writeText(text); setDone(what); } catch { setDone(''); }
    setTimeout(() => setDone(''), 1600);
  }, []);
  return [done, copy];
}

// A run the author recorded at a larger setting (`scale <name>:` in the
// header, its output beside the post as <file>.<name>.out; scripts/feed-build.mjs):
// too slow for a browser, so its figure and tests are shown as recorded.
function RecordedRuns({ recorded, file }) {
  const charts = useMemo(() => recorded.map((r) => parseOwnOutput(r.figure || '').charts), [recorded]);
  if (!recorded.length) return null;
  return recorded.map((r, k) => {
    const sets = Object.entries(r.sets);
    return (
      <section key={r.scale} className="pv-recorded" aria-label={`The recorded ${r.scale} run`}>
        <h2>The {r.scale} run, recorded by the author</h2>
        <p className="fp-recorded-note">
          {sets.map(([a, v]) => `${a} = ${v}`).join(' · ')}{r.note ? ` · ${r.note}` : ''}
          {' · '}<code>viv run{sets.map(([a, v]) => ` --set ${a}=${v}`).join('')} posts/{file}</code>
        </p>
        {charts[k].map((c, i) => <Chart key={i} chart={c} />)}
        {r.tests && r.tests.length > 0 && (
          <ul className="pv-recorded-tests">
            {r.tests.map((t) => <li key={t}>✓ {t}</li>)}
          </ul>
        )}
      </section>
    );
  });
}

function CodeView({ cells, katex }) {
  const pieces = useMemo(() => codePieces(cells), [cells]);
  const [done, copy] = useCopy();
  const prog = useMemo(() => programText(pieces), [pieces]);
  const lines = prog.split('\n').length - 1;
  return (
    <div className="pv-codeview">
      <div className="pv-filebar">
        <span>The program: {lines} lines of code, comments and tests. Prose and <code>#|</code> notes are folded; open one to read it.</span>
        <button type="button" onClick={() => copy(prog, 'code')}>{done === 'code' ? 'Copied' : 'Copy the code'}</button>
      </div>
      <div className="pv-file">
        {pieces.map((p, i) => <CodePiece key={i} p={p} katex={katex} />)}
      </div>
    </div>
  );
}

function SourceView({ src, cells, file }) {
  const [done, copy] = useCopy();
  const html = useMemo(() => cells.map((c) => (c.kind === 'code'
    ? highlight(c.text, 'viv')
    : `<span class="pv-src-${c.kind}">${c.text.replace(/[&<>]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[ch]))}</span>`)).join(''), [cells]);
  const lines = src.split('\n').length - (src.endsWith('\n') ? 1 : 0);
  return (
    <div className="pv-sourceview">
      <div className="pv-filebar">
        <span><code>posts/{file}</code> · {lines} lines · {new Blob([src]).size.toLocaleString('en-US')} bytes</span>
        <span className="pv-filebar-act">
          <button type="button" onClick={() => copy(src, 'src')}>{done === 'src' ? 'Copied' : 'Copy'}</button>
          <a href={`/programs/posts/${file}`} download={file}>Download</a>
        </span>
      </div>
      <pre className="pv-source" tabIndex={0} aria-label={`The file ${file}`}><code dangerouslySetInnerHTML={{ __html: html }} /></pre>
    </div>
  );
}

// ---- the page ------------------------------------------------------------------

export default function PostViews({ src, post, author, dateLabel, proseHtml = {}, cellOut = {}, expectError = '', explain = null }) {
  const [cells, setCells] = useState(() => splitNotebook(src));
  const runner = useCellRunner(cells, setCells, `posts/${post.file.replace(/\.viv$/, '')}`);
  const { results, running, all, runCell, runAll, onStop, onEdit, onKey, taRefFor, codeIdx, numberOf } = runner;

  // SSR renders Article; the reader's choice (the address, then what they chose
  // last) is applied once the page is live
  const [view, setViewState] = useState('article');
  useEffect(() => {
    let stored = null;
    try { stored = localStorage.getItem(STORE); } catch { /* private window */ }
    setViewState(initialView(window.location, stored));
    // a link to #view=code on this same page changes only the hash
    const onHash = () => {
      const h = (window.location.hash.match(/(?:^#|&)view=(\w+)/) || [])[1];
      if (VIEWS.some((v) => v.id === h)) setViewState(h);
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  const setView = useCallback((v) => {
    setViewState(v);
    try { localStorage.setItem(STORE, v); } catch { /* not kept */ }
    try {
      const u = new URL(window.location.href);
      if (v === 'article') u.searchParams.delete('view'); else u.searchParams.set('view', v);
      if (/(?:^#|&)view=/.test(u.hash)) u.hash = '';
      history.replaceState(history.state, '', u.toString());
    } catch { /* the address stays */ }
  }, []);

  // each cell's output: this tab's run when there is a current one, else the build's
  const built = useMemo(() => {
    const o = {};
    for (const [k, text] of Object.entries(cellOut)) o[k] = { text, parsed: parseOwnOutput(text), error: /^error\b/m.test(text), built: true };
    return o;
  }, [cellOut]);
  const outs = useMemo(() => {
    const o = { ...built };
    for (const [k, r] of Object.entries(results)) if (r.parsed && !r.stale) o[k] = { text: r.out, parsed: r.parsed, error: r.error, built: false };
    return o;
  }, [built, results]);
  const liveAny = Object.values(results).some((r) => r.parsed && !r.stale);

  // KaTeX for prose rendered here, and for math in comments and test names
  const needKatex = post.kind === 'problem' || (post.kind === 'ladder' && !(post.ladder && post.ladder.rungs.length))
    || /^[ \t]*(#|test[ \t]+").*\$[^$\n]+\$/m.test(src);
  const katex = useKatex(needKatex);
  const edited = joinCells(cells) !== src;
  const isRunning = running !== null;
  const kind = post.kind || 'demo';
  const canRun = view === 'article' || view === 'notebook';

  return (
    <article className="pv" data-view={view} data-kind={kind} data-nb-tests-fail={kind === 'problem' ? 'yes' : undefined} data-nb-expect-error={expectError ? 'yes' : undefined}>
      <p className="pv-record">
        <span className={`fp-kind fp-kind-${kind}`}>{kind}</span>
        {post.lane && <span className={`fp-lane fp-lane-${post.lane}`}>{post.lane === 'ai' ? 'AI' : post.lane[0].toUpperCase() + post.lane.slice(1)}</span>}
        {post.channel && <a className="fp-channel" href={`/c/${post.channel}`}>{post.channelTitle || post.channel}</a>}
        {post.parent && <a href={`/p/${post.parent.slug}`}>fork of {post.parent.title || post.parent.slug}</a>}
        {post.solves && <a href={`/p/${post.solves}`}>solves {post.problem?.title || post.solves}</a>}
        {post.claims && (
          <a className={`pv-claims${post.claims.failing ? ' pv-claims-bad' : ''}`} href="#claims" title="numbers in the prose that a test checks, and the ones nothing establishes">
            {post.claims.checked}/{post.claims.total} numbers checked{post.claims.failing ? ` · ${post.claims.failing} mismatch` : post.claims.orphan ? ` · ${post.claims.orphan} orphan` : ''}
          </a>
        )}
      </p>
      <header className="nb-head pv-head">
        <h1 className="nb-title">{post.title}</h1>
        <p className="nb-byline">{author.name} · {dateLabel}{post.agent ? ' · written with an agent' : ''}</p>
        {post.equationHtml && <div className="nb-equation" dangerouslySetInnerHTML={{ __html: post.equationHtml }} />}
        {post.summaryHtml && <p className="nb-summary" dangerouslySetInnerHTML={{ __html: post.summaryHtml }} />}
      </header>

      <div className="nb-bar pv-bar" role="toolbar" aria-label="Post">
        <ViewSwitch view={view} setView={setView} />
        <span className="nb-bar-status" aria-live="polite">
          {all && !all.done && isRunning && `Running cell ${all.at} of ${all.total}`}
          {all && all.done && !all.stoppedAt && `✓ ${all.total} cell${all.total === 1 ? '' : 's'} · ${fmtSecs(all.secs)}`}
          {all && all.done && all.stoppedAt && (all.stopped ? `Stopped at cell ${all.stoppedAt}` : `Cell ${all.stoppedAt} failed; the cells after it did not run`)}
        </span>
        {canRun && (isRunning
          ? <button type="button" className="nb-runall nb-stop" onClick={onStop}>Stop</button>
          : <button type="button" className="nb-runall" onClick={runAll} disabled={!!expectError || !codeIdx.length}>Run all</button>)}
      </div>
      <p className="pv-links">
        <a href={`/workbench?open=posts/${post.file}`}>Open in the workbench</a>
        <a href={`/programs/posts/${post.file}`} download={post.file}>Download .viv</a>
        {view === 'article' && <span className="pv-provenance">{liveAny ? 'Outputs below are from your run.' : 'Outputs below are from the site build’s run; Run all runs it again in your browser.'}</span>}
      </p>
      <LookInside links={explain ? lookInsideLinks({ slug: explain.slug }) : lookInsideLinks({ open: `posts/${post.file}` })} />
      {expectError && <p className="nb-expect">{expectError}</p>}

      <div id="pv-panel" role="tabpanel" aria-labelledby={`pv-tab-${view}`} className="pv-panel">
        {view === 'article' && (
          <ArticleView cells={cells} proseHtml={proseHtml} outs={outs} post={post} katex={katex} liveAny={liveAny}
            onSolve={() => { setView('notebook'); requestAnimationFrame(() => document.querySelector('.nb-code textarea')?.focus()); }} />
        )}
        {view === 'article' && post.recorded && <RecordedRuns recorded={post.recorded} file={post.file} />}
        {view === 'notebook' && (
          <div className="nb pv-nb">
            {kind === 'problem' && <p className="nb-expect">This is a problem: its tests fail until the functions in it are written. Edit a cell and run it to try; the cells below it show whether the tests pass.</p>}
            <p className="nb-keys">Shift+Enter in a cell runs it and moves to the next. Running a cell runs every cell above it too, as one program.{edited ? ' You have edited the file here; Source still shows it as stored.' : ''}</p>
            {cells.map((c, i) => {
              if (c.kind === 'header') return c.prose ? <ProseCell key={i} html={proseHtml[i]} body={c.prose} /> : null;
              if (c.kind === 'prose') return <ProseCell key={i} html={proseHtml[i]} body={c.body} />;
              if (c.empty) return null;
              return (
                <CodeCell key={i} cell={c} idx={i} number={numberOf(i)} res={results[i]} running={running === i}
                  onEdit={onEdit} onKey={onKey} taRef={taRefFor(i)} onRun={() => runCell(i)} disabled={isRunning || !!expectError} />
              );
            })}
          </div>
        )}
        {view === 'code' && <CodeView cells={cells} katex={katex} />}
        {view === 'source' && <SourceView src={src} cells={splitNotebook(src)} file={post.file} />}
      </div>
    </article>
  );
}
