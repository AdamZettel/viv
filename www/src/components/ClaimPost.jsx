// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { withModules, parseOwnOutput, Chart, ExpectedCompare, comparableOutput } from './VivCell.jsx';
import { highlight } from '../lib/highlight.js';
import FigureBox from './FigureBox.jsx';
import { vivModule } from '../lib/vivWasm.js';
import { atlasHref } from '../lib/atlasLink.js';
import { FIGURE_LINE } from '../lib/postCheck.js';
import { sciNotation } from '../lib/richText.js';

// One claim as a post (kind `claim`, scripts/feed-build.mjs): a statement,
// specific and bounded, checked by a Viv program and, where vipyr has the API,
// its vipyr twin (claims/<topic>/<slug>.viv and .py, tools/claims/run.py).
//
// Two forms. The CARD (the feed, the claims channel): its title, the line
// with its key numbers, and the figure the program draws. The PAGE (`page`):
// the title, the line, the figure, a table of the numbers the tests check,
// the program with its output and checks, then the full statement, its scope
// and its reference. Run runs the Viv program in this tab, redraws the figure
// from that run and marks whether the output matches the recorded run's. The
// page passes the sources and recorded outputs in as `full`
// (/programs/claims/<slug>.json).

const STATUS_LABEL = { checked: 'checked', mismatch: 'mismatch', stale: 'stale', unrun: 'not run' };
const fmtMs = (ms) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);
const esc = (s) => String(s).replace(/[&<>]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[ch]));
// a value a program printed (4.3e7), set as a power of ten
const Num = ({ v }) => <span className="clp-num" dangerouslySetInnerHTML={{ __html: sciNotation(esc(v)) }} />;
// the title, line, statement, scope and reference are typeset at build time (KaTeX)
const Html = ({ html, text, as: Tag = 'span', className }) => (html ? <Tag className={className} dangerouslySetInnerHTML={{ __html: html }} /> : <Tag className={className}>{text}</Tag>);
// a claim's run happens in a worker (src/lib/claimRun.worker.js): some take
// minutes in the interpreter, and the page must stay usable and stoppable
let runSerial = 0;
function runInWorker(worker, src) {
  const id = ++runSerial;
  return vivModule().then((m) => new Promise((resolve, reject) => {
    worker.onmessage = (e) => {
      if (e.data.id !== id) return;
      if (e.data.error) reject(new Error(e.data.error));
      else resolve(e.data);
    };
    worker.onerror = (e) => reject(new Error(e.message || 'the run stopped'));
    worker.postMessage({ id, src, vivModule: m });
  }));
}
const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
// the output a reader reads: no figure records, no engine records
const textOutput = (out) => comparableOutput(String(out || '').split('\n').filter((l) => !FIGURE_LINE.test(l)).join('\n'));
const quantity = (name) => name.replace(/_/g, ' ');
const longDate = (d) => {
  if (!d) return '';
  const t = new Date(`${d}T12:00:00Z`);
  return Number.isNaN(t.getTime()) ? d : t.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC' });
};

function Meta({ post, c }) {
  return (
    <p className="clp-meta">
      <a href={`/c/claims#${c.topic}`}>{c.topicTitle || c.topic}{c.section ? ` · ${c.section}` : ''}</a>
      <span className={`clp-st clp-st-${c.status}`}>{c.status === 'checked' ? `✓ checked ${longDate(c.ran)}` : STATUS_LABEL[c.status] || c.status}</span>
      {post.agent && <span className="clp-prov" title={`Written with an AI agent (${post.agent}); not yet reviewed by a person`}>agent-written</span>}
    </p>
  );
}

function Card({ post, c, charts, boxes }) {
  const href = `/p/${post.slug}`;
  const vt = c.vivTests;
  return (
    <article className="clp-card" id={post.slug}>
      <Meta post={post} c={c} />
      <h3 className="clp-card-title"><a href={href}><Html html={post.titleHtml} text={post.title} /></a></h3>
      {c.lineHtml && <Html as="p" className="clp-card-line" html={c.lineHtml} text={c.line} />}
      {charts.length > 0 && (
        <a className="clp-thumb" href={href} aria-label={`The claim ${post.titleText || post.title}`}>
          {boxes && boxes[0] ? <FigureBox box={boxes[0]} /> : <Chart chart={charts[0]} />}
        </a>
      )}
      <p className="clp-card-foot">
        <span>{plural(vt ? vt.pass : 0, 'test')}{c.vivOnly ? ' · Viv' : ' · Viv and vipyr agree'}</span>
        <a href={href}>The program and its numbers</a>
      </p>
    </article>
  );
}

export default function ClaimPost({ post, page = false, full = null, boxes = null }) {
  const c = post.claim;
  const built = useMemo(() => parseOwnOutput(post.figure || '').charts, [post.figure]);
  if (!page) return <Card post={post} c={c} charts={built} boxes={boxes} />;
  return <Page post={post} c={c} built={built} full={full} />;
}

function Page({ post, c, built, full }) {
  const hasTwin = !c.vivOnly;
  const heavy = c.cost === 'heavy';
  const data = full;
  const [tab, setTab] = useState('viv');
  const [code, setCode] = useState(full ? full.viv.source : '');
  const [charts, setCharts] = useState(built);
  const [state, setState] = useState('idle'); // idle | running | done | failed | error
  const [report, setReport] = useState(null);
  const taRef = useRef(null);

  const worker = useRef(null);
  const stopRef = useRef(null);
  useEffect(() => () => worker.current && worker.current.terminate(), []);
  const stop = useCallback(() => {
    if (worker.current) worker.current.terminate();
    worker.current = null;
    if (stopRef.current) stopRef.current();
  }, []);

  const run = useCallback(async () => {
    setState('running');
    setTab('viv');
    try {
      // a module beside the claim (claims/math/qrlib.viv) resolves from its folder
      const src = await withModules(code, c.file ? c.file.replace(/\.viv$/, '') : null);
      if (!worker.current) worker.current = new Worker(new URL('../lib/claimRun.worker.js', import.meta.url), { type: 'module' });
      const stopped = new Promise((resolve) => { stopRef.current = () => resolve(null); });
      const r = await Promise.race([runInWorker(worker.current, src), stopped]);
      stopRef.current = null;
      if (!r) { setReport(null); setState('idle'); return; }
      const { out, ms } = r;
      const lines = out.split('\n');
      const diagnostic = lines.find((l) => /^error\b/.test(l));
      if (diagnostic) { setReport({ message: diagnostic }); setState('error'); return; }
      const failing = lines.filter((l) => l.startsWith('FAIL ')).map((l) => l.slice(5).split(' -- ')[0]);
      const drawn = parseOwnOutput(out).charts;
      if (drawn.length) setCharts(drawn);
      const text = textOutput(out);
      setReport({ fail: failing.length, failing, ms, out: text, same: text === textOutput(data.viv.output), edited: code !== data.viv.source });
      setState(failing.length ? 'failed' : 'done');
    } catch (e) {
      setReport({ message: String(e.message || e) });
      setState('error');
    }
  }, [code, data, c.file]);

  const vivHtml = useMemo(() => `${highlight(code, 'viv')}\n`, [code]);
  const pyHtml = useMemo(() => (data && data.vipyr ? highlight(data.vipyr.source.replace(/\n$/, ''), 'python') : ''), [data]);
  useEffect(() => {
    const ta = taRef.current;
    if (ta) { ta.style.height = 'auto'; ta.style.height = `${ta.scrollHeight + 2}px`; }
  }, [code, tab]);
  const py = tab === 'vipyr' && hasTwin;
  const side = data && (py ? data.vipyr : data.viv);
  const live = !!(report && report.out && !py);
  const outLines = (live ? report.out : textOutput(side ? side.output : '')).split('\n').filter(Boolean);
  const values = outLines.filter((l) => !/^(PASS|FAIL) /.test(l) && !/^\d+ test\(s\) FAILED/.test(l));
  const checks = outLines.filter((l) => /^(PASS|FAIL) /.test(l));
  const api = c.links.filter((l) => l.kind === 'api');
  const map = c.links.find((l) => l.kind === 'map');
  const others = c.links.filter((l) => l.kind === 'gallery' || l.kind === 'page');

  return (
    <article className="clp-doc">
      <Meta post={post} c={c} />
      <h1 className="clp-title"><Html html={post.titleHtml} text={post.title} /></h1>
      {c.lineHtml && <Html as="p" className="clp-line" html={c.lineHtml} text={c.line} />}

      {charts.length > 0 && (
        <figure className="clp-fig">
          {charts.map((ch, i) => <Chart key={`${live ? 'run' : 'rec'}-${i}`} chart={ch} />)}
          {live && <figcaption>Redrawn from your run.</figcaption>}
        </figure>
      )}

      {c.table && c.table.length > 0 && (
        <div className="clp-table-wrap">
          <table className="clp-table">
            <thead>
              <tr><th scope="col">Quantity</th><th scope="col">Viv</th>{hasTwin && <th scope="col">vipyr</th>}<th scope="col">Checked by</th></tr>
            </thead>
            <tbody>
              {c.table.map((r) => (
                <tr key={r.name}>
                  <th scope="row"><code>{quantity(r.name)}</code></th>
                  <td data-k="Viv"><Num v={r.viv} /></td>
                  {hasTwin && <td data-k="vipyr">{r.vipyr ? <Num v={r.vipyr} /> : <span className="clp-dim">not shown</span>}</td>}
                  <td className="clp-by">{r.tests.length ? r.tests.join('; ') : <span className="clp-dim">Viv and vipyr agree</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {c.whyHtml && <Html as="p" className="clp-why" html={c.whyHtml} />}

      <section className="clp-prog" aria-label="The program">
        <div className="clp-switch" role="tablist" aria-label="Which program">
          <button type="button" role="tab" aria-selected={!py} onClick={() => setTab('viv')}>Viv</button>
          {hasTwin && <button type="button" role="tab" aria-selected={py} onClick={() => setTab('vipyr')} title="the same quantities through the vipyr API, from Python">vipyr</button>}
          <span className="clp-file">{((py ? c.twin : c.file) || '').split('/').pop()}</span>
          {!py && data && code !== data.viv.source && <button type="button" className="clp-reset" onClick={() => setCode(data.viv.source)}>Reset</button>}
        </div>
        {!py && data && (
          <div className="fp-code-scroll clp-code">
            <div className="fp-editor">
              <pre aria-hidden="true" dangerouslySetInnerHTML={{ __html: vivHtml }} />
              <textarea ref={taRef} value={code} spellCheck="false" aria-label={`The Viv program of the claim ${post.slug}`} readOnly={heavy}
                onChange={(e) => setCode(e.target.value)}
                onKeyDown={(e) => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); if (!heavy) run(); } }} />
            </div>
          </div>
        )}
        {py && data && data.vipyr && <pre className="clp-py"><code dangerouslySetInnerHTML={{ __html: pyHtml }} /></pre>}
        <div className="clp-out">
          <p className="clp-out-head">
            {live
              ? <>Your run · {fmtMs(report.ms)}{report.edited ? ' · your edit' : ''} · {report.same ? 'the same output as the recorded run' : 'differs from the recorded run'}</>
              : <>Output · {py ? 'Python with vipyr' : 'native build'} · {longDate(c.ran)}{side && side.seconds != null ? ` · ${side.seconds} s` : ''}</>}
          </p>
          {values.length > 0 && <pre className="clp-values">{values.join('\n')}</pre>}
          {checks.length > 0 && (
            <ul className="clp-checks">
              {checks.map((l) => <li key={l} className={l.startsWith('PASS') ? 'ok' : 'bad'}>{l.startsWith('PASS') ? '✓' : '✗'} {l.slice(5).split(' -- ')[0]}</li>)}
            </ul>
          )}
          {live && !report.same && <ExpectedCompare actual={report.out} expected={textOutput(data.viv.output)} expectedLabel="recorded" />}
          {state === 'error' && <p className="clp-diag">{report.message}</p>}
        </div>
      </section>

      <p className="clp-actions">
        {state === 'running' ? (
          <button type="button" onClick={stop} className="clp-run" title="stops the run; the recorded output stays">
            <svg viewBox="0 0 24 24" width="11" height="11" fill="currentColor" aria-hidden="true"><path d="M6 6h12v12H6z" /></svg>
            Stop
          </button>
        ) : (
          <button type="button" onClick={run} disabled={heavy || !data} className="clp-run" title={heavy ? 'too heavy to run in a browser; its recorded run is from the nightly check' : 'runs the Viv program in your browser and redraws the figure'}>
            <svg viewBox="0 0 24 24" width="11" height="11" fill="currentColor" aria-hidden="true"><path d="M7 4.5v15l12-7.5z" /></svg>
            {live ? 'Run again' : 'Run'}
          </button>
        )}
        <a href={`/workbench?open=${c.file}`}>Open in the workbench</a>
        {atlasHref(c.file) && <a href={atlasHref(c.file)}>Atlas</a>}
        {map ? <a href={map.href}>Code map</a> : <a href={`/c/claims#${c.topic}`}>More {c.topicTitle ? c.topicTitle.toLowerCase() : ''} claims</a>}
      </p>
      {api.length > 0 && (
        <p className="clp-uses">Uses {api.map((l, i) => <span key={l.href}>{i ? ', ' : ''}<a href={l.href}><code>{l.label}</code></a></span>)}</p>
      )}
      {others.length > 0 && (
        <p className="clp-uses">See also {others.map((l, i) => <span key={l.href}>{i ? ', ' : ''}<a href={l.href}>{l.label}</a></span>)}</p>
      )}

      <dl className="clp-about">
        <div><dt>Full statement</dt><Html as="dd" html={c.statementHtml} text={c.statement} /></div>
        {c.scope && <div><dt>Scope</dt><Html as="dd" html={c.scopeHtml} text={c.scope} /></div>}
        {c.reference && <div><dt>Reference</dt><Html as="dd" html={c.referenceHtml} text={c.reference} /></div>}
        {c.citedBy && c.citedBy.length > 0 && <div><dt>Cited by</dt><dd>{c.citedBy.map((a, i) => <span key={a.slug}>{i ? ', ' : ''}<a href={`/p/${a.slug}`}>{a.title}</a></span>)}</dd></div>}
      </dl>

      <p className="clp-foot">
        <span>{c.file}{c.twin ? ` · ${c.twin}` : ''}</span>
        <span title="a hash of the claim's programs as recorded">#{post.address}</span>
      </p>
    </article>
  );
}
