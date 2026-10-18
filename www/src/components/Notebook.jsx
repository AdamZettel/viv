// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CellEditor, Chart, parseOwnOutput } from './VivCell.jsx';
import { splitNotebook, withCode, runUnits, joinCells, docHeader } from '../lib/notebook.js';
import { markdown } from '../lib/richText.js';
import { expandUses, hasUses, browserResolver } from '../lib/uses.js';
import '../styles/notebook.css';
import { vivWorker } from '../lib/vivWasm.js';
import LookInside from './LookInside.jsx';

// A .viv file read as a notebook (src/lib/notebook.js says how it is cut into
// cells). Prose cells are rendered; code cells are editable and each has its
// own Run. Running cell N runs cells 1..N as one program, through the
// interpreter's incremental session in a worker (public/nb-worker.js): when the
// cells above are exactly what the session ran last, only cell N executes.
// What a cell prints while it runs -- text, PASS/FAIL lines of its tests,
// figures -- shows under that cell.
//
// props:
//   src          the file's text; or
//   srcUrl       where to fetch it, for files too big to put in the page
//   name         the file's repository path, shown in the bar
//   proseHtml    { [cell index]: html } rendered at build time; prose cells
//                missing from it are rendered here
//   intro        { title, byline, summaryHtml, equationHtml } for a post: shown
//                in place of the header's fields (its prose still shows)
//   expectError  why this file cannot run in the interpreter, or ''
//   links        [{ href, label, download? }] for the bar
//   inside       lookInsideLinks()'s { atlas, explore } for the "Look inside
//                this program" block under the bar, or null
//   problem      true for a problem post, whose tests fail until it is solved
//   onSource     (text) => void, called with the whole file after each edit
//                (the workbench puts it back in its editor)

const CAP = 40; // output lines shown before "Show all"
const FOLD = 400; // a code cell longer than this starts folded
const FOLD_SHOW = 60;


// One worker per notebook; runs are queued in it one at a time.
export function useWorker() {
  const ref = useRef(null);
  const seq = useRef(0);
  const pending = useRef(new Map());
  const start = () => {
    const w = vivWorker('/nb-worker.js');
    w.onmessage = (e) => {
      const p = pending.current.get(e.data.id);
      if (p) { pending.current.delete(e.data.id); p(e.data); }
    };
    w.onerror = (e) => {
      for (const p of pending.current.values()) p({ error: String(e.message || e) });
      pending.current.clear();
    };
    return w;
  };
  useEffect(() => () => { if (ref.current) ref.current.terminate(); }, []);
  const send = useCallback((prefix, cell) => new Promise((resolve) => {
    if (!ref.current) ref.current = start();
    const id = ++seq.current;
    pending.current.set(id, resolve);
    ref.current.postMessage({ id, prefix, cell });
  }), []);
  // Stop: the only way to interrupt wasm is to end its worker. The session
  // goes with it, so the next run rebuilds the cells above.
  const stop = useCallback(() => {
    if (ref.current) ref.current.terminate();
    ref.current = null;
    for (const p of pending.current.values()) p({ stopped: true });
    pending.current.clear();
  }, []);
  return { send, stop };
}

// The text a cell printed, one line per row: PASS and FAIL lines marked,
// the rest as printed. Capped at CAP lines until the reader asks for all.
export function OutText({ blocks, error }) {
  const [all, setAll] = useState(false);
  const rows = [];
  for (const b of blocks) {
    if (b.type === 'narrate') rows.push({ kind: 'narrate', s: b.s });
    else for (const l of b.s.split('\n')) rows.push({ kind: error && /^error\b/.test(l) ? 'error' : /^PASS\b/.test(l) ? 'pass' : /^FAIL\b/.test(l) ? 'fail' : 'text', s: l });
  }
  if (!rows.length) return null;
  const shown = all ? rows : rows.slice(0, CAP);
  return (
    <div className="nb-text">
      {shown.map((r, i) => (
        <div key={i} className={`nb-line nb-l-${r.kind}`}>{r.kind === 'narrate' ? `▸ ${r.s}` : r.s || ' '}</div>
      ))}
      {rows.length > CAP && (
        <button type="button" className="nb-more" onClick={() => setAll((a) => !a)}>
          {all ? 'Show the first ' + CAP + ' lines' : `Show all ${rows.length} lines`}
        </button>
      )}
    </div>
  );
}

export function Output({ res }) {
  if (!res) return null;
  if (res.stopped) return <div className="nb-out"><div className="nb-note">Stopped.</div></div>;
  const parsed = res.parsed;
  const empty = !parsed.blocks.length && !parsed.charts.length;
  return (
    <div className={`nb-out${res.stale ? ' nb-out-stale' : ''}`} aria-live="polite">
      {res.stale && <div className="nb-note">A cell above this one, or this one, changed since this ran. Run it again to update the output.</div>}
      {empty && <div className="nb-note">This cell printed nothing.</div>}
      <OutText blocks={parsed.blocks} error={res.error} />
      {!res.error && parsed.charts.map((c, i) => <div className="nb-fig" key={i}><Chart chart={c} /></div>)}
    </div>
  );
}

export function ProseCell({ html, body }) {
  const [out, setOut] = useState(html || null);
  useEffect(() => {
    if (html) { setOut(html); return undefined; }
    let alive = true;
    markdown(body).then((h) => { if (alive) setOut(h); });
    return () => { alive = false; };
  }, [html, body]);
  return <div className="nb-prose" dangerouslySetInnerHTML={{ __html: out || '' }} />;
}

export function Header({ cell, idx, intro, html, name }) {
  const title = intro?.title || cell?.fields?.post || null;
  return (
    <header className="nb-head">
      {title && <h1 className="nb-title">{title}</h1>}
      {intro?.byline && <p className="nb-byline">{intro.byline}</p>}
      {intro?.summaryHtml && <p className="nb-summary" dangerouslySetInnerHTML={{ __html: intro.summaryHtml }} />}
      {intro?.equationHtml && <div className="nb-equation" dangerouslySetInnerHTML={{ __html: intro.equationHtml }} />}
      {cell && (cell.prose ?? cell.body) && <ProseCell html={html} body={cell.prose ?? cell.body} key={idx} />}
      {!title && !cell && name && <h1 className="nb-title nb-title-file">{name.split('/').map((part, i, all) => <Fragment key={i}>{part}{i < all.length - 1 && <>/<wbr /></>}</Fragment>)}</h1>}
    </header>
  );
}

// A code cell longer than FOLD lines (a table of reference data, say) starts
// folded: its first lines, read-only, and a button for the rest.
// A file header (`#! pde ...`, `#! equation ...`, `#! | prose`) read as text:
// the title, the equation as display math, the prose, then the other fields.
const TITLE_KEYS = new Set(['pde', 'engine', 'program', 'title', 'method_name']);
export function headerMarkdown({ fields, prose }) {
  const out = [];
  const rest = [];
  fields.forEach(([k, v], i) => {
    if (i === 0 && TITLE_KEYS.has(k)) out.push(`## ${v}`);
    else if (k === 'equation') out.push(`$$${v}$$`);
    else rest.push(`- **${k}**: ${v}`);
  });
  if (prose.trim()) out.push(prose);
  if (rest.length) out.push(rest.join('\n'));
  return out.join('\n\n');
}

export function CodeCell({ cell, idx, number, res, running, onEdit, onKey, taRef, onRun, disabled }) {
  const head = useMemo(() => docHeader(cell.code), [cell.code]);
  const headMd = useMemo(() => (head.doc ? headerMarkdown(head) : ''), [head]);
  const code = head.body.replace(/\n+$/, '');
  const first = cell.codeStart + (head.lines || 0);
  const lines = code.split('\n').length;
  const [open, setOpen] = useState(lines <= FOLD);
  const status = running ? 'running' : res ? (res.stopped ? 'stopped' : res.stale ? 'stale' : res.error ? 'error' : 'done') : 'idle';
  return (
    <section className={`nb-code nb-s-${status}`} data-cell={number} data-status={status} data-continued={res && !res.stale && res.continued ? 'yes' : undefined}>
      <div className="nb-cellbar">
        <span className="nb-num" title={`lines ${first}–${first + lines - 1} of the file`}>[{number}]</span>
        <button type="button" className="nb-run" onClick={onRun} disabled={disabled} aria-label={`Run cell ${number}`}>
          {running ? 'running…' : '▶ Run'}
        </button>
        {res && !running && !res.stopped && (
          <span className={`nb-meta${res.error ? ' nb-meta-err' : ''}`}>
            {res.error ? 'error' : `${res.ms < 10 ? res.ms.toFixed(1) : Math.round(res.ms)} ms`}
            {!res.error && (res.continued || number === 1 ? ' · this cell alone' : ` · rebuilt cells 1–${number - 1}, then this one`)}
          </span>
        )}
        <span className="nb-lines">lines {first}–{first + lines - 1}</span>
      </div>
      {headMd && (
        <div className="nb-dochead" title={`the file's #! header, lines ${cell.codeStart}–${first - 1}`}>
          <ProseCell body={headMd} />
        </div>
      )}
      {open ? (
        <CellEditor code={code} onChange={(t) => onEdit(idx, head.doc + t)} onKeyDown={(e) => onKey(e, idx)} label={`Code cell ${number}`} taRef={taRef} />
      ) : (
        <>
          <CellEditor code={code.split('\n').slice(0, FOLD_SHOW).join('\n')} onChange={() => {}} readOnly label={`Code cell ${number}, first ${FOLD_SHOW} lines`} />
          <button type="button" className="nb-more nb-unfold" onClick={() => setOpen(true)}>Show all {lines} lines</button>
        </>
      )}
      <Output res={res} />
    </section>
  );
}

/**
 * Running a file's cells: one worker, its incremental session, and each code
 * cell's latest result. Shared by the notebook view and a post's other views
 * (PostViews.jsx), which run the same cells.
 */
export function useCellRunner(cells, setCells, base = null) {
  const [results, setResults] = useState({}); // cell index -> result
  const [running, setRunning] = useState(null); // cell index running now
  const [all, setAll] = useState(null); // { at, total } during Run all; { done, secs } after
  const { send, stop } = useWorker();
  // What the worker's session last ran, as the text it was given; a run whose
  // prefix is this continues the session, anything else rebuilds it.
  const session = useRef(null);
  const busy = useRef(false);
  const cellsRef = useRef(cells);
  cellsRef.current = cells;
  const tas = useRef({}); // cell index -> textarea ref
  const taRefFor = (k) => (tas.current[k] ||= { current: null });

  const codeIdx = useMemo(() => cells.map((c, i) => (c.kind === 'code' && !c.empty ? i : -1)).filter((i) => i >= 0), [cells]);
  const numberOf = (k) => codeIdx.indexOf(k) + 1;

  // The units the interpreter gets, with imported modules spliced in. One set
  // of modules for the whole file: a module the first cell uses is not
  // included again when a later cell names it.
  const resolver = useMemo(() => (typeof window === 'undefined' ? null : browserResolver(base)), [base]);
  const expanded = useCallback(async (cs) => {
    const units = runUnits(cs);
    const seen = new Set();
    const out = [];
    for (const u of units) out.push(hasUses(u.text) ? await expandUses(u.text, resolver, [], seen) : u.text);
    return { units, texts: out };
  }, [resolver]);

  const runOne = useCallback(async (k) => {
    const cs = cellsRef.current;
    const { units, texts } = await expanded(cs);
    const j = units.findIndex((u) => u.index === k);
    if (j < 0) return { ok: true };
    const prefix = texts.slice(0, j).join('\n');
    const cell = texts[j];
    const continued = j > 0 && session.current === prefix;
    setRunning(k);
    const r = await send(prefix, cell);
    setRunning(null);
    if (r.stopped) {
      session.current = null;
      setResults((old) => ({ ...old, [k]: { stopped: true } }));
      return { ok: false, stopped: true };
    }
    const out = r.error ? `error: ${r.error}` : r.out;
    // a diagnostic is a LINE that starts with `error`; a program may well show a
    // variable called shape_error
    const error = /^error\b/m.test(out);
    session.current = error ? null : prefix ? `${prefix}\n${cell}` : cell;
    setResults((old) => ({ ...old, [k]: { parsed: parseOwnOutput(out), out, error, ms: (r.seconds || 0) * 1000, continued, stale: false } }));
    return { ok: !error };
  }, [expanded, send]);

  const runCell = useCallback(async (k) => {
    if (busy.current) return;
    busy.current = true;
    try { await runOne(k); } finally { busy.current = false; }
  }, [runOne]);

  const runAll = useCallback(async () => {
    if (busy.current) return;
    busy.current = true;
    const t0 = performance.now();
    const ks = codeIdx;
    try {
      for (let n = 0; n < ks.length; n++) {
        setAll({ at: n + 1, total: ks.length });
        const r = await runOne(ks[n]);
        if (!r.ok) { setAll({ done: true, stoppedAt: n + 1, total: ks.length, secs: (performance.now() - t0) / 1000, stopped: r.stopped }); return; }
      }
      setAll({ done: true, total: ks.length, secs: (performance.now() - t0) / 1000 });
    } finally { busy.current = false; }
  }, [codeIdx, runOne]);

  const onStop = useCallback(() => { stop(); session.current = null; }, [stop]);

  // An edit makes this cell's output, and every later cell's, out of date.
  const onEdit = useCallback((k, code) => {
    setCells((cs) => cs.map((c, i) => (i === k ? withCode(c, code) : c)));
    setResults((old) => {
      const next = { ...old };
      for (const key of Object.keys(next)) if (Number(key) >= k && next[key].parsed) next[key] = { ...next[key], stale: true };
      return next;
    });
  }, []);

  // Shift+Enter runs the cell and moves to the next one; Ctrl/Cmd+Enter runs it and stays.
  const onKey = useCallback((e, k) => {
    if (e.key !== 'Enter') return;
    if (e.shiftKey) {
      e.preventDefault();
      runCell(k);
      const next = codeIdx[codeIdx.indexOf(k) + 1];
      if (next !== undefined) {
        const ta = tas.current[next]?.current;
        if (ta) { ta.focus(); ta.setSelectionRange(0, 0); ta.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }
      }
    } else if (e.metaKey || e.ctrlKey) {
      e.preventDefault();
      runCell(k);
    }
  }, [codeIdx, runCell]);

  return { results, running, all, runCell, runAll, onStop, onEdit, onKey, taRefFor, codeIdx, numberOf };
}

export default function Notebook({ src = null, srcUrl = null, name = '', proseHtml = {}, intro = null, expectError = '', links = [], inside = null, onSource = null, problem = false }) {
  const [text, setText] = useState(src);
  const [loadErr, setLoadErr] = useState('');
  useEffect(() => {
    if (src !== null || !srcUrl) return undefined;
    let alive = true;
    fetch(srcUrl).then((r) => (r.ok ? r.text() : Promise.reject(new Error(`${r.status}`))))
      .then((t) => { if (alive) setText(t); })
      .catch((e) => { if (alive) setLoadErr(`Could not load ${srcUrl} (${e.message}).`); });
    return () => { alive = false; };
  }, [src, srcUrl]);

  const [cells, setCells] = useState(() => (src !== null ? splitNotebook(src) : []));
  useEffect(() => { if (text !== null && src === null) setCells(splitNotebook(text)); }, [text, src]);

  const { results, running, all, runCell, runAll, onStop, onEdit, onKey, taRefFor, codeIdx, numberOf } = useCellRunner(cells, setCells, name && name !== 'scratch.viv' ? name.replace(/\.viv$/, '') : null);

  const [dirty, setDirty] = useState(false);
  useEffect(() => {
    if (text === null) return;
    const now = joinCells(cells);
    setDirty(now !== text);
    if (onSource) onSource(now);
  }, [cells, text, onSource]);
  const download = () => {
    const blob = new Blob([joinCells(cells)], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = (name.split('/').pop() || 'notebook.viv');
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };

  if (loadErr) return <div className="nb"><p className="nb-note">{loadErr}</p></div>;
  if (text === null) return <div className="nb"><p className="nb-note">Loading {name}…</p></div>;

  const headerIdx = cells.findIndex((c) => c.kind === 'header');
  const isRunning = running !== null;
  return (
    <div className="nb" data-nb-expect-error={expectError ? 'yes' : undefined} data-nb-tests-fail={problem ? 'yes' : undefined}>
      <Header cell={headerIdx >= 0 ? cells[headerIdx] : null} idx={headerIdx} intro={intro} html={proseHtml[headerIdx]} name={name} />
      <div className="nb-bar" role="toolbar" aria-label="Notebook">
        <span className="nb-bar-name">{name ? <code>{name}</code> : 'Notebook'} · {codeIdx.length} code cell{codeIdx.length === 1 ? '' : 's'}</span>
        <span className="nb-bar-status" aria-live="polite">
          {all && !all.done && isRunning && `Running cell ${all.at} of ${all.total}`}
          {all && all.done && !all.stoppedAt && `✓ ${all.total} cell${all.total === 1 ? '' : 's'} · ${all.secs < 0.1 ? `${Math.max(1, Math.round(all.secs * 1000))} ms` : `${all.secs.toFixed(1)} s`}`}
          {all && all.done && all.stoppedAt && (all.stopped ? `Stopped at cell ${all.stoppedAt}` : `Cell ${all.stoppedAt} failed; the cells after it did not run`)}
        </span>
        <span className="nb-bar-links">
          {links.map((l) => <a key={l.href} href={l.href} download={l.download || undefined}>{l.label}</a>)}
          {dirty && <button type="button" className="nb-linkbtn" onClick={download} title="Your edited copy of the file">Download your edits</button>}
        </span>
        {isRunning
          ? <button type="button" className="nb-runall nb-stop" onClick={onStop}>Stop</button>
          : <button type="button" className="nb-runall" onClick={runAll} disabled={!!expectError || !codeIdx.length}>Run all</button>}
      </div>
      {inside && <LookInside links={inside} />}
      {expectError && <p className="nb-expect">{expectError}</p>}
      {problem && <p className="nb-expect">This is a problem: its tests fail until the functions in it are written. Edit a cell and run it to try; the cells below it show whether the tests pass.</p>}
      <p className="nb-keys">Shift+Enter in a cell runs it and moves to the next. Running a cell runs every cell above it too, as one program.</p>
      {cells.map((c, i) => {
        if (c.kind === 'header') return null;
        if (c.kind === 'prose') return <ProseCell key={i} html={proseHtml[i]} body={c.body} />;
        if (c.empty) return null;
        return (
          <CodeCell key={i} cell={c} idx={i} number={numberOf(i)} res={results[i]} running={running === i}
            onEdit={onEdit} onKey={onKey} taRef={taRefFor(i)} onRun={() => runCell(i)} disabled={isRunning || !!expectError} />
        );
      })}
    </div>
  );
}
