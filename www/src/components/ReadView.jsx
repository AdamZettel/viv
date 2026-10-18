// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The explorer's reading view ("The reading
// view"): the code is the spine, on the left, and beside each line its notes
// fill in as it scrolls into view: what each variable assigned or first used
// there is for (its meaning, its symbol, and the value it had in the recorded
// run, labeled with its pass), what a step or a statement does when that is
// not obvious from the docs, and the equations the intent author wrote for
// the real math, with the tests and holds that check them as marks in the
// margin.
//
// Two ways to read, switched at the top:
//   Core path     the stops of the program's `path` block (its .intent file),
//                 each with its narrative, its figures and its code; what is
//                 skipped between two stops is one line that opens.
//   All the code  every file, top to bottom, as written, every def annotated.
// A pass selector, shared by the whole page, picks the pass (and trial) the
// values come from. Values come from the recording the timeline uses; nothing
// is run again.
import { Fragment, useEffect, useLayoutEffect, useMemo, useRef, useState, createContext, useContext } from 'react';
import { highlight } from '../lib/highlight.js';
import { renderProse } from '../lib/richText.js';
import {
  buildReading, buildFiles, buildPath, defNotes, chunksOf, indexRun, passModel, valueAt, passLabel, fmtValue, fillRun, equationParts,
  skippedText, mainPieces,
} from '../lib/readView.js';
import { parseOwnOutput, Chart } from './VivCell.jsx';
import ProgramGraph from './ProgramGraph.jsx';

const MAIN = '<main>';
/** How far down the viewport the reading line sits: the row crossing it is the current one. */
const READ_LINE = 0.35;
/** A row's notes fill in when the row is this far up the viewport. */
const FILL_LINE = 0.85;
// a jump from the program's graph: how often, and how far apart, the scroll is set again
const SETTLE_TRIES = 25;
const SETTLE_MS = 120;
/** The smallest a display equation is set to fit its column; past that it scrolls. */
const MATH_MIN_SCALE = 0.7;

const Ctx = createContext(null);

function useMedia(q) {
  const [on, setOn] = useState(() => typeof window !== 'undefined' && window.matchMedia(q).matches);
  useEffect(() => {
    const mq = window.matchMedia(q);
    const f = () => setOn(mq.matches);
    mq.addEventListener('change', f);
    f();
    return () => mq.removeEventListener('change', f);
  }, [q]);
  return on;
}

const escHtml = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/**
 * A mark in the margin for a test or a hold that checks what it sits beside:
 * a check (or a cross, for a hold that failed in this run), with the test's
 * name on hover or focus.
 */
function checkHtml(kind, name, state = 'test', extra = '') {
  const what = `checked by the ${kind} “${name}”${extra}`;
  const glyph = state === 'bad' ? '✗' : '✓';
  return `<button type="button" class="rv-check ${state}" data-kind="${kind}" data-tip="${escHtml(what)}" aria-label="${escHtml(what)}">${glyph}</button>`;
}

/** Intent prose with its math typeset and each `[test "..."]` citation as a mark. */
function proseHtml(text, katex) {
  const cites = [];
  const t = String(text || '').replace(/\s*\[(test|hold) "([^"]*)"\]/g, (_, kind, name) => {
    cites.push([kind, name]);
    return `\u0001${cites.length - 1}\u0001`;
  });
  return renderProse(t, katex).replace(/\u0001(\d+)\u0001/g, (_, k) => {
    const [kind, name] = cites[Number(k)];
    return checkHtml(kind, name);
  });
}

function Prose({ text, katex, as: Tag = 'p', className }) {
  const html = useMemo(() => proseHtml(text, katex), [text, katex]);
  return <Tag className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}

/** A piece of LaTeX, typeset (or its source until KaTeX has loaded). */
function Tex({ tex, display = false, katex }) {
  const html = useMemo(() => (katex ? katex.renderToString(tex, { throwOnError: false, displayMode: display, output: 'html' }) : null), [tex, display, katex]);
  if (html === null) return <code className="rv-texsrc">{tex}</code>;
  return <span className={display ? 'rv-tex rv-texd' : 'rv-tex'} dangerouslySetInnerHTML={{ __html: html }} />;
}

// `unrun`: the program is not run here (an atlas made without a run, lib/atlas.js), so no values come
export default function ReadView({ ix, run, katex, recording, unrun = false }) {
  const book = useMemo(() => buildReading(ix), [ix]);
  const files = useMemo(() => buildFiles(ix).filter((f) => f.defs.length || f.top.length), [ix]);
  const path = useMemo(() => ((ix.bundle.paths || [])[0] ? buildPath(ix, ix.bundle.paths[0], book) : null), [ix, book]);
  const rx = useMemo(() => (run ? indexRun(run.tree, run.rec.events) : null), [run]);
  const pm = useMemo(() => (run && rx ? passModel(ix, run.tree, rx) : null), [ix, run, rx]);
  const charts = useMemo(() => (run ? parseOwnOutput(run.rec.printed || '').charts : []), [run]);
  const notesOf = useMemo(() => {
    const cache = new Map();
    return (def) => {
      if (!cache.has(def)) cache.set(def, defNotes(ix, def));
      return cache.get(def);
    };
  }, [ix]);
  const [mode, setMode] = useState(() => {
    const h = typeof window !== 'undefined' ? window.location.hash : '';
    return h === '#all' || !path ? 'all' : 'core';
  });
  const [sel, setSel] = useState({ pass: 1, trial: 1 });
  const narrow = useMedia('(max-width: 899px)');
  const reduced = useMedia('(prefers-reduced-motion: reduce)');
  const [where, setWhere] = useState({ file: '', def: '', step: '' });
  const [jump, setJump] = useState(false);
  const bodyRef = useRef(null);

  const switchMode = (m) => {
    setMode(m);
    setJump(false);
    try {
      window.history.replaceState(null, '', m === 'all' ? '#all' : window.location.pathname + window.location.search);
    } catch { /* a sandboxed page cannot change its URL */ }
    const top = bodyRef.current ? bodyRef.current.getBoundingClientRect().top + window.scrollY - 160 : 0;
    if (window.scrollY > top) window.scrollTo({ top: Math.max(0, top), behavior: 'auto' });
  };

  // a row's notes fill in as it comes into view; the row at the reading line is current
  useEffect(() => {
    const root = bodyRef.current;
    if (!root) return undefined;
    const fill = new IntersectionObserver((es) => {
      for (const e of es) {
        if (e.isIntersecting || e.boundingClientRect.bottom < 0) {
          e.target.classList.add('seen');
          fill.unobserve(e.target);
        }
      }
    }, { rootMargin: `0px 0px -${Math.round((1 - FILL_LINE) * 100)}% 0px` });
    const cur = new IntersectionObserver((es) => {
      for (const e of es) {
        if (!e.isIntersecting) continue;
        for (const r of root.querySelectorAll('.rv-row.cur')) r.classList.remove('cur');
        e.target.classList.add('cur');
        const d = e.target.dataset;
        setWhere({ file: d.file || '', def: d.def || '', step: d.step || '' });
      }
    }, { rootMargin: `-${Math.round(READ_LINE * 100)}% 0px -${Math.round((1 - READ_LINE) * 100) - 1}% 0px` });
    const watch = (r) => {
      if (r.dataset.watched) return;
      r.dataset.watched = '1';
      if (reduced) r.classList.add('seen');
      else fill.observe(r);
      cur.observe(r);
    };
    for (const r of root.querySelectorAll('.rv-row')) watch(r);
    // rows added later (a skipped part opened, a collapsed def shown)
    const mo = new MutationObserver(() => {
      for (const r of root.querySelectorAll('.rv-row:not([data-watched])')) watch(r);
    });
    mo.observe(root, { childList: true, subtree: true });
    return () => {
      fill.disconnect();
      cur.disconnect();
      mo.disconnect();
      for (const r of root.querySelectorAll('.rv-row[data-watched]')) delete r.dataset.watched;
    };
  }, [mode, reduced, ix, narrow]);

  const ctx = { ix, rx, pm, sel, katex, charts, notesOf, narrow, run, book, path };
  const nPasses = pm ? pm.passes.length : 0;
  const pass = pm ? pm.passes.find((p) => p.n === sel.pass) : null;
  const nTrials = pass ? pass.trials.length : 0;
  const setPass = (n) => setSel({ pass: Math.max(1, Math.min(nPasses, n)), trial: 1 });
  const go = (id) => {
    const el = document.getElementById(id);
    setJump(false);
    if (!el) return;
    window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - window.innerHeight * 0.2, behavior: reduced ? 'auto' : 'smooth' });
  };
  // from the graph's panel: that def's code in the full reading, in all the code if the path does not show it
  const pickDef = (def) => {
    const find = () => bodyRef.current && [...bodyRef.current.querySelectorAll('.rv-row')].find((r) => r.dataset.def === def);
    // rows grow as their notes fill in, so the scroll is set again until the row stops moving
    const settle = (k) => {
      const el = find();
      if (!el) return;
      const off = el.getBoundingClientRect().top - window.innerHeight * READ_LINE + 4;
      if (Math.abs(off) > 2) window.scrollTo({ top: window.scrollY + off, behavior: 'auto' });
      if (k < SETTLE_TRIES) setTimeout(() => settle(k + 1), SETTLE_MS);
    };
    if (!find() && mode !== 'all') switchMode('all');
    setTimeout(() => settle(0), 0);
  };
  const crumb = [where.file ? `${where.file}.viv` : null, where.def === MAIN ? 'the program' : where.def || null, where.step || null].filter(Boolean);

  return (
    <Ctx.Provider value={ctx}>
      <div className={`rv${reduced ? ' rv-still' : ''}${narrow ? ' rv-narrow' : ''}`} data-mode={mode}>
        {ix.bundle.drawing && <ProgramGraph ix={ix} katex={katex} def={where.def} step={where.step} onRead={pickDef} run={run} />}
        <div className="rv-bar">
          <div className="rv-modes" role="radiogroup" aria-label="How to read the code">
            {path && (
              <button type="button" role="radio" aria-checked={mode === 'core'} className={mode === 'core' ? 'on' : ''} onClick={() => switchMode('core')}>
                Core path <span className="rv-count">{path.stops.length} stops</span>
              </button>
            )}
            <button type="button" role="radio" aria-checked={mode === 'all'} className={mode === 'all' ? 'on' : ''} onClick={() => switchMode('all')}>
              All the code <span className="rv-count">{files.length} files</span>
            </button>
          </div>
          <div className="rv-passsel" role="group" aria-label="Which pass of the run the values come from">
            {pm ? (
              <>
                <button type="button" className="rv-step" onClick={() => setPass(sel.pass - 1)} disabled={sel.pass <= 1} aria-label="Previous pass">‹</button>
                <label>
                  pass{' '}
                  <select value={sel.pass} onChange={(e) => setPass(Number(e.target.value))} aria-label="Pass">
                    {pm.passes.map((p) => <option key={p.n} value={p.n}>{p.n}</option>)}
                  </select>
                  <span className="rv-of"> of {nPasses}</span>
                </label>
                <button type="button" className="rv-step" onClick={() => setPass(sel.pass + 1)} disabled={sel.pass >= nPasses} aria-label="Next pass">›</button>
                {nTrials > 1 && (
                  <label className="rv-trial">
                    trial{' '}
                    <select value={sel.trial} onChange={(e) => setSel({ pass: sel.pass, trial: Number(e.target.value) })} aria-label="Trial">
                      {pass.trials.map((t) => <option key={t.t} value={t.t}>{t.t}</option>)}
                    </select>
                    <span className="rv-of"> of {nTrials}</span>
                  </label>
                )}
              </>
            ) : (
              <span className="rv-of">{run ? 'values from the one run' : unrun ? 'not run here' : 'values after the recording'}</span>
            )}
          </div>
          <div className="rv-jumpwrap">
            <button type="button" className="rv-jumpbtn" aria-expanded={jump} aria-controls="rv-jump" onClick={() => setJump(!jump)}>{mode === 'core' ? 'Stops' : 'Files'} ▾</button>
            {jump && (
              <nav className="rv-jump" id="rv-jump" aria-label={mode === 'core' ? 'The stops of the core path' : 'The files'}>
                <ol>
                  {mode === 'core'
                    ? path.stops.map((s) => (
                        <li key={s.i}>
                          <button type="button" onClick={() => go(`rv-stop-${s.i}`)}><span className="rv-n">{s.i + 1}</span> <Prose as="span" text={s.stop.text || s.stop.target} katex={katex} /></button>
                        </li>
                      ))
                    : files.map((f) => (
                        <li key={f.name}>
                          <button type="button" onClick={() => go(`rv-file-${f.index}`)}><code>{f.name}.viv</code> <span className="rv-of">{f.lines} lines{f.annotated ? '' : ', not annotated'}</span></button>
                        </li>
                      ))}
                </ol>
              </nav>
            )}
          </div>
          {ix.bundle.drawing && (
            <button type="button" className="rv-jumpbtn rv-tograph" onClick={() => document.getElementById('rv-graph')?.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' })}>Graph ▴</button>
          )}
          <nav className="rv-crumb" aria-label="Where you are">
            <ol>{crumb.map((c, k) => <li key={k} className={k === crumb.length - 1 ? 'on' : ''}>{c}</li>)}</ol>
          </nav>
        </div>
        {!run && !unrun && <p className="rv-wait ex-quiet" role="status">{recording || 'Recording the run'}; the values it took fill in beside the code.</p>}
        <div className="rv-body" ref={bodyRef}>
          {mode === 'core' && path ? <CorePath path={path} /> : <AllCode files={files} />}
        </div>
      </div>
    </Ctx.Provider>
  );
}

// ---------------------------------------------------------------------------
// The annotated code: a run of lines, each chunk's notes beside its first line
// ---------------------------------------------------------------------------

/**
 * Lines of one def with their notes. `extra` adds notes by line (the core
 * path's docs for variables first met there); `skipStep` leaves out a step's
 * own note when the narrative above already shows it.
 */
function Annotated({ def, lines, extra = null, skipStep = null, file }) {
  const { ix, notesOf } = useContext(Ctx);
  const chunks = useMemo(() => {
    const own = notesOf(def);
    const merged = new Map();
    for (const x of lines) {
      const ns = [...(own.get(x) || []), ...((extra && extra.get(x)) || [])].filter((n) => !(skipStep && n.kind === 'step' && n.stepId === skipStep));
      if (ns.length) merged.set(x, ns);
    }
    return chunksOf(lines, merged);
  }, [def, lines, extra, skipStep, notesOf]);
  const stepAt = useMemo(() => {
    // the innermost step around each line, for the bar's crumb
    const m = new Map();
    const steps = ix.bundle.nodes.filter((n) => n.kind === 'step' && n.def === def);
    for (const x of lines) {
      let best = null;
      for (const s of steps) if (s.line <= x && s.end >= x && (!best || s.line >= best.line)) best = s;
      m.set(x, best ? best.step : '');
    }
    return m;
  }, [ix, def, lines]);
  return (
    <div className="rv-code" data-def={def}>
      {chunks.map((c) => (
        <div key={c.line} className={`rv-row${c.notes.length ? ' has' : ''}`} data-line={c.line} data-def={def} data-file={file || ''} data-step={stepAt.get(c.line) || ''}>
          <div className="rv-src">
            {c.lines.map((x, k) => (
              <Fragment key={x}>
                {k > 0 && x !== c.lines[k - 1] + 1 && <div className="rv-gap" aria-hidden="true">⋯</div>}
                <CodeLine x={x} />
              </Fragment>
            ))}
          </div>
          <div className="rv-notes">
            {c.notes.map((n, k) => <Note key={k} n={n} def={def} line={c.line} />)}
          </div>
        </div>
      ))}
    </div>
  );
}

function CodeLine({ x }) {
  const { ix } = useContext(Ctx);
  const m = ix.toModule(x);
  const t = ix.programLines[x - 1] || '';
  return (
    <div className="rv-cl" data-line={x}>
      <span className="rv-ln" aria-hidden="true">{m ? m.line : ''}</span>
      <code dangerouslySetInnerHTML={{ __html: highlight(t || ' ', 'viv') || ' ' }} />
    </div>
  );
}

function Note({ n, def, line }) {
  if (n.kind === 'doc') return <DocNote n={n} def={def} line={line} />;
  if (n.kind === 'step') return <StepNote n={n} def={def} />;
  if (n.kind === 'anchor') return <AnchorNote n={n} def={def} line={line} />;
  if (n.kind === 'def') return <div className="rv-note rv-defnote"><Details dets={n.details} unit={{ kind: 'intro', def }} /></div>;
  return null;
}

/** A variable's doc: its symbol and name, what it is for, and its value in the chosen pass. */
function DocNote({ n, def, line }) {
  const { ix, katex } = useContext(Ctx);
  const stepEnd = n.step && !n.early ? ix.nodes.get(`${def}/${n.step}`)?.end ?? null : null;
  return (
    <div className={`rv-note rv-doc${n.reuse ? ' reuse' : ''}${n.early ? ' early' : ''}`} data-var={n.name} data-key={n.key}>
      <span className="rv-dname">
        {n.symbol ? <><Tex tex={n.symbol} katex={katex} /> </> : null}
        <code>{n.name}</code>
        {n.reuse ? <span className="rv-now"> now holds</span> : null}
      </span>{' '}
      <Prose as="span" className="rv-dtext" text={n.text} katex={katex} />
      <Value def={def} line={line} name={n.name} stepEnd={stepEnd} />
    </div>
  );
}

function Value({ def, line, name, stepEnd }) {
  const { ix, rx, pm, sel } = useContext(Ctx);
  if (!rx) return null;
  const v = valueAt(ix, rx, pm, sel, def, line, name, stepEnd);
  if (!v) return null;
  const f = fmtValue(v.value);
  if (f === null) return null;
  const label = passLabel(v.loc, pm);
  return (
    <span className="rv-val" data-pass={label}>
      <span className="rv-eqs">=</span> <span className="rv-num">{f}</span> <span className="rv-passtag">{label}</span>
    </span>
  );
}

/**
 * A step's note beside its first line: its text and its equations; its why,
 * how, predict and run lines one click away, so a long explanation does not
 * push the code apart (the core path shows them in full above the code).
 */
function StepNote({ n, def }) {
  const { katex } = useContext(Ctx);
  const unit = { kind: 'step', def, stepId: n.stepId };
  const eqs = n.details.filter((d) => d.label === '$$');
  const more = n.details.filter((d) => d.label !== '$$');
  return (
    <div className="rv-note rv-stepnote" data-step={n.stepId}>
      <p className="rv-steph"><span className="rv-kw">step</span> <code>{n.stepId}</code>{n.text ? <> <Prose as="span" text={n.text} katex={katex} /></> : null}</p>
      <Details dets={eqs} unit={unit} />
      {more.length > 0 && (
        <details className="rv-more">
          <summary>{moreLabel(more)}</summary>
          <Details dets={more} unit={unit} />
        </details>
      )}
    </div>
  );
}

/** What a step's hidden details hold, in a few words: "Why, how, and a prediction". */
function moreLabel(dets) {
  const has = (l) => dets.some((d) => d.label === l);
  const parts = [has('why') && 'why', has('how') && 'how', has('predict') && 'a prediction', !has('predict') && has('run') && 'what the run did'].filter(Boolean);
  const text = parts.length > 1 ? `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}` : parts[0] || 'more';
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function AnchorNote({ n, def, line }) {
  const { katex } = useContext(Ctx);
  return (
    <div className="rv-note rv-anchor">
      {n.text.split('\n').map((p, k) => <Prose key={k} text={p} katex={katex} className="rv-p" />)}
      <Details dets={n.details} unit={{ kind: 'anchor', def, stepId: n.step, anchor: { anchor: n.anchor, line, step: n.step } }} />
    </div>
  );
}

/** Why, how, equations, predict and run lines, and a path's own prose. */
function Details({ dets, unit }) {
  const { katex } = useContext(Ctx);
  const blocks = [];
  for (const d of dets) {
    const last = blocks[blocks.length - 1];
    if (d.label === 'run' && last && last.kind === 'predict') last.runs.push(d);
    else if (d.label === 'predict') blocks.push({ kind: 'predict', d, runs: [] });
    else blocks.push({ kind: d.label, d });
  }
  const u = { focus: [], ...unit };
  return blocks.map((b, k) => {
    if (b.kind === '$$') return <Equation key={k} text={b.d.text} />;
    if (b.kind === 'predict') return <Predict key={k} b={b} unit={u} />;
    if (b.kind === 'run') return <p key={k} className="rv-run"><span className="rv-label">In this run.</span> <RunText text={b.d.text} unit={u} /></p>;
    if (b.kind === 'figure') return null;
    const label = b.kind === 'why' ? 'Why' : b.kind === 'how' ? 'How' : null;
    return (
      <div key={k} className="rv-field">
        {b.d.text.split('\n').map((p, j) => (
          <p key={j} className="rv-p">
            {j === 0 && label ? <><span className="rv-label">{label}.</span> </> : null}
            <Prose as="span" text={p} katex={katex} />
          </p>
        ))}
      </div>
    );
  });
}

function Predict({ b, unit }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rv-predict">
      <p><span className="rv-label">Predict.</span> <RunText text={b.d.text} unit={unit} /></p>
      {b.runs.length > 0 && (
        <>
          <button type="button" className="rv-reveal" aria-expanded={open} onClick={() => setOpen(!open)}>{open ? 'Hide what the run did' : 'Show what the run did'}</button>
          {open && b.runs.map((r, k) => <p key={k} className="rv-run rv-answer"><RunText text={r.text} unit={unit} /></p>)}
        </>
      )}
    </div>
  );
}

function RunText({ text, unit }) {
  const { rx, katex } = useContext(Ctx);
  const pieces = useMemo(() => (rx ? fillRun(text, unit, rx) : null), [text, unit, rx]);
  if (!pieces) return <span className="ex-quiet"><Prose as="span" text={text.replace(/\{([A-Za-z_]\w*)[^}]*\}/g, '…')} katex={katex} /></span>;
  return (
    <>
      {pieces.map((p, k) => {
        if (typeof p === 'string') return p ? <Prose key={k} as="span" text={p} katex={katex} /> : null;
        if (p.missing) return <span key={k} className="rv-missing" title="the recording has no value for this">{p.missing}</span>;
        return <span key={k} className="rv-runval" title={p.name}>{p.value}</span>;
      })}
    </>
  );
}

/** A display equation the intent wrote, with the holds and tests that check it as marks. */
function Equation({ text }) {
  const { rx, katex } = useContext(Ctx);
  const { tex, holds, tests } = equationParts(text);
  const marks = [
    ...holds.map((s) => {
      const h = rx ? rx.bySentence.get(s) : null;
      const state = !rx ? 'wait' : !h ? 'none' : h.fail ? 'bad' : 'ok';
      const extra = h ? (h.fail ? `: FAILED ${h.fail} of ${h.ok + h.fail} times in this run` : `: held all ${h.ok} times in this run`) : rx ? ': not reached in this run' : '';
      return checkHtml('hold', s, state, extra);
    }),
    ...tests.map((s) => checkHtml('test', s)),
  ];
  const box = useRef(null);
  useFitMath(box, [tex, katex]);
  return (
    <figure className="rv-eq">
      <div className="rv-eqbox" ref={box}><Tex tex={tex} display katex={katex} /></div>
      {marks.length > 0 && <span className="rv-marks" dangerouslySetInnerHTML={{ __html: marks.join('') }} />}
    </figure>
  );
}

/**
 * Math cannot wrap the way prose does, so a display wider than its column is
 * set smaller, down to MATH_MIN_SCALE, and past that scrolls sideways, the
 * box marked `wide` so a fade at its right edge says there is more.
 */
function useFitMath(ref, deps) {
  useLayoutEffect(() => {
    const box = ref.current;
    if (!box) return undefined;
    const fit = () => {
      const el = box.firstElementChild;
      if (!el) return;
      el.style.fontSize = '';
      let scale = 1;
      // not quite proportional (KaTeX has fixed-width pieces): a few rounds
      for (let round = 0; round < 4 && scale > MATH_MIN_SCALE; round++) {
        const room = box.clientWidth;
        const need = box.scrollWidth;
        if (!(room > 0 && need > room + 1)) break;
        scale = Math.max(MATH_MIN_SCALE, (scale * room) / need - 0.005);
        el.style.fontSize = `${(scale * 100).toFixed(1)}%`;
      }
      box.classList.toggle('wide', box.scrollWidth > box.clientWidth + 1);
    };
    fit();
    let width = box.clientWidth;
    const ro = new ResizeObserver(() => {
      if (box.clientWidth === width) return;
      width = box.clientWidth;
      fit();
    });
    ro.observe(box);
    let live = true;
    document.fonts?.ready.then(() => live && fit());
    return () => {
      live = false;
      ro.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

// ---------------------------------------------------------------------------
// All the code
// ---------------------------------------------------------------------------

function AllCode({ files }) {
  const { ix, katex } = useContext(Ctx);
  return files.map((f) => {
    const isRoot = f.index === 0;
    const moduleText = ix.bundle.moduleText?.[f.name];
    return (
      <section key={f.name} className="rv-file" id={`rv-file-${f.index}`} data-file={f.name} aria-labelledby={`rv-fh-${f.index}`}>
        <header className="rv-filehead">
          <h2 id={`rv-fh-${f.index}`}><code>{f.name}.viv</code></h2>
          <p className="rv-of">{f.lines} lines{f.defs.length ? ` · ${f.annotated} of ${f.defs.length} ${f.defs.length === 1 ? 'def' : 'defs'} annotated` : ''}</p>
          {moduleText && (
            <details className="rv-about" open={isRoot}>
              <summary>About this file</summary>
              <ProseBlock text={moduleText} katex={katex} />
            </details>
          )}
        </header>
        {isRoot ? <MainCode /> : f.top.length > 0 && <Collapsed label={<>The file's top level: {f.top.length} lines of constants and comments</>} def={MAIN} lines={f.top} file={f.name} open={f.annotated > 0} />}
        {f.defs.map((d) => (d.annotated ? (
          <DefBlock key={d.def} d={d} file={f.name} />
        ) : (
          <Collapsed key={d.def} label={<><code>def {ix.bundle.intent[d.def]?.sig || d.def}</code> <span className="rv-of">({d.lines.length} lines; {d.test ? 'a helper for the tests' : 'no intent'})</span></>} def={d.def} lines={[...(d.comment || []), ...d.lines]} file={f.name} />
        )))}
      </section>
    );
  });
}

/** The root program: its prose blocks and its code, annotated. */
function MainCode() {
  const { ix, katex } = useContext(Ctx);
  const groups = useMemo(() => {
    const out = [];
    for (const p of mainPieces(ix)) {
      if (p.kind === 'prose') out.push({ prose: p.text });
      else {
        const last = out[out.length - 1];
        if (last && last.lines) last.lines.push(p.line);
        else out.push({ lines: [p.line] });
      }
    }
    return out.map((g) => (g.lines ? { lines: trimBlank(ix, g.lines) } : g)).filter((g) => g.prose || g.lines.length);
  }, [ix]);
  const it = ix.bundle.intent[MAIN] || {};
  return (
    <div className="rv-def" data-def={MAIN}>
      <div className="rv-defhead">
        <h3>The program</h3>
        {it.what && <Prose text={it.what} katex={katex} className="rv-p rv-what" />}
      </div>
      {groups.map((g, k) => (g.prose ? <div key={k} className="rv-prose"><ProseBlock text={g.prose} katex={katex} blankLines /></div> : <Annotated key={k} def={MAIN} lines={g.lines} file={ix.bundle.name} />))}
    </div>
  );
}

function trimBlank(ix, lines) {
  const out = [...lines];
  while (out.length && !(ix.programLines[out[0] - 1] || '').trim()) out.shift();
  while (out.length && !(ix.programLines[out[out.length - 1] - 1] || '').trim()) out.pop();
  return out;
}

/** One annotated def: its signature and what it is for, then its code with the docs beside it. */
function DefBlock({ d, file }) {
  const { ix, katex } = useContext(Ctx);
  const it = ix.bundle.intent[d.def] || {};
  return (
    <div className="rv-def" id={`rv-def-${d.def}`} data-def={d.def}>
      <div className="rv-defhead">
        <h3><code>def {it.sig || d.def}</code></h3>
        {it.what && <Prose text={it.what} katex={katex} className="rv-p rv-what" />}
        {(it.how || it.why) && (
          <details className="rv-hw">
            <summary>How it works{it.why ? ', and why' : ''}</summary>
            {it.how && <p className="rv-p"><span className="rv-label">How.</span> <Prose as="span" text={it.how} katex={katex} /></p>}
            {it.why && <p className="rv-p"><span className="rv-label">Why.</span> <Prose as="span" text={it.why} katex={katex} /></p>}
          </details>
        )}
      </div>
      {d.comment && d.comment.length > 0 && <div className="rv-code rv-comment">{d.comment.map((x) => <CodeLine key={x} x={x} />)}</div>}
      <Annotated def={d.def} lines={d.lines} file={file} />
    </div>
  );
}

function Collapsed({ label, def, lines, file, open: startOpen = false }) {
  const [open, setOpen] = useState(startOpen);
  return (
    <details className="rv-collapsed" open={startOpen} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>{label}</summary>
      {open && <Annotated def={def} lines={lines} file={file} />}
    </details>
  );
}

// An .intent file's module text comes with its lines joined and one newline
// between paragraphs; a post's `###` prose has a blank line between
// paragraphs (`blankLines`), and its line breaks inside one are spaces.
function ProseBlock({ text, katex, blankLines = false }) {
  const paras = String(text || '')
    .split(blankLines ? /\n\s*\n/ : /\n\s*\n|\n/)
    .map((p) => p.replace(/\s*\n\s*/g, ' ').trim())
    .filter(Boolean);
  return paras.map((p, k) => {
    const heading = p.match(/^#{1,4}\s+(.*)$/);
    if (heading) return <Prose key={k} as="h4" className="rv-ph" text={heading[1]} katex={katex} />;
    return <Prose key={k} text={p} katex={katex} className="rv-p" />;
  });
}

// ---------------------------------------------------------------------------
// The core path
// ---------------------------------------------------------------------------

function CorePath({ path }) {
  const { ix } = useContext(Ctx);
  // a variable's doc is shown where the path first meets it, even when its
  // glossary line starts to hold in code the path skips
  const extras = useMemo(() => {
    const seen = new Map();
    const texRows = new Map();
    for (const r of ix.bundle.tex || []) {
      if (!texRows.has(r.def)) texRows.set(r.def, []);
      texRows.get(r.def).push(r);
    }
    return path.stops.map((st) => {
      const def = st.def;
      if (!seen.has(def)) seen.set(def, new Set());
      const s = seen.get(def);
      const docs = (ix.bundle.intent[def] || {}).docs || [];
      const inStop = new Set(st.lines);
      const extra = new Map();
      // a doc whose site is in the stop is shown there anyway
      for (const d of docs) if (d.line !== null && inStop.has(d.line)) s.add(d.name);
      const rows = (texRows.get(def) || []).filter((r) => inStop.has(r.line)).sort((a, b) => a.line - b.line);
      for (const r of rows) {
        for (const name of r.names) {
          if (s.has(name)) continue;
          const mine = docs.filter((d) => d.name === name && d.line !== null);
          if (!mine.length) continue;
          // the doc in effect at this line: the last one to start before it, else the variable's own
          const before = mine.filter((d) => d.line <= r.line).sort((a, b) => b.line - a.line)[0];
          const d = before || mine.find((x) => !x.key.includes('@')) || mine[0];
          s.add(name);
          if (!extra.has(r.line)) extra.set(r.line, []);
          extra.get(r.line).push({ kind: 'doc', name, key: d.key, symbol: d.symbol, text: d.text, reuse: false, early: true, step: d.step });
        }
      }
      return extra;
    });
  }, [ix, path]);
  const stopModules = useMemo(() => new Set(path.stops.map((s) => (ix.bundle.intent[s.def] || {}).module)), [ix, path]);
  return (
    <>
      <header className="rv-pathhead">
        <h2>{path.title}</h2>
        <p className="rv-p">{path.stops.length} stops through the code that carries the algorithm, in the order a first reading needs them. What the path skips between two stops is one line that opens; All the code shows every file, top to bottom.</p>
      </header>
      {path.stops.map((st, k) => (
        <Fragment key={st.i}>
          {st.skipped.length > 0 && <Skipped units={st.skipped} stopModules={stopModules} />}
          <Stop st={st} n={path.stops.length} extra={extras[k]} />
        </Fragment>
      ))}
      {path.tail.length > 0 && <Skipped units={path.tail} stopModules={stopModules} after />}
      <p className="rv-end ex-quiet">The end of the core path. The Timeline view steps through the same run pass by pass and line by line.</p>
    </>
  );
}

function Stop({ st, n, extra }) {
  const { ix, katex, charts, run } = useContext(Ctx);
  const it = ix.bundle.intent[st.def] || {};
  const figs = st.pathDetails.filter((d) => d.label === 'figure');
  const lead = st.pathDetails.filter((d) => d.label === '>');
  const more = st.pathDetails.filter((d) => d.label === 'run' || d.label === 'predict');
  const stepText = st.stepId ? ix.bundle.steps[`${st.def}/${st.stepId}`] : null;
  const unit = st.stepId ? { kind: 'step', def: st.def, stepId: st.stepId } : { kind: 'intro', def: st.def };
  const title = st.stop.text || stepText || st.stop.target;
  return (
    <section className="rv-stop" id={`rv-stop-${st.i}`} data-stop={st.stop.target} aria-labelledby={`rv-sh-${st.i}`}>
      <div className={`rv-stophead${figs.length ? ' withfig' : ''}`}>
        <div className="rv-narr">
          <p className="rv-kicker">Stop {st.i + 1} of {n} · <code>{st.def}{st.stepId ? ` › ${st.stepId}` : ''}</code> · {it.module}.viv</p>
          <h2 id={`rv-sh-${st.i}`}><Prose as="span" text={title} katex={katex} /></h2>
          {lead.map((d, k) => <Prose key={k} text={d.text} katex={katex} className="rv-p rv-lead" />)}
          {!st.stepId && it.what && <Prose text={it.what} katex={katex} className="rv-p" />}
          {!st.stepId && (it.how || it.why) && (
            <details className="rv-hw">
              <summary>How {st.def} works{it.why ? ', and why' : ''}: the whole story in two paragraphs, numbers included (they answer the predictions below)</summary>
              {it.how && <p className="rv-p"><span className="rv-label">How.</span> <Prose as="span" text={it.how} katex={katex} /></p>}
              {it.why && <p className="rv-p"><span className="rv-label">Why.</span> <Prose as="span" text={it.why} katex={katex} /></p>}
            </details>
          )}
          {stepText && <p className="rv-p rv-codesays"><span className="rv-label">In the code:</span> <code>{st.stepId}</code> <Prose as="span" text={stepText} katex={katex} /></p>}
          <Details dets={st.details} unit={unit} />
          <Details dets={more} unit={unit} />
        </div>
        {figs.length > 0 && (
          <div className="rv-figs">
            {figs.map((f, k) => {
              const c = charts.find((ch) => ch.fig && ch.fig.title === f.text);
              return (
                <figure key={k} className="rv-fig" data-figure={f.text}>
                  {c ? <Chart chart={c} /> : <p className="ex-quiet">{run ? `The run drew no figure “${f.text}”.` : `“${f.text}” is drawn by the program; it appears when the run has been recorded.`}</p>}
                  {figureModule(ix, f.text) && <figcaption className="ex-quiet">Drawn by <code>{figureModule(ix, f.text)}.viv</code> in this run, from the values the run computed.</figcaption>}
                </figure>
              );
            })}
          </div>
        )}
      </div>
      {/* a stop with no line of its own (the top level's opening, line 0) has no code panel */}
      {st.lines.some((x) => x > 0) && <Annotated def={st.def} lines={st.lines.filter((x) => x > 0)} extra={extra} skipStep={st.stepId} file={it.module} />}
    </section>
  );
}

/** The module whose `figure("<title>", ...)` call draws a figure, or null. */
function figureModule(ix, title) {
  const lit = `figure(${JSON.stringify(title)}`;
  const k = ix.programLines.findIndex((l) => l.includes(lit));
  return k >= 0 ? ix.toModule(k + 1)?.module || null : null;
}

function Skipped({ units, stopModules, after = false }) {
  const { ix } = useContext(Ctx);
  const [open, setOpen] = useState(false);
  const text = skippedText(units, ix, stopModules).join('; ');
  // what the skipped part shows when opened: a def whole, or its steps
  const parts = useMemo(() => {
    const whole = new Set();
    for (const u of units) {
      if (u.kind !== 'intro' || u.def === MAIN) continue;
      const steps = ix.bundle.nodes.filter((n) => n.kind === 'step' && n.def === u.def).length;
      const listed = units.filter((v) => v.def === u.def && v.kind === 'step').length;
      if (!steps || listed >= steps) whole.add(u.def);
    }
    const out = [];
    const done = new Set();
    for (const u of units) {
      if (whole.has(u.def)) {
        if (done.has(u.def)) continue;
        done.add(u.def);
        const node = ix.nodes.get(u.def);
        if (node) out.push({ def: u.def, lines: range(node.line, node.end) });
      } else if (u.kind === 'step') {
        const node = ix.nodes.get(`${u.def}/${u.stepId}`);
        // a parent step's lines include its children's: each line once
        if (node && !out.some((o) => o.def === u.def && o.lines.includes(node.line))) out.push({ def: u.def, lines: range(node.line, node.end) });
      } else if (u.kind === 'intro' && u.def === MAIN) {
        const lines = trimBlank(ix, u.reveal || []);
        if (lines.length) out.push({ def: MAIN, lines });
      }
    }
    return out;
  }, [units, ix]);
  return (
    <details className="rv-skip" onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary><span className="rv-label">{after ? 'After the last stop' : 'Skipped'}:</span> {text}</summary>
      {open && parts.map((p, k) => <Annotated key={k} def={p.def} lines={p.lines} file={(ix.bundle.intent[p.def] || {}).module || ix.bundle.name} />)}
    </details>
  );
}

function range(a, b) {
  const out = [];
  for (let x = a; x <= b; x++) out.push(x);
  return out;
}
