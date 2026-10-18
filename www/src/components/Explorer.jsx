// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The explorer: run a program in
// this tab, then move through what it did. The program's bundle (its graph,
// line table and lenses) comes from scripts/explore-build.mjs; the run is
// recorded here, by viv.wasm in a worker, once (in two passes: the second
// records line by line every pass of a small run, and the loop passes the
// first found worth reading of a big one). Moving never runs anything.
//
// Three views of the one recording: Read (the code with its docs beside it,
// on a core path or file by file, components/ReadView.jsx; the default when
// the program has intent), Timeline (step through the run) and Notebook (one cell
// per step, in the order they ran).
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { highlight } from '../lib/highlight.js';
import { renderProse, loadKatex } from '../lib/richText.js';
import { hasIntent, equationParts } from '../lib/readView.js';
import ReadView from './ReadView.jsx';
import { parseOwnOutput } from './VivCell.jsx';
import {
  parseRecording, indexBundle, buildTree, stopsOf, pathOf, callOf, stepOf, indexOfNode, reveal, graphIdOf,
  valuesAt, historyOf, labelOf, defLabel, numText, secondPassIters, secondPassCalls, wholeRunRequest, iterValues, hasLines,
  digitsLine,
  splitDigits,
} from '../lib/explore.js';
import { vivWorker } from '../lib/vivWasm.js';

const EMPTY_VIEW = { open: new Set(), closed: new Set(), shown: new Set() };

// ---------------------------------------------------------------------------
// Playing
// ---------------------------------------------------------------------------

/** Play's speeds; 1x is one stop per PLAY_STOP_MS. */
export const PLAY_SPEEDS = [0.5, 1, 2, 4];
export const PLAY_STOP_MS = 1500;
/** How long the graph and the code take to glide to the next stop while playing. */
export const PLAY_EASE_MS = 400;
const SPEED_KEY = 'viv.explore.speed';

function savedSpeed() {
  try {
    const v = Number(window.localStorage.getItem(SPEED_KEY));
    return PLAY_SPEEDS.includes(v) ? v : 1;
  } catch {
    return 1;
  }
}
function saveSpeed(v) {
  try {
    window.localStorage.setItem(SPEED_KEY, String(v));
  } catch {
    /* a browser that keeps nothing plays at 1x next time */
  }
}
const reducedMotion = () => typeof window !== 'undefined' && !!window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const gliding = new WeakMap();
/** Scroll `el` to `top`: at once, or (while playing) eased over PLAY_EASE_MS. */
function scrollPane(el, top, ease) {
  const to = Math.max(0, top);
  const prev = gliding.get(el);
  if (prev) cancelAnimationFrame(prev);
  gliding.delete(el);
  const from = el.scrollTop;
  if (!ease || Math.abs(to - from) < 1) {
    el.scrollTop = to;
    return;
  }
  const t0 = performance.now();
  const frame = (t) => {
    const k = Math.min(1, (t - t0) / PLAY_EASE_MS);
    const e = k < 0.5 ? 2 * k * k : 1 - (2 - 2 * k) ** 2 / 2;
    el.scrollTop = from + (to - from) * e;
    if (k < 1) gliding.set(el, requestAnimationFrame(frame));
    else gliding.delete(el);
  };
  gliding.set(el, requestAnimationFrame(frame));
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

function record(worker, src, req) {
  return new Promise((resolve, reject) => {
    const id = Math.random().toString(36).slice(2);
    const on = (e) => {
      if (e.data.id !== id) return;
      worker.removeEventListener('message', on);
      if (e.data.error) reject(new Error(e.data.error));
      else resolve(e.data);
    };
    worker.addEventListener('message', on);
    worker.postMessage({ id, src, ...req });
  });
}

/** A promise that rejects after `ms` (0: never), for a recording with a budget. */
function within(promise, ms) {
  if (!ms) return promise;
  let timer = null;
  const late = new Promise((_, reject) => {
    timer = setTimeout(() => {
      const err = new Error('budget');
      err.budget = true;
      reject(err);
    }, ms);
  });
  return Promise.race([promise, late]).finally(() => clearTimeout(timer));
}

// `slug` opens a bundle the build made (public/explore/<slug>.json); `load`,
// in its place, makes one in the browser (lib/atlas.js, the /atlas page): it
// is called with a function that takes what is happening and resolves to
// `{ bundle, ran, note }`. A bundle made without a run (`ran` false) is shown
// as its atlas and reading alone, with `note` saying why it was not run, and
// so is one whose recording runs past `recordBudget` ms.
export default function Explorer({ slug, title, load = null, recordBudget = 0 }) {
  const [phase, setPhase] = useState({ what: 'loading', detail: 'Loading the program and its lenses' });
  const [ix, setIx] = useState(null);
  const [run, setRun] = useState(null); // { rec, tree, seconds, passes }
  const [view, setView] = useState(EMPTY_VIEW);
  const [pos, setPos] = useState(0);
  const [mode, setMode] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const [katex, setKatex] = useState(null);
  const [quiet, setQuiet] = useState(null); // why the program was not run, for an atlas without a run
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeedState] = useState(1);
  const rootRef = useRef(null);
  const playRef = useRef(null); // the Play button and the speed menu, which do not pause
  useEffect(() => setSpeedState(savedSpeed()), []);
  const setSpeed = (v) => {
    setSpeedState(v);
    saveSpeed(v);
  };

  useEffect(() => {
    let dead = false;
    loadKatex().then((k) => {
      if (!dead) setKatex(k);
    });
    return () => {
      dead = true;
    };
  }, []);

  useEffect(() => {
    let dead = false;
    let worker = null;
    const t0 = performance.now();
    const tick = setInterval(() => setElapsed((performance.now() - t0) / 1000), 250);
    (async () => {
      try {
        let bundle;
        if (load) {
          const made = await load((detail) => !dead && setPhase({ what: 'loading', detail }));
          bundle = made.bundle;
          if (dead) return;
          setIx(indexBundle(bundle));
          setMode('read');
          if (!made.ran) {
            setQuiet(made.note);
            setPhase({ what: 'static' });
            return;
          }
        } else {
          const res = await fetch(`/explore/${slug}.json`);
          if (!res.ok) throw new Error(`no bundle for ${slug} (${res.status})`);
          bundle = await res.json();
        }
        const ix = indexBundle(bundle);
        if (dead) return;
        setIx(ix);
        setMode((m) => m || (hasIntent(bundle) ? 'read' : 'timeline'));
        worker = vivWorker('/explore-worker.js');
        setPhase({ what: 'recording', detail: 'Recording the run: every step, and the first pass of each loop line by line' });
        const defs = bundle.lineDefs || [];
        const r1 = await within(record(worker, bundle.program, { defs }), recordBudget);
        let rec = parseRecording(r1.out);
        if (rec.error) throw new Error(rec.error);
        let tree = buildTree(rec.events, ix);
        let seconds = r1.seconds;
        let passes = 1;
        // a small run is recorded again whole, so every pass and trial has
        // its values; a big one, in the passes and calls worth reading
        const whole = wholeRunRequest(tree, rec.total);
        const iters = whole ? whole.iters : secondPassIters(tree);
        const calls = whole ? whole.calls : secondPassCalls(tree);
        const every = whole ? whole.every : [];
        if (Object.keys(iters).length || Object.keys(calls).length || every.length) {
          if (dead) return;
          setPhase({ what: 'recording', detail: whole ? 'Recording again, every pass line by line' : 'Recording again, line by line in the passes and calls worth reading' });
          const r2 = await within(record(worker, bundle.program, { defs, iters, calls, every }), recordBudget);
          const rec2 = parseRecording(r2.out);
          if (rec2.error) throw new Error(rec2.error);
          rec = rec2;
          tree = buildTree(rec.events, ix);
          seconds += r2.seconds;
          passes = 2;
        }
        if (dead) return;
        setRun({ rec, tree, seconds, passes, whole: !!whole && passes === 2 });
        setPhase({ what: 'ready' });
      } catch (err) {
        if (dead) return;
        if (err.budget) {
          setQuiet(`Its recording ran past ${recordBudget / 1000} s and was stopped: the atlas shows the run's calls, without the values.`);
          setPhase({ what: 'static' });
        } else setPhase({ what: 'error', detail: String(err.message || err) });
      } finally {
        clearInterval(tick);
        if (worker) worker.terminate();
      }
    })();
    return () => {
      dead = true;
      clearInterval(tick);
      if (worker) worker.terminate();
    };
  }, [slug, load, recordBudget]);

  const stops = useMemo(() => (run ? stopsOf(run.tree, view) : []), [run, view]);
  const stop = stops[Math.min(pos, stops.length - 1)];
  const node = stop ? stop.node : null;

  /** Move to a node: open what must be open for it to be a stop, then go there. */
  const goNode = useCallback(
    (n) => {
      if (!run || !n) return;
      const v = reveal(view, n);
      setView(v);
      setPos(indexOfNode(stopsOf(run.tree, v), n));
    },
    [run, view]
  );
  const setOpen = useCallback(
    (n, open) => {
      const v = { open: new Set(view.open), closed: new Set(view.closed), shown: new Set(view.shown) };
      if (open) {
        v.open.add(n.key);
        v.closed.delete(n.key);
      } else {
        v.closed.add(n.key);
        v.open.delete(n.key);
      }
      setView(v);
      setPos(indexOfNode(stopsOf(run.tree, v), n));
    },
    [run, view]
  );
  const isOpen = (n) => {
    if (!n) return false;
    if (view.closed.has(n.key)) return false;
    if (view.open.has(n.key)) return true;
    if (n.kind === 'call' || n.kind === 'step') return true;
    if (n.kind === 'iter') return n.n === 1;
    return false;
  };
  const canOpen = (n) => n && (n.kind === 'loop' || n.kind === 'group' || ((n.kind === 'iter' || n.kind === 'call' || n.kind === 'step') && n.children.length > 0));

  const move = useCallback(
    (how) => {
      if (!stops.length) return;
      const k = Math.min(pos, stops.length - 1);
      const find = (dir, pred) => {
        for (let j = k + dir; j >= 0 && j < stops.length; j += dir) if (pred(stops[j].node)) return j;
        return k;
      };
      const big = (n) => n.kind === 'step' || n.kind === 'call' || n.kind === 'loop' || n.kind === 'group';
      switch (how) {
        case 'next': return setPos(Math.min(k + 1, stops.length - 1));
        case 'prev': return setPos(Math.max(k - 1, 0));
        case 'nextStep': return setPos(find(1, big));
        case 'prevStep': return setPos(find(-1, big));
        case 'nextIter': return setPos(find(1, (n) => n.kind === 'iter'));
        case 'prevIter': return setPos(find(-1, (n) => n.kind === 'iter'));
        case 'out': {
          const c = callOf(stops[k].node);
          if (!c || !c.parent) return;
          return setPos(find(1, (n) => n.i >= c.end && pathOf(n).indexOf(c) < 0));
        }
        case 'first': return setPos(0);
        case 'last': return setPos(stops.length - 1);
        default:
      }
    },
    [stops, pos]
  );

  useEffect(() => {
    const onKey = (e) => {
      if (!run || mode !== 'timeline') return;
      const t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const map = { ArrowRight: e.shiftKey ? 'nextStep' : 'next', ArrowLeft: e.shiftKey ? 'prevStep' : 'prev', ']': 'nextIter', '[': 'prevIter', Home: 'first', End: 'last', o: 'out' };
      if (map[e.key]) {
        e.preventDefault();
        move(map[e.key]);
      } else if (e.key === 'Enter' && canOpen(node) && (!t || t === document.body || rootRef.current?.contains(t)) && !(t && (t.tagName === 'BUTTON' || t.tagName === 'A'))) {
        e.preventDefault();
        setOpen(node, !isOpen(node));
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  // Play: one stop per PLAY_STOP_MS / speed, stopping at the last
  const last = stops.length - 1;
  useEffect(() => {
    if (!playing) return;
    if (mode !== 'timeline' || !run) return setPlaying(false);
    if (pos >= last) return setPlaying(false);
    const t = setTimeout(() => setPos((p) => Math.min(p + 1, last)), PLAY_STOP_MS / speed);
    return () => clearTimeout(t);
  }, [playing, pos, speed, last, mode, run]);
  const togglePlay = useCallback(() => {
    if (playing) return setPlaying(false);
    if (pos >= last) setPos(0);
    setPlaying(true);
  }, [playing, pos, last]);
  // Space plays and pauses; any other key, and any click outside Play and its speed menu, pauses
  useEffect(() => {
    if (!run || mode !== 'timeline') return;
    const onKey = (e) => {
      const t = e.target;
      const field = t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable);
      if (e.key === ' ' && !field && !(t && (t.tagName === 'BUTTON' || t.tagName === 'A'))) {
        e.preventDefault();
        togglePlay();
      } else if (playing && e.key !== ' ' && !['Shift', 'Control', 'Alt', 'Meta'].includes(e.key)) setPlaying(false);
    };
    const onDown = (e) => {
      if (playing && !(playRef.current && playRef.current.contains(e.target))) setPlaying(false);
    };
    window.addEventListener('keydown', onKey, true);
    window.addEventListener('pointerdown', onDown, true);
    return () => {
      window.removeEventListener('keydown', onKey, true);
      window.removeEventListener('pointerdown', onDown, true);
    };
  }, [run, mode, playing, togglePlay]);
  // glide while playing; with reduced motion, playing moves the highlight and not the panes
  const glide = playing ? (reducedMotion() ? 'still' : 'ease') : null;

  const modeTabs = phase.what === 'static' ? null : (
    <div className="ex-modes" role="tablist" aria-label="How to read the run">
      <button type="button" role="tab" aria-selected={mode === 'read'} onClick={() => setMode('read')}>Read</button>
      <button type="button" role="tab" aria-selected={mode === 'timeline'} onClick={() => setMode('timeline')}>Timeline</button>
      <button type="button" role="tab" aria-selected={mode === 'notebook'} onClick={() => setMode('notebook')}>Notebook</button>
    </div>
  );

  // the reading view needs only the bundle; the run's values fill in when the recording is done
  if (ix && mode === 'read' && phase.what !== 'error') {
    return (
      <div className="ex" ref={rootRef}>
        {run ? <RecordedLine run={run} /> : phase.what === 'static' ? <p className="ex-recorded ex-static">{quiet}</p> : <p className="ex-recorded" role="status">{phase.detail}{phase.what === 'recording' ? ` · ${elapsed.toFixed(0)} s` : ''}</p>}
        {modeTabs}
        <ReadView ix={ix} run={run} katex={katex} recording={phase.what === 'ready' || phase.what === 'static' ? null : phase.detail} unrun={phase.what === 'static'} />
      </div>
    );
  }

  if (phase.what !== 'ready') {
    return (
      <div className="ex-loading" role="status" aria-live="polite">
        {phase.what === 'error' ? (
          <p className="ex-error">The run could not be recorded: {phase.detail}</p>
        ) : (
          <>
            <div className="ex-spinner" aria-hidden="true" />
            <p>{phase.detail}{phase.what === 'recording' ? ` · ${elapsed.toFixed(0)} s` : ''}</p>
            <p className="ex-quiet">viv.wasm runs the program in your browser. Nothing is sent to a server.</p>
          </>
        )}
      </div>
    );
  }

  const { rec, tree } = run;
  const path = pathOf(node);
  const gid = graphIdOf(node, ix);
  const call = callOf(node);

  return (
    <div className="ex" ref={rootRef}>
      <RecordedLine run={run} />
      {modeTabs}

      {mode === 'notebook' ? (
        <Notebook ix={ix} tree={tree} events={rec.events} printed={rec.printed} katex={katex} onOpen={(n) => { goNode(n); setMode('timeline'); }} />
      ) : (
        <>
          <Controls stops={stops} pos={Math.min(pos, stops.length - 1)} setPos={setPos} move={move} node={node} ix={ix} path={path} goNode={goNode}
            canOpen={canOpen(node)} open={isOpen(node)} toggle={() => setOpen(node, !isOpen(node))}
            playing={playing} togglePlay={togglePlay} speed={speed} setSpeed={setSpeed} playRef={playRef} />
          <div className="ex-grid">
            <section className="ex-pane ex-graph-pane" aria-label="Where you are in the program">
              <h2 className="ex-h">The graph of <code>{defLabel(ix, call.def)}</code></h2>
              <Graph ix={ix} tree={tree} def={call.def} current={gid} path={path} node={node} goNode={goNode} glide={glide} />
            </section>
            <section className="ex-pane ex-intent-pane" aria-label="What this part is for">
              <h2 className="ex-h">Intent</h2>
              <Intent ix={ix} node={node} goNode={goNode} tree={tree} katex={katex} />
            </section>
            <section className="ex-pane ex-code-pane" aria-label="The code that ran">
              <h2 className="ex-h">Code</h2>
              <Code ix={ix} node={node} glide={glide} />
            </section>
            <section className="ex-pane ex-values-pane" aria-label="Values at this point">
              <h2 className="ex-h">Values</h2>
              <Values ix={ix} node={node} events={rec.events} goNode={goNode} />
            </section>
          </div>
          {ix.bundle.digits && <DigitsLens digits={ix.bundle.digits} />}
          <section className="ex-printed">
            <h3 className="ex-printed-title">What the program printed</h3>
            <pre className="dump">{printedText(rec.printed) || '(nothing)'}</pre>
          </section>
        </>
      )}
    </div>
  );
}

/** The digits lens on a source line: the fewest digits a printed number
 * keeps when this line alone is perturbed. */
function DigitsMark({ line }) {
  const worst = line.effects.reduce((m, e) => (e.digits < m.digits ? e : m), line.effects[0]);
  const sev = worst.digits < 6 ? 'bad' : worst.digits < 12 ? 'mid' : 'ok';
  const title = line.effects.map((e) => `#${e.output + 1} keeps ${e.digits.toFixed(1)} digits (${e.by})`).join('; ');
  return <span className={`ex-dg-mark ex-dg-${sev}`} title={`Perturbed alone, this line moves printed numbers: ${title}`}>{worst.digits.toFixed(1)}</span>;
}

/** The digits lens (viv digits): each printed number with its stable digits
 * apart from the ones a perturbed run changes. */
function DigitsLens({ digits }) {
  const m = digits.modes;
  return (
    <details className="ex-printed ex-digits" open>
      <summary>Which printed digits are real</summary>
      <p className="ex-quiet">
        From <code>viv digits</code>: {digits.runs} runs with the reductions reordered{m.fma ? ', a*b + c fused' : ''}
        {m.mca !== 'off' ? ` and Monte Carlo arithmetic (${m.samples} samples)` : ''}. The leading digits that stayed the same are in the first color; the marks beside the code say how many digits a line leaves when it alone is perturbed.
      </p>
      <ol className="ex-dg-list">
        {digits.outputs.map((o, k) => {
          const [ok, rest] = splitDigits(o.value, o.stable);
          return (
            <li key={k}>
              <code><span className="ex-dg-ok">{ok}</span><span className="ex-dg-bad">{rest}</span></code>
              <span className="ex-quiet"> {o.label ? <code>{o.label}</code> : `“${o.text.trim().slice(0, 60)}”`}: {o.stable >= 15.95 ? 'all digits' : `${o.stable.toFixed(1)} digits`}</span>
            </li>
          );
        })}
      </ol>
    </details>
  );
}

/** What the program printed as text: its figure records are drawn elsewhere, not listed. */
function printedText(printed) {
  const { blocks, charts } = parseOwnOutput(printed || '');
  const text = blocks.filter((b) => b.type === 'text').map((b) => b.s).join('\n');
  return charts.length ? `${text}${text ? '\n' : ''}(and ${charts.length} ${charts.length === 1 ? 'figure' : 'figures'}, drawn from its figure records)` : text;
}

function RecordedLine({ run }) {
  const { rec } = run;
  return (
    <p className="ex-recorded">
      Recorded in your browser in {run.seconds.toFixed(1)} s{run.whole ? ' (two passes, the second every pass line by line)' : run.passes === 2 ? ' (two passes)' : ''}: {rec.total.toLocaleString('en-US')} events
      {rec.total > rec.events.length ? ` (the first ${rec.events.length.toLocaleString('en-US')} kept)` : ''}, {(rec.bytes / 1024).toFixed(0)} KB.{' '}
      {rec.fails || rec.holdFails ? <strong className="ex-bad">{rec.fails} failed asserts, {rec.holdFails} failed holds.</strong> : 'Every hold passed.'}
    </p>
  );
}

/** Intent text with its math typeset. */
function Md({ text, katex, as: Tag = 'p', className }) {
  const html = useMemo(() => renderProse(text || '', katex), [text, katex]);
  return <Tag className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}

/** A step's or a def's why, how, equations and run lines, as the timeline and the notebook show them. */
function Details({ dets, katex }) {
  return dets.map((d, k) => {
    if (d.label === '$$') {
      const { tex, holds, tests } = equationParts(d.text);
      return (
        <figure key={k} className="ex-eq">
          <Md as="div" text={`$$${tex}$$`} katex={katex} />
          {(holds.length > 0 || tests.length > 0) && (
            <figcaption className="ex-quiet">
              {holds.map((h) => <span key={h}>checked by the hold “{h}”. </span>)}
              {tests.map((t) => <span key={t}>checked by the test “{t}”. </span>)}
            </figcaption>
          )}
        </figure>
      );
    }
    const label = { why: 'Why', how: 'How', run: 'In the run', predict: 'Predict' }[d.label] || null;
    return (
      <div key={k} className="ex-stepdet">
        {label && <p className="ex-label">{label}</p>}
        {d.text.split('\n').map((p, j) => <Md key={j} text={p.replace(/\{([A-Za-z_]\w*)[^}]*\}/g, '`$1`')} katex={katex} />)}
      </div>
    );
  });
}

// ---------------------------------------------------------------------------
// Moving
// ---------------------------------------------------------------------------

function Controls({ stops, pos, setPos, move, node, ix, path, goNode, canOpen, open, toggle, playing, togglePlay, speed, setSpeed, playRef }) {
  const label = labelOf(node, ix);
  const crumbs = path.filter((n) => n.kind === 'call' || n.kind === 'step' || n.kind === 'iter' || n.kind === 'loop');
  return (
    <div className="ex-controls">
      <nav className="ex-path" aria-label="Where you are">
        <ol>
          {crumbs.map((n) => (
            <li key={n.key}>
              <button type="button" className={n === node ? 'on' : ''} onClick={() => goNode(n)}>
                {n.kind === 'call' ? (n.parent ? defLabel(ix, n.def) : 'run') : n.kind === 'iter' ? `pass ${n.n}` : n.kind === 'loop' ? 'loop' : n.id}
              </button>
            </li>
          ))}
          {node.kind === 'line' && <li><span>{label}</span></li>}
        </ol>
      </nav>
      <div className="ex-buttons">
        <div className="ex-bgroup" role="group" aria-label="By position">
          <button type="button" onClick={() => move('prev')} disabled={pos === 0} aria-keyshortcuts="ArrowLeft">Back</button>
          <button type="button" onClick={() => move('next')} disabled={pos >= stops.length - 1} aria-keyshortcuts="ArrowRight" className="ex-primary">Next</button>
        </div>
        <div className="ex-bgroup ex-play" role="group" aria-label="Play" ref={playRef}>
          <button type="button" className="ex-playbtn" onClick={togglePlay} aria-pressed={playing} aria-keyshortcuts="Space" disabled={stops.length < 2}>
            {playing ? 'Pause' : 'Play'}
          </button>
          <select className="ex-speed" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="Playing speed">
            {PLAY_SPEEDS.map((v) => <option key={v} value={v}>{v === 0.5 ? '0.5×' : `${v}×`}</option>)}
          </select>
        </div>
        <div className="ex-bgroup" role="group" aria-label="By step">
          <button type="button" onClick={() => move('prevStep')} aria-keyshortcuts="Shift+ArrowLeft">Previous step</button>
          <button type="button" onClick={() => move('nextStep')} aria-keyshortcuts="Shift+ArrowRight">Next step</button>
        </div>
        <div className="ex-bgroup" role="group" aria-label="By pass">
          <button type="button" onClick={() => move('prevIter')} aria-keyshortcuts="[">Previous pass</button>
          <button type="button" onClick={() => move('nextIter')} aria-keyshortcuts="]">Next pass</button>
        </div>
        <button type="button" onClick={() => move('out')} aria-keyshortcuts="o" disabled={!callOf(node)?.parent}>Out of this call</button>
        {canOpen && (
          <button type="button" className="ex-open" onClick={toggle} aria-expanded={open} aria-keyshortcuts="Enter">
            {open ? 'Close' : node.kind === 'loop' ? 'Open the loop' : 'Open'}
          </button>
        )}
      </div>
      <div className="ex-scrub">
        <input type="range" min="0" max={Math.max(0, stops.length - 1)} value={pos} onChange={(e) => setPos(Number(e.target.value))} aria-label="Position in the run" aria-valuetext={label} />
        <span className="ex-where" aria-live="polite">
          <b>{label}</b> <span className="ex-quiet">· position {pos + 1} of {stops.length}</span>
        </span>
      </div>
      <p className="ex-keys ex-quiet">
        Keys: <kbd>Space</kbd> plays and pauses, <kbd>←</kbd> <kbd>→</kbd> one position, <kbd>Shift</kbd>+arrows one step, <kbd>[</kbd> <kbd>]</kbd> one pass, <kbd>o</kbd> out of the call, <kbd>Enter</kbd> opens or closes a loop.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// The graph
// ---------------------------------------------------------------------------

function Graph({ ix, tree, def, current, path, node, goNode, glide = null }) {
  const wrap = useRef(null);
  const boxes = useRef(new Map());
  const [arcs, setArcs] = useState([]);
  const defNode = ix.nodes.get(def);
  const kids = (id) => (ix.kids.get(id) || []).filter((n) => n.kind === 'step' || n.kind === 'loop');
  const lit = new Set();
  for (let g = ix.nodes.get(current); g; g = g.parent ? ix.nodes.get(g.parent) : null) lit.add(g.id);
  const edges = ix.bundle.edges.filter((e) => e.kind === 'data' && ix.nodes.get(e.to)?.def === def);

  // occurrences in the run, for jumping from a box
  const occurrences = useMemo(() => {
    const m = new Map();
    const go = (n) => {
      const g = n.kind === 'step' ? `${n.def}/${n.id}` : n.kind === 'loop' ? n.graphId : n.kind === 'call' ? n.def : null;
      if (g) {
        if (!m.has(g)) m.set(g, []);
        m.get(g).push(n);
      }
      for (const c of n.children) go(c);
    };
    go(tree);
    return m;
  }, [tree]);
  const jump = (gidTo) => {
    const occ = occurrences.get(gidTo) || [];
    if (!occ.length) return;
    const next = occ.find((n) => n.i > node.i) || occ[0];
    goNode(next);
  };

  // keep the current box in view inside the pane
  useLayoutEffect(() => {
    const b = boxes.current.get(current);
    const pane = wrap.current && wrap.current.closest('.ex-graph-pane');
    if (!b || !pane || pane.scrollHeight <= pane.clientHeight || glide === 'still') return;
    const pr = pane.getBoundingClientRect();
    const br = b.getBoundingClientRect();
    if (br.top < pr.top + 40 || br.bottom > pr.bottom - 20) scrollPane(pane, pane.scrollTop + br.top - pr.top - pane.clientHeight / 3, glide === 'ease');
  }, [current]);

  useLayoutEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const measure = () => {
      const top = el.getBoundingClientRect().top;
      const at = (id) => {
        const b = boxes.current.get(id);
        if (!b) return null;
        const r = b.getBoundingClientRect();
        return r.top - top + 18;
      };
      setArcs(
        edges
          .map((e) => ({ e, y1: at(e.from), y2: at(e.to) }))
          .filter((a) => a.y1 !== null && a.y2 !== null && a.y1 !== a.y2)
      );
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [def, current]);

  const Box = ({ n }) => {
    const here = n.id === current;
    const on = lit.has(n.id);
    const occ = occurrences.get(n.id) || [];
    const into = edges.filter((e) => e.to === n.id);
    if (n.kind === 'loop') {
      return (
        <div className={`ex-gloop${on ? ' on' : ''}${here ? ' here' : ''}`} ref={(el) => el && boxes.current.set(n.id, el)}>
          <button type="button" className="ex-gloop-head" onClick={() => jump(n.id)} disabled={!occ.length} aria-current={here ? 'location' : undefined}>
            <code>{n.header}</code>
            {occ.length > 0 && <span className="ex-quiet"> · {occ.map((l) => l.children.length).reduce((a, b) => a + b, 0)} passes</span>}
          </button>
          {kids(n.id).map((c) => <Box key={c.id} n={c} />)}
        </div>
      );
    }
    const text = ix.bundle.steps[n.id] || n.text;
    return (
      <div className={`ex-gbox${on ? ' on' : ''}${here ? ' here' : ''}`} ref={(el) => el && boxes.current.set(n.id, el)}>
        <button type="button" className="ex-gbox-head" onClick={() => jump(n.id)} disabled={!occ.length} aria-current={here ? 'location' : undefined}
          title={occ.length ? `ran ${occ.length} ${occ.length === 1 ? 'time' : 'times'}; go to the next time` : 'did not run'}>
          <b>{n.step}</b>
          {text && <span className="ex-gtext">{text}</span>}
        </button>
        <dl className="ex-io">
          {n.reads.length > 0 && (<><dt>in</dt><dd>{n.reads.map((v) => <code key={v}>{v}</code>)}</dd></>)}
          {n.writes.length > 0 && (<><dt>out</dt><dd>{n.writes.map((v) => <code key={v}>{v}</code>)}</dd></>)}
          {n.calls.length > 0 && (
            <>
              <dt>calls</dt>
              <dd>
                {n.calls.map((c) => (occurrences.get(c) ? <button type="button" key={c} className="ex-callchip" onClick={() => jump(c)}>{c}</button> : <code key={c}>{c}</code>))}
              </dd>
            </>
          )}
        </dl>
        {into.length > 0 && (
          <p className="ex-from ex-quiet">
            {into.map((e, k) => (
              <span key={k}>{k ? '; ' : ''}{e.vars.join(', ')} from <b>{ix.nodes.get(e.from)?.step}</b>{e.carried ? ' (previous pass)' : ''}</span>
            ))}
          </p>
        )}
      </div>
    );
  };

  const stack = path.filter((n) => n.kind === 'call');
  const top = kids(def);
  const intent = ix.bundle.intent[def];
  return (
    <div className="ex-graph">
      <ol className="ex-stack" aria-label="Call stack">
        {stack.map((c) => (
          <li key={c.key}><button type="button" onClick={() => goNode(c)} className={c.def === def ? 'on' : ''}>{c.parent ? defLabel(ix, c.def) : '<main>'}</button></li>
        ))}
      </ol>
      {defNode && (
        <p className="ex-gdef">
          <code>{def === '<main>' ? 'the top level' : `${def}(${defNode.params.join(', ')})`}</code>
          {intent && intent.module && <span className="ex-quiet"> · {intent.module}.viv</span>}
          {defNode.writes.length > 0 && <span className="ex-quiet"> · writes into {defNode.writes.join(', ')}</span>}
        </p>
      )}
      {top.length ? (
        <div className="ex-gflow" ref={wrap}>
          <svg className="ex-arcs" aria-hidden="true" width="26" height="100%">
            {arcs.map(({ e, y1, y2 }, k) => {
              const hot = e.from === current || e.to === current;
              const bulge = Math.min(22, 6 + Math.abs(y2 - y1) / 30);
              return (
                <path key={k} className={`${e.carried ? 'carried' : ''}${hot ? ' hot' : ''}`}
                  d={`M 26 ${y1} C ${26 - bulge} ${y1}, ${26 - bulge} ${y2}, 26 ${y2}`}>
                  <title>{`${e.vars.join(', ')}: ${ix.nodes.get(e.from)?.step} to ${ix.nodes.get(e.to)?.step}${e.carried ? ', on the next pass' : ''}`}</title>
                </path>
              );
            })}
          </svg>
          <div className="ex-gboxes">{top.map((n) => <Box key={n.id} n={n} />)}</div>
        </div>
      ) : (
        <p className="ex-quiet">{def === '<main>' ? 'The top level has no steps: it sets up the input and calls the defs above.' : `${def} has no steps.`}</p>
      )}
      {defNode && defNode.calls.length > 0 && (
        <p className="ex-gcalls ex-quiet">
          Calls:{' '}
          {defNode.calls.filter((c) => occurrences.get(c)).map((c) => <button type="button" key={c} className="ex-callchip" onClick={() => jump(c)}>{c}</button>)}
          {(() => {
            const k = defNode.calls.filter((c) => !occurrences.get(c)).length;
            const some = defNode.calls.some((c) => occurrences.get(c));
            return k > 0 ? <>{some ? ' and ' : ''}{k} {k === 1 ? 'def' : 'defs'} without steps</> : null;
          })()}
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Intent
// ---------------------------------------------------------------------------

function holdsIn(n) {
  const out = [...n.holds];
  if (n.kind !== 'line') for (const c of n.children) if (c.kind === 'step' || c.kind === 'iter') out.push(...holdsIn(c));
  return out;
}

function Intent({ ix, node, goNode, tree, katex }) {
  const call = callOf(node);
  const def = call.def;
  const intent = ix.bundle.intent[def] || {};
  const step = stepOf(node);
  const stepText = step ? ix.bundle.steps[`${step.def}/${step.id}`] : null;
  const gloss = new Map((intent.glossary || []).map(([k, t]) => [k, t]));
  const row = node.kind === 'line' ? ix.rows.get(`${node.def}:${node.line}`) : null;
  const holds = node.kind === 'line' ? [] : holdsIn(node);
  const dets = ix.details.get(def) || [];
  const stepDet = step ? dets.filter((d) => d.step === step.id && d.label !== '=' && !d.under) : [];
  const lineDet = node.kind === 'line' && row ? dets.filter((d) => d.label === '=' && d.line !== null && d.line >= row.line && d.line <= row.end) : [];
  const lensAt = step ? ix.debugAt.get(`${def}/${step.id}`) || [] : [];
  return (
    <div className="ex-intent">
      {node.kind === 'loop' || node.kind === 'iter' ? (
        <LoopSummary ix={ix} loop={node.kind === 'loop' ? node : node.parent} cur={node.kind === 'iter' ? node : null} goNode={goNode} />
      ) : null}
      {step && (
        <div className="ex-steptext">
          <p className="ex-label">Step <code>{step.id}</code></p>
          {stepText ? <Md className="ex-big" text={stepText} katex={katex} /> : <p className="ex-big"><span className="ex-quiet">(no text for this step yet)</span></p>}
          <Details dets={stepDet} katex={katex} />
        </div>
      )}
      {lineDet.length > 0 && (
        <div className="ex-linedet">
          <p className="ex-label">This line</p>
          {lineDet.map((d, k) => (
            <div key={k}>{lineDet.length > 1 || (row && row.kind === 'loop') ? <p className="ex-quiet"><code>= {d.anchor}</code></p> : null}{d.text.split('\n').map((p, j) => <Md key={j} text={p} katex={katex} />)}<Details dets={dets.filter((x) => x.under === d.anchor && x.step === d.step)} katex={katex} /></div>
          ))}
        </div>
      )}
      {node.kind === 'line' && (
        <div className="ex-line-expl">
          {row && row.kind === 'loop' && <p className="ex-note">A loop without steps: it is one line here, however many times it went round. What it assigned is under Values.</p>}
          {row && row.kind === 'exit' && <p className="ex-note">Leaves {row.step ? 'the step and ' : ''}the {def === '<main>' ? 'program' : 'def'} here.</p>}
          {Object.keys(node.values).filter((k) => gloss.has(k)).map((k) => (
            <p key={k}><code>{k}</code> <Md as="span" text={gloss.get(k)} katex={katex} /></p>
          ))}
          {!step && !lineDet.length && !Object.keys(node.values).some((k) => gloss.has(k)) && row && row.kind !== 'loop' && <p className="ex-quiet">No intent entry for this line.</p>}
        </div>
      )}
      {node.kind === 'group' && (
        <p>Before the program's own code runs, the modules it uses set their constants and tables: {node.modules.join(', ')}. Open it to go through them line by line.</p>
      )}
      {holds.length > 0 && (
        <div className="ex-holds">
          <p className="ex-label">Holds checked here</p>
          <ul>
            {dedupeHolds(holds).map((h, k) => (
              <li key={k} className={h.ok ? 'ok' : 'bad'}>
                <span className="ex-dot" aria-hidden="true" />
                <span>{h.ok ? 'held' : 'FAILED'}{h.count > 1 ? ` ${h.count} times` : ''}: {h.sentence}{h.worst !== null && h.worst !== undefined ? ` (worst ${numText(h.worst)})` : ''}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {lensAt.length > 0 && node.kind === 'step' && (
        <p className="ex-quiet">Debug lens at this step: {lensAt.map((d) => d.text).join('; ')}</p>
      )}
      <div className="ex-defintent">
        <p className="ex-label">{def === '<main>' ? 'The program' : <>The def <code>{def}</code></>}</p>
        {intent.what ? <Md text={intent.what} katex={katex} /> : <p className="ex-quiet">No intent written for {def === '<main>' ? 'the top level' : 'this def'}.</p>}
        {(intent.how || intent.why) && (
          <details>
            <summary>How and why</summary>
            {intent.how && <Md text={`**How.** ${intent.how}`} katex={katex} />}
            {intent.why && <Md text={`**Why.** ${intent.why}`} katex={katex} />}
          </details>
        )}
      </div>
    </div>
  );
}

function dedupeHolds(hs) {
  const m = new Map();
  for (const h of hs) {
    const k = `${h.ok}|${h.sentence}`;
    if (m.has(k)) {
      const x = m.get(k);
      x.count++;
      if (h.worst !== null && typeof h.worst === 'number' && (x.worst === null || h.worst > x.worst)) x.worst = h.worst;
    } else m.set(k, { ...h, count: 1 });
  }
  return [...m.values()].sort((a, b) => a.ok - b.ok);
}

function LoopSummary({ ix, loop, cur, goNode }) {
  const g = loop.graphId && ix.nodes.get(loop.graphId);
  const series = useMemo(() => {
    const names = new Map();
    for (const it of loop.children) for (const [k, v] of Object.entries(iterValues(it))) if (typeof v === 'number') names.set(k, (names.get(k) || 0) + 1);
    // a pass counter (1, 2, 3, ...) says nothing the pass numbers do not
    const counter = (k) => loop.children.every((it) => iterValues(it)[k] === it.n || iterValues(it)[k] === it.n - 1);
    return [...names.entries()].filter(([k, c]) => c >= Math.min(2, loop.children.length) && !counter(k)).map(([k]) => k).slice(0, 4);
  }, [loop]);
  return (
    <div className="ex-loopsum">
      <p className="ex-label">Loop <code>{g ? g.header : ''}</code></p>
      <p className="ex-big">{loop.children.length} {loop.children.length === 1 ? 'pass' : 'passes'}{cur ? `; this is pass ${cur.n}` : ''}.</p>
      {series.map((k) => {
        const pts = loop.children.map((it) => ({ x: it.n, y: iterValues(it)[k], it })).filter((p) => typeof p.y === 'number');
        return (
          <div key={k} className="ex-series">
            <span className="ex-sname"><code>{k}</code> per pass</span>
            <Spark pts={pts} cur={cur ? cur.n : null} onPick={(p) => goNode(p.it)} label={`${k} per pass`} log={pts.every((p) => p.y > 0) && Math.max(...pts.map((p) => p.y)) / Math.min(...pts.map((p) => p.y)) > 1e3} />
            {pts.length > 0 && <span className="ex-quiet ex-range">{numText(pts[0].y)} to {numText(pts[pts.length - 1].y)}{pts.every((p) => p.y > 0) && Math.max(...pts.map((p) => p.y)) / Math.min(...pts.map((p) => p.y)) > 1e3 ? ', log scale' : ''}</span>}
          </div>
        );
      })}
      <p className="ex-label">Passes worth reading</p>
      <ul className="ex-notable">
        {[...loop.notable.entries()].sort((a, b) => a[0] - b[0]).map(([n, why]) => {
          const it = loop.children[n - 1];
          return (
            <li key={n}>
              <button type="button" onClick={() => goNode(it)} aria-current={cur && cur.n === n ? 'step' : undefined}>pass {n}</button>{' '}
              <span>{why.join('; ')}</span>
              {hasLines(it) ? '' : <span className="ex-quiet"> (steps only)</span>}
            </li>
          );
        })}
      </ul>
      {loop.children.length > loop.notable.size && (
        <details className="ex-allpasses">
          <summary>Any pass</summary>
          <div className="ex-passes">
            {loop.children.map((it) => (
              <button type="button" key={it.key} onClick={() => goNode(it)} aria-current={cur === it ? 'step' : undefined}>{it.n}</button>
            ))}
          </div>
          <p className="ex-quiet">Passes other than these were recorded step by step, not line by line.</p>
        </details>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Code
// ---------------------------------------------------------------------------

function Code({ ix, node, glide = null }) {
  const box = useRef(null);
  const call = callOf(node);
  const defNode = ix.nodes.get(call.def);
  const step = stepOf(node);
  const stepG = step ? ix.nodes.get(`${step.def}/${step.id}`) : null;
  // the lines on show: the def, or for top-level code, a window around the line
  let lo;
  let hi;
  let focus = null;
  let focusEnd = null;
  if (node.kind === 'line') {
    const row = ix.rows.get(`${node.def}:${node.line}`);
    focus = node.line;
    focusEnd = row ? row.end : node.line;
  } else if (node.kind === 'step' && stepG) {
    focus = stepG.line;
    focusEnd = stepG.end;
  } else if ((node.kind === 'loop' || node.kind === 'iter')) {
    const loop = node.kind === 'loop' ? node : node.parent;
    const g = loop.graphId && ix.nodes.get(loop.graphId);
    if (g) {
      focus = g.line;
      focusEnd = g.line;
    }
  } else if (node.kind === 'group') {
    focus = node.children[0].line;
    focusEnd = focus;
  }
  if (defNode && defNode.line > 0) {
    lo = defNode.line;
    hi = defNode.end;
  } else {
    const at = focus || 1;
    const m = ix.toModule(at);
    lo = Math.max(1, at - 12);
    hi = Math.min(ix.programLines.length, at + 14);
    // stay inside the module the line came from
    while (lo < at && ix.toModule(lo)?.module !== m?.module) lo++;
    while (hi > at && ix.toModule(hi)?.module !== m?.module) hi--;
  }
  const stepLo = stepG ? stepG.line : null;
  const stepHi = stepG ? stepG.end : null;
  const mod = ix.toModule(focus || lo);
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const cur = el.querySelector('.cur');
    if (!cur || glide === 'still') return;
    scrollPane(el, cur.offsetTop - el.clientHeight / 3, glide === 'ease');
  }, [node, lo]);
  const rows = [];
  for (let x = lo; x <= hi; x++) {
    const m = ix.toModule(x);
    const cls = [focus !== null && x >= focus && x <= focusEnd ? 'cur' : '', stepLo !== null && x >= stepLo && x <= stepHi ? 'instep' : ''].join(' ').trim();
    const dl = digitsLine(ix, m);
    rows.push(
      <div key={x} className={`ex-cl ${cls}`}>
        <span className="ex-ln" aria-hidden="true">{m ? m.line : ''}</span>
        {dl && dl.effects && dl.effects.length > 0 && <DigitsMark line={dl} />}
        <code dangerouslySetInnerHTML={{ __html: highlight(ix.programLines[x - 1] || ' ', 'viv') || ' ' }} />
      </div>
    );
  }
  return (
    <div className="ex-code">
      <p className="ex-file ex-quiet">{mod ? `${mod.module}.viv` : ''}{focus && mod ? `, line ${ix.toModule(focus)?.line}` : ''}</p>
      <div className="ex-lines" ref={box} tabIndex={0} aria-label="Source">{rows}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

function Values({ ix, node, events, goNode }) {
  const { vars, changed } = useMemo(() => valuesAt(node, events), [node, events]);
  const [all, setAll] = useState(false);
  const call = callOf(node);
  const gloss = new Map((ix.bundle.intent[call.def]?.glossary || []).map(([k, t]) => [k, t]));
  if (node.kind === 'group') return <p className="ex-quiet">Open the group to see what each line set.</p>;
  const names = [...vars.keys()];
  const order = [...changed, ...names.filter((n) => !changed.includes(n)).sort((a, b) => vars.get(b).i - vars.get(a).i)];
  const shown = all ? order : order.slice(0, 8);
  if (!order.length) return <p className="ex-quiet">Nothing is watched in {call.def === '<main>' ? 'the top level' : <code>{defLabel(ix, call.def)}</code>} up to here.</p>;
  return (
    <div className="ex-values">
      {changed.length > 0 ? <p className="ex-label">Set on this line</p> : node.kind === 'line' ? <p className="ex-quiet">This line set nothing a watch can record (a number, true or false, or a real array).</p> : null}
      {shown.map((k, j) => (
        <div key={k}>
          {changed.length > 0 && j === changed.length && <p className="ex-label">Earlier in this call</p>}
          {changed.length === 0 && j === 0 && <p className="ex-label">Latest values in this call</p>}
          <Value name={k} v={vars.get(k).value} gloss={gloss.get(k)} node={node} events={events} fresh={changed.includes(k)} goNode={goNode} />
        </div>
      ))}
      {order.length > shown.length && <button type="button" className="ex-more" onClick={() => setAll(true)}>Show all {order.length}</button>}
    </div>
  );
}

function Value({ name, v, gloss, node, events, fresh }) {
  const hist = useMemo(() => (typeof v === 'number' ? historyOf(name, node, events) : null), [name, node, events, v]);
  return (
    <div className={`ex-value${fresh ? ' fresh' : ''}`}>
      <p className="ex-vhead">
        <code>{name}</code>
        {gloss && <span className="ex-gloss">{gloss.length > 90 ? `${gloss.slice(0, 88)}…` : gloss}</span>}
      </p>
      {typeof v === 'number' || typeof v === 'boolean' || typeof v === 'string' ? (
        <div className="ex-scalar">
          <span className="ex-num">{numText(v)}</span>
          {hist && hist.pts.length > 1 && <Spark pts={hist.pts} cur={hist.cur} label={`${name} ${hist.by === 'pass' ? 'per pass' : 'as it was set'}`} small />}
        </div>
      ) : (
        <ArrayView a={v} name={name} />
      )}
    </div>
  );
}

function ArrayView({ a, name }) {
  const shape = a.shape || [];
  const [slice, setSlice] = useState(0);
  if (!a.values) {
    return (
      <p className="ex-summary">
        {shape.join(' × ')} array, too large to keep whole: min {numText(a.min)}, max {numText(a.max)}, norm {numText(a.norm)}{a.trace !== undefined ? `, trace ${numText(a.trace)}` : ''}
      </p>
    );
  }
  const vals = a.values.map(Number);
  if (shape.length <= 1 || (shape.length === 2 && (shape[0] === 1 || shape[1] === 1))) {
    return <Bars vals={vals} name={name} />;
  }
  const r = shape[shape.length - 2];
  const c = shape[shape.length - 1];
  const nSlices = vals.length / (r * c);
  const s = Math.min(slice, nSlices - 1);
  const part = vals.slice(s * r * c, (s + 1) * r * c);
  return (
    <div>
      {nSlices > 1 && (
        <div className="ex-slices" role="group" aria-label={`${name}: which ${r} × ${c} slice`}>
          <span className="ex-quiet">{shape.join(' × ')}; slice</span>
          {Array.from({ length: Math.min(nSlices, 12) }, (_, k) => (
            <button type="button" key={k} aria-pressed={k === s} onClick={() => setSlice(k)}>{k}</button>
          ))}
        </div>
      )}
      <Heatmap vals={part} rows={r} cols={c} name={name} />
    </div>
  );
}

// viridis, sampled at nine points (the colormap Viv's plot2d draws with)
const VIRIDIS = [[68, 1, 84], [71, 44, 122], [59, 81, 139], [44, 113, 142], [33, 144, 141], [39, 173, 129], [92, 200, 99], [170, 220, 50], [253, 231, 37]];
function viridis(t) {
  const x = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const k = Math.min(VIRIDIS.length - 2, Math.floor(x));
  const f = x - k;
  const c = VIRIDIS[k].map((a, j) => Math.round(a + (VIRIDIS[k + 1][j] - a) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function Heatmap({ vals, rows, cols, name }) {
  const ref = useRef(null);
  const [hover, setHover] = useState(null);
  const fin = vals.filter(Number.isFinite);
  const lo = Math.min(...fin);
  const hi = Math.max(...fin);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    cv.width = cols;
    cv.height = rows;
    const g = cv.getContext('2d');
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const v = vals[i * cols + j];
      g.fillStyle = Number.isFinite(v) ? viridis(hi > lo ? (v - lo) / (hi - lo) : 0.5) : 'rgb(128,128,128)';
      g.fillRect(j, i, 1, 1);
    }
  }, [vals, rows, cols, lo, hi]);
  const onMove = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const j = Math.floor(((e.clientX - r.left) / r.width) * cols);
    const i = Math.floor(((e.clientY - r.top) / r.height) * rows);
    if (i >= 0 && j >= 0 && i < rows && j < cols) setHover({ i, j, v: vals[i * cols + j] });
  };
  return (
    <figure className="ex-heat">
      <canvas ref={ref} onMouseMove={onMove} onMouseLeave={() => setHover(null)} role="img"
        aria-label={`${name}: ${rows} by ${cols} matrix, values from ${numText(lo)} to ${numText(hi)}`} style={{ aspectRatio: `${cols} / ${rows}` }} />
      <figcaption className="ex-quiet">
        <span className="ex-ramp" aria-hidden="true" /> {numText(lo)} to {numText(hi)}
        {hover ? <> · [{hover.i}, {hover.j}] = {numText(hover.v)}</> : ` · ${rows} × ${cols}`}
      </figcaption>
    </figure>
  );
}

function Bars({ vals, name }) {
  const n = vals.length;
  const fin = vals.filter(Number.isFinite);
  const lo = Math.min(0, ...fin);
  const hi = Math.max(0, ...fin);
  const W = 300;
  const H = 70;
  const y = (v) => (hi > lo ? H - ((v - lo) / (hi - lo)) * H : H / 2);
  const bw = W / Math.max(1, n);
  return (
    <figure className="ex-bars">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label={`${name}: ${n} values from ${numText(Math.min(...fin))} to ${numText(Math.max(...fin))}`}>
        <line x1="0" x2={W} y1={y(0)} y2={y(0)} className="ex-axis" />
        {vals.map((v, k) => (
          <rect key={k} x={k * bw + bw * 0.1} width={Math.max(0.5, bw * 0.8)} y={Math.min(y(v), y(0))} height={Math.max(0.5, Math.abs(y(v) - y(0)))} className="ex-bar">
            <title>{`${name}[${k}] = ${numText(v)}`}</title>
          </rect>
        ))}
      </svg>
      <figcaption className="ex-quiet">{n} values{n <= 8 ? `: ${vals.map(numText).join(', ')}` : `, ${numText(Math.min(...fin))} to ${numText(Math.max(...fin))}`}</figcaption>
    </figure>
  );
}

function Spark({ pts, cur, onPick, label, small, log }) {
  const W = small ? 160 : 300;
  const H = small ? 34 : 64;
  const P = 4;
  const ys = pts.map((p) => (log ? Math.log10(Math.abs(p.y)) : p.y)).filter(Number.isFinite);
  if (ys.length < 2) return null;
  const lo = Math.min(...ys);
  const hi = Math.max(...ys);
  const xs = pts.map((p) => p.x);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const X = (x) => P + (x1 > x0 ? ((x - x0) / (x1 - x0)) * (W - 2 * P) : (W - 2 * P) / 2);
  const Y = (v) => {
    const t = log ? Math.log10(Math.abs(v)) : v;
    return H - P - (hi > lo ? ((t - lo) / (hi - lo)) * (H - 2 * P) : (H - 2 * P) / 2);
  };
  const d = pts.filter((p) => Number.isFinite(p.y)).map((p, k) => `${k ? 'L' : 'M'} ${X(p.x).toFixed(1)} ${Y(p.y).toFixed(1)}`).join(' ');
  const curP = pts.find((p) => p.x === cur);
  return (
    <svg className={`ex-spark${small ? ' small' : ''}`} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${label}: ${pts.length} values from ${numText(pts[0].y)} to ${numText(pts[pts.length - 1].y)}${log ? ', log scale' : ''}`}>
      <path d={d} className="ex-sline" />
      {pts.map((p) => (
        onPick ? (
          <circle key={p.x} cx={X(p.x)} cy={Y(p.y)} r={p === curP ? 4 : 2.6} className={`ex-sdot${p === curP ? ' cur' : ''}`} onClick={() => onPick(p)} style={{ cursor: 'pointer' }}>
            <title>{`pass ${p.x}: ${numText(p.y)}`}</title>
          </circle>
        ) : p === curP ? <circle key={p.x} cx={X(p.x)} cy={Y(p.y)} r="3.2" className="ex-sdot cur" /> : null
      ))}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Notebook
// ---------------------------------------------------------------------------

/** The run in the order it ran, one cell per step, loops as one cell each. */
function Notebook({ ix, tree, events, printed, onOpen, katex }) {
  const cells = [];
  const go = (n, depth) => {
    for (const c of n.children) {
      if (c.kind === 'call') {
        cells.push({ kind: 'call', node: c, depth });
        go(c, depth + 1);
      } else if (c.kind === 'step') {
        cells.push({ kind: 'step', node: c, depth });
        if (c.children.some((k) => k.kind === 'call' || k.kind === 'step' || k.kind === 'loop')) go(c, depth + 1);
      } else if (c.kind === 'loop') {
        cells.push({ kind: 'loop', node: c, depth });
        // the loop's first pass, step by step; the other passes are in the plot
        if (c.children[0]) go(c.children[0], depth + 1);
      }
    }
  };
  go(tree, 0);
  return (
    <div className="ex-nb">
      <p className="ex-quiet">The program in the order it ran: every call of a def with steps and every step, each with its intent and equations. A loop is one cell with its values per pass, followed by the steps of its first pass. Each cell's button opens the timeline at that point.</p>
      {cells.map(({ kind, node, depth }) => <Cell key={node.key} kind={kind} node={node} depth={Math.min(depth, 4)} ix={ix} events={events} onOpen={onOpen} katex={katex} />)}
      <section className="ex-printed">
        <h3 className="ex-printed-title">What the program printed</h3>
        <pre className="dump">{printedText(printed) || '(nothing)'}</pre>
      </section>
    </div>
  );
}

function Cell({ kind, node, depth, ix, events, onOpen, katex }) {
  const intent = ix.bundle.intent[node.def] || {};
  const style = { marginLeft: `${depth * 14}px` };
  const dets = ix.details.get(node.def) || [];
  if (kind === 'call') {
    const defDets = dets.filter((d) => !d.step && !d.under && d.label !== '=');
    return (
      <div className="ex-cell ex-cell-call" style={style}>
        <p><code>{defLabel(ix, node.def)}</code></p>
        {intent.what ? <Md className="ex-callwhat" text={intent.what} katex={katex} /> : null}
        {(intent.how || intent.why) && (
          <details className="ex-callmore">
            <summary>How and why</summary>
            {intent.how && <Md text={`**How.** ${intent.how}`} katex={katex} />}
            {intent.why && <Md text={`**Why.** ${intent.why}`} katex={katex} />}
          </details>
        )}
        {defDets.length > 0 && <div className="ex-callmore"><Details dets={defDets} katex={katex} /></div>}
        <button type="button" onClick={() => onOpen(node)}>Go to this call</button>
      </div>
    );
  }
  if (kind === 'loop') {
    const g = node.graphId && ix.nodes.get(node.graphId);
    const names = Object.keys(iterValues(node.children[node.children.length - 1] || node)).filter((k) => node.children.filter((it) => typeof iterValues(it)[k] === 'number').length > 1).slice(0, 3);
    return (
      <div className="ex-cell ex-cell-loop" style={style}>
        <p className="ex-label">Loop <code>{g ? g.header : ''}</code>: {node.children.length} passes</p>
        {names.map((k) => (
          <div key={k} className="ex-series">
            <span className="ex-sname"><code>{k}</code></span>
            <Spark pts={node.children.map((it) => ({ x: it.n, y: iterValues(it)[k] })).filter((p) => typeof p.y === 'number')} label={`${k} per pass`} />
          </div>
        ))}
        <button type="button" onClick={() => onOpen(node)}>Go to this loop</button>
      </div>
    );
  }
  const g = ix.nodes.get(`${node.def}/${node.id}`);
  const text = ix.bundle.steps[`${node.def}/${node.id}`];
  const end = valuesAt({ ...node, kind: 'step' }, events);
  const set = g ? g.writes.filter((w) => end.vars.has(w)) : [];
  const failed = node.holds.filter((h) => !h.ok);
  return (
    <div className="ex-cell" style={style}>
      <p className="ex-cellhead"><code>{node.def === '<main>' ? '' : `${node.def} / `}{node.id}</code></p>
      {text ? <Md className="ex-big" text={text} katex={katex} /> : <p className="ex-big"><span className="ex-quiet">(no text)</span></p>}
      <Details dets={dets.filter((d) => d.step === node.id && d.label !== '=' && !d.under)} katex={katex} />
      {g && (
        <details>
          <summary>Code, lines {ix.toModule(g.line)?.line} to {ix.toModule(g.end)?.line} of {ix.toModule(g.line)?.module}.viv</summary>
          <pre className="ex-cellcode"><code dangerouslySetInnerHTML={{ __html: highlight(ix.programLines.slice(g.line - 1, g.end).join('\n'), 'viv') }} /></pre>
        </details>
      )}
      {set.length > 0 && (
        <div className="ex-cellout">
          <p className="ex-label">Hands on</p>
          {set.map((k) => {
            const v = end.vars.get(k).value;
            return (
              <div key={k} className="ex-cellval">
                <code>{k}</code>{' '}
                {typeof v === 'object' ? <ArrayView a={v} name={k} /> : <span className="ex-num">{numText(v)}</span>}
              </div>
            );
          })}
        </div>
      )}
      {node.holds.length > 0 && <p className={failed.length ? 'ex-bad' : 'ex-quiet'}>{failed.length ? `${failed.length} of ${node.holds.length} holds failed` : `${node.holds.length} ${node.holds.length === 1 ? 'hold' : 'holds'} held`}</p>}
      <button type="button" onClick={() => onOpen(node)}>Go to this step</button>
    </div>
  );
}
