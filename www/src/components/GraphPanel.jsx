// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The panel beside the program graph (ProgramGraph.jsx): what the reader
// picked in the picture. For a def: its signature, what its intent says it
// does (what, how, why, glossary), who called it and how often, what it calls,
// and its code; for a step or a loop: its text, what it reads and writes, and
// its def's code with its own lines marked and scrolled to. Every def named
// here is a link that moves the picture to its box (opening what must be
// opened for it to be drawn).
//
// The explanation comes first, as prose (the intent, or the def's `#|` lines
// when it has no intent), cut to its first sentence when it is long. The code
// under it is code only (lib/panelCode.js): no `###` post blocks, no `#|` or
// `#:` lines, no comment paragraphs, only one-line comments of at most
// NOTE_CHARS characters; the gutter keeps the source's line numbers and marks
// where lines were left out. "Show comments" brings back the full text,
// with the comments above the def.
//
// With nothing picked, the panel says how to use the picture and lists, by
// module, the defs the program defines that this run did not call.
import { Fragment, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { highlight } from '../lib/highlight.js';
import { parseOwnOutput } from './VivCell.jsx';
import { renderProse } from '../lib/richText.js';
import { MAIN, closes, isMp, mpOf, contractWords, robustWords } from '../lib/graphView.js';
import { kindsOf, codeOnly, docAbove, inlineIntent, firstSentence } from '../lib/panelCode.js';
import { fmtValue, shapeOf, loopSeries, shortGloss, programPurpose, holdsIn, hotBelow, recordedDefs } from '../lib/tour.js';

/** The most values a stop lists; the rest are counted. */
const MAX_VALUES = 6;
/** The most lines of the program's printed output the last stop shows. */
const MAX_PRINTED = 12;

/** Where the reader's choice to see the comments is remembered, in this browser. */
const COMMENTS_KEY = 'viv.graph.comments';
/** An explanation longer than this many characters opens at its first sentence. */
const SHORT_CHARS = 220;

/** The most lines of a def the panel shows. */
const MAX_LINES = 900;

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/** Intent prose: its math typeset, a `[test "..."]` citation as a check mark with the test's name. */
function proseHtml(text, katex) {
  const cites = [];
  const t = String(text || '').replace(/\s*\[(test|hold) "([^"]*)"\]/g, (_, kind, name) => {
    cites.push([kind, name]);
    return `\u0001${cites.length - 1}\u0001`;
  });
  return renderProse(t, katex).replace(/\u0001(\d+)\u0001/g, (_, k) => {
    const [kind, name] = cites[Number(k)];
    return `<span class="pg-check" title="${esc(`checked by the ${kind} “${name}”`)}">✓</span>`;
  });
}

function Prose({ text, katex, as: Tag = 'p', className }) {
  const html = useMemo(() => proseHtml(text, katex), [text, katex]);
  return <Tag className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}

const isStepNode = (n) => n && (n.kind === 'step' || n.kind === 'loop');
const fmtCount = (n) => Number(n || 0).toLocaleString('en-US');

/** A def's name as a link that moves the picture there. */
function DefLink({ id, go, extra = null, note = '' }) {
  return (
    <button type="button" className="pg-link" onClick={() => go(id)} title={note || `Show ${id} in the picture`}>
      <code>{id === MAIN ? 'the program' : id}</code>
      {extra}
    </button>
  );
}

/**
 * The lines the panel shows for a def: its own, or for `<main>` the top-level
 * statements of the program (not its defs and records, but all of its steps). Code only (lib/panelCode.js), unless `full`:
 * then every line, with the comments above a def. At most MAX_LINES, around
 * `mark` when it is given.
 */
function linesOf(ix, def, mark, full) {
  const kinds = kindsOf(ix);
  let lines = [];
  if (def.id === MAIN) {
    // the defs written in the text (not the ones the compiler makes, such as
    // the `viv__with_...` of a record update, whose line is a statement), and the records
    const written = (n) => n.kind === 'record' || (n.kind === 'def' && n.id !== MAIN && /^\s*def\b/.test(ix.programLines[n.line - 1] || ''));
    const skip = ix.bundle.nodes.filter(written).map((n) => [n.line, n.end]).sort((a, b) => a[0] - b[0]);
    // the root module and the modules that hold the program's steps (a deck
    // such as dft/pwdft.viv), as the full reading has them; not the constants
    // at the top of every module it uses
    const mods = new Set([ix.toModule(1)?.module]);
    for (const n of ix.bundle.nodes) if (n.def === MAIN && n.kind !== 'def') mods.add(ix.toModule(n.line)?.module);
    let k = 0;
    for (let x = 1; x <= ix.programLines.length; x++) {
      while (k < skip.length && skip[k][1] < x) k++;
      if (k < skip.length && skip[k][0] <= x) {
        x = skip[k][1];
        continue;
      }
      const m = ix.toModule(x);
      if (m && mods.has(m.module)) lines.push(x);
    }
  } else {
    const above = full ? docAbove(ix.programLines, def.line) : null;
    for (let x = above ? above[0] : def.line; x <= def.end; x++) lines.push(x);
  }
  if (!full) lines = codeOnly(kinds, lines);
  // no blank lines at the ends
  while (lines.length && !(ix.programLines[lines[0] - 1] || '').trim()) lines.shift();
  while (lines.length && !(ix.programLines[lines[lines.length - 1] - 1] || '').trim()) lines.pop();
  if (lines.length > MAX_LINES) {
    const at = mark ? Math.max(0, lines.findIndex((x) => x >= mark.line) - Math.floor(MAX_LINES / 4)) : 0;
    lines = lines.slice(at, at + MAX_LINES);
  }
  return lines;
}

/** How many lines between shown lines `a` and `b` were left out, not counting blank ones. */
function hiddenBetween(ix, a, b) {
  let n = 0;
  for (let x = a + 1; x < b; x++) if ((ix.programLines[x - 1] || '').trim()) n++;
  return n;
}

/** The lines of a def, with the lines of `mark` (a step or loop) marked. */
function Code({ ix, def, mark, scrollKey, reduced, full, setFull, ownScroll = false }) {
  const ref = useRef(null);
  const lines = useMemo(() => linesOf(ix, def, mark, full), [ix, def, mark, full]);
  useLayoutEffect(() => {
    const box = ref.current && ref.current.closest('.pg-panel');
    if (!box) return;
    const behavior = reduced ? 'auto' : 'smooth';
    const first = ref.current.querySelector('.pg-cl.on');
    if (ownScroll) {
      // the walkthrough: the code block scrolls, the panel and the page stay put
      const code = ref.current;
      const top = first ? first.offsetTop - code.clientHeight * 0.25 : 0;
      const bottom = first ? first.offsetTop + first.offsetHeight : 0;
      if (first && first.offsetTop >= code.scrollTop + 4 && bottom <= code.scrollTop + code.clientHeight - 4) return;
      code.scrollTo({ top: Math.max(0, top), behavior });
      return;
    }
    // a step's lines a little below the panel's top; for a def, the panel from its top
    const target = first || ref.current.closest('.pg-sel') || ref.current;
    const lead = first ? 0.18 : 0;
    if (getComputedStyle(box).overflowY !== 'visible') {
      // a wide screen: the panel scrolls, the page does not
      if (box.scrollHeight <= box.clientHeight + 1) return;
      const top = target.getBoundingClientRect().top - box.getBoundingClientRect().top;
      if (!first && top >= 0 && top < box.clientHeight * 0.6) return;
      box.scrollTo({ top: box.scrollTop + top - box.clientHeight * lead - 8, behavior });
      return;
    }
    // a narrow screen: the panel is part of the page, under the picture, which stays at the top
    const graph = box.parentElement && box.parentElement.querySelector('.pg-left');
    const under = graph ? graph.getBoundingClientRect().bottom : 0;
    const top = target.getBoundingClientRect().top;
    const room = window.innerHeight - under;
    if (!first && top >= under && top < under + room * 0.6) return;
    window.scrollTo({ top: window.scrollY + top - under - room * lead - 8, behavior });
  }, [scrollKey, reduced]);
  const kinds = kindsOf(ix);
  // highlighted once per set of lines, not on every render of the panel
  const html = useMemo(() => new Map(lines.map((x) => [x, highlight(ix.programLines[x - 1] || ' ', 'viv') || ' '])), [ix, lines]);
  const all = def.id === MAIN ? null : def.end - def.line + 1;
  const shown = lines.filter((x) => x >= def.line && x <= def.end).length;
  const where = ix.toModule(lines[0] || def.line);
  const hidden = def.id === MAIN ? 0 : hiddenBetween(ix, def.line - 1, def.end + 1) - lines.filter((x) => x >= def.line && x <= def.end && (ix.programLines[x - 1] || '').trim()).length;
  return (
    <div className="pg-codebox">
      <div className="pg-codebar">
        <span className="pg-file">{where ? `${where.module}.viv` : ''}</span>
        <label className="pg-toggle" title="Show the source as written: the comments, the intent lines and the post's prose">
          <input type="checkbox" className="pg-comments" checked={full} onChange={(e) => setFull(e.target.checked)} /> show comments
        </label>
      </div>
      <div className="pg-code" ref={ref} data-from={lines.find((x) => x >= def.line) || def.line} data-def={def.id} data-full={full ? '1' : ''}>
        <div className="pg-lines">
          {lines.map((x, k) => {
            const m = ix.toModule(x);
            const on = mark && x >= mark.line && x <= mark.end;
            const gap = k > 0 && x !== lines[k - 1] + 1 ? hiddenBetween(ix, lines[k - 1], x) : 0;
            const other = gap && m && ix.toModule(lines[k - 1])?.module !== m.module;
            const kind = kinds[x - 1];
            return (
              <Fragment key={x}>
                {gap > 0 && (
                  <div className="pg-gap" title={other ? `${m.module}.viv` : `${gap} line${gap === 1 ? '' : 's'} of comments or prose left out`}>
                    <span className="pg-ln">⋯</span>
                    <span>{other ? `${m.module}.viv` : ''}</span>
                  </div>
                )}
                <div className={`pg-cl${on ? ' on' : ''}${full && kind !== 'code' && kind !== 'note' && kind !== 'blank' ? ' pg-prose' : ''}`} data-line={x} data-kind={kind}>
                  <span className="pg-ln" aria-hidden="true">{m ? m.line : ''}</span>
                  <code dangerouslySetInnerHTML={{ __html: html.get(x) }} />
                </div>
              </Fragment>
            );
          })}
        </div>
      </div>
      {all !== null && !full && hidden > 0 && <p className="ex-quiet pg-more">{hidden} line{hidden === 1 ? '' : 's'} of comments left out.</p>}
      {all !== null && shown < all && full && <p className="ex-quiet pg-more">{all - shown} more lines; the full reading below has them all.</p>}
    </div>
  );
}

/**
 * Prose that opens at its first sentence when it is long: `head` is always
 * shown, `rest` (any nodes) behind "more".
 */
function Explain({ it, stepText, katex, resetKey }) {
  const [open, setOpen] = useState(false);
  useEffect(() => setOpen(false), [resetKey]);
  const lead = stepText || it.what || '';
  const extra = (it.how || '').length + (it.why || '').length + (it.glossary || []).reduce((s, [n, t]) => s + n.length + t.length, 0);
  const long = lead.length + extra > SHORT_CHARS;
  const [first, rest] = long ? firstSentence(lead) : [lead, ''];
  const more = rest || extra > 0;
  if (!lead && !extra) return null;
  const body = (
    <>
      {it.how && <><p className="pg-k">How</p><Prose text={it.how} katex={katex} /></>}
      {it.why && <><p className="pg-k">Why</p><Prose text={it.why} katex={katex} /></>}
      {(it.glossary || []).length > 0 && (
        <dl className="pg-gloss">
          {it.glossary.map(([name, text]) => (
            <div key={name}><dt><code>{name}</code></dt><dd><Prose as="span" text={text} katex={katex} /></dd></div>
          ))}
        </dl>
      )}
    </>
  );
  if (!long) {
    return (
      <div className="pg-explain">
        {lead && <Prose className="pg-what" text={lead} katex={katex} />}
        {body}
      </div>
    );
  }
  return (
    <div className="pg-explain" data-open={open ? '1' : ''}>
      <Prose className="pg-what" text={open ? lead : first} katex={katex} />
      {open && body}
      {more && (
        <button type="button" className="pg-morebtn" aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? 'less' : 'more'}
        </button>
      )}
    </div>
  );
}

function Names({ label, names }) {
  if (!names || !names.length) return null;
  return (
    <div className="pg-names pg-chips">
      <span className="pg-k">{label}</span>
      {names.map((n) => <code key={n} className="pg-chip">{n}</code>)}
    </div>
  );
}

export default function GraphPanel({ ix, katex, model, shown, open, sel: pick, go, toggle, onRead, reduced, tour = null }) {
  const bundle = ix.bundle;
  // during the walkthrough the panel tells the stop, unless the reader opened a def from it
  const sel = tour ? tour.peek : pick;
  const node = sel ? model.byId.get(sel) : null;
  const def = node ? (node.kind === 'def' ? node : model.defs.get(node.def)) : null;
  const panelTop = useRef(null);
  useEffect(() => {
    if (!def && panelTop.current) panelTop.current.closest('.pg-panel')?.scrollTo({ top: 0 });
  }, [def]);
  // code only unless the reader asked for the comments (remembered in this browser)
  const [full, setFullState] = useState(false);
  useEffect(() => {
    try {
      if (localStorage.getItem(COMMENTS_KEY) === '1') setFullState(true);
    } catch { /* no storage: code only */ }
  }, []);
  const setFull = (v) => {
    setFullState(v);
    try {
      localStorage.setItem(COMMENTS_KEY, v ? '1' : '0');
    } catch { /* not remembered */ }
  };

  if (tour && !tour.peek) return <TourStop ix={ix} katex={katex} model={model} tour={tour} reduced={reduced} full={full} setFull={setFull} />;
  if (sel && isMp(sel) && model.mps.get(mpOf(sel))) return <MultiPath mp={model.mps.get(mpOf(sel))} member={sel.includes('/m:') ? sel.split('/m:')[1] : null} open={open} go={go} toggle={toggle} katex={katex} refTop={panelTop} />;
  if (!node || !def) return <Intro ix={ix} model={model} shown={shown} go={go} refTop={panelTop} />;
  const it = (bundle.intent || {})[def.id] || {};
  // no intent: the def's `#|` lines, as prose
  const explained = it.what || it.how || it.why || def.id === MAIN ? it : { ...it, ...inlineIntent(ix.programLines, def.line) };
  const where = ix.toModule(def.id === MAIN ? (isStepNode(node) ? node.line : 1) : def.line);
  const callers = (model.callers.get(def.id) || []).filter((c) => model.ranSet.has(c));
  const callees = model.callees.get(def.id) || [];
  const calledTimes = callers.reduce((s, c) => s + (model.counts.get(`${c}>${def.id}`) || 0), 0);
  const hidden = callees.filter((c) => !shown.has(c));
  const canClose = closes(model, open, shown, def.id);
  const steps = (model.inner.get(def.id) || []).filter((n) => n.kind === 'step');
  const isStep = node.kind === 'step' || node.kind === 'loop';
  const stepText = node.kind === 'step' ? node.text || (bundle.steps || {})[node.id] || '' : '';
  const stepCalls = isStep ? (node.calls || []).filter((c) => callees.includes(c)) : [];
  return (
    <div className="pg-sel" ref={panelTop}>
      {tour && <button type="button" className="pg-btn pg-tourback" onClick={tour.back}>◀ Back to the walk, stop {tour.stop.k + 1}</button>}
      <p className="pg-where ex-quiet">
        {where ? <><code>{where.module}.viv</code> line {where.line}</> : null}
        {isStep && <> · in <DefLink id={def.id} go={go} /></>}
      </p>
      {isStep ? (
        <>
          <h3 className="pg-title"><code>{node.kind === 'loop' ? node.header : `${def.id} / ${node.step}`}</code></h3>
          <Explain it={{}} stepText={stepText} katex={katex} resetKey={sel} />
          <Names label="reads" names={node.reads} />
          <Names label="writes" names={node.writes} />
          {stepCalls.length > 0 && (
            <div className="pg-names pg-chips">
              <span className="pg-k">calls</span>
              {stepCalls.map((c) => <DefLink key={c} id={c} go={go} extra={<span className="pg-n">×{fmtCount(model.counts.get(`${def.id}>${c}`))}</span>} />)}
            </div>
          )}
        </>
      ) : (
        <>
          <h3 className="pg-title"><code>{it.sig || `${def.id}(${(def.params || []).join(', ')})`}</code></h3>
          <Explain it={explained} katex={katex} resetKey={sel} />
        </>
      )}
      <div className="pg-acts">
        {!isStep && (hidden.length > 0 || canClose) && (
          <button type="button" className="pg-btn" onClick={() => toggle(def.id)} aria-pressed={!hidden.length}>
            {hidden.length ? `Show the ${hidden.length} def${hidden.length === 1 ? '' : 's'} it calls` : 'Hide what it calls'}
          </button>
        )}
        {onRead && <button type="button" className="pg-btn pg-quiet" onClick={() => onRead(def.id)}>In the full reading ↓</button>}
      </div>
      {!isStep && (
        <div className="pg-rel pg-chips">
          {def.id !== MAIN && (
            <p>
              <span className="pg-k">Called {fmtCount(calledTimes)} time{calledTimes === 1 ? '' : 's'}</span>
              {callers.map((c) => <DefLink key={c} id={c} go={go} extra={<span className="pg-n">×{fmtCount(model.counts.get(`${c}>${def.id}`))}</span>} note={`${c} called it ${fmtCount(model.counts.get(`${c}>${def.id}`))} times`} />)}
            </p>
          )}
          {callees.length > 0 && (
            <p>
              <span className="pg-k">Calls</span>
              {callees.map((c) => (
                <DefLink key={c} id={c} go={go} extra={<span className="pg-n">×{fmtCount(model.counts.get(`${def.id}>${c}`))}{shown.has(c) ? '' : ' ·'}</span>}
                  note={shown.has(c) ? `Show ${c}` : `${c} is not drawn yet: open it in the picture`} />
              ))}
            </p>
          )}
          {steps.length > 0 && (
            <p>
              <span className="pg-k">Steps</span>
              {steps.map((s) => <button key={s.id} type="button" className="pg-link" onClick={() => go(s.id)}><code>{s.step}</code></button>)}
            </p>
          )}
        </div>
      )}
      <Code ix={ix} def={def} mark={isStep ? node : null} scrollKey={sel} reduced={reduced} full={full} setFull={setFull} />
    </div>
  );
}

/**
 * A multi-path: one intent, several
 * implementations held to agree. Its members (the default first), which of
 * them share code (members that share a def outside the stage machinery are
 * one implementation, not two), the contract they are held to, the tests that
 * run them side by side, and how the production path reaches the default.
 */
function MultiPath({ mp, member, open, go, toggle, katex, refTop }) {
  const s = mp.stage;
  const familyOf = (m) => s.families.findIndex((f) => f.includes(m));
  const isOpen = open.has(mp.id);
  return (
    <div className="pg-sel pg-mp" ref={refTop}>
      <p className="pg-where ex-quiet">multi-path of <code>{s.id}</code> · used by <DefLink id={mp.host} go={go} /></p>
      <h3 className="pg-title">{s.title}</h3>
      {s.what && <Prose text={s.what} katex={katex} className="pg-mp-what" />}
      <p className="pg-mp-score">
        <b>{s.independent}</b> independent implementation{s.independent === 1 ? '' : 's'} of {s.members.length}
        {s.agreeing_families === null || s.agreeing_families === undefined
          ? ', agreement not measured yet'
          : <>, <b>{s.agreeing_families}</b> seen agreeing{s.agreeing_families >= 2 ? ' ✓' : ''}</>}
      </p>
      <div className="pg-acts">
        <button type="button" className="pg-btn" onClick={() => toggle(mp.id)} aria-pressed={isOpen}>{isOpen ? 'Close it into one box' : 'Open it: fork, members, join'}</button>
      </div>
      <table className="pg-mp-members">
        <thead>
          <tr><th>member</th><th>implementation</th><th>this run</th></tr>
        </thead>
        <tbody>
          {s.members.map((m) => (
            <tr key={m} className={m === member ? 'on' : ''}>
              <td><DefLink id={m} go={go} />{m === s.default && <span className="pg-n">default</span>}</td>
              <td className="pg-quiet">{String.fromCharCode(65 + familyOf(m))}{(s.agreeing || []).includes(m) ? ' · agreed' : ''}</td>
              <td className="pg-quiet">{mp.members.has(m) ? `ran ×${fmtCount(mp.members.get(m))}` : '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {s.ties.length > 0 && (
        <details className="pg-mp-ties">
          <summary>Why some members count as one implementation</summary>
          <ul>
            {s.ties.map((t) => <li key={`${t.a}>${t.b}`}><code>{t.a}</code> and <code>{t.b}</code> both reach {t.shared.map((x, i) => <Fragment key={x}>{i ? ', ' : ''}<code>{x}</code></Fragment>)}</li>)}
          </ul>
        </details>
      )}
      <div className="pg-rel pg-chips">
        <p><span className="pg-k">Contract</span> {contractWords(s) || s.contract}{s.tolerance ? <span className="pg-n">({s.tolerance.name})</span> : null}</p>
        {s.checked.length > 0 && (
          <p>
            <span className="pg-k">Checked by</span>
            {s.checked.map((c) => <span key={c.name} className="pg-mp-cite"><span className="pg-check">✓</span> {c.kind} “{c.name}”</span>)}
          </p>
        )}
        <p>
          <span className="pg-k">Production path</span>
          {s.direct ? <>calls <code>{s.default}</code> directly</> : <>reaches <code>{s.default}</code> through {s.through.map((t, i) => <Fragment key={t}>{i ? ' → ' : ''}<code>{t}</code></Fragment>)} and the registry <code>{s.registry}</code></>}
        </p>
        {s.intent && <p><span className="pg-k">Intent</span><DefLink id={s.intent} go={go} /></p>}
      </div>
      <p className="ex-quiet">{robustWords(s)}. Measured by <code>viv arch report --verify</code>; the lens is <code>{mp.arch}</code>.</p>
    </div>
  );
}

/** The multi-paths of the program and how many stages have two agreeing, independent implementations. */
function MultiPaths({ model, go }) {
  const byStage = new Map();
  for (const mp of model.mps.values()) if (!byStage.has(mp.stage.id)) byStage.set(mp.stage.id, mp);
  const stages = [...byStage.values()];
  if (!stages.length) return null;
  const robust = stages.filter((mp) => (mp.stage.agreeing_families || 0) >= 2);
  return (
    <details className="pg-notrun pg-mps" open>
      <summary>
        Multi-paths: {robust.length} of {stages.length} stages this run used have two or more independent implementations seen agreeing
      </summary>
      <p className="ex-quiet">A stage with several implementations of one intent is drawn as one box, <b>multi-path ×n</b>; opened, it is a fork, a path per member and a join. The count is of implementations that share no code.</p>
      <ul className="pg-mp-list">
        {stages.map((mp) => (
          <li key={mp.id}>
            <button type="button" className="pg-link" onClick={() => go(mp.id)}><code>{mp.stage.id}</code></button>{' '}
            <span className="pg-quiet">{mp.stage.title}</span>{' '}
            <span className="pg-n">{mp.stage.independent}/{mp.stage.members.length} independent, {mp.stage.agreeing_families ?? '?'} agreeing{(mp.stage.agreeing_families || 0) >= 2 ? ' ✓' : ''}</span>
          </li>
        ))}
      </ul>
    </details>
  );
}

/** With nothing picked: how to use the picture, and what the run did not call. */
function Intro({ ix, model, shown, go, refTop }) {
  const byModule = useMemo(() => {
    const m = new Map();
    for (const d of model.notRun) {
      const n = model.defs.get(d);
      const w = ix.toModule(n.line);
      const key = w ? w.module : '(the program)';
      if (!m.has(key)) m.set(key, []);
      m.get(key).push(d);
    }
    return [...m.entries()].sort((a, b) => b[1].length - a[1].length);
  }, [ix, model]);
  const nDefs = model.defs.size;
  return (
    <div className="pg-intro" ref={refTop}>
      <p>
        Pick a box to read its code here, with what its intent says and who calls it. The picture shows the defs {model.bundle.calls ? 'this run called' : 'the text calls'}, {shown.size} of{' '}
        {model.ran.length} so far; a box marked <b>+n</b> hides the n defs it calls and opens with a click, and <b>−</b> closes it again.
      </p>
      <p className="ex-quiet">
        Drag to move the picture, and scroll or pinch to zoom; with the picture focused, the arrow keys move from box to box, Enter opens or closes the
        box's calls, + and − zoom and 0 shows the whole picture.
      </p>
      {model.mps && model.mps.size > 0 && <MultiPaths model={model} go={go} />}
      {model.fromRun && byModule.length > 0 && (
        <details className="pg-notrun">
          <summary>
            The program's text defines {fmtCount(nDefs)} defs; the run did not call {fmtCount(model.notRun.length)} of them
          </summary>
          <p className="ex-quiet">Other methods, other bases, reference data and the tests' helpers. They are not in the picture; a click shows one's code here.</p>
          {byModule.map(([mod, ds]) => (
            <details key={mod} className="pg-notrun-mod">
              <summary><code>{mod}</code> <span className="ex-quiet">{ds.length}</span></summary>
              <p className="pg-notrun-list">
                {ds.map((d) => <button key={d} type="button" className="pg-link" onClick={() => go(d)}><code>{d}</code></button>)}
              </p>
            </details>
          ))}
        </details>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// The walkthrough (lib/tour.js): one stop of the recorded run
// ---------------------------------------------------------------------------

/** A recorded value in words: a number, a short array's entries, or a larger one's shape. */
function valueText(v) {
  if (v && typeof v === 'object') {
    if (Array.isArray(v.values) && v.values.length <= 4 && (v.shape || []).length <= 1) return `[${v.values.map(fmtValue).join(', ')}]`;
    const sh = shapeOf(v);
    if (!sh) return '…';
    return (v.shape || []).length === 1 ? `${sh} values` : `${sh} array`;
  }
  return fmtValue(v);
}

/** The values a stop names: the debug lens's watched names first, then numbers, then arrays; at most MAX_VALUES. */
function pickValues(ix, def, at, values) {
  const watched = new Set();
  for (const d of ix.bundle.debug || []) {
    if (d.def === def && d.kind === 'watch' && (d.at === at || d.at === 'iteration')) for (const n of d.names || d.text.replace(/^watch\s+(per\s+iteration|at\s+\S+)\s+/, '').split(/\s*,\s*/)) watched.add(n);
  }
  const rank = ([k, v]) => (watched.has(k) ? 0 : typeof v === 'object' ? 2 : 1);
  const sorted = [...values].sort((a, b) => rank(a) - rank(b));
  return { shown: sorted.slice(0, MAX_VALUES), more: Math.max(0, sorted.length - MAX_VALUES) };
}

function Values({ ix, def, at, values, katex }) {
  if (!values || !values.length) return null;
  const { shown, more } = pickValues(ix, def, at, values);
  return (
    <dl className="pg-vals">
      {shown.map(([k, v]) => {
        const g = shortGloss(ix, def, k);
        return (
          <div key={k}>
            <dt><code>{k}</code></dt>
            <dd>
              <span className="pg-val">{valueText(v)}</span>
              {g && <span className="ex-quiet pg-valg"> <Prose as="span" text={g} katex={katex} /></span>}
            </dd>
          </div>
        );
      })}
      {more > 0 && <div className="pg-valmore ex-quiet">and {more} more</div>}
    </dl>
  );
}

/** A small line chart of one watched value across a loop's passes, the current pass marked. */
function Spark({ name, pts, cur, gloss }) {
  const W = 120;
  const H = 34;
  const pad = 3;
  const ys = pts.map((p) => p.y);
  const pos = ys.every((y) => y > 0);
  const neg = ys.every((y) => y < 0);
  const mags = ys.map(Math.abs);
  const spread = Math.max(...mags) / Math.max(Math.min(...mags), 1e-300);
  const log = (pos || neg) && spread > 1e3;
  const f = (y) => (log ? Math.log10(Math.abs(y)) : y);
  const fy = ys.map(f);
  const lo = Math.min(...fy);
  const hi = Math.max(...fy);
  const x0 = pts[0].x;
  const x1 = pts[pts.length - 1].x;
  const X = (x) => pad + ((x - x0) / Math.max(1, x1 - x0)) * (W - 2 * pad);
  const Y = (y) => H - pad - ((f(y) - lo) / Math.max(hi - lo, 1e-300)) * (H - 2 * pad);
  const d = pts.map((p, k) => `${k ? 'L' : 'M'}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join(' ');
  const at = pts.find((p) => p.x === cur);
  return (
    <figure className="pg-spark">
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`${name} over the passes${log ? ', log scale' : ''}`}>
        <path d={d} className="pg-spark-line" />
        {at && <circle cx={X(at.x)} cy={Y(at.y)} r="3" className="pg-spark-dot" />}
      </svg>
      <figcaption title={gloss || undefined}>
        <code>{name}</code>{log ? <span className="ex-quiet"> log</span> : null}
        {at && <span className="pg-val"> {fmtValue(at.y)}</span>}
      </figcaption>
    </figure>
  );
}

function Sparks({ ix, loop, cur }) {
  const series = useMemo(() => loopSeries(ix, loop), [ix, loop]);
  if (!series.length) return null;
  return (
    <div className="pg-sparks">
      {series.slice(0, 4).map((s) => <Spark key={s.name} name={s.name} pts={s.pts} cur={cur} gloss={shortGloss(ix, loop.def, s.name)} />)}
    </div>
  );
}

/** The debug lens's holds checked under a node: each sentence, or, when there are many, a count that opens to them. */
function Holds({ node, many = 3 }) {
  const h = holdsIn(node);
  if (!h.ok && !h.fail) return null;
  const items = [...h.sentences].sort((a, b) => b[1].fail - a[1].fail);
  const list = (xs) => (
    <ul className="pg-holds">
      {xs.map(([sent, c]) => (
        <li key={sent} className={c.fail ? 'bad' : 'ok'}>
          <span aria-hidden="true">{c.fail ? '✗' : '✓'}</span> {sent}
          {c.ok + c.fail > 1 && <span className="ex-quiet"> ({c.fail ? `failed ${c.fail} of ${c.ok + c.fail}` : `${c.ok} times`})</span>}
        </li>
      ))}
    </ul>
  );
  if (items.length <= many) return list(items);
  return (
    <details className="pg-holdsum">
      <summary>
        <span aria-hidden="true" className={h.fail ? 'bad' : 'ok'}>{h.fail ? '✗' : '✓'}</span>{' '}
        {h.fail ? `${h.fail} of ${h.ok + h.fail} checks failed` : `all ${h.ok} checks held`} ({items.length} kinds)
      </summary>
      {list(items)}
    </details>
  );
}

const plural = (n, one, many = `${one}s`) => `${n.toLocaleString('en-US')} ${n === 1 ? one : many}`;

/** The panel during the walkthrough: where the run is, what is happening there, and the code with the lines marked. */
function TourStop({ ix, katex, model, tour, reduced, full, setFull }) {
  const { stop, stops, run, peekAt, openPass } = tour;
  const ref = useRef(null);
  const bundle = ix.bundle;
  const n = stop.node;
  const intent = bundle.intent || {};
  const recorded = useMemo(() => recordedDefs(run.tree), [run]);
  const defNode = model.defs.get(stop.def);
  // on a narrow screen the page scrolls: keep the narration under the picture
  useLayoutEffect(() => {
    const el = ref.current;
    const panel = el && el.closest('.pg-panel');
    if (!el || !panel || getComputedStyle(panel).overflowY !== 'visible') return;
    // where the picture's bottom is once it sticks under the site's header
    const graph = panel.parentElement.querySelector('.pg-left');
    const head = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--head-h')) || 56;
    const under = head + (graph ? graph.getBoundingClientRect().height : 0);
    const top = el.getBoundingClientRect().top;
    if (Math.abs(top - under - 4) > 12) window.scrollTo({ top: window.scrollY + top - under - 4, behavior: reduced ? 'auto' : 'smooth' });
  }, [stop.k, reduced]);

  const link = (id, extra = null) => (
    <button key={id} type="button" className="pg-link" onClick={() => peekAt(id)} title={`Open ${id} in the panel (the walk pauses)`}>
      <code>{id}</code>{extra}
    </button>
  );
  const where = stop.mark ? ix.toModule(stop.mark.line) : null;
  let head = stop.title;
  let body = null;
  if (stop.kind === 'intro') {
    const loops = stops.filter((s) => s.kind === 'loop').length;
    head = 'What the program is for';
    body = (
      <>
        <Prose className="pg-what" text={programPurpose(ix)} katex={katex} />
        <p className="ex-quiet">
          This walk goes through the run this tab recorded, in {plural(stops.length, 'stop')}. Calls and steps come in the order they ran
          {loops ? '; a loop shows its first pass step by step and each later pass as one stop' : ''}. The panel shows the code with the current lines marked.
        </p>
      </>
    );
  } else if (stop.kind === 'setup') {
    body = <p>Before the program's own statements run, the modules it uses set up their constants and tables: {plural(n.children.length, 'line')} in {plural(n.modules.length, 'module')}. The walk does not go through them.</p>;
  } else if (stop.kind === 'lines') {
    const P = ix.programLines;
    const text = stop.nodes.map((l) => (P[l.line - 1] || '').trim()).filter(Boolean);
    const calls = [];
    // the recorded calls this statement makes come next in the walk
    for (let j = stop.k + 1; j < stops.length && stops[j].kind === 'call' && stop.def === '<main>'; j++) calls.push(stops[j].def);
    body = (
      <>
        <p>{stop.def === '<main>' ? (text.length > 1 ? 'The program runs these statements:' : 'The program runs this statement:') : <>In <code>{stop.def}</code>, outside its steps:</>}</p>
        {stop.def === '<main>' && <pre className="pg-stmt"><code dangerouslySetInnerHTML={{ __html: text.slice(0, 4).map((t) => highlight(t, 'viv')).join('\n') + (text.length > 4 ? '\n…' : '') }} /></pre>}
        <Values ix={ix} def={stop.def} at="" values={stop.values} katex={katex} />
      </>
    );
  } else if (stop.kind === 'call') {
    const it = intent[stop.def] || {};
    const what = it.what || inlineIntent(ix.programLines, defNode ? defNode.line : 0).what;
    const caller = (() => {
      for (let m = n.parent; m; m = m.parent) if (m.kind === 'call') return m.def;
      return null;
    })();
    const steps = (model.inner.get(stop.def) || []).filter((x) => x.kind === 'step');
    body = (
      <>
        <p className="ex-quiet">{caller && caller !== '<main>' ? <>Called from {link(caller)}.</> : 'Called by the program.'}</p>
        {what && <Prose className="pg-what" text={what} katex={katex} />}
        {steps.length > 0 && <p className="pg-chips"><span className="pg-k">Its steps</span>{steps.map((x) => <code key={x.id} className="pg-chip">{x.step}</code>)}</p>}
      </>
    );
  } else if (stop.kind === 'step') {
    const text = (bundle.steps || {})[`${n.def}/${n.id}`] || model.byId.get(`${n.def}/${n.id}`)?.text || '';
    const hot = hotBelow(model, `${n.def}/${n.id}`, recorded);
    body = (
      <>
        {text && <Prose className="pg-what" text={text} katex={katex} />}
        <Values ix={ix} def={n.def} at={n.id} values={stop.values} katex={katex} />
        <Holds node={n} />
        {hot.length > 0 && (
          <p className="pg-hot">
            <span className="pg-k">Below it, not walked</span>
            {hot.map((h) => link(h.id, <span className="pg-n">×{h.count.toLocaleString('en-US')}</span>))}
            <span className="ex-quiet">calls in the whole run; open one to read its code.</span>
          </p>
        )}
      </>
    );
  } else if (stop.kind === 'loop') {
    const g = n.graphId && ix.nodes.get(n.graphId);
    body = (
      <>
        <p>
          {g && g.header ? <>The loop <code>{g.header}</code> runs</> : 'A loop runs'} {plural(n.children.length, 'pass', 'passes')}.
          {stop.again ? ' The walk went through its passes the first time the loop ran, so this time it is one stop.' : n.children.length > 1 ? ' The walk shows the first pass step by step, then each later pass as one stop with the values it ended with.' : ''}
        </p>
        <Sparks ix={ix} loop={n} cur={null} />
      </>
    );
  } else if (stop.kind === 'pass') {
    const loop = n.parent;
    const last = n.n === loop.children.length;
    body = (
      <>
        <p>
          {last ? 'The last pass. ' : ''}
          {stop.walked && n.n === 1 && loop.children.length > 1 ? 'The walk goes through this pass step by step; each later pass is one stop. ' : ''}
          {stop.values.length ? 'At its end:' : 'It records no values.'}
        </p>
        <Values ix={ix} def={n.def} at="iteration" values={stop.values} katex={katex} />
        <Sparks ix={ix} loop={loop} cur={n.n} />
        <Holds node={n} />
        {!stop.walked && (
          <button type="button" className="pg-btn pg-quiet pg-openpass" onClick={() => openPass(n.key)}>Walk through this pass step by step</button>
        )}
      </>
    );
  } else if (stop.kind === 'end') {
    // the text it printed; its figure records are counted, not listed
    const own = parseOwnOutput(run.rec.printed || '');
    const printed = own.blocks.filter((b) => b.type === 'text').map((b) => b.s).join('\n').split('\n').filter((l) => l.trim() && l.trim() !== '(no output)');
    const figures = own.charts.length;
    const loops = stops.filter((s) => s.kind === 'loop');
    const h = holdsIn(run.tree);
    head = 'The end of the run';
    body = (
      <>
        {printed.length > 0 && (
          <>
            <p>The program printed:</p>
            <pre className="pg-printed">{printed.slice(0, MAX_PRINTED).join('\n')}{printed.length > MAX_PRINTED ? `\n… ${printed.length - MAX_PRINTED} more lines` : ''}</pre>
          </>
        )}
        {figures > 0 && <p className="ex-quiet">It also drew {plural(figures, 'figure')}; the full reading below shows them.</p>}
        {loops.length > 0 && <Sparks ix={ix} loop={loops[0].node} cur={loops[0].node.children.length} />}
        {loops.length > 0 && (
          <ul className="pg-loops">
            {loops.map((s) => {
              const g = s.node.graphId && ix.nodes.get(s.node.graphId);
              return <li key={s.k}><code>{s.def}</code>: {g && g.header ? <code>{g.header}</code> : 'a loop'} ran {plural(s.node.children.length, 'pass', 'passes')}</li>;
            })}
          </ul>
        )}
        {h.ok + h.fail > 0 ? (
          <>
            <p>{h.fail ? `${plural(h.fail, 'check')} of ${plural(h.ok + h.fail, 'check')} failed:` : `All ${plural(h.ok, 'check')} of the debug lens held:`}</p>
            <Holds node={run.tree} many={6} />
          </>
        ) : <p className="ex-quiet">The debug lens checks nothing in this run.</p>}
      </>
    );
  }
  return (
    <div className="pg-sel pg-tourstop" ref={ref} data-kind={stop.kind} data-k={stop.k}>
      <p className="pg-where ex-quiet">
        <span className="pg-stopn">Stop {stop.k + 1} of {stops.length}</span>
        {where ? <> · <code>{where.module}.viv</code> line {where.line}</> : null}
      </p>
      <h3 className="pg-title">{stop.kind === 'step' || stop.kind === 'call' ? <code>{head}</code> : head}</h3>
      <div className="pg-narr">{body}</div>
      {defNode && stop.kind !== 'intro' && stop.kind !== 'end' && stop.kind !== 'setup' && (
        <div className="pg-tourcode">
          <Code ix={ix} def={defNode} mark={stop.mark} scrollKey={`${stop.k}`} reduced={reduced} full={full} setFull={setFull} ownScroll />
        </div>
      )}
    </div>
  );
}
