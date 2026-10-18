// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// A showcase page's run (/show/<slug>): the Run button, the run's progress,
// the figure or animation the program drew, and the tests it passed against
// its reference. Before the reader runs it, the figure and the tests are the
// build-time run's (scripts/show-build.mjs); after, they are the reader's own.
import { useEffect, useRef, useState } from 'react';
import { parseOwnOutput, Chart } from './VivCell.jsx';
import SceneViz from './SceneViz.jsx';
import { runShowcase, testsOf } from '../lib/showRun.js';
import { sceneOf } from '../data/scenes.mjs';


/** The charts a run drew; `only` picks one of them by its order. */
export function Figures({ text, only = null }) {
  const all = parseOwnOutput(text).charts;
  const charts = only == null ? all : all.slice(only, only + 1);
  return (
    <>
      {charts.map((c, i) => (
        <Chart key={i} chart={c} />
      ))}
    </>
  );
}

export default function ShowRun({ slug, figure, tests, seconds, reference }) {
  const scene = sceneOf(slug);
  const readout = useRef(null);
  const [state, setState] = useState({ what: 'idle' });
  const [elapsed, setElapsed] = useState(0);
  const t0 = useRef(0);

  useEffect(() => {
    if (state.what !== 'running') return undefined;
    const tick = setInterval(() => setElapsed((performance.now() - t0.current) / 1000), 100);
    return () => clearInterval(tick);
  }, [state.what]);

  const run = async () => {
    t0.current = performance.now();
    setElapsed(0);
    setState({ what: 'running' });
    try {
      const r = await runShowcase(slug);
      setState({ what: 'done', out: r.out, seconds: (performance.now() - t0.current) / 1000 });
    } catch (e) {
      setState({ what: 'error', error: String(e.message || e) });
    }
  };

  const mine = state.what === 'done';
  const shownFigure = mine ? state.out : figure;
  const shownTests = mine ? testsOf(state.out) : tests.map((name) => ({ name, pass: true, why: '' }));
  const passed = shownTests.filter((t) => t.pass).length;
  const progress = Math.min(1, elapsed / Math.max(seconds, 0.1));

  return (
    <>
      {scene && (
        <div className="scn-hero">
          <SceneViz kind={scene.kind} slug={slug} fig={scene.fig} text={mine ? state.out : null} readout={readout} aspect="16 / 9" label={scene.title} />
          <p className="scn-readout" ref={readout} aria-live="off" />
        </div>
      )}
      <h2 id="run">Run</h2>
      <div className="show-run">
        <button type="button" className="show-runbtn" onClick={run} disabled={state.what === 'running'}>
          {state.what === 'running' ? 'Running' : mine ? 'Run again' : 'Run'}
        </button>
        <div className="show-progress" role="status" aria-live="polite">
          {state.what === 'running' && (
            <>
              <div className="show-bar" aria-hidden="true">
                <span style={{ width: `${Math.round(progress * 100)}%` }} />
              </div>
              <p>Running, {elapsed.toFixed(1)} s.</p>
            </>
          )}
          {mine && (
            <p>
              Your run: {passed} of {shownTests.length} tests pass.
            </p>
          )}
          {state.what === 'error' && <p className="show-err">The run stopped: {state.error}</p>}
        </div>
      </div>

      <h2 id="shows">What the run shows</h2>
      <div className="run-fig show-fig">
        <Figures text={shownFigure} />
      </div>

      <h2 id="checked">How it is checked</h2>
      <p>
        The reference is {reference}. {passed} of {shownTests.length} tests pass{mine ? ' in your run' : ''}.
      </p>
      <ul className="run-tests">
        {shownTests.map((t, i) => (
          <li key={i} className={t.pass ? '' : 'is-fail'}>
            <span aria-hidden="true">{t.pass ? '✓' : '✗'}</span>
            <span>
              {t.name}
              {t.why && <small> {t.why}</small>}
            </span>
          </li>
        ))}
      </ul>
    </>
  );
}
