// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The DFT page's instrument: pick a material, run its self-consistent
// plane-wave calculation in this tab (viv.wasm, in a worker), and read the band
// structure it printed.
//
// Nothing here knows any physics. The plot is drawn from the program's own
// `plot:` lines, the numbers from its `show`s, the checklist from its PASS /
// FAIL lines -- parsed by the same src/lib/dftParse.js that read the build-time
// result the page opened with.
import { useEffect, useMemo, useRef, useState } from 'react';
import { parseDft, isReferenceTest } from '../lib/dftParse.js';
import { VIV_WASM_URL, withVivModule, postVivMessage } from '../lib/vivWasm.js';

// viridis' hues, indigo to green, lifted so a 2px line reads on the dark ground.
// Its yellow end is left out on purpose: on this site yellow marks a number that
// matched, and a band is not one.
const RAMP = ['#5a7bd0', '#3f8dc2', '#2a9fae', '#25ae92', '#44bf70', '#7fd34e'];
const bandColor = (b, n) => RAMP[Math.min(RAMP.length - 1, Math.floor((b / Math.max(1, n - 1)) * (RAMP.length - 1) + 0.5))];

const fmt = (v, d = 2) => (v === undefined || v === null || Number.isNaN(v) ? '–' : v.toFixed(d));

function niceStep(range) {
  if (range > 26) return 5;
  if (range > 13) return 4;
  return 2;
}

export function BandPlot({ result, metal, drawKey }) {
  // Drawn at the container's own width, one SVG unit per CSS pixel, so an 11px
  // tick label is 11px on a phone too instead of shrinking with the viewBox.
  const holder = useRef(null);
  const [W, setW] = useState(760);
  useEffect(() => {
    const el = holder.current;
    if (!el || typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(() => setW(Math.max(300, Math.round(el.clientWidth))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const narrow = W < 520;
  const H = Math.round(narrow ? W * 0.92 : W * 0.62);
  const L = narrow ? 34 : 60, R = narrow ? 68 : 92, T = 16, B = 44;
  const { x, bands } = result;
  const geom = useMemo(() => {
    const lo = Math.min(...bands[0]);
    const ymin = Math.floor(lo - 0.8);
    const ymax = 7;
    const x0 = x[0], x1 = x[x.length - 1];
    const sx = (v) => L + ((v - x0) / (x1 - x0)) * (W - L - R);
    const sy = (v) => T + ((ymax - v) / (ymax - ymin)) * (H - T - B);
    const step = niceStep(ymax - ymin);
    const ticks = [];
    for (let t = Math.ceil(ymin / step) * step; t <= ymax; t += step) ticks.push(t);
    // an insulator's gap: from the top of the bands that never rise above zero
    // to the bottom of the ones that never fall below it
    let gap = null;
    if (!metal) {
      const empty = bands.filter((e) => Math.min(...e) > 1e-6);
      if (empty.length) gap = Math.min(...empty.map((e) => Math.min(...e)));
    }
    return { ymin, ymax, sx, sy, ticks, gap };
  }, [x, bands, metal, W, H, L, R]);
  const { ymin, ymax, sx, sy, ticks, gap } = geom;
  const nseg = (x.length - 1) / 2;
  const corners = [[0, 'L'], [nseg, 'Γ'], [x.length - 1, 'X']];
  const path = (e) => e.map((v, i) => `${i ? 'L' : 'M'}${sx(x[i]).toFixed(1)} ${sy(Math.max(ymin - 5, Math.min(ymax + 5, v))).toFixed(1)}`).join(' ');

  return (
    <div ref={holder}>
    <svg className="bs-plot" viewBox={`0 0 ${W} ${H}`} role="img"
      aria-label={`Band structure along L, Gamma, X: ${bands.length} bands, energies in electron-volts from ${metal ? 'the Fermi level' : 'the valence-band maximum'}`}>
      <defs>
        <clipPath id="bs-clip"><rect x={L} y={T} width={W - L - R} height={H - T - B} /></clipPath>
      </defs>
      {gap !== null && (
        <g>
          <rect className="bs-gap" x={L} y={sy(gap)} width={W - L - R} height={sy(0) - sy(gap)} />
          <path className="bs-bracket" d={`M${W - R + 5} ${sy(gap)} h5 V${sy(0)} h-5`} />
          <text className="bs-gaplabel" x={W - R + 14} y={sy(gap) - 18}>gap</text>
          <text className="bs-gaplabel" x={W - R + 14} y={sy(gap) - 5}>{fmt(gap)} eV</text>
        </g>
      )}
      {ticks.map((t) => (
        <g key={t}>
          <line className="bs-grid" x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} />
          <text className="bs-tick" x={L - 9} y={sy(t) + 4} textAnchor="end">{t}</text>
        </g>
      ))}
      {corners.map(([i, label]) => (
        <g key={label}>
          <line className="bs-warp" x1={sx(x[i])} x2={sx(x[i])} y1={T} y2={H - B} />
          <text className="bs-corner" x={sx(x[i])} y={H - B + 27} textAnchor="middle">{label}</text>
        </g>
      ))}
      <line className="bs-zero" x1={L} x2={W - R} y1={sy(0)} y2={sy(0)} />
      <text className="bs-zerolabel" x={W - R + 14} y={sy(0) + 14}>{metal ? 'Fermi' : narrow ? 'band' : 'valence'}</text>
      <text className="bs-zerolabel" x={W - R + 14} y={sy(0) + 27}>{metal ? 'level' : narrow ? 'top' : 'band top'}</text>
      <g clipPath="url(#bs-clip)" key={drawKey}>
        {bands.map((e, b) => (
          <path key={b} className="bs-band" d={path(e)} pathLength="1" stroke={bandColor(b, bands.length)}
            style={{ animationDelay: `${b * 45}ms` }} />
        ))}
      </g>
      {!narrow && <text className="bs-ylabel" transform={`translate(15 ${(T + H - B) / 2}) rotate(-90)`} textAnchor="middle">energy (eV)</text>}
    </svg>
    </div>
  );
}

export function Convergence({ drho }) {
  if (!drho || drho.length < 2) return null;
  const W = 300, H = 96, L = 38, R = 8, T = 8, B = 20;
  const logs = drho.map((v) => Math.log10(Math.max(v, 1e-12)));
  const lo = Math.floor(Math.min(...logs)), hi = Math.ceil(Math.max(...logs));
  const sx = (i) => L + (i / (drho.length - 1)) * (W - L - R);
  const sy = (v) => T + ((hi - v) / (hi - lo)) * (H - T - B);
  return (
    <svg className="bs-conv" viewBox={`0 0 ${W} ${H}`} role="img"
      aria-label={`Density residual per SCF iteration, falling from ${drho[0].toExponential(1)} to ${drho[drho.length - 1].toExponential(1)} in ${drho.length} iterations`}>
      {[hi, Math.round((hi + lo) / 2), lo].map((t) => (
        <g key={t}>
          <line className="bs-grid" x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} />
          <text className="bs-tick" x={L - 6} y={sy(t) + 3.5} textAnchor="end">1e{t}</text>
        </g>
      ))}
      <path className="bs-convline" d={logs.map((v, i) => `${i ? 'L' : 'M'}${sx(i).toFixed(1)} ${sy(v).toFixed(1)}`).join(' ')} />
      {logs.map((v, i) => <circle key={i} className="bs-convdot" cx={sx(i)} cy={sy(v)} r="2.4" />)}
      <text className="bs-tick" x={L} y={H - 4}>1</text>
      <text className="bs-tick" x={W - R} y={H - 4} textAnchor="end">{drho.length}</text>
    </svg>
  );
}

// Test names are ASCII in the Viv source. Typeset the symmetry labels and signs.
const pretty = (s) => s.replace(/\bGamma(\d*)/g, 'Γ$1').replace(/(\d)'/g, '$1′').replace(/ -- /g, ' — ').replace(/ -(\d)/g, ' −$1').replace(/ - /g, ' − ');

function TestList({ title, note, names, failed }) {
  if (!names.length && !failed.length) return null;
  return (
    <div className="bs-tests">
      <h3>{title}</h3>
      <p className="bs-testnote">{note}</p>
      <ul>
        {names.map((n) => <li key={n} className="ok"><span aria-hidden="true">✓</span>{pretty(n)}</li>)}
        {failed.map((n) => <li key={n} className="bad"><span aria-hidden="true">✗</span>{pretty(n)}</li>)}
      </ul>
    </div>
  );
}

const ENGINES = [
  { id: 'compiled', label: 'Compiled', sub: 'Viv → C → wasm' },
  { id: 'interpreted', label: 'Interpreted', sub: 'viv.wasm interprets the source' },
];

export default function BandStructure({ materials }) {
  const [sel, setSel] = useState(materials[0].id);
  const [engine, setEngine] = useState('compiled');
  const [live, setLive] = useState({}); // `${id}:${engine}` -> { result, seconds } once computed in this tab
  const [running, setRunning] = useState(null); // `${id}:${engine}` being computed
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState(null);
  const worker = useRef(null);

  const mat = materials.find((m) => m.id === sel);
  const key = `${sel}:${engine}`;
  const mine = live[key];
  const result = mine ? mine.result : mat.result;
  const metal = mat.smear > 0;
  // copper is minutes of work: offered compiled, never interpreted
  const heavy = !!mat.recorded;
  const canRun = !(heavy && engine === 'interpreted');

  useEffect(() => {
    if (!running) return undefined;
    const t0 = performance.now();
    const id = setInterval(() => setElapsed((performance.now() - t0) / 1000), 100);
    return () => clearInterval(id);
  }, [running]);
  useEffect(() => () => worker.current && worker.current.terminate(), []);

  const stop = () => {
    if (worker.current) worker.current.terminate();
    worker.current = null;
    setRunning(null);
  };
  const compute = () => {
    stop();
    setError(null);
    setElapsed(0);
    const k = key;
    const w = new Worker(`/dft-worker.js?wasm=${encodeURIComponent(VIV_WASM_URL)}`, { type: 'module' });
    worker.current = w;
    setRunning(k);
    w.onmessage = (e) => {
      if (e.data.started) return;
      w.terminate();
      worker.current = null;
      setRunning(null);
      if (!e.data.ok) { setError(e.data.error); return; }
      setLive((prev) => ({ ...prev, [k]: { result: parseDft(e.data.text), seconds: e.data.seconds } }));
    };
    w.onerror = (e) => { stop(); setError(e.message || 'the engine could not start in your browser; reload the page and try again'); };
    // interpreted: the worker gets the page's compiled viv.wasm with the job
    if (engine === 'interpreted') withVivModule({ mat: sel, engine }).then((m) => postVivMessage(w, m), () => w.postMessage({ mat: sel, engine }));
    else w.postMessage({ mat: sel, engine });
  };

  const refTests = result ? result.pass.filter(isReferenceTest) : [];
  const physTests = result ? result.pass.filter((n) => !isReferenceTest(n)) : [];
  const refFail = result ? result.fail.filter(isReferenceTest) : [];
  const physFail = result ? result.fail.filter((n) => !isReferenceTest(n)) : [];
  const total = result ? result.pass.length + result.fail.length : 0;
  const isRunning = running === key;
  const busy = running !== null;
  const secs = (x) => (x >= 100 ? `${Math.round(x / 60)} min` : `${x >= 10 ? Math.round(x) : x} s`);
  const expected = engine === 'compiled' ? (heavy ? mat.wasmSeconds : mat.result.seconds) : mat.interpreted && mat.interpreted.seconds;

  let provenance;
  if (isRunning) provenance = `${engine === 'compiled' ? 'Running the compiled program' : 'Interpreting the source'}… ${elapsed.toFixed(1)} s`;
  else if (mine) provenance = `Computed in your browser in ${mine.seconds.toFixed(1)} s — ${engine === 'compiled' ? 'the compiled program' : 'the interpreter'}.`;
  else if (heavy) provenance = `A recorded run: compiled to C, run natively, ${mat.result.seconds} s on ${mat.result.machine}.`;
  else provenance = `Computed when this page was built, by the compiled program, in ${mat.result.seconds} s.`;

  return (
    <div className="bs">
      <div className="bs-pick" role="radiogroup" aria-label="Material">
        {materials.map((m) => (
          <button key={m.id} type="button" role="radio" aria-checked={m.id === sel} disabled={busy}
            className={`bs-mat${m.id === sel ? ' on' : ''}`} onClick={() => setSel(m.id)}>
            <span className="bs-formula">{m.formula}</span>
            <span className="bs-name">{m.name}</span>
            <span className="bs-size">{m.result.shown.NPWMAX} plane waves</span>
            <span className="bs-time">{m.result.shown.NKPT} k-points · {m.recorded ? secs(m.wasmSeconds || m.result.seconds) : secs(m.result.seconds)}</span>
          </button>
        ))}
      </div>

      <div className="bs-main">
        <div className={`bs-frame${isRunning ? ' busy' : ''}`}>
          {result && result.bands.length ? <BandPlot result={result} metal={metal} drawKey={`${key}-${mine ? mine.seconds : 'built'}`} />
            : <div className="bs-empty">No band structure yet for {mat.name}.</div>}
          {isRunning && <div className="bs-progress" aria-hidden="true" />}
        </div>

        <div className="bs-bar">
          <div className="bs-engines" role="radiogroup" aria-label="Engine">
            {ENGINES.map((e) => (
              <button key={e.id} type="button" role="radio" aria-checked={e.id === engine} disabled={busy}
                className={`bs-engine${e.id === engine ? ' on' : ''}`} onClick={() => setEngine(e.id)}>
                <b>{e.label}</b><span>{e.sub}</span>
              </button>
            ))}
          </div>
          {isRunning ? (
            <button type="button" className="bs-btn ghost" onClick={stop}>Stop</button>
          ) : (
            <button type="button" className="bs-btn" onClick={compute} disabled={!canRun || busy}>
              {mine ? 'Compute it again' : 'Compute it'}{canRun && expected ? ` · about ${secs(expected)}` : ''}
            </button>
          )}
        </div>
        <p className="bs-prov" aria-live="polite">{provenance}</p>
        {heavy && (
          <p className="bs-cant">
            {engine === 'compiled'
              ? `Copper is ${mat.result.shown.NPWMAX.toLocaleString()} plane waves at ${mat.result.shown.NKPT} k-points. The compiled program runs in your browser in about ${secs(mat.wasmSeconds)}, using about ${mat.wasmMemoryMB} MB, and the page stays usable while it runs.`
              : 'Copper cannot be run with the interpreter here: the compiled program takes minutes, and the interpreter would take several times longer.'}
          </p>
        )}
        {error && <p className="bs-error" role="alert">The run did not finish: {error}</p>}

        {result && (
          <div className="bs-facts">
            <dl>
              <div><dt>cutoff</dt><dd>{mat.ecut} Ha</dd></div>
              <div><dt>plane waves</dt><dd>{result.shown.NPWMAX}</dd></div>
              <div><dt>k-points</dt><dd>{result.shown.NKPT}</dd></div>
              <div><dt>FFT grid</dt><dd>{result.shown.NG}<sup>3</sup></dd></div>
              <div><dt>SCF iterations</dt><dd>{result.shown.ITER}</dd></div>
              <div><dt>tests passing</dt><dd className={result.fail.length ? 'bad' : 'hot'}>{result.pass.length} of {total}</dd></div>
            </dl>
            <figure className="bs-convfig">
              <Convergence drho={result.drho} />
              <figcaption>density residual, per SCF iteration</figcaption>
            </figure>
          </div>
        )}
      </div>

      {result && (
        <div className="bs-checks">
          <TestList title="Against an independent implementation"
            note={heavy
              ? 'A numpy code of the same iterative method, run at these settings; its numbers are pinned in the check file.'
              : 'A numpy code that diagonalizes H densely (a different algorithm as well as a different language), run at these settings; its numbers are pinned in the check file.'}
            names={refTests} failed={refFail} />
          <TestList title="Against published LDA results"
            note="These tolerances are wide because published values differ in pseudopotential and k-point mesh. A failure here would mean the method itself is wrong."
            names={physTests} failed={physFail} />
        </div>
      )}
      <p className="bs-pseudo">
        Pseudopotential{mat.pseudos.length > 1 ? 's' : ''}: {mat.pseudos.join(', ')} (Goedecker–Teter–Hutter, from CP2K’s table).
        The compiled program is {(mat.moduleBytes / 1024).toFixed(0)} KB of WebAssembly; the source it came from is {mat.lines.toLocaleString()} lines of Viv —{' '}
        <a href={`/dft/source/pwdft_it`}>read the solver</a>, or <a href={`/dft/source/${mat.id}`}>this deck</a>.
      </p>
    </div>
  );
}
