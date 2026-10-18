// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// /engines/dft-bands: a crystal file in, a band structure out.
//
// Everything is decided in this tab, before anything runs: the adapter reads
// the file (src/lib/engines/formats/), reduces the crystal to the engine's fcc
// cell and writes the deck (deck.js), the sizes come from the engine's own
// formulas (estimate.js), and anything the engine cannot do is refused here
// with the reason. Then the compiled engine (engines/wasm/dft-bands.wasm) runs
// in a worker with the deck on its standard input, or viv.wasm reads the
// program with the deck pasted in.
import { useEffect, useMemo, useRef, useState } from 'react';
import EngineInput, { Setting, download, startRun } from './EngineInput.jsx';
import { Convergence } from './BandStructure.jsx';
import { parseOwnOutput, Chart } from './VivCell.jsx';
import { parseStructure } from '../lib/engines/formats/index.js';
import { buildDeck, deckBinary, encodeVariables, DECK_VARIABLES, CORNER } from '../lib/engines/deck.js';
import { sizes, estimateSeconds } from '../lib/engines/estimate.js';
import { dftInputBlock, dftReferenceVariables, parseReport, precise, withInput } from '../lib/engines/report.js';
import WorkerRun, { WorkerProvenance } from './WorkerRun.jsx';
import { BOHR_ANGSTROM } from '../lib/engines/crystal.js';
import COST from '../lib/engines/dft-cost.json';

const MAX_SECONDS = 90; // estimated compiled seconds the tab is offered
// the interpreter against the compiled module, measured by enginetest.mjs --run:
// silicon 20.3 s against 8.2, gallium arsenide 34.3 against 11.6
const INTERPRETED = 2.8;
// A worker runs the program compiled to native code, one core. Measured on an
// Apple M2 Pro with a local worker: silicon 6.0 s against 7.4 s
// estimated here, copper 180 s against 172 s. So the estimate above is also
// the worker's, to within a fifth; the timeout leaves three times that.
const workerTimeout = (est) => Math.max(120, Math.ceil(est * 3));

const EXAMPLES = [
  { file: 'si.poscar', label: 'Silicon', note: 'the primitive cell, as VASP writes it' },
  { file: 'si.cif', label: 'Silicon', note: 'the cubic cell and the space group Fd-3m' },
  { file: 'si.pw.in', label: 'Silicon', note: 'a Quantum ESPRESSO input, ibrav = 2' },
  { file: 'si.xyz', label: 'Silicon', note: 'extended XYZ with a Lattice' },
  { file: 'gaas.poscar', label: 'GaAs', note: 'the 8-atom cubic cell, cartesian' },
  { file: 'gaas.cif', label: 'GaAs', note: 'the cubic cell and the space group F-43m' },
  { file: 'gaas.pw.in', label: 'GaAs', note: 'ibrav = 0, cell in angstrom' },
  { file: 'cu.pw.in', label: 'Copper', note: 'about 3 minutes, too long for a browser run' },
  { file: 'fe.pw.in', label: 'Iron', note: 'magnetic, so the engine refuses it' },
];

const f6 = (v) => (v === null || v === undefined || Number.isNaN(v) ? '–' : Number(v).toFixed(6));
const frac = (v) => { const r = Math.round(v * 1e6) / 1e6; return (Object.is(r, -0) ? 0 : r).toString(); };
const secs = (s) => (s >= 100 ? `${Math.round(s / 60)} min` : `${s >= 10 ? Math.round(s) : s.toFixed(1)} s`);

export default function DftBandsEngine({ refs }) {
  const [text, setText] = useState('');
  const [name, setName] = useState('');
  const [over, setOver] = useState({ ecut: '', nkmesh: '', smear: '', nbands: '' });
  const [mode, setMode] = useState('compiled');
  const [run, setRun] = useState(null); // { state: 'running'|'done'|'error', lines, report, seconds, key }
  const [elapsed, setElapsed] = useState(0);
  const handle = useRef(null);

  const onInput = (t, n) => { setText(t); setName(n); setRun(null); setOver({ ecut: '', nkmesh: '', smear: '', nbands: '' }); };
  // open on silicon, so the page shows what it does before anything is dropped
  useEffect(() => {
    fetch('/engines/examples/si.poscar').then((r) => r.text()).then((t) => { setText((cur) => cur || t); setName((cur) => cur || 'si.poscar'); });
    return () => handle.current && handle.current.stop();
  }, []);

  // ---- what the adapter understood, recomputed as the input changes
  const understood = useMemo(() => {
    if (!text.trim()) return null;
    try {
      const parsed = parseStructure(text, name);
      const overrides = Object.fromEntries(Object.entries(over).filter(([, v]) => v !== ''));
      const built = buildDeck(parsed, overrides);
      let sz = null, est = null, input = null, tooBig = false;
      if (built.deck) {
        sz = sizes(built.deck);
        est = estimateSeconds(sz, built.deck.NBANDS, COST);
        tooBig = est > MAX_SECONDS;
        if (tooBig) {
          built.refusals.push(`Estimated at ${secs(est)} compiled in your browser (${sz.npwmax} plane waves at ${sz.nkpt} k-points): more than the ${MAX_SECONDS} s a browser run allows. Run it on your own machine with dft/run.sh (see dft/README.md).`);
        }
        input = dftInputBlock(built, name, refs);
      }
      return { parsed, built, sz, est, input, tooBig };
    } catch (e) {
      if (e.refusal) return { refusal: e.message, detail: e.detail };
      return { refusal: `Could not read this input: ${e.message}` };
    }
  }, [text, name, over, refs]);

  const built = understood && understood.built;
  const canRun = built && built.deck && !built.refusals.length;
  // refused only because the tab would take too long: a worker can take it
  const forWorker = built && built.deck && understood.tooBig && built.refusals.length === 1;

  const workerSource = async () => {
    const r = await fetch('/engines/dft-bands.viv');
    if (!r.ok) throw new Error(`could not load the engine program (${r.status})`);
    return withInput(await r.text(), understood.input.block);
  };
  const workerDone = ({ output, backend }) => {
    const { input } = understood;
    setRun({
      state: 'done', mode: 'worker', seconds: output.timings.run_s, text: output.stdout, report: parseReport(output.stdout),
      deck: input.text, ref: input.ref, name, worker: { backend, machine: output.machine, timings: output.timings, viv: output.viv },
    });
  };

  useEffect(() => {
    if (!run || run.state !== 'running') return undefined;
    const t0 = performance.now();
    const id = setInterval(() => setElapsed((performance.now() - t0) / 1000), 200);
    return () => clearInterval(id);
  }, [run && run.state]);

  const start = () => {
    if (!canRun) return;
    if (handle.current) handle.current.stop();
    const { input } = understood;
    const lines = [];
    setElapsed(0);
    setRun({ state: 'running', lines: [], mode });
    const message = mode === 'compiled'
      ? { mode, wasm: '/engines/dft-bands.wasm', stdin: encodeVariables({ ...built.deck, ...dftReferenceVariables(input.ref) }, [...DECK_VARIABLES, 'HAS_REF', 'REF_ETOT', 'REF_NB', 'REFPATH']) }
      : { mode, program: '/engines/dft-bands.viv', block: input.block };
    handle.current = startRun(message, {
      onLine: (l) => { lines.push(l); if (/^(narrate|ETOT|DRHO|DECK)/.test(l)) setRun((r) => (r && r.state === 'running' ? { ...r, lines: [...lines] } : r)); },
      onDone: (d) => { handle.current = null; setRun({ state: 'done', mode, seconds: d.seconds, memoryMB: d.memoryMB, text: d.text, report: parseReport(d.text), deck: input.text, ref: input.ref, name }); },
      onError: (msg) => { handle.current = null; setRun({ state: 'error', error: msg, mode }); },
    });
  };
  const stop = () => { if (handle.current) handle.current.stop(); handle.current = null; setRun(null); };

  const rep = run && run.report;
  const bandCharts = useMemo(() => (run && run.text ? parseOwnOutput(run.text).charts.filter((c) => c.fig) : []), [run]);
  const etot = rep ? precise(rep.shown, 'ETOT') : null;
  const progress = run && run.state === 'running' ? progressOf(run.lines) : null;

  return (
    <div className="eng">
      <EngineInput examples={EXAMPLES} text={text} name={name} onInput={onInput} accept=".cif,.vasp,.poscar,.in,.pwi,.xyz,.extxyz,*"
        placeholder="A POSCAR, a CIF, a pw.x input, or an extended XYZ file with a Lattice=&quot;...&quot; cell" />

      {understood && understood.refusal && (
        <div className="eng-refuse" role="alert" data-testid="refusal">
          <b>Cannot run this.</b> {understood.refusal}
          {understood.detail && <p>{understood.detail}</p>}
        </div>
      )}

      {built && <Understood u={understood} over={over} setOver={setOver} />}

      {built && built.refusals.length > 0 && (
        <div className="eng-refuse" role="alert" data-testid="refusal">
          <b>Cannot run this{built.refusals.length > 1 ? `, for ${built.refusals.length} reasons` : ''}.</b>
          <ul>{built.refusals.map((r) => <li key={r}>{r}</li>)}</ul>
        </div>
      )}

      {forWorker && (
        <WorkerRun key={`${name}:${understood.est}`} makeSource={workerSource} timeoutS={workerTimeout(understood.est)}
          estimate={`about ${secs(understood.est)} on one core`} seconds={understood.est} onDone={workerDone} />
      )}

      {canRun && (
        <div className="eng-run">
          <div className="eng-modes" role="radiogroup" aria-label="Engine">
            {[['compiled', 'Compiled', 'Viv → C → wasm, one module that runs any deck'], ['interpreted', 'Interpreted', 'viv.wasm interprets the program']].map(([id, label, sub]) => (
              <button key={id} type="button" role="radio" aria-checked={mode === id} className={`eng-mode${mode === id ? ' on' : ''}`}
                disabled={run && run.state === 'running'} onClick={() => setMode(id)}>
                <b>{label}</b><span>{sub}</span>
              </button>
            ))}
          </div>
          {run && run.state === 'running' ? (
            <button type="button" className="eng-btn ghost" onClick={stop}>Stop</button>
          ) : (
            <button type="button" className="eng-btn" onClick={start} data-testid="run">
              Run · about {secs(understood.est * (mode === 'interpreted' ? INTERPRETED : 1))}
            </button>
          )}
          <p className="eng-prov" aria-live="polite">
            {run && run.state === 'running' && <>{progress} · {elapsed.toFixed(1)} s</>}
            {run && run.state === 'done' && <>Computed in your browser in {run.seconds.toFixed(1)} s by the {run.mode} engine.</>}
            {!run && `Time estimated from the sizes above, for the ${mode} engine.`}
          </p>
        </div>
      )}
      {run && run.state === 'error' && <p className="eng-refuse" role="alert">The run did not finish: {run.error}</p>}

      {rep && run.state === 'done' && (
        <div className="eng-result" data-testid="result">
          {run.worker && <WorkerProvenance worker={run.worker} />}
          {/* the program's own figure: its bands, its L/Γ/X ticks, its zero line and gap band */}
          {bandCharts.map((c) => <div className="bs-frame" key={c.key}><Chart chart={c} /></div>)}
          <dl className="eng-facts">
            <div><dt>total energy</dt><dd className="num" data-testid="etot">{etot !== null ? etot.toFixed(10) : '–'} Ha</dd></div>
            {rep.shown.GAP > 0 && <div><dt>gap along L–Γ–X</dt><dd className="num" data-testid="gap">{f6(rep.shown.GAP).slice(0, 5)} eV</dd></div>}
            <div><dt>SCF iterations</dt><dd>{rep.shown.ITER}</dd></div>
            <div><dt>plane waves</dt><dd>{rep.shown.NPWMAX}</dd></div>
            <div><dt>tests</dt><dd className={rep.fail.length ? 'bad' : 'hot'} data-testid="tests">{rep.pass.length} of {rep.pass.length + rep.fail.length} pass</dd></div>
          </dl>
          <p className="eng-note">
            The total energy leaves out the ion–ion (Ewald) term, as dft/ does: it compares runs on the same
            atoms, not different structures.
          </p>
          {run.ref ? (
            <div className="eng-ref" data-testid="reference">
              <b>Checked against the dft/ reference.</b> This deck is identical to <code>dft/{run.ref.id}.viv</code>, so the numpy
              reference pinned in <code>dft/{run.ref.id}_it_check.viv</code> applies: the energy is{' '}
              <span className="num">{rep.shown.REF_DE !== undefined ? rep.shown.REF_DE.toExponential(1) : '–'}</span> Ha from it
              (tolerance {run.ref.etol}), and every band at every path point within{' '}
              <span className="num">{rep.shown.REF_DBAND !== undefined ? rep.shown.REF_DBAND.toExponential(1) : '–'}</span> eV
              (tolerance {run.ref.btol}).
              {rep.shown.REF_DE < run.ref.etol && rep.shown.REF_DBAND < run.ref.btol ? ' Both are within tolerance.' : ' AT LEAST ONE IS OUTSIDE ITS TOLERANCE.'}
            </div>
          ) : (
            <div className="eng-ref none">
              No reference: dft/ has no deck identical to this one{built && built.unvalidated ? ', and some settings were not measured for these elements' : ''}.
              The engine’s own tests (convergence of the density and of every band) are the only checks on this run.
            </div>
          )}
          <ul className="eng-tests">
            {rep.pass.map((n) => <li key={n} className="ok"><span aria-hidden="true">✓</span>{n}</li>)}
            {rep.fail.map((n) => <li key={n} className="bad"><span aria-hidden="true">✗</span>{n}</li>)}
          </ul>
          <figure className="eng-conv">
            <Convergence drho={rep.series.DRHO ? rep.series.DRHO.slice(0, -1) : []} />
            <figcaption>density residual, per SCF iteration</figcaption>
          </figure>
          <section className="eng-output">
            <h3 className="eng-output-title">What the program printed</h3>
            <p className="eng-note">Everything on this page comes from these lines: the figure from its <code>figure</code>, <code>plot</code>, <code>xtick</code>, <code>hline</code>, <code>rect</code> and <code>label</code> records, the numbers from its <code>show</code>s, the checklist from its tests.</p>
            <pre data-testid="program-output">{run.text}</pre>
          </section>
          <div className="eng-downloads">
            <button type="button" className="eng-pick" onClick={() => download(`${stem(run.name)}.viv`, run.deck)}>Download the deck (.viv)</button>
            <button type="button" className="eng-pick" onClick={() => download(`${stem(run.name)}-bands.csv`, bandsCsv(rep), 'text/csv')}>Bands (.csv)</button>
            <button type="button" className="eng-pick" onClick={() => download(`${stem(run.name)}-result.json`, JSON.stringify(resultJson(run, built, etot), null, 1), 'application/json')}>Everything (.json)</button>
          </div>
        </div>
      )}
    </div>
  );
}

const stem = (n) => (n || 'crystal').replace(/\.[^.]*$/, '').replace(/[^A-Za-z0-9_-]+/g, '-') || 'crystal';

function progressOf(lines) {
  const stage = [...lines].reverse().find((l) => l.startsWith('narrate: '));
  const iters = lines.filter((l) => l.startsWith('DRHO: ')).length;
  const last = [...lines].reverse().find((l) => l.startsWith('DRHO: '));
  if (!stage) return 'Starting';
  const s = stage.slice(9);
  if (s === 'self-consistent field') return `Self-consistent field: iteration ${iters}${last ? `, density residual ${Number(last.slice(6)).toExponential(1)}` : ''}`;
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function bandsCsv(rep) {
  const head = ['k_path_length_inverse_bohr', ...rep.bands.map((_, b) => `band_${b + 1}_eV`)];
  const rows = rep.x.map((x, i) => [x, ...rep.bands.map((e) => e[i])].join(','));
  return `${head.join(',')}\n${rows.join('\n')}\n`;
}

function resultJson(run, built, etot) {
  const r = run.report;
  return {
    engine: 'dft-bands', input: run.name, format: built.formatName, mode: run.mode, seconds: run.seconds,
    worker: run.worker || null,
    total_energy_hartree: etot, total_energy_note: 'no ion-ion (Ewald) term',
    gap_eV: r.shown.GAP, scf_iterations: r.shown.ITER, npwmax: r.shown.NPWMAX, nkpt: r.shown.NKPT, fft_grid: r.shown.NG,
    reference: run.ref ? { deck: `dft/${run.ref.id}.viv`, energy_error: r.shown.REF_DE, band_error_eV: r.shown.REF_DBAND } : null,
    tests: { pass: r.pass, fail: r.fail },
    band_path: { corners: { L: CORNER[0], Gamma: CORNER[1], X: CORNER[2] }, x: r.x, bands_eV: r.bands },
    deck: run.deck,
  };
}

// ---- what the adapter understood ---------------------------------------------
function Understood({ u, over, setOver }) {
  const { parsed, built, sz, est } = u;
  const c = built.crystal;
  const set = (k) => (v) => setOver((o) => ({ ...o, [k]: v }));
  return (
    <div className="eng-understood" data-testid="understood">
      <h3>What it read</h3>
      <dl className="eng-kv">
        <div><dt>format</dt><dd>{built.formatName}{parsed.title ? ` — “${parsed.title}”` : ''}</dd></div>
        <div><dt>lattice</dt><dd>face-centered cubic, a = {c.alat.toFixed(6)} bohr ({(c.alat * BOHR_ANGSTROM).toFixed(6)} Å)</dd></div>
        <div><dt>cell</dt><dd>the file’s cell has {c.nInput} atom{c.nInput > 1 ? 's' : ''}; the primitive cell the engine computes has {c.nPrimitive}</dd></div>
        {built.material && <div><dt>recognized</dt><dd>the composition of {built.material} in <code>dft/ref/materials.py</code></dd></div>}
      </dl>
      <table className="eng-table">
        <caption>Atoms in the primitive cell, in fractions of dft/’s fcc vectors (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)</caption>
        <thead><tr><th>#</th><th>element</th><th className="r">f₁</th><th className="r">f₂</th><th className="r">f₃</th></tr></thead>
        <tbody>
          {c.atoms.map((a, i) => (
            <tr key={i}><td>{i + 1}</td><td>{a.el}</td>{a.frac.map((v, k) => <td key={k} className="num r">{frac(v)}</td>)}</tr>
          ))}
        </tbody>
      </table>
      <table className="eng-table">
        <caption>Species and the pseudopotentials the engine will use (Goedecker–Teter–Hutter, LDA, from CP2K’s table)</caption>
        <thead><tr><th>element</th><th>pseudopotential</th><th className="r">valence electrons</th></tr></thead>
        <tbody>
          {built.species.map((el, i) => (
            <tr key={el}><td>{el}</td><td>{built.pseudos[i] ? built.pseudos[i].name : 'none'}</td><td className="num r">{built.pseudos[i] ? built.pseudos[i].zion : '–'}</td></tr>
          ))}
        </tbody>
      </table>
      <div className="eng-settings">
        <h4>Settings</h4>
        <Setting label="cutoff" unit="Ha" setting={built.settings.ecut} value={over.ecut} onChange={set('ecut')} hint="no measured cutoff: give one" />
        <Setting label="k-mesh" unit="× 4 shifts" setting={built.settings.nkmesh} value={over.nkmesh} onChange={set('nkmesh')} step="1" />
        <Setting label="smearing" unit="Ha" setting={built.settings.smear} value={over.smear} onChange={set('smear')} />
        <Setting label="bands" unit="" setting={built.settings.nbands} value={over.nbands} onChange={set('nbands')} step="1" />
      </div>
      {sz && (
        <dl className="eng-kv">
          <div><dt>electrons</dt><dd>{built.nelec} per cell ({built.deck.SMEAR > 0 ? 'a metal: Fermi–Dirac occupations' : 'an insulator: the lowest bands filled'})</dd></div>
          <div><dt>k-points</dt><dd>{sz.nkpt} for the self-consistent field (the four-shift fcc mesh of <code>dft/kmesh.viv</code>, halved by time reversal)</dd></div>
          <div><dt>band path</dt><dd>L ({CORNER[0].join(', ')}) → Γ (0, 0, 0) → X ({CORNER[2].join(', ')}), {sz.npts} points</dd></div>
          <div><dt>plane waves</dt><dd className="num" data-testid="npw">up to {sz.npwmax} per k-point; FFT grid {sz.ng}³</dd></div>
          <div><dt>cost</dt><dd>about {secs(est)} compiled, {secs(est * INTERPRETED)} interpreted, and {Math.round(sz.memoryMB)} MB</dd></div>
        </dl>
      )}
      {built.notes.length > 0 && <ul className="eng-notes">{built.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
    </div>
  );
}
