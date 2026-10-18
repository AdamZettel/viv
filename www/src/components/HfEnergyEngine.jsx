// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// /engines/hf-energy: an XYZ file and a basis name in, a Hartree-Fock energy
// and orbital energies out.
//
// The adapter (src/lib/engines/molecule.js) builds lib/hf's basis packs from
// PySCF's basis table, says what it built, and refuses what lib/hf or the tab
// cannot take. viv.wasm then runs engines/hf-energy.viv with the molecule
// pasted into its input block, in a worker.
import { useEffect, useMemo, useRef, useState } from 'react';
import EngineInput, { download, startRun } from './EngineInput.jsx';
import { parseMolecule, buildMolecule, moleculeBlock, hfReferenceFor, resolvedMethod, primitiveFunctions, BASIS_NAMES, METHOD_NAMES, BASES } from '../lib/engines/molecule.js';
import { parseReport, precise, withInput } from '../lib/engines/report.js';
import WorkerRun, { WorkerProvenance } from './WorkerRun.jsx';

const EXAMPLES = [
  { file: 'water.xyz', label: 'Water', note: '6-31G, closed shell: RHF' },
  { file: 'oh.xyz', label: 'Hydroxyl', note: '6-31G, one unpaired electron: UHF' },
  { file: 'ammonia.xyz', label: 'Ammonia', note: '6-31G*, with d functions' },
];
const HA_EV = 27.211386245988;
const secs = (s) => (s >= 100 ? `${Math.round(s / 60)} min` : `${s >= 10 ? Math.round(s) : s.toFixed(1)} s`);
// The tab's estimate is for the interpreter. Compiled to native code on a worker
// lib/hf is far faster, by a factor that grows with the molecule, so no worker
// time is promised; measured with a local worker on an Apple M2 Pro:
// water in 6-31G 0.2 s (about 3 s in the tab), benzene in STO-3G 0.6 s against
// 142 s estimated here, each after about 4 s of compiling the C.
const WORKER_TIMEOUT = 600;

export default function HfEnergyEngine({ refs }) {
  const [text, setText] = useState('');
  const [name, setName] = useState('');
  const [opts, setOpts] = useState({ basis: '', charge: '', spin: '', method: '', spherical: '' });
  const [run, setRun] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const handle = useRef(null);
  const onInput = (t, n) => { setText(t); setName(n); setRun(null); setOpts({ basis: '', charge: '', spin: '', method: '', spherical: '' }); };
  useEffect(() => {
    fetch('/engines/examples/water.xyz').then((r) => r.text()).then((t) => { setText((c) => c || t); setName((c) => c || 'water.xyz'); });
    return () => handle.current && handle.current.stop();
  }, []);

  const u = useMemo(() => {
    if (!text.trim()) return null;
    try {
      const parsed = parseMolecule(text);
      const o = {};
      if (opts.basis) o.basis = opts.basis;
      if (opts.charge !== '') o.charge = Number(opts.charge);
      if (opts.spin !== '') o.spin = Number(opts.spin);
      if (opts.method) o.method = opts.method;
      if (opts.spherical !== '') o.spherical = opts.spherical === 'true';
      const built = buildMolecule(parsed, o);
      // a molecule refused only for its size still has lib/hfref's reference, for a worker's run
      const onlySize = built.tooBig && built.refusals.length === 1;
      const ref = built.refusals.length && !onlySize ? null : hfReferenceFor(built, refs);
      return { parsed, built, ref };
    } catch (e) {
      return { refusal: e.refusal ? e.message : `Could not read this input: ${e.message}` };
    }
  }, [text, opts, refs]);
  const built = u && u.built;
  const canRun = built && built.packs && !built.refusals.length;
  const forWorker = built && built.packs && built.tooBig && built.refusals.length === 1;
  const workerSource = async () => {
    const r = await fetch('/engines/hf-energy.viv');
    if (!r.ok) throw new Error(`could not load the engine program (${r.status})`);
    return withInput(await r.text(), moleculeBlock(built, u.ref));
  };
  const workerDone = ({ output, backend }) => {
    setRun({
      state: 'done', seconds: output.timings.run_s, report: parseReport(output.stdout), block: moleculeBlock(built, u.ref), ref: u.ref, built, name,
      worker: { backend, machine: output.machine, timings: output.timings, viv: output.viv },
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
    const block = moleculeBlock(built, u.ref);
    setElapsed(0);
    setRun({ state: 'running' });
    handle.current = startRun({ mode: 'interpreted', program: '/engines/hf-energy.viv', block }, {
      onDone: (d) => { handle.current = null; setRun({ state: 'done', seconds: d.seconds, report: parseReport(d.text), block, ref: u.ref, built, name }); },
      onError: (msg) => { handle.current = null; setRun({ state: 'error', error: msg }); },
    });
  };
  const stop = () => { if (handle.current) handle.current.stop(); handle.current = null; setRun(null); };
  const set = (k) => (e) => setOpts((o) => ({ ...o, [k]: e.target.value }));

  const rep = run && run.state === 'done' ? run.report : null;
  const energy = rep ? precise(rep.shown, 'E') : null;

  return (
    <div className="eng">
      <EngineInput examples={EXAMPLES} text={text} name={name} onInput={onInput} accept=".xyz,*"
        placeholder={'3\nwater  basis=6-31g\nO 0 0 0\nH 0 -0.757 0.587\nH 0 0.757 0.587'} />
      {u && u.refusal && <div className="eng-refuse" role="alert" data-testid="refusal"><b>Cannot run this.</b> {u.refusal}</div>}
      {built && (
        <div className="eng-understood" data-testid="understood">
          <h3>What it read</h3>
          <div className="eng-hfopts">
            <label>basis
              <select value={opts.basis || built.basis} onChange={set('basis')} aria-label="basis">
                {BASIS_NAMES.map((b) => <option key={b} value={b}>{b}</option>)}
              </select>
            </label>
            <label>charge <input type="number" step="1" value={opts.charge} placeholder={String(built.charge)} onChange={set('charge')} aria-label="charge" /></label>
            <label><span>N<sub>α</sub> − N<sub>β</sub></span> <input type="number" step="1" min="0" value={opts.spin} placeholder={String(built.spin)} onChange={set('spin')} aria-label="spin" /></label>
            <label>method
              <select value={opts.method} onChange={set('method')} aria-label="method">
                <option value="">auto ({METHOD_NAMES[resolvedMethod(built)]})</option>
                <option value="rhf">RHF</option><option value="uhf">UHF</option><option value="rohf">ROHF</option>
              </select>
            </label>
            <label>functions
              <select value={opts.spherical === '' ? String(built.spherical) : opts.spherical} onChange={set('spherical')} aria-label="spherical or Cartesian">
                <option value="true">spherical</option><option value="false">Cartesian</option>
              </select>
            </label>
          </div>
          <table className="eng-table">
            <caption>Atoms, in angstrom as the file gives them{u.parsed.comment ? ` — “${u.parsed.comment}”` : ''}</caption>
            <thead><tr><th>#</th><th>element</th><th className="r">x</th><th className="r">y</th><th className="r">z</th></tr></thead>
            <tbody>{built.atoms.map((a, i) => <tr key={i}><td>{i + 1}</td><td>{a.el}</td>{a.x.map((v, k) => <td key={k} className="num r">{v.toFixed(6)}</td>)}</tr>)}</tbody>
          </table>
          <dl className="eng-kv">
            <div><dt>electrons</dt><dd>{built.nelec}: {built.nalpha} α and {built.nbeta} β (charge {built.charge})</dd></div>
            <div><dt>method</dt><dd>{METHOD_NAMES[resolvedMethod(built)]}{built.method === 0 ? ` (${built.settings.method.from})` : ''}</dd></div>
            <div><dt>basis</dt><dd>{built.basis}, from PySCF {BASES.pyscf}’s library ({built.settings.basis.from}); {built.spherical ? 'spherical' : 'Cartesian'} functions</dd></div>
            {built.packs && <div><dt>size</dt><dd className="num" data-testid="nao">{built.nao} basis functions, {built.packs.shells.length} shells, {built.packs.prims.length} primitives ({primitiveFunctions(built)} primitive Cartesian functions)</dd></div>}
            {built.seconds !== null && <div><dt>cost</dt><dd>about {secs(built.seconds)} in your browser</dd></div>}
            <div><dt>reference</dt><dd>{u.ref ? <>lib/hfref holds this exact calculation: <code>{u.ref.name}</code> = {u.ref.energy} Ha, from PySCF</> : 'none: lib/hfref holds no calculation identical to this one'}</dd></div>
          </dl>
          {built.notes.length > 0 && <ul className="eng-notes">{built.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
        </div>
      )}
      {built && built.refusals.length > 0 && (
        <div className="eng-refuse" role="alert" data-testid="refusal">
          <b>Cannot run this.</b>
          <ul>{built.refusals.map((r) => <li key={r}>{r}</li>)}</ul>
        </div>
      )}
      {forWorker && (
        <WorkerRun key={`${name}:${built.nao}:${built.basis}`} makeSource={workerSource} timeoutS={WORKER_TIMEOUT}
          estimate={`far faster than the ${secs(built.seconds)} this tab would take (benzene in STO-3G: 0.6 s, against 142 s estimated here)`} seconds={null} onDone={workerDone} />
      )}
      {canRun && (
        <div className="eng-run">
          {run && run.state === 'running'
            ? <button type="button" className="eng-btn ghost" onClick={stop}>Stop</button>
            : <button type="button" className="eng-btn" onClick={start} data-testid="run">Run · about {secs(built.seconds)}</button>}
          <p className="eng-prov" aria-live="polite">
            {run && run.state === 'running' && <>Computing integrals and iterating the SCF… {elapsed.toFixed(1)} s</>}
            {run && run.state === 'done' && <>Computed in your browser in {run.seconds.toFixed(1)} s by viv.wasm.</>}
            {!run && 'Runs lib/hf in viv.wasm, with the default algorithm at every step.'}
          </p>
        </div>
      )}
      {run && run.state === 'error' && <p className="eng-refuse" role="alert">The run did not finish: {run.error}</p>}
      {rep && <HfResult run={run} rep={rep} energy={energy} />}
    </div>
  );
}

function HfResult({ run, rep, energy }) {
  const b = run.built;
  const m = resolvedMethod(b);
  const orbitals = (vals, nocc) => vals && (
    <tbody>
      {vals.map((e, i) => (
        <tr key={i} className={i < nocc ? 'occ' : ''}>
          <td>{i + 1}</td><td>{i < nocc ? 'occupied' : 'virtual'}{i === nocc - 1 ? ' (HOMO)' : i === nocc ? ' (LUMO)' : ''}</td>
          <td className="num r">{e.toFixed(6)}</td><td className="num r">{(e * HA_EV).toFixed(4)}</td>
        </tr>
      ))}
    </tbody>
  );
  return (
    <div className="eng-result" data-testid="result">
      {run.worker && <WorkerProvenance worker={run.worker} />}
      <dl className="eng-facts">
        <div><dt>{METHOD_NAMES[m]} energy</dt><dd className="num hot" data-testid="energy">{energy !== null ? energy.toFixed(10) : '–'} Ha</dd></div>
        <div><dt>nuclear repulsion</dt><dd className="num">{rep.shown.E_NUC} Ha</dd></div>
        <div><dt>SCF iterations</dt><dd>{rep.shown.ITERATIONS}</dd></div>
        {rep.shown.S2 !== undefined && <div><dt>⟨S²⟩</dt><dd className="num">{rep.shown.S2}</dd></div>}
        <div><dt>tests</dt><dd className={rep.fail.length ? 'bad' : 'hot'}>{rep.pass.length} of {rep.pass.length + rep.fail.length} pass</dd></div>
      </dl>
      {run.ref ? (
        <div className="eng-ref" data-testid="reference">
          <b>Checked against PySCF.</b> <code>{run.ref.module}</code> holds this calculation: <code>{run.ref.name}</code> = {run.ref.energy} Ha.
          This run is <span className="num">{rep.shown.REF_DE !== undefined ? rep.shown.REF_DE.toExponential(1) : '–'}</span> Ha from it
          (lib/hf’s tolerance for matching PySCF is 1e-9 Ha).{rep.shown.REF_DE < 1e-9 ? ' Within tolerance.' : ' OUTSIDE TOLERANCE.'}
        </div>
      ) : (
        <div className="eng-ref none">No reference: lib/hfref holds no calculation identical to this one, so the convergence test is the only check on this run.</div>
      )}
      {rep.shown.MO_ALPHA ? (
        <table className="eng-table orbitals">
          <caption>Orbital energies{rep.shown.MO_BETA ? ', α spin' : ''}: the eigenvalues of the converged Fock matrix</caption>
          <thead><tr><th>#</th><th /><th className="r">hartree</th><th className="r">eV</th></tr></thead>
          {orbitals(rep.shown.MO_ALPHA, b.nalpha)}
        </table>
      ) : <p className="eng-note">ROHF has no unique orbital energies (they depend on how the effective Fock matrix is built), so none are shown.</p>}
      {rep.shown.MO_BETA && (
        <table className="eng-table orbitals">
          <caption>Orbital energies, β spin</caption>
          <thead><tr><th>#</th><th /><th className="r">hartree</th><th className="r">eV</th></tr></thead>
          {orbitals(rep.shown.MO_BETA, b.nbeta)}
        </table>
      )}
      <div className="eng-downloads">
        <button type="button" className="eng-pick" onClick={() => download(`${stem(run.name)}-hf-input.viv`, run.block)}>Download the input block (.viv)</button>
        <button type="button" className="eng-pick" onClick={() => download(`${stem(run.name)}-hf.json`, JSON.stringify({
          engine: 'hf-energy', input: run.name, basis: b.basis, spherical: b.spherical, charge: b.charge, spin: b.spin, method: METHOD_NAMES[m],
          energy_hartree: energy, nuclear_repulsion: rep.shown.E_NUC, iterations: rep.shown.ITERATIONS, s2: rep.shown.S2 ?? null,
          orbital_energies_alpha: rep.shown.MO_ALPHA || null, orbital_energies_beta: rep.shown.MO_BETA || null,
          reference: run.ref ? { constant: run.ref.name, module: run.ref.module, energy: run.ref.energy, difference: rep.shown.REF_DE } : null,
          seconds: run.seconds, worker: run.worker || null, tests: { pass: rep.pass, fail: rep.fail },
        }, null, 1), 'application/json')}>Everything (.json)</button>
      </div>
    </div>
  );
}

const stem = (n) => (n || 'molecule').replace(/\.[^.]*$/, '').replace(/[^A-Za-z0-9_-]+/g, '-') || 'molecule';
