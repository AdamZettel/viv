// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// "Run on a cloud worker": for an input an engine page refuses as too large
// for the tab. It appears only when the server behind /api says jobs are
// available (GET /api/health -> jobs.available); without a server the page is
// what it was. The program goes to POST /api/jobs, the page polls the job, and
// when the worker has answered it hands the output to the engine page, which
// renders it with the components a run in the tab uses.
//
// Workers cost the site's owner money, so they are for signed-in members, and
// the server enforces a few at once and a few a day per member.
import { useEffect, useRef, useState } from 'react';
import { api, hasServer } from '../lib/api.js';

const POLL_MS = 2000;
const secs = (s) => (s >= 100 ? `${Math.round(s / 60)} min` : `${s >= 10 ? Math.round(s) : s.toFixed(1)} s`);

/**
 * props:
 *   makeSource  async () => the whole Viv program to run
 *   estimate    a phrase for how long the worker should take, or null
 *   seconds     the same as a number, for the cost estimate, or null
 *   timeoutS    the most the program may run on the worker
 *   onDone      (job) => void, with job = {output, seconds, backend}: the worker answered and the program ran
 */
export default function WorkerRun({ makeSource, estimate, seconds, timeoutS, onDone }) {
  const [jobs, setJobs] = useState(undefined); // undefined: asking; null: none here
  const [session, setSession] = useState(undefined);
  const [job, setJob] = useState(null); // {id, status, error, t0}
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState('');
  const live = useRef(true);

  useEffect(() => {
    live.current = true;
    hasServer().then((h) => {
      if (!live.current) return;
      const j = h && h.jobs && h.jobs.available ? h.jobs : null;
      setJobs(j);
      if (j) api.me().then((m) => live.current && setSession(m.user || null)).catch(() => live.current && setSession(null));
    });
    return () => { live.current = false; };
  }, []);

  // poll the job until the worker has answered
  useEffect(() => {
    if (!job || job.status === 'done' || job.status === 'failed') return undefined;
    const tick = setInterval(() => setElapsed((Date.now() - job.t0) / 1000), 250);
    const poll = setTimeout(async () => {
      try {
        const j = await api.job(job.id);
        if (!live.current) return;
        if (j.status === 'done') {
          const o = j.output;
          setJob({ ...job, status: 'done' });
          // an output with a program's stdout is a run the page can show, even
          // one whose tests failed; one that stopped before running is an error
          if (o.phase === 'done' || (o.phase === 'run' && o.stdout && !/ran longer than/.test(o.error || ''))) onDone({ output: o, seconds: j.seconds, backend: j.backend });
          else setError(o.error || 'the worker did not run the program');
        } else if (j.status === 'failed') {
          setJob({ ...job, status: 'failed' });
          setError(j.error || 'the job failed');
        } else {
          setJob({ ...job, status: j.status });
        }
      } catch (e) {
        if (live.current) { setJob({ ...job, status: 'failed' }); setError(e.message); }
      }
    }, POLL_MS);
    return () => { clearInterval(tick); clearTimeout(poll); };
  }, [job]);

  if (!jobs) return null;
  const where = jobs.backend === 'cloud' ? 'a cloud worker' : 'a local worker';
  const cost = jobs.usdPerHour && seconds ? (seconds * jobs.usdPerHour) / 3600 : null;
  const running = job && (job.status === 'queued' || job.status === 'running');

  const start = async () => {
    setError('');
    try {
      const source = await makeSource();
      const r = await api.submitJob(source, { timeout_s: Math.min(timeoutS, jobs.limits.maxTimeoutS) });
      setElapsed(0);
      setJob({ id: r.id, status: r.status, t0: Date.now() });
    } catch (e) {
      setError(e.message);
    }
  };
  const stop = async () => {
    if (!job) return;
    try { await api.cancelJob(job.id); } catch { /* it may have finished */ }
    setJob(null);
  };

  return (
    <div className="eng-worker" data-testid="worker">
      <p>
        <b>{jobs.backend === 'cloud' ? 'A cloud worker can run it.' : 'A worker on this server can run it.'}</b>{' '}
        The worker compiles the same program to native code and runs it
        {estimate ? <>: {estimate}</> : ''}{cost !== null ? <>, about ${cost < 0.01 ? cost.toFixed(4) : cost.toFixed(2)} at the worker’s rate</> : ''}.
        {jobs.backend === 'local' && ' (Local mode: the server runs jobs on its own machine, the way a cloud worker would.)'}
      </p>
      {session === null && (
        <p><a href={`/account?next=${encodeURIComponent(typeof location !== 'undefined' ? location.pathname : '/engines')}`}>Sign in</a> to use one. Each run costs the site money, so workers are for members.</p>
      )}
      {session && !running && (
        <button type="button" className="eng-btn" onClick={start} data-testid="worker-run">Run on {where}</button>
      )}
      {running && (
        <div className="eng-run">
          <button type="button" className="eng-btn ghost" onClick={stop}>Stop</button>
          <p className="eng-prov" aria-live="polite" data-testid="worker-status">
            {job.status === 'queued' ? 'Waiting for a worker' : 'Running on the worker'} · {elapsed.toFixed(0)} s
          </p>
        </div>
      )}
      {error && <p className="eng-refuse" role="alert" data-testid="worker-error">The worker did not finish: {error}</p>}
      {session && !running && jobs.limits && <p className="eng-note">Up to {jobs.limits.concurrent} at once and {jobs.limits.perDay} a day per member.</p>}
    </div>
  );
}

/** The line a result carries when a worker computed it. */
export function WorkerProvenance({ worker }) {
  const m = worker.machine || {};
  const t = worker.timings || {};
  const build = (t.compile_s || 0) + (t.cc_s || 0);
  return (
    <p className="eng-prov" data-testid="worker-prov">
      Computed on {worker.backend === 'cloud' ? 'a cloud worker' : 'a local worker'}
      {m.model ? ` (${m.model}, a machine with ${m.cpus} core${m.cpus === 1 ? '' : 's'})` : ''} in {(t.run_s || 0).toFixed(1)} s,
      after {build.toFixed(1)} s compiling. Viv {String(worker.viv || '').slice(0, 10)}.
    </p>
  );
}
