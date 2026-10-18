// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The input side of an engine page: drop a file, pick one, paste text, or load
// an example. It hands the text and a file name up and knows nothing about
// what the text means; the engine's adapter decides that.
import { useRef, useState } from 'react';
import { VIV_WASM_URL, withVivModule, postVivMessage } from '../lib/vivWasm.js';

export default function EngineInput({ examples, text, name, onInput, accept, placeholder }) {
  const [over, setOver] = useState(false);
  const file = useRef(null);
  const readFile = (f) => {
    if (!f) return;
    f.text().then((t) => onInput(t, f.name));
  };
  const loadExample = async (ex) => {
    const r = await fetch(`/engines/examples/${ex.file}`);
    onInput(await r.text(), ex.file);
  };
  return (
    <div className="eng-input">
      <div className="eng-examples" role="group" aria-label="Example inputs">
        <span className="eng-label">Examples</span>
        {examples.map((ex) => (
          <button key={ex.file} type="button" className={`eng-ex${ex.file === name ? ' on' : ''}`} onClick={() => loadExample(ex)}
            title={ex.note || ex.file}>
            {ex.label}<span>{ex.file}</span>
          </button>
        ))}
      </div>
      <div className={`eng-drop${over ? ' over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); readFile(e.dataTransfer.files[0]); }}>
        <div className="eng-drop-bar">
          <span>{name ? <>Reading <b>{name}</b></> : 'Drop a file here, or paste one below'}</span>
          <button type="button" className="eng-pick" onClick={() => file.current && file.current.click()}>Choose a file…</button>
          <input ref={file} type="file" accept={accept} hidden onChange={(e) => readFile(e.target.files[0])} data-testid="engine-file" />
        </div>
        <textarea className="eng-text" spellCheck={false} value={text} placeholder={placeholder}
          aria-label="Input file" onChange={(e) => onInput(e.target.value, '')} rows={10} />
      </div>
    </div>
  );
}

/** A setting with where its value came from, editable. */
export function Setting({ label, unit, setting, value, onChange, step, hint }) {
  return (
    <label className="eng-setting">
      <span className="eng-setting-name">{label}</span>
      <input type="number" step={step || 'any'} value={value} placeholder={setting ? String(setting.value) : ''}
        onChange={(e) => onChange(e.target.value)} aria-label={label} />
      <span className="eng-unit">{unit}</span>
      <span className={`eng-from${setting && setting.user ? ' user' : ''}`}>{setting ? setting.from : hint || ''}</span>
    </label>
  );
}

/** Save text or bytes as a file. */
export function download(name, content, type = 'text/plain') {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Run an engine in public/engine-worker.js; returns a handle with stop(). */
export function startRun(message, { onLine, onDone, onError }) {
  const w = new Worker(`/engine-worker.js?wasm=${encodeURIComponent(VIV_WASM_URL)}`, { type: 'module' });
  w.onmessage = (e) => {
    const d = e.data;
    if (d.line !== undefined) { onLine && onLine(d.line); return; }
    w.terminate();
    if (d.error) onError(d.error);
    else onDone(d);
  };
  w.onerror = (e) => { w.terminate(); onError(e.message || 'the engine could not start in your browser; reload the page and try again'); };
  // interpreted: the worker gets the page's compiled viv.wasm with the job
  if (message.mode === 'interpreted') withVivModule(message).then((m) => postVivMessage(w, m), () => w.postMessage(message));
  else w.postMessage(message);
  return { stop: () => w.terminate() };
}
