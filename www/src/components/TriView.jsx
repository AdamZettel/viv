// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
import { useCallback, useEffect, useRef, useState } from 'react';
import { highlight } from '../lib/highlight.js';
import { loadWasm } from './VivCell.jsx';
import { runDirect } from '../lib/vivDirect.js';

// THE THREE RENDERINGS, at once.
//
// Viv has three backends for one source: a tree-walking interpreter, a C99
// backend, and a direct wasm backend; tools/both-engines.sh checks that the
// interpreter and the compiled C print the same thing. The reason this page can show you all three *simultaneously, while
// you type*, is that the entire compiler — lexer, parser, IR store, all three
// backends — is the 2 MB wasm module this component loads. No server is
// involved in anything below.
//
// So: type in the left pane, and the middle pane is the real C99 the knotkit
// backend emits, the right readout is the real size of the module the direct
// wasm backend produces. The answer strip shows what the interpreter printed
// and what that wasm module printed when it ran in the tab, side by side. The
// C is not run in the tab (tools/three-engines.sh runs it natively).

export const DEFAULT_TRI_SRC = `def sq(x) {
    return x * x
}

total = 0.0
for i to 10 {
    total = total + sq(i)
}
show total`;

// Decode a (ptr<<32 | len) result as text.
function callText(ex, fn, text) {
  const enc = new TextEncoder().encode(text);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const packed = fn(ptr, enc.length);
  return new TextDecoder().decode(
    new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn))
  );
}

// viv_emit_wasm hands back raw module bytes on success and a UTF-8 error
// string on failure. A real module always starts with the wasm magic
// \0asm — no error message does — so the magic is the discriminator.
const MAGIC = [0x00, 0x61, 0x73, 0x6d];
function callWasmBytes(ex, text) {
  const enc = new TextEncoder().encode(text);
  const ptr = ex.viv_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
  const packed = ex.viv_emit_wasm(ptr, enc.length);
  const bytes = new Uint8Array(
    ex.memory.buffer,
    Number(packed >> 32n),
    Number(packed & 0xffffffffn)
  );
  const isModule = MAGIC.every((b, i) => bytes[i] === b);
  return isModule
    ? { size: bytes.length, bytes: bytes.slice(), err: null }
    : { size: null, bytes: null, err: new TextDecoder().decode(bytes) };
}

// The C backend puts its runtime (output, errors, Viv's libm) at the top of
// every program; the program's own C is the run of top-level blocks just
// before main and main itself. Show that, and say how long the runtime is.
export function splitC(c) {
  const lines = c.split('\n');
  const main = lines.findIndex((l) => /^int main\(/.test(l));
  if (main < 0) return { own: c, runtime: 0 };
  let start = main;
  for (let i = main - 1; i >= 0; i--) {
    if (lines[i].trim() === '') continue;
    let j = i;
    while (j > 0 && lines[j - 1].trim() !== '') j--;
    if (/^(static|#|\/\*| \*|extern)/.test(lines[j])) break;
    start = j;
    i = j;
  }
  return { own: lines.slice(start).join('\n').trimEnd(), runtime: start };
}

// What a program produced: the last `show`/`print` line that ends in a number,
// with its label (`total: 285`).
function lastValue(out) {
  const lines = out.trim().split('\n').filter((l) => !/^(narrate|__)/.test(l));
  for (let i = lines.length - 1; i >= 0; i--) {
    if (/-?[\d.]+(?:e[-+]?\d+)?\s*$/i.test(lines[i])) return lines[i].trim();
  }
  return null;
}

/** Run the module the direct wasm backend made (`tri.wasmBytes`) against the
 *  runtime in `ex`, and add what it printed to `tri`. */
export async function withWasmRun(ex, tri) {
  if (!tri.wasmBytes || !ex.vrt_out_len) return { ...tri, wasmBytes: null };
  const r = await runDirect(ex, tri.wasmBytes);
  return { ...tri, wasmBytes: null, wasmRun: r.error ? `error: ${r.error}` : r.output, wasmValue: r.error ? null : lastValue(r.output) };
}

/** The three renderings of `text` by the compiler `ex` (without the time they took). */
export function triOf(ex, text) {
  const run = callText(ex, ex.viv_run, text);
  const c = callText(ex, ex.viv_compile, text);
  const w = callWasmBytes(ex, text);
  return {
    status: 'ok',
    run,
    c,
    wasmSize: w.size,
    wasmBytes: w.bytes,
    wasmErr: w.err,
    value: lastValue(run),
    runErr: /^error/m.test(run) ? run : null,
  };
}

// `prebuilt`: triOf the default program, made when the site was built
// (src/pages/index.astro), so the panes are full without loading the compiler;
// its C is already split (cParts), so the page does not carry the C runtime.
// viv.wasm loads on the reader's first edit.
export default function TriView({ src: initial = DEFAULT_TRI_SRC, prebuilt = null }) {
  const [src, setSrc] = useState(initial);
  const [state, setState] = useState(prebuilt ? { ...prebuilt, prebuilt: true } : { status: 'loading' });
  const taRef = useRef(null);
  const timer = useRef(null);

  const compute = useCallback(async (text) => {
    let ex;
    try {
      ex = await loadWasm();
    } catch (e) {
      setState({ status: 'error', msg: e.message || String(e) });
      return;
    }
    const t0 = performance.now();
    const tri = await withWasmRun(ex, triOf(ex, text));
    setState({ ...tri, ms: performance.now() - t0 });
  }, []);

  // Recompute on a short idle so every keystroke does not run three backends.
  useEffect(() => {
    clearTimeout(timer.current);
    if (prebuilt && state.prebuilt && src === initial) return undefined;
    timer.current = setTimeout(() => compute(src), 180);
    return () => clearTimeout(timer.current);
  }, [src, compute]);

  const cOk = state.status === 'ok' && !state.c.startsWith('error');
  const cParts = state.cParts || (cOk ? splitC(state.c) : null);
  const hlRef = useRef(null);

  return (
    <div className="tv">
      <div className="tv-grid">
        <section className="tv-pane tv-src">
          <header>
            <span className="tv-k">your program</span>
            <span className="tv-h">program.viv</span>
          </header>
          <div className="tv-edit">
          <pre className="tv-hl" ref={hlRef} aria-hidden="true" dangerouslySetInnerHTML={{ __html: highlight(src + '\n', 'viv') }} />
          <textarea
            ref={taRef}
            onScroll={(e) => {
              if (hlRef.current) {
                hlRef.current.scrollTop = e.target.scrollTop;
                hlRef.current.scrollLeft = e.target.scrollLeft;
              }
            }}
            value={src}
            spellCheck={false}
            aria-label="Viv source, editable"
            onChange={(e) => setSrc(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Tab') {
                e.preventDefault();
                const el = e.target;
                const { selectionStart: a, selectionEnd: b } = el;
                const next = src.slice(0, a) + '    ' + src.slice(b);
                setSrc(next);
                requestAnimationFrame(() => el.setSelectionRange(a + 4, a + 4));
              }
            }}
          />
          </div>
        </section>

        <section className="tv-pane">
          <header>
            <span className="tv-k">the C99 it emits</span>
            <span className="tv-h">{cParts ? `after a ${cParts.runtime.toLocaleString()}-line runtime` : 'C backend'}</span>
          </header>
          {state.status === 'loading' || state.status === 'error' ? (
            <pre>{state.status === 'loading' ? 'loading the compiler…' : state.msg}</pre>
          ) : (
            <pre dangerouslySetInnerHTML={{ __html: highlight(cParts ? cParts.own : state.c || '', 'c') }} />
          )}
        </section>

        <section className="tv-pane">
          <header>
            <span className="tv-k">and as wasm</span>
            <span className="tv-h">compiled directly, without C</span>
          </header>
          <div className="tv-wasm">
            {state.status !== 'ok' ? (
              <p className="tv-dim">…</p>
            ) : state.wasmSize !== null ? (
              <>
                <b className="tv-bytes">{state.wasmSize.toLocaleString()}</b>
                <span className="tv-dim">bytes of WebAssembly</span>
              </>
            ) : (
              <>
                <b className="tv-declined">not supported</b>
                <pre className="tv-small">{state.wasmErr}</pre>
                <p className="tv-dim">
                  The direct wasm backend does not handle this program yet; the error above
                  names the feature.
                </p>
              </>
            )}
          </div>
        </section>
      </div>

      <div className="tv-answers">
        <div className="tv-a">
          <span className="tv-k">the interpreter prints</span>
          <b className={state.value ? 'tv-hot' : ''}>
            {state.status === 'ok' ? (state.runErr ? 'error' : state.value ?? '—') : '…'}
          </b>
        </div>
        <div className="tv-a">
          <span className="tv-k">the wasm module prints</span>
          <b className={state.wasmValue ? 'tv-hot' : ''}>
            {state.status === 'ok' ? (state.wasmSize === null ? 'not compiled' : state.wasmValue ?? (state.wasmRun ? 'error' : '—')) : '…'}
          </b>
        </div>
        <div className="tv-a">
          <span className="tv-k">all three, in your browser</span>
          <b>{state.prebuilt ? <span className="tv-dim tv-later">timed on your first edit</span> : state.status === 'ok' ? (state.ms < 1 ? '<1 ms' : `${state.ms.toFixed(state.ms < 10 ? 1 : 0)} ms`) : '…'}</b>
        </div>
      </div>

      {state.status === 'ok' && state.runErr && (
        <pre className="tv-runerr">{state.run}</pre>
      )}


      <style>{`
        .tv { margin: 1.6rem 0 0; }
        .tv-grid {
          display: grid; gap: 1px; background: var(--border);
          border: 1px solid var(--border-strong); border-radius: var(--radius);
          overflow: hidden;
          grid-template-columns: 1.08fr 1.08fr 0.84fr;
        }
        @media (max-width: 980px) { .tv-grid { grid-template-columns: 1fr; } }
        .tv-pane { background: var(--bg-card); display: flex; flex-direction: column; min-width: 0; }
        .tv-pane header {
          display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
          padding: 9px 13px; border-bottom: 1px solid var(--border); background: var(--bg-soft);
        }
        .tv-k {
          font: 500 11px var(--font-mono); letter-spacing: 0.1em;
          text-transform: uppercase; color: var(--accent);
        }
        .tv-h { font: 11.5px var(--font-mono); color: var(--text-faint); margin-left: auto; }
        .tv-pane textarea, .tv-pane pre {
          margin: 0; border: 0; background: none; color: var(--text);
          font: 12.5px/1.6 var(--font-mono); padding: 12px 13px;
          height: 330px; overflow: auto; tab-size: 4; white-space: pre;
        }
        .tv-edit { position: relative; }
        .tv-edit .tv-hl, .tv-edit textarea { position: absolute; inset: 0; height: 100%; width: 100%; box-sizing: border-box; }
        .tv-edit { height: 330px; }
        .tv-edit .tv-hl { pointer-events: none; overflow: hidden; color: var(--heading); }
        .tv-edit textarea {
          resize: none; outline: none; color: transparent; caret-color: var(--heading);
          -webkit-text-fill-color: transparent;
        }
        .tv-edit textarea::selection { background: color-mix(in srgb, var(--accent) 25%, transparent); }
        .tv-edit:focus-within { background: color-mix(in srgb, var(--accent) 4%, transparent); }
        .tv-wasm { padding: 18px 15px; height: 330px; overflow: auto; }
        .tv-later { font: 400 13px var(--font-mono); }
        .tv-bytes {
          display: block; font: 600 2.5rem/1 var(--font-mono);
          font-variant-numeric: tabular-nums; color: var(--hot); letter-spacing: -0.03em;
        }
        .tv-declined { display: block; font: 600 1.4rem var(--font-mono); color: var(--text-muted); }
        .tv-wasm .tv-dim { display: block; font: 12px var(--font-mono); color: var(--text-faint); }
        .tv-wasm p.tv-dim { margin: 12px 0 0; line-height: 1.6; }
        .tv-small { font: 11.5px/1.5 var(--font-mono); color: var(--text-muted); margin: 8px 0 0; white-space: pre-wrap; }
        .tv-answers {
          display: flex; flex-wrap: wrap; gap: 1px; margin-top: 1px;
          background: var(--border); border: 1px solid var(--border-strong); border-top: 0;
          border-radius: 0 0 var(--radius) var(--radius); overflow: hidden;
        }
        .tv-a { flex: 1 1 190px; background: var(--bg-card); padding: 11px 14px; }
        .tv-a b {
          display: block; font: 600 1.5rem/1.2 var(--font-mono);
          font-variant-numeric: tabular-nums; color: var(--heading); letter-spacing: -0.02em;
        }
        .tv-a b.tv-hot { color: var(--hot); }
        .tv-runerr {
          font: 12px/1.55 var(--font-mono); color: #ff9a9a; background: var(--bg-soft);
          border: 1px solid var(--border); border-left: 2px solid #b91c1c;
          border-radius: 0 8px 8px 0; padding: 10px 13px; margin: 12px 0 0;
          white-space: pre-wrap; overflow-x: auto;
        }
        .tv-foot {
          font-size: 0.86rem; color: var(--text-muted); margin: 0.9rem 0 0; max-width: 62ch;
        }
      `}</style>
    </div>
  );
}
