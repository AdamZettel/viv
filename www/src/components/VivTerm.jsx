// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { loadWasm, callWasm, joinKept } from './VivCell.jsx';
import { compileDirect, runDirect } from '../lib/vivDirect.js';

/* global __VIV_VERSION__ */
const VIV_VERSION = typeof __VIV_VERSION__ === 'string' ? __VIV_VERSION__ : '';

// The Viv CLI, in the browser. Every verb below is the same function the native
// `viv` binary calls — lib.rs exports them across the wasm ABI, and main.rs is
// a thin argv wrapper around the identical entry points. So this is not a
// simulation of the CLI: `hash`, `diff`, `canon`, `doc`, `whatis` and the rest
// return byte-for-byte what the installed compiler returns on the same file.
//
// Props match the shape the IDE passes:
//   files     [{ name, url }]  fetched on demand, cached after first read
//   liveFiles { name: () => string }  virtual files backed by a live buffer
//   fill      stretch to the container instead of a fixed height

const VERBS = [
  ['run <file>', 'run it and print its output, PASS/FAIL test lines included'],
  ['c <file>', 'compile it to C99 and print the C'],
  ['wasm <file>', 'compile it directly to a WebAssembly module, without C, and run that'],
  ['doc <file> [fn]', "the program's doc comments, by function: what, how, why, tags, glossary"],
  ['whatis <file> <name>', 'what a name is: its kind, its type, and where it is defined'],
  ['lsfunc <file>', 'every function with its parameters'],
  ['lsvar <file> [fn]', 'every variable, with its glossary line if it has one'],
  ['src <file> <fn>', "one function's source, annotations stripped"],
  ['deps <file>', 'the static call graph'],
  ['canon <file>', 'canonical dependency order (idempotent)'],
  ['hash <file>', 'the content address of each definition: a hash of the definition and a hash of its body'],
  ['diff <a> <b>', 'structural diff: edits, renames, unchanged'],
  ['features [name]', "what Viv can do, by area; with a name or an area, one entry: what it does, when to use it, an example and its output"],
  ['features --search <words>', 'the catalog entries that mention all the words'],
  ['cat <file>', 'print the source'],
  ['ls', 'list the programs loaded in this terminal'],
  ['clear', 'clear the screen'],
  ['help', 'this list'],
];

const esc = (s) =>
  String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

export default function VivTerm({ files = [], liveFiles = {}, fill = false }) {
  const [lines, setLines] = useState([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [ready, setReady] = useState(false);
  const histRef = useRef([]);
  const histPos = useRef(-1);
  const srcCache = useRef(new Map());
  const scrollRef = useRef(null);
  const inputRef = useRef(null);

  const liveNames = useMemo(() => Object.keys(liveFiles), [liveFiles]);
  const allNames = useMemo(
    () => [...liveNames, ...files.map((f) => f.name)].sort(),
    [liveNames, files]
  );

  const emit = useCallback((html, cls = '') => {
    setLines((ls) => [...ls, { html, cls }]);
  }, []);

  useEffect(() => {
    loadWasm().then(
      () => setReady(true),
      (e) => emit(`could not load the compiler: ${esc(e.message || e)}`, 'vt-err')
    );
  }, [emit]);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines, busy]);

  // Resolve a file name to source text: live buffers first, then fetch+cache.
  const readFile = useCallback(
    async (name) => {
      if (liveFiles[name]) return liveFiles[name]();
      if (srcCache.current.has(name)) return srcCache.current.get(name);
      const hit = files.find((f) => f.name === name || f.name.endsWith(`/${name}`));
      if (!hit) return null;
      const res = await fetch(hit.url);
      if (!res.ok) return null;
      const text = await res.text();
      srcCache.current.set(name, text);
      srcCache.current.set(hit.name, text);
      return text;
    },
    [files, liveFiles]
  );

  const run = useCallback(
    async (raw) => {
      const line = raw.trim();
      if (!line) return;
      emit(`<span class="vt-ps1">viv</span> ${esc(line)}`, 'vt-cmd');
      // the site writes commands as the installed binary takes them (`viv run x.viv`);
      // the prompt already says viv, so a leading `viv` is the same command
      const words = line.split(/\s+/);
      if (words[0] === 'viv' && words.length > 1) words.shift();
      const [verb, ...rest] = words;

      if (verb === '--version' || verb === '-V' || verb === 'version') {
        emit(VIV_VERSION ? `viv ${esc(VIV_VERSION)} (viv.wasm, built from the same sources as the binary)` : 'viv.wasm, built from the same sources as the binary', 'vt-dim');
        return;
      }

      if (verb === 'clear') {
        setLines([]);
        return;
      }
      if (verb === 'help' || verb === '?') {
        emit(
          `<table class="vt-help">${VERBS.map(
            ([v, d]) => `<tr><td><b>${esc(v)}</b></td><td>${esc(d)}</td></tr>`
          ).join('')}</table>`
        );
        emit(
          'Each command runs the Viv compiler itself, compiled to WebAssembly: the same code as the installed viv binary. Before writing code, <b>features --search</b> what you need: Viv may already have it.',
          'vt-dim'
        );
        return;
      }
      if (verb === 'ls') {
        if (!allNames.length) {
          emit('no programs loaded', 'vt-dim');
          return;
        }
        emit(
          allNames
            .map(
              (n) =>
                `<span class="vt-file" data-cmd="run ${esc(n)}">${esc(n)}</span>${
                  liveFiles[n] ? ' <span class="vt-dim">(the text in the editor now)</span>' : ''
                }`
            )
            .join('<br>')
        );
        emit('click one to run it', 'vt-dim');
        return;
      }

      // `features`: the catalog (crate::features), the same text `viv features` prints
      if (verb === 'features') {
        const ex = await loadWasm();
        if (!ex.viv_features) {
          emit('this build of the compiler has no feature catalog', 'vt-err');
          return;
        }
        emit(`<pre class="vt-out">${esc(callWasm(ex, ex.viv_features, rest.join(' ')))}</pre>`);
        return;
      }

      const needsFile = new Set([
        'run', 'c', 'wasm', 'doc', 'whatis', 'lsfunc', 'lsvar', 'src', 'deps', 'canon', 'hash', 'cat',
      ]);
      if (!needsFile.has(verb) && verb !== 'diff') {
        emit(`unknown command \`${esc(verb)}\`; type <b>help</b> for the list`, 'vt-err');
        return;
      }

      setBusy(true);
      try {
        const ex = await loadWasm();
        const fileArg = rest[0];
        if (!fileArg) {
          emit(`usage: ${esc(verb)} &lt;file&gt;`, 'vt-err');
          return;
        }

        if (verb === 'diff') {
          const [a, b] = rest;
          if (!a || !b) {
            emit('usage: diff &lt;old.viv&gt; &lt;new.viv&gt;', 'vt-err');
            return;
          }
          const [sa, sb] = await Promise.all([readFile(a), readFile(b)]);
          if (sa === null || sb === null) {
            emit(`no such file: ${esc(sa === null ? a : b)}`, 'vt-err');
            return;
          }
          // viv_diff takes both sources: (old ptr,len, new ptr,len)
          const enc = new TextEncoder();
          const ba = enc.encode(sa);
          const bb = enc.encode(sb);
          const pa = ex.viv_alloc(ba.length);
          new Uint8Array(ex.memory.buffer, pa, ba.length).set(ba);
          const pb = ex.viv_alloc(bb.length);
          new Uint8Array(ex.memory.buffer, pb, bb.length).set(bb);
          const packed = ex.viv_diff(pa, ba.length, pb, bb.length);
          const out = new TextDecoder().decode(
            new Uint8Array(
              ex.memory.buffer,
              Number(packed >> 32n),
              Number(packed & 0xffffffffn)
            )
          );
          emit(`<pre class="vt-out">${esc(out)}</pre>`);
          return;
        }

        const src = await readFile(fileArg);
        if (src === null) {
          emit(
            `no such file: ${esc(fileArg)} — <b>ls</b> lists what is loaded`,
            'vt-err'
          );
          return;
        }

        // Two-string verbs (source + a name) use the 4-arg ABI.
        const twoArg = { whatis: ex.viv_whatis, src: ex.viv_src, lsvar: ex.viv_lsvar };
        if (verb in twoArg) {
          const name = rest[1] ?? '';
          if (verb !== 'lsvar' && !name) {
            emit(`usage: ${esc(verb)} &lt;file&gt; &lt;name&gt;`, 'vt-err');
            return;
          }
          const enc = new TextEncoder();
          const bs = enc.encode(src);
          const bn = enc.encode(name);
          const ps = ex.viv_alloc(bs.length);
          new Uint8Array(ex.memory.buffer, ps, bs.length).set(bs);
          const pn = ex.viv_alloc(bn.length);
          new Uint8Array(ex.memory.buffer, pn, bn.length).set(bn);
          const packed = twoArg[verb](ps, bs.length, pn, bn.length);
          const out = new TextDecoder().decode(
            new Uint8Array(
              ex.memory.buffer,
              Number(packed >> 32n),
              Number(packed & 0xffffffffn)
            )
          );
          emit(`<pre class="vt-out">${esc(out)}</pre>`);
          return;
        }

        if (verb === 'cat') {
          emit(`<pre class="vt-out">${esc(src)}</pre>`);
          return;
        }
        if (verb === 'wasm') {
          // Compile with the direct backend and run the module right here:
          // it imports its runtime from the viv.wasm already loaded, so what
          // it prints is what `run` prints, at compiled speed.
          const mod = compileDirect(ex, src);
          if (mod.error) {
            emit(`<pre class="vt-out vt-warn">${esc(mod.error)}</pre>`);
            emit(
              'the direct WebAssembly backend cannot compile this program; the message above names the feature it lacks',
              'vt-dim'
            );
            return;
          }
          const r = await runDirect(ex, mod.bytes);
          const body = r.error ? `error: ${r.error}` : r.output.replace(/__(graph|trace)__.*$/gm, '').trim();
          const cls = r.error || /FAIL/.test(body) ? 'vt-out vt-warn' : 'vt-out';
          emit(`<pre class="${cls}">${esc(body)}</pre>`);
          emit(
            `compiled to a WebAssembly module of <b>${mod.bytes.length.toLocaleString()}</b> bytes without going through C; ran in ${Math.round(r.seconds * 1000)} ms in your browser`,
            'vt-dim'
          );
          return;
        }

        const fn = {
          run: ex.viv_run_keep || ex.viv_run,
          c: ex.viv_compile,
          doc: ex.viv_doc,
          lsfunc: ex.viv_lsfunc,
          deps: ex.viv_deps,
          canon: ex.viv_canon,
          hash: ex.viv_hash,
        }[verb];
        const t0 = performance.now();
        const out = joinKept(callWasm(ex, fn, src));
        const ms = Math.round(performance.now() - t0);
        const cls = /^error/m.test(out) || /FAIL/.test(out) ? 'vt-out vt-warn' : 'vt-out';
        emit(`<pre class="${cls}">${esc(out.replace(/__(graph|trace)__.*$/gm, '').trim())}</pre>`);
        emit(`${esc(verb)} — ${ms} ms in your browser`, 'vt-dim');
      } catch (e) {
        emit(esc(e.message || String(e)), 'vt-err');
      } finally {
        setBusy(false);
      }
    },
    [allNames, emit, liveFiles, readFile]
  );

  // The file chips and the help table are inert HTML; one delegated listener
  // turns a data-cmd into a command so nothing needs per-node handlers.
  const onClick = useCallback(
    (e) => {
      const t = e.target.closest('[data-cmd]');
      if (t) {
        run(t.getAttribute('data-cmd'));
        return;
      }
      if (!window.getSelection()?.toString()) inputRef.current?.focus();
    },
    [run]
  );

  const onKey = useCallback(
    (e) => {
      if (e.key === 'Enter') {
        const v = input;
        setInput('');
        if (v.trim()) {
          histRef.current = [v, ...histRef.current].slice(0, 100);
          histPos.current = -1;
        }
        run(v);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        const next = Math.min(histPos.current + 1, histRef.current.length - 1);
        if (next >= 0) {
          histPos.current = next;
          setInput(histRef.current[next]);
        }
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        const next = histPos.current - 1;
        histPos.current = next;
        setInput(next >= 0 ? histRef.current[next] : '');
      }
    },
    [input, run]
  );

  return (
    <div className={`vt${fill ? ' vt-fill' : ''}`} onClick={onClick}>
      <div className="vt-bar">
        <span className="vt-title">viv</span>
        <span className="vt-dim">
          the viv command line, compiled to WebAssembly · {allNames.length} program
          {allNames.length === 1 ? '' : 's'} loaded
        </span>
        <span className={`vt-dot${ready ? ' vt-live' : ''}`}>
          {ready ? 'compiler loaded' : 'loading…'}
        </span>
      </div>
      <div className="vt-scroll" ref={scrollRef}>
        {lines.length === 0 && (
          <div className="vt-dim vt-intro">
            The Viv compiler, running in your browser. Try{' '}
            <span className="vt-file" data-cmd="help">help</span>,{' '}
            <span className="vt-file" data-cmd="ls">ls</span>, or{' '}
            <span className="vt-file" data-cmd="hash control.viv">hash control.viv</span> to
            see a hash of each definition.
          </div>
        )}
        {lines.map((l, i) => (
          <div key={i} className={`vt-line ${l.cls}`} dangerouslySetInnerHTML={{ __html: l.html }} />
        ))}
        {busy && <div className="vt-line vt-dim">working…</div>}
        <div className="vt-prompt">
          <span className="vt-ps1">viv</span>
          <input
            ref={inputRef}
            value={input}
            spellCheck={false}
            autoComplete="off"
            aria-label="viv command"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
          />
        </div>
      </div>
    </div>
  );
}
