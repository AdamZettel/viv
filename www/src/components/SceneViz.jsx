// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// One scene's picture: a canvas drawn by a scene module (src/components/
// scenes/), from a program's own records. The module and the build-time
// records load when the scene comes near the screen; the canvas plays only
// while it is on screen and draws one still frame for a reader who prefers
// reduced motion. A program that runs quickly runs again in the reader's tab
// when the scene arrives, and its records replace the build's; a slower one
// keeps the build's records until the reader presses Run.
//
// The canvas sizes itself from its box (ResizeObserver, device pixel ratio),
// and every module draws inside the box it is given, so nothing is cropped
// at any width. The canvas takes no wheel or touch input: the page scrolls.
import { useEffect, useRef, useState } from 'react';
import { runShowcase } from '../lib/showRun.js';

const MODULES = {
  density: () => import('./scenes/density.js'),
  spins: () => import('./scenes/spins.js'),
  waterfall: () => import('./scenes/waterfall.js'),
  wave: () => import('./scenes/wave.js'),
  cartpole: () => import('./scenes/cartpole.js'),
  bands: () => import('./scenes/bands.js'),
  glowplot: () => import('./scenes/glowplot.js'),
};

/** Plain text with T_c set as T with a subscript c. */
export function typeset(text) {
  return String(text)
    .split(/(T_c)/)
    .map((part, i) => (part === 'T_c' ? (
      <span key={i}>
        <i>T</i>
        <sub>c</sub>
      </span>
    ) : part));
}

const reducedMotion = () => typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/**
 * kind     the scene module
 * slug     the staged records, public/programs/show/<slug>.fig.txt
 * data     records passed in directly (the band structure's)
 * live     'auto' or 'click': a Run button runs the program again in the reader's
 *          browser (never on arrival: viv.wasm loads only when asked for); null: no button
 * fig      for glowplot: the figure's title, or its start
 * text     records to draw instead of fetching (a /show page passes the reader's run)
 * readout  a ref to the element the readout line goes in
 */
export default function SceneViz({ kind, slug = null, data = null, live = null, fig = null, text = null, readout = null, aspect = '4 / 3', label = '' }) {
  const box = useRef(null);
  const canvas = useRef(null);
  const scene = useRef(null);
  const [near, setNear] = useState(false);
  const [state, setState] = useState('idle'); // idle | running | done | error
  const [error, setError] = useState('');

  // load when near the screen
  useEffect(() => {
    const el = box.current;
    if (!el) return undefined;
    if (typeof IntersectionObserver === 'undefined') {
      setNear(true);
      return undefined;
    }
    const io = new IntersectionObserver((es) => es.some((e) => e.isIntersecting) && setNear(true), { rootMargin: '600px 0px' });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    if (!near) return undefined;
    let dead = false;
    let io = null;
    let ro = null;
    let mo = null;
    const setReadout = (s) => {
      if (readout && readout.current && readout.current.textContent !== s) readout.current.textContent = s;
    };
    (async () => {
      const [mod, kit] = await Promise.all([MODULES[kind](), import('./scenes/kit.js')]);
      let src = text;
      if (src == null && slug) src = await fetch(`/programs/show/${slug}.fig.txt`).then((r) => (r.ok ? r.text() : '')).catch(() => '');
      if (dead || !canvas.current) return;
      const reduced = reducedMotion();
      const s = mod.default(canvas.current, { text: src || '', data, fig, reduced, readout: setReadout });
      scene.current = s;
      // records missing or stale: hold a drawn frame, never a blank canvas,
      // and run the program for its own records
      const hold = () => !s.ok() && kit.placeholder(canvas.current, label);
      hold();
      s.redraw();
      if (!s.ok() && slug && text == null) run(true).then(() => !dead && (hold(), s.redraw()));
      io = new IntersectionObserver((es) => {
        const on = es.some((e) => e.isIntersecting);
        if (on && !document.hidden && s.ok()) s.start();
        else s.stop();
      }, { threshold: 0.15 });
      io.observe(box.current);
      ro = new ResizeObserver(() => (s.ok() ? s.redraw() : hold()));
      ro.observe(box.current);
      // the site's light/dark switch repaints a paused scene too
      mo = new MutationObserver(() => (s.ok() ? s.redraw() : hold()));
      mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    })();
    const vis = () => document.hidden && scene.current && scene.current.stop();
    document.addEventListener('visibilitychange', vis);
    return () => {
      dead = true;
      if (io) io.disconnect();
      if (ro) ro.disconnect();
      if (mo) mo.disconnect();
      document.removeEventListener('visibilitychange', vis);
      if (scene.current) scene.current.stop();
      scene.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [near, kind, slug]);

  // a /show page hands over the reader's own run
  useEffect(() => {
    if (text != null && scene.current) scene.current.setText(text);
  }, [text]);

  async function run(quiet = false) {
    const prog = slug;
    if (!prog) return;
    if (!quiet) setState('running');
    try {
      const r = await runShowcase(prog);
      if (scene.current) {
        scene.current.setText(r.out);
        if (scene.current.ok() && box.current && box.current.getBoundingClientRect().bottom > 0) scene.current.start();
      }
      if (!quiet) setState('done');
    } catch (e) {
      if (!quiet) {
        setState('error');
        setError(String(e.message || e));
      }
    }
  }

  return (
    <div className="scn-viz" ref={box} style={{ aspectRatio: aspect }}>
      <canvas ref={canvas} className="scn-canvas" role="img" aria-label={label} />
      {live && text == null && (
        <button type="button" className="scn-run" onClick={() => run(false)} disabled={state === 'running'}>
          {state === 'running' ? 'Running' : state === 'done' ? 'Run again' : 'Run'}
        </button>
      )}
      {state === 'error' && <p className="scn-err">The run stopped: {error}</p>}
    </div>
  );
}
