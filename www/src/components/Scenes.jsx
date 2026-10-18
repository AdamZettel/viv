// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The home page's scenes: full-width dark sections in the page's own scroll,
// one per program (src/data/scenes.mjs), text and picture alternating sides.
// Nothing here scrolls on its own, snaps, or takes the wheel: the page
// scrolls as the browser scrolls it, and each scene fades in as it arrives.
import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import SceneViz, { typeset } from './SceneViz.jsx';

const ZoomWalk = lazy(() => import('./ZoomWalk.jsx'));

// a scene's text reveals once its top is above this fraction of the window
const REVEAL_AT = 0.9;

function useReveal(ref) {
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setShown(true);
      return undefined;
    }
    let ticking = false;
    const check = () => {
      ticking = false;
      if (el.getBoundingClientRect().top < window.innerHeight * REVEAL_AT) {
        setShown(true);
        window.removeEventListener('scroll', on);
        window.removeEventListener('resize', on);
      }
    };
    const on = () => {
      if (!ticking) {
        ticking = true;
        requestAnimationFrame(check);
      }
    };
    window.addEventListener('scroll', on, { passive: true });
    window.addEventListener('resize', on);
    check();
    return () => {
      window.removeEventListener('scroll', on);
      window.removeEventListener('resize', on);
    };
  }, [ref]);
  return shown;
}

function useOnScreen(ref) {
  const [state, setState] = useState({ seen: false, on: false });
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === 'undefined') {
      setState({ seen: true, on: true });
      return undefined;
    }
    const io = new IntersectionObserver((es) => {
      const on = es.some((e) => e.isIntersecting);
      setState((s) => ({ seen: s.seen || on, on }));
    }, { rootMargin: '300px 0px' });
    io.observe(el);
    return () => io.disconnect();
  }, [ref]);
  return state;
}

// A walk starts when the reader presses Run, not when it scrolls into view: it
// fetches the program's bundle and viv.wasm and records the run, which a
// reader passing by should not have to download.
function Walk({ scene }) {
  const ref = useRef(null);
  const { on } = useOnScreen(ref);
  const [started, setStarted] = useState(false);
  return (
    // before Run the card is only as tall as its sentence and button: the walk takes its full height when it starts
    <div className={`scn-walk${started ? '' : ' is-idle'}`} ref={ref}>
      {started ? (
        <Suspense fallback={<p className="scn-hold">Loading the walk</p>}>
          <ZoomWalk slug={scene.slug} title={scene.title} card active={on} />
        </Suspense>
      ) : (
        <div className="scn-walk-start">
          <p className="scn-hold">Run records the program in your browser, then the walk follows the recording call by call.</p>
          <button type="button" className="scn-run scn-walk-run" onClick={() => setStarted(true)}>
            Run
          </button>
        </div>
      )}
    </div>
  );
}

function Scene({ scene, i, extra }) {
  const ref = useRef(null);
  const readout = useRef(null);
  const shown = useReveal(ref);
  const flip = i % 2 === 1;
  const walk = scene.kind === 'walk';
  return (
    <section className={`scn${flip ? ' scn-flip' : ''}${walk ? ' scn-is-walk' : ''}${shown ? ' is-in' : ''}`} ref={ref} aria-labelledby={`scn-${scene.id}`}>
      <div className="scn-in">
        <div className="scn-text">
          <h2 className="scn-title" id={`scn-${scene.id}`}>
            {scene.title}
          </h2>
          <p className="scn-line">{typeset(scene.line)}</p>
          {!walk && <p className="scn-readout" ref={readout} aria-live="off" />}
          <p className="scn-ref">{typeset(scene.check || `Checked against ${scene.ref}`)}</p>
          <a className="scn-link" href={scene.href}>
            {scene.cta} →
          </a>
          {scene.more && (
            <a className="scn-link scn-link-more" href={scene.more.href}>
              {scene.more.cta} →
            </a>
          )}
        </div>
        {walk ? (
          <Walk scene={scene} />
        ) : (
          <SceneViz kind={scene.kind} slug={scene.slug} data={extra[scene.id] || null} live={scene.live} fig={scene.fig} readout={readout} label={scene.title} />
        )}
      </div>
    </section>
  );
}

export default function Scenes({ scenes, extra = {} }) {
  return (
    <div className="scn-list">
      {scenes.map((s, i) => (
        <Scene key={s.id} scene={s} i={i} extra={extra} />
      ))}
    </div>
  );
}
