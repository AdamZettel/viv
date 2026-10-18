// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The walk through a run (/explore/walk/<slug>): a full-window app drawn like
// the vipyr atlas (components/CodeMap.jsx). The run's call graph is laid out
// by the atlas's own code (lib/codemap.js: a region per module, a tile per
// def, the calls routed around them) and styled by its stylesheet
// (styles/codemap.css). On each stop the camera flies to the def that runs
// (van Wijk and Nuij's path) and the calls open at that moment are drawn as
// one path; the panel on the right shows that def, its calls and its code
// with the running line marked. A thin timeline along the bottom scrubs the
// stops.
//
// The recording is the explorer's (public/explore-worker.js, viv.wasm), with
// every def traced so each call is an event (src/lib/zoomwalk.js, traceSource).
// Moving never runs anything again.
import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { highlight } from '../lib/highlight.js';
import { renderProse, loadKatex, texToPlain } from '../lib/richText.js';
import { parseRecording, indexBundle, buildTree, secondPassIters, secondPassCalls, wholeRunRequest, defLabel, numText } from '../lib/explore.js';
import {
  traceSource, hotDefs, TRACE_LIMIT, layoutWorld, buildFrames, interpolateZoom, ease, flightMs, dwellMs,
  nextOver, nextOut, codeFiles, codeTarget, callSite, defAt, walkMap, builtinsOf,
} from '../lib/zoomwalk.js';
import { intentOf, layoutMap, makeRouter, orthoPath, trunks } from '../lib/codemap.js';

import { REPO_PUBLIC, GITHUB } from '../lib/repo.js';
import { vivWorker } from '../lib/vivWasm.js';
const GH = `${GITHUB}/blob/main/`;

function record(worker, src, req) {
  return new Promise((resolve, reject) => {
    const id = Math.random().toString(36).slice(2);
    const on = (e) => {
      if (e.data.id !== id) return;
      worker.removeEventListener('message', on);
      if (e.data.error) reject(new Error(e.data.error));
      else resolve(e.data);
    };
    worker.addEventListener('message', on);
    worker.postMessage({ id, src, ...req });
  });
}

function useReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReduced(mq.matches);
    const on = (e) => setReduced(e.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, []);
  return reduced;
}


// `load`, in place of `slug`, makes the bundle in the browser (lib/atlas.js,
// /atlas/walk): called with a function that hears what is happening, it
// resolves to `{ bundle, ran, note }`. `back` is the page the bar links back to.
export default function ZoomWalk({ slug, title, card = false, active = true, load = null, back = null }) {
  const home = back || `/explore/${slug}`;
  const [phase, setPhase] = useState({ what: 'loading', detail: 'Loading the program' });
  const [ix, setIx] = useState(null);
  const [run, setRun] = useState(null);
  const [katex, setKatex] = useState(null);

  useEffect(() => {
    let dead = false;
    loadKatex().then((k) => !dead && setKatex(k));
    return () => {
      dead = true;
    };
  }, []);

  useEffect(() => {
    let dead = false;
    let worker = null;
    (async () => {
      try {
        let bundle;
        if (load) {
          const made = await load((detail) => !dead && setPhase({ what: 'loading', detail }));
          if (!made.ran) throw new Error(`there is no walk without a run. ${made.note}`);
          bundle = made.bundle;
        } else {
          const res = await fetch(`/explore/${slug}.json`);
          if (!res.ok) throw new Error(`no bundle for ${slug} (${res.status})`);
          bundle = await res.json();
        }
        const ix = indexBundle(bundle);
        if (dead) return;
        setIx(ix);
        worker = vivWorker('/explore-worker.js');
        const defs = bundle.lineDefs || [];
        const t0 = performance.now();
        setPhase({ what: 'recording', detail: 'Recording the run with every call traced' });
        let traced = true;
        let note = null;
        let hot = new Set();
        let rec;
        let r1 = await record(worker, traceSource(bundle), { defs }).catch((e) => ({ out: `error: ${e.message}` }));
        rec = parseRecording(r1.out);
        if (!rec.error && rec.total > TRACE_LIMIT) {
          // too many calls to keep: leave the hot defs without steps untraced
          hot = hotDefs(rec.events, ix);
          setPhase({ what: 'recording', detail: `Recording again without tracing ${hot.size} defs called thousands of times` });
          r1 = await record(worker, traceSource(bundle, hot), { defs });
          rec = parseRecording(r1.out);
          note = `Calls of ${[...hot].join(', ')} are not traced: the run made too many to keep.`;
        }
        if (rec.error) {
          // the walk still works from the explorer's own recording, with calls only of defs with steps
          traced = false;
          note = `This run could not be traced call by call (${rec.error}); the walk shows calls of the defs with steps only.`;
          r1 = await record(worker, bundle.program, { defs });
          rec = parseRecording(r1.out);
          if (rec.error) throw new Error(rec.error);
        }
        let tree = buildTree(rec.events, ix);
        const whole = wholeRunRequest(tree, rec.total);
        const iters = whole ? whole.iters : secondPassIters(tree);
        const calls = whole ? whole.calls : secondPassCalls(tree);
        const every = whole ? whole.every : [];
        if (Object.keys(iters).length || Object.keys(calls).length || every.length) {
          if (dead) return;
          setPhase({ what: 'recording', detail: 'Recording again, line by line in more passes and calls' });
          const r2 = await record(worker, traced ? traceSource(bundle, hot) : bundle.program, { defs, iters, calls, every });
          const rec2 = parseRecording(r2.out);
          if (!rec2.error) {
            rec = rec2;
            tree = buildTree(rec.events, ix);
          }
        }
        if (dead) return;
        const world = layoutWorld(ix, tree);
        setRun({ rec, tree, world, note, seconds: (performance.now() - t0) / 1000 });
        setPhase({ what: 'ready' });
      } catch (err) {
        if (!dead) setPhase({ what: 'error', detail: String(err.message || err) });
      } finally {
        if (worker) worker.terminate();
      }
    })();
    return () => {
      dead = true;
      if (worker) worker.terminate();
    };
  }, [slug, load]);


  if (phase.what !== 'ready' && card) {
    return (
      <div className="zw-card zw-card-loading" role="status" aria-live="polite">
        {phase.what === 'error' ? <p className="ex-error">The run could not be recorded: {phase.detail}</p> : <p>{phase.detail}</p>}
      </div>
    );
  }
  if (phase.what !== 'ready') {
    return (
      <div className="cm zw">
        <div className="cm-bar zw-bar">
          <a className="cm-back-ref zw-back" href={home}>
            ← <span>{title}</span>
          </a>
        </div>
        <div className="cm-empty zw-loading" role="status" aria-live="polite">
          {phase.what === 'error' ? (
            <p className="ex-error">The run could not be recorded: {phase.detail}</p>
          ) : (
            <>
              <div className="ex-spinner" aria-hidden="true" />
              <p>{phase.detail}</p>
              <p className="zw-loading-what">
                Your browser runs the program once and records every call. Then the whole program
                appears as a map, files as regions and defs as boxes, and a camera follows the
                recording from call to call.
              </p>
            </>
          )}
        </div>
      </div>
    );
  }
  return <Walk ix={ix} run={run} katex={katex} home={home} title={title} card={card} active={active} />;
}

// ---------------------------------------------------------------------------

// the zoom at which the camera reads a def while following (a narrow screen
// reads closer to the whole); the zoom levels below which the tiles drop their
// intent line, their name, and the regions their titles
const K_READ = 1.05;
const K_READ_PHONE = 0.85;
const K_SUB = 0.45;
const K_NAME = 0.32;
const K_REGION = 0.6;
const zoomLevel = (k) => (k < K_NAME ? 0 : k < K_SUB ? 1 : k < K_REGION ? 2 : 3);
// how many of the defs the run left last are drawn as a fading trail
const TRAIL = 3;
const SPEEDS = [0.5, 1, 2, 4];
// a card on the home page rests this long on the last stop before it starts again
const CARD_REST_MS = 2500;

const nameOf = (id) => (id === '<main>' ? 'top level' : id);

function stopFromHash() {
  const m = /^#stop-(\d+)$/.exec(typeof location === 'undefined' ? '' : location.hash);
  return m ? Math.max(0, Number(m[1]) - 1) : 0;
}

/**
 * The walk itself, given a recording: `run` is { rec, tree, world, note,
 * seconds }. On the site `home` is the program's page; the page `viv atlas`
 * writes (src/atlas/main.jsx) hands it the recording the compiler made
 * (lang/viv/src/atlas_walk.rs), with `back` null (no program page to go back
 * to) and `recorded` saying where it was made.
 */
export function Walk({ ix, run, katex, home, slug, title, card = false, active = true, back, recorded = 'Recorded' }) {
  const backHref = back === undefined ? (home ?? `/explore/${slug}`) : back;
  const { world, tree, rec } = run;
  const map = useMemo(() => walkMap(ix, world), [ix, world]);
  const lay = useMemo(() => layoutMap(map), [map]);
  const router = useMemo(() => makeRouter(lay), [lay]);
  const trunkList = useMemo(() => trunks(lay, router), [lay, router]);
  const defs = useMemo(() => new Map(map.defs.map((d) => [d.id, d])), [map]);

  const [loops, setLoops] = useState('one');
  const { frames, truncated, via } = useMemo(() => buildFrames(tree, ix, { loops }), [tree, ix, loops]);
  const [i, setI] = useState(() => (card ? 0 : Math.min(stopFromHash(), frames.length - 1)));
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [follow, setFollow] = useState(true);
  const [whole, setWhole] = useState(false);
  // what the reader picked to look at apart from the run: a def (from the
  // graph, the panel or a call in the code) or a call on the stack; the next
  // move in the run clears it
  const [focus, setFocus] = useState(null);
  const [hover, setHover] = useState(null);
  const [query, setQuery] = useState('');
  const [legend, setLegend] = useState(false);
  const [lvl, setLvl] = useState(3);
  const reduced = useReducedMotion();
  const frame = frames[Math.min(i, frames.length - 1)];

  const rootRef = useRef(null);
  const svgRef = useRef(null);
  const camG = useRef(null);
  const searchRef = useRef(null);
  const size = useRef({ vw: 800, vh: 520 });
  const cam = useRef(null); // [cx, cy, w]: the centre and the view's width in graph units
  const anim = useRef(null);
  const lastFlight = useRef(0);
  const prevI = useRef(i);
  const viewSub = useRef(null);

  const apply = useCallback(() => {
    const c = cam.current;
    if (!c || !camG.current) return;
    const { vw, vh } = size.current;
    const k = vw / c[2];
    const v = { x: vw / 2 - c[0] * k, y: vh / 2 - c[1] * k, k, vw, vh };
    camG.current.setAttribute('transform', `translate(${v.x} ${v.y}) scale(${k})`);
    setLvl(zoomLevel(k));
    if (viewSub.current) viewSub.current(v);
  }, []);

  const flyTo = useCallback(
    (target, animate) => {
      if (anim.current) cancelAnimationFrame(anim.current);
      anim.current = null;
      if (!cam.current || !animate) {
        cam.current = target;
        lastFlight.current = 0;
        apply();
        return;
      }
      const path = interpolateZoom(cam.current, target);
      const ms = flightMs(path.S) / speed;
      lastFlight.current = ms;
      const t0 = performance.now();
      const tick = (now) => {
        const t = Math.min(1, (now - t0) / ms);
        cam.current = path(ease(t));
        apply();
        anim.current = t < 1 ? requestAnimationFrame(tick) : null;
      };
      anim.current = requestAnimationFrame(tick);
    },
    [apply, speed],
  );

  // the views: the whole graph, and a def at reading size (kept where it is
  // when it is already in view at about that size)
  const wholeView = useCallback(() => {
    const { vw, vh } = size.current;
    const k = Math.min(1.2, Math.max(0.1, Math.min((vw - 76) / lay.width, (vh - 40) / lay.height)));
    return [lay.width / 2, lay.height / 2, vw / k];
  }, [lay]);
  const defView = useCallback(
    (id, keep) => {
      const t = lay.tiles.get(id);
      const { vw, vh } = size.current;
      const kRead = vw < 480 ? K_READ_PHONE : K_READ;
      const w0 = wholeView();
      if (!t || vw / w0[2] >= kRead * 0.9) return w0;
      const c = cam.current;
      if (keep && c) {
        const k = vw / c[2];
        const sx = vw / 2 + (t.x + t.w / 2 - c[0]) * k;
        const sy = vh / 2 + (t.y + t.h / 2 - c[1]) * k;
        if (k >= kRead * 0.7 && k <= kRead * 1.6 && sx > vw * 0.15 && sx < vw * 0.85 && sy > vh * 0.15 && sy < vh * 0.85) return c;
      }
      return [t.x + t.w / 2, t.y + t.h / 2, vw / kRead];
    },
    [lay, wholeView],
  );
  const viewFor = useCallback(
    (f, keep) => {
      if (f.kind === 'start' || f.kind === 'end') {
        const w0 = wholeView();
        return size.current.vw / w0[2] >= 0.5 ? w0 : defView('<main>', keep);
      }
      return defView(f.def, keep);
    },
    [wholeView, defView],
  );

  // the graph's size
  useLayoutEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      size.current = { vw: Math.max(200, r.width), vh: Math.max(160, r.height) };
      if (!cam.current) cam.current = viewFor(frame, false);
      apply();
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [apply, viewFor]); // eslint-disable-line react-hooks/exhaustive-deps

  // the camera follows the run, or goes to what the reader picked
  const focusDef = focus ? focus.def : null;
  useEffect(() => {
    const step = Math.abs(i - prevI.current) === 1;
    prevI.current = i;
    if (focusDef) {
      flyTo(defView(focusDef, true), !reduced);
      return;
    }
    if (!follow || !frame) {
      lastFlight.current = 0;
      return;
    }
    flyTo(viewFor(frame, true), (step || playing) && !reduced);
  }, [i, frame, follow, focusDef, flyTo, viewFor, defView, reduced]); // eslint-disable-line react-hooks/exhaustive-deps

  // playing: rest on the stop, then move on
  useEffect(() => {
    if (!playing) return;
    if (i >= frames.length - 1) {
      if (card) {
        const t = setTimeout(() => setI(0), CARD_REST_MS);
        return () => clearTimeout(t);
      }
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => setI((x) => Math.min(x + 1, frames.length - 1)), lastFlight.current + dwellMs(frame) / speed);
    return () => clearTimeout(t);
  }, [playing, i, frames.length, frame, speed]);

  useEffect(() => () => anim.current && cancelAnimationFrame(anim.current), []);

  // a card plays while it is on screen and pauses off it; a reader who
  // prefers reduced motion starts it with the button
  useEffect(() => {
    if (!card) return;
    setPlaying(active && !reduced);
  }, [card, active, reduced]);

  // the stop in the address, so a link opens the walk where it was
  useEffect(() => {
    if (playing || card) return;
    try {
      history.replaceState(null, '', i > 0 ? `#stop-${i + 1}` : location.pathname + location.search);
    } catch {
      /* a sandboxed frame */
    }
  }, [i, playing]);

  // a new loop setting keeps the reader near where they were
  const setLoopMode = (m) => {
    const node = frame.node;
    setLoops(m);
    const next = buildFrames(tree, ix, { loops: m }).frames;
    let k = next.findIndex((f) => f.node === node && f.kind === frame.kind);
    if (k < 0) k = next.findIndex((f) => f.node && f.node.i >= node.i);
    prevI.current = -10;
    setFocus(null);
    setI(Math.max(0, k));
  };

  const go = useCallback(
    (how) => {
      setFocus(null);
      setWhole(false);
      setI((x) => {
        const n = frames.length - 1;
        if (how === 'into') return Math.min(n, x + 1);
        if (how === 'back') return Math.max(0, x - 1);
        if (how === 'over') return nextOver(frames, x);
        if (how === 'out') return nextOut(frames, x);
        if (how === 'first') return 0;
        if (how === 'last') return n;
        return x;
      });
    },
    [frames],
  );
  const togglePlay = useCallback(() => {
    setFocus(null);
    setWhole(false);
    setFollow(true);
    setPlaying((p) => {
      if (!p && i >= frames.length - 1) setI(0);
      return !p;
    });
  }, [i, frames.length]);

  const overview = () => {
    setFollow(false);
    setFocus(null);
    setWhole(true);
    flyTo(wholeView(), !reduced);
  };
  const toggleFollow = () => {
    setFocus(null);
    setWhole(false);
    setFollow((f) => (focus ? true : !f));
  };

  // keys: the arrows move by stop, space plays, f follows, w the whole graph, s or / find
  useEffect(() => {
    if (card) return undefined;
    const onKey = (e) => {
      const t = e.target;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) {
        if (e.key === 'Escape') t.blur();
        return;
      }
      const moves = { ArrowRight: 'into', ArrowLeft: 'back', ArrowDown: 'over', ArrowUp: 'out', Home: 'first', End: 'last' };
      if (e.key === ' ') {
        e.preventDefault();
        togglePlay();
      } else if (moves[e.key]) {
        e.preventDefault();
        setPlaying(false);
        go(moves[e.key]);
      } else if (e.key === 'f') toggleFollow();
      else if (e.key === 'w') overview();
      else if (e.key === 's' || e.key === '/') {
        e.preventDefault();
        searchRef.current?.focus();
      } else if (e.key === 'Escape') setLegend(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }); // eslint-disable-line react-hooks/exhaustive-deps

  // the reader picks a def: the graph goes to it and the panel shows it
  const showDef = useCallback(
    (def) => {
      if (!defs.has(def)) return;
      setPlaying(false);
      setWhole(false);
      setFocus({ def, line: null, why: 'def' });
    },
    [defs],
  );

  // pan and zoom: a drag, the wheel, a pinch; any of them stops following.
  // A press on a tile that does not move is a click on it.
  useEffect(() => {
    const el = svgRef.current;
    if (!el || card) return;
    const pts = new Map();
    let pinch = null;
    let press = null;
    const worldAt = (x, y) => {
      const c = cam.current;
      const { vw, vh } = size.current;
      const k = vw / c[2];
      return [c[0] + (x - vw / 2) / k, c[1] + (y - vh / 2) / k];
    };
    const zoomAt = (x, y, f) => {
      const c = cam.current;
      const { vw, vh } = size.current;
      const [wx, wy] = worldAt(x, y);
      const w = Math.max(vw / 3, Math.min(vw / 0.1, c[2] * f));
      const k = vw / w;
      cam.current = [wx - (x - vw / 2) / k, wy - (y - vh / 2) / k, w];
      apply();
    };
    const local = (e) => {
      const r = el.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    };
    const stopFollowing = () => {
      if (anim.current) cancelAnimationFrame(anim.current);
      anim.current = null;
      setFollow(false);
      setWhole(false);
    };
    const onWheel = (e) => {
      e.preventDefault();
      stopFollowing();
      const [x, y] = local(e);
      zoomAt(x, y, Math.exp(e.deltaY * (e.deltaMode === 1 ? 0.05 : e.ctrlKey ? 0.01 : 0.0015)));
    };
    const onDown = (e) => {
      if (e.button !== 0 && e.pointerType === 'mouse') return;
      el.setPointerCapture(e.pointerId);
      const p = local(e);
      pts.set(e.pointerId, p);
      const box = e.target.closest && e.target.closest('[data-def]');
      press = pts.size === 1 ? { p, def: box ? box.getAttribute('data-def') : null, moved: false } : null;
      if (pts.size === 2) {
        const [a, b] = [...pts.values()];
        pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]) };
      }
    };
    const onMove = (e) => {
      if (!pts.has(e.pointerId)) return;
      const p = local(e);
      const q = pts.get(e.pointerId);
      pts.set(e.pointerId, p);
      if (pts.size === 2 && pinch) {
        const [a, b] = [...pts.values()];
        const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
        if (d > 0 && pinch.d > 0) zoomAt((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, pinch.d / d);
        pinch.d = d;
        stopFollowing();
        return;
      }
      if (press && Math.hypot(p[0] - press.p[0], p[1] - press.p[1]) < 4) return;
      if (press) press.moved = true;
      stopFollowing();
      const c = cam.current;
      const k = size.current.vw / c[2];
      cam.current = [c[0] - (p[0] - q[0]) / k, c[1] - (p[1] - q[1]) / k, c[2]];
      apply();
    };
    const onUp = (e) => {
      pts.delete(e.pointerId);
      if (pts.size < 2) pinch = null;
      if (press && !press.moved && press.def && e.type === 'pointerup') showDef(press.def);
      press = null;
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    el.addEventListener('pointerdown', onDown);
    el.addEventListener('pointermove', onMove);
    el.addEventListener('pointerup', onUp);
    el.addEventListener('pointercancel', onUp);
    return () => {
      el.removeEventListener('wheel', onWheel);
      el.removeEventListener('pointerdown', onDown);
      el.removeEventListener('pointermove', onMove);
      el.removeEventListener('pointerup', onUp);
      el.removeEventListener('pointercancel', onUp);
    };
  }, [apply, showDef]);

  const zoomBy = (f) => {
    const c = cam.current;
    if (!c) return;
    setFollow(false);
    setWhole(false);
    flyTo([c[0], c[1], c[2] / f], !reduced);
  };

  // the call stack: picking a call shows where it is waiting, without moving the run
  const pickFrame = (k) => {
    const st = frame.stack;
    setPlaying(false);
    if (k >= st.length - 1) {
      setFocus(null);
      return;
    }
    const v = via.get(st[k + 1].key);
    setFocus({ def: st[k].def, line: callSite(ix, st[k].def, st[k + 1].def, v ? v.line : null), why: 'stack', k });
  };

  // what the graph marks: the def that runs, the calls open now, the call
  // just returned from, the defs the run left last
  const cur = frame.kind === 'start' || frame.kind === 'end' ? '<main>' : frame.def;
  const stackIds = frame.stack.map((c) => c.def);
  const trail = useMemo(() => {
    const out = [];
    const open = new Set(stackIds);
    for (let j = i - 1; j >= 0 && j > i - 400 && out.length < TRAIL; j--) {
      const d = frames[j].kind === 'start' || frames[j].kind === 'end' ? '<main>' : frames[j].def;
      if (!open.has(d) && !out.includes(d)) out.push(d);
    }
    return out;
  }, [i, frames]); // eslint-disable-line react-hooks/exhaustive-deps
  const returned = frame.kind === 'ret' ? [frame.back.caller, frame.from] : null;
  const hits = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    // a def named exactly what was typed first, then names that start with
    // it, so Enter on `coulomb` opens coulomb and not proton_coulomb
    const rank = (d) => (nameOf(d.id).toLowerCase() === q ? 0 : nameOf(d.id).toLowerCase().startsWith(q) ? 1 : 2);
    return map.defs
      .filter((d) => nameOf(d.id).toLowerCase().includes(q) || map.modules[d.module].path.toLowerCase().includes(q))
      .map((d, i) => [rank(d), i, d])
      .sort((a, b) => a[0] - b[0] || a[1] - b[1])
      .map((x) => x[2])
      .slice(0, 12);
  }, [query, map]);

  const code = focus ? { def: focus.def, line: focus.line, mark: focus.why === 'stack' ? 'is-picked' : null } : { ...codeTarget(frame, ix), mark: 'is-run' };
  const nFiles = map.modules.length;
  const nDefs = map.defs.filter((d) => !d.main).length;

  if (card) {
    return (
      <div className="zw-card" ref={rootRef}>
        <div className="zw-card-bar">
          <button type="button" className="zw-playbtn zw-stepbtn" onClick={() => { setPlaying(false); go('back'); }} aria-label="A stop back" title="A stop back">
            ‹
          </button>
          <button type="button" className="zw-playbtn" onClick={() => setPlaying((p) => !p)} aria-pressed={playing}>
            {playing ? 'Pause' : 'Play'}
          </button>
          <button type="button" className="zw-playbtn zw-stepbtn" onClick={() => { setPlaying(false); go('into'); }} aria-label="A stop forward" title="A stop forward">
            ›
          </button>
          <span className="zw-card-stop" title="A stop is one place the walk pauses: a call, a return, a step or a line">
            Stop {(i + 1).toLocaleString('en-US')} of {frames.length.toLocaleString('en-US')}
          </span>
          <nav className="zw-card-route" aria-label="The calls open at this stop">
            {frame.stack.map((c, k) => (
              <span key={c.key} className={k === frame.stack.length - 1 ? 'is-cur' : ''}>
                {k > 0 && <span className="cm-sep">›</span>}
                {c.parent ? defLabel(ix, c.def) : 'top level'}
              </span>
            ))}
          </nav>
        </div>
        <div className="zw-card-now">
          <Now ix={ix} frame={frame} katex={katex} playing={playing} />
        </div>
        <div className="zw-card-body">
          <div className="cm-stage zw-stage zw-card-stage">
            <svg ref={svgRef} className="cm-svg zw-svg" role="img" aria-label={`The call graph of the run, now at ${frameLabel(frame, ix)}`}>
              <defs>
                {['out', 'in', 'route', 'ret'].map((k) => (
                  <marker key={k} id={`zw-arrow-${k}`} className={`cm-arrow cm-arrow-${k}`} viewBox="0 0 8 8" refX="7.5" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto">
                    <path d="M0.5,0.8 L7.5,4 L0.5,7.2 Z" />
                  </marker>
                ))}
              </defs>
              <g ref={camG}>
                <Canvas lay={lay} router={router} trunkList={trunkList} map={map} defs={defs} cur={cur} sel={cur} hover={null}
                  stackKey={stackIds.join('\u0000')} trail={trail.join('\u0000')} returned={returned ? returned.join('\u0000') : ''}
                  hits="" lvl={lvl} setHover={() => {}} />
              </g>
            </svg>
          </div>
          <Pane ix={ix} map={map} defs={defs} target={code} frame={null} katex={katex} reduced={reduced} playing={playing} onDef={() => {}} compact />
        </div>
      </div>
    );
  }

  return (
    <div className="cm zw" ref={rootRef}>
      <div className="cm-bar zw-bar">
        {backHref ? (
          <a className="cm-back-ref zw-back" href={backHref} title="The program's page">
            ← <span>{title}</span>
          </a>
        ) : null}
        <span className="cm-count">
          {nDefs} defs in {nFiles} {nFiles === 1 ? 'file' : 'files'}
        </span>
        <div className="cm-nav zw-play" role="group" aria-label="Playback">
          <button type="button" onClick={() => { setPlaying(false); go('back'); }} disabled={i === 0} aria-label="Previous stop" title="Previous stop (←)">
            ‹
          </button>
          <button type="button" className="zw-playbtn" onClick={togglePlay} aria-pressed={playing} title="Play or pause (space)">
            {playing ? 'Pause' : i >= frames.length - 1 ? 'Replay' : 'Play'}
          </button>
          <button type="button" onClick={() => { setPlaying(false); go('into'); }} disabled={i >= frames.length - 1} aria-label="Next stop" title="Next stop (→)">
            ›
          </button>
          <select className="zw-speed" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="Speed">
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {s}×
              </option>
            ))}
          </select>
        </div>
        <div className="cm-nav zw-cam" role="group" aria-label="Camera">
          <button type="button" className="cm-all" onClick={overview} aria-pressed={whole} title="The whole graph (w)">
            Whole graph
          </button>
          <button type="button" className="cm-all" onClick={toggleFollow} aria-pressed={follow && !focus && !whole} title="Follow the run (f)">
            Following
          </button>
        </div>
        <div className="cm-search">
          <input
            ref={searchRef}
            type="search"
            placeholder="Find a def (s)"
            value={query}
            aria-label="Find a def"
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && hits[0]) {
                showDef(hits[0].id);
                setQuery('');
                e.currentTarget.blur();
              }
            }}
          />
          {hits.length > 0 && (
            <ul className="cm-hits" role="listbox">
              {hits.map((d) => (
                <li key={d.id}>
                  <button
                    type="button"
                    onClick={() => {
                      showDef(d.id);
                      setQuery('');
                    }}>
                    <code>{nameOf(d.id)}</code>
                    <span>{map.modules[d.module].path}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="cm-nav">
          <button type="button" onClick={() => setLegend((x) => !x)} aria-expanded={legend}>
            Key
          </button>
        </div>
      </div>
      <nav className="cm-route" aria-label="The calls open at this stop">
        {frame.stack.map((c, k) => (
          <span key={c.key}>
            {k > 0 && <span className="cm-sep">›</span>}
            <button
              type="button"
              className={`${k === frame.stack.length - 1 ? 'is-cur' : ''}${focus && focus.why === 'stack' && focus.k === k ? ' is-picked' : ''}`}
              onClick={() => pickFrame(k)}
              title={k === frame.stack.length - 1 ? 'Where the run is' : 'Where this call is waiting'}>
              {c.parent ? defLabel(ix, c.def) : 'top level'}
            </button>
          </span>
        ))}
      </nav>
      <div className="cm-body zw-body">
        <div className="cm-stage zw-stage">
          <svg ref={svgRef} className="cm-svg zw-svg" role="img" aria-label={`The call graph of the run, now at ${frameLabel(frame, ix)}`}>
            <defs>
              {['out', 'in', 'route', 'ret'].map((k) => (
                <marker key={k} id={`zw-arrow-${k}`} className={`cm-arrow cm-arrow-${k}`} viewBox="0 0 8 8" refX="7.5" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto">
                  <path d="M0.5,0.8 L7.5,4 L0.5,7.2 Z" />
                </marker>
              ))}
            </defs>
            <g ref={camG}>
              <Canvas lay={lay} router={router} trunkList={trunkList} map={map} defs={defs} cur={cur} sel={focusDef || cur} hover={hover}
                stackKey={stackIds.join('\u0000')} trail={trail.join('\u0000')} returned={returned ? returned.join('\u0000') : ''}
                hits={hits.map((d) => d.id).join('\u0000')} lvl={lvl} setHover={setHover} />
            </g>
          </svg>
          <Overlay lay={lay} map={map} curModule={defs.get(focusDef || cur)?.module} viewSub={viewSub} lvl={lvl} onJump={(x, y) => {
            const c = cam.current;
            setFollow(false);
            setWhole(false);
            flyTo([x, y, c[2]], !reduced);
          }} />
          <div className="cm-zoom">
            <button type="button" onClick={() => zoomBy(1.25)} aria-label="Zoom in">
              +
            </button>
            <button type="button" onClick={() => zoomBy(0.8)} aria-label="Zoom out">
              −
            </button>
            <button type="button" onClick={overview} aria-label="The whole graph (w)" title="The whole graph (w)">
              ⤢
            </button>
          </div>
          {legend && (
            <div className="cm-legend" role="note">
              <p>
                A region is a file, a tile a def the run called. <span className="cm-k cm-k-sel" /> the def that runs ·{' '}
                <span className="cm-k cm-k-out" /> the calls open now · <span className="cm-k cm-k-in" /> calls into the def ·{' '}
                <span className="cm-k cm-k-trail" /> the defs the run left last
              </p>
              <p>
                Keys: <kbd>←</kbd> <kbd>→</kbd> a stop back and forward, <kbd>↓</kbd> over a call, <kbd>↑</kbd> out of it, <kbd>space</kbd> play,{' '}
                <kbd>f</kbd> follow, <kbd>w</kbd> the whole graph, <kbd>s</kbd> find. Click a tile, a call in the code or a call above the graph to read it.
              </p>
              <p className="cm-of">
                {recorded} in {run.seconds.toFixed(1)} s, {rec.total.toLocaleString('en-US')} events and {frames.length.toLocaleString('en-US')} stops
                {truncated ? ' (the first ones of the run)' : ''}.{run.note ? <> {run.note}</> : null}
              </p>
            </div>
          )}
        </div>
        <div className="zw-time">
          <span className="zw-stop">
            Stop {(i + 1).toLocaleString('en-US')} of {frames.length.toLocaleString('en-US')}
          </span>
          <input type="range" min="0" max={frames.length - 1} value={i}
            onChange={(e) => { setPlaying(false); setFocus(null); setI(Number(e.target.value)); }}
            aria-label="Position in the run" aria-valuetext={frameLabel(frame, ix)} />
          <label className="zw-loops">
            Loops{' '}
            <select value={loops} onChange={(e) => setLoopMode(e.target.value)}>
              <option value="one">one pass each</option>
              <option value="notable">passes worth reading</option>
              <option value="every">every pass</option>
            </select>
          </label>
        </div>
        <Pane ix={ix} map={map} defs={defs} target={code} frame={focus ? null : frame} katex={katex} reduced={reduced} playing={playing} onDef={showDef} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// The graph, drawn as the atlas draws it
// ---------------------------------------------------------------------------

const Canvas = memo(function Canvas({ lay, router, trunkList, map, defs, cur, sel, hover, stackKey, trail, returned, hits, lvl, setHover }) {
  const cache = useMemo(() => new Map(), [router]);
  const pathOf = (a, b) => {
    const k = `${a}>${b}`;
    if (!cache.has(k)) {
      const pts = router.call(a, b);
      cache.set(k, pts ? orthoPath(pts) : null);
    }
    return cache.get(k);
  };
  const stack = stackKey.split('\u0000');
  const onStack = new Set(stack);
  const routeEdges = new Set();
  for (let j = 1; j < stack.length; j++) routeEdges.add(`${stack[j - 1]}>${stack[j]}`);
  const trailAt = new Map(trail ? trail.split('\u0000').map((d, k) => [d, k + 1]) : []);
  const hitSet = new Set(hits ? hits.split('\u0000') : []);
  // the def whose calls are drawn: the one pointed at, else the one shown
  const focus = hover || sel;
  const callees = new Set();
  const callers = new Set();
  for (const [a, b] of lay.edges) {
    if (a === focus && b !== focus) callees.add(b);
    if (b === focus && a !== focus) callers.add(a);
  }
  const edges = [];
  const seen = new Set();
  const add = (a, b, kind, n = 1) => {
    const k = `${a}>${b}`;
    if (seen.has(k) || a === b) return;
    seen.add(k);
    const d = pathOf(a, b);
    if (d) edges.push({ k, d, kind, n });
  };
  // the calls open now, then the call just returned from, then the focused def's own
  for (const k of routeEdges) add(...k.split('>'), 'route');
  if (returned) add(...returned.split('\u0000'), 'ret');
  for (const [a, b, n] of lay.edges) if (a === focus) add(a, b, 'out', n);
  for (const [a, b, n] of lay.edges) if (b === focus) add(a, b, 'in', n);
  // the drawing order: the quiet ones first
  const order = { in: 0, out: 1, ret: 2, route: 3 };
  edges.sort((x, y) => order[x.kind] - order[y.kind]);
  return (
    <>
      <g className="cm-trunks is-quiet">
        {trunkList.map((t) => (
          <path key={`${t.from}>${t.to}`} d={t.d} strokeWidth={Math.round(Math.min(3, 1 + 0.6 * Math.log2(1 + t.calls)) * 10) / 10} />
        ))}
      </g>
      {lay.order.map((k) => {
        const r = lay.regions.get(k);
        const m = map.modules[k];
        return (
          <g key={`r${k}`} className={`cm-region${defs.get(sel)?.module === k ? ' is-cur' : ''}`}>
            <rect x={r.x} y={r.y} width={r.w} height={r.h} rx="6" />
            {lvl >= 3 && (
              <text x={r.x + 10} y={r.y + 17} fontSize={12} className="cm-region-name">
                {m.path.replace(/\.viv$/, '')}
                <tspan className="cm-region-n"> {r.n}</tspan>
              </text>
            )}
          </g>
        );
      })}
      <g className="cm-edges">
        {edges.map((e) => (
          <path key={e.k} d={e.d} className={`is-${e.kind}`} markerEnd={`url(#zw-arrow-${e.kind})`} strokeWidth={e.kind === 'route' ? 3 : e.kind === 'ret' ? 2.4 : 1.8} />
        ))}
      </g>
      {[...lay.tiles.values()].map((t) => {
        const d = defs.get(t.id);
        const cls = [
          'cm-tile',
          t.id === sel && 'is-sel',
          t.id === cur && 'is-run',
          onStack.has(t.id) && t.id !== sel && 'is-route',
          callees.has(t.id) && 'is-callee',
          callers.has(t.id) && 'is-caller',
          hitSet.has(t.id) && 'is-hit',
          trailAt.has(t.id) && `is-trail is-trail-${trailAt.get(t.id)}`,
        ]
          .filter(Boolean)
          .join(' ');
        return (
          <g key={t.id} className={cls} data-def={t.id} tabIndex={0} role="button"
            aria-label={`${nameOf(t.id)}${d.count > 1 ? `, called ${d.count} times` : ''}`}
            onPointerEnter={() => setHover(t.id)} onPointerLeave={() => setHover((h) => (h === t.id ? null : h))}
            onFocus={() => setHover(t.id)} onBlur={() => setHover((h) => (h === t.id ? null : h))}>
            <rect x={t.x} y={t.y} width={t.w} height={t.h} rx="4" />
            {lvl >= 1 && (
              <text x={t.x + 8} y={t.y + 17}>
                {t.label}
              </text>
            )}
            {lvl >= 2 && t.sub && (
              <text className="cm-sub" x={t.x + 8} y={t.y + 32}>
                {t.sub}
              </text>
            )}
            <title>{`${nameOf(t.id)}${d.count > 1 ? ` (called ${d.count} times)` : ''}${t.intent ? `\n${texToPlain(t.intent)}` : ''}`}</title>
          </g>
        );
      })}
    </>
  );
});

/** The module names at small zoom and the minimap: they follow the camera. */
function Overlay({ lay, map, curModule, viewSub, lvl, onJump }) {
  const [v, setV] = useState(null);
  useEffect(() => {
    viewSub.current = setV;
    return () => {
      viewSub.current = null;
    };
  }, [viewSub]);
  if (!v) return null;
  const mini = { w: 180, h: Math.max(40, Math.min(140, (180 * lay.height) / lay.width)) };
  const mk = Math.min(mini.w / lay.width, mini.h / lay.height);
  const inside = v.x >= 0 && v.y >= 0 && v.x + lay.width * v.k <= v.vw && v.y + lay.height * v.k <= v.vh;
  const labels = [];
  if (lvl < 3) {
    const placed = [];
    for (const k of lay.order) {
      const r = lay.regions.get(k);
      const name = map.modules[k].path.replace(/\.viv$/, '').replace(/^(lib|study\/ports)\//, '');
      const x = v.x + r.x * v.k;
      const y = v.y + r.y * v.k - 2;
      const w = name.length * 6.6 + 12;
      if (placed.some((p) => x < p.x + p.w && p.x < x + w && Math.abs(p.y - y) < 16)) continue;
      placed.push({ x, y, w });
      labels.push(
        <span key={k} className={curModule === k ? 'is-cur' : ''} style={{ left: `${x}px`, top: `${y}px` }}>
          {name}
        </span>,
      );
    }
  }
  return (
    <>
      {labels.length > 0 && (
        <div className="cm-labels" aria-hidden="true">
          {labels}
        </div>
      )}
      {!inside && (
        <svg className="cm-mini" width={mini.w} height={mini.h} aria-hidden="true"
          onClick={(ev) => {
            const r = ev.currentTarget.getBoundingClientRect();
            onJump((ev.clientX - r.left) / mk, (ev.clientY - r.top) / mk);
          }}>
          {lay.order.map((k) => {
            const r = lay.regions.get(k);
            return <rect key={k} x={r.x * mk} y={r.y * mk} width={Math.max(1, r.w * mk)} height={Math.max(1, r.h * mk)} className={curModule === k ? 'is-cur' : ''} />;
          })}
          <rect className="cm-mini-view" x={(-v.x / v.k) * mk} y={(-v.y / v.k) * mk} width={(v.vw / v.k) * mk} height={(v.vh / v.k) * mk} />
        </svg>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// The panel: the def, its calls, what is happening, its code
// ---------------------------------------------------------------------------

const HANG = 4;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

/** The program lines the panel shows for a def: its own (with the comment above it), the top level's, or its whole file. */
function rowsOf(ix, files, d, whole) {
  const lineOfMod = (x) => ix.toModule(x)?.line ?? x;
  if (whole || d.main) {
    const path = d.main ? null : `${(ix.toModule(d.line)?.module) || ''}.viv`;
    const f = d.main ? files[0] : files.find((f) => `${f.module}.viv` === path) || files[0];
    let rows = f.rows;
    if (d.main && !whole) {
      // the top level's statements: not the post header (a fenced ### block), not blank lines
      let fenced = false;
      rows = rows.filter((r) => {
        const t = (ix.programLines[r.x - 1] || '').trim();
        if (t.startsWith('###')) {
          fenced = !fenced;
          return false;
        }
        return !fenced && t && defAt(ix, r.x) === '<main>';
      });
    }
    return rows.map((r, k) => ({ x: r.x, n: r.n, use: r.use, gap: k > 0 && d.main && !whole && r.n > rows[k - 1].n + 1 }));
  }
  let a = d.line;
  while (a > 1 && /^\s*#(?![!:|])/.test(ix.programLines[a - 2] || '')) a--;
  const out = [];
  for (let x = a; x <= d.end; x++) out.push({ x, n: lineOfMod(x) });
  return out;
}

const Pane = memo(function Pane({ ix, map, defs, target, frame, katex, reduced, playing, onDef, compact = false }) {
  const { files } = useMemo(() => codeFiles(ix), [ix]);
  const [whole, setWhole] = useState(false);
  const paneRef = useRef(null);
  const d = defs.get(target.def) || defs.get('<main>');
  useEffect(() => setWhole(false), [d.id]);
  const path = d.main ? map.modules[0].path : map.modules[d.module].path;
  const rows = useMemo(() => rowsOf(ix, files, d, whole), [ix, files, d, whole]);
  const callers = useMemo(() => map.defs.filter((x) => x.calls.some(([to]) => to === d.id) && x.id !== d.id).map((x) => x.id), [map, d]);
  const builtins = useMemo(() => Object.entries(builtinsOf(ix, d.id)).sort((a, b) => b[1] - a[1]), [ix, d]);
  const names = useMemo(() => new Set(defs.keys()), [defs]);
  const lines = useMemo(
    () =>
      rows.map((r) => {
        // an import line: the spliced program has the module in its place
        if (r.use) return { ...r, html: `<span class="tk-com">import of ${esc(r.use)}.viv, spliced in here</span>`, indent: 0, step: false, out: false };
        const raw = (ix.programLines[r.x - 1] || '').replace(/\t/g, '    ');
        const indent = raw.length - raw.trimStart().length;
        let html = highlight(raw.trimStart(), 'viv');
        html = html.replace(/(^|>)([^<]+)/g, (m, open, txt) =>
          open + txt.replace(/(\bstep\s+)?\b([A-Za-z_]\w*)\b/g, (all, step, w) => (!step && names.has(w) && w !== d.id ? `<a class="cm-ref" data-def="${esc(w)}" href="#">${w}</a>` : all)),
        );
        return { ...r, html, indent, step: /^\s*#:/.test(raw), out: !d.main && !whole ? r.x < d.line : !d.main && (r.x < d.line || r.x > d.end) };
      }),
    [rows, ix, names, d, whole],
  );

  // the running line marked and kept in view; a new def starts at its head
  const lastDef = useRef(null);
  useLayoutEffect(() => {
    const pane = paneRef.current;
    if (!pane) return;
    pane.querySelectorAll('.cm-ln.is-run, .cm-ln.is-picked').forEach((el) => el.classList.remove('is-run', 'is-picked'));
    const ln = target.line != null ? pane.querySelector(`.cm-ln[data-line="${target.line}"]`) : null;
    if (ln && target.mark) ln.classList.add(target.mark);
    const scrolls = pane.scrollHeight > pane.clientHeight + 1 && getComputedStyle(pane).overflowY !== 'visible';
    const newDef = lastDef.current !== d.id;
    lastDef.current = d.id;
    if (!scrolls) return;
    if (!ln) {
      if (newDef) pane.scrollTop = 0;
      return;
    }
    const top = ln.offsetTop - pane.offsetTop;
    const lh = ln.offsetHeight || 20;
    const h = pane.clientHeight;
    if (!newDef && top >= pane.scrollTop + 2 * lh && top + lh <= pane.scrollTop + h - 2 * lh) return;
    const head = pane.querySelector('.cm-pane-head');
    const to = newDef && head && top + 2 * lh <= h ? 0 : Math.max(0, top - h * 0.35);
    pane.scrollTo({ top: to, behavior: reduced || newDef ? 'auto' : 'smooth' });
  }, [target.line, target.mark, d.id, lines, reduced]);

  const first = lines.length ? lines.find((l) => !l.out) || lines[0] : null;
  const intent = intentOf(d);
  return (
    <aside className="cm-pane zw-pane" ref={paneRef} aria-label="The def's code">
      <header className="cm-pane-head">
        <h1>
          <code>{nameOf(d.id)}</code>
          {!d.main && <span className="cm-params">({d.params.join(', ')})</span>}
        </h1>
        <p className="cm-where">
          {REPO_PUBLIC ? (
            <a href={`${GH}${path}${d.main || !first ? '' : `#L${first.n}-L${lines[lines.length - 1].n}`}`} rel="noopener" target="_blank">
              {path}
              {!d.main && first && `:${first.n}`}
            </a>
          ) : (
            <span>
              {path}
              {!d.main && first && `:${first.n}`}
            </span>
          )}
          {!d.main && <> · {d.end - d.line + 1} lines</>}
          {d.count > 1 && <> · called {d.count.toLocaleString('en-US')} times in this run</>}
        </p>
        {intent && katex && d.tex ? <Md className="cm-intent" text={d.tex} katex={katex} /> : intent ? <p className="cm-intent">{intent}</p> : <p className="cm-intent is-missing">No intent written for this def.</p>}
        {frame && <Now ix={ix} frame={frame} katex={katex} playing={playing} />}
        {!compact && callers.length > 0 && (
          <p className="cm-rel">
            <span>Called by</span>
            {callers.map((c) => (
              <button type="button" key={c} className="is-caller" onClick={() => onDef(c)}>
                {nameOf(c)}
              </button>
            ))}
          </p>
        )}
        {!compact && d.calls.length > 0 && (
          <p className="cm-rel">
            <span>Calls</span>
            {d.calls.map(([to, , , , n]) => (
              <button type="button" key={to} className="is-callee" onClick={() => onDef(to)} title={`${n === 1 ? 'once' : `${n} times`} in this run`}>
                {nameOf(to)}
                {n > 1 && <small>×{n}</small>}
              </button>
            ))}
          </p>
        )}
        {!compact && builtins.length > 0 && (
          <p className="cm-rel cm-bi">
            <span>Builtins</span>
            {builtins.map(([b, c]) => (
              <code key={b}>
                {b}
                {c > 1 && <small>×{c}</small>}
              </code>
            ))}
          </p>
        )}
      </header>
      <pre
        className="cm-code"
        onClick={(e) => {
          const a = e.target.closest('a.cm-ref');
          if (a) {
            e.preventDefault();
            onDef(a.dataset.def);
          }
        }}>
        {lines.map((l) => (
          <div key={l.x} data-line={l.x} className={`cm-ln${l.gap ? ' is-gap' : ''}${l.step ? ' is-step' : ''}${l.out ? ' is-out' : ''}`}>
            <span className="cm-num">{l.n}</span>
            <span className="cm-txt" style={{ paddingLeft: `${l.indent + HANG}ch`, textIndent: `-${HANG}ch` }} dangerouslySetInnerHTML={{ __html: l.html || ' ' }} />
          </div>
        ))}
      </pre>
      {!d.main && !compact && (
        <button type="button" className="cm-whole" onClick={() => setWhole((w) => !w)}>
          {whole ? 'Show this def only' : `Show all of ${path}`}
        </button>
      )}
    </aside>
  );
});

// ---------------------------------------------------------------------------
// What is happening at a stop
// ---------------------------------------------------------------------------

function frameLabel(f, ix) {
  const name = (d) => (d === '<main>' ? 'the top level' : defLabel(ix, d));
  switch (f.kind) {
    case 'start': return 'the start of the run';
    case 'end': return 'the end of the run';
    case 'call': return `the call of ${name(f.def)}`;
    case 'ret': return `the return to ${name(f.def)}`;
    case 'step': return `step ${f.gid.split('/')[1]} of ${name(f.def)}`;
    case 'loop': return `a loop of ${name(f.def)}`;
    case 'iter': return `pass ${f.pass.n} of a loop in ${name(f.def)}`;
    case 'group': return 'the setup of the modules';
    case 'line': {
      const m = ix.toModule(f.line);
      return m ? `${m.module}.viv line ${m.line}` : `line ${f.line}`;
    }
    default: return f.kind;
  }
}

function Md({ text, katex, as: Tag = 'p', className }) {
  const html = useMemo(() => renderProse(text || '', katex), [text, katex]);
  return <Tag className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}

function Now({ ix, frame: f, katex, playing }) {
  const name = (d) => (d === '<main>' ? 'the top level' : defLabel(ix, d));
  const lineNo = (line) => {
    const m = line ? ix.toModule(line) : null;
    return m ? `line ${m.line}` : null;
  };
  const stepText = f.step ? ix.bundle.steps[f.step] : f.gid && ix.bundle.steps[f.gid];
  let head;
  switch (f.kind) {
    case 'start':
      head = <>The run starts at the top level.</>;
      break;
    case 'end':
      head = <>The run is over.</>;
      break;
    case 'call':
      head = (
        <>
          <code>{name(f.via.caller)}</code> calls <code>{name(f.def)}</code>
          {f.via.line ? <> at {lineNo(f.via.line)}</> : null}
          {f.count > 1 ? <>, {f.count} times in a row (the walk follows the first)</> : null}.
        </>
      );
      break;
    case 'ret':
      head = <><code>{name(f.from)}</code> returns to <code>{name(f.def)}</code>{f.count > 1 ? <> after {f.count} calls</> : null}.</>;
      break;
    case 'step':
      head = <>Step <code>{f.gid.split('/')[1]}</code></>;
      break;
    case 'loop': {
      const g = f.gid && ix.nodes.get(f.gid);
      head = <>The loop <code>{g ? g.header : ''}</code> runs {f.total} {f.total === 1 ? 'pass' : 'passes'}.</>;
      break;
    }
    case 'iter':
      head = <>Pass {f.pass.n} of {f.pass.total}{f.shown < f.pass.total ? <> (the walk shows {f.shown === 1 ? 'this one' : `${f.shown} of them`})</> : null}.</>;
      break;
    case 'group':
      head = <>The modules set their constants and tables.</>;
      break;
    case 'line':
      head = <>Running {lineNo(f.line)}{f.pass ? <>, pass {f.pass.n} of {f.pass.total}</> : null}</>;
      break;
    default:
      head = null;
  }
  const values = f.kind === 'line' ? Object.entries(f.values || {}).slice(0, 4) : [];
  return (
    <div className="zw-now" aria-live={playing ? 'off' : 'polite'}>
      <p className="zw-now-head">{head}</p>
      {values.length > 0 && (
        <p className="zw-values">
          {values.map(([k, v]) => (
            <span key={k}>
              <code>{k}</code> = {typeof v === 'object' && v ? (v.shape ? `${v.shape.join(' × ')} array` : 'an array') : numText(v)}
            </span>
          ))}
        </p>
      )}
      {stepText && <Md className="zw-step" text={stepText} katex={katex} />}
    </div>
  );
}
