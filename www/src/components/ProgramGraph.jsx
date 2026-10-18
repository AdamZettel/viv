// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The whole program as a graph, at the top of the reading view, with a panel
// beside it (below it on a narrow screen) for what the reader picks.
//
// The picture is drawn by Viv: lib/graphdraw.viv's `draw_view`, at stage time
// for the first view (www/scripts/explore-build.mjs) and in this tab's
// viv.wasm, in a worker, when the reader opens or closes a box. The page
// renders its figure records (DiagramView) and works on top of the ids the
// drawing gave its marks: it pans and zooms, lights a box's calls and values
// when the pointer is on it, outlines the reader's place, the picked box and
// the keyboard's, and shows the picked def's or step's code in the panel.
// Which defs are boxes is src/lib/graphView.js: one box per def the run
// called, from a first depth (the defs with steps, the defs on a call path to
// one, and what they call); a box marked `+n` hides n callees.
//
//   ix        the explorer's indexed bundle (lib/explore.js indexBundle)
//   katex     KaTeX, once loaded
//   def, step where the reader's line is in the full reading below
//   onRead    called with a def's name to show it in the full reading
//   run       the recorded run (Explorer.jsx), once it is recorded: the Run
//             button then walks through it (lib/tour.js), the picture
//             following each stop and the panel narrating it
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { tourStops } from '../lib/tour.js';
import DiagramView, { diagramFrame, boxRect } from './DiagramView.jsx';
import GraphPanel from './GraphPanel.jsx';
import { parseOwnOutput } from './VivCell.jsx';
import { MAIN, graphModel, defaultOpen, shownOf, hiddenCallees, closes, revealed, drawSource, hintsOf, wayOf, drawingOf, isMp, mpOf, robustWords, boxFor } from '../lib/graphView.js';
import { vivWorker } from '../lib/vivWasm.js';

/** The first view is no smaller than this many pixels to a unit, so its text can be read. */
const FIRST_SCALE = 0.85;
const FIRST_SCALE_NARROW = 0.62;
/** A pane narrower than this many pixels is a phone's. */
const NARROW_PANE = 520;
const MIN_SCALE = 0.08;
const MAX_SCALE = 3;
/** How much one button press or key zooms. */
const ZOOM_STEP = 1.25;
/** A press that moves less than this many pixels is a click, not a drag. */
const CLICK_SLOP = 5;
/** After the reader moves the picture, the picture does not follow the reading for this long. */
const HANDS_OFF_MS = 2500;
/** How long a move to a box takes. */
const GLIDE_MS = 260;
/** Room kept around a box brought into view, in pixels. */
const VIEW_PAD = 28;
/** The most search results listed. */
const MAX_RESULTS = 10;
/** A wheel turn this large, in whole pixels and straight down, is a mouse's notch: it zooms. */
const NOTCH = 50;
/** The picture's share of the width: with nothing picked, and with a box picked (the panel then has code to show). */
const SPLIT_NONE = 0.615;
const SPLIT_PICKED = 0.5;
/** How far the divider can be dragged, as the picture's share; one arrow key moves it this much. */
const SPLIT_MIN = 0.3;
const SPLIT_MAX = 0.8;
const SPLIT_KEY_STEP = 0.03;
/** Where a dragged divider is remembered, in this browser. */
const SPLIT_KEY = 'viv.graph.split';
/** How long the walkthrough stays at a stop at 1x, in ms; the first, loop and last stops longer. */
const TOUR_MS = 2600;
const TOUR_LONG_MS = 4200;
/** The walkthrough's speeds. */
const TOUR_SPEEDS = [0.5, 1, 2, 4];

const keyOf = (open) => [...open].sort().join('\n');
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const defOf = (id) => String(id).split('/')[0];
const within = (id, box) => id === box || id.startsWith(`${box}/`);

/** The arrows of a drawing, by kind and ends: { id, kind, from, to }. */
function arrowsOf(chart) {
  const out = [];
  const boxIds = new Set((chart ? chart.marks : []).filter((m) => m.kind === 'box').map((m) => m.id));
  for (const m of chart ? chart.marks : []) {
    if (m.kind !== 'arrow') continue;
    const k = m.id.indexOf(':');
    const kind = m.id.slice(0, k);
    if (kind === 'legend') continue;
    if (kind === 'mp') {
      // a multi-path's arrows: `<node>/in` from its host, `<node>/to:<member>`
      // from the fork, `<node>/from:<member>` to the join
      const [node, part] = [mpOf(m.id), m.id.slice(mpOf(m.id).length + 1)];
      const host = node.slice(node.indexOf('@') + 1);
      const member = (x) => (boxIds.has(`${node}/m:${x}`) ? `${node}/m:${x}` : x);
      const head = boxIds.has(node) ? node : `${node}/fork`;
      if (part === 'in') out.push({ id: m.id, kind: 'call', from: host, to: head });
      else if (part.startsWith('to:')) out.push({ id: m.id, kind: 'call', from: `${node}/fork`, to: member(part.slice(3)) });
      else if (part.startsWith('from:')) out.push({ id: m.id, kind: 'call', from: member(part.slice(5)), to: `${node}/join` });
      continue;
    }
    const [from, to] = m.id.slice(k + 1).split('->');
    out.push({ id: m.id, kind, from, to });
  }
  return out;
}

export default function ProgramGraph({ ix, katex, def, step, onRead, run = null }) {
  const bundle = ix.bundle;
  const model = useMemo(() => graphModel(bundle), [bundle]);
  const first = useMemo(() => defaultOpen(model), [model]);
  const [open, setOpen] = useState(first);
  const [records, setRecords] = useState(bundle.drawing || '');
  const cache = useRef(new Map([[keyOf(first), bundle.drawing || '']]));
  const chart = useMemo(() => parseOwnOutput(records || '').charts.find((c) => c.kind === 'diagram') || null, [records]);
  const frame = useMemo(() => (chart ? diagramFrame(chart) : null), [chart]);
  const boxes = useMemo(() => new Map((chart ? chart.marks : []).filter((m) => m.kind === 'box').map((m) => [m.id, m])), [chart]);
  const arrows = useMemo(() => arrowsOf(chart), [chart]);
  const shown = useMemo(() => shownOf(model, open), [model, open]);
  const [sel, setSel] = useState(null);
  const [tourOn, setTourOn] = useState(false); // the walkthrough of the run (below)
  const [vb, setVb] = useState(null);
  const [pane, setPane] = useState({ w: 0, h: 0 });
  const [busy, setBusy] = useState(false);
  const [redraw, setRedraw] = useState(null); // how long the last redrawing took
  const [err, setErr] = useState('');
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const [say, setSay] = useState('');
  const reduced = useMemo(() => typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches, []);
  const paneRef = useRef(null);
  const svgRef = useRef(null);
  const tipRef = useRef(null);
  const hoverRef = useRef(null);
  const worker = useRef(null);
  const anchor = useRef(null);
  const pendingGo = useRef(null);
  const handsOff = useRef(0);
  const glide = useRef(0);
  const vbRef = useRef(vb);
  vbRef.current = vb;
  // the divider: where the reader put it, or the default for picked or not
  const [split, setSplit] = useState(null);
  const [dragging, setDragging] = useState(false);
  const splitRef = useRef(null);
  useEffect(() => {
    try {
      const v = Number(localStorage.getItem(SPLIT_KEY));
      if (v >= SPLIT_MIN && v <= SPLIT_MAX) setSplit(v);
    } catch { /* no storage: the default */ }
  }, []);
  const keepSplit = (v) => {
    setSplit(v);
    try {
      if (v === null) localStorage.removeItem(SPLIT_KEY);
      else localStorage.setItem(SPLIT_KEY, String(v));
    } catch { /* not remembered */ }
  };
  const frac = split ?? (sel || tourOn ? SPLIT_PICKED : SPLIT_NONE);
  const splitAt = (clientX) => {
    const r = splitRef.current.getBoundingClientRect();
    return clamp((clientX - r.left) / r.width, SPLIT_MIN, SPLIT_MAX);
  };
  const onDividerDown = (e) => {
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging(true);
  };
  const onDividerMove = (e) => {
    if (dragging) setSplit(splitAt(e.clientX));
  };
  const onDividerUp = (e) => {
    if (!dragging) return;
    setDragging(false);
    keepSplit(splitAt(e.clientX));
  };
  const onDividerKey = (e) => {
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault();
      keepSplit(clamp(frac + (e.key === 'ArrowLeft' ? -SPLIT_KEY_STEP : SPLIT_KEY_STEP), SPLIT_MIN, SPLIT_MAX));
    } else if (e.key === 'Home' || e.key === 'Escape') keepSplit(null);
  };

  // ---- the walkthrough of the recorded run
  const [tourK, setTourK] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [tourOpen, setTourOpen] = useState(() => new Set()); // passes walked step by step
  const [peek, setPeek] = useState(null); // a def the reader opened from a stop, the walk paused
  const stops = useMemo(() => (run && tourOn ? tourStops(run.tree, ix, tourOpen) : []), [run, ix, tourOn, tourOpen]);
  const tourStop = tourOn && stops.length ? stops[Math.min(tourK, stops.length - 1)] : null;

  // ---- the worker that redraws: viv.wasm, loaded the first time the reader
  // changes the picture (the first view is the bundle's)
  useEffect(() => () => {
    if (worker.current) worker.current.terminate();
    worker.current = null;
  }, []);
  const runViv = useCallback((src) => new Promise((resolve, reject) => {
    if (!worker.current) {
      try {
        worker.current = vivWorker('/graph-worker.js');
      } catch {
        worker.current = null;
      }
    }
    const w = worker.current;
    if (!w) return reject(new Error('this browser cannot start the worker that redraws the picture'));
    const id = Math.random().toString(36).slice(2);
    const on = (e) => {
      if (e.data.id !== id) return;
      w.removeEventListener('message', on);
      if (e.data.error) reject(new Error(e.data.error));
      else resolve(e.data);
    };
    w.addEventListener('message', on);
    w.postMessage({ id, src });
  }), []);

  // ---- the pane's size, and the first view
  useLayoutEffect(() => {
    const el = paneRef.current;
    if (!el) return undefined;
    const ro = new ResizeObserver(() => setPane({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setPane({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);
  const fitView = useCallback(() => {
    if (!frame || !pane.w) return null;
    const s = clamp(Math.min(pane.w / frame.W, pane.h / frame.H), MIN_SCALE, 1);
    return { x: (frame.W - pane.w / s) / 2, y: (frame.H - pane.h / s) / 2, s };
  }, [frame, pane]);
  useLayoutEffect(() => {
    if (vb || !frame || !pane.w) return;
    const fit = fitView();
    if (fit.s >= (pane.w < NARROW_PANE ? FIRST_SCALE_NARROW : FIRST_SCALE)) {
      setVb(fit);
      return;
    }
    // the picture's start, where `<main>` is, at a size that can be read
    // (smaller on a phone, where less of it fits)
    const s = pane.w < NARROW_PANE ? FIRST_SCALE_NARROW : FIRST_SCALE;
    const m = boxes.get(MAIN);
    const r = m ? boxRect(frame, m) : { x: 0, y: 0, w: 0, h: 0 };
    const down = wayOf(records) === 'down';
    setVb(down
      ? { x: r.x + r.w / 2 - pane.w / (2 * s), y: 0, s }
      : { x: 0, y: clamp(r.y + r.h / 2 - pane.h / (2 * s), 0, Math.max(0, frame.H - pane.h / s)), s });
  }, [vb, frame, pane, fitView, boxes, records]);

  /** The rectangle of a box, in the picture's units. */
  const rectOf = useCallback((id) => (boxes.has(id) && frame ? boxRect(frame, boxes.get(id)) : null), [boxes, frame]);

  // the pane wider or narrower (the divider moved, a box picked, the window
  // resized): the picture keeps its center, and the picked box stays in view
  const lastW = useRef(0);
  useLayoutEffect(() => {
    const old = lastW.current;
    lastW.current = pane.w;
    const v = vbRef.current;
    if (!old || !pane.w || old === pane.w || !v) return;
    const next = { ...v, x: v.x + (old - pane.w) / (2 * v.s) };
    const r = sel ? rectOf(boxes.has(sel) ? sel : defOf(sel)) : null;
    if (r) {
      const vw = pane.w / v.s;
      const pad = VIEW_PAD / v.s;
      if (r.x < next.x + pad || r.x + Math.min(r.w, vw / 2) > next.x + vw - pad) next.x = r.w > vw - 2 * pad ? r.x - pad : r.x + r.w / 2 - vw / 2;
    }
    setVb(next);
  }, [pane.w]); // eslint-disable-line react-hooks/exhaustive-deps

  /** Move the view to `to`, gliding unless the reader asked for no motion. */
  const moveTo = useCallback((to) => {
    cancelAnimationFrame(glide.current);
    const from = vbRef.current;
    if (!from || reduced) {
      setVb(to);
      return;
    }
    const t0 = performance.now();
    const tick = (t) => {
      const u = Math.min(1, (t - t0) / GLIDE_MS);
      const e = u < 0.5 ? 2 * u * u : 1 - (-2 * u + 2) ** 2 / 2;
      setVb({ x: from.x + (to.x - from.x) * e, y: from.y + (to.y - from.y) * e, s: from.s + (to.s - from.s) * e });
      if (u < 1) glide.current = requestAnimationFrame(tick);
    };
    glide.current = requestAnimationFrame(tick);
  }, [reduced]);

  /** The view with box `id` in it: null when it is in view already (unless `center`), else one centered on it. */
  const viewWith = useCallback((id, center = false) => {
    const r = rectOf(id);
    const v = vbRef.current;
    if (!r || !v || !pane.w) return null;
    const pad = VIEW_PAD / v.s;
    const vw = pane.w / v.s;
    const vh = pane.h / v.s;
    const inside = r.x >= v.x + pad && r.x + Math.min(r.w, vw / 2) <= v.x + vw - pad && r.y >= v.y + pad && r.y + Math.min(r.h, vh / 2) <= v.y + vh - pad;
    if (inside && !center) return null;
    // a box larger than the view shows its top left
    const cx = r.w > vw - 2 * pad ? r.x - pad + vw / 2 : r.x + r.w / 2;
    const cy = r.h > vh - 2 * pad ? r.y - pad + vh / 2 : r.y + r.h / 2;
    return { x: cx - vw / 2, y: cy - vh / 2, s: v.s };
  }, [rectOf, pane]);

  // ---- redrawing: a box opened or closed, a def revealed, all or the first view
  const redrawTo = useCallback(async (next, keep, then = null) => {
    const key = keyOf(next);
    const t0 = performance.now();
    let recs = cache.current.get(key);
    // the box that stays where it is on the screen while the others move
    const r = keep ? rectOf(keep) : null;
    const v = vbRef.current;
    const mark = r && v ? { id: keep, sx: (r.x - v.x) * v.s, sy: (r.y - v.y) * v.s, then, t0 } : { id: null, then, t0 };
    if (!recs) {
      setBusy(true);
      try {
        const { src } = drawSource(model, next, { hints: hintsOf(records), way: wayOf(records), title: bundle.name });
        const res = await runViv(bundle.drawLib + src);
        recs = drawingOf(res.out);
        if (!recs.startsWith('figure: ')) throw new Error(`lib/graphdraw drew nothing: ${res.out.slice(0, 200)}`);
        cache.current.set(key, recs);
        setErr('');
      } catch (e) {
        setBusy(false);
        setErr(String(e.message || e));
        return;
      }
      setBusy(false);
    }
    anchor.current = mark;
    setOpen(next);
    setRecords(recs);
    if (recs === records) {
      // the same drawing (a key that maps to it): nothing re-renders, so finish here
      anchor.current = null;
      setRedraw((r) => ({ ms: Math.round(performance.now() - t0), n: (r ? r.n : 0) + 1 }));
      if (then) then();
    }
  }, [model, records, bundle, rectOf, runViv]);

  // after a redrawing: the kept box where it was on the screen; how long it took
  useLayoutEffect(() => {
    const a = anchor.current;
    if (!a || !frame) return;
    anchor.current = null;
    const v = vbRef.current;
    if (a.id && v) {
      const r = rectOf(a.id);
      if (r) setVb({ x: r.x - a.sx / v.s, y: r.y - a.sy / v.s, s: v.s });
    }
    if (a.then) a.then();
    setRedraw((r) => ({ ms: Math.round(performance.now() - a.t0), n: (r ? r.n : 0) + 1 }));
  }, [chart]); // eslint-disable-line react-hooks/exhaustive-deps
  // a redrawing that revealed a box: bring it into view once the new boxes are known
  useEffect(() => {
    const id = pendingGo.current;
    if (!id) return;
    pendingGo.current = null;
    const to = viewWith(boxes.has(id) ? id : defOf(id), true);
    if (to) moveTo(to);
  }, [boxes, viewWith, moveTo]);

  const toggle = useCallback((d) => {
    if (isMp(d)) {
      // a multi-path opens into its fork, members and join, and closes into one box
      const node = mpOf(d);
      const mp = model.mps.get(node);
      const next = new Set(open);
      const opening = !open.has(node);
      if (opening) next.add(node);
      else next.delete(node);
      handsOff.current = performance.now();
      setSay(opening ? `Opened the multi-path ${mp ? mp.stage.id : node}: its ${mp ? mp.stage.members.length : ''} members, the fork and the join` : `Closed the multi-path ${mp ? mp.stage.id : node}`);
      redrawTo(next, mp ? mp.host : null);
      return;
    }
    const next = new Set(open);
    const hidden = hiddenCallees(model, shown, d);
    if (hidden.length) next.add(d);
    else if (closes(model, open, shown, d)) next.delete(d);
    else return;
    handsOff.current = performance.now();
    setSay(hidden.length ? `Opened ${d}: ${hidden.length} more def${hidden.length === 1 ? '' : 's'} drawn` : `Closed ${d}`);
    redrawTo(next, d);
  }, [open, model, shown, redrawTo]);

  const sayOf = useCallback((id) => {
    const n = model.byId.get(id);
    const t = n && n.kind === 'step' ? n.text || (bundle.steps || {})[id] || '' : '';
    return t ? `${id}: ${t}` : id;
  }, [model, bundle]);

  /** Pick a def or step and bring its box into view, opening what must be opened. */
  const go = useCallback((id, { center = true } = {}) => {
    handsOff.current = performance.now();
    setSel(id);
    setSay(sayOf(id));
    if (!model.ranSet.has(defOf(id))) return;
    const next = revealed(model, open, id);
    if (next && next !== open) {
      redrawTo(next, null, () => {
        pendingGo.current = id;
      });
      return;
    }
    const to = viewWith(boxes.has(id) ? id : defOf(id), center);
    if (to) moveTo(to);
  }, [model, open, boxes, viewWith, moveTo, redrawTo, sayOf]);

  // the walkthrough: the picture follows the stop, opening the path to its box
  const tourGo = useRef(null);
  tourGo.current = go;
  useEffect(() => {
    if (!tourStop || peek) return;
    // a def folded into a multi-path (a sweep driver) is shown by the multi-path's box
    tourGo.current(boxFor(model, tourStop.box, tourStop.node), { center: false });
  }, [tourStop && tourStop.k, tourStop && tourStop.box, peek]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!playing || !tourStop || peek) return undefined;
    if (tourStop.k >= stops.length - 1) {
      setPlaying(false);
      return undefined;
    }
    const long = tourStop.kind === 'intro' || tourStop.kind === 'loop' || tourStop.kind === 'end';
    const t = setTimeout(() => setTourK((k) => Math.min(k + 1, stops.length - 1)), (long ? TOUR_LONG_MS : TOUR_MS) / speed);
    return () => clearTimeout(t);
  }, [playing, tourStop, speed, stops.length, peek]);
  const startTour = () => {
    setTourOn(true);
    setPeek(null);
    if (!tourOn || (tourStop && tourStop.k >= stops.length - 1)) setTourK(0);
    setPlaying(true);
    const sec = paneRef.current && paneRef.current.closest('.pg');
    if (sec) {
      const r = sec.getBoundingClientRect();
      if (r.top < 0 || r.bottom > window.innerHeight) sec.scrollIntoView({ block: window.innerWidth < 900 ? 'start' : 'end', behavior: reduced ? 'auto' : 'smooth' });
    }
  };
  const tourMove = (d) => {
    setPeek(null);
    setTourK((k) => clamp(k + d, 0, Math.max(0, stops.length - 1)));
  };
  const exitTour = () => {
    setTourOn(false);
    setPlaying(false);
    setPeek(null);
  };
  const openPass = (key) => {
    setTourOpen((s) => new Set(s).add(key));
    setTourK((k) => k + 1);
  };
  const peekAt = (id) => {
    setPlaying(false);
    setPeek(id);
    go(id, { center: false });
  };
  // the keyboard during the walk: space plays or pauses, the arrows step, Escape leaves it
  useEffect(() => {
    if (!tourOn) return undefined;
    const onKeyDown = (e) => {
      const t = e.target;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
      if (e.key === ' ') {
        if (t && t.tagName === 'BUTTON') return; // the button's own click
        e.preventDefault();
        setPlaying((p) => !p);
      } else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        if (t && t.closest && t.closest('.pg-divider')) return;
        e.preventDefault();
        setPlaying(false);
        tourMove(e.key === 'ArrowRight' ? 1 : -1);
      } else if (e.key === 'Escape') exitTour();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [tourOn, stops.length]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- the reader's place in the full reading, outlined; the view follows it
  // when it leaves the view, unless the reader has just moved the picture
  const readerBox = useMemo(() => {
    if (!def) return null;
    if (step && boxes.has(`${def}/${step}`)) return `${def}/${step}`;
    return boxes.has(def) ? def : null;
  }, [def, step, boxes]);
  const nearBox = useMemo(() => {
    // the reader is in a def that is not drawn: its nearest drawn caller
    if (!def || readerBox || !model.depth.has(def)) return null;
    let d = def;
    for (let k = 0; k < model.ran.length && d && !boxes.has(d); k++) {
      const ps = (model.callers.get(d) || []).filter((p) => model.depth.has(p) && model.depth.get(p) < model.depth.get(d));
      d = ps.find((p) => boxes.has(p)) || ps[0];
    }
    return d && boxes.has(d) ? d : null;
  }, [def, readerBox, boxes, model]);
  useEffect(() => {
    const target = readerBox || nearBox;
    if (!target || performance.now() - handsOff.current < HANDS_OFF_MS) return;
    const to = viewWith(target);
    if (to) moveTo(to);
  }, [readerBox, nearBox]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- lighting: the box or arrow under the pointer and what it touches; the rest dimmed
  const lightUp = useCallback((hit) => {
    const svg = svgRef.current;
    if (!svg) return;
    const lit = new Set();
    if (hit) {
      const touch = (a) => [a.id, a.from, a.to, defOf(a.from), defOf(a.to)].forEach((x) => lit.add(x));
      if (hit.kind === 'arrow') {
        const a = arrows.find((x) => x.id === hit.id);
        if (a) touch(a);
      } else {
        lit.add(hit.id);
        lit.add(defOf(hit.id));
        for (const b of boxes.keys()) if (within(b, hit.id)) lit.add(b);
        for (const a of arrows) if (within(a.from, hit.id) || within(a.to, hit.id)) touch(a);
      }
      // a lit step's loops stay lit, so it is not drawn inside a dimmed frame
      for (const b of boxes.keys()) {
        if (!b.includes('/loop@') || lit.has(b)) continue;
        const r = boxes.get(b);
        for (const x of lit) {
          const m = boxes.get(x);
          if (m && x !== b && defOf(x) === defOf(b) && x.includes('/') && Math.min(m.x0, m.x1) >= Math.min(r.x0, r.x1) && Math.max(m.x0, m.x1) <= Math.max(r.x0, r.x1) && Math.min(m.y0, m.y1) >= Math.min(r.y0, r.y1) && Math.max(m.y0, m.y1) <= Math.max(r.y0, r.y1)) {
            lit.add(b);
            break;
          }
        }
      }
    }
    svg.classList.toggle('pg-dimming', !!hit);
    for (const el of svg.querySelectorAll('[data-id]')) {
      const own = el.getAttribute('data-of') || el.getAttribute('data-id');
      el.classList.toggle('pg-lit', !!hit && lit.has(own));
    }
  }, [arrows, boxes]);

  const tipFor = useCallback((hit) => {
    const intent = bundle.intent || {};
    const firstSentence = (t) => String(t || '').split(/(?<=[.:;])\s/)[0].replace(/`([^`]*)`/g, '$1');
    const times = (c) => `${c.toLocaleString('en-US')} time${c === 1 ? '' : 's'}`;
    if (hit.kind === 'mark') {
      const n = hiddenCallees(model, shown, hit.id).length;
      return n ? `Show the ${n} def${n === 1 ? '' : 's'} ${hit.id} calls` : `Hide what ${hit.id} calls`;
    }
    if (hit.kind === 'arrow') {
      const a = arrows.find((x) => x.id === hit.id);
      if (!a) return '';
      if (isMp(a.id)) {
        const mp = model.mps.get(mpOf(a.id));
        if (!mp) return '';
        const part = a.id.slice(mpOf(a.id).length + 1);
        if (part === 'in') return `${mp.host} uses ${mp.stage.id}: ${mp.stage.title}`;
        const m = part.slice(part.indexOf(':') + 1);
        const c = mp.members.get(m);
        return `${m}, ${m === mp.stage.default ? 'the default' : 'a cross-check'}${c ? `: ran ${times(c)} in this run` : ': did not run in this run'}`;
      }
      if (a.kind === 'call') {
        const c = model.counts.get(`${defOf(a.from)}>${a.to}`);
        return `${a.from} calls ${a.to}${c ? `, ${times(c)} in this run` : ''}`;
      }
      const e = (bundle.edges || []).find((x) => x.kind === 'data' && x.from === a.from && x.to === a.to);
      const vars = e && e.vars.length ? e.vars.join(', ') : 'values';
      if (a.kind === 'next') return `${a.from} hands ${vars} to ${a.to} for the next pass`;
      if (a.kind === 'then') return `${a.to} runs after ${a.from}`;
      return `${a.from} hands ${vars} to ${a.to}`;
    }
    if (isMp(hit.id)) {
      const mp = model.mps.get(mpOf(hit.id));
      if (!mp) return hit.id;
      const m = hit.id.includes('/m:') ? hit.id.split('/m:')[1] : null;
      if (m) return `${m}: ${m === mp.stage.default ? 'the default, which the production path runs' : 'a cross-check member'} of ${mp.stage.id}`;
      return `${mp.stage.id}: ${mp.stage.members.length} implementations of one intent; ${robustWords(mp.stage)}. ${open.has(mpOf(hit.id)) ? 'Click the fork or the join to close it.' : 'Click to open it.'}`;
    }
    const n = model.byId.get(hit.id);
    if (!n) return hit.id;
    if (n.kind === 'def') {
      const callers = (model.callers.get(n.id) || []).filter((c) => model.ranSet.has(c));
      const count = callers.reduce((s, c) => s + (model.counts.get(`${c}>${n.id}`) || 0), 0);
      const it = intent[n.id] || {};
      const lines = [it.sig || n.id];
      if (it.what) lines.push(firstSentence(it.what));
      if (n.id !== MAIN && count) lines.push(`called ${times(count)}, from ${callers.slice(0, 3).join(', ')}${callers.length > 3 ? ' and others' : ''}`);
      return lines.join('\n');
    }
    if (n.kind === 'step') {
      const lines = [`${n.def} / ${n.step}`];
      const t = n.text || (bundle.steps || {})[n.id];
      if (t) lines.push(firstSentence(t));
      if ((n.reads || []).length) lines.push(`reads ${n.reads.join(', ')}`);
      if ((n.writes || []).length) lines.push(`writes ${n.writes.join(', ')}`);
      return lines.join('\n');
    }
    return n.header || n.id;
  }, [bundle, model, shown, arrows, open]);

  /** What the pointer is on: a box, an arrow, or a def's +n / − mark. */
  const hitOf = useCallback((target) => {
    const el = target && target.closest ? target.closest('[data-id]') : null;
    if (!el) return null;
    const id = el.getAttribute('data-id');
    if (/#(more|less)$/.test(id)) return { kind: 'mark', id: id.split('#')[0] };
    const own = el.getAttribute('data-of') || id;
    if (boxes.has(own)) return { kind: 'box', id: own };
    if (arrows.some((a) => a.id === own)) return { kind: 'arrow', id: own };
    return null;
  }, [boxes, arrows]);

  // ---- the pointer: hover, click, drag to pan, wheel and pinch to zoom
  const drag = useRef(null);
  const pointers = useRef(new Map());
  const zoomAt = useCallback((factor, cx, cy) => {
    const v = vbRef.current;
    if (!v) return;
    const s = clamp(v.s * factor, MIN_SCALE, MAX_SCALE);
    setVb({ x: v.x + cx / v.s - cx / s, y: v.y + cy / v.s - cy / s, s });
  }, []);
  const local = (e) => {
    const r = paneRef.current.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };
  const showTip = (hit, e) => {
    const tip = tipRef.current;
    if (!tip) return;
    const text = hit ? tipFor(hit) : '';
    if (!text) {
      tip.hidden = true;
      return;
    }
    tip.textContent = text;
    tip.hidden = false;
    const [x, y] = local(e);
    const w = paneRef.current.clientWidth;
    const h = paneRef.current.clientHeight;
    tip.style.left = `${Math.max(4, Math.min(x + 14, w - tip.offsetWidth - 4))}px`;
    tip.style.top = `${y + 18 + tip.offsetHeight > h ? Math.max(4, y - tip.offsetHeight - 10) : y + 18}px`;
  };
  const onPointerDown = (e) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    pointers.current.set(e.pointerId, local(e));
    try {
      paneRef.current.setPointerCapture(e.pointerId);
    } catch { /* a synthetic event has no pointer to capture */ }
    const v = vbRef.current;
    if (pointers.current.size === 1) drag.current = { at: local(e), vb: v, moved: false, target: e.target };
    else if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()];
      drag.current = { pinch: Math.hypot(a[0] - b[0], a[1] - b[1]) || 1, vb: v, mid: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], moved: true };
    }
  };
  const onPointerMove = (e) => {
    const d = drag.current;
    if (pointers.current.has(e.pointerId)) pointers.current.set(e.pointerId, local(e));
    if (d && d.pinch && pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()];
      const s = clamp(d.vb.s * (Math.hypot(a[0] - b[0], a[1] - b[1]) / d.pinch), MIN_SCALE, MAX_SCALE);
      const [cx, cy] = d.mid;
      setVb({ x: d.vb.x + cx / d.vb.s - cx / s, y: d.vb.y + cy / d.vb.s - cy / s, s });
      handsOff.current = performance.now();
      return;
    }
    if (d && d.at) {
      const [x, y] = local(e);
      const dx = x - d.at[0];
      const dy = y - d.at[1];
      if (!d.moved && Math.hypot(dx, dy) < CLICK_SLOP) return;
      d.moved = true;
      cancelAnimationFrame(glide.current);
      handsOff.current = performance.now();
      setVb({ x: d.vb.x - dx / d.vb.s, y: d.vb.y - dy / d.vb.s, s: d.vb.s });
      showTip(null, e);
      return;
    }
    if (e.pointerType !== 'mouse') return;
    const hit = hitOf(e.target);
    const key = hit ? `${hit.kind}:${hit.id}` : '';
    if (key !== hoverRef.current) {
      hoverRef.current = key;
      lightUp(hit ? (hit.kind === 'mark' ? { kind: 'box', id: hit.id } : hit) : null);
    }
    showTip(hit, e);
  };
  const onPointerUp = (e) => {
    const d = drag.current;
    pointers.current.delete(e.pointerId);
    if (pointers.current.size === 0) drag.current = null;
    if (!d || d.moved || d.pinch) return;
    const hit = hitOf(d.target);
    if (!hit) return;
    if (hit.kind === 'mark') {
      toggle(hit.id);
      return;
    }
    if (hit.kind === 'arrow') {
      const a = arrows.find((x) => x.id === hit.id);
      if (a) go(a.to, { center: false });
      return;
    }
    handsOff.current = performance.now();
    if (isMp(hit.id)) {
      // the closed box opens; the fork or the join closes it; a member is picked
      const node = mpOf(hit.id);
      if (hit.id === node || hit.id.endsWith('/fork') || hit.id.endsWith('/join')) toggle(node);
      setSel(hit.id.includes('/m:') ? hit.id : node);
      return;
    }
    setSel(hit.id);
    setSay(sayOf(hit.id));
  };
  const onLeave = () => {
    hoverRef.current = null;
    lightUp(null);
    if (tipRef.current) tipRef.current.hidden = true;
  };
  useEffect(() => {
    // the wheel: a mouse's wheel or a trackpad's pinch zooms; two fingers on a trackpad move the picture
    const el = paneRef.current;
    if (!el) return undefined;
    const onWheel = (e) => {
      e.preventDefault();
      handsOff.current = performance.now();
      cancelAnimationFrame(glide.current);
      const r = el.getBoundingClientRect();
      const [x, y] = [e.clientX - r.left, e.clientY - r.top];
      const notch = e.deltaMode === 1 || (Math.abs(e.deltaY) >= NOTCH && e.deltaX === 0 && Number.isInteger(e.deltaY));
      if (e.ctrlKey || e.metaKey || notch) {
        const k = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
        zoomAt(Math.exp(-clamp(k, -120, 120) * (e.ctrlKey ? 0.01 : 0.002)), x, y);
      } else {
        const v = vbRef.current;
        if (v) setVb({ x: v.x + e.deltaX / v.s, y: v.y + e.deltaY / v.s, s: v.s });
      }
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [zoomAt]);
  // a new drawing: nothing lit, no tooltip about the old one
  useEffect(() => {
    hoverRef.current = null;
    lightUp(null);
    if (tipRef.current) tipRef.current.hidden = true;
  }, [chart, lightUp]);

  const zoomBy = (f) => {
    handsOff.current = performance.now();
    zoomAt(f, pane.w / 2, pane.h / 2);
  };
  const fitAll = () => {
    handsOff.current = performance.now();
    const f = fitView();
    if (f) moveTo(f);
  };

  // ---- the keyboard: from box to box by direction
  const stepTo = (dir) => {
    const from = sel && boxes.has(sel) ? sel : sel && boxes.has(defOf(sel)) ? defOf(sel) : readerBox || MAIN;
    const r0 = rectOf(from);
    if (!r0) return;
    const [cx, cy] = [r0.x + r0.w / 2, r0.y + r0.h / 2];
    let best = null;
    let bestScore = Infinity;
    for (const id of boxes.keys()) {
      if (id === from) continue;
      const r = rectOf(id);
      const [dx, dy] = [r.x + r.w / 2 - cx, r.y + r.h / 2 - cy];
      const along = dir === 'right' ? dx : dir === 'left' ? -dx : dir === 'down' ? dy : -dy;
      const across = dir === 'right' || dir === 'left' ? Math.abs(dy) : Math.abs(dx);
      if (along <= 1) continue;
      const score = along + 2.5 * across;
      if (score < bestScore) {
        bestScore = score;
        best = id;
      }
    }
    if (best) go(best, { center: false });
  };
  const onKey = (e) => {
    if (e.target !== paneRef.current || e.metaKey || e.ctrlKey || e.altKey) return;
    if (tourOn && (e.key === 'ArrowRight' || e.key === 'ArrowLeft' || e.key === ' ' || e.key === 'Escape')) return; // the walk's keys
    const dirs = { ArrowRight: 'right', ArrowLeft: 'left', ArrowDown: 'down', ArrowUp: 'up' };
    if (dirs[e.key]) {
      e.preventDefault();
      stepTo(dirs[e.key]);
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      const d = sel ? defOf(sel) : null;
      if (sel && isMp(sel)) toggle(mpOf(sel));
      else if (d && boxes.has(d) && (model.callees.get(d) || []).length) toggle(d);
    } else if (e.key === '+' || e.key === '=') zoomBy(ZOOM_STEP);
    else if (e.key === '-' || e.key === '_') zoomBy(1 / ZOOM_STEP);
    else if (e.key === '0') fitAll();
    else if (e.key === 'Escape') setSel(null);
    else if (e.key === '/') {
      e.preventDefault();
      document.getElementById('pg-search')?.focus();
    }
  };

  // ---- search: any def the run called, or a step of one
  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    const out = [];
    for (const d of model.ran) {
      const cands = [[d, d]];
      for (const n of model.inner.get(d) || []) if (n.kind === 'step') cands.push([n.id, `${n.id} ${n.text || (bundle.steps || {})[n.id] || ''}`]);
      for (const [id, text] of cands) {
        const at = text.toLowerCase().indexOf(q);
        if (at < 0) continue;
        out.push({ id, score: (at === 0 ? 0 : 1) + (id.toLowerCase().includes(q) ? 0 : 2) + id.length / 1000 });
      }
    }
    return out.sort((a, b) => a.score - b.score).slice(0, MAX_RESULTS);
  }, [query, model, bundle]);
  const pickResult = (id) => {
    setQuery('');
    setActive(0);
    go(id);
    paneRef.current?.focus({ preventScroll: true });
  };
  const onSearchKey = (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActive((a) => Math.min(a + 1, results.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActive((a) => Math.max(a - 1, 0));
    } else if (e.key === 'Enter' && results[active]) {
      e.preventDefault();
      pickResult(results[active].id);
    } else if (e.key === 'Escape') setQuery('');
  };

  if (!chart || !bundle.drawLib) return null;
  const outlines = [];
  if (nearBox) outlines.push({ id: nearBox, kind: 'near' });
  if (tourStop && !peek && tourStop.box !== defOf(tourStop.box) && boxes.has(defOf(tourStop.box))) outlines.push({ id: defOf(tourStop.box), kind: 'near' });
  if (sel && boxes.has(sel)) outlines.push({ id: sel, kind: 'sel' });
  else if (sel && boxes.has(defOf(sel))) outlines.push({ id: defOf(sel), kind: 'sel' });
  const v = vb || { x: 0, y: 0, s: 1 };
  const viewBox = `${v.x.toFixed(2)} ${v.y.toFixed(2)} ${(pane.w / v.s).toFixed(2)} ${(pane.h / v.s).toFixed(2)}`;
  return (
    <section className="pg" id="rv-graph" aria-label="The program as a graph" data-shown={shown.size} data-ran={model.ran.length}
      data-redraw-ms={redraw ? redraw.ms : ''} data-draws={redraw ? redraw.n : 0} data-busy={busy ? '1' : ''} data-scale={v.s.toFixed(3)} data-sel={sel || ''}>
      <div className="pg-head">
        <h2 className="ex-h">The program as a graph</h2>
        <p className="ex-quiet">
          One box per def {ix.bundle.calls ? 'the run called' : 'its text calls'} ({shown.size} of {model.ran.length} drawn), with its steps inside; <b>+n</b> on a box opens the n defs it
          calls. Drawn by <code>lib/graphdraw.viv</code> in your browser.
        </p>
      </div>
      <div className="pg-split" ref={splitRef} style={{ gridTemplateColumns: `minmax(0, ${frac}fr) 12px minmax(300px, ${1 - frac}fr)` }} data-tour={tourStop ? tourStop.k : ''} data-tour-kind={tourStop ? tourStop.kind : ''} data-playing={playing ? '1' : ''}>
        <div className={`pg-left${tourOn ? ' touring' : ''}`}>
          {run && (
            <div className={`pg-tour${tourOn ? ' on' : ''}`} role="group" aria-label="Walk through the run">
              {!tourOn ? (
                <>
                  <button type="button" className="pg-btn pg-run" onClick={startTour}>▶ Run</button>
                  <span className="ex-quiet pg-tour-say" title="Walk through the recorded run, stop by stop: the picture follows it and the panel says what is happening.">Walk through the recorded run, stop by stop: the picture follows it and the panel says what is happening.</span>
                </>
              ) : (
                <>
                  <button type="button" className="pg-btn pg-tback" onClick={() => { setPlaying(false); tourMove(-1); }} disabled={!tourStop || tourStop.k === 0} aria-label="Back one stop" title="Back (←)">◀ Back</button>
                  <button type="button" className="pg-btn pg-run" onClick={() => (playing ? setPlaying(false) : tourStop && tourStop.k >= stops.length - 1 ? startTour() : (setPeek(null), setPlaying(true)))} aria-pressed={playing} title="Play or pause (space)">
                    {playing ? '⏸ Pause' : tourStop && tourStop.k >= stops.length - 1 ? '▶ Again' : '▶ Play'}
                  </button>
                  <button type="button" className="pg-btn pg-tnext" onClick={() => { setPlaying(false); tourMove(1); }} disabled={!tourStop || tourStop.k >= stops.length - 1} aria-label="Next stop" title="Next (→)">Next ▶</button>
                  <label className="pg-speed">
                    <span className="ex-quiet">speed</span>
                    <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
                      {TOUR_SPEEDS.map((x) => <option key={x} value={x}>{x}×</option>)}
                    </select>
                  </label>
                  <input type="range" className="pg-scrub" min={0} max={Math.max(0, stops.length - 1)} value={tourStop ? tourStop.k : 0} aria-label="Stop"
                    onChange={(e) => { setPlaying(false); setPeek(null); setTourK(Number(e.target.value)); }} />
                  <span className="pg-count ex-quiet" aria-live="off">{tourStop ? `${tourStop.k + 1} / ${stops.length}` : ''}</span>
                  <button type="button" className="pg-btn pg-quiet pg-texit" onClick={exitTour} title="Leave the walk (Esc)">Exit</button>
                </>
              )}
            </div>
          )}
          <div className="pg-tools">
            <div className="pg-search">
              <input id="pg-search" type="search" placeholder="Find a def or a step" value={query} autoComplete="off" spellCheck={false}
                role="combobox" aria-expanded={results.length > 0} aria-autocomplete="list" aria-label="Find a def or a step" aria-controls="pg-results"
                aria-activedescendant={results[active] ? `pg-r-${active}` : undefined}
                onChange={(e) => { setQuery(e.target.value); setActive(0); }} onKeyDown={onSearchKey} />
              {results.length > 0 && (
                <ul id="pg-results" role="listbox" className="pg-results" aria-label="Defs and steps found">
                  {results.map((r, k) => (
                    <li key={r.id} id={`pg-r-${k}`} role="option" aria-selected={k === active} className={k === active ? 'on' : ''}
                      onMouseDown={(e) => { e.preventDefault(); pickResult(r.id); }}>
                      <code>{r.id}</code>
                      {r.id.includes('/') && <span className="ex-quiet"> {model.byId.get(r.id)?.text || (bundle.steps || {})[r.id] || ''}</span>}
                      {!shown.has(defOf(r.id)) && <span className="pg-tag">not drawn yet</span>}
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <div className="pg-btns" role="group" aria-label="Zoom">
              <button type="button" className="pg-zoomout" onClick={() => zoomBy(1 / ZOOM_STEP)} aria-label="Zoom out" title="Zoom out (−)">−</button>
              <button type="button" className="pg-zoomin" onClick={() => zoomBy(ZOOM_STEP)} aria-label="Zoom in" title="Zoom in (+)">+</button>
              <button type="button" className="pg-fit" onClick={fitAll} title="The whole picture (0)">Fit</button>
            </div>
            <div className="pg-btns" role="group" aria-label="How many defs to draw">
              <button type="button" className="pg-all" onClick={() => redrawTo(new Set(model.ran), sel ? defOf(sel) : MAIN)} disabled={busy || shown.size === model.ran.length} title="Draw every def the run called">All</button>
              <button type="button" className="pg-reset" onClick={() => redrawTo(first, sel && shownOf(model, first).has(defOf(sel)) ? defOf(sel) : MAIN)} disabled={busy || keyOf(open) === keyOf(first)} title="Back to the first view">Reset</button>
            </div>
          </div>
          <div className="pg-canvas" ref={paneRef} tabIndex={0} role="application"
            aria-label="The program graph. Arrow keys move from box to box and show its code; Enter opens or closes a def's calls; plus and minus zoom; 0 shows the whole picture; slash finds a def."
            onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp} onPointerLeave={onLeave} onKeyDown={onKey}
            onDoubleClick={(e) => { const h = hitOf(e.target); if (h && h.kind === 'box' && !h.id.includes('/') && !isMp(h.id)) toggle(h.id); }}>
            {pane.w > 0 && vb && <DiagramView chart={chart} viewBox={viewBox} highlight={readerBox ? [readerBox] : []} outlines={outlines} hits svgRef={svgRef} />}
            <div className="pg-tip" ref={tipRef} hidden role="tooltip" />
            <p className="pg-stamp" aria-live="off">{busy ? 'drawing…' : redraw ? `drawn in ${redraw.ms} ms` : ''}</p>
            {err && <p className="pg-err" role="alert">The picture could not be redrawn: {err}</p>}
          </div>
          <p className="pg-live" aria-live="polite">{say}</p>
        </div>
        <div className={`pg-divider${dragging ? ' on' : ''}`} role="separator" aria-orientation="vertical" aria-label="Divide the width between the picture and the panel"
          aria-valuemin={SPLIT_MIN * 100} aria-valuemax={SPLIT_MAX * 100} aria-valuenow={Math.round(frac * 100)} tabIndex={0}
          title="Drag to give the picture or the panel more room; double-click for the default"
          onPointerDown={onDividerDown} onPointerMove={onDividerMove} onPointerUp={onDividerUp} onPointerCancel={onDividerUp}
          onKeyDown={onDividerKey} onDoubleClick={() => keepSplit(null)} />
        <aside className="pg-panel" aria-label="What you picked in the graph">
          <GraphPanel ix={ix} katex={katex} model={model} shown={shown} open={open} sel={sel} go={go} toggle={toggle} onRead={onRead} reduced={reduced}
            tour={tourStop ? { stop: tourStop, stops, run, peek, peekAt, back: () => { setPeek(null); tourGo.current(boxFor(model, tourStop.box, tourStop.node), { center: false }); }, openPass, playing } : null} />
        </aside>
      </div>
    </section>
  );
}
