// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The code map of one vipyr function (/vipyr/map/<function>): every def of the
// Viv program behind it that its entry reaches, grouped by the module it is
// written in, with the calls between them. The data is the compiler's
// (`viv graph --map`, lang/viv/src/codemap.rs), written by `python -m
// vipyr.docs` into public/codemap/<key>.json. Where the explorer has recorded
// a run of the same library (public/explore/<slug>.calls.json), the calls it
// counted are shown too.
//
// The picture: one column per call distance from the entry (the entry's
// module on the left), a region per module, a tile per def whose width grows
// with its length in lines. Click a tile (or a call in the source) to read a
// def; the route from the entry to it is drawn through the regions. Only the
// def's own calls (or a hovered tile's) are drawn one by one, routed through
// the gutters (lib/codemap.js, callPath); the calls between two modules run
// as one trunk behind the regions. "All calls" (key a) draws every call.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { highlight } from '../lib/highlight.js';
import { intentOf, intentText, layoutMap, makeRouter, orthoPath, routeOf, trunks } from '../lib/codemap.js';
import { renderProse, texToPlain } from '../lib/richText.js';

import { REPO_PUBLIC, GITHUB } from '../lib/repo.js';
const GH = `${GITHUB}/blob/main/`;

/** An intent's words, its math typeset when a KaTeX is handed in (`viv
 *  atlas`'s page bundles one); without one, the text as it is written. */
export function Intent({ text, katex, as: Tag = 'span', className }) {
  const html = useMemo(() => (katex ? renderProse(text || '', katex) : null), [text, katex]);
  return html == null ? <Tag className={className}>{text}</Tag> : <Tag className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}

/**
 * A map as the reader sees it: a contract's def has the parameters the reader
 * writes (the compiler's own `viv__` method selector and settings are not
 * among them) and the contract's intent when the def has none of its own.
 */
function forReader(map) {
  if (!map) return map;
  const intentOfContract = new Map((map.contracts || []).map((c) => [c.name, c.intent || '']));
  const defs = map.defs.map((d) => ({
    ...d,
    params: (d.params || []).filter((p) => !p.startsWith('viv__')),
    intent: (d.intent || '').trim() ? d.intent : intentOfContract.get(d.id) || d.intent,
  }));
  return { ...map, defs };
}

function useMap(mapKey, explore, given) {
  const [state, setState] = useState(() => (given ? { map: forReader(given.map), runs: given.runs || null, error: null } : { map: null, runs: null, error: null }));
  useEffect(() => {
    // a map handed in whole (`viv atlas`'s offline file) is not fetched
    if (given) return undefined;
    let live = true;
    fetch(`/codemap/${mapKey}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`the map file /codemap/${mapKey}.json did not load (${r.status})`))))
      .then(async (map) => {
        let runs = null;
        if (explore) {
          try {
            const r = await fetch(`/explore/${explore}.calls.json`);
            if (r.ok) runs = await r.json();
          } catch {
            runs = null;
          }
        }
        if (live) setState({ map: forReader(map), runs, error: null });
      })
      .catch((e) => live && setState({ map: null, runs: null, error: String(e.message || e) }));
    return () => {
      live = false;
    };
  }, [mapKey, explore, given]);
  return state;
}

const r1 = (v) => Math.round(v * 10) / 10;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

/** Highlighted lines of a def's source, each def name of the map made a link. */
function sourceLines(text, first, last, names) {
  const all = text.split('\n');
  const out = [];
  for (let n = first; n <= last; n++) {
    const raw = all[n - 1] ?? '';
    // the line's own indent becomes padding, so a wrapped line hangs under
    // its first character, a step further in
    const indent = raw.length - raw.trimStart().length;
    let html = highlight(raw.trimStart(), 'viv');
    // link def names in the text parts of the highlighted line
    html = html.replace(/(^|>)([^<]+)/g, (m, open, txt) => open + txt.replace(/(\bstep\s+)?\b([A-Za-z_]\w*)\b/g, (all, step, w) => (!step && names.has(w) ? `<a class="cm-ref" data-def="${esc(w)}" href="#${esc(w)}">${w}</a>` : all)));
    out.push({ n, html, indent, step: /^\s*#:/.test(raw), doc: /^\s*#(?![!:|])/.test(raw) });
  }
  return out;
}

/** Line ranges sorted, the overlapping and adjacent ones joined. */
function mergeSpans(spans) {
  const out = [];
  for (const [a, b] of spans.slice().sort((x, y) => x[0] - y[0])) {
    const last = out[out.length - 1];
    if (last && a <= last[1] + 1) last[1] = Math.max(last[1], b);
    else out.push([a, b]);
  }
  return out;
}

// a map with more defs than this opens with the modules far from the entry
// folded: the entry's module and those of the defs it calls stay open
const FOLD_ABOVE = 80;
// how much further in a wrapped source line continues than its first line
const HANG = 4;

const secs = (x) => (x == null ? '' : x < 0.01 ? `${(x * 1000).toFixed(1)} ms` : x < 10 ? `${x.toFixed(2)} s` : `${x.toFixed(0)} s`);
const sci = (x) => (x == null ? '' : x.toExponential(0).replace('e-', 'e−'));

// a contract's methods under their group labels, in the order listed; the
// methods outside any group come first, with no label
const groupsOf = (c) => {
  const grouped = new Set((c.groups || []).flatMap((g) => g.methods));
  const free = c.methods.filter((m) => !grouped.has(m));
  return [...(free.length ? [{ label: null, methods: free }] : []), ...(c.groups || [])];
};

// A contract: what it computes, its methods (the
// default first), what they must agree on, and what its sweep measured
function ContractPanel({ contract, cur, defs, go, katex }) {
  const methods = contract.methods;
  const rows = contract.sweep?.rows || [];
  return (
    <div className="cm-contract">
      <p className="cm-contract-head">
        <span>Contract</span> <code>{contract.name}</code> <Intent text={contract.intent} katex={katex} />
      </p>
      {groupsOf(contract).map((g) => (
        <div key={g.label || '-'}>
          {g.label && <p className="cm-contract-group">{g.label}</p>}
          <ul>
            {g.methods.map((m) => (
              <li key={m} className={m === cur ? 'is-cur' : ''}>
                <button type="button" onClick={() => go(m)} disabled={m === cur || !defs.has(m)}>
                  {m}
                </button>
                {m === contract.default && <small> default</small>}
                {m !== cur && <Intent text={(defs.get(m)?.intent || '').split(/(?<=\.) /)[0]} katex={katex} />}
              </li>
            ))}
          </ul>
        </div>
      ))}
      {contract.agree.length > 0 && (
        <p className="cm-contract-note">
          <code>using every method</code> runs them all
          {(contract.groups || []).map((g) => (
            <span key={g.label}>
              , <code>using every {g.label} method</code> the {g.label} ones
            </span>
          ))}
          ; a run stops when one differs from the first by more than the contract allows:{' '}
          {contract.agree.map((a, i) => (
            <span key={a.output}>
              {i > 0 && ', '}
              <code>{a.output}</code> {a.within == null ? 'exactly' : <>within <code>{a.within}</code></>}
            </span>
          ))}
          .
        </p>
      )}
      {rows.length > 0 && (
        <table className="cm-contract-sweep">
          <thead>
            <tr>
              <th>case</th>
              <th>dim</th>
              {methods.map((m) => <th key={m}>{m}</th>)}
              <th>max |Δ|</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.case}>
                <td>{r.case}</td>
                <td>{r.dim.toLocaleString('en-US')}</td>
                {methods.map((m) => <td key={m}>{r[m] ? secs(r[m].seconds) : '–'}</td>)}
                <td>{sci(r.agreement)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {rows.length > 0 && (
        <p className="cm-contract-note">
          Wall time of the whole run; max |Δ| is the largest difference between the methods' energies. From{' '}
          {REPO_PUBLIC ? <a href={`${GH}${contract.sweep.file}`} rel="noopener" target="_blank">{contract.sweep.file.split('/').pop()}</a> : contract.sweep.file.split('/').pop()}.
        </p>
      )}
    </div>
  );
}

// `data` ({ map, runs }) hands the map in instead of fetching it, and `label`
// names it in the bar instead of the link back to a vipyr function: the page
// `viv atlas` writes (lang/viv/src/atlas.rs) does both, and its counts are
// from its own run (`runLabel`).
export default function CodeMap({ fn, mapKey, explore, exploreTitle, refHref, data, label, runLabel, katex = null }) {
  const { map, runs, error } = useMap(mapKey, explore, data);
  const ranIn = runLabel || "the explorer's run";
  const [sel, setSel] = useState(null);
  const [collapsed, setCollapsed] = useState(() => new Set());
  const [query, setQuery] = useState('');
  const [view, setView] = useState({ x: 0, y: 0, k: 1 });
  const [whole, setWhole] = useState(false);
  const [legend, setLegend] = useState(false);
  const back = useRef([]);
  const fwd = useRef([]);
  const [, bump] = useState(0);
  const svgRef = useRef(null);
  const paneRef = useRef(null);
  const searchRef = useRef(null);
  const drag = useRef(null);

  const defs = useMemo(() => (map ? new Map(map.defs.map((d) => [d.id, d])) : new Map()), [map]);
  // def -> the contract it is a method of, or is
  const contractOf = useMemo(() => {
    const m = new Map();
    for (const c of map?.contracts || []) {
      m.set(c.name, c);
      for (const d of c.methods) m.set(d, c);
    }
    return m;
  }, [map]);
  const names = useMemo(() => new Set(defs.keys()), [defs]);
  // the run's counts: calls into each def, and per edge
  const heat = useMemo(() => {
    if (!runs) return null;
    const into = new Map();
    const edge = new Map();
    for (const [a, b, c] of runs.calls || []) {
      if (!defs.has(b)) continue;
      into.set(b, (into.get(b) || 0) + c);
      edge.set(`${a}\u0000${b}`, c);
    }
    if (!into.size) return null;
    return { into, edge, max: Math.max(...into.values()) };
  }, [runs, defs]);
  useEffect(() => {
    if (!map || map.defs.length <= FOLD_ABOVE) return;
    const open = new Set(map.defs.filter((d) => d.dist <= 1).map((d) => d.module));
    // and the module of the def the address names
    const named = map.defs.find((d) => d.id === decodeURIComponent(location.hash.slice(1)));
    if (named) open.add(named.module);
    setCollapsed(new Set(map.modules.map((_, k) => k).filter((k) => !open.has(k))));
  }, [map]);
  const lay = useMemo(() => (map ? layoutMap(map, collapsed) : null), [map, collapsed]);
  const router = useMemo(() => (lay ? makeRouter(lay) : null), [lay]);
  const trunkList = useMemo(() => (lay ? trunks(lay, router) : []), [lay, router]);
  const [allCalls, setAllCalls] = useState(false);
  const [hover, setHover] = useState(null);

  // the def in the URL's hash, else the entry
  useEffect(() => {
    if (!map) return;
    const pick = () => {
      const h = decodeURIComponent(location.hash.slice(1));
      setSel(defs.has(h) ? h : map.entry);
    };
    pick();
    window.addEventListener('popstate', pick);
    return () => window.removeEventListener('popstate', pick);
  }, [map, defs]);

  const fit = useCallback(() => {
    const svg = svgRef.current;
    if (!svg || !lay) return;
    const r = svg.getBoundingClientRect();
    const k = Math.min(1.2, Math.max(0.15, Math.min((r.width - 76) / lay.width, (r.height - 40) / lay.height)));
    setView({ k, x: Math.max(56, (r.width - lay.width * k) / 2), y: Math.max(20, (r.height - lay.height * k) / 2) });
  }, [lay]);
  // first view: the whole map when it is legible at that size, else the
  // selected def at reading size
  const first = useRef(true);
  useEffect(() => {
    if (!lay || !sel || !first.current) return;
    first.current = false;
    const svg = svgRef.current;
    const r = svg.getBoundingClientRect();
    const k = Math.min((r.width - 40) / lay.width, (r.height - 40) / lay.height);
    if (k >= 0.9) return fit();
    const t = lay.tiles.get(sel);
    if (!t) return fit();
    const kk = 0.95;
    setView({ k: kk, x: Math.min(40, r.width / 3 - (t.x + t.w / 2) * kk), y: r.height / 2 - (t.y + t.h / 2) * kk });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lay, sel]);

  const centerOn = useCallback(
    (id) => {
      const t = lay && lay.tiles.get(id);
      const svg = svgRef.current;
      if (!t || !svg) return;
      const r = svg.getBoundingClientRect();
      setView((v) => {
        const sx = v.x + (t.x + t.w / 2) * v.k;
        const sy = v.y + (t.y + t.h / 2) * v.k;
        if (sx > 60 && sx < r.width - 60 && sy > 40 && sy < r.height - 40) return v;
        const k = Math.max(v.k, 0.6);
        return { k, x: r.width / 2 - (t.x + t.w / 2) * k, y: r.height / 2 - (t.y + t.h / 2) * k };
      });
    },
    [lay],
  );

  // the pane scrolls (on a wide screen) so the def's first line is in view:
  // from the top when the header and that line fit, else with the line a
  // little below the pane's top
  const showSignature = useCallback(
    (id) => {
      const pane = paneRef.current;
      const d = defs.get(id);
      if (!pane || !d || pane.scrollHeight <= pane.clientHeight + 1) return;
      if (getComputedStyle(pane).overflowY === 'visible') return;
      const el = pane.querySelector(`[data-line="${d.main ? '' : d.line}"]`) || pane.querySelector('.cm-code [data-line]');
      if (!el) return;
      const top = el.offsetTop - pane.offsetTop;
      const lineH = el.offsetHeight || 20;
      pane.scrollTop = top + lineH * 2 <= pane.clientHeight ? 0 : Math.max(0, top - pane.clientHeight * 0.25);
    },
    [defs],
  );
  const go = useCallback(
    (id, how = 'push') => {
      if (!defs.has(id) || id === sel) return;
      if (how === 'push') {
        if (sel) back.current.push(sel);
        fwd.current = [];
      }
      setSel(id);
      setWhole(false);
      const d = defs.get(id);
      if (d && collapsed.has(d.module)) {
        const c = new Set(collapsed);
        c.delete(d.module);
        setCollapsed(c);
      }
      try {
        history.replaceState(null, '', `#${encodeURIComponent(id)}`);
      } catch {
        /* a sandboxed frame */
      }
      bump((x) => x + 1);
      requestAnimationFrame(() => centerOn(id));
    },
    [defs, sel, collapsed, centerOn],
  );
  useEffect(() => {
    if (sel) showSignature(sel);
  }, [sel, showSignature]);
  const goBack = useCallback(() => {
    const p = back.current.pop();
    if (p) {
      fwd.current.push(sel);
      go(p, 'none');
    }
  }, [go, sel]);
  const goFwd = useCallback(() => {
    const n = fwd.current.pop();
    if (n) {
      back.current.push(sel);
      go(n, 'none');
    }
  }, [go, sel]);

  const cur = sel ? defs.get(sel) : null;
  useEffect(() => {
    if (whole && cur) paneRef.current?.querySelector(`[data-line="${cur.line}"]`)?.scrollIntoView({ block: 'start' });
  }, [whole, cur]);
  const route = useMemo(() => (cur ? routeOf(defs, cur.id) : []), [cur, defs]);
  const callers = useMemo(() => {
    if (!cur || !map) return [];
    return map.defs.filter((d) => d.calls.some(([to]) => to === cur.id)).map((d) => d.id);
  }, [cur, map]);
  const callees = useMemo(() => {
    if (!cur) return [];
    const by = new Map();
    for (const [to, line, depth, value] of cur.calls) {
      const o = by.get(to) || { to, sites: 0, loop: 0, value: true, line };
      o.sites++;
      o.loop = Math.max(o.loop, depth);
      o.value = o.value && !!value;
      by.set(to, o);
    }
    return [...by.values()];
  }, [cur]);

  // keys: s or / search, [ and ] (or Alt+arrows) back and forward, u the
  // caller on the route, e the entry, j and k the next and previous callee,
  // 0 fit, c collapse the selected def's module
  useEffect(() => {
    const onKey = (ev) => {
      const t = ev.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) {
        if (ev.key === 'Escape') t.blur();
        return;
      }
      if (ev.metaKey || ev.ctrlKey) return;
      const k = ev.key;
      if (k === 's' || k === '/') {
        ev.preventDefault();
        ev.stopPropagation();
        searchRef.current?.focus();
      } else if (k === '[' || (ev.altKey && k === 'ArrowLeft')) goBack();
      else if (k === ']' || (ev.altKey && k === 'ArrowRight')) goFwd();
      else if (k === 'u' && cur?.from) go(cur.from);
      else if (k === 'e' && map) go(map.entry);
      else if ((k === 'j' || k === 'k') && callees.length) {
        // step through the selected def's callees; the last one visited is
        // remembered on the def being read
        const list = callees.map((c) => c.to);
        go(list[k === 'j' ? 0 : list.length - 1]);
      } else if (k === '0') fit();
      else if (k === 'a') setAllCalls((x) => !x);
      else if (k === 'c' && cur && !cur.main) {
        const c = new Set(collapsed);
        c.has(cur.module) ? c.delete(cur.module) : c.add(cur.module);
        setCollapsed(c);
      } else return;
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [goBack, goFwd, go, cur, map, callees, fit, collapsed]);

  // pan with a drag, zoom with the wheel or a pinch
  const pointers = useRef(new Map());
  const onDown = (ev) => {
    if (ev.target.closest('[data-tile],[data-region]')) return;
    ev.currentTarget.setPointerCapture(ev.pointerId);
    pointers.current.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
    drag.current = { x: ev.clientX, y: ev.clientY, v: view, pinch: null };
  };
  const onMove = (ev) => {
    if (!drag.current || !pointers.current.has(ev.pointerId)) return;
    pointers.current.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
    const ps = [...pointers.current.values()];
    if (ps.length === 2) {
      const dist = Math.hypot(ps[0].x - ps[1].x, ps[0].y - ps[1].y);
      if (!drag.current.pinch) drag.current.pinch = { dist, v: view };
      const r = svgRef.current.getBoundingClientRect();
      const cx = (ps[0].x + ps[1].x) / 2 - r.left;
      const cy = (ps[0].y + ps[1].y) / 2 - r.top;
      const p = drag.current.pinch;
      const k = Math.min(3, Math.max(0.1, (p.v.k * dist) / p.dist));
      setView({ k, x: cx - ((cx - p.v.x) * k) / p.v.k, y: cy - ((cy - p.v.y) * k) / p.v.k });
      return;
    }
    const d = drag.current;
    setView({ ...d.v, x: d.v.x + ev.clientX - d.x, y: d.v.y + ev.clientY - d.y });
  };
  const onUp = (ev) => {
    pointers.current.delete(ev.pointerId);
    if (!pointers.current.size) drag.current = null;
  };
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const onWheel = (ev) => {
      ev.preventDefault();
      const r = svg.getBoundingClientRect();
      const cx = ev.clientX - r.left;
      const cy = ev.clientY - r.top;
      setView((v) => {
        const k = Math.min(3, Math.max(0.1, v.k * Math.exp(-ev.deltaY * (ev.ctrlKey ? 0.01 : 0.0015))));
        return { k, x: cx - ((cx - v.x) * k) / v.k, y: cy - ((cy - v.y) * k) / v.k };
      });
    };
    svg.addEventListener('wheel', onWheel, { passive: false });
    return () => svg.removeEventListener('wheel', onWheel);
  }, [lay]);
  const zoomBy = (f) => {
    const r = svgRef.current.getBoundingClientRect();
    setView((v) => {
      const k = Math.min(3, Math.max(0.1, v.k * f));
      return { k, x: r.width / 2 - ((r.width / 2 - v.x) * k) / v.k, y: r.height / 2 - ((r.height / 2 - v.y) * k) / v.k };
    });
  };

  const hits = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q || !map) return [];
    // a def named exactly what was typed first, then names that start with
    // it, so Enter on `lmder` opens lmder and not lmder1_maxfev
    const rank = (d) => (d.id.toLowerCase() === q ? 0 : d.id.toLowerCase().startsWith(q) ? 1 : 2);
    return map.defs
      .filter((d) => d.id.toLowerCase().includes(q) || (map.modules[d.module]?.path || '').toLowerCase().includes(q))
      .map((d, i) => [rank(d), i, d])
      .sort((a, b) => a[0] - b[0] || a[1] - b[1])
      .map((x) => x[2])
      .slice(0, 12);
  }, [query, map]);

  if (error) return <div className="cm-empty">{error}</div>;
  if (!map || !lay) return <div className="cm-empty">Loading the map of {label || `vp.${fn}`}…</div>;

  const onRoute = new Set(route);
  // the def whose calls are drawn (the one pointed at, else the one shown):
  // its callees and its callers, marked apart
  const focusCallees = new Set(lay.edges.filter(([a, b]) => a === (hover || sel) && b !== a).map(([, b]) => b));
  const focusCallers = new Set(lay.edges.filter(([a, b]) => b === (hover || sel) && b !== a).map(([a]) => a));
  const tileOf = (id) => lay.tiles.get(id);
  const anchor = (id) => {
    const t = tileOf(id);
    if (t) return t;
    const d = defs.get(id);
    return d ? lay.regions.get(d.module) : null;
  };
  // the calls drawn one by one: the selected (or hovered) def's own, the
  // route's, or every call when "All calls" is on; the rest travel as trunks
  const focus = hover || sel;
  const shown = lay.edges.filter(([a, b]) => allCalls || a === focus || b === focus || (onRoute.has(a) && onRoute.has(b) && route.indexOf(b) === route.indexOf(a) + 1));
  const mod = (k) => map.modules[k];
  const curMod = cur && !cur.main ? mod(cur.module) : mod(0);
  const lines = cur
    ? cur.main
      ? mergeSpans((cur.top || []).filter(([k]) => k === 0).map(([, a, b]) => [a, b]))
          .flatMap(([a, b], i, all) => {
            const ls = sourceLines(mod(0).src || '', a, b, names);
            // a gap between statements that are not next to each other
            if (i > 0 && a > all[i - 1][1] + 1) ls[0] = { ...ls[0], gap: true };
            return ls;
          })
      : sourceLines(curMod.src || '', whole ? 1 : Math.max(1, cur.line - (cur.doc ? cur.doc.split('\n').length : 0)), whole ? curMod.lines : cur.end, names)
    : [];
  const loopAt = new Map();
  if (cur) for (const l of cur.loops) loopAt.set(l.line, l);
  const callAt = new Map();
  if (cur) for (const [to, line, depth] of cur.calls) if (depth > 0) callAt.set(line, Math.max(callAt.get(line) || 0, depth));
  const builtins = cur ? Object.entries(cur.builtins).sort((a, b) => b[1] - a[1]) : [];
  const legendBuiltins = (() => {
    const tot = new Map();
    for (const d of map.defs) for (const [b, c] of Object.entries(d.builtins)) tot.set(b, (tot.get(b) || 0) + c);
    return [...tot.entries()].sort((a, b) => b[1] - a[1]);
  })();
  const nDefs = map.defs.filter((d) => !d.main).length;
  const nMods = lay.order.length;
  const mini = { w: 180, h: Math.max(40, Math.min(140, (180 * lay.height) / lay.width)) };
  const miniK = Math.min(mini.w / lay.width, mini.h / lay.height);
  const svgBox = svgRef.current?.getBoundingClientRect();

  return (
    <div className="cm">
      <div className="cm-bar">
        {label ? (
          <span className="cm-back-ref">
            <code>{label}</code>
          </span>
        ) : (
          <a className="cm-back-ref" href={refHref}>
            ← <code>vp.{fn}</code>
          </a>
        )}
        <span className="cm-count">
          {nDefs} defs in {nMods} {nMods === 1 ? 'module' : 'modules'}
          <span className="cm-of"> of {map.defsInProgram} in the program</span>
        </span>
        <div className="cm-search">
          <input
            ref={searchRef}
            type="search"
            placeholder="Find a def or module (s)"
            value={query}
            aria-label="Find a def or module"
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && hits[0]) {
                go(hits[0].id);
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
                      go(d.id);
                      setQuery('');
                    }}>
                    <code>{d.main ? 'top level' : d.id}</code>
                    <span>{mod(d.module).path}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="cm-nav">
          <button type="button" onClick={goBack} disabled={!back.current.length} aria-label="Back ([)" title="Back ([)">
            ‹
          </button>
          <button type="button" onClick={goFwd} disabled={!fwd.current.length} aria-label="Forward (])" title="Forward (])">
            ›
          </button>
          <button type="button" className="cm-all" onClick={() => setAllCalls((x) => !x)} aria-pressed={allCalls} title="Draw every call (a)">
            All calls
          </button>
          <button type="button" onClick={() => setLegend((x) => !x)} aria-expanded={legend}>
            Key
          </button>
        </div>
      </div>
      <nav className="cm-route" aria-label="The calls from the entry to the def shown">
        {route.map((id, i) => (
          <span key={id}>
            {i > 0 && <span className="cm-sep">›</span>}
            <button type="button" className={id === sel ? 'is-cur' : ''} onClick={() => go(id)}>
              {defs.get(id)?.main ? 'top level' : id}
            </button>
          </span>
        ))}
      </nav>
      <div className="cm-body">
        <div className="cm-stage">
          <svg
            ref={svgRef}
            className="cm-svg"
            role="application"
            aria-label={label ? `Atlas of ${label}` : `Map of the Viv code behind vp.${fn}`}
            onPointerDown={onDown}
            onPointerMove={onMove}
            onPointerUp={onUp}
            onPointerCancel={onUp}>
            <defs>
              {['out', 'in', 'route', 'all'].map((k) => (
                <marker key={k} id={`cm-arrow-${k}`} className={`cm-arrow cm-arrow-${k}`} viewBox="0 0 8 8" refX="7.5" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto">
                  <path d="M0.5,0.8 L7.5,4 L0.5,7.2 Z" />
                </marker>
              ))}
            </defs>
            <g transform={`translate(${view.x},${view.y}) scale(${view.k})`}>
              <g className={`cm-trunks${allCalls ? ' is-quiet' : ''}`}>
                {trunkList.map((t) => (
                  <path key={`${t.from}>${t.to}`} d={t.d} strokeWidth={r1(Math.min(3.5, 1 + 0.7 * Math.log2(1 + t.calls)))}>
                    <title>{`${mod(t.from).path} → ${mod(t.to).path}: ${t.calls} ${t.calls === 1 ? 'call' : 'calls'}`}</title>
                  </path>
                ))}
              </g>
              {lay.order.map((k) => {
                const r = lay.regions.get(k);
                const m = mod(k);
                const shut = collapsed.has(k);
                const fs = 12;
                return (
                  <g key={`r${k}`} className={`cm-region${curMod === m ? ' is-cur' : ''}`}>
                    <rect x={r.x} y={r.y} width={r.w} height={r.h} rx="6" />
                    <text
                      visibility={view.k < 0.6 ? 'hidden' : undefined}
                      x={r.x + 10}
                      y={r.y + 17}
                      fontSize={fs}
                      className="cm-region-name"
                      data-region={k}
                      onClick={() => {
                        const c = new Set(collapsed);
                        shut ? c.delete(k) : c.add(k);
                        setCollapsed(c);
                      }}>
                      {shut ? '▸ ' : '▾ '}
                      {m.path.replace(/\.viv$/, '')}
                      <tspan className="cm-region-n"> {r.n}</tspan>
                    </text>
                  </g>
                );
              })}
              <g className="cm-edges">
                {shown.map(([a, b, n]) => {
                  if (a === b) return null;
                  const pts = router.call(a, b);
                  if (!pts) return null;
                  const onR = onRoute.has(a) && onRoute.has(b) && route.indexOf(b) === route.indexOf(a) + 1;
                  const kind = onR ? 'route' : a === focus ? 'out' : b === focus ? 'in' : 'all';
                  return <path key={`${a}>${b}`} d={orthoPath(pts)} className={`is-${kind}`} markerEnd={`url(#cm-arrow-${kind})`} strokeWidth={kind === 'route' ? 2.6 : kind === 'all' ? 1 : Math.min(2.2, 1.3 + Math.log2(n) * 0.3)} />;
                })}
              </g>
              {[...lay.tiles.values()].map((t) => {
                const d = defs.get(t.id);
                const h = heat && heat.into.get(t.id);
                const hot = h ? Math.log(1 + h) / Math.log(1 + heat.max) : 0;
                const deep = Math.max(0, ...d.loops.map((l) => l.depth));
                const cls = ['cm-tile', t.id === sel && 'is-sel', onRoute.has(t.id) && 'is-route', focusCallees.has(t.id) && 'is-callee', focusCallers.has(t.id) && 'is-caller', hits.some((x) => x.id === t.id) && 'is-hit'].filter(Boolean).join(' ');
                return (
                  <g
                    key={t.id}
                    className={cls}
                    data-tile={t.id}
                    tabIndex={0}
                    role="button"
                    aria-label={`${d.main ? 'top level' : t.id}, ${d.end - d.line + 1} lines`}
                    onClick={() => go(t.id)}
                    onPointerEnter={() => setHover(t.id)}
                    onPointerLeave={() => setHover((h) => (h === t.id ? null : h))}
                    onFocus={() => setHover(t.id)}
                    onBlur={() => setHover((h) => (h === t.id ? null : h))}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        go(t.id);
                      }
                    }}>
                    <rect x={t.x} y={t.y} width={t.w} height={t.h} rx="4" />
                    {hot > 0 && <rect className="cm-heat" x={t.x} y={t.y + t.h - 4} width={t.w * hot} height="4" rx="1" />}
                    {view.k > 0.32 && (
                      <text x={t.x + 8} y={t.y + 17}>
                        {t.label}
                      </text>
                    )}
                    {view.k > 0.45 && (
                      <text className={`cm-sub${t.sub ? '' : ' is-missing'}`} x={t.x + 8} y={t.y + 32}>
                        {t.sub || 'no intent written'}
                      </text>
                    )}
                    {deep >= 2 && (
                      <text className="cm-loop" x={t.x + t.w - 6} y={t.y + 11} textAnchor="end">
                        ↻{deep}
                      </text>
                    )}
                    {contractOf.has(t.id) && view.k > 0.32 && (
                      <text className="cm-loop cm-contract-mark" x={t.x + t.w - (deep >= 2 ? 22 : 6)} y={t.y + 11} textAnchor="end">
                        ⇄
                      </text>
                    )}
                    {/* after the rects: the styles find the tile's box as its first child */}
                    <title>{t.intent ? `${d.main ? 'top level' : t.id}: ${texToPlain(t.intent)}` : `${t.id}: no intent written`}</title>
                  </g>
                );
              })}
            </g>
          </svg>
          {view.k < 0.6 && (
            // zoomed out, the module names are set at reading size over
            // the map, each where its region starts, skipping any that would
            // overlap a name already placed
            <div className="cm-labels" aria-hidden="true">
              {(() => {
                const placed = [];
                const out = [];
                for (const k of lay.order) {
                  const r = lay.regions.get(k);
                  const name = mod(k).path.replace(/\.viv$/, '').replace(/^(lib|study\/ports)\//, '');
                  const x = view.x + r.x * view.k;
                  const y = view.y + r.y * view.k - 2;
                  const w = name.length * 6.6 + 12;
                  if (placed.some((p) => x < p.x + p.w && p.x < x + w && Math.abs(p.y - y) < 16)) continue;
                  placed.push({ x, y, w });
                  out.push(
                    <span key={k} className={curMod === mod(k) ? 'is-cur' : ''} style={{ left: `${x}px`, top: `${y}px` }}>
                      {name}
                    </span>,
                  );
                }
                return out;
              })()}
            </div>
          )}
          <div className="cm-zoom">
            <button type="button" onClick={() => zoomBy(1.25)} aria-label="Zoom in">
              +
            </button>
            <button type="button" onClick={() => zoomBy(0.8)} aria-label="Zoom out">
              −
            </button>
            <button type="button" onClick={fit} aria-label="Fit the map (0)" title="Fit (0)">
              ⤢
            </button>
          </div>
          {svgBox && !(view.x >= 0 && view.y >= 0 && view.x + lay.width * view.k <= svgBox.width && view.y + lay.height * view.k <= svgBox.height) && (
          <svg
            className="cm-mini"
            width={mini.w}
            height={mini.h}
            aria-hidden="true"
            onClick={(ev) => {
              const r = ev.currentTarget.getBoundingClientRect();
              const mx = (ev.clientX - r.left) / miniK;
              const my = (ev.clientY - r.top) / miniK;
              const b = svgRef.current.getBoundingClientRect();
              setView((v) => ({ ...v, x: b.width / 2 - mx * v.k, y: b.height / 2 - my * v.k }));
            }}>
            {lay.order.map((k) => {
              const r = lay.regions.get(k);
              return <rect key={k} x={r.x * miniK} y={r.y * miniK} width={Math.max(1, r.w * miniK)} height={Math.max(1, r.h * miniK)} className={curMod === mod(k) ? 'is-cur' : ''} />;
            })}
            <rect className="cm-mini-view" x={(-view.x / view.k) * miniK} y={(-view.y / view.k) * miniK} width={(svgBox.width / view.k) * miniK} height={(svgBox.height / view.k) * miniK} />
          </svg>
          )}
          {legend && (
            <div className="cm-legend" role="note">
              <p>
                A region is a module, a tile is a def; a tile is wider the more lines the def has. Columns run left to right
                by how many calls the def is from the entry. The thick line is the route from the entry to the def you are reading.
                Only that def's calls are drawn one by one (point at another tile to see its calls); the calls between two
                modules run as one grey line behind the regions, thicker for more calls. <i>All calls</i> draws every call.
              </p>
              <p>
                <span className="cm-k cm-k-out" /> calls made by the def · <span className="cm-k cm-k-in cm-k-dash" /> calls into it · <b>↻2</b> a loop
                inside a loop
                {heat && (
                  <>
                    {' '}
                    · <span className="cm-k cm-k-heat" /> calls counted in {runLabel || <>the explorer's run of {exploreTitle}</>} (log scale)
                  </>
                )}
              </p>
              <p>
                Keys: <kbd>s</kbd> find, <kbd>[</kbd> <kbd>]</kbd> back and forward, <kbd>u</kbd> up the route, <kbd>e</kbd> the entry,{' '}
                <kbd>j</kbd> first callee, <kbd>c</kbd> fold the module, <kbd>a</kbd> all calls, <kbd>0</kbd> fit. Click a module's name to fold it.
              </p>
              {legendBuiltins.length > 0 && (
                <p className="cm-builtins">
                  Builtins called (not drawn):{' '}
                  {legendBuiltins.slice(0, 40).map(([b, c]) => (
                    <code key={b}>
                      {b}
                      <small>×{c}</small>
                    </code>
                  ))}
                  {legendBuiltins.length > 40 && <> and {legendBuiltins.length - 40} more</>}
                </p>
              )}
            </div>
          )}
        </div>
        <aside className="cm-pane" ref={paneRef} aria-label="The def's source">
          {cur && (
            <>
              <header className="cm-pane-head">
                <h1>
                  <code>{cur.main ? 'top level' : cur.id}</code>
                  {!cur.main && <span className="cm-params">({cur.params.join(', ')})</span>}
                </h1>
                <p className="cm-where">
                  {REPO_PUBLIC ? (
                    <a href={`${GH}${curMod.path}${cur.main ? '' : `#L${cur.line}-L${cur.end}`}`} rel="noopener" target="_blank">
                      {curMod.path}
                      {!cur.main && `:${cur.line}`}
                    </a>
                  ) : (
                    <span>
                      {curMod.path}
                      {!cur.main && `:${cur.line}`}
                    </span>
                  )}
                  {!cur.main && <> · {cur.end - cur.line + 1} lines</>}
                  {heat && heat.into.get(cur.id) && <> · called {heat.into.get(cur.id).toLocaleString('en-US')} times in {ranIn}</>}
                </p>
                {(() => {
                  const ct = contractOf.get(cur.id);
                  const text = intentText(cur) || (ct?.name === cur.id ? ct.intent : '');
                  return text ? <Intent as="p" className="cm-intent" text={text} katex={katex} /> : <p className="cm-intent is-missing">No intent written for this def.</p>;
                })()}
                {contractOf.has(cur.id) && <ContractPanel contract={contractOf.get(cur.id)} cur={cur.id} defs={defs} go={go} katex={katex} />}
                {callers.length > 0 && (
                  <p className="cm-rel">
                    <span>Called by</span>
                    {callers.map((c) => (
                      <button type="button" key={c} className="is-caller" onClick={() => go(c)}>
                        {defs.get(c)?.main ? 'top level' : c}
                      </button>
                    ))}
                  </p>
                )}
                {callees.length > 0 && (
                  <p className="cm-rel">
                    <span>Calls</span>
                    {callees.map((c) => (
                      <button type="button" key={c.to} onClick={() => go(c.to)} title={c.loop ? `inside ${c.loop} loop${c.loop > 1 ? 's' : ''}` : ''}>
                        {c.to}
                        {c.sites > 1 && <small>×{c.sites}</small>}
                        {c.loop > 0 && <small className="cm-inloop">↻{c.loop}</small>}
                        {c.value && <small> (as a value)</small>}
                      </button>
                    ))}
                  </p>
                )}
                {cur.indirect.length > 0 && (
                  <p className="cm-rel cm-ind">
                    <span>Through a function value</span>
                    {[...new Set(cur.indirect.map((x) => x[0]))].map((v) => (
                      <code key={v}>{v}(…)</code>
                    ))}
                  </p>
                )}
                {cur.steps.length > 0 && (
                  <p className="cm-rel cm-steps">
                    <span>Steps</span>
                    {cur.steps.map((s) => {
                      // a step's marker comment is not among the top level's
                      // statement lines: go to the first line shown from it on
                      const at = (lines.find((l) => l.n >= s.line) || { n: s.line }).n;
                      return (
                        <a key={s.id} href={`#L${at}`} onClick={(e) => {
                          e.preventDefault();
                          paneRef.current?.querySelector(`[data-line="${at}"]`)?.scrollIntoView({ block: 'center' });
                        }}>
                          {s.text || s.id}
                        </a>
                      );
                    })}
                  </p>
                )}
                {builtins.length > 0 && (
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
                data-lang="viv"
                onClick={(e) => {
                  const a = e.target.closest('a.cm-ref');
                  if (a) {
                    e.preventDefault();
                    go(a.dataset.def);
                  }
                }}>
                {lines.map((l) => {
                  const lp = loopAt.get(l.n);
                  const inloop = callAt.get(l.n);
                  const inDef = !cur.main && l.n >= cur.line && l.n <= cur.end;
                  return (
                    <div key={l.n} id={`L${l.n}`} data-line={l.n} className={`cm-ln${l.gap ? ' is-gap' : ''}${l.step ? ' is-step' : ''}${inDef || cur.main ? '' : ' is-out'}${lp && lp.depth >= 2 ? ' is-deep' : ''}`}>
                      <span className="cm-num">{l.n}</span>
                      <span className="cm-txt" style={{ paddingLeft: `${l.indent + HANG}ch`, textIndent: `-${HANG}ch` }} dangerouslySetInnerHTML={{ __html: l.html || ' ' }} />
                      {inloop ? <span className="cm-mark" title={`a call inside ${inloop} loop${inloop > 1 ? 's' : ''}`}>↻{inloop}</span> : null}
                    </div>
                  );
                })}
              </pre>
              {!cur.main && (
                <button type="button" className="cm-whole" onClick={() => setWhole((w) => !w)}>
                  {whole ? 'Show this def only' : `Show all of ${curMod.path}`}
                </button>
              )}
            </>
          )}
        </aside>
      </div>
    </div>
  );
}
