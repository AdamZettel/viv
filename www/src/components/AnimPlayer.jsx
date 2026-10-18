// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import { buildTimeline, stateAt, captionAt } from '../lib/animParse.js';
import { SERIES_VARS } from './LineChart.jsx';

// The player for an animation a Viv program wrote.
//
// The program says WHAT happens and for how long -- draw this curve on, morph it
// into that one, put a label there, say this. The player only interpolates and
// draws: it has no presentation of its own to add. It opens on the LAST frame,
// paused, so a page shows the finished picture; Play runs it from the start.
//
// With "reduce motion" set, Play steps from one finished transition to the next
// instead of tweening between them.

function niceTicks(a, b, n) {
  const span = b - a || 1;
  const step0 = span / Math.max(1, n);
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const out = [];
  for (let t = Math.ceil(a / step - 1e-9) * step; t <= b + step * 1e-9; t += step) out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
  return { ticks: out, step };
}
const tickLabel = (step) => {
  const d = Math.max(0, Math.min(8, -Math.floor(Math.log10(step) + 1e-9)));
  return (v) => (v === 0 ? '0' : v.toFixed(d).replace('-', '−'));
};

// Text may carry $math$. KaTeX is fetched only when some text needs it, so a page
// of plain animations never pays for it; until it arrives the dollars are dropped.
function useKatex(wanted) {
  const [katex, setKatex] = useState(null);
  useEffect(() => {
    if (!wanted || katex) return;
    let alive = true;
    Promise.all([import('katex'), import('katex/dist/katex.min.css')]).then(([k]) => { if (alive) setKatex(() => k.default || k); }).catch(() => {});
    return () => { alive = false; };
  }, [wanted, katex]);
  return katex;
}
const esc = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
function richText(text, katex) {
  return text.split(/(\$[^$]+\$)/).map((part) => {
    if (part.startsWith('$') && part.endsWith('$') && part.length > 2) {
      const tex = part.slice(1, -1);
      if (!katex) return esc(tex);
      try { return katex.renderToString(tex, { throwOnError: false, output: 'html' }); } catch { return esc(tex); }
    }
    return esc(part);
  }).join('');
}

export default function AnimPlayer({ chart }) {
  const tl = useMemo(() => buildTimeline(chart.lines), [chart.lines]);
  const total = Math.max(tl.duration, 0.01);
  const wrapRef = useRef(null);
  const [width, setWidth] = useState(640);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [reduced, setReduced] = useState(false);
  // whether the reader has touched the controls, and whether this player has
  // already had its one automatic run
  const touched = useRef(false);
  const autoplayed = useRef(false);
  const wantsMath = useMemo(() => chart.lines.some((l) => /^(label|say): .*\$/.test(l)), [chart.lines]);
  const katex = useKatex(wantsMath);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return undefined;
    const measure = () => { const w = Math.round(el.clientWidth); if (w > 0) setWidth((p) => (Math.abs(p - w) > 1 ? w : p)); };
    measure();
    if (typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useEffect(() => {
    if (typeof matchMedia === 'undefined') return undefined;
    const mq = matchMedia('(prefers-reduced-motion: reduce)');
    const on = () => setReduced(mq.matches);
    on();
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, []);

  // Someone who asked for less motion gets the finished picture instead of an
  // empty stage they would have to press play to fill.
  useEffect(() => {
    if (!reduced) return;
    setPlaying(false);
    setTime(total);
  }, [reduced, total]);

  // Play once, when it scrolls into view. An animation that is sitting on its
  // last frame when you first look at it is a diagram, and the reader has no
  // way to know they missed anything.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el || reduced || typeof IntersectionObserver === 'undefined') return undefined;
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        // off screen it stops where it is (a demo scroll keeps every card mounted)
        if (!e.isIntersecting) {
          setPlaying(false);
          continue;
        }
        if (autoplayed.current || touched.current) continue;
        autoplayed.current = true;
        setTime(0);
        setPlaying(true);
      }
    }, { threshold: 0.25 });
    io.observe(el);
    return () => io.disconnect();
  }, [reduced]);

  // the clock
  useEffect(() => {
    if (!playing) return undefined;
    let raf = 0;
    let last = performance.now();
    const tick = (now) => {
      // a frame's timestamp is when the frame began, which can be before the
      // performance.now() read above: the first step would run time backward
      const dt = Math.max(0, now - last) / 1000;
      last = now;
      setTime((t) => {
        const next = t + dt;
        if (next >= total) { setPlaying(false); return total; }
        return next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, total]);

  const toggle = useCallback(() => {
    touched.current = true;
    setPlaying((p) => {
      if (!p && time >= total - 1e-6) setTime(0);
      return !p;
    });
  }, [time, total]);

  const [xlo, xhi, ylo, yhi] = tl.window;
  const W = width;
  const xT = niceTicks(xlo, xhi, W < 420 ? 4 : 7);
  const yT = niceTicks(ylo, yhi, 5);
  const yfmt = tickLabel(yT.step);
  const xfmt = tickLabel(xT.step);
  const widest = tl.axes ? Math.max(...yT.ticks.map((t) => yfmt(t).length), 1) : 0;
  const M = tl.axes
    ? { l: 16 + widest * 6.4 + (tl.axes.ylabel ? 18 : 0), r: 14, t: 10, b: tl.axes.xlabel ? 44 : 26 }
    : { l: 10, r: 10, t: 10, b: 10 };
  const plotW = Math.max(40, W - M.l - M.r);
  // a stage that draws shapes keeps the window's aspect ratio, so a circle is round
  const aspect = tl.spatial ? Math.max(0.3, Math.min(1.25, (yhi - ylo) / (xhi - xlo))) : 0.54;
  const plotH = Math.round(tl.spatial ? plotW * aspect : Math.max(200, Math.min(360, plotW * aspect)));
  const SX = plotW / (xhi - xlo); // data units -> px, for radii
  const H = plotH + M.t + M.b;
  const X = (v) => M.l + ((v - xlo) / (xhi - xlo)) * plotW;
  const Y = (v) => M.t + (1 - (v - ylo) / (yhi - ylo)) * plotH;
  const clipId = `an${useId().replace(/:/g, '')}`;

  const objects = stateAt(tl, time, reduced);
  const caption = captionAt(tl, time);
  const pathOf = (o) => {
    let d = '';
    for (let i = 0; i < o.x.length; i++) {
      const yy = Math.max(-1e4, Math.min(1e4, Y(o.y[i])));
      d += `${i ? 'L' : 'M'}${X(o.x[i]).toFixed(2)} ${yy.toFixed(2)}`;
    }
    return d;
  };

  if (!tl.order.length && tl.errors.length) {
    return <figure className="vc"><figcaption className="vc-title">{tl.title}</figcaption><p className="an-error">{tl.errors[0]}</p></figure>;
  }

  return (
    <figure className="vc an">
      {tl.title && <figcaption className="vc-title">{tl.title}</figcaption>}
      <div className="an-stage" ref={wrapRef}>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={tl.title || 'animation'}>
          <defs><clipPath id={clipId}><rect x={M.l} y={M.t - 1} width={plotW} height={plotH + 2} /></clipPath></defs>
          {tl.axes && (
            <g>
              {yT.ticks.map((t) => (
                <g key={'y' + t}>
                  <line className="vc-grid" x1={M.l} x2={W - M.r} y1={Y(t)} y2={Y(t)} />
                  <text className="vc-tick" x={M.l - 8} y={Y(t) + 3.8} textAnchor="end">{yfmt(t)}</text>
                </g>
              ))}
              <line className="vc-axis" x1={M.l} x2={W - M.r} y1={M.t + plotH} y2={M.t + plotH} />
              {xT.ticks.map((t) => (
                <g key={'x' + t}>
                  <line className="vc-axis" x1={X(t)} x2={X(t)} y1={M.t + plotH} y2={M.t + plotH + 4} />
                  <text className="vc-tick" x={X(t)} y={M.t + plotH + 17} textAnchor="middle">{xfmt(t)}</text>
                </g>
              ))}
              {tl.axes.xlabel && <text className="vc-label" x={M.l + plotW / 2} y={H - 6} textAnchor="middle">{tl.axes.xlabel}</text>}
              {tl.axes.ylabel && (
                <text className="vc-label" x={12} y={M.t + plotH / 2} textAnchor="middle"
                  transform={`rotate(-90 12 ${M.t + plotH / 2})`}>{tl.axes.ylabel}</text>
              )}
            </g>
          )}
          <g clipPath={`url(#${clipId})`}>
            {objects.map((o) => {
              if (o.alpha <= 0.001) return null;
              const color = SERIES_VARS[o.n % SERIES_VARS.length];
              if (o.kind === 'hline') return <line key={o.id} className="an-ref" opacity={o.alpha} x1={M.l} x2={M.l + plotW} y1={Y(o.at)} y2={Y(o.at)} />;
              if (o.kind === 'vline') return <line key={o.id} className="an-ref" opacity={o.alpha} x1={X(o.at)} x2={X(o.at)} y1={M.t} y2={M.t + plotH} />;
              if (o.kind === 'curve') {
                return (
                  <path key={o.id} d={pathOf(o)} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"
                    opacity={o.alpha} pathLength="1" strokeDasharray={o.progress < 1 ? '1' : undefined} strokeDashoffset={o.progress < 1 ? 1 - o.progress : undefined} />
                );
              }
              if (o.kind === 'dots') {
                const r = o.r !== undefined && o.r !== null ? Math.max(0.8, o.r * SX) : 3.4;
                return <g key={o.id} opacity={o.alpha}>{o.x.map((v, i) => <circle key={i} className={o.r !== undefined ? 'an-ball' : 'vc-dot'} cx={X(v)} cy={Y(o.y[i])} r={r} fill={color} />)}</g>;
              }
              if (o.kind === 'circle') return <circle key={o.id} className="an-ball" opacity={o.alpha} cx={X(o.x)} cy={Y(o.y)} r={Math.max(0.8, o.r * SX)} fill={color} />;
              if (o.kind === 'rect') {
                const cx = X(o.x), cy = Y(o.y);
                return <rect key={o.id} className="an-shape" opacity={o.alpha} x={cx - o.w * SX / 2} y={cy - o.h * SX / 2} width={o.w * SX} height={o.h * SX} rx="2" fill={color}
                  transform={o.angle ? `rotate(${-o.angle * 180 / Math.PI} ${cx} ${cy})` : undefined} />;
              }
              if (o.kind === 'segment') return <line key={o.id} className="an-seg" opacity={o.alpha} x1={X(o.x1)} y1={Y(o.y1)} x2={X(o.x2)} y2={Y(o.y2)} />;
              if (o.kind === 'poly') return <polygon key={o.id} className="an-shape" opacity={o.alpha} points={o.x.map((v, i) => `${X(v).toFixed(2)},${Y(o.y[i]).toFixed(2)}`).join(' ')} fill={color} />;
              return null;
            })}
          </g>
        </svg>
        {objects.filter((o) => o.kind === 'label' && o.alpha > 0.001).map((o) => (
          <div key={o.id} className="an-label" style={{ left: `${(X(o.x) / W) * 100}%`, top: `${(Y(o.y) / H) * 100}%`, opacity: o.alpha }}
            dangerouslySetInnerHTML={{ __html: richText(o.text, katex) }} />
        ))}
      </div>
      {/* a stage with nothing to play is a DIAGRAM: no transport, no caption line */}
      {tl.duration > 0 && (
      <>
      <p className="an-caption" aria-live="polite" dangerouslySetInnerHTML={{ __html: caption ? richText(caption, katex) : '&nbsp;' }} />
      <div className="an-controls">
        <button type="button" className="an-play" onClick={toggle} aria-label={playing ? 'Pause' : time >= total - 1e-6 ? 'Play from the start' : 'Play'}>
          <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true">
            {playing ? <path d="M6 4h4v16H6zM14 4h4v16h-4z" /> : <path d="M7 4.5v15l12-7.5z" />}
          </svg>
        </button>
        <input type="range" className="an-scrub" min="0" max={total} step="0.01" value={time} aria-label="Time"
          onChange={(e) => { touched.current = true; setPlaying(false); setTime(Number(e.target.value)); }} />
        <span className="an-time">{time.toFixed(1)} / {total.toFixed(1)} s</span>
      </div>
      </>
      )}
      {tl.errors.length > 0 && <p className="an-error">{tl.errors[0]}</p>}
    </figure>
  );
}
