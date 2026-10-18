// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The animation protocol: records -> a timeline.
//
// A Viv program that animates prints one record per verb, in order:
//
//     stage: <title> ; xlo xhi ylo yhi          opens an animation
//     axes: <xlabel> ; <ylabel>                  ticks and a grid, optional
//     curve: <id> ; [x…] ; [y…]                  objects: defined HIDDEN
//     dots: <id> ; [x…] ; [y…] [; r]               (r: a radius in data units)
//     label: <id> ; <text> ; x y
//     hline: <id> ; y          vline: <id> ; x
//     circle: <id> ; x y r     rect: <id> ; x y w h [angle]     segment: <id> ; x1 y1 x2 y2
//     poly: <id> ; [x…] ; [y…]
//     frame: s                                   a KEYFRAME: advance s seconds; an object set
//                                                again later moves there over the time between
//     draw: <id> ; s           fade_in: <id> ; s      fade_out: <id> ; s
//     morph: <id> ; [x…] ; [y…] ; s              move_to: <id> ; x y s
//     say: <caption>[ ; value]                   wait: s
//     together:  …  apart:                       transitions between them start at once
//
// This module is plain JavaScript with no dependencies: the player uses it in the
// browser, and the build uses it to refuse a post whose animation names an object
// that does not exist.

export const ANIM_VERBS = ['stage', 'axes', 'curve', 'dots', 'label', 'hline', 'vline', 'xtick', 'ytick', 'circle', 'rect', 'segment', 'poly', 'draw', 'fade_in', 'fade_out', 'morph', 'move_to', 'say', 'wait', 'frame', 'together', 'apart'];
const SHAPES = new Set(['circle', 'rect', 'segment', 'poly']);
const RECORD = new RegExp(`^(${ANIM_VERBS.join('|')}):(?: (.*))?$`);

/** Is this output line an animation record? */
export const isAnimRecord = (line) => RECORD.test(line);

// Inside a figure() with no stage open, these records are static marks on that
// figure's chart (lower.rs ANIM_VERBS): drawn exactly as printed, nothing added.
//     hline: <id> ; y        vline: <id> ; x        xtick: <text> ; x        ytick: <text> ; y
//     label: <id> ; <text> ; x y     rect: <id> ; x y w h (centered)     segment: <id> ; x1 y1 x2 y2
//     box: <id> ; <style> ; x0 y0 x1 y1     arrow: <id> ; <style> ; [xs] ; [ys]
//     note: <id> ; <text> ; <style> ; x y       (a diagram's marks)
const MARK = /^(hline|vline|xtick|ytick|label|rect|segment|box|arrow|note): (.*)$/;
/** The static mark an output line describes, or null. */
export function parseMark(line) {
  const m = line.match(MARK);
  if (!m) return null;
  const parts = m[2].split(' ; ');
  const n = (s) => (s || '').trim().split(/\s+/).map(recNum);
  let mark = null;
  if (m[1] === 'hline') mark = { kind: 'hline', id: parts[0], y: n(parts[1])[0] };
  else if (m[1] === 'vline') mark = { kind: 'vline', id: parts[0], x: n(parts[1])[0] };
  else if (m[1] === 'xtick') mark = { kind: 'xtick', text: parts[0], x: n(parts[1])[0] };
  else if (m[1] === 'ytick') mark = { kind: 'ytick', text: parts[0], y: n(parts[1])[0] };
  else if (m[1] === 'label') { const [x, y] = n(parts[2]); mark = { kind: 'label', id: parts[0], text: parts[1], x, y }; }
  else if (m[1] === 'rect') { const [x, y, w, h] = n(parts[1]); mark = { kind: 'rect', id: parts[0], x, y, w, h }; }
  else if (m[1] === 'segment') { const [x1, y1, x2, y2] = n(parts[1]); mark = { kind: 'segment', id: parts[0], x1, y1, x2, y2 }; }
  else if (m[1] === 'box') {
    const v = n(parts[2]);
    if (v.length !== 4) return null;
    const [x0, y0, x1, y1] = v;
    mark = { kind: 'box', id: parts[0], style: (parts[1] || '').trim(), x0, y0, x1, y1 };
  } else if (m[1] === 'arrow') {
    const am = (s) => { const t = (s || '').trim(); return t.startsWith('[') && t.endsWith(']') ? recArr(t.slice(1, -1)) : null; };
    const x = am(parts[2]);
    const y = am(parts[3]);
    // a non-finite point is a gap in the arrow, not a reason to drop it
    return x && y && x.length === y.length && x.length >= 2 ? { kind: 'arrow', id: parts[0], style: (parts[1] || '').trim(), x, y } : null;
  } else if (m[1] === 'note') {
    const v = n(parts[3]);
    if (v.length !== 2) return null;
    mark = { kind: 'note', id: parts[0], text: parts[1] || '', style: (parts[2] || '').trim(), x: v[0], y: v[1] };
  }
  if (!mark) return null;
  const vals = Object.entries(mark).filter(([k]) => !['kind', 'id', 'text', 'style'].includes(k)).map(([, v]) => v);
  return vals.every(Number.isFinite) ? mark : null;
}

/**
 * One number of a record. The engines print C's `%g` / `%.17g` spellings, which
 * include `nan`, `-nan`, `inf` and `-inf`; JavaScript's Number() reads the last
 * two as NaN, so they are spelled out here. A non-finite value is kept: a chart
 * draws it as a gap, it does not drop the series.
 */
export function recNum(s) {
  const t = String(s).trim();
  if (t === 'inf' || t === '+inf') return Infinity;
  if (t === '-inf') return -Infinity;
  return Number(t);
}
/** The inside of a record's `[...]`, as numbers; `[]` is an empty array (not [0]). */
export function recArr(s) {
  const t = String(s).trim();
  return t === '' ? [] : t.split(',').map(recNum);
}
const nums = (s) => s.trim().split(/\s+/).map(recNum);

// A series style: the words of a
// `style: <label> ; <words>` record, one per aspect. The compiler refuses any
// other word, so an unknown one here is a record from a newer compiler, and
// is left out rather than guessed at.
export const STYLE_WORDS = {
  line: ['solid', 'dash', 'dot', 'noline'],
  color: ['s1', 's2', 's3', 's4', 's5', 's6'],
  marker: ['circle', 'ring', 'square', 'cross', 'nomarker'],
  width: ['thin', 'thick'],
};
/** `style: exact ; dash s2` -> { label: 'exact', style: { line: 'dash', color: 's2' } }, or null. */
export function parseStyle(line) {
  const m = line.match(/^style: (.*?) ; (.*?)\s*$/);
  if (!m) return null;
  const style = {};
  for (const w of m[2].split(/\s+/)) {
    for (const [aspect, words] of Object.entries(STYLE_WORDS)) if (words.includes(w)) style[aspect] = w;
  }
  return { label: m[1], style };
}
const arr = (s) => recArr(s.trim().replace(/^\[|\]$/g, ''));
const fmtValue = (v) => (Number.isInteger(v) ? String(v) : String(+v.toPrecision(6)));

/**
 * Build a timeline from the records of ONE animation (from its `stage:` line on).
 * Returns { title, window, axes, objects, order, tracks, captions, duration, errors }.
 *   objects[id] = { id, kind, n }            n: creation index among its kind's color class
 *   tracks[id]  = [{ t0, t1, kind, ... }]    that object's transitions, in time order
 */
export function buildTimeline(lines) {
  // keyframed: the program advances with frame(); objects are then visible from
  // creation, and an object set again later moves there. spatial: it draws
  // shapes, so the stage keeps the window's aspect ratio.
  const out = { title: '', window: [0, 1, 0, 1], axes: null, objects: {}, order: [], tracks: {}, captions: [], duration: 0, errors: [], keyframed: false, spatial: false };
  const colorOf = {}; // color class (the id's prefix before `:`) -> palette index
  let t = 0;
  let group = null; // { t0, longest } while inside together: … apart:
  let colored = 0;
  const span = (id, dur, step) => {
    const d = Math.max(0, dur);
    const t0 = group ? group.t0 : t;
    if (group) group.longest = Math.max(group.longest, d);
    else t += d;
    if (id !== null) out.tracks[id].push({ t0, t1: t0 + d, ...step });
  };
  const need = (verb, id, kinds) => {
    const o = out.objects[id];
    if (!o) { out.errors.push(`${verb}: there is no object named "${id}"`); return null; }
    if (kinds && !kinds.includes(o.kind)) { out.errors.push(`${verb}: "${id}" is a ${o.kind}; ${verb} works on ${kinds.join(' or ')}`); return null; }
    return o;
  };
  const define = (id, kind, state) => {
    const prev = out.objects[id];
    if (prev && prev.kind !== kind) { out.errors.push(`${kind}: "${id}" is already a ${prev.kind}`); return; }
    if (!prev) {
      // a segment is a line, drawn neutral like hline; the filled things take the palette
      const hasColor = kind === 'curve' || kind === 'dots' || (SHAPES.has(kind) && kind !== 'segment');
      // Objects whose id shares a prefix before `:` share a color: `rect("xor:1")`
      // and `rect("xor:2")` are two of the same thing, and a schematic draws them
      // alike. An id with no colon is its own class, as before.
      const cls = id.includes(':') ? id.slice(0, id.indexOf(':')) : id;
      if (hasColor && !(cls in colorOf)) colorOf[cls] = colored++;
      out.objects[id] = { id, kind, n: hasColor ? colorOf[cls] : 0 };
      out.order.push(id);
      out.tracks[id] = [];
      if (SHAPES.has(kind) || (kind === 'dots' && state.r !== undefined)) out.spatial = true;
    }
    out.tracks[id].push({ t0: t, t1: t, kind: 'set', ...state });
  };

  for (const line of lines) {
    const m = line.match(RECORD);
    if (!m) continue;
    const verb = m[1];
    const parts = (m[2] || '').split(' ; ');
    if (verb === 'stage') {
      out.title = parts[0] || '';
      const w = nums(parts[1] || '');
      if (w.length === 4 && w.every(Number.isFinite) && w[0] !== w[1] && w[2] !== w[3]) out.window = w;
      else out.errors.push('stage: the window must be four numbers, xlo < xhi and ylo < yhi');
    } else if (verb === 'axes') {
      out.axes = { xlabel: parts[0] || '', ylabel: parts[1] || '' };
    } else if (verb === 'curve' || verb === 'dots' || verb === 'poly') {
      const x = arr(parts[1] || '');
      const y = arr(parts[2] || '');
      if (x.length !== y.length || !x.length) out.errors.push(`${verb}: "${parts[0]}" needs x and y of the same length`);
      else define(parts[0], verb, verb === 'dots' && parts[3] !== undefined ? { x, y, r: Number(parts[3]) } : { x, y });
    } else if (verb === 'circle') {
      const [x, y, r] = nums(parts[1] || '');
      define(parts[0], 'circle', { x, y, r });
    } else if (verb === 'rect') {
      const [x, y, w, h, angle] = nums(parts[1] || '');
      define(parts[0], 'rect', { x, y, w, h, angle: Number.isFinite(angle) ? angle : 0 });
    } else if (verb === 'segment') {
      const [x1, y1, x2, y2] = nums(parts[1] || '');
      define(parts[0], 'segment', { x1, y1, x2, y2 });
    } else if (verb === 'frame') {
      out.keyframed = true;
      span(null, Number(parts[0]), {});
    } else if (verb === 'label') {
      const [x, y] = nums(parts[2] || '');
      define(parts[0], 'label', { text: parts[1] || '', x, y });
    } else if (verb === 'hline' || verb === 'vline') {
      define(parts[0], verb, { at: Number(parts[1]) });
    } else if (verb === 'draw' || verb === 'fade_in' || verb === 'fade_out') {
      if (need(verb, parts[0])) span(parts[0], Number(parts[1]), { kind: verb });
    } else if (verb === 'morph') {
      if (need(verb, parts[0], ['curve', 'dots'])) {
        const x = arr(parts[1] || '');
        const y = arr(parts[2] || '');
        if (x.length !== y.length || !x.length) out.errors.push(`morph: "${parts[0]}" needs x and y of the same length`);
        else span(parts[0], Number(parts[3]), { kind: 'morph', x, y });
      }
    } else if (verb === 'move_to') {
      if (need(verb, parts[0], ['label'])) {
        const [x, y, s] = nums(parts[1] || '');
        span(parts[0], s, { kind: 'move', x, y });
      }
    } else if (verb === 'say') {
      const text = parts.length > 1 ? `${parts[0]} ${fmtValue(Number(parts[1]))}` : parts[0] || '';
      out.captions.push({ t: group ? group.t0 : t, text });
    } else if (verb === 'wait') {
      span(null, Number(parts[0]), {});
    } else if (verb === 'together') {
      if (group) out.errors.push('together: already inside a together … apart group');
      else group = { t0: t, longest: 0 };
    } else if (verb === 'apart') {
      if (!group) out.errors.push('apart: there is no together to close');
      else { t = group.t0 + group.longest; group = null; }
    }
  }
  if (group) { out.errors.push('together: never closed with apart()'); t = group.t0 + group.longest; }
  out.duration = t;
  // A stage with nothing to play is a DIAGRAM: what it places is simply there.
  // (In an authored animation, objects start hidden and a transition reveals
  // them; with no transitions at all that rule would show an empty box.)
  if (out.duration === 0 && !out.keyframed) out.keyframed = true;
  return out;
}

const ease = (u) => (u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u));
const lerp = (a, b, u) => a + (b - a) * u;
// the same curve sampled at n points, by index: lets a morph change the point count
function resample(v, n) {
  if (v.length === n) return v;
  const outv = new Array(n);
  for (let i = 0; i < n; i++) {
    const p = n === 1 ? 0 : (i / (n - 1)) * (v.length - 1);
    const k = Math.min(v.length - 2, Math.floor(p));
    outv[i] = v.length === 1 ? v[0] : lerp(v[k], v[k + 1], p - k);
  }
  return outv;
}

/**
 * The state of every object at time `time`: { id, kind, n, alpha, progress, x, y, text, at }.
 * `progress` is how much of a curve has been drawn (0..1). With `snap`, a transition
 * is either not started or already finished -- what reduced motion asks for.
 */
const NUMERIC = ['x', 'y', 'r', 'w', 'h', 'angle', 'at', 'x1', 'y1', 'x2', 'y2'];
function assign(s, k) {
  for (const f of NUMERIC) if (k[f] !== undefined) s[f] = k[f];
  if (k.text !== undefined) s.text = k.text;
}
// the state between two keyframes of the same object: numbers and arrays move
// linearly -- unless the move is more than half the stage, which is a periodic
// wrap (an atom leaving the box on the right and entering on the left), not a
// motion: then the object holds where it was until the keyframe, and jumps.
const SPAN_OF = { x: 0, x1: 0, x2: 0, y: 1, y1: 1, y2: 1 };
function blend(s, from, to, u, spans) {
  const step = (a, b, f) => {
    const half = spans && SPAN_OF[f] !== undefined ? spans[SPAN_OF[f]] / 2 : Infinity;
    return Math.abs(b - a) > half ? a : lerp(a, b, u);
  };
  for (const f of NUMERIC) {
    if (to[f] === undefined) continue;
    if (Array.isArray(to[f])) {
      const a = Array.isArray(from[f]) ? resample(from[f], to[f].length) : to[f];
      s[f] = to[f].map((v, i) => step(a[i], v, f));
    } else if (typeof from[f] === 'number') s[f] = step(from[f], to[f], f);
    else s[f] = to[f];
  }
  if (to.text !== undefined) s.text = to.text;
}

export function stateAt(tl, time, snap = false) {
  const spans = [tl.window[1] - tl.window[0], tl.window[3] - tl.window[2]];
  return tl.order.map((id) => {
    const o = tl.objects[id];
    const s = { id, kind: o.kind, n: o.n, alpha: tl.keyframed ? 1 : 0, progress: 1, x: null, y: null, text: '', at: 0 };
    let last = null; // the previous 'set' keyframe: what a later one moves from
    for (const k of tl.tracks[id]) {
      if (k.t0 > time) {
        // a keyframe still ahead: move toward it from the last one
        if (k.kind === 'set' && last && tl.keyframed && k.t0 > last.t0) {
          const before = { ...s };
          blend(s, before, k, (time - last.t0) / (k.t0 - last.t0), spans);
        }
        break;
      }
      // snapped, a transition that has STARTED is shown finished: the picture then
      // changes on the same beat as the caption that introduces it
      const u = snap || k.t1 <= k.t0 ? 1 : Math.min(1, (time - k.t0) / (k.t1 - k.t0));
      if (k.kind === 'set') {
        assign(s, k);
        last = k;
      } else if (k.kind === 'draw') {
        // a curve is drawn ON, left to right; anything else has no stroke to follow and fades in
        if (o.kind === 'curve') {
          if (u > 0 || k.t1 === k.t0) s.alpha = 1;
          s.progress = u;
        } else {
          s.alpha = Math.max(s.alpha, ease(u));
        }
      } else if (k.kind === 'fade_in') {
        s.alpha = Math.max(s.alpha, ease(u));
        s.progress = 1;
      } else if (k.kind === 'fade_out') {
        s.alpha = Math.min(s.alpha, 1 - ease(u));
      } else if (k.kind === 'morph') {
        const e = ease(u);
        const fx = resample(s.x, k.x.length);
        const fy = resample(s.y, k.y.length);
        s.x = k.x.map((v, i) => lerp(fx[i], v, e));
        s.y = k.y.map((v, i) => lerp(fy[i], v, e));
      } else if (k.kind === 'move') {
        const e = ease(u);
        s.x = lerp(s.x, k.x, e);
        s.y = lerp(s.y, k.y, e);
      }
    }
    // not created yet (its first `set` is still ahead): nothing to draw
    if (!last) s.alpha = 0;
    return s;
  });
}

/** The caption showing at `time`, or ''. */
export function captionAt(tl, time) {
  let text = '';
  for (const c of tl.captions) if (c.t <= time + 1e-9) text = c.text;
  return text;
}
