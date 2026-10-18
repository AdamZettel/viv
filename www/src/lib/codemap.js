// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The layout of a code map (components/CodeMap.jsx): a region per module, a
// tile per def, placed in columns by how far the module's nearest def is from
// the entry, in calls, ordered by their callers' positions. No randomness: the
// same map is laid out the same way every time. `routeEdges` draws the calls.
import { texToPlain } from './richText.js';

// a tile holds two lines: the def's name and its intent
const TILE_H = 40;
const PAD = 12;
const HEAD = 34;
const GAP_X = 14;
const GAP_Y = 16;
const COL_GAP = 96;
const ROW_GAP = 26;
const CHAR_W = 7.3;
const MAX_TILE_W = 260;
// room for a few words of the intent under the name
const MIN_TILE_W = 190;
const INTENT_CHAR_W = 6.1;
const SHUT_W = 220;

const labelOf = (d) => (d.main ? 'top level' : d.id);

/** A def's intent in one line, as written: its `what`, else the first paragraph of its doc; '' when it has none (`code` marks and $math$ kept, for the pane). */
export function intentText(d) {
  if (d.main) return 'the program as it runs';
  // a def `grad` wrote: what it differentiates
  const g = /^viv__ad_grad__(.+)_by_(.+)$/.exec(d.id);
  if (g) return `the derivative of ${g[1]} by ${g[2]}, written by grad`;
  return (d.intent || '').trim() || (d.doc || '').split('\n\n')[0].replace(/\s+/g, ' ').trim();
}

/** A def's intent for a tile's line, which is plain text: `code` marks go. */
export function intentOf(d) {
  return intentText(d).replace(/`([^`]*)`/g, '$1');
}

const cut = (t, n) => (t.length > n ? t.slice(0, Math.max(1, n - 1)).trimEnd() + '…' : t);

/** The width of a def's tile: room for its name, more for more lines. */
function tileWidth(d) {
  const n = d.main ? 0 : Math.max(1, d.end - d.line + 1);
  const deep = Math.max(0, ...d.loops.map((l) => l.depth)) >= 2 ? 16 : 0;
  const name = labelOf(d).length * CHAR_W + 22 + deep;
  return Math.min(MAX_TILE_W, Math.max(name, MIN_TILE_W, 34 + deep + 9 * Math.sqrt(n)));
}

export function layoutMap(map, collapsed = new Set()) {
  const byMod = new Map();
  for (const d of map.defs) {
    if (!byMod.has(d.module)) byMod.set(d.module, []);
    byMod.get(d.module).push(d);
  }
  // a module's column: the call distance of its nearest def
  const rank = new Map();
  for (const [k, ds] of byMod) rank.set(k, Math.min(...ds.map((d) => d.dist)));
  const ranks = [...new Set(rank.values())].sort((a, b) => a - b);
  const columns = ranks.map((r) =>
    [...byMod.keys()].filter((k) => rank.get(k) === r).sort((a, b) => byMod.get(b).length - byMod.get(a).length || map.modules[a].path.localeCompare(map.modules[b].path)),
  );
  const tiles = new Map();
  const regions = new Map();
  const order = [];
  // who calls each def, for the barycenter ordering below
  const callersOf = new Map();
  for (const d of map.defs) {
    for (const [to, , , value] of d.calls) {
      if (value || to === d.id) continue;
      if (!callersOf.has(to)) callersOf.set(to, new Set());
      callersOf.get(to).add(d.id);
    }
  }
  // the mean of a placed caller's centre along one axis, or null
  const bary = (id, axis) => {
    const ps = [...(callersOf.get(id) || [])].map((c) => tiles.get(c)).filter(Boolean);
    if (!ps.length) return null;
    return ps.reduce((s, t) => s + (axis === 'x' ? t.x + t.w / 2 : t.y + t.h / 2), 0) / ps.length;
  };
  const byBary = (vals, tie) => (a, b) => {
    const va = vals.get(a);
    const vb = vals.get(b);
    if (va == null && vb == null) return tie(a, b);
    if (va == null) return 1;
    if (vb == null) return -1;
    return va - vb || tie(a, b);
  };
  let x = 0;
  let height = 0;
  for (const col0 of columns) {
    // a column's modules in the order of their callers' heights (the
    // Sugiyama barycenter heuristic), so calls between columns cross less
    const modY = new Map(
      col0.map((k) => {
        const ys = byMod.get(k).map((d) => bary(d.id, 'y')).filter((v) => v != null);
        return [k, ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : null];
      }),
    );
    const col = col0.slice().sort(byBary(modY, (a, b) => col0.indexOf(a) - col0.indexOf(b)));
    let y = 0;
    let colW = 0;
    for (const k of col) {
      order.push(k);
      // rows by call distance; inside a row, under the callers' mean x
      // (one sweep: the callers above are placed first)
      const groups = new Map();
      for (const d of byMod.get(k).slice().sort((a, b) => a.dist - b.dist || a.line - b.line)) {
        if (!groups.has(d.dist)) groups.set(d.dist, []);
        groups.get(d.dist).push(d);
      }
      const ds = [];
      const placedHere = new Map();
      for (const g of groups.values()) {
        const vx = new Map(g.map((d) => {
          const ps = [...(callersOf.get(d.id) || [])].map((c) => placedHere.get(c)).filter((v) => v != null);
          return [d, ps.length ? ps.reduce((a, b) => a + b, 0) / ps.length : null];
        }));
        g.sort(byBary(vx, (a, b) => a.line - b.line));
        g.forEach((d, i) => placedHere.set(d.id, ds.length + i));
        ds.push(...g);
      }
      const name = map.modules[k].path.replace(/\.viv$/, '');
      const headW = name.length * CHAR_W + 60;
      if (collapsed.has(k)) {
        const w = Math.max(SHUT_W, headW);
        regions.set(k, { x, y, w, h: HEAD + 8, n: ds.length, rows: [], shut: true });
        y += HEAD + 8 + ROW_GAP;
        colW = Math.max(colW, w);
        continue;
      }
      const ws = ds.map(tileWidth);
      const area = ws.reduce((s, w) => s + (w + GAP_X) * (TILE_H + GAP_Y), 0);
      const inner = Math.max(headW - 2 * PAD, Math.min(720, Math.max(MAX_TILE_W, Math.sqrt(area) * 1.7)));
      let cx = 0;
      let cy = 0;
      let used = 0;
      const rowTops = [y + HEAD];
      ds.forEach((d, i) => {
        const w = ws[i];
        // each call distance starts a row: the defs a module's first def
        // calls sit below it
        const newDist = i > 0 && d.dist !== ds[i - 1].dist;
        if (cx > 0 && (cx + w > inner || newDist)) {
          cx = 0;
          cy += TILE_H + GAP_Y;
          rowTops.push(y + HEAD + cy);
        }
        const maxChars = Math.floor((w - 16) / CHAR_W);
        const label = labelOf(d).length > maxChars ? labelOf(d).slice(0, Math.max(3, maxChars - 1)) + '…' : labelOf(d);
        const intent = intentOf(d);
        const sub = intent ? cut(texToPlain(intent), Math.floor((w - 16) / INTENT_CHAR_W)) : '';
        tiles.set(d.id, { id: d.id, x: x + PAD + cx, y: y + HEAD + cy, w, h: TILE_H, label, sub, intent, module: k, row: rowTops.length - 1 });
        cx += w + GAP_X;
        used = Math.max(used, cx - GAP_X);
      });
      const w = Math.max(used, headW - 2 * PAD) + 2 * PAD;
      const h = HEAD + cy + TILE_H + PAD;
      regions.set(k, { x, y, w, h, n: ds.length, rows: rowTops });
      y += h + ROW_GAP;
      colW = Math.max(colW, w);
    }
    height = Math.max(height, y - ROW_GAP);
    x += colW + COL_GAP;
  }
  // the calls, one edge per caller and callee, by the number of call sites
  const count = new Map();
  for (const d of map.defs) {
    for (const [to, , , value] of d.calls) {
      if (value) continue;
      const key = `${d.id}\u0000${to}`;
      count.set(key, (count.get(key) || 0) + 1);
    }
  }
  const edges = [...count.entries()].map(([k, n]) => [...k.split('\u0000'), n]);
  // the lines of the root module inside some def (the rest is the top level)
  const spans = map.defs.filter((d) => !d.main).map((d) => [d.module, d.line, d.end]);
  const inDef = (k, n) => spans.some(([m, a, b]) => m === k && n >= a && n <= b);
  const modOf = new Map(map.defs.map((d) => [d.id, d.module]));
  return { tiles, regions, order, edges, modOf, width: Math.max(1, x - COL_GAP), height: Math.max(1, height), inDef };
}

/** The calls from the entry to `id`, along the first route the map found. */
export function routeOf(defs, id) {
  const out = [];
  const seen = new Set();
  let cur = defs.get(id);
  while (cur && !seen.has(cur.id)) {
    seen.add(cur.id);
    out.unshift(cur.id);
    cur = cur.from ? defs.get(cur.from) : null;
  }
  return out;
}

// ---------------------------------------------------------------- the calls
//
// A call is drawn as an orthogonal line that never passes over or touches a
// tile it does not connect (lib/codemap.test.mjs measures every map):
//
//   - it leaves the caller from the bottom, into the gap below the caller's
//     row, and enters the callee from the top, out of the gap above its row;
//   - inside one region it runs along those gaps between rows and up or down
//     the region's left padding, where there are no tiles;
//   - between regions it leaves through the side of its region along the same
//     gap, and outside the regions it is routed around them (a shortest path
//     with few bends on the grid of the regions' edges).
//
// The corners are rounded by one radius. Calls that share a gap are spread
// over three lanes.

const r2 = (v) => Math.round(v * 10) / 10;
const RADIUS = 6;
const LANE = 3;
// how far outside a region a line turns, and how far outside it the router
// keeps its lines
const STUB = 11;
const CLEAR = 6;
const BEND = 40;

/** A polyline through `pts` with each corner rounded. */
export function orthoPath(pts) {
  const p = simplify(pts);
  let d = `M${r2(p[0][0])},${r2(p[0][1])}`;
  for (let i = 1; i < p.length - 1; i++) {
    const [x0, y0] = p[i - 1];
    const [x1, y1] = p[i];
    const [x2, y2] = p[i + 1];
    const l1 = Math.hypot(x1 - x0, y1 - y0);
    const l2 = Math.hypot(x2 - x1, y2 - y1);
    const r = Math.min(RADIUS, l1 / 2, l2 / 2);
    const ax = x1 - ((x1 - x0) / (l1 || 1)) * r;
    const ay = y1 - ((y1 - y0) / (l1 || 1)) * r;
    const bx = x1 + ((x2 - x1) / (l2 || 1)) * r;
    const by = y1 + ((y2 - y1) / (l2 || 1)) * r;
    d += ` L${r2(ax)},${r2(ay)} Q${r2(x1)},${r2(y1)} ${r2(bx)},${r2(by)}`;
  }
  const z = p[p.length - 1];
  return `${d} L${r2(z[0])},${r2(z[1])}`;
}

/** The points with repeats and the middle of straight runs taken out. */
export function simplify(pts) {
  const out = [];
  for (const q of pts) {
    const last = out[out.length - 1];
    if (last && Math.abs(last[0] - q[0]) < 0.01 && Math.abs(last[1] - q[1]) < 0.01) continue;
    if (out.length >= 2) {
      const [a, b] = [out[out.length - 2], last];
      const colinear = (Math.abs(a[0] - b[0]) < 0.01 && Math.abs(b[0] - q[0]) < 0.01) || (Math.abs(a[1] - b[1]) < 0.01 && Math.abs(b[1] - q[1]) < 0.01);
      if (colinear) out.pop();
    }
    out.push(q);
  }
  return out;
}

/** A small binary heap of [cost, value]. */
class Heap {
  constructor() {
    this.a = [];
  }
  push(c, v) {
    const a = this.a;
    a.push([c, v]);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (a[p][0] <= a[i][0]) break;
      [a[p], a[i]] = [a[i], a[p]];
      i = p;
    }
  }
  pop() {
    const a = this.a;
    const top = a[0];
    const last = a.pop();
    if (a.length) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < a.length && a[l][0] < a[m][0]) m = l;
        if (r < a.length && a[r][0] < a[m][0]) m = r;
        if (m === i) break;
        [a[m], a[i]] = [a[i], a[m]];
        i = m;
      }
    }
    return top;
  }
  get size() {
    return this.a.length;
  }
}

/**
 * The router of one layout. `call(a, b)` gives the points of a call between
 * two defs (null when both are inside one folded module); `trunk(ma, mb)` the
 * points of the line that carries every call from module `ma` to module `mb`.
 */
export function makeRouter(lay) {
  const rects = [...lay.regions.values()].map((r) => ({ l: r.x - CLEAR, r: r.x + r.w + CLEAR, t: r.y - CLEAR, b: r.y + r.h + CLEAR }));
  const xs0 = new Set([-STUB * 4, lay.width + STUB * 4]);
  const ys0 = new Set([-STUB * 4, lay.height + STUB * 4]);
  for (const q of rects) {
    xs0.add(q.l).add(q.r);
    ys0.add(q.t).add(q.b);
  }
  const strictlyIn = (x, y) => rects.some((q) => x > q.l + 1e-6 && x < q.r - 1e-6 && y > q.t + 1e-6 && y < q.b - 1e-6);

  /** A shortest path with few bends from s to t around the regions. */
  function around(s, t) {
    const xs = [...new Set([...xs0, s[0], t[0]])].sort((a, b) => a - b);
    const ys = [...new Set([...ys0, s[1], t[1]])].sort((a, b) => a - b);
    const nx = xs.length;
    const ny = ys.length;
    const id = (i, j) => j * nx + i;
    const si = xs.indexOf(s[0]);
    const sj = ys.indexOf(s[1]);
    const ti = xs.indexOf(t[0]);
    const tj = ys.indexOf(t[1]);
    // dir 0 +x, 1 -x, 2 +y, 3 -y
    const DI = [1, -1, 0, 0];
    const DJ = [0, 0, 1, -1];
    const best = new Float64Array(nx * ny * 4).fill(Infinity);
    const prev = new Int32Array(nx * ny * 4).fill(-1);
    const h = new Heap();
    for (let d = 0; d < 4; d++) {
      best[id(si, sj) * 4 + d] = 0;
      h.push(0, id(si, sj) * 4 + d);
    }
    let goal = -1;
    while (h.size) {
      const [c, st] = h.pop();
      if (c > best[st]) continue;
      const node = st >> 2;
      const dir = st & 3;
      const i = node % nx;
      const j = (node - i) / nx;
      if (i === ti && j === tj) {
        goal = st;
        break;
      }
      for (let d = 0; d < 4; d++) {
        if ((d ^ 1) === dir && c > 0) continue;
        const i2 = i + DI[d];
        const j2 = j + DJ[d];
        if (i2 < 0 || j2 < 0 || i2 >= nx || j2 >= ny) continue;
        const mx = (xs[i] + xs[i2]) / 2;
        const my = (ys[j] + ys[j2]) / 2;
        if (strictlyIn(mx, my) || strictlyIn(xs[i2], ys[j2])) continue;
        const len = Math.abs(xs[i2] - xs[i]) + Math.abs(ys[j2] - ys[j]);
        const nc = c + len + (d !== dir && c > 0 ? BEND : 0);
        const ns = id(i2, j2) * 4 + d;
        if (nc < best[ns]) {
          best[ns] = nc;
          prev[ns] = st;
          h.push(nc, ns);
        }
      }
    }
    if (goal < 0) return [s, [s[0], t[1]], t];
    const out = [];
    for (let st = goal; st >= 0; st = prev[st]) {
      const node = st >> 2;
      const i = node % nx;
      out.push([xs[i], ys[(node - i) / nx]]);
      if (prev[st] < 0) break;
    }
    return out.reverse();
  }

  const regionOf = (id) => lay.regions.get(lay.modOf.get(id));
  // the gap below a tile's row and above it
  const below = (t, R) => (t.row + 1 < R.rows.length ? R.rows[t.row + 1] - GAP_Y / 2 : t.y + t.h + PAD / 2);
  const above = (t, R) => R.rows[t.row] - GAP_Y / 2;
  const outX = (t) => t.x + Math.min(t.w * 0.3, 22);
  const inX = (t) => t.x + Math.min(t.w * 0.5, 34);
  const lane = (key) => {
    let h = 0;
    for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) | 0;
    return (((h % 3) + 3) % 3) - 1;
  };

  /** Which sides of two regions a line between them leaves and enters by. */
  const sides = (RA, RB) => (RB.x >= RA.x + RA.w ? ['r', 'l'] : RB.x + RB.w <= RA.x ? ['l', 'r'] : ['l', 'l']);
  const edgeX = (R, side) => (side === 'l' ? R.x : R.x + R.w);
  const stubX = (R, side) => (side === 'l' ? R.x - STUB : R.x + R.w + STUB);

  function call(a, b) {
    const A = lay.tiles.get(a);
    const B = lay.tiles.get(b);
    const RA = regionOf(a);
    const RB = regionOf(b);
    if (!RA || !RB || (RA === RB && (!A || !B))) return null;
    const l = lane(`${a}>${b}`) * LANE;
    if (RA === RB) {
      // one open region: along the gaps and the left padding
      const ys = below(A, RA) + l;
      const yt = above(B, RB) + l;
      const sx = outX(A);
      const ex = inX(B);
      if (Math.abs(ys - yt) < 0.01) return [[sx, A.y + A.h], [sx, ys], [ex, ys], [ex, B.y]];
      const px = RA.x + PAD / 2 + l / 1.5;
      return [[sx, A.y + A.h], [sx, ys], [px, ys], [px, yt], [ex, yt], [ex, B.y]];
    }
    const [sa, sb] = sides(RA, RB);
    const head = [];
    let s;
    if (A) {
      const ys = below(A, RA) + l;
      head.push([outX(A), A.y + A.h], [outX(A), ys], [edgeX(RA, sa), ys]);
      s = [stubX(RA, sa), ys];
    } else {
      const y = RA.y + RA.h / 2;
      head.push([edgeX(RA, sa), y]);
      s = [stubX(RA, sa), y];
    }
    const tail = [];
    let t;
    if (B) {
      const yt = above(B, RB) + l;
      t = [stubX(RB, sb), yt];
      tail.push([edgeX(RB, sb), yt], [inX(B), yt], [inX(B), B.y]);
    } else {
      const y = RB.y + RB.h / 2;
      t = [stubX(RB, sb), y];
      tail.push([edgeX(RB, sb), y]);
    }
    return [...head, ...around(s, t), ...tail];
  }

  // the trunks' ports, spread along each side of each region
  const ports = new Map();
  function trunk(ma, mb, slotA, slotB) {
    const RA = lay.regions.get(ma);
    const RB = lay.regions.get(mb);
    const [sa, sb] = sides(RA, RB);
    const ya = RA.y + Math.min(RA.h - 6, 14 + slotA * 7);
    const yb = RB.y + Math.min(RB.h - 6, 14 + slotB * 7);
    return [[edgeX(RA, sa), ya], ...around([stubX(RA, sa), ya], [stubX(RB, sb), yb]), [edgeX(RB, sb), yb]];
  }
  return { call, trunk, ports };
}

/**
 * The calls between modules gathered into trunks: one per pair of modules,
 * with the number of distinct calls it carries, routed around the regions.
 */
export function trunks(lay, router) {
  const by = new Map();
  for (const [a, b, n] of lay.edges) {
    const ma = lay.modOf.get(a);
    const mb = lay.modOf.get(b);
    if (ma == null || mb == null || ma === mb) continue;
    const key = `${ma}>${mb}`;
    if (!by.has(key)) by.set(key, { from: ma, to: mb, calls: 0, sites: 0 });
    by.get(key).calls += 1;
    by.get(key).sites += n;
  }
  // each region side's trunks take the next port down that side
  const used = new Map();
  const slot = (k, side) => {
    const key = `${k}${side}`;
    const n = used.get(key) || 0;
    used.set(key, n + 1);
    return n;
  };
  return [...by.values()].map((t) => {
    const RA = lay.regions.get(t.from);
    const RB = lay.regions.get(t.to);
    const fwd = RB.x >= RA.x + RA.w;
    const back = RB.x + RB.w <= RA.x;
    const pts = router.trunk(t.from, t.to, slot(t.from, fwd ? 'r' : 'l'), slot(t.to, back ? 'r' : 'l'));
    return { ...t, pts, d: orthoPath(pts) };
  });
}

/** How many of the lines pass over or touch a tile they do not connect. */
export function tileHits(lay, lines) {
  const tiles = [...lay.tiles.values()];
  let hits = 0;
  const bad = [];
  for (const { pts, ends } of lines) {
    const p = simplify(pts);
    let hit = null;
    for (let i = 1; i < p.length && !hit; i++) {
      const [x0, y0] = p[i - 1];
      const [x1, y1] = p[i];
      const sx = [Math.min(x0, x1), Math.max(x0, x1)];
      const sy = [Math.min(y0, y1), Math.max(y0, y1)];
      for (const t of tiles) {
        if (ends.includes(t.id)) continue;
        // touching counts: the tile grown by one pixel
        if (sx[1] >= t.x - 1 && sx[0] <= t.x + t.w + 1 && sy[1] >= t.y - 1 && sy[0] <= t.y + t.h + 1) {
          hit = t.id;
          break;
        }
      }
    }
    if (hit) {
      hits++;
      bad.push([ends.join('>') || 'trunk', hit]);
    }
  }
  return { hits, bad };
}
