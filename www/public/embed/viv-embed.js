// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
/* viv-embed.js — one script tag, runnable Viv cells on any page.
 *
 *   <script src="https://.../embed/viv-embed.js" defer></script>
 *   <viv-cell>
 *   x = linspace(0.0, 6.0, 61)
 *   ...
 *   </viv-cell>
 *
 * Every <viv-cell> becomes an editable, runnable cell: syntax highlighting,
 * text output, live narration lines, plot() charts, and the animated call
 * graph — the same experience as the Viv docs, with zero dependencies and no
 * server (the whole compiler is a ~1.7 MB wasm fetched next to this script).
 * Markdown-rendered blocks (<pre><code class="language-viv">) are upgraded
 * too. Add the `run` attribute to auto-run a cell on load. Styles live in
 * shadow DOM, so the host page's CSS is untouched and untouching.
 */
(function () {
  'use strict';

  // ---- wasm ---------------------------------------------------------------
  var SCRIPT_SRC = (document.currentScript && document.currentScript.src) || '';
  var WASM_URL = new URL('viv.wasm', SCRIPT_SRC || location.href).href;
  var wasmPromise = null;
  function loadWasm() {
    if (!wasmPromise) {
      wasmPromise = (async function () {
        var instance;
        // a page that already has the compiler (viv-lang.org's own pages) lends
        // its compiled module, so viv.wasm is not fetched a second time
        if (typeof window.__vivModule === 'function') {
          instance = await WebAssembly.instantiate(await window.__vivModule(), {});
          return instance.exports;
        }
        try {
          instance = (await WebAssembly.instantiateStreaming(fetch(WASM_URL), {})).instance;
        } catch (e) {
          var bytes = await (await fetch(WASM_URL)).arrayBuffer();
          instance = (await WebAssembly.instantiate(bytes, {})).instance;
        }
        return instance.exports;
      })();
    }
    return wasmPromise;
  }
  function runViv(ex, text) {
    var enc = new TextEncoder().encode(text);
    var ptr = ex.viv_alloc(enc.length);
    new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
    var packed = ex.viv_run(ptr, enc.length);
    return new TextDecoder().decode(
      new Uint8Array(ex.memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn)));
  }

  // ---- syntax highlighting ------------------------------------------------
  var KEYWORDS = ['def', 'if', 'else', 'while', 'for', 'to', 'return', 'break', 'continue', 'print', 'show', 'narrate', 'test', 'fn', 'record', 'parallel', 'and', 'or', 'not'];
  var BUILTINS = ['zeros', 'len', 'sqrt', 'exp', 'log', 'sin', 'cos', 'tan', 'abs', 'floor', 'ceil', 'pow', 'min', 'max', 'eig_sym', 'transpose', 'simpson', 'trapezoid', 'deriv', 'assert_near', 'linspace', 'plot', 'plot2d', 'plot3d', 'complex', 're', 'im', 'conj', 'fill', 'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh', 'erf', 'gamma', 'round', 'mod'];
  var KW = new Set(KEYWORDS), BI = new Set(BUILTINS);
  function esc(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  function highlight(src) {
    return src.split('\n').map(function (line) {
      var out = '', i = 0;
      while (i < line.length) {
        var rest = line.slice(i), m;
        if ((m = rest.match(/^#.*$/))) out += '<span class="c">' + esc(m[0]) + '</span>';
        else if ((m = rest.match(/^"[^"]*"?/))) out += '<span class="s">' + esc(m[0]) + '</span>';
        else if ((m = rest.match(/^\d+\.?\d*([eE][+-]?\d+)?/))) out += '<span class="n">' + esc(m[0]) + '</span>';
        else if ((m = rest.match(/^[a-zA-Z_][a-zA-Z0-9_]*/))) {
          var w = m[0];
          out += KW.has(w) ? '<span class="k">' + w + '</span>'
            : BI.has(w) ? '<span class="f">' + w + '</span>' : w;
        } else if ((m = rest.match(/^(->|[<>=!+\-*\/@%]+)/))) out += '<span class="o">' + esc(m[0]) + '</span>';
        else { m = [rest[0]]; out += esc(rest[0]); }
        i += m[0].length;
      }
      return out;
    }).join('\n');
  }

  // ---- output parsing (plot / narrate / graph / trace) --------------------
  function parseOutput(text) {
    var blocks = [], charts = [], graph = null, trace = [];
    // figure(title, xlabel, ylabel) prints a `figure:` record; the plots after
    // it are one chart with that title and those axis labels
    var fig = null, freshFig = false;
    function pushText(line) {
      var last = blocks[blocks.length - 1];
      if (last && last.type === 'text') last.s += '\n' + line;
      else blocks.push({ type: 'text', s: line });
    }
    text.split('\n').forEach(function (line) {
      var f = line.match(/^figure: (.*?) ; (.*?) ; (.*?)\s*$/);
      if (f) { fig = { title: f[1].trim(), x: f[2].trim(), y: f[3].trim() }; freshFig = true; return; }
      var p = line.match(/^plot: (.*?) ; \[([^\]]*)\] ; \[([^\]]*)\]\s*$/);
      if (p) {
        var xs = p[2].split(',').map(Number).filter(isFinite);
        var ys = p[3].split(',').map(Number).filter(isFinite);
        if (xs.length && xs.length === ys.length) {
          var last = charts[charts.length - 1];
          if (last && last.key === p[2] && !freshFig) last.series.push({ name: p[1].trim(), values: ys });
          else charts.push({ key: p[2], x: xs, fig: fig, series: [{ name: p[1].trim(), values: ys }] });
          freshFig = false;
        }
        return;
      }
      var p2 = line.match(/^plot2d: (.*?) ; (\d+) (\d+) ;/);
      if (p2) { blocks.push({ type: 'narrate', s: '[heatmap] ' + (p2[1].trim() || 'unnamed') + ' (' + p2[2] + 'x' + p2[3] + ') - open in the full IDE to view' }); return; }
      var p3 = line.match(/^plot3d: (.*?) ; \[/);
      if (p3) { blocks.push({ type: 'narrate', s: '[3d curve] ' + (p3[1].trim() || 'unnamed') + ' - open in the full IDE to rotate' }); return; }
      var nar = line.match(/^narrate: (.*)$/);
      if (nar) { blocks.push({ type: 'narrate', s: nar[1] }); return; }
      var g = line.match(/^__graph__ (.*)$/);
      if (g) { try { graph = JSON.parse(g[1]); } catch (e) {} return; }
      var t = line.match(/^__trace__ (.*)$/);
      if (t) { try { trace = JSON.parse(t[1]).map(function (e) { return e.split('>'); }); } catch (e) {} return; }
      if (line.indexOf('__steps__ ') === 0) return; // step events: lens data, not output
      if (line.indexOf('__story__ ') === 0) return; // the story of a debug run: lens data, not output
      pushText(line);
    });
    blocks = blocks.map(function (b) {
      return b.type === 'text' ? { type: 'text', s: b.s.replace(/^\n+|\n+$/g, '') } : b;
    }).filter(function (b) { return b.type !== 'text' || b.s !== ''; });
    return { blocks: blocks, charts: charts, graph: graph, trace: trace };
  }

  // ---- charts -------------------------------------------------------------
  var COLORS = ['#2f7fd1', '#1a9e5b', '#c05fd6', '#e0913a'];
  function chartSVG(chart) {
    // Mirrors the site's professional chart: 1-2-5 nice ticks, gridlines,
    // padded y-range, markers on sparse series.
    var fg = chart.fig;
    var W = 520, H = fg ? 290 : 260, P = { l: fg && fg.y ? 70 : 58, r: 14, t: fg && fg.title ? 34 : 16, b: fg && fg.x ? 50 : 36 };
    var xs = chart.x, all = [];
    chart.series.forEach(function (se) { all = all.concat(se.values); });
    var xlo = Math.min.apply(null, xs), xhi = Math.max.apply(null, xs);
    var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
    var pad = (hi - lo || Math.abs(hi) || 1) * 0.06;
    lo -= pad; hi += pad;
    function X(x) { return P.l + (x - xlo) / (xhi - xlo || 1) * (W - P.l - P.r); }
    function Y(v) { return P.t + (1 - (v - lo) / (hi - lo || 1)) * (H - P.t - P.b); }
    function ticks(a, b, n) {
      var span = b - a || 1, step0 = span / Math.max(1, n);
      var mag = Math.pow(10, Math.floor(Math.log(step0) / Math.LN10));
      var norm = step0 / mag;
      var step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
      var out = [];
      for (var t = Math.ceil(a / step) * step; t <= b + step * 1e-9; t += step)
        out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
      return out;
    }
    function fmt(v) {
      if (v === 0) return '0';
      var a = Math.abs(v);
      if (a >= 1e5 || a < 1e-3) return v.toExponential(1).replace('e+', 'e');
      return String(+v.toPrecision(4));
    }
    var sparse = xs.length <= 24;
    var svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="chart" role="img">';
    // labels: the embed sets them as plain text ($ signs dropped); the site's
    // own charts typeset the math
    var plain = function (t) { return esc(String(t).replace(/\$/g, '').replace(/\\(?=[a-zA-Z])/g, '')); };
    if (fg && fg.title) svg += '<text x="' + P.l + '" y="18" class="ttl">' + plain(fg.title) + '</text>';
    if (fg && fg.x) svg += '<text x="' + ((P.l + W - P.r) / 2) + '" y="' + (H - 10) + '" class="axl" text-anchor="middle">' + plain(fg.x) + '</text>';
    if (fg && fg.y) svg += '<text transform="translate(16 ' + ((P.t + H - P.b) / 2) + ') rotate(-90)" class="axl" text-anchor="middle">' + plain(fg.y) + '</text>';
    ticks(lo, hi, 4).forEach(function (t) {
      svg += '<line x1="' + P.l + '" y1="' + Y(t) + '" x2="' + (W - P.r) + '" y2="' + Y(t) + '" class="grid"/>';
      svg += '<text x="' + (P.l - 7) + '" y="' + (Y(t) + 3.5) + '" class="tick" text-anchor="end">' + fmt(t) + '</text>';
    });
    ticks(xlo, xhi, 5).forEach(function (t) {
      svg += '<line x1="' + X(t) + '" y1="' + (H - P.b) + '" x2="' + X(t) + '" y2="' + (H - P.b + 4) + '" class="ax"/>';
      svg += '<text x="' + X(t) + '" y="' + (H - P.b + 16) + '" class="tick" text-anchor="middle">' + fmt(t) + '</text>';
    });
    if (lo < 0 && hi > 0) svg += '<line x1="' + P.l + '" y1="' + Y(0) + '" x2="' + (W - P.r) + '" y2="' + Y(0) + '" class="zero"/>';
    svg += '<line x1="' + P.l + '" y1="' + P.t + '" x2="' + P.l + '" y2="' + (H - P.b) + '" class="ax"/>';
    svg += '<line x1="' + P.l + '" y1="' + (H - P.b) + '" x2="' + (W - P.r) + '" y2="' + (H - P.b) + '" class="ax"/>';
    chart.series.forEach(function (se, k) {
      var col = COLORS[k % COLORS.length];
      var pts = se.values.map(function (v, i) { return X(xs[i]).toFixed(1) + ',' + Y(v).toFixed(1); }).join(' ');
      svg += '<polyline points="' + pts + '" fill="none" stroke="' + col + '" stroke-width="2"' + (k > 0 ? ' stroke-dasharray="7 5"' : '') + ' stroke-linecap="round" stroke-linejoin="round"/>';
      if (sparse) se.values.forEach(function (v, i) {
        svg += '<circle cx="' + X(xs[i]) + '" cy="' + Y(v) + '" r="2.6" fill="' + col + '"/>';
      });
    });
    var ly = P.t + 6;
    chart.series.forEach(function (se, k) {
      if (!se.name) return;
      svg += '<text x="' + (W - P.r - 3) + '" y="' + ly + '" text-anchor="end" class="leg" fill="' + COLORS[k % COLORS.length] + '">' + esc(se.name) + '</text>';
      ly += 15;
    });
    return svg + '</svg>';
  }

  // ---- call graph (layered layout + collision-aware labels + replay) ------
  function layoutGraph(graph) {
    var nodes = graph.nodes.slice();
    var mi = nodes.indexOf('<main>');
    if (mi > 0) { nodes.splice(mi, 1); nodes.unshift('<main>'); }
    var layer = {};
    nodes.forEach(function (n) { layer[n] = n === '<main>' ? 0 : 1; });
    for (var pass = 0; pass <= nodes.length; pass++) {
      var moved = false;
      graph.edges.forEach(function (e) {
        if (e.from === e.to) return;
        if (layer[e.to] < layer[e.from] + 1 && layer[e.from] + 1 <= nodes.length) {
          layer[e.to] = layer[e.from] + 1; moved = true;
        }
      });
      if (!moved) break;
    }
    var cols = [];
    nodes.forEach(function (n) { (cols[layer[n]] = cols[layer[n]] || []).push(n); });
    var columns = cols.filter(Boolean);
    var row = {};
    columns.forEach(function (col) { col.forEach(function (n, i) { row[n] = i; }); });
    columns.forEach(function (col, ci) {
      if (!ci) return;
      col.sort(function (a, b) {
        function bary(n) {
          var rs = graph.edges.filter(function (e) { return e.to === n && e.from !== n; })
            .map(function (e) { return row[e.from] || 0; });
          return rs.length ? rs.reduce(function (x, y) { return x + y; }, 0) / rs.length : 0;
        }
        return bary(a) - bary(b);
      });
      col.forEach(function (n, i) { row[n] = i; });
    });
    var COLW = 148, ROWH = 44, PADX = 12, PADY = 26;
    var H = Math.max.apply(null, columns.map(function (c) { return c.length; })) * ROWH + PADY * 2;
    var W = columns.length * COLW + PADX * 2 + 30;
    var pos = {};
    columns.forEach(function (col, ci) {
      var y0 = (H - col.length * ROWH) / 2 + ROWH / 2;
      col.forEach(function (n, ri) { pos[n] = { x: PADX + ci * COLW + 46, y: y0 + ri * ROWH }; });
    });
    return { pos: pos, W: W, H: H };
  }

  function boxW(n) { return 16 + n.length * 7.2; }

  function graphSVG(graph) {
    var L = layoutGraph(graph), pos = L.pos;
    var maxCount = Math.max.apply(null, graph.edges.map(function (e) { return e.count; }));
    function cubic(P0, P1, P2, P3) {
      return function (t) {
        var u = 1 - t;
        return {
          x: u * u * u * P0.x + 3 * u * u * t * P1.x + 3 * u * t * t * P2.x + t * t * t * P3.x,
          y: u * u * u * P0.y + 3 * u * u * t * P1.y + 3 * u * t * t * P2.y + t * t * t * P3.y,
        };
      };
    }
    var geoms = graph.edges.map(function (e) {
      var a = pos[e.from], b = pos[e.to];
      if (e.from === e.to) {
        var s0 = { x: a.x - 8, y: a.y - 12 }, s1 = { x: a.x - 20, y: a.y - 34 };
        var s2 = { x: a.x + 20, y: a.y - 34 }, s3 = { x: a.x + 8, y: a.y - 12 };
        return { e: e, self: true, d: 'M ' + s0.x + ' ' + s0.y + ' C ' + s1.x + ' ' + s1.y + ', ' + s2.x + ' ' + s2.y + ', ' + s3.x + ' ' + s3.y, at: cubic(s0, s1, s2, s3) };
      }
      var x1 = a.x + boxW(e.from) / 2 + 2, y1 = a.y;
      var x2 = b.x - boxW(e.to) / 2 - 4, y2 = b.y;
      if (x2 < x1) {
        var b0 = { x: x1, y: y1 + 8 }, b1 = { x: x1 + 24, y: y1 + 34 };
        var b2 = { x: x2 - 24, y: y2 + 34 }, b3 = { x: x2, y: y2 + 8 };
        return { e: e, back: true, d: 'M ' + b0.x + ' ' + b0.y + ' C ' + b1.x + ' ' + b1.y + ', ' + b2.x + ' ' + b2.y + ', ' + b3.x + ' ' + b3.y, at: cubic(b0, b1, b2, b3) };
      }
      var mx = (x1 + x2) / 2;
      var span = Math.abs(b.x - a.x);
      var bow = span > 190 ? Math.min(34, 12 + span / 14) : 0;
      var f0 = { x: x1, y: y1 }, f1 = { x: mx, y: y1 - bow }, f2 = { x: mx, y: y2 - bow }, f3 = { x: x2, y: y2 };
      return { e: e, d: 'M ' + f0.x + ' ' + f0.y + ' C ' + f1.x + ' ' + f1.y + ', ' + f2.x + ' ' + f2.y + ', ' + f3.x + ' ' + f3.y, at: cubic(f0, f1, f2, f3) };
    });
    // collision-aware count labels
    var taken = graph.nodes.map(function (n) {
      return { x: pos[n].x - boxW(n) / 2 - 2, y: pos[n].y - 14, w: boxW(n) + 4, h: 28 };
    });
    function clear(r) {
      return !taken.some(function (o) {
        return r.x < o.x + o.w && o.x < r.x + r.w && r.y < o.y + o.h && o.y < r.y + r.h;
      });
    }
    var CAND = [0.5, 0.42, 0.58, 0.34, 0.66, 0.26, 0.74, 0.18, 0.82];
    var labels = {};
    geoms.slice().sort(function (a, b) { return b.e.count - a.e.count; }).forEach(function (g) {
      var text = String(g.e.count), w = text.length * 6.4 + 4, h = 12, best = null;
      outer: for (var ci = 0; ci < CAND.length; ci++) {
        var c = g.at(CAND[ci]);
        var dys = g.self ? [-8, -14] : [-7, 9, -15, 17];
        for (var di = 0; di < dys.length; di++) {
          var r = { x: c.x - w / 2, y: c.y + dys[di] - h + 3, w: w, h: h };
          if (clear(r)) { best = { x: c.x, y: c.y + dys[di] }; break outer; }
        }
      }
      if (!best) { var c2 = g.at(0.5); best = { x: c2.x, y: c2.y - 7 }; }
      taken.push({ x: best.x - w / 2, y: best.y - h + 3, w: w, h: h });
      labels[g.e.from + '>' + g.e.to] = best;
    });
    var svg = '<svg viewBox="0 0 ' + L.W + ' ' + L.H + '" class="cg" style="max-width:' + L.W + 'px">';
    svg += '<defs><marker id="varr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" class="arrhead"/></marker></defs>';
    geoms.forEach(function (g) {
      var w = 1.1 + 2.2 * Math.log(1 + g.e.count) / Math.log(1 + maxCount);
      svg += '<g class="edge" data-edge="' + esc(g.e.from + '>' + g.e.to) + '"><path d="' + g.d + '" fill="none" stroke-width="' + w + '" marker-end="url(#varr)" class="eline"/></g>';
    });
    graph.nodes.forEach(function (n) {
      var p = pos[n];
      svg += '<g class="node" data-node="' + esc(n) + '"><rect x="' + (p.x - boxW(n) / 2) + '" y="' + (p.y - 12) + '" width="' + boxW(n) + '" height="24" rx="12" class="nbox"/><text x="' + p.x + '" y="' + (p.y + 4) + '" text-anchor="middle" class="nname">' + esc(n) + '</text></g>';
    });
    geoms.forEach(function (g) {
      var lp = labels[g.e.from + '>' + g.e.to];
      svg += '<text x="' + lp.x + '" y="' + lp.y + '" text-anchor="middle" class="count" data-count="' + esc(g.e.from + '>' + g.e.to) + '">' + g.e.count + '</text>';
    });
    return svg + '</svg>';
  }

  // ---- the component ------------------------------------------------------
  var STYLE = [
    // the host page can set the colors: --viv-cell-accent, -bg, -card, -border,
    // -text and -muted on the <viv-cell> (or any ancestor) replace the defaults
    ':host { display: block; margin: 1em 0; font-family: ui-sans-serif, system-ui, sans-serif; --acc: var(--viv-cell-accent, #6c7bdd); --bg: var(--viv-cell-bg, #12172b); --card: var(--viv-cell-card, #1a2138); --border: var(--viv-cell-border, #2c3554); --text: var(--viv-cell-text, #e6e9f5); --muted: var(--viv-cell-muted, #9aa3bf); }',
    ':host([light]) { --bg: var(--viv-cell-bg, #f7f8fc); --card: var(--viv-cell-card, #ffffff); --border: var(--viv-cell-border, #d5dae8); --text: var(--viv-cell-text, #1c2237); --muted: var(--viv-cell-muted, #5c6580); }',
    '.cell { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; background: var(--card); }',
    '.head { display: flex; align-items: center; gap: 10px; padding: 6px 10px; border-bottom: 1px solid var(--border); }',
    '.tag { font: 600 11px ui-monospace, monospace; color: var(--muted); letter-spacing: 0.03em; }',
    '.run { font: 600 12px inherit; padding: 3px 12px; border-radius: 6px; border: 1px solid var(--acc); background: var(--acc); color: #fff; cursor: pointer; }',
    '.run:disabled { opacity: 0.6; }',
    '.ms { font: 11px ui-monospace, monospace; color: var(--muted); }',
    '.err { color: #d33; font-weight: 700; }',
    '.ed { position: relative; background: var(--bg); }',
    '.hl, .src { margin: 0; padding: 10px 12px; font: 12.5px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre; overflow-x: auto; tab-size: 4; }',
    '.hl { pointer-events: none; color: var(--text); min-height: 100%; }',
    '.src { position: absolute; inset: 0; width: 100%; height: 100%; border: 0; outline: 0; resize: none; background: transparent; color: transparent; caret-color: var(--text); }',
    '.c { color: #8b95a7; font-style: italic; } .s { color: #d0a35c; } .n { color: #a48fe8; } .k { color: #e06c8c; font-weight: 600; } .f { color: #5f96e0; } .o { color: #7fae8b; }',
    '.out { font: 12.5px/1.55 ui-monospace, monospace; color: var(--text); padding: 8px 12px; border-top: 1px dashed var(--border); margin: 0; overflow-x: auto; white-space: pre; }',
    '.nar { font: italic 600 12px inherit; color: var(--acc); padding: 4px 12px; border-top: 1px dashed var(--border); }',
    '.nar + .nar { border-top: 0; padding-top: 0; }',
    '.chart { display: block; width: 100%; max-width: 560px; margin: 6px auto; }',
    '.ax { stroke: var(--muted); } .grid { stroke: var(--border); } .zero { stroke: var(--muted); stroke-dasharray: 2 3; }',
    '.tick { font: 10.5px ui-monospace, monospace; fill: var(--muted); } .leg { font: 700 10.5px ui-monospace, monospace; }',
    '.ttl { font: 600 13px system-ui, sans-serif; fill: var(--fg, currentColor); } .axl { font: 11.5px system-ui, sans-serif; fill: var(--muted); }',
    '.gwrap { border-top: 1px dashed var(--border); padding: 2px 8px 8px; }',
    '.ghead { display: flex; align-items: center; gap: 10px; padding: 2px 4px; }',
    '.disc { font: 600 11px ui-monospace, monospace; padding: 4px 8px; border: 0; background: none; color: var(--muted); cursor: pointer; }',
    '.disc:hover { color: var(--acc); }',
    '.replay { font: 600 11px inherit; padding: 2px 9px; border-radius: 6px; border: 1px solid var(--border); background: none; color: var(--muted); cursor: pointer; }',
    '.replay:hover { border-color: var(--acc); color: var(--acc); }',
    '.step { font: 11px ui-monospace, monospace; color: var(--acc); }',
    '.cg { display: block; margin: 0 auto; width: 100%; }',
    '.nbox { fill: var(--bg); stroke: var(--border); } .nname { font: 600 11.5px ui-monospace, monospace; fill: var(--text); }',
    '.eline { stroke: var(--muted); opacity: 0.75; } .arrhead { fill: var(--muted); }',
    '.count { font: 10px ui-monospace, monospace; fill: var(--muted); paint-order: stroke; stroke: var(--card); stroke-width: 3px; stroke-linejoin: round; }',
    '.edge.dim .eline { opacity: 0.18; } .count.dim { opacity: 0.3; }',
    '.edge.active .eline { stroke: var(--acc); opacity: 1; } .count.active { fill: var(--acc); font-weight: 700; }',
    '.node.pulse .nbox { stroke: var(--acc); stroke-width: 2; }',
  ].join('\n');

  function dedent(text) {
    var lines = text.replace(/^\n+/, '').replace(/\s+$/, '').split('\n');
    var indents = lines.filter(function (l) { return l.trim(); })
      .map(function (l) { return l.match(/^[ \t]*/)[0].length; });
    var cut = indents.length ? Math.min.apply(null, indents) : 0;
    return lines.map(function (l) { return l.slice(cut); }).join('\n');
  }

  var VivCell = function () { return Reflect.construct(HTMLElement, [], VivCell); };
  VivCell.prototype = Object.create(HTMLElement.prototype);
  VivCell.prototype.connectedCallback = function () {
    if (this.__init) return;
    this.__init = true;
    var self = this;
    var code = this.getAttribute('code') || dedent(this.textContent || '');
    this.textContent = '';
    var root = this.attachShadow({ mode: 'open' });
    root.innerHTML = '<style>' + STYLE + '</style>' +
      '<div class="cell"><div class="head"><span class="tag">Viv</span>' +
      '<button class="run">▶ Run</button><span class="ms"></span></div>' +
      '<div class="ed"><pre class="hl" aria-hidden="true"></pre>' +
      '<textarea class="src" spellcheck="false" aria-label="Viv code"></textarea></div>' +
      '<div class="results"></div></div>';
    var hl = root.querySelector('.hl'), ta = root.querySelector('.src');
    var btn = root.querySelector('.run'), ms = root.querySelector('.ms');
    var results = root.querySelector('.results');
    function sync() {
      hl.innerHTML = highlight(ta.value) + '\n';
      ta.style.height = 'auto';
      ta.style.height = hl.scrollHeight + 'px';
    }
    ta.value = code;
    ta.addEventListener('input', sync);
    ta.addEventListener('scroll', function () { hl.scrollLeft = ta.scrollLeft; });
    ta.addEventListener('keydown', function (e) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); go(); }
    });
    sync();

    var replayTimer = null;
    function renderGraph(graph, trace) {
      var wrap = document.createElement('div');
      wrap.className = 'gwrap';
      var total = graph.edges.reduce(function (a, e) { return a + e.count; }, 0);
      wrap.innerHTML = '<div class="ghead"><button class="disc">▸ call graph — ' +
        graph.nodes.length + ' nodes, ' + total + ' calls</button>' +
        '<button class="replay" hidden>▶ replay</button><span class="step"></span></div>' +
        '<div class="gsvg" hidden></div>';
      var disc = wrap.querySelector('.disc'), rep = wrap.querySelector('.replay');
      var step = wrap.querySelector('.step'), holder = wrap.querySelector('.gsvg');
      function replay() {
        clearInterval(replayTimer);
        var edges = holder.querySelectorAll('.edge'), counts = holder.querySelectorAll('.count');
        var nodes = holder.querySelectorAll('.node');
        var i = 0;
        replayTimer = setInterval(function () {
          if (i >= trace.length) {
            clearInterval(replayTimer);
            edges.forEach(function (e) { e.classList.remove('dim', 'active'); });
            counts.forEach(function (c) { c.classList.remove('dim', 'active'); });
            nodes.forEach(function (n) { n.classList.remove('pulse'); });
            step.textContent = '';
            return;
          }
          var key = trace[i][0] + '>' + trace[i][1];
          var seen = {};
          for (var j = 0; j <= i; j++) seen[trace[j][0] + '>' + trace[j][1]] = 1;
          edges.forEach(function (e) {
            var k = e.getAttribute('data-edge');
            e.classList.toggle('active', k === key);
            e.classList.toggle('dim', !seen[k]);
          });
          counts.forEach(function (c) {
            var k = c.getAttribute('data-count');
            c.classList.toggle('active', k === key);
            c.classList.toggle('dim', !seen[k]);
          });
          nodes.forEach(function (n) {
            n.classList.toggle('pulse', n.getAttribute('data-node') === trace[i][1]);
          });
          step.textContent = 'call ' + (i + 1) + ' / ' + trace.length;
          i++;
        }, 140);
      }
      disc.addEventListener('click', function () {
        var open = holder.hidden;
        holder.hidden = !open;
        rep.hidden = !open;
        disc.textContent = open ? '▾ call graph' : '▸ call graph — ' + graph.nodes.length + ' nodes, ' + total + ' calls';
        if (open) { holder.innerHTML = graphSVG(graph); if (trace.length) replay(); }
        else { clearInterval(replayTimer); step.textContent = ''; }
      });
      rep.addEventListener('click', replay);
      return wrap;
    }

    function go() {
      btn.disabled = true;
      ms.textContent = 'running…';
      ms.classList.remove('err');
      loadWasm().then(function (ex) {
        var t0 = performance.now();
        var raw = runViv(ex, ta.value);
        var dt = performance.now() - t0;
        clearInterval(replayTimer);
        results.textContent = '';
        var parsed = parseOutput(raw);
        var isErr = raw.indexOf('error:') === 0 || raw.indexOf('\nerror:') >= 0;
        parsed.blocks.forEach(function (b) {
          var el = document.createElement(b.type === 'narrate' ? 'div' : 'pre');
          el.className = b.type === 'narrate' ? 'nar' : 'out';
          el.textContent = b.type === 'narrate' ? '▸ ' + b.s : b.s;
          results.appendChild(el);
        });
        if (!isErr) {
          parsed.charts.forEach(function (c) {
            var d = document.createElement('div');
            d.innerHTML = chartSVG(c);
            results.appendChild(d.firstChild);
          });
          if (parsed.graph && parsed.graph.edges.length) {
            results.appendChild(renderGraph(parsed.graph, parsed.trace));
          }
        }
        ms.textContent = dt.toFixed(0) + ' ms';
        if (isErr) { ms.textContent = 'error'; ms.classList.add('err'); }
        btn.disabled = false;
      }).catch(function (e) {
        results.textContent = '';
        var pre = document.createElement('pre');
        pre.className = 'out';
        pre.textContent = String(e);
        results.appendChild(pre);
        ms.textContent = 'error';
        ms.classList.add('err');
        btn.disabled = false;
      });
    }
    btn.addEventListener('click', go);
    if (this.hasAttribute('run')) loadWasm().then(go);
  };
  customElements.define('viv-cell', VivCell);

  // Upgrade markdown-rendered code blocks: <pre><code class="language-viv">
  function upgradeMarkdown() {
    document.querySelectorAll('pre > code.language-viv, pre.viv').forEach(function (el) {
      var pre = el.tagName === 'PRE' ? el : el.parentElement;
      var cell = document.createElement('viv-cell');
      cell.setAttribute('code', dedent(el.textContent || ''));
      pre.replaceWith(cell);
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', upgradeMarkdown);
  else upgradeMarkdown();
})();
