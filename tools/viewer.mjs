#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// A local code viewer for the repository, reading git LIVE on every request.
//
//     node tools/viewer.mjs            then open http://localhost:4400
//     node tools/viewer.mjs 4500       on another port
//
// It exists so work in flight can be read before it is merged: every branch is a
// choice in the top bar, and a branch opens on the files it changed against main,
// each with its diff. Nothing is cached and nothing is written -- reload the page
// and it shows what the branch holds now. The syntax colors are the site's own
// (www/src/lib/highlight.js), so code looks here the way it will on viv-lang.org.
//
// No dependencies beyond Node. It only runs `git` read commands, and, for the
// lenses of a .viv file, the viv binary of
// the worktree that has the viewed branch checked out:
//
//     code | intent | debug | story   tabs on a .viv file; `l` rotates the lens
//                                     (`L` backwards) keeping the same def and
//                                     step in view; `s` splits the view into
//                                     two lenses, scroll-locked by anchor.
//
// The lenses read the files on disk in that worktree (a lens needs the .intent
// and .debug files and the modules a program uses, not one blob), and say so
// when those files differ from the commit. VIV=<path> runs a different binary.
// Without a worktree or a binary the File view works as before and the lens
// panes say what is missing.
import http from 'node:http';
import { execFileSync, execFile } from 'node:child_process';
import { readFileSync, existsSync, realpathSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { highlight } from '../www/src/lib/highlight.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PORT = Number(process.argv[2] || 4400);
// The site's stylesheet: its palette variables and tk-* token classes are what
// the highlighted code is colored with.
const CSS = readFileSync(path.join(ROOT, 'www/src/styles/viv.css'), 'utf8');

const git = (...args) =>
  execFileSync('git', ['-C', ROOT, ...args], { encoding: 'utf8', maxBuffer: 256 << 20 });
const tryGit = (...args) => {
  try {
    return git(...args);
  } catch {
    return '';
  }
};

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

// A branch name comes from the URL, so it is checked against the real list
// before it reaches git.
function refs() {
  const out = git('for-each-ref', '--sort=-committerdate', '--format=%(refname:short)\t%(committerdate:relative)\t%(subject)', 'refs/heads');
  return out
    .trim()
    .split('\n')
    .filter(Boolean)
    .map((l) => {
      const [name, when, subject] = l.split('\t');
      const ahead = name === 'main' ? 0 : Number(tryGit('rev-list', '--count', `main..${name}`).trim() || 0);
      return { name, when, subject, ahead };
    });
}

const LANG = (p) => {
  if (p.endsWith('.viv')) return 'viv';
  if (p.endsWith('.py')) return 'py';
  if (p.endsWith('.rs')) return 'rs';
  if (p.endsWith('.js') || p.endsWith('.mjs') || p.endsWith('.jsx')) return 'js';
  if (p.endsWith('.sh')) return 'sh';
  if (p.endsWith('.c') || p.endsWith('.h')) return 'c';
  return '';
};

const SOURCE = /\.(viv|py|rs|mjs|js|jsx|sh|md|tsv|c|h|toml)$/;

function tree(ref) {
  return git('ls-tree', '-r', '--name-only', ref, '--')
    .split('\n')
    .filter((p) => p && SOURCE.test(p) && !p.startsWith('www/public/') && !p.includes('/node_modules/'));
}

function changed(ref) {
  if (ref === 'main') return [];
  const base = tryGit('merge-base', 'main', ref).trim();
  if (!base) return [];
  return git('diff', '--numstat', base, ref, '--')
    .trim()
    .split('\n')
    .filter(Boolean)
    .map((l) => {
      const [add, del, p] = l.split('\t');
      return { path: p, add: Number(add) || 0, del: Number(del) || 0 };
    });
}

// Generated data modules hold literals thousands of numbers long. They are
// shown, but a line past this many characters is cut with a count of what was
// left out, so a data file does not bury the code around it.
const LONG = 400;
function clip(line) {
  if (line.length <= LONG) return { text: line, cut: 0 };
  return { text: line.slice(0, LONG), cut: line.length - LONG };
}

function outline(src) {
  const items = [];
  src.split('\n').forEach((l, i) => {
    let m;
    if ((m = /^def\s+([A-Za-z_][\w]*)\s*\(([^)]*)\)/.exec(l))) items.push({ kind: 'def', name: m[1], args: m[2], line: i + 1 });
    else if ((m = /^test\s+"([^"]*)"/.exec(l))) items.push({ kind: 'test', name: m[1], line: i + 1 });
    else if ((m = /^(?:from\s+([\w.]+)\s+import\b|import\s+([\w.]+))/.exec(l))) items.push({ kind: 'use', name: m[1] || m[2], line: i + 1 });
    else if ((m = /^([A-Z][A-Z0-9_]+)\s*=/.exec(l))) items.push({ kind: 'const', name: m[1], line: i + 1 });
  });
  return items;
}

function renderFile(ref, p) {
  const src = git('show', `${ref}:${p}`);
  const lang = LANG(p);
  const lines = src.replace(/\n$/, '').split('\n');
  let cuts = 0;
  const rows = lines
    .map((l, i) => {
      const { text, cut } = clip(l);
      if (cut) cuts++;
      const code = highlight(text, lang);
      const more = cut ? `<span class="cut">… ${cut.toLocaleString()} more characters</span>` : '';
      return `<tr id="L${i + 1}"><td class="ln"><a href="#L${i + 1}">${i + 1}</a></td><td class="src">${code}${more}</td></tr>`;
    })
    .join('');
  const out = outline(src);
  const defs = out.filter((o) => o.kind === 'def').length;
  const tests = out.filter((o) => o.kind === 'test').length;
  return { rows, out, lines: lines.length, defs, tests, cuts, bytes: src.length };
}

function renderDiff(ref, p) {
  const base = tryGit('merge-base', 'main', ref).trim();
  const d = tryGit('diff', '-U4', base, ref, '--', p);
  if (!d) return '';
  const lang = LANG(p);
  const rows = [];
  for (const raw of d.split('\n').slice(4)) {
    if (raw.startsWith('@@')) {
      rows.push(`<tr class="hunk"><td colspan="2">${esc(raw)}</td></tr>`);
      continue;
    }
    const sign = raw[0] || ' ';
    const { text, cut } = clip(raw.slice(1));
    const cls = sign === '+' ? 'add' : sign === '-' ? 'del' : '';
    const more = cut ? `<span class="cut">… ${cut.toLocaleString()} more characters</span>` : '';
    rows.push(`<tr class="${cls}"><td class="sg">${esc(sign)}</td><td class="src">${highlight(text, lang)}${more}</td></tr>`);
  }
  return `<table class="code diff">${rows.join('')}</table>`;
}

function page({ ref, file, all, view, lens, split, panes }) {
  const R = refs();
  if (!R.some((r) => r.name === ref)) ref = 'main';
  const ch = changed(ref);
  const chSet = new Map(ch.map((c) => [c.path, c]));
  const files = all || ref === 'main' ? tree(ref) : ch.map((c) => c.path).filter((p) => SOURCE.test(p));
  if (file && !tree(ref).includes(file)) file = '';
  // Open on the code that matters: the Hartree-Fock library first, then any
  // library, then any Viv program at all.
  if (!file) file = files.find((f) => f.startsWith('lib/hf/')) || files.find((f) => f.startsWith('lib/') && f.endsWith('.viv')) || files.find((f) => f.endsWith('.viv')) || files[0] || '';
  const q = (o) => '/?' + new URLSearchParams({ ref, ...(all ? { all: '1' } : {}), ...o }).toString();

  // group files by directory for the list
  const groups = {};
  for (const f of files) {
    const dir = f.includes('/') ? f.slice(0, f.lastIndexOf('/')) : '.';
    (groups[dir] ||= []).push(f);
  }
  const list = Object.keys(groups)
    .sort()
    .map((dir) => {
      const items = groups[dir]
        .map((f) => {
          const c = chSet.get(f);
          const stat = c ? `<span class="stat"><span class="a">+${c.add}</span>${c.del ? ` <span class="d">−${c.del}</span>` : ''}</span>` : '';
          const cur = f === file ? ' class="cur"' : '';
          return `<li${cur}><a href="${q({ file: f })}">${esc(f.slice(dir.length + 1) || f)}</a>${stat}</li>`;
        })
        .join('');
      return `<div class="dir">${esc(dir)}/</div><ul>${items}</ul>`;
    })
    .join('');

  const refOpts = R.map((r) => `<option value="${esc(r.name)}"${r.name === ref ? ' selected' : ''}>${esc(r.name)}${r.ahead ? ` (+${r.ahead})` : ''} — ${esc(r.when)}</option>`).join('');
  const log =
    ref === 'main'
      ? ''
      : tryGit('log', '--format=%h\t%cr\t%s', `main..${ref}`)
          .trim()
          .split('\n')
          .filter(Boolean)
          .map((l) => {
            const [h, when, s] = l.split('\t');
            return `<li><code>${esc(h)}</code> <span class="when">${esc(when)}</span> ${esc(s)}</li>`;
          })
          .join('');

  let body = '<p class="empty">No files.</p>';
  let side = '';
  if (file) {
    const isChanged = chSet.has(file);
    const mode = view === 'diff' && isChanged ? 'diff' : 'file';
    const f = renderFile(ref, file);
    side = f.out.length
      ? `<nav class="outline"><div class="h">Outline</div><ol>${f.out
          .map((o) => `<li class="k-${o.kind}"><a href="${mode === 'diff' ? q({ file }) : ''}#L${o.line}"><span class="kind">${o.kind}</span> ${esc(o.name)}${o.kind === 'def' ? `<span class="args">(${esc(o.args)})</span>` : ''}</a></li>`)
          .join('')}</ol></nav>`
      : '';
    // A .viv file also gets the four lenses; every other file keeps its tabs.
    const isViv = file.endsWith('.viv');
    const lensMode = isViv && LENSES.includes(lens) && mode === 'file' && Array.isArray(panes);
    const tabs = isViv
      ? lensTabs({ q, file, isChanged, mode, lens: lensMode ? lens : '', split: lensMode ? split : '' })
      : isChanged
      ? `<div class="tabs"><a class="${mode === 'file' ? 'on' : ''}" href="${q({ file })}">File</a><a class="${mode === 'diff' ? 'on' : ''}" href="${q({ file, view: 'diff' })}">Diff against main</a></div>`
      : '';
    const facts = `${f.lines.toLocaleString()} lines · ${f.defs} defs · ${f.tests} tests${f.cuts ? ` · ${f.cuts} long lines cut` : ''}`;
    body = `<header class="fh"><h1>${esc(file)}</h1><div class="facts">${facts}</div>${tabs}</header>${
      mode === 'diff'
        ? renderDiff(ref, file)
        : lensMode
        ? `<div class="lenses${panes.length > 1 ? ' two' : ''}">${panes.map((h, i) => `<section class="lpane${i === 0 ? ' act' : ''}">${h}</section>`).join('')}</div>`
        : `<table class="code">${f.rows}</table>`
    }${isViv && mode === 'file' ? lensScript({ ref, file, all, lens: lensMode ? lens : '', split: lensMode ? split : '', anchors: panes && panes.anchors }) : ''}`;
  }

  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Viv source</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">
<style>${CSS}
${STYLE}
${MAPSTYLE}${file && file.endsWith('.viv') ? LENSSTYLE : ''}</style></head><body>
${bar('browse')}
  <form method="get" action="/"><select name="ref" onchange="this.form.submit()">${refOpts}</select>${all ? '<input type="hidden" name="all" value="1">' : ''}</form>
  ${ref !== 'main' ? `<a class="toggle" href="/?${new URLSearchParams({ ref, ...(all ? {} : { all: '1' }) })}">${all ? 'Show changed files only' : 'Show all files'}</a>` : ''}
  <span class="note">Read live from git; reload for the latest commits.</span>
</div>
<div class="wrap">
  <aside class="files">${log ? `<div class="h">Commits on ${esc(ref)}</div><ol class="log">${log}</ol>` : ''}<div class="h">${ref === 'main' || all ? 'Files' : `Changed against main (${ch.length})`}</div>${list}</aside>
  <main>${body}</main>
  ${side ? `<aside class="side">${side}</aside>` : ''}
</div>
</body></html>`;
}

// ---------------------------------------------------------------------------
// Lenses: code, intent, debug and story
// renderings of one .viv file, each element tagged with the anchor it belongs
// to (`def` or `def/step`), so that switching lenses, or scrolling one of two
// lenses side by side, can keep the same def and step in view.

const LENSES = ['code', 'intent', 'debug', 'story'];
// `viv view` answers in well under a second even on lib/hf; `viv story` runs
// the program, which for a lib/hf module takes about 25 s.
const VIEW_MS = 60e3;
const STORY_MS = 300e3;

function worktrees() {
  const list = [];
  let cur = null;
  for (const l of tryGit('worktree', 'list', '--porcelain').split('\n')) {
    if (l.startsWith('worktree ')) list.push((cur = { path: l.slice(9), head: '', branch: '' }));
    else if (cur && l.startsWith('HEAD ')) cur.head = l.slice(5);
    else if (cur && l.startsWith('branch ')) cur.branch = l.slice(7);
  }
  return list;
}

const real = (p) => {
  try {
    return realpathSync(p);
  } catch {
    return p;
  }
};

// Where the lenses of `ref` come from: the worktree that has the branch
// checked out, else one sitting on the same commit (this viewer's own first),
// and the viv binary built there.
function lensHome(ref) {
  const sha = tryGit('rev-parse', '--verify', '-q', `${ref}^{commit}`).trim();
  const wts = worktrees();
  const root = real(ROOT);
  const w =
    wts.find((w) => w.branch === `refs/heads/${ref}`) ||
    wts.find((w) => w.head === sha && real(w.path) === root) ||
    wts.find((w) => w.head === sha);
  if (!w) {
    return { err: `No worktree has ${ref} checked out, so there are no files on disk for viv to read. Check the branch out in a worktree and build viv there to see this lens.` };
  }
  const bin = process.env.VIV || path.join(w.path, 'target/release/viv');
  if (!existsSync(bin)) {
    return { dir: w.path, err: `No viv binary at ${bin}. Build it in that worktree (cargo build --release -p viv) to see this lens.` };
  }
  return { dir: w.path, bin };
}

// One run of viv at a time per command line: a split view and a rotation that
// both ask for the story share the run instead of starting two.
const inflight = new Map();
function runViv(home, args, ms) {
  const key = JSON.stringify([home.bin, home.dir, args]);
  if (inflight.has(key)) return inflight.get(key);
  const p = new Promise((resolve) => {
    execFile(home.bin, args, { cwd: home.dir, timeout: ms, maxBuffer: 256 << 20, encoding: 'utf8' }, (err, stdout, stderr) => {
      resolve({ ok: !err, killed: !!(err && err.killed), stdout: stdout || '', stderr: stderr || (err && !err.code ? String(err.message) : '') });
    });
  }).finally(() => inflight.delete(key));
  inflight.set(key, p);
  return p;
}

// The JSON a viv subcommand prints, or why there is none.
async function vivJson(home, args, ms) {
  const r = await runViv(home, args, ms);
  const text = r.stdout.trim();
  // `viv story` exits nonzero when a hold failed and still prints the story
  if (text.startsWith('{')) {
    try {
      const j = JSON.parse(text);
      // viv's --json envelope carries the object as `result`
      return { json: j.viv === 1 && 'result' in j && 'cmd' in j ? j.result : j };
    } catch {}
  }
  if (r.killed) return { err: `viv ${args[0]} did not finish in ${ms / 1000} s.` };
  const e = r.stderr.trim();
  if (e.startsWith('usage:') && !e.includes(`viv ${args[0]}`)) {
    return { err: `This viv build (${home.bin}) has no \`${args[0]}\` command: it predates the lenses. Rebuild it from a branch that has them.` };
  }
  return { err: `viv ${args.join(' ')} failed:`, detail: (e || text).split('\n').slice(0, 30).join('\n') };
}

// The preorder list of anchors of a file (every def, then its steps): the
// order in which a lens that lacks an anchor looks for the nearest one it has.
// Ranges map a line of the file to the innermost def or step holding it.
function anchorsOf(v) {
  const order = [];
  const ranges = [];
  for (const d of (v && v.defs) || []) {
    if (d.name === '<main>' || !d.line) continue;
    order.push(d.name);
    ranges.push({ a: d.name, line: d.line, end: d.end });
    const walk = (items) => {
      for (const o of items || []) {
        if (o.kind === 'step') {
          order.push(`${d.name}/${o.id}`);
          ranges.push({ a: `${d.name}/${o.id}`, line: o.line, end: o.end });
        }
        walk(o.children);
      }
    };
    walk(d.outline);
  }
  return { order, ranges };
}

const note = (msg, detail) => `<div class="lnote">${esc(msg)}${detail ? `<pre>${esc(detail)}</pre>` : ''}</div>`;

// A number as the story prints one: ten significant digits, no trailing zeros.
function num(x) {
  if (typeof x !== 'number') return esc(String(x));
  if (Number.isInteger(x) && Math.abs(x) < 1e15) return String(x);
  return String(Number(x.toPrecision(10)));
}

function valueHtml(v) {
  if (v === null || typeof v !== 'object') return `<span class="lv">${num(v)}</span>`;
  const shape = (v.shape || []).join('×');
  if (Array.isArray(v.values)) {
    const xs = v.values;
    const [r, c] = v.shape || [];
    if (v.shape && v.shape.length === 2 && r <= 8 && c <= 8) {
      let t = '';
      for (let i = 0; i < r; i++) t += `<tr>${xs.slice(i * c, i * c + c).map((x) => `<td>${num(x)}</td>`).join('')}</tr>`;
      return `<span class="lv">${shape}</span><table class="lmat">${t}</table>`;
    }
    if (xs.length <= 12) return `<span class="lv">[${xs.map(num).join(', ')}]</span>`;
    const norm = Math.sqrt(xs.reduce((s, x) => s + x * x, 0));
    return `<span class="lv">${shape} |${num(norm)}|</span>`;
  }
  const parts = ['min', 'max', 'norm', 'trace'].filter((k) => k in v).map((k) => `${k} ${num(v[k])}`);
  return `<span class="lv">${shape} ${parts.join(' ')}</span>`;
}

function watchHtml(w) {
  const ks = Object.keys(w || {});
  if (!ks.length) return '';
  return `<span class="lw">${ks.map((k) => `<span class="lwk">${esc(k)}</span> ${valueHtml(w[k])}`).join('<span class="lsep"></span>')}</span>`;
}

function holdsHtml(hs) {
  if (!hs || !hs.length) return '';
  return `<ul class="lholds">${hs
    .map((h) => `<li class="${h.ok ? 'ok' : 'bad'}"><span class="lhk">${h.ok ? 'holds' : 'FAILED'}</span> ${esc(h.sentence)}${h.worst != null ? ` <span class="lfaint">worst ${num(h.worst)}</span>` : ''}</li>`)
    .join('')}</ul>`;
}

// code: the file with every intent comment (#| #|| #||| #: #@) hidden, since
// those belong to the intent lens; each row carries its line and its anchor.
function codeLens(src, lang, v) {
  const lines = src.replace(/\n$/, '').split('\n');
  const at = [];
  for (const r of anchorsOf(v).ranges) for (let i = r.line; i <= r.end; i++) at[i] = r.a; // steps come after their def: innermost wins
  const rows = lines
    .map((l, i) => {
      const n = i + 1;
      const { text, cut } = clip(l);
      const more = cut ? `<span class="cut">… ${cut.toLocaleString()} more characters</span>` : '';
      const ic = /^\s*#(\||:|@)/.test(l) ? ' class="ic"' : '';
      return `<tr${ic} data-l="${n}"${at[n] ? ` data-a="${esc(at[n])}"` : ''}><td class="ln">${n}</td><td class="src">${highlight(text, lang)}${more}</td></tr>`;
    })
    .join('');
  return `<table class="code lcode">${rows}</table>`;
}

function fieldHtml(k, f) {
  if (!f || !f.text) return '';
  return `<div class="lf"><span class="lfk">${k}</span><span class="lft">${esc(f.text)}</span></div>`;
}

function outlineHtml(def, items) {
  return (items || [])
    .map((o) =>
      o.kind === 'step'
        ? `<div class="lstep" data-a="${esc(`${def}/${o.id}`)}"><span class="lid">${esc(o.id)}</span><span class="lt">${o.text ? esc(o.text) : '<span class="lfaint">(no text)</span>'}</span></div>${
            o.children && o.children.length ? `<div class="lkids">${outlineHtml(def, o.children)}</div>` : ''
          }`
        : `<div class="lctl">${esc(o.text || o.control || '')}</div><div class="lkids">${outlineHtml(def, o.children)}</div>`
    )
    .join('');
}

// intent: what, how and why of each def, its glossary and its steps, in file
// order -- the pseudocode the .intent file and the #| comments hold together.
function intentLens(v) {
  const out = [];
  if (v.entry && v.entry.text) out.push(`<div class="lmod"><span class="lk">module</span> ${esc(v.entry.name)}<p>${esc(v.entry.text)}</p></div>`);
  for (const d of v.defs) {
    if (d.name === '<main>') continue;
    const chips = [];
    if (/^placeholder\b/.test(d.sig)) chips.push('<span class="chip planned">placeholder: no body yet</span>');
    for (const [where, h] of Object.entries(d.stamp || {})) {
      if (h && h !== d.hash) chips.push(`<span class="chip branch">stale: ${where === 'file' ? '.intent' : '#@'} written for @${esc(h)}, code is @${esc(d.hash)}</span>`);
    }
    const i = d.intent || {};
    const gl = (d.glossary || []).map((g) => `<div class="lf"><span class="lfk lg">${esc(g.name)}</span><span class="lft">${esc(g.text)}</span></div>`).join('');
    const fields = fieldHtml('what', i.what) + fieldHtml('how', i.how) + fieldHtml('why', i.why);
    out.push(
      `<section class="ldef" data-a="${esc(d.name)}"><h3>${esc(d.name)} <span class="lfaint lsig">${esc(d.sig)}</span> ${chips.join(' ')}</h3>${
        fields || '<div class="lfaint">(no intent written)</div>'
      }${gl}${d.outline && d.outline.length ? `<div class="lsteps">${outlineHtml(d.name, d.outline)}</div>` : ''}</section>`
    );
  }
  return out.join('') || note('This file has no defs.');
}

// intent in run order: `viv explain`, the intent walked along the static call
// graph from the program's entry, calls expanded in place.
function explainLens(root) {
  const go = (n, def) => {
    let head = '';
    let a = '';
    let inner = def;
    if (n.kind === 'call' || n.kind === 'def') {
      a = n.name;
      inner = n.name;
      head = `<span class="lk">${n.kind === 'call' ? '→' : ''}</span> <span class="lid">${esc(n.name)}</span> <span class="lt">${esc(n.text)}</span>${n.module ? ` <span class="lfaint">${esc(n.module)}</span>` : ''}`;
    } else if (n.kind === 'step') {
      a = `${def}/${n.name}`;
      head = `<span class="lid">${esc(n.name)}</span> <span class="lt">${n.text ? esc(n.text) : '<span class="lfaint">(no text)</span>'}</span>`;
    } else {
      head = `<span class="lctl">${esc(n.text || n.name)}</span>`;
    }
    if (n.note) head += ` <span class="lfaint">${esc(n.note)}</span>`;
    const kids = (n.children || []).map((c) => go(c, inner)).join('');
    return `<div class="lx lx-${esc(n.kind)}"${a ? ` data-a="${esc(a)}"` : ''}>${head}</div>${kids ? `<div class="lkids">${kids}</div>` : ''}`;
  };
  return `<div class="lexplain">${go(root, '')}</div>`;
}

// debug: the .debug layout `viv view --lens debug` prints; a `watch at` or
// `hold at` line belongs to its step, the rest to its def.
function debugLens(text) {
  if (!text.trim()) return note('No debug lens: this module has no .debug file and no inline debug blocks.');
  let def = '';
  return `<div class="ldebug">${text
    .replace(/\n$/, '')
    .split('\n')
    .map((l) => {
      let a = def;
      let cls = 'dl';
      const h = /^([A-Za-z_*][\w]*)(\s+@[0-9a-f]+)?\s*$/.exec(l);
      if (h && !/^\s/.test(l)) {
        def = a = h[1];
        cls += ' dh';
      } else {
        const s = /^\s+(?:watch|hold)\s+at\s+([A-Za-z_]\w*)/.exec(l);
        if (s && def) a = `${def}/${s[1]}`;
      }
      const body = esc(l)
        .replace(/^(\s+)(watch|hold|trace)\b/, '$1<span class="dk">$2</span>')
        .replace(/(&quot;[^&]*?&quot;)\s*$/, '<span class="ds">$1</span>');
      return `<div class="${cls}"${a ? ` data-a="${esc(a)}"` : ''}>${body || ' '}</div>`;
    })
    .join('')}</div>`;
}

// story: one run in debug mode. Each loop iteration is one line with its
// watched values; the first iteration of each loop is open, so every step has
// a visible place to be scrolled to, and the rest open on a click.
function storyLens(s) {
  const firstClause = (w) => (w || '').split(/[.;]\s/)[0];
  const go = (n) => {
    const kids = [];
    const opened = new Set();
    for (const c of n.children || []) {
      if (c.kind === 'iteration') {
        const open = !opened.has(c.def);
        opened.add(c.def);
        kids.push(go1(c, open));
      } else kids.push(go(c));
    }
    const k = kids.join('');
    const tail = holdsHtml(n.holds) + (k ? `<div class="lkids">${k}</div>` : '');
    if (n.kind === 'main') return `<div class="sn" data-a="${esc(n.def)}"><span class="lid">${esc(n.def)}</span> ${watchHtml(n.watch)}</div>${tail}`;
    if (n.kind === 'call') return `<div class="sn" data-a="${esc(n.def)}"><span class="lk">→</span> <span class="lid">${esc(n.def)}</span> <span class="lt">${esc(firstClause(n.what))}</span> ${watchHtml(n.watch)}</div>${tail}`;
    if (n.kind === 'step')
      return `<div class="sn" data-a="${esc(`${n.def}/${n.id}`)}"><span class="lt">${esc(n.text || n.id)}</span> <span class="lfaint">${esc(n.id)}</span> ${watchHtml(n.watch)}</div>${tail}`;
    return tail;
  };
  // an iteration's anchor is its first step, so scrolling past iteration lines
  // keeps the other lens on the loop, not on the top of the def
  const go1 = (n, open) => {
    const first = (n.children || []).find((c) => c.kind === 'step');
    const a = first ? `${n.def}/${first.id}` : n.def;
    const bad = JSON.stringify(n).includes('"ok":false');
    return `<details class="sit"${open ? ' open' : ''}><summary data-a="${esc(a)}"><span class="lid">iteration ${n.n}</span> ${watchHtml(n.watch)}${bad ? ' <span class="lhk bad">a hold FAILED</span>' : ''}</summary>${go({ ...n, kind: '' })}</details>`;
  };
  const h = s.holds || { checked: 0, failed: 0 };
  const head = `<div class="lsum">${s.total ? `${s.total.toLocaleString()} events recorded` : 'The run recorded no events: no def that ran has steps, watches, holds or traces.'}${
    h.checked ? ` · holds: ${h.checked} checked, <span class="${h.failed ? 'bad' : 'ok'}">${h.failed} failed</span>` : ''
  }${s.kept < s.total ? ` · the story keeps the first ${s.kept.toLocaleString()}` : ''}</div>`;
  return `${head}<div class="lstory">${go(s.root)}</div>`;
}

// One lens pane: a header naming the lens and where it was read from, and the
// lens itself, scrollable.
async function lensPane(ref, file, lens, order) {
  const home = lensHome(ref);
  const lang = LANG(file);
  const label = { code: 'Code', intent: 'Intent', debug: 'Debug', story: 'Story' }[lens];
  const ctl =
    lens === 'intent'
      ? `<span class="lord"><button data-order="" class="${order === 'run' ? '' : 'on'}">file order</button><button data-order="run" class="${order === 'run' ? 'on' : ''}">run order</button></span>`
      : lens === 'code'
      ? '<label class="lic"><input type="checkbox" class="lic-toggle"> intent comments</label>'
      : '';
  const head = (src) => `<div class="lhead"><span class="lname">${label}</span>${ctl}<span class="lsrc">${src}</span></div>`;
  let body;
  let src = '';
  if (home.err) {
    body = lens === 'code' ? codeLens(git('show', `${ref}:${file}`), lang, null) : '';
    body = note(home.err) + body;
    src = `from git (${esc(ref)})`;
  } else {
    // the lenses read the working tree: say so when it is not the commit
    const dirty = tryGitIn(home.dir, 'status', '--porcelain', '--', file, file.replace(/\.viv$/, '.intent'), file.replace(/\.viv$/, '.debug')).trim();
    const where = real(home.dir) === real(ROOT) ? 'this checkout' : home.dir;
    const bin = path.relative(home.dir, home.bin);
    src = `${esc(where)} · ${esc(bin.startsWith('..') ? home.bin : bin)}${
      dirty ? ` · <span class="bad" title="${esc(dirty)}">uncommitted changes on disk</span>` : ''
    }`;
    if (lens === 'code' || lens === 'intent') {
      const r = lens === 'intent' && order === 'run' ? await vivJson(home, ['explain', file, '--json'], VIEW_MS) : await vivJson(home, ['view', file, '--lens', 'code', '--json'], VIEW_MS);
      if (lens === 'code') {
        let text;
        try {
          text = readFileSync(path.join(home.dir, file), 'utf8');
        } catch {
          text = git('show', `${ref}:${file}`);
        }
        body = (r.err ? note(`${r.err} The code is shown without its anchors.`, r.detail) : '') + codeLens(text, lang, r.json);
      } else body = r.err ? note(r.err, r.detail) : order === 'run' ? explainLens(r.json) : intentLens(r.json);
    } else if (lens === 'debug') {
      const r = await vivJson(home, ['view', file, '--lens', 'debug', '--json'], VIEW_MS);
      body = r.err ? note(r.err, r.detail) : debugLens(r.json.text || '');
    } else {
      const t0 = Date.now();
      const r = await vivJson(home, ['story', file, '--json'], STORY_MS);
      body = r.err ? note(r.err, r.detail) : storyLens(r.json);
      src += ` · ran in ${((Date.now() - t0) / 1000).toFixed(1)} s`;
    }
  }
  return `${head(src)}<div class="lbody" data-lens="${lens}" data-order="${esc(order || '')}">${body}</div>`;
}

function tryGitIn(dir, ...args) {
  try {
    return execFileSync('git', ['-C', dir, ...args], { encoding: 'utf8', maxBuffer: 16 << 20 });
  } catch {
    return '';
  }
}

// The anchors of a file for the page script, from the same `viv view` the
// code lens reads; empty when there is no binary.
async function fileAnchors(ref, file) {
  const home = lensHome(ref);
  if (home.err) return { order: [], ranges: [] };
  const r = await vivJson(home, ['view', file, '--lens', 'code', '--json'], VIEW_MS);
  return anchorsOf(r.json);
}

function lensTabs({ q, file, isChanged, mode, lens, split }) {
  const t = (on, href, text, extra = '') => `<a class="${on ? 'on' : ''}" href="${href}"${extra}>${text}</a>`;
  const opts = ['', ...LENSES].map((l) => `<option value="${l}"${l === split ? ' selected' : ''}>${l ? `beside ${l}` : 'one lens'}</option>`).join('');
  return `<div class="tabs">${t(mode === 'file' && !lens, q({ file }), 'File')}${LENSES.map((l) =>
    t(l === lens, q({ file, lens: l, ...(split ? { split } : {}) }), l, ` data-lens="${l}"`)
  ).join('')}${isChanged ? t(mode === 'diff', q({ file, view: 'diff' }), 'Diff against main') : ''}<span class="lsplit"><select class="lsplit-sel" title="a second lens beside the first, scroll-locked by anchor (key s)">${opts}</select><span class="lkeys">l rotate · s split</span></span></div>`;
}

function lensScript({ ref, file, all, lens, split, anchors }) {
  const data = { ref, file, all: !!all, lens, split, lenses: LENSES, order: (anchors && anchors.order) || [], ranges: (anchors && anchors.ranges) || [] };
  return `<script type="application/json" id="lens-data">${JSON.stringify(data).replace(/</g, '\\u003c')}</script><script>${LENSJS}</script>`;
}

// The page script: rotation, split and the scroll lock. An anchor is found in
// a lens by exact match, then by its def, then by the nearest anchor before it
// (or after it) in file order, so a lens that has no entry for a step still
// shows the part of the program around it.
const LENSJS = String.raw`(() => {
  const D = JSON.parse(document.getElementById('lens-data').textContent);
  const L = D.lenses;
  const wrap = document.querySelector('.lenses');
  const panes = () => [...document.querySelectorAll('.lpane')];
  const bodyOf = (p) => p.querySelector('.lbody');
  // a closed <details> keeps layout boxes for its content, so getClientRects
  // alone would call a step inside a collapsed iteration visible
  const vis = (el) => (el.checkVisibility ? el.checkVisibility() : el.getClientRects().length > 0);
  const lensOf = (p) => (bodyOf(p) && bodyOf(p).dataset.lens) || '';
  const orderOf = (p) => (bodyOf(p) && bodyOf(p).dataset.order) || '';
  const url = (lens, split) => '/?' + new URLSearchParams({ ref: D.ref, ...(D.all ? { all: '1' } : {}), file: D.file, ...(lens ? { lens } : {}), ...(split ? { split } : {}) });
  let active = 0;
  let showIC = false;
  let tok = 0;
  const cache = new Map();
  const ck = (lens, order) => lens + '|' + (order || '');

  // the visible anchored elements of a pane, in document order
  const els = (p) => p._els || (p._els = bodyOf(p) ? [...bodyOf(p).querySelectorAll('[data-a]')].filter(vis) : []);
  const leaf = (el) => !el.querySelector('[data-a]');
  // the height of el, or of the run of code rows that share its anchor
  function span(p, el) {
    const list = els(p);
    let j = list.indexOf(el);
    if (j < 0) return el.getBoundingClientRect().height;
    while (el.tagName === 'TR' && j + 1 < list.length && list[j + 1].dataset.a === el.dataset.a) j++;
    return list[j].getBoundingClientRect().bottom - el.getBoundingClientRect().top;
  }
  // the anchor at the top of a pane, and how far into it the pane has scrolled
  function here(p) {
    const b = bodyOf(p);
    const list = els(p);
    if (!b || !list.length) return null;
    const top = b.getBoundingClientRect().top;
    let lo = 0, hi = list.length - 1, k = 0;
    while (lo <= hi) {
      const m = (lo + hi) >> 1;
      if (list[m].getBoundingClientRect().top <= top + 1) { k = m; lo = m + 1; } else hi = m - 1;
    }
    // the rows of one step in the code lens count as one element; elsewhere
    // two neighbors with one anchor are two places (two iterations of a loop)
    while (list[k].tagName === 'TR' && k > 0 && list[k - 1].dataset.a === list[k].dataset.a) k--;
    const el = list[k];
    const s = span(p, el);
    const frac = leaf(el) && s > 0 ? Math.min(1, Math.max(0, (top - el.getBoundingClientRect().top) / s)) : 0;
    return { a: el.dataset.a, frac };
  }
  function find(p, a) {
    const b = bodyOf(p);
    if (!b) return null;
    const q = (x) => { const all = [...b.querySelectorAll('[data-a="' + CSS.escape(x) + '"]')]; return all.find(vis) || all[0] || null; };
    let el = q(a);
    if (!el && a.includes('/')) el = q(a.split('/')[0]);
    if (!el) {
      let i = D.order.indexOf(a);
      if (i < 0) i = D.order.indexOf(a.split('/')[0]);
      for (let k = i - 1; i >= 0 && k >= 0 && !el; k--) el = q(D.order[k]);
      for (let k = i + 1; i >= 0 && k < D.order.length && !el; k++) el = q(D.order[k]);
    }
    if (el && !vis(el)) {
      for (let d = el.closest('details'); d; d = d.parentElement && d.parentElement.closest('details')) d.open = true;
      p._els = null;
    }
    return el;
  }
  const mute = (p) => { p._mute = performance.now(); };
  function go(p, a, frac) {
    const el = find(p, a);
    if (!el) return false;
    const b = bodyOf(p);
    const off = el.getBoundingClientRect().top - b.getBoundingClientRect().top + (leaf(el) ? (frac || 0) * span(p, el) : 0);
    mute(p);
    b.scrollTop += off;
    return true;
  }
  function follow(p) {
    const h = here(p);
    if (h) for (const o of panes()) if (o !== p) go(o, h.a, h.frac);
  }

  // one line of the file, in whichever lenses are showing
  function goLine(n) {
    const ps = panes();
    const code = ps.find((p) => lensOf(p) === 'code');
    if (code) {
      let el = bodyOf(code).querySelector('[data-l="' + n + '"]');
      while (el && !vis(el)) el = el.nextElementSibling;
      if (el) {
        mute(code);
        bodyOf(code).scrollTop += el.getBoundingClientRect().top - bodyOf(code).getBoundingClientRect().top;
        follow(code);
        return;
      }
    }
    let r = null;
    for (const x of D.ranges) if (x.line <= n && n <= x.end) r = x;
    if (r) for (const p of ps) go(p, r.a, (n - r.line) / (r.end - r.line + 1));
  }

  function load(lens, order) {
    const k = ck(lens, order);
    if (!cache.has(k)) {
      const u = new URLSearchParams({ ref: D.ref, file: D.file, lens, ...(order ? { order } : {}) });
      cache.set(k, fetch('/lens?' + u).then((r) => r.text()).catch((e) => '<div class="lbody"><div class="lnote">' + String(e) + '</div></div>'));
    }
    return cache.get(k);
  }
  function setup(p) {
    p._els = null;
    p.classList.toggle('showic', showIC);
    const t = p.querySelector('.lic-toggle');
    if (t) { t.checked = showIC; t.onchange = () => { const h = here(p); showIC = t.checked; panes().forEach((x) => { x.classList.toggle('showic', showIC); x._els = null; }); if (h) go(p, h.a, h.frac); }; }
    p.querySelectorAll('[data-order]').forEach((btn) => { if (btn.tagName === 'BUTTON') btn.onclick = () => show(p, 'intent', btn.dataset.order); });
  }
  async function show(p, lens, order) {
    // a lens without the anchor shows its nearest neighbor; the pane still
    // wants the original, so rotating on reaches it again unless the reader
    // has scrolled since
    const h = p._want || here(p) || p._last || null;
    const mine = ++tok;
    p._tok = mine;
    if (!cache.has(ck(lens, order))) {
      p.classList.add('loading');
      const n = p.querySelector('.lname');
      if (n) n.textContent = lens === 'story' ? 'Story: running the program in debug mode…' : lens + '…';
    }
    const html = await load(lens, order);
    if (p._tok !== mine) return;
    p.classList.remove('loading');
    p.innerHTML = html;
    setup(p);
    if (h) go(p, h.a, h.frac);
    p._want = h && !bodyOf(p).querySelector('[data-a="' + CSS.escape(h.a) + '"]') ? h : null;
    p._last = h;
    marks();
    save();
  }
  function setSplit(lens) {
    const ps = panes();
    if (!lens) {
      ps.slice(1).forEach((x) => x.remove());
      active = 0;
      wrap.classList.remove('two');
    } else if (ps.length === 1) {
      const n = document.createElement('section');
      n.className = 'lpane';
      n.innerHTML = '<div class="lhead"><span class="lname"></span></div><div class="lbody"></div>';
      wrap.appendChild(n);
      wrap.classList.add('two');
      n._last = here(ps[0]);
      show(n, lens, '');
    } else show(ps[1], lens, '');
    marks();
    save();
  }
  function marks() {
    const ps = panes();
    if (active >= ps.length) active = 0;
    ps.forEach((p, i) => p.classList.toggle('act', i === active && ps.length > 1));
    const on = lensOf(ps[active]);
    document.querySelectorAll('.tabs a').forEach((a) => a.classList.toggle('on', a.dataset.lens === on));
    const sel = document.querySelector('.lsplit-sel');
    if (sel) sel.value = ps.length > 1 ? lensOf(ps[1]) : '';
  }
  let saveT = 0;
  function save() {
    clearTimeout(saveT);
    saveT = setTimeout(() => {
      const ps = panes();
      const h = here(ps[active]);
      history.replaceState(null, '', url(lensOf(ps[0]), ps[1] ? lensOf(ps[1]) : '') + (h ? '#a=' + encodeURIComponent(h.a) : ''));
    }, 250);
  }
  function size() {
    if (!wrap) return;
    const top = wrap.getBoundingClientRect().top + window.scrollY;
    wrap.style.height = Math.max(320, window.innerHeight - top - 12) + 'px';
  }
  // the first File row below the top bar, for leaving the File view on the same line
  function topLine() {
    const rows = [...document.querySelectorAll('table.code tr[id^="L"]')];
    const bar = document.querySelector('.bar');
    const y = bar ? bar.getBoundingClientRect().bottom : 0;
    const r = rows.find((x) => x.getBoundingClientRect().bottom > y);
    return r ? r.id.slice(1) : '1';
  }

  document.addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey || /^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(e.target.tagName)) return;
    if (e.key === 'l' || e.key === 'L') {
      e.preventDefault();
      if (!wrap) { location.href = url('code') + '#L' + topLine(); return; }
      const p = panes()[active];
      const k = L.indexOf(lensOf(p));
      show(p, L[(k + (e.key === 'l' ? 1 : L.length - 1) + L.length) % L.length], '');
    } else if (e.key === 's') {
      e.preventDefault();
      if (!wrap) { location.href = url('code', 'intent') + '#L' + topLine(); return; }
      if (panes().length > 1) setSplit('');
      else setSplit(L[(L.indexOf(lensOf(panes()[0])) + 1) % L.length]);
    }
  });
  document.querySelectorAll('.tabs a[data-lens]').forEach((a) =>
    a.addEventListener('click', (e) => {
      if (!wrap) { a.href = url(a.dataset.lens) + '#L' + topLine(); return; }
      e.preventDefault();
      show(panes()[active], a.dataset.lens, '');
    })
  );
  const sel = document.querySelector('.lsplit-sel');
  if (sel) sel.addEventListener('change', () => {
    if (!wrap) { location.href = url('code', sel.value) + '#L' + topLine(); return; }
    setSplit(sel.value);
  });
  if (!wrap) return;

  document.addEventListener('scroll', (e) => {
    const p = e.target && e.target.closest ? e.target.closest('.lpane') : null;
    if (!p || performance.now() - (p._mute || 0) < 120) return;
    p._want = null;
    if (p._raf) return;
    p._raf = requestAnimationFrame(() => { p._raf = 0; if (panes().length > 1) follow(p); save(); });
  }, true);
  document.addEventListener('toggle', (e) => { const p = e.target.closest && e.target.closest('.lpane'); if (p) p._els = null; }, true);
  document.addEventListener('pointerdown', (e) => {
    const p = e.target.closest && e.target.closest('.lpane');
    if (p) { active = panes().indexOf(p); marks(); }
  });
  document.querySelectorAll('.outline a').forEach((a) =>
    a.addEventListener('click', (e) => {
      const m = /#L(\d+)$/.exec(a.getAttribute('href') || '');
      if (!m) return;
      e.preventDefault();
      goLine(Number(m[1]));
      save();
    })
  );
  window.addEventListener('resize', size);

  panes().forEach((p) => { cache.set(ck(lensOf(p), orderOf(p)), Promise.resolve(p.innerHTML)); setup(p); });
  size();
  marks();
  const ma = /^#a=(.+)$/.exec(location.hash);
  const ml = /^#L(\d+)$/.exec(location.hash);
  if (ma) { const a = decodeURIComponent(ma[1]); panes().forEach((p) => go(p, a, 0)); }
  else if (ml) goLine(Number(ml[1]));
})();`;

const LENSSTYLE = `
.lsplit { margin-left: auto; display: flex; gap: 10px; align-items: center; }
.lsplit select { font: 12px var(--font-mono); background: var(--bg-card); color: var(--text); border: 1px solid var(--border-strong); border-radius: 6px; padding: 3px 6px; }
.lkeys { font-size: 11px; color: var(--text-faint); font-family: var(--font-mono); }
.tabs { align-items: center; flex-wrap: wrap; }
/* rows of minmax(0, 1fr): an auto row would grow to the code and push the pane past the window */
.lenses { display: grid; grid-template-columns: minmax(0, 1fr); grid-template-rows: minmax(0, 1fr); gap: 10px; height: calc(100vh - 170px); min-height: 320px; margin-top: -1px; }
.lenses.two { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
.lpane { display: flex; flex-direction: column; min-height: 0; min-width: 0; background: var(--bg-card); border: 1px solid var(--border); border-radius: 0 8px 8px 8px; }
.lpane.act { border-color: var(--accent); }
.lpane.loading .lbody { opacity: .45; }
.lhead { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; padding: 6px 10px; border-bottom: 1px solid var(--border); font-size: 12px; }
.lname { font-weight: 600; color: var(--heading); text-transform: capitalize; }
.lsrc { margin-left: auto; color: var(--text-faint); font-family: var(--font-mono); font-size: 11px; overflow-wrap: anywhere; }
.lord button { font: 11.5px var(--font-sans); background: none; color: var(--text-muted); border: 1px solid var(--border); padding: 2px 8px; cursor: pointer; }
.lord button:first-child { border-radius: 5px 0 0 5px; } .lord button:last-child { border-radius: 0 5px 5px 0; border-left: 0; }
.lord button.on { color: var(--heading); background: var(--bg-soft); border-color: var(--border-strong); }
.lic { color: var(--text-muted); display: flex; gap: 4px; align-items: center; }
.lbody { flex: 1; min-height: 0; overflow: auto; padding: 8px 12px 0; font-size: 13.5px; line-height: 1.5; }
.lbody[data-lens="code"] { padding: 0; }
/* room below the last line, so that any anchor can be scrolled to the top
   (a block, not padding: padding would set the pane's minimum height) */
.lbody::after { content: ''; display: block; height: calc(100vh - 160px); }
.lcode { border: 0; border-radius: 0; margin: 0; }
.lcode tr.ic { display: none; }
.showic .lcode tr.ic { display: table-row; }
.lnote { border: 1px dashed var(--border-strong); border-radius: 6px; padding: 8px 10px; margin: 8px 0; color: var(--text-muted); font-size: 13px; }
.lnote pre { white-space: pre-wrap; font: 12px var(--font-mono); margin: 6px 0 0; color: var(--text); }
.lfaint { color: var(--text-faint); }
.lk { color: var(--tk-kw); font-family: var(--font-mono); font-size: 12px; }
.lid { font-family: var(--font-mono); color: var(--accent); font-size: 12.5px; }
.lmod { margin: 4px 0 14px; } .lmod p { margin: 4px 0 0; color: var(--text); }
.ldef { margin: 0 0 16px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
.ldef h3 { font: 600 14px var(--font-mono); color: var(--heading); margin: 0 0 6px; }
.lsig { font-weight: 400; font-size: 12px; }
.lf { display: grid; grid-template-columns: 64px minmax(0, 1fr); gap: 8px; margin: 2px 0; }
.lfk { font: 12px var(--font-mono); color: var(--text-faint); } .lfk.lg { color: var(--tk-num); }
.lsteps { margin-top: 8px; }
.lstep { display: grid; grid-template-columns: minmax(80px, max-content) minmax(0, 1fr); gap: 12px; padding: 2px 4px; border-radius: 4px; }
.lstep .lid { min-width: 80px; }
.lctl { font-style: italic; color: var(--text-muted); padding: 2px 4px; }
.lkids { margin-left: 18px; border-left: 1px solid var(--border); padding-left: 8px; }
.lx { padding: 2px 4px; }
.ldebug { font: 13px/1.6 var(--font-mono); }
.dl { white-space: pre-wrap; padding: 0 4px 0 calc(4px + 6ch); text-indent: -6ch; } .dl.dh { color: var(--heading); font-weight: 600; margin-top: 8px; }
.dk { color: var(--tk-kw); } .ds { color: var(--tk-str); }
.lsum { font-size: 12.5px; color: var(--text-muted); margin: 0 0 8px; }
.sn { padding: 2px 4px; }
.sit > summary { cursor: pointer; padding: 1px 4px; list-style-position: inside; }
.lw { font: 12px var(--font-mono); color: var(--text-muted); margin-left: 8px; }
.lw > .lv, .lwk { white-space: nowrap; }
.lwk { color: var(--tk-num); } .lsep { display: inline-block; width: 12px; }
.lmat { display: inline-table; border-collapse: collapse; vertical-align: top; margin-left: 6px; font: 11px var(--font-mono); }
.lmat td { padding: 0 5px; border: 1px solid var(--border); text-align: right; }
.lholds { margin: 0 0 2px 22px; font-size: 12px; }
.lholds li.ok { color: var(--text-faint); } .lholds li.bad, .bad { color: var(--tk-kw); } .ok { color: var(--ok); }
.lhk { font-family: var(--font-mono); font-size: 11px; }
@media (max-width: 900px) { .lenses.two { grid-template-columns: minmax(0, 1fr); grid-template-rows: minmax(0, 1fr) minmax(0, 1fr); } }
`;

// The Hartree-Fock map: the files of the SCF project in the order a calculation
// runs, each marked with where it lives right now -- on main, on a branch still
// being built, or not yet written. tools/hf_map.json holds the order and the
// one-line descriptions; where each file is found is read from git per request.
function fileOn(ref, file) {
  try {
    execFileSync('git', ['-C', ROOT, 'cat-file', '-e', `${ref}:${file}`], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}
function lineCount(ref, file) {
  const t = tryGit('show', `${ref}:${file}`);
  return t ? t.split('\n').length : 0;
}

function mapPage() {
  const map = JSON.parse(readFileSync(path.join(ROOT, 'tools/hf_map.json'), 'utf8'));
  const branches = refs().filter((r) => r.name.startsWith('hf/'));
  let done = 0, wip = 0, todo = 0;
  const stages = map.stages
    .map((st, i) => {
      const rows = st.files
        .map(([file, what]) => {
          const w = fileOn('main', file) ? { ref: 'main', state: 'main' } : (branches.find((b) => fileOn(b.name, file)) ? { ref: branches.find((b) => fileOn(b.name, file)).name, state: 'branch' } : { ref: '', state: 'planned' });
          if (w.state === 'main') done++; else if (w.state === 'branch') wip++; else todo++;
          const n = w.ref ? lineCount(w.ref, file) : 0;
          const label = w.state === 'main' ? 'merged' : w.state === 'branch' ? `in progress · ${esc(w.ref)}` : 'not written yet';
          const name = w.ref
            ? `<a href="/?${new URLSearchParams({ ref: w.ref, file, ...(w.ref === 'main' ? {} : { all: '1' }) })}">${esc(file)}</a>`
            : `<span class="todo">${esc(file)}</span>`;
          return `<tr class="st-${w.state}"><td class="fname">${name}</td><td class="what">${esc(what)}</td><td class="state"><span class="chip ${w.state}">${label}</span>${n ? `<span class="lines">${n.toLocaleString()} lines</span>` : ''}</td></tr>`;
        })
        .join('');
      const body = `<p class="why">${esc(st.why)}</p><table class="map">${rows}</table>`;
      return st.collapsed
        ? `<section class="stage"><details><summary><span class="num">${i + 1}</span> ${esc(st.name)}</summary>${body}</details></section>`
        : `<section class="stage"><h2><span class="num">${i + 1}</span> ${esc(st.name)}</h2>${body}</section>`;
    })
    .join('');
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hartree-Fock map</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">
<style>${CSS}
${STYLE}
${MAPSTYLE}</style></head><body>
${bar('map')}</div>
<main class="mapmain">
  <h1>${esc(map.title)}</h1>
  <p class="tally"><span class="chip main">${done} merged</span> <span class="chip branch">${wip} in progress</span> <span class="chip planned">${todo} not written yet</span> · read live from git; reload for the latest</p>
  ${stages}
</main></body></html>`;
}

function bar(on) {
  return `<div class="bar"><strong>Viv source</strong>
  <nav class="modes"><a class="${on === 'map' ? 'on' : ''}" href="/hf">Hartree-Fock map</a><a class="${on === 'browse' ? 'on' : ''}" href="/?ref=main">Browse branches</a></nav>`;
}

const MAPSTYLE = `
.modes { display: flex; gap: 4px; }
.modes a { font-size: 13px; padding: 4px 10px; border-radius: 6px; text-decoration: none; color: var(--text-muted); }
.modes a.on { background: var(--bg-card); color: var(--heading); outline: 1px solid var(--border-strong); }
.mapmain { max-width: 1080px; margin: 0 auto; padding: 24px 16px 80px; }
.mapmain h1 { font-size: 22px; color: var(--heading); margin: 0 0 6px; }
.tally { color: var(--text-muted); font-size: 13px; margin: 0 0 22px; }
.stage { margin: 0 0 26px; }
.stage h2, .stage summary { font-size: 16px; color: var(--heading); margin: 0 0 4px; font-weight: 600; cursor: default; }
.stage summary { cursor: pointer; }
.num { display: inline-block; width: 22px; color: var(--text-faint); font-family: var(--font-mono); }
.why { color: var(--text-muted); font-size: 14px; margin: 0 0 8px 22px; max-width: 72ch; }
table.map { width: calc(100% - 22px); margin-left: 22px; border-collapse: collapse; background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; }
table.map td { padding: 7px 10px; border-top: 1px solid var(--border); vertical-align: top; font-size: 13.5px; }
table.map tr:first-child td { border-top: 0; }
.fname { font-family: var(--font-mono); font-size: 13px; white-space: nowrap; }
.fname a { color: var(--accent); }
.fname .todo { color: var(--text-faint); }
.what { color: var(--text); }
.state { white-space: nowrap; text-align: right; }
.lines { display: block; font-size: 11px; color: var(--text-faint); margin-top: 2px; }
.chip { font-size: 11.5px; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--border-strong); }
.chip.main { color: var(--ok); border-color: color-mix(in srgb, var(--ok) 50%, transparent); }
.chip.branch { color: var(--hot); border-color: color-mix(in srgb, var(--hot) 45%, transparent); }
.chip.planned { color: var(--text-faint); }
tr.st-planned .what { color: var(--text-faint); }
@media (max-width: 720px) { .fname { white-space: normal; overflow-wrap: anywhere; } .state { text-align: left; } table.map td { display: block; border-top: 0; } table.map tr { display: block; border-top: 1px solid var(--border); } }
`;

const STYLE = `
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--text); font-family: var(--font-sans); }
a { color: inherit; }
.bar { position: sticky; top: 0; z-index: 2; display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
  padding: 10px 16px; background: var(--bg-soft); border-bottom: 1px solid var(--border); }
.bar select { font: 14px var(--font-mono); background: var(--bg-card); color: var(--text); border: 1px solid var(--border-strong); border-radius: 6px; padding: 5px 8px; }
.bar .toggle { font-size: 13px; color: var(--accent); }
.bar .note { font-size: 12px; color: var(--text-faint); margin-left: auto; }
.wrap { display: grid; grid-template-columns: 280px minmax(0, 1fr) 260px; min-height: calc(100vh - 50px); }
aside { border-right: 1px solid var(--border); padding: 12px 14px; font-size: 13px; overflow: auto; max-height: calc(100vh - 50px); position: sticky; top: 50px; }
aside.side { border-right: 0; border-left: 1px solid var(--border); }
.h { font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: var(--text-faint); margin: 14px 0 6px; }
.h:first-child { margin-top: 0; }
.dir { font-family: var(--font-mono); color: var(--text-muted); margin-top: 10px; font-size: 12px; }
ul, ol { list-style: none; margin: 4px 0; padding: 0; }
.files li { display: flex; justify-content: space-between; gap: 6px; padding: 2px 6px; border-radius: 5px; font-family: var(--font-mono); font-size: 12.5px; }
.files li a { text-decoration: none; overflow-wrap: anywhere; }
.files li.cur { background: var(--bg-card); outline: 1px solid var(--border-strong); }
.files li:hover { background: var(--bg-card); }
.stat { white-space: nowrap; font-size: 11px; }
.stat .a { color: var(--ok); } .stat .d { color: var(--tk-kw); }
.log li { padding: 3px 0; line-height: 1.35; font-size: 12.5px; border-bottom: 1px solid var(--border); }
.log code { color: var(--accent); } .log .when { color: var(--text-faint); font-size: 11px; }
main { min-width: 0; padding: 14px 18px 60px; }
.fh h1 { font: 600 16px var(--font-mono); margin: 0 0 4px; color: var(--heading); overflow-wrap: anywhere; }
.facts { font-size: 12.5px; color: var(--text-muted); }
.tabs { display: flex; gap: 4px; margin: 10px 0 0; }
.tabs a { font-size: 13px; padding: 5px 12px; border: 1px solid var(--border); border-radius: 6px 6px 0 0; text-decoration: none; color: var(--text-muted); }
.tabs a.on { background: var(--bg-card); color: var(--heading); border-color: var(--border-strong); }
table.code { width: 100%; border-collapse: collapse; font: 13px/1.55 var(--font-mono); background: var(--bg-card); border: 1px solid var(--border); border-radius: 0 8px 8px 8px; margin-top: -1px; }
table.code td { vertical-align: top; padding: 0 10px; }
td.ln, td.sg { width: 1%; white-space: nowrap; text-align: right; color: var(--text-faint); user-select: none; border-right: 1px solid var(--border); }
td.ln a { text-decoration: none; }
td.src { white-space: pre-wrap; overflow-wrap: anywhere; }
tr:target { background: color-mix(in srgb, var(--accent) 18%, transparent); }
tr.add { background: color-mix(in srgb, var(--ok) 11%, transparent); }
tr.del { background: color-mix(in srgb, var(--tk-kw) 12%, transparent); }
tr.hunk td { color: var(--text-faint); background: var(--bg-soft); padding: 4px 10px; font-size: 12px; }
.cut { color: var(--text-faint); font-style: italic; margin-left: 6px; }
.outline li a { display: block; text-decoration: none; padding: 2px 4px; border-radius: 4px; font-family: var(--font-mono); font-size: 12px; overflow-wrap: anywhere; }
.outline li a:hover { background: var(--bg-card); }
.outline .kind { font-size: 10px; text-transform: uppercase; color: var(--text-faint); display: inline-block; width: 36px; }
.outline .k-test .kind { color: var(--ok); } .outline .k-use .kind { color: var(--tk-pre); } .outline .k-const .kind { color: var(--tk-num); }
.outline .args { color: var(--text-faint); }
.empty { color: var(--text-muted); }
@media (max-width: 1100px) { .wrap { grid-template-columns: 240px minmax(0, 1fr); } aside.side { display: none; } }
@media (max-width: 720px) { .wrap { display: block; } aside { position: static; max-height: none; border-right: 0; border-bottom: 1px solid var(--border); } }
`;

// A ref and a .viv file from the URL, both checked against what git holds
// before either reaches git or viv; null when they are not real.
function lensTarget(ref, file) {
  if (!refs().some((r) => r.name === ref)) return null;
  if (!file.endsWith('.viv') || !tree(ref).includes(file)) return null;
  return { ref, file };
}

const server = http
  .createServer(async (req, res) => {
    const u = new URL(req.url, 'http://x');
    if (u.pathname !== '/' && u.pathname !== '/hf' && u.pathname !== '/lens') {
      res.writeHead(404).end('not found');
      return;
    }
    try {
      if (u.pathname === '/hf' || (u.pathname === '/' && ![...u.searchParams.keys()].length)) {
        res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' }).end(mapPage());
        return;
      }
      const lens = u.searchParams.get('lens') || '';
      // one lens pane, for the page script to swap in
      if (u.pathname === '/lens') {
        const t = lensTarget(u.searchParams.get('ref') || '', u.searchParams.get('file') || '');
        if (!t || !LENSES.includes(lens)) {
          res.writeHead(400, { 'content-type': 'text/plain' }).end('no such ref, .viv file or lens');
          return;
        }
        const html = await lensPane(t.ref, t.file, lens, u.searchParams.get('order') === 'run' ? 'run' : '');
        res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' }).end(html);
        return;
      }
      const opts = {
        ref: u.searchParams.get('ref') || 'main',
        file: u.searchParams.get('file') || '',
        all: u.searchParams.get('all') === '1',
        view: u.searchParams.get('view') || '',
      };
      // a lens view renders its panes here, so the page reads without the script
      const t = LENSES.includes(lens) && opts.view !== 'diff' ? lensTarget(opts.ref, opts.file) : null;
      if (t) {
        const split = LENSES.includes(u.searchParams.get('split')) ? u.searchParams.get('split') : '';
        const order = u.searchParams.get('order') === 'run' ? 'run' : '';
        const panes = await Promise.all([lensPane(t.ref, t.file, lens, order), ...(split ? [lensPane(t.ref, t.file, split, '')] : [])]);
        panes.anchors = await fileAnchors(t.ref, t.file);
        Object.assign(opts, { lens, split, panes });
      }
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' }).end(page(opts));
    } catch (e) {
      res.writeHead(500, { 'content-type': 'text/plain' }).end(String(e && e.stack || e));
    }
  })
  .listen(PORT, '127.0.0.1', () => console.log(`viewer: http://localhost:${server.address().port}`));
