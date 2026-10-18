// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The rules a post is held to -- ONE copy, used by three hosts:
//   - the site build (scripts/feed-build.mjs), for the posts committed in posts/
//   - the composer in the browser, so an author sees the verdict before publishing
//   - the server, which re-runs a submitted post and applies the same rules
// Plain JavaScript, no dependencies: the host supplies the program's run output.

export const BUDGET_SECONDS = 10;
export const FIGURE_BYTES = 160 * 1024;
export const SLUG_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
export const FIGURE_LINE = /^(figure|plot|plot2d|plot3d|scatter|quiver|bars|fill_between|rangebars|stage|axes|curve|dots|label|hline|vline|xtick|ytick|xscale|yscale|aspect|yright|grid|share|style|origin|colormap|clim|levels|circle|rect|segment|box|arrow|note|poly|draw|fade_in|fade_out|morph|move_to|say|wait|frame|together|apart):/;

// A post's header is a FENCED BLOCK: a line that is exactly `###`, the fields,
// a blank line, then the prose, then `###` again. The lexer skips it, so none
// of it needs a `#` in front and a paragraph is a paragraph:
//
//     ###
//     post: Doubling time
//     date: 2026-09-22
//     tags: algebra, logarithms
//     summary: Something growing 7% a year doubles in 10.24 years, not 14.3.
//
//     Growth by a fixed fraction each year is $A(t) = A_0 (1 + r)^t$. Asking
//     when it doubles means solving for an exponent, which is what a logarithm
//     is for.
//
//     ## A heading, in an article
//     ###
//
// The older form -- every line prefixed `#! ` and every prose line `#! |` --
// still reads, so posts written before this do not have to move.
const FENCE = /^###[ \t]*$/;

/** The header block's text, or null when the post uses the older `#!` lines. */
export function fencedHeader(src) {
  const lines = String(src || '').split('\n');
  let start = -1;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim() === '') continue;
    if (FENCE.test(lines[i])) { start = i; }
    break; // the fence opens the file, or there is none
  }
  if (start < 0) return null;
  for (let i = start + 1; i < lines.length; i++) if (FENCE.test(lines[i])) return { start, end: i, text: lines.slice(start + 1, i).join('\n') };
  return null;
}

/** One header line, `key: value`. A key is one word, or two (`scale large:`,
 *  `recorded large:`); every reader of a fenced header uses this pattern
 *  (src/lib/notebook.js, scripts/show-build.mjs, export.rs fence_text). */
export const HEADER_FIELD = /^([a-z][a-z0-9_]*(?: [a-z0-9_-]+)?):[ \t]*(.*)$/i;

/** { fields, body } of a fenced header: `key: value` lines, a blank line, then the prose. */
export function parseFenced(text) {
  const lines = String(text || '').split('\n');
  const fields = {};
  let i = 0;
  for (; i < lines.length; i++) {
    const m = lines[i].match(HEADER_FIELD);
    if (!m) break;
    fields[m[1].toLowerCase()] = m[2].trim();
  }
  return { fields, body: lines.slice(i).join('\n').replace(/^\n+|\n+$/g, '') };
}

/** The value of a header field: `key: value` in the fenced block, or `#! key ...`. */
export const meta = (src, key) => {
  const fence = fencedHeader(src);
  if (fence) return parseFenced(fence.text).fields[key] || '';
  return (src.match(new RegExp(`^#! ${key}\\s+(.+)$`, 'm')) || [])[1]?.trim() || '';
};

// every non-blank line starts with four spaces or a tab (richText.js's rule)
const indented = (ls) => ls.some((l) => l.trim()) && ls.every((l) => !l.trim() || /^( {4}|\t)/.test(l));
/** The prose paragraphs: the fenced block's body, or the `#! |` lines. */
export function notesOf(src) {
  const paras = [];
  let cur = [];
  // a paragraph indented four spaces (a table, a formula in columns) keeps
  // its lines and indentation; the renderer sets it as a block (richText.js)
  const flush = () => { if (cur.length) paras.push(indented(cur) ? cur.join('\n') : cur.map((l) => l.trim()).join(' ')); cur = []; };
  const fence = fencedHeader(src);
  if (fence) {
    for (const l of parseFenced(fence.text).body.split('\n')) {
      if (!l.trim()) flush(); else cur.push(l);
    }
    flush();
    return paras;
  }
  for (const l of src.split('\n')) {
    const m = l.match(/^#! \|\s?(.*)$/);
    if (!m) continue;
    if (!m[1].trim()) flush(); else cur.push(m[1]);
  }
  flush();
  return paras;
}

/** Scales: `#! scale <name> KEY=value KEY=value`. The program as written is the
 *  browser scale; a named scale sets top-level constants to run it bigger --
 *  `viv run --set N=20000 --set STEPS=100000 post.viv` -- the same program by
 *  the content address of every function, with nothing but those numbers
 *  changed. A RECORDED run is the output of one such run the author did
 *  elsewhere (a night on a cluster), attached to the post with a note
 *  (`#! recorded <name> <note>`), shown beside the browser run and marked as
 *  not run by the server. */
export function scalesOf(src) {
  const out = [];
  const fence = fencedHeader(src);
  const source = fence ? fence.text.split('\n').map((l) => `#! ${l.replace(/^(scale [a-z0-9_-]+):/i, '$1')}`).join('\n') : src;
  for (const l of source.split('\n')) {
    const m = l.match(/^#! scale\s+([a-z][a-z0-9_-]*)\s+(.*)$/i);
    if (!m) continue;
    const sets = {};
    for (const kv of m[2].trim().split(/\s+/)) {
      const e = kv.match(/^([A-Za-z_]\w*)=(.+)$/);
      if (e) sets[e[1]] = e[2];
    }
    out.push({ name: m[1], sets, line: l });
  }
  return out;
}
export function recordedNotes(src) {
  const notes = {};
  const fence = fencedHeader(src);
  const source = fence ? fence.text.split('\n').map((l) => `#! ${l.replace(/^(recorded [a-z0-9_-]+):/i, '$1')}`).join('\n') : src;
  for (const l of source.split('\n')) {
    const m = l.match(/^#! recorded\s+([a-z][a-z0-9_-]*)\s*(.*)$/i);
    if (m) notes[m[1]] = m[2].trim();
  }
  return notes;
}
const TOP_CONST = (name) => new RegExp(`^${name}\\s*=\\s*(-?\\d[\\d.]*(?:[eE][+-]?\\d+)?)\\s*(#.*)?$`);
/** The program with a scale's constants set: the same substitution `viv --set` makes. */
export function applyScale(src, sets) {
  const lines = src.split('\n');
  const missing = [];
  for (const [name, value] of Object.entries(sets)) {
    const re = TOP_CONST(name);
    const i = lines.findIndex((l) => re.test(l));
    if (i < 0) { missing.push(name); continue; }
    const comment = (lines[i].match(/#.*$/) || [''])[0];
    lines[i] = `${name} = ${value}${comment ? `    ${comment}` : ''}`;
  }
  return { src: lines.join('\n'), missing };
}

/** The program with its header taken off: what is left is what runs. An
 *  import line is program, not header, and stays. */
export function withoutHeader(src) {
  const fence = fencedHeader(src);
  const lines = String(src || '').split('\n');
  const rest = fence ? lines.slice(fence.end + 1) : lines.filter((l) => !l.startsWith('#!'));
  return rest.join('\n').replace(/^\n+/, '');
}

/** Five kinds of post.
 *    DEMO      leads with its figure: a picture and the run that backs it
 *    ARTICLE   leads with its prose: the notes are the body, the program sits inside
 *    LADDER    a program built one `rung` at a time; the page walks the rungs
 *    PROBLEM   a program whose tests do not pass yet: stubs, and tests that say what the
 *              right answers are. Solving it is making them pass.
 *    SOLUTION  a program that `#! solves <slug>`: the problem's tests, verbatim, all passing
 *  A demo or an article draws a figure; the other three may. */
export const KINDS = { demo: 'Demo', article: 'Article', ladder: 'Ladder', problem: 'Problem', solution: 'Solution' };
export const NEEDS_FIGURE = { demo: true, article: true };

/** The `test "..." { ... }` blocks of a program, at any depth, each with its body
 *  normalized (lines trimmed, blank lines dropped) so two copies compare by what
 *  they say. A solution must carry its problem's tests unchanged. */
export function testBlocksOf(src) {
  const out = [];
  const lines = String(src || '').split('\n');
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^\s*test\s+"((?:[^"\\]|\\.)*)"\s*\{/);
    if (!m) continue;
    let depth = 0;
    let j = i;
    for (; j < lines.length; j++) {
      const code = lines[j].replace(/"(?:[^"\\]|\\.)*"/g, '""').replace(/#.*$/, '');
      for (const ch of code) { if (ch === '{') depth++; else if (ch === '}') depth--; }
      if (depth <= 0) break;
    }
    out.push({ name: m[1], text: lines.slice(i, j + 1).map((l) => l.trim()).filter(Boolean).join('\n') });
    i = j;
  }
  return out;
}

/** Four lanes, least checked to most. The author chooses one of the first two:
 *    AI         written by an agent and posted without review
 *    DRAFT      its author wrote the tests and read the output; nobody else has
 *  REVIEWED and PUBLISHED are set by the feed server's rating, from
 *  other people's reviews and votes; LANE_NOTES below says it for readers.
 *  A post with an `agent:` line starts in AI unless its author says draft --
 *  which is a claim about the work, not about who typed it: tests written for
 *  it, and its output read. */
export const LANES = { ai: 'AI', draft: 'Draft' };
export const LANE_LABELS = { ai: 'AI', draft: 'Draft', reviewed: 'Reviewed', published: 'Published' };
export const LANE_NOTES = {
  ai: 'Posted by an AI model without review, or moved down here after reviews found problems. Its own tests pass; nobody else has confirmed it.',
  draft: 'A person wrote the tests and read the output. Nobody else has confirmed it yet, or reviews found problems.',
  reviewed: 'People other than the author approved it in reviews or up-votes. From Draft, one approving review or two up-votes gets a post here. Neither requires running the program; a review shows the reviewer\'s test results only if they ran it.',
  published: 'People other than the author approved it further. From Draft that takes four approving reviews or six up-votes. Later reviews and votes can move it back down.',
};
/** Said under the AI lane's note wherever a reader can post. */
export const AI_LANE_INVITE = 'You can post AI-written programs here without reviewing them first.';
export const defaultLane = (h) => (h.agent ? 'ai' : 'draft');

/** The kinds of work a version may claim on its parent (`edge:`); lineage.js checks them. */
export const EDGE_NAMES = ['fork', 'fix', 'extend', 'rewrite', 'adversarial-review', 'validate'];

/** The header fields, as the composer's form sees them. */
export function headerOf(src) {
  const h = {
    title: meta(src, 'post'),
    date: meta(src, 'date'),
    kind: meta(src, 'kind') || 'demo',
    tags: meta(src, 'tags').split(',').map((t) => t.trim()).filter(Boolean),
    summary: meta(src, 'summary'),
    agent: meta(src, 'agent'),
    from: meta(src, 'from'),
    edge: meta(src, 'edge'),
    reference: meta(src, 'reference'),
    solves: meta(src, 'solves'),
    channel: meta(src, 'channel'),
    equation: meta(src, 'equation'),
    notes: notesOf(src),
  };
  h.lane = meta(src, 'lane') || defaultLane(h);
  return h;
}

/** A header from form fields, in front of a program body. */
export function withHeader({ title, date, kind, lane, channel, equation, tags, summary, agent, from, edge, reference, solves, notes, scales, recordedNotes: rnotes }, body) {
  const lines = ['###', `post: ${title.trim()}`, `date: ${date}`];
  if (kind && kind !== 'demo') lines.push(`kind: ${kind}`);
  if (channel && channel.trim()) lines.push(`channel: ${channel.trim()}`);
  lines.push(`tags: ${(tags || []).map((t) => t.trim()).filter(Boolean).join(', ')}`);
  if (equation && equation.trim()) lines.push(`equation: ${equation.trim().replace(/\s*\n\s*/g, ' ')}`);
  if (summary && summary.trim()) lines.push(`summary: ${summary.trim().replace(/\s*\n\s*/g, ' ')}`);
  if (agent && agent.trim()) lines.push(`agent: ${agent.trim()}`);
  if (lane && lane !== defaultLane({ agent })) lines.push(`lane: ${lane}`);
  if (from && from.trim()) lines.push(`from: ${from.trim()}`);
  if (from && from.trim() && edge && edge !== 'fork') lines.push(`edge: ${edge}`);
  if (from && from.trim() && reference && reference.trim()) lines.push(`reference: ${reference.trim().replace(/\s*\n\s*/g, ' ')}`);
  if (solves && solves.trim()) lines.push(`solves: ${solves.trim()}`);
  for (const sc of scales || []) {
    const sets = Object.entries(sc.sets || {}).map(([k, v]) => `${k}=${v}`).join(' ');
    if (sc.name) lines.push(`scale ${sc.name}: ${sets}`);
    const note = rnotes && rnotes[sc.name];
    if (note && note.trim()) lines.push(`recorded ${sc.name}: ${note.trim()}`);
  }
  for (const para of notes || []) {
    lines.push('');
    for (const l of wrap(para.trim(), 92)) lines.push(l);
  }
  lines.push('###');
  return `${lines.join('\n')}\n\n${withoutHeader(body)}`;
}
function wrap(text, width) {
  const out = [];
  let line = '';
  for (const word of text.split(/\s+/).filter(Boolean)) {
    if (line && line.length + 1 + word.length > width) { out.push(line); line = word; } else line = line ? `${line} ${word}` : word;
  }
  if (line) out.push(line);
  return out.length ? out : [''];
}

/**
 * Judge a post from its source and the output of running it.
 * `seconds` is how long the run took, or null when the host cannot say.
 * `extra.ladder` is the parsed report of `viv ladder` (the host runs it when the
 * kind is ladder); `extra.problem` is the source of the problem a solution
 * `#! solves` (the host looks it up), or null when there is no such post;
 * `extra.channels` is the list of channel slugs that exist, when the host knows.
 * Returns { errors, tests, failed, figure, shown, seconds } -- errors empty means publishable.
 */
export function judge(src, out, seconds = null, extra = {}) {
  const errors = [];
  const h = headerOf(src);
  if (!h.title) errors.push('the header has no `post:` line; a post needs a title');
  if (!h.date) errors.push('the header has no `date:` line (YYYY-MM-DD)');
  else if (!/^\d{4}-\d{2}-\d{2}$/.test(h.date)) errors.push(`\`date: ${h.date}\` is not a date in the form YYYY-MM-DD`);
  if (!h.tags.length) errors.push('the header has no `tags:` line; give the post at least one tag');
  if (h.summary.length > 600) errors.push('the summary is over 600 characters; move the rest into the notes, the paragraphs after the header fields');
  if (h.equation.length > 300) errors.push('`equation:` is over 300 characters; it holds one formula, the one the post is about');
  if (/(^|[^\\])\$/.test(h.equation)) errors.push('`equation:` is already LaTeX, so leave the $ off (write `\\$` for a dollar sign)');
  if (!KINDS[h.kind]) errors.push(`\`kind: ${h.kind}\` is not a kind of post; use ${Object.keys(KINDS).slice(0, -1).join(', ')} or ${Object.keys(KINDS).slice(-1)[0]}`);
  if (h.kind === 'article' && !h.notes.length) errors.push('an article has a body: write the notes (the paragraphs after the header fields), or make it a demo');
  if (h.kind === 'ladder' && !/^\s*rung\s+"/m.test(src)) errors.push('a ladder has rungs: `rung "title" { ... }` blocks, each adding a function or two and a test');
  if (h.kind === 'ladder' && extra.ladder) {
    if (extra.ladder.error) errors.push(extra.ladder.error);
    for (const r of extra.ladder.rungs || []) for (const b of r.broken || []) errors.push(`rung ${r.n} (${r.title}): ${b}`);
  }
  if (h.solves && h.kind !== 'solution') errors.push(`\`solves: ${h.solves}\` is for a solution; set \`kind: solution\``);
  if (h.kind === 'solution') {
    if (!h.solves) errors.push('a solution names the problem it solves; add `solves: <the problem\'s slug>`');
    else if (!SLUG_RE.test(h.solves)) errors.push(`\`solves: ${h.solves}\` is not a valid slug (lowercase words joined by hyphens)`);
    else if (extra.problem === null) errors.push(`\`solves: ${h.solves}\`: there is no problem post with that slug`);
    else if (typeof extra.problem === 'string') {
      if (headerOf(extra.problem).kind !== 'problem') errors.push(`\`solves: ${h.solves}\`: that post is not a problem`);
      const mine = testBlocksOf(src);
      for (const t of testBlocksOf(extra.problem)) {
        const hit = mine.find((x) => x.name === t.name);
        if (!hit) errors.push(`the problem's test "${t.name}" is missing: a solution keeps every test of its problem, word for word`);
        else if (hit.text !== t.text) errors.push(`the problem's test "${t.name}" was changed: a solution keeps every test of its problem, word for word`);
      }
    }
  }
  if (!LANES[h.lane]) errors.push(`\`lane: ${h.lane}\`: a post can only be posted to ${Object.keys(LANES).map((l) => `\`${l}\``).join(' or ')}${LANE_LABELS[h.lane] ? ` (a post reaches ${h.lane} through other people\'s reviews and votes; its author cannot choose it)` : ''}`);
  if (h.from && !SLUG_RE.test(h.from)) errors.push(`\`from: ${h.from}\` is not a post's slug (lowercase words joined by hyphens)`);
  // lineage: `edge:` says what kind of work this version is on the one `from:` names.
  // Whether the claim is TRUE is lineage.js's check, which needs the parent's run.
  if (h.edge && !h.from) errors.push(`\`edge: ${h.edge}\` says what this version does to the post it came from: name that post with \`from: <slug>\``);
  if (h.edge && !EDGE_NAMES.includes(h.edge)) errors.push(`\`edge: ${h.edge}\` is not one of ${EDGE_NAMES.join(', ')}`);
  if (h.reference && h.edge !== 'validate') errors.push('`reference:` names what a validation compared against, so it goes with `edge: validate`');
  if (h.channel && !SLUG_RE.test(h.channel)) errors.push(`\`channel: ${h.channel}\` is not a channel's slug (lowercase words joined by hyphens)`);
  if (h.channel && extra.channels && !extra.channels.includes(h.channel)) errors.push(`\`channel: ${h.channel}\`: there is no channel by that name${extra.channels.length ? ` (there are: ${extra.channels.join(', ')})` : ''}`);
  for (const sc of scalesOf(src)) {
    if (!Object.keys(sc.sets).length) errors.push(`\`scale ${sc.name}\` changes no constant: write it as \`scale ${sc.name}: N=20000\``);
    for (const [k, v] of Object.entries(sc.sets)) if (!Number.isFinite(Number(v))) errors.push(`\`scale ${sc.name}\`: ${k}=${v} is not a number`);
    const { missing } = applyScale(src, sc.sets);
    for (const k of missing) errors.push(`\`scale ${sc.name}\` sets ${k}, but the program has no top-level \`${k} = <number>\` line`);
  }
  const notes = recordedNotes(src);
  for (const name of Object.keys(notes)) if (!scalesOf(src).some((sc) => sc.name === name)) errors.push(`\`recorded ${name}\` names no scale; add a \`scale ${name}:\` line`);
  const lines = (out || '').split('\n');
  const diagnostic = lines.find((l) => /^error\b/.test(l));
  if (diagnostic) errors.push(diagnostic);
  const tests = lines.filter((l) => l.startsWith('PASS ')).map((l) => l.slice(5));
  const failed = lines.filter((l) => l.startsWith('FAIL ')).map((l) => l.slice(5));
  if (h.kind === 'problem') {
    // a problem is posted UNSOLVED: its stubs run, and at least one test says what is missing
    if (!diagnostic && tests.length && !failed.length) errors.push('every test passes already: a problem is a program whose tests do not pass yet. Leave the functions unfinished, and write tests that check for the right answers');
  } else {
    for (const f of failed) errors.push(`FAIL ${f}`);
  }
  if (!diagnostic && !tests.length && !failed.length) errors.push(h.kind === 'problem' ? 'no test ran: a problem needs tests that its unfinished functions fail. Write `test "..." { ... }` blocks' : 'no test ran: a post needs at least one test. End the program with a `test "..." { ... }` block that checks its claim');
  const figure = lines.filter((l) => FIGURE_LINE.test(l)).join('\n');
  if (!diagnostic && NEEDS_FIGURE[h.kind] && !lines.some((l) => l.startsWith('figure:') || l.startsWith('stage:'))) errors.push(`the program draws no figure and stages no animation, and ${h.kind === 'article' ? 'an article' : 'a demo'} needs one: call figure() and plot(), or stage()`);
  if (seconds !== null && seconds > BUDGET_SECONDS) errors.push(`the program ran for ${seconds.toFixed(1)} s; a post must finish in ${BUDGET_SECONDS} s, because readers run it in their browser`);
  if (figure.length > FIGURE_BYTES) errors.push(`the figure data is ${(figure.length / 1024).toFixed(0)} KB; the limit is ${FIGURE_BYTES / 1024} KB. Plot fewer points, or use plot2d for a dense picture`);
  const shown = lines.filter((l) => !FIGURE_LINE.test(l)).map((l) => l.match(/^([A-Za-z_]\w*): (-?[\d.]+(?:e[+-]?\d+)?|-?inf|nan)$/)).filter(Boolean).map((m) => ({ name: m[1], value: m[2] }));
  return { errors, tests, failed, figure, shown, seconds, header: h, scales: scalesOf(src), recordedNotes: notes };
}

/** Judge a RECORDED run's output: it must pass its own tests and draw something. The
 *  server cannot rerun it; this is what it can check. */
export function judgeRecorded(out) {
  const errors = [];
  const lines = (out || '').split('\n');
  const diagnostic = lines.find((l) => /^error\b/.test(l));
  if (diagnostic) errors.push(diagnostic);
  const tests = lines.filter((l) => l.startsWith('PASS ')).map((l) => l.slice(5));
  const failed = lines.filter((l) => l.startsWith('FAIL ')).map((l) => l.slice(5));
  for (const f of failed) errors.push(`FAIL ${f}`);
  if (!tests.length && !failed.length && !diagnostic) errors.push('no test ran in the recorded output');
  const figure = lines.filter((l) => FIGURE_LINE.test(l)).join('\n');
  if (!lines.some((l) => l.startsWith('figure:') || l.startsWith('stage:'))) errors.push('the recorded output draws no figure');
  if (figure.length > FIGURE_BYTES * 4) errors.push(`the figure data in the recorded output is ${(figure.length / 1024).toFixed(0)} KB; the limit is ${FIGURE_BYTES * 4 / 1024} KB`);
  const shown = lines.filter((l) => !FIGURE_LINE.test(l)).map((l) => l.match(/^([A-Za-z_]\w*): (-?[\d.]+(?:e[+-]?\d+)?|-?inf|nan)$/)).filter(Boolean).map((m) => ({ name: m[1], value: m[2] }));
  return { errors, tests, failed, figure, shown };
}

/** "full N=20000 STEPS=100000" lines -> scales, for the composer's field. */
export function parseScaleLines(text) {
  return String(text || '').split('\n').map((l) => l.trim()).filter(Boolean).map((l) => {
    const [name, ...rest] = l.split(/\s+/);
    const sets = {};
    for (const kv of rest) { const e = kv.match(/^([A-Za-z_]\w*)=(.+)$/); if (e) sets[e[1]] = e[2]; }
    return { name: name.toLowerCase(), sets };
  });
}

/** A slug from a title: "The Gibbs overshoot never shrinks" -> "the-gibbs-overshoot-never-shrinks". */
export const slugify = (title) => title.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/['\u2019]/g, '').replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 72).replace(/-+$/, '');
