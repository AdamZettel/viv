// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
import { useEffect, useMemo, useState } from 'react';
import { api, hasServer } from '../lib/api.js';
import { lineageView, EDGE_KINDS, RECORDED } from '../lib/lineage.js';
import { LANE_LABELS, LANE_NOTES } from '../lib/postCheck.js';
import { PUBLIC_SITE } from '../lib/repo.js';

// The history of a post: every version of its family as a tree, each edge
// labeled with the kind of work it was, and per version what has actually been
// checked -- tests, reviews of that exact hash, validations, and adversarial
// findings still open there. The build supplies the family of committed posts
// (src/data/lineage.json); when a server answers, its record replaces that one,
// since it also has the reviews and the versions published from the site.
//
// `raw` is the family as lineage.js describes it; `focus` the hash of this
// page's version.
const KINDS = { ...EDGE_KINDS, ...RECORDED };
/** A review's verdict key, as a reader should see it. */
const VERDICT_LABEL = { reproduces: 'output matches', notes: 'output matches, with notes', issue: 'found a problem', fails: 'output does not match' };
/** An edge kind as a noun, for the show/hide buttons. */
const EDGE_NOUN = { fork: 'fork', fix: 'fix', extend: 'extension', rewrite: 'rewrite', 'adversarial-review': 'adversarial review', validate: 'validation' };
const short = (h) => (h ? h.slice(0, 8) : '');
const plural = (n, one, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

export default function History({ raw: fromBuild, focus: buildFocus, slug }) {
  const [raw, setRaw] = useState(fromBuild);
  const [focus, setFocus] = useState(buildFocus);
  const [source, setSource] = useState('build');
  const [open, setOpen] = useState(() => new Set());
  useEffect(() => {
    let alive = true;
    (async () => {
      if (!(await hasServer()) || !alive) return;
      try {
        const r = await api.lineage(slug);
        if (!alive || !r.lineage) return;
        setRaw(r.lineage);
        setFocus(r.lineage.current);
        setSource('server');
      } catch { /* the server does not have this post: keep what the build recorded */ }
    })();
    return () => { alive = false; };
  }, [slug]);
  const view = useMemo(() => (raw ? lineageView(raw, focus) : null), [raw, focus]);
  if (!view || !view.versions.length) return null;
  const byHash = new Map(view.versions.map((v) => [v.hash, v]));
  const toggle = (h) => setOpen((s) => { const n = new Set(s); if (n.has(h)) n.delete(h); else n.add(h); return n; });
  const shown = view.versions.filter((v) => !v.branch || open.has(v.branch));
  const branches = view.versions.filter((v) => v.branch === v.hash).length;
  const alone = view.versions.length === 1;
  // one version, and no composer to fork it with: nothing to show
  if (alone && PUBLIC_SITE) return null;
  return (
    <section className="lh" id="history">
      <h2>History <span className="lh-count">{plural(view.versions.length, 'version')}{view.kinds.length ? ` · ${view.kinds.map((k) => KINDS[k]?.label || k).join(', ')}` : ''}</span></h2>
      {alone ? (
        <p className="lh-lede">
          To add a version, fork this post and put an <code>edge:</code> line in the header saying what your version does to this one
          (<em>fix</em>, <em>extend</em>, <em>rewrite</em>, <em>adversarial-review</em> or <em>validate</em>). The site runs both
          programs to check that claim, and your version then appears here.
        </p>
      ) : (
        <p className="lh-lede">
          Every version of this program, and what each one changed from the one before it. Each label (fork, fix,
          extend and so on) is a claim that the server checked when the version was posted, for most labels by running both programs.
          {branches > 0 && ` Versions that branch off the path to the current best version are hidden until you open them with the buttons below.`}
        </p>
      )}
      <ol className="lh-tree">
        {shown.map((v, idx) => {
          const e = v.parentEdge;
          // the version this one came from: "the version above" only when it is
          // the one listed directly above; otherwise it is named
          const parent = e ? byHash.get(e.from) : null;
          const from = !parent || shown[idx - 1] === parent ? 'the version above' : `“${parent.title}”${parent.at ? ` (${String(parent.at).slice(0, 10)})` : ''}`;
          const kind = e ? KINDS[e.kind] : null;
          const closedKids = v.childEdges.map((x) => byHash.get(x.to)).filter((k) => k && k.branch === k.hash);
          const reviews = v.reviews || [];
          const reproduced = reviews.filter((r) => r.verdict === 'reproduces' || r.verdict === 'notes').length;
          return (
            <li key={v.hash} className={`lh-node${v.hash === view.best ? ' lh-best' : ''}${v.hash === focus ? ' lh-here' : ''}${v.kind === 'problem' ? ' lh-problem' : ''}`} style={{ '--depth': Math.min(v.depth, 6) }}>
              {e && (
                <p className={`lh-edge lh-edge-${e.kind}`} title={kind ? kind.says : ''}>
                  <span className="lh-edge-kind">{kind ? kind.label : e.kind}</span>
                  <span className="lh-edge-check">{describe(e, from)}</span>
                </p>
              )}
              <div className="lh-card">
                <p className="lh-title">
                  {v.deleted ? <span>{v.title}</span> : <a href={`/p/${v.slug}`}>{v.title}</a>}
                  {v.lane && <span className={`fp-lane fp-lane-${v.lane}`} title={LANE_NOTES[v.lane]}>{LANE_LABELS[v.lane] || v.lane}</span>}
                  {v.kind === 'problem' && <span className="lh-tag">problem</span>}
                  {v.hash === view.best && <span className="lh-tag lh-tag-best" title="Chosen among versions that are not problem posts: fewest failing tests from adversarial reviews, then the highest list (Published, Reviewed, Draft, AI), then the most validations, the most reviews that confirmed it, the most tests, and the newest">current best</span>}
                  {v.hash === focus && <span className="lh-tag lh-tag-here">this page</span>}
                </p>
                <p className="lh-meta">
                  @{v.author?.handle}{v.agent ? `, written with ${v.agent}` : ''}
                  {v.at ? ` · ${String(v.at).slice(0, 10)}` : ''}
                  {v.deleted ? ' · its post was deleted' : !v.current ? ' · an earlier version of its post' : ''}
                </p>
                <ul className="lh-checks">
                  <li className={v.failed && v.failed.length ? 'lh-warn' : 'lh-ok'}>
                    {plural((v.tests || []).length, 'test')} pass{v.failed && v.failed.length ? `, ${v.failed.length} fail${v.kind === 'problem' ? ' (expected: a problem post\'s failing tests are what it asks someone to fix)' : ''}` : ''}
                  </li>
                  <li className={reproduced ? 'lh-ok' : ''}>
                    {plural(reviews.length, 'review')} of this exact version{reviews.length ? `, ${reproduced} confirming it` : source === 'build' ? ' in this build (reviews are stored on the site\'s server)' : ''}
                    {reviews.some((r) => r.inferred) ? ' (some were written before versions were tracked, and are assigned to this one)' : ''}
                  </li>
                  {v.validations.length > 0 && <li className="lh-ok">{plural(v.validations.length, 'validation')}: {v.validations.map((x) => x.reference).filter(Boolean).join('; ')}</li>}
                  {v.kind !== 'problem' && (v.open.length + v.closed.length + v.inapplicable.length > 0) && (
                    <li className={v.open.length ? 'lh-warn' : 'lh-ok'}>
                      tests from adversarial reviews: {v.open.length} fail here{v.closed.length ? `, ${v.closed.length} pass here` : ''}{v.inapplicable.length ? `, ${v.inapplicable.length} cannot run here` : ''}
                    </li>
                  )}
                </ul>
                {(v.open.length > 0 || v.inapplicable.length > 0) && v.kind !== 'problem' && (
                  <details className="lh-more">
                    <summary>Which tests</summary>
                    <ul>
                      {v.open.map((f) => <li key={`o${f.test}`}><span className="lh-x">fails</span> {f.test}</li>)}
                      {v.inapplicable.map((f) => <li key={`i${f.test}`}><span className="lh-q">cannot run</span> {f.test} <small>(it calls a function this version does not define)</small></li>)}
                    </ul>
                  </details>
                )}
                {reviews.length > 0 && (
                  <details className="lh-more">
                    <summary>Reviews of this version</summary>
                    <ul>{reviews.map((r, i) => <li key={i}>@{r.reviewer}: {VERDICT_LABEL[r.verdict] || r.verdict}{r.inferred ? ' (written before versions were tracked)' : ''}</li>)}</ul>
                  </details>
                )}
                {closedKids.length > 0 && (
                  <p className="lh-branches">
                    {closedKids.map((k) => (
                      <button type="button" key={k.hash} className="lh-branch" aria-expanded={open.has(k.hash)} onClick={() => toggle(k.hash)}>
                        {open.has(k.hash) ? 'Hide' : 'Show'} the {EDGE_NOUN[k.parentEdge.kind] || KINDS[k.parentEdge.kind]?.label || k.parentEdge.kind} “{k.title}”{k.author?.handle ? ` by @${k.author.handle}` : ''}
                      </button>
                    ))}
                  </p>
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

/** One line on what the check of an edge found. */
function describe(e, from = 'the version above') {
  const f = e.facts || {};
  const n = (x) => (Array.isArray(x) ? x.length : 0);
  if (e.inferred) return 'recorded before these claims were checked';
  switch (e.kind) {
    case 'fork': return `copied from ${from}`;
    case 'extend': return `kept the ${plural(f.kept || 0, 'test')} of ${from} word for word, all passing; added ${plural(n(f.added), 'test')}`;
    case 'rewrite': return f.republished ? `republished; the output is the same, line for line (${f.lines} lines)` : `both run; the output is the same, line for line (${f.lines} lines)`;
    case 'fix': return `${plural(n(f.fixed), 'test')} that ${n(f.fixed) === 1 ? 'fails' : 'fail'} on ${from} ${n(f.fixed) === 1 ? 'passes' : 'pass'} here`;
    case 'adversarial-review': return `${plural(n(f.findings), 'added test')} ${n(f.findings) === 1 ? 'fails' : 'fail'} on the reviewed version`;
    case 'validate': return `compared against ${f.reference || 'an outside reference'}`;
    case 'edit': return 'republished by its author; the output changed';
    default: return '';
  }
}

function Ownership({ o }) {
  if (!o || !o.total) return <span>no definitions</span>;
  const share = o.person / o.total;
  return (
    <span className="lh-own">
      <span className="lh-own-bar" role="img" aria-label={`${o.person} of ${o.total} definitions owned by a person`}>
        <span style={{ width: `${Math.round(share * 100)}%` }} />
      </span>
      owned by a person: {o.person} of {o.total} definitions{o.model ? `; ${o.model} as the model wrote ${o.model === 1 ? 'it' : 'them'}` : ''}
    </span>
  );
}
