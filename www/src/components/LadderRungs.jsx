// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
import { useEffect, useMemo, useState } from 'react';
import { highlight } from '../lib/highlight.js';
import { renderProse, loadKatex } from '../lib/richText.js';
import { atlasHref } from '../lib/atlasLink.js';

// A ladder, rung by rung: on /ladders/<name>, on a ladder post's page and in
// the composer's preview.
//
// Each rung is its prose, what it prints and its tests, followed by the whole
// program at that rung. Lines the rung below already had are dimmed; lines it
// adds carry a green bar and lines it changes in an existing function an amber
// one. A page that adds `ld-live` to .ld-steps (the /ladders page's script
// does, on wide screens) gets two columns: the prose on the left and, on the
// right, one pane held in view that shows the program of the rung being read.
//
// A rung from the site build carries `program` (the whole program, each def in
// the place it first appeared), `lineKinds` ('s' same, 'a' added, 'e' edited,
// one per line) and `printed` ([{ text, isNew }]). A rung from the composer or
// the server has only `code`, the rung's own body, and that is what is shown.
//
// Prose: a rung's `why` and `notes` are raw text with $math$. Rendered at build
// time they arrive as `whyHtml` / `notesHtml`; otherwise they are rendered
// here, with KaTeX loaded on demand.
//
// `openHref(rung)` gives the link that opens the program at that rung, or null.

function Prose({ text, html, katex, as: Tag = 'p', className }) {
  const out = html !== undefined && html !== null ? html : renderProse(text, katex);
  return <Tag className={className} dangerouslySetInnerHTML={{ __html: out }} />;
}

/** Highlighted code as one HTML string per source line, each with its spans closed. */
function linesHtml(code) {
  const out = [];
  const open = [];
  for (const line of highlight(code || '', 'viv').split('\n')) {
    let html = open.join('') + line;
    for (const m of line.matchAll(/<span\b[^>]*>|<\/span>/g)) {
      if (m[0] === '</span>') open.pop();
      else open.push(m[0]);
    }
    html += '</span>'.repeat(open.length);
    out.push(html);
  }
  return out;
}

const KIND = { s: 'same', a: 'add', e: 'edit' };

function Program({ r }) {
  const whole = typeof r.program === 'string';
  const code = whole ? r.program : r.code;
  const html = useMemo(() => linesHtml(code), [code]);
  const kinds = whole && r.lineKinds ? r.lineKinds : html.map(() => 'a');
  const added = kinds.filter((k) => k === 'a').length;
  const edited = kinds.filter((k) => k === 'e').length;
  return (
    <figure className="ld-prog">
      <figcaption>
        <span>{whole ? `The program at rung ${r.n}, ${html.length} lines` : `Rung ${r.n}`}</span>
        {whole && r.n > 1 && (
          <span className="ld-key">
            {added > 0 && <span className="ld-key-add">{added} new</span>}
            {edited > 0 && <span className="ld-key-edit">{edited} changed</span>}
          </span>
        )}
      </figcaption>
      <pre className="ld-code" tabIndex={0}>
        <code>
          {html.map((h, i) => (
            <span key={i} className={`ll ll-${KIND[kinds[i]] || 'same'}`} dangerouslySetInnerHTML={{ __html: h || ' ' }} />
          ))}
        </code>
      </pre>
    </figure>
  );
}

export default function LadderRungs({ rungs, openHref = () => null, showBroken = true }) {
  const needKatex = useMemo(() => rungs.some((r) => r.whyHtml === undefined && /\$[^$]+\$/.test([r.title, r.why, ...(r.notes || [])].join(' '))), [rungs]);
  const [katex, setKatex] = useState(null);
  useEffect(() => { if (needKatex) loadKatex().then(setKatex); }, [needKatex]);

  return (
    <div className="ld-frame">
    <div className="ld-steps" style={{ '--rungs': rungs.length }}>
      {rungs.map((r) => {
        const href = openHref(r);
        return (
          <section className="ld-rung" key={r.n}>
            <div className="ld-text" id={`rung-${r.n}`} data-n={r.n}>
              <header>
                <span className="ld-n" aria-hidden="true">{r.n}</span>
                <div>
                  <Prose as="h2" text={r.title} html={r.titleHtml} katex={katex} />
                  {r.why && <Prose className="ld-why" text={r.why} html={r.whyHtml} katex={katex} />}
                </div>
              </header>

              {(r.notes || []).map((t, i) => <Prose key={i} className="ld-note" text={t} html={r.notesHtml ? r.notesHtml[i] : undefined} katex={katex} />)}

              {showBroken && r.broken && r.broken.length > 0 && (
                <ul className="ld-broken">
                  {r.broken.map((b, i) => <li key={i}>{b}</li>)}
                </ul>
              )}

              {r.printed && r.printed.length > 0 && (
                <figure className="ld-out">
                  <figcaption>What rung {r.n} prints</figcaption>
                  <pre className="ld-printed">
                    {r.printed.map((p, i) => <span key={i} className={p.isNew ? 'ol ol-new' : 'ol'}>{p.text}</span>)}
                  </pre>
                </figure>
              )}

              <ul className="ld-tests">
                {r.tests.map((t) => (
                  <li className={t.isNew ? 'new' : ''} key={t.name}><span aria-hidden="true">✓</span><span className="ld-tname">{t.name}{t.isNew && <em> new</em>}</span></li>
                ))}
              </ul>
              {r.n > 1 && r.carried && r.carried.length > 0 && (
                // aispeak-ok: the one quiet line per rung
                <p className="ld-quiet">Everything else is unchanged; the compiler checks it.</p>
              )}
              {href && (
                <p className="ld-actions">
                  <a href={href}>Run rung {r.n} in the workbench</a>
                  {atlasHref(r.file) && <> · <a href={atlasHref(r.file)}>The atlas of rung {r.n}</a></>}
                </p>
              )}
            </div>
            <div className={r.n === rungs[0].n ? 'ld-side is-active' : 'ld-side'} data-n={r.n}>
              <Program r={r} />
            </div>
          </section>
        );
      })}
    </div>
    </div>
  );
}

/** The rules, as a list: on /ladders and under a ladder post. */
export function LadderRules() {
  return (
    <ol className="stages">
      <li><strong>Every test on every rung passes.</strong><span>The rungs are run, by the same <code>viv.wasm</code> your browser loads, every time the site is built and every time a ladder is published.</span></li>
      <li><strong>A test, once written, stays.</strong><span>Each rung reruns every test from the rungs below it, under the same name, and they must still pass.</span></li>
      <li><strong>Every rung adds a test.</strong><span>Each rung after the first adds at least one new test, and no rung may be only comments.</span></li>
      <li><strong>A function stays the same until a rung redefines it.</strong><span>A function that a rung does not redefine keeps its content address (a hash of its parsed code), and the compiler checks this. A redefinition is marked <span className="ld-chip edited">edited</span>, and its address must change: redefining a function with the code it already had is an error.</span></li>
    </ol>
  );
}
