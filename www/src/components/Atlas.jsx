// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The atlas of any program on the site (/atlas and /atlas/walk): which
// program comes from the address, `?open=<path>` for a staged program (the
// path `/workbench?open=` takes) or `?draft` for the text the workbench left
// (lib/atlas.js, writeDraft). The bundle is made in the browser
// (lib/atlas.js, buildAtlas) and handed to the explorer (Explorer.jsx) or the
// walk (ZoomWalk.jsx). The eight programs of /explore open there instead,
// with the bundles the build made.
import { useEffect, useMemo, useState } from 'react';
import Explorer from './Explorer.jsx';
import ZoomWalk from './ZoomWalk.jsx';
import { buildAtlas, readDraft, RECORD_BUDGET_MS } from '../lib/atlas.js';
import LookInside from './LookInside.jsx';

/** A staged path the page accepts: segments of word characters, dots and dashes, ending in `.viv`. */
const SAFE_PATH = /^(?!.*\.\.)[\w.-]+(\/[\w.-]+)*\.viv$/;

/** The program the address names: `{ name, file, src?, draft }`, or `{ error }`. */
function wanted() {
  const q = new URLSearchParams(window.location.search);
  if (q.has('draft')) {
    const d = readDraft();
    if (!d) return { error: 'The workbench left no program here. Open the atlas from the workbench again.' };
    const file = d.name ? `${d.name}.viv` : 'the workbench';
    return { name: d.name || 'scratch', file, src: d.src, draft: true };
  }
  const open = q.get('open') || '';
  if (!SAFE_PATH.test(open)) return { error: open ? `No program at ${open}.` : 'No program named. Every program page links to its atlas.' };
  return { name: open.replace(/\.viv$/, ''), file: open, draft: false };
}

/** The program's title: its file's name without the directory and the date a post's file starts with. */
const titleOf = (file) => file.split('/').pop().replace(/\.viv$/, '').replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/[-_]+/g, ' ');

/** The page with no program to show: what it does, and where its programs are. */
function AtlasEmpty({ error, walk }) {
  const base = walk ? '/atlas/walk' : '/atlas';
  return (
    <div className="atlas-empty">
      <h1>{walk ? 'Walk' : 'Atlas'}</h1>
      <LookInside links={{ atlas: '/atlas/', explore: '/explore/' }} here="atlas" heading="Look inside a program" />
      <p className="ex-error">{error}</p>
      <p>
        {walk
          ? 'The walk plays one run of a program: a camera follows each call to the def that runs it and marks the running line.'
          : 'The atlas draws one program as its defs and the calls between them, with its code and the values of one run beside it.'}{' '}
        It needs a program to open. Some to start from:
      </p>
      <ul>
        <li><a href="/explore/">The eight programs of Explore</a>, with their runs recorded at build time</li>
        <li><a href={`${base}?open=${encodeURIComponent('posts/2026-09-22-the-geometric-series-adds-up.viv')}`}>The geometric series adds up</a>, a short post</li>
        <li><a href="/posts/">Posts</a>: every post links to its atlas</li>
        <li><a href="/workbench/">The workbench</a>: write a program, then open its atlas from there</li>
      </ul>
    </div>
  );
}

export default function Atlas({ walk = false }) {
  const [want, setWant] = useState(null);
  useEffect(() => setWant(wanted()), []);

  const load = useMemo(() => {
    if (!want || want.error) return null;
    return async (onPhase) => {
      let src = want.src;
      if (src === undefined) {
        onPhase(`Fetching ${want.file}`);
        const r = await fetch(`/programs/${want.file}`);
        if (!r.ok) throw new Error(`no program at ${want.file} (${r.status})`);
        src = await r.text();
      }
      return buildAtlas({ name: want.name, src, onPhase });
    };
  }, [want]);

  if (!want) return null;
  if (want.error) return <AtlasEmpty error={want.error} walk={walk} />;
  const title = want.draft ? (want.name === 'scratch' ? 'the workbench program' : `${titleOf(want.file)}, as edited in the workbench`) : titleOf(want.file);
  const self = want.draft ? '?draft' : `?open=${encodeURIComponent(want.file)}`;

  if (walk) return <ZoomWalk load={load} title={title} back={`/atlas${self}`} />;
  return (
    <>
      <p className="eyebrow"><a href="/explore">Explore a run</a> · <code>{want.file}</code></p>
      <h1>The atlas of {title}</h1>
      <p className="lede ex-lede">
        The program as a picture of its defs and the calls between them, made by viv.wasm from {want.draft ? 'the text in the workbench' : 'the program and the modules it imports'}.
        Your browser runs it once for the call counts and records it for the values; a box opens to the steps and loops inside it.
      </p>
      <p className="ex-walklink">
        <a href={`/atlas/walk${self}`}>Walk through the run</a>: the atlas with a camera that follows each call as it runs.
        {!want.draft && <> · <a href={`/workbench?open=${want.file}`}>Open it in the workbench</a></>}
      </p>
      <Explorer load={load} title={title} recordBudget={RECORD_BUDGET_MS} />
    </>
  );
}
