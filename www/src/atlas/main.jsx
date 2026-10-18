// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The page `viv atlas` writes (lang/viv/src/atlas.rs): one file that opens
// with no network. The compiler puts the program's data in the
// `#atlas-data` script element (the code map of `viv graph --map`, the call
// counts of one run, the picture `viv graph --svg` draws) and this bundle
// beside it (scripts/atlas-viewer-build.mjs builds it from the site's own
// components/CodeMap.jsx).
//
// The walk is the site's (components/ZoomWalk.jsx, /explore/walk/<slug>),
// replaying the recording the compiler made during the same command
// (lang/viv/src/atlas_walk.rs): `walk` holds the explorer bundle and the run's
// story, so nothing runs here.
import { useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
// bundled whole, with its stylesheet and fonts inlined in viewer.css
// (scripts/atlas-viewer-build.mjs): the math is typeset with no network
import katex from 'katex';
import CodeMap from '../components/CodeMap.jsx';
import { Walk } from '../components/ZoomWalk.jsx';
import { indexBundle, buildTree } from '../lib/explore.js';
import { layoutWorld } from '../lib/zoomwalk.js';

const DATA = JSON.parse(document.getElementById('atlas-data').textContent);
// handed to CodeMap once, so it is never fetched
const GIVEN = { map: DATA.map, runs: DATA.calls ? { program: DATA.name, calls: DATA.calls } : null };

/** The recording as ZoomWalk.jsx's Walk takes it (lib/explore.js parseRecording, then buildTree and layoutWorld). */
function walkRun(w) {
  const ix = indexBundle(w.bundle);
  const rec = { events: w.story.events, total: w.story.total, printed: '', fails: w.fails, holdFails: w.holds };
  const tree = buildTree(rec.events, ix);
  const world = layoutWorld(ix, tree);
  return { ix, run: { rec, tree, world, note: w.note, seconds: w.seconds } };
}

function WalkView() {
  const r = useMemo(() => walkRun(DATA.walk), []);
  return <Walk ix={r.ix} run={r.run} katex={katex} slug={DATA.name} title={DATA.name} back={null} recorded="Recorded by viv atlas" />;
}

function Atlas() {
  const [view, setView] = useState('atlas');
  const defs = DATA.map.defs.filter((d) => !d.main).length;
  const mods = DATA.map.modules.length;
  const ran = DATA.calls ? `${DATA.calls.reduce((s, c) => s + c[2], 0).toLocaleString('en-US')} calls in one run` : 'not run: the calls are the ones the text makes';
  return (
    <>
      <header className="at-head">
        <h1>
          Atlas of <code>{DATA.name}</code>
        </h1>
        <span className="at-facts">
          {defs} defs in {mods} {mods === 1 ? 'module' : 'modules'} · {ran}
        </span>
        <div className="at-tabs">
          <button type="button" aria-pressed={view === 'atlas'} onClick={() => setView('atlas')}>
            Atlas
          </button>
          <button type="button" aria-pressed={view === 'walk'} onClick={() => setView('walk')} title={DATA.walk ? 'Walk through the run, line by line' : DATA.walkNote || 'No walk'}>
            Walk
          </button>
          <button type="button" aria-pressed={view === 'picture'} onClick={() => setView('picture')}>
            Picture
          </button>
        </div>
      </header>
      <div className="at-body">
        {view === 'atlas' ? (
          <CodeMap data={GIVEN} katex={katex} label={DATA.name} runLabel={DATA.calls ? 'the run viv atlas made' : null} />
        ) : view === 'walk' ? (
          DATA.walk ? <WalkView /> : <p className="at-none">{DATA.walkNote || 'No walk.'}</p>
        ) : DATA.svg ? (
          <div className="at-picture" dangerouslySetInnerHTML={{ __html: DATA.svg }} />
        ) : (
          <p className="at-none">No picture: lib/graphdraw could not draw this program ({DATA.svgError || 'no drawing'}).</p>
        )}
      </div>
    </>
  );
}

createRoot(document.getElementById('atlas')).render(<Atlas />);
