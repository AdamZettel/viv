// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The top of /gallery: every scene (src/data/scenes.mjs) as a large dark
// card, its picture playing from the program's records while it is on
// screen, its title, readout and reference. The title's link covers the
// card (a stretched ::after), and the Run button sits above that layer, so
// a button never nests inside an <a>.
import { useRef } from 'react';
import SceneViz, { typeset } from './SceneViz.jsx';

function Card({ scene, extra }) {
  const readout = useRef(null);
  return (
    <div className="scn-card">
      <SceneViz kind={scene.kind} slug={scene.slug} data={extra[scene.id] || null} live={scene.live === 'auto' ? 'auto' : null} fig={scene.fig} readout={readout} aspect="4 / 3" label={scene.title} />
      <div className="scn-card-text">
        <h3 className="scn-card-title"><a className="scn-card-link" href={scene.href}>{scene.title}</a></h3>
        <p className="scn-readout" ref={readout} aria-live="off" />
        <p className="scn-ref">{typeset(scene.check || `Checked against ${scene.ref}`)}</p>
      </div>
    </div>
  );
}

export default function SceneCards({ scenes, extra = {} }) {
  return (
    <div className="scn-cards">
      {scenes.map((s) => (
        <Card key={s.id} scene={s} extra={extra} />
      ))}
    </div>
  );
}
