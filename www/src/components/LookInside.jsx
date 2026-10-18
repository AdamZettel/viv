// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// "Look inside this program": the program's atlas (its map) and, when the
// build recorded a run, its explorer, as two labeled buttons with one line
// each. `links` is lookInsideLinks()'s result (src/lib/atlasLink.js); `here`
// marks the page the reader is on ('atlas' or 'explore'), which shows as the
// current one. The Astro twin is LookInside.astro.
import { LOOK_INSIDE } from '../lib/atlasLink.js';
import '../styles/lookinside.css';

export const MapIcon = () => (
  <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" aria-hidden="true">
    <rect x="3" y="4" width="7" height="5" rx="1" /><rect x="14" y="4" width="7" height="5" rx="1" /><rect x="8.5" y="15" width="7" height="5" rx="1" />
    <path d="M6.5 9v3h11V9M12 12v3" />
  </svg>
);
export const StepIcon = () => (
  <svg viewBox="0 0 24 24" width="15" height="15" fill="currentColor" aria-hidden="true">
    <path d="M5 4.5v15l10-7.5z" /><rect x="16.5" y="4.5" width="2.5" height="15" rx="0.6" />
  </svg>
);

export default function LookInside({ links, compact = false, here = null, heading = LOOK_INSIDE.heading }) {
  if (!links || !links.atlas) return null;
  const cur = (k) => (here === k ? { 'aria-current': 'page' } : {});
  return (
    <section className={`li-block${compact ? ' li-compact' : ''}`} aria-label={heading}>
      <p className="li-head">{heading}</p>
      <ul className="li-ways">
        <li className="li-way">
          <a className="li-btn" href={links.atlas} {...cur('atlas')}><MapIcon />Atlas</a>
          <p className="li-line">{LOOK_INSIDE.atlas}</p>
        </li>
        <li className="li-way">
          {links.explore ? (
            <>
              <a className="li-btn" href={links.explore} {...cur('explore')}><StepIcon />Explore</a>
              <p className="li-line">{LOOK_INSIDE.explore}</p>
            </>
          ) : (
            <p className="li-line li-norun">{LOOK_INSIDE.noRun}</p>
          )}
        </li>
      </ul>
    </section>
  );
}
