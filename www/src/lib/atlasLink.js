// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The link to a program's atlas, for every page that shows a program. `open`
// is the program's path under /programs, the one `/workbench?open=` takes. A
// program of /explore (src/data/explorables.mjs) opens there, with the bundle
// the build made; any other opens /atlas, which makes its bundle in the
// browser (src/lib/atlas.js). Kept apart from atlas.js so a page that only
// links pulls in none of it.
import { EXPLORABLES } from '../data/explorables.mjs';

/** The atlas page of the program at `open`, or null when there is none. */
export function atlasHref(open) {
  if (!open || !/\.viv$/.test(open)) return null;
  const curated = EXPLORABLES.find((e) => e.file === open);
  return curated ? `/explore/${curated.slug}` : `/atlas?open=${encodeURIComponent(open)}`;
}

/** Where the workbench leaves the editor's text for /atlas?draft (this browser only). */
export const DRAFT_KEY = 'viv.atlas.draft';

/** The workbench's text for /atlas?draft, or null. */
export function readDraft() {
  try {
    const t = window.localStorage.getItem(DRAFT_KEY);
    const d = t ? JSON.parse(t) : null;
    return d && typeof d.src === 'string' ? d : null;
  } catch {
    return null;
  }
}

/** Leave the workbench's text where /atlas?draft finds it; false when this browser will not keep it. */
export function writeDraft(name, src) {
  try {
    window.localStorage.setItem(DRAFT_KEY, JSON.stringify({ name, src }));
    return true;
  } catch {
    return false;
  }
}

/** The two ways into a program, for the "Look inside this program" block
 *  (components/LookInside.jsx and .astro): `atlas`, the program's map (the
 *  walk, which draws every def by module and the calls between them), and
 *  `explore`, the explorer page of a run recorded at build time, or null when
 *  the program has none. `open` is the path under /programs; `slug` names an
 *  /explore program directly. Null when there is no program to show. */
export function lookInsideLinks({ open = null, slug = null } = {}) {
  const curated = slug ? EXPLORABLES.find((e) => e.slug === slug) : open ? EXPLORABLES.find((e) => e.file === open) : null;
  if (curated) return { atlas: `/explore/walk/${curated.slug}`, explore: `/explore/${curated.slug}` };
  if (!open || !/\.viv$/.test(open)) return null;
  return { atlas: `/atlas/walk?open=${encodeURIComponent(open)}`, explore: null };
}

/** The block's words, shared by the React and the Astro component. */
export const LOOK_INSIDE = {
  heading: 'Look inside this program',
  atlas: 'The program’s map: every function and which calls which.',
  explore: 'Run it and step through what each part did, with the code, values and plots beside it.',
  noRun: 'There’s no recorded run of this program yet.',
};
