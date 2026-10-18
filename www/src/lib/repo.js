// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Two switches every page reads. Kept apart from site.ts so the browser
// islands can import them without pulling in the site's data.

export const GITHUB = 'https://github.com/AdamZettel/viv';

// Whether the repository at GITHUB is public. While this is false, no page
// links to it: no "Source on GitHub", no clone command, no "view this file"
// link; pages that would point at a file show the comparison's numbers or the
// reference's name instead. It is true for the launch, so the site must not
// deploy before the repository is public at exactly this address (every link
// and the first install command 404 until then): publish the repository
// first, or set this to false for a deploy that goes out before it.
export const REPO_PUBLIC = true;

/** A link into the repository (`path` relative to its root), or null while it is private. */
export const repoLink = (path = '', kind = 'blob') =>
  REPO_PUBLIC ? (path ? `${GITHUB}/${kind}/main/${path}` : GITHUB) : null;

// The public build (PUBLIC_SITE=1, set in netlify.toml) runs without the
// site's server: no /api/health probe, no sign-in, composing or reviewing.
// The dev server and local builds keep all of it.
// (Vite inlines import.meta.env.PUBLIC_SITE; a node script reads the environment.)
const env = import.meta.env || (typeof process !== 'undefined' ? process.env : {});
export const PUBLIC_SITE = env.PUBLIC_SITE === '1';

// The per-number checks (the "N / M numbers checked" badge on a post, its
// Claims panel of every number and what establishes it, and the tables on
// /claims). Hidden until the number linker matches exactly; the claim posts'
// own quantity tables and every test stay.
export const SHOW_NUMBER_CHECKS = false;
