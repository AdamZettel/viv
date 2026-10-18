// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// netlify.toml's `[[headers]]` rules, and the headers a path gets from them --
// for serve-dist.mjs (which serves dist/ with them, so the site test runs every
// page under them) and csp.mjs (which checks them against the build).
//
// Matching follows Netlify's path patterns as this file uses them: an exact
// path ("/"), or a prefix ending in "/*" ("/feed/*" matches /feed/ and all
// below it). Every rule that matches applies. When two matching rules set the
// same header their values are joined with ", " -- how Netlify is reported to
// combine them; netlify.toml avoids depending on it by never setting one
// header in two rules that can match the same path (the framing rules and the
// policy are separate headers).
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parse } from 'smol-toml';

export const TOML_PATH = join(dirname(dirname(dirname(fileURLToPath(import.meta.url)))), 'netlify.toml');

/** [{ for, values }] in file order. */
export function headerRules(text = readFileSync(TOML_PATH, 'utf8')) {
  return (parse(text).headers || []).map((h) => ({ for: h.for, values: h.values || {} }));
}

export function matches(pattern, path) {
  if (pattern.endsWith('/*')) return path.startsWith(pattern.slice(0, -1)) || path === pattern.slice(0, -2);
  if (pattern.endsWith('*')) return path.startsWith(pattern.slice(0, -1));
  return path === pattern;
}

/** { name: value } for a URL path. */
export function headersFor(rules, path) {
  const out = {};
  for (const r of rules) {
    if (!matches(r.for, path)) continue;
    for (const [k, v] of Object.entries(r.values)) out[k] = out[k] ? `${out[k]}, ${v}` : String(v);
  }
  return out;
}
