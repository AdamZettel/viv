// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Claim declarations in a post. A claim is a statement checked by a program
// (claims/<topic>/<slug>.viv, scripts/claims/claimfiles.mjs), and an article
// that relies on one says so twice:
//
//     ###
//     post: Finite differences and spectral methods
//     kind: article
//     claims: spectral-derivative, fd-local-near-a-jump
//
//     ... the spectral derivative converges geometrically [claim: spectral-derivative] ...
//     ###
//
// The header's `claims:` line declares them; `[claim: <slug>]` in the prose
// places one. On a line of its own it shows as a card (the sentence, its
// status, links to the claim's post and program); inside a sentence as a badge.
// The build refuses an article that cites a claim that does not exist, is not
// `checked`, or is placed in the prose without being declared.

export const CLAIM_REF = /\[claim:\s*([a-z0-9]+(?:-[a-z0-9]+)*)\s*\]/g;

/** The slugs a header's `claims:` field declares. */
export const declaredClaims = (field) => String(field || '').split(',').map((s) => s.trim()).filter(Boolean);

/** The slugs `[claim: slug]` places in a text, in order, each once. */
export function placedClaims(text) {
  return [...new Set([...String(text || '').matchAll(CLAIM_REF)].map((m) => m[1]))];
}

/** What is wrong with a post's claim references, given the claims that exist
 *  ({ slug: { status } }). An empty list means none. */
export function claimRefErrors(declared, placed, claims) {
  const errors = [];
  for (const s of declared) {
    const c = claims[s];
    if (!c) errors.push(`\`claims:\` names ${s}, which is not a claim in claims/`);
    else if (c.status !== 'checked') errors.push(`\`claims:\` names ${s}, whose status is ${c.status}, not checked (python3 tools/claims/run.py --only ${s} --write)`);
  }
  for (const s of placed) if (!declared.includes(s)) errors.push(`the prose places [claim: ${s}] but the header's \`claims:\` line does not declare it`);
  return errors;
}

import { texToPlain } from './richText.js';

const esc = (s) => String(s).replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));
const STATUS_MARK = { checked: '✓ checked', mismatch: '✗ mismatch', stale: 'stale', unrun: 'not run' };

/** One claim as a card (a paragraph of its own) or a badge (inside a sentence). */
export function claimHtml(c, slug, block) {
  if (!c) return `<span class="cr-badge cr-missing">claim ${esc(slug)}: no such claim</span>`;
  const mark = `<span class="cr-status cr-${esc(c.status)}">${STATUS_MARK[c.status] || esc(c.status)}</span>`;
  if (!block) return `<a class="cr-badge cr-${esc(c.status)}" href="/p/${esc(slug)}" title="${esc(texToPlain(c.claim))}">claim ${esc(slug)} ${mark}</a>`;
  return `<aside class="cr-card cr-${esc(c.status)}" aria-label="A claim this article relies on">`
    + `<p class="cr-head"><span class="cr-kind">claim</span>${mark}<span class="cr-topic">${esc(c.topicTitle || c.topic)}${c.section ? ` · ${esc(c.section)}` : ''}</span></p>`
    + `<p class="cr-text"><a href="/p/${esc(slug)}">${c.claimHtml || esc(c.claim)}</a></p>`
    + `<p class="cr-links"><a href="/p/${esc(slug)}">The claim and its program</a><a href="/workbench?open=${esc(c.file)}">Open the program in the workbench</a></p>`
    + '</aside>';
}

/** Rendered prose with every `[claim: slug]` replaced: a paragraph holding only
 *  the reference becomes a card, any other a badge. `claims` is { slug: claim }. */
export function withClaimRefs(html, claims) {
  return String(html || '')
    .replace(/<p>\s*\[claim:\s*([a-z0-9-]+)\s*\]\s*<\/p>/g, (_, s) => claimHtml(claims[s], s, true))
    .replace(CLAIM_REF, (_, s) => claimHtml(claims[s], s, false));
}
