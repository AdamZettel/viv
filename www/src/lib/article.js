// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// An article's body, from its `#! |` paragraphs.
//
// Two marks give the prose structure without a markup language:
//     #! | ## A heading            a paragraph that starts with `## ` is a section heading
//     #! | [figure]                a paragraph that is only `[figure]` is where the figure goes
// A paragraph indented four spaces arrives rendered as a <pre> and stays a block.
// With no `[figure]`, the figure follows the lede. Works on the rendered
// paragraphs (html) and on the raw ones alike, since neither mark contains a
// character the renderer changes.
export function articleBlocks(paragraphs) {
  const blocks = [];
  let placed = false;
  for (const p of paragraphs) {
    const text = String(p).trim();
    if (text === '[figure]') { if (!placed) { blocks.push({ type: 'figure' }); placed = true; } continue; }
    if (text.startsWith('<pre')) { blocks.push({ type: 'pre', html: text }); continue; }
    const h = text.match(/^## (.+)$/s);
    if (h) blocks.push({ type: 'h2', html: h[1] });
    else blocks.push({ type: 'p', html: text });
  }
  if (!placed) blocks.unshift({ type: 'figure' });
  return blocks;
}

/** The reading time, from the words in the body and the lede. */
export const readMinutes = (summary, paragraphs) => Math.max(1, Math.round((`${summary} ${paragraphs.join(' ')}`.replace(/<[^>]+>/g, '').split(/\s+/).length + 60) / 220));
