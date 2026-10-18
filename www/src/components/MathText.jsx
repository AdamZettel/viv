// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Chart labels with math: the `$...$` parts of a figure label set by
// src/lib/mathtext.js, the same layout `viv plot` draws (lang/viv/src/mathtext.rs).
// A label without math is drawn as before, as plain text.
import { chartLabel, hasMath, shapesSvg } from '../lib/mathtext.js';

const plainText = (s) => (s && s.includes('\\$') ? s.replace(/\\\$/g, '$') : s);

/** An SVG <text> at (x, y): `size` is its font size in px, `color` the CSS
 *  color of its fraction bars and radicals, `rotate` its turn in degrees
 *  (it sets the transform). Other props go to the <text>. */
export function SvgText({ text, size, x, y, textAnchor = 'start', rotate = 0, color = 'var(--text-muted)', ...rest }) {
  const transform = rotate ? `rotate(${rotate} ${x} ${y})` : undefined;
  if (!hasMath(text)) {
    return <text x={x} y={y} textAnchor={textAnchor} transform={transform} {...rest}>{plainText(text)}</text>;
  }
  const l = chartLabel(text, size);
  const shapes = shapesSvg(l, x, y, textAnchor, rotate, color);
  return (
    <>
      <text x={x} y={y} textAnchor={textAnchor} transform={transform} {...rest} dangerouslySetInnerHTML={{ __html: l.inner }} />
      {shapes && <g aria-hidden="true" dangerouslySetInnerHTML={{ __html: shapes }} />}
    </>
  );
}

/** A label inside HTML (a title, a legend entry): plain text, or a small
 *  inline SVG when it has math. `size` is the surrounding font size in px.
 *  `vertical` is for a label in vertical writing mode (writing-mode:
 *  vertical-rl): the SVG is drawn turned a quarter, reading top to bottom. */
export function MathInline({ text, size = 12, color = 'currentColor', vertical = false }) {
  if (!hasMath(text)) return plainText(text);
  const l = chartLabel(text, size);
  const pad = 1;
  const w = l.w + 2 * pad;
  const h = l.h + l.d + 2 * pad;
  const [bw, bh] = vertical ? [h, w] : [w, h];
  // the SVG's own text is laid out horizontally whatever the page's writing mode
  const style = vertical
    ? { display: 'inline-block', writingMode: 'horizontal-tb' }
    : { display: 'inline-block', writingMode: 'horizontal-tb', verticalAlign: `${-(l.d + pad)}px` };
  return (
    <svg className="math-inline" width={bw} height={bh} viewBox={`0 0 ${bw} ${bh}`} overflow="visible" role="img" aria-label={text} style={style}>
      <g transform={vertical ? `translate(${bw} 0) rotate(90)` : undefined}>
        <text x={pad} y={l.h + pad} fill={color} fontSize={size} dangerouslySetInnerHTML={{ __html: l.inner }} />
        <g dangerouslySetInnerHTML={{ __html: shapesSvg(l, pad, l.h + pad, 'start', 0, color) }} />
      </g>
    </svg>
  );
}
