// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// A figure's place, empty: the static stand-in /posts renders for a card that
// loads later (its `boxes` prop, measured at build by src/lib/figureBoxes.js).
// The box is as tall as the figure will be at the card's width, so nothing
// below it moves when the card arrives. A line chart's height follows
// LineChart.jsx's rule, written with the box's own width (cqw).
export default function FigureBox({ box }) {
  if (box.kind === 'line') {
    return (
      <div className="fp-figbox-wrap fp-figbox-vc" aria-hidden="true">
        <div className="fp-figbox fp-figbox-line" style={{ height: `calc(${box.head}px + min(var(--figbox-cap, 9999px), ${box.hx}px + clamp(200px, (100cqw - ${box.wx}px) * 0.52, 340px)))` }} />
      </div>
    );
  }
  return (
    <div className="fp-figbox-wrap" aria-hidden="true">
      <div className="fp-figbox fp-figbox-ratio" style={{ marginTop: 8 + (box.head || 0), aspectRatio: `${box.w} / ${box.h}`, maxWidth: box.max ? `${box.max}px` : undefined }} />
    </div>
  );
}
