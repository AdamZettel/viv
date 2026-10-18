// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The data palette: the colors of DATA, the same on the vipyr and the Viv side
// (figure lines and fills, the scenes, the walk's running def and call path,
// readouts). Identity colors (vipyr green, Viv indigo: marks, links, buttons)
// are the sides' own tokens in viv.css and are not here.
//
// This file is the one source. `node scripts/data-palette.mjs` writes it out as
//   www/src/styles/data-palette.css   --data-1 .. --data-4, --data-text, --data-glow, --data-bg/ink/muted/grid
//   lang/viv/src/data_palette.rs      the same table for the SVG figure renderer
// and `--check` fails when either is stale.
//
// The values are the owner-approved mock (palette/mock.py). Series order:
// cyan, amber, indigo, green. Light mode draws crisp strokes; dark mode gives
// the primary series a soft glow. Contrast (WCAG) of each series on its
// figure background, light #fbfbfd and dark #0d0f16; 3:1 is the bar for a
// stroke. Text in the data color (a readout) uses `text`, 4.5:1 or better:
//   cyan    #0891b2 3.6:1    #7dd3fc 11.5:1     text #0e7490 5.2:1 / #7dd3fc
//   amber   #d97706 3.1:1    #fbbf24 11.5:1
//   indigo  #4a3fc4 7.3:1    #8e86f5  6.3:1
//   green   #1f8a5b 4.2:1    #3fd08e  9.7:1
// bg, ink, muted and grid are a figure's own background, text, secondary
// text and grid lines.
export const DATA_PALETTE = {
  light: { series: ['#0891b2', '#d97706', '#4a3fc4', '#1f8a5b'], text: '#0e7490', glow: 'transparent', bg: '#fbfbfd', ink: '#1d1f2b', muted: '#6b7080', grid: '#e6e8ef' },
  dark: { series: ['#7dd3fc', '#fbbf24', '#8e86f5', '#3fd08e'], text: '#7dd3fc', glow: 'rgba(125, 211, 252, 0.35)', bg: '#0d0f16', ink: '#e8ebf5', muted: '#8a90a6', grid: '#22263a' },
};
