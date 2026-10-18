// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Named visualizations a VivCell can render from its own text output, keyed by
// the string a page passes as `viz="..."`. A viz parses only the `name: value`
// lines it needs, so the picture is drawn entirely from quantities the cell
// just computed.
//
// This site's registry is empty on purpose. Every viz written so far is a
// picture of a specific physics result — shell-model orbitals, a tokamak
// cross-section, an interaction fit's convergence — which belongs with the
// science that motivated it, not in a language's documentation. The tour and
// the docs draw figures through the language's own plot/plot2d/plot3d/scene
// builtins instead, which is the mechanism a reader can actually reuse.
//
// To add one: export a component taking { text } and list it here.
export const VIZ = {};
