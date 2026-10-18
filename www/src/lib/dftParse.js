// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Read what the plane-wave DFT program prints. One parser, used by the build
// (scripts/dft-precompute.mjs, on viv.wasm's output), by the page (on the
// worker's output), and on copper's recorded native run -- so all three are
// the same view of the same text.
//
//   ETOT: <Ha>                     one per SCF iteration
//   DRHO: <e/bohr^3 rms>           one per SCF iteration, then once more at the end
//   NAME: <number>                 every `show`
//   PASS <test name> / FAIL <test name> -- <why>
//   plot: band ; [x...] ; [e...]   one per band, along L - Gamma - X
export function parseDft(text) {
  const etot = [];
  let drho = [];
  const shown = {};
  const pass = [];
  const fail = [];
  const bands = [];
  let x = null;
  let figure = null;
  for (const line of text.split('\n')) {
    let m;
    if ((m = line.match(/^ETOT: (.+)$/))) etot.push(Number(m[1]));
    else if ((m = line.match(/^DRHO: (.+)$/))) { drho.push(Number(m[1])); shown.DRHO = Number(m[1]); }
    else if ((m = line.match(/^PASS (.*)$/))) pass.push(m[1]);
    else if ((m = line.match(/^FAIL (.*)$/))) fail.push(m[1]);
    else if ((m = line.match(/^error: (.*)$/))) fail.push(m[1]);
    else if ((m = line.match(/^figure: (.*?) ; (.*?) ; ([^;]*)/))) figure = { title: m[1], ylabel: m[3].trim() };
    else if ((m = line.match(/^plot: .*? ; \[([^\]]*)\] ; \[([^\]]*)\]\s*$/))) {
      const nums = (s) => s.split(',').map((v) => Number(v));
      x = nums(m[1]);
      bands.push(nums(m[2]));
    } else if ((m = line.match(/^([A-Za-z_][A-Za-z0-9_]*): (-?[0-9.]+(?:e[-+]?[0-9]+)?)$/))) shown[m[1]] = Number(m[2]);
  }
  // the engine shows DRHO once more after the loop: the same number as the last iteration's
  if (drho.length === etot.length + 1) drho = drho.slice(0, -1);
  return { etot, drho, shown, pass, fail, x, bands, figure };
}

// Tests come in two kinds, and the page shows them apart: agreement with the
// numpy reference (a bug detector) and agreement with published LDA values (a
// method detector). The check files put the reference tests first, then a
// marker comment; here the split is by wording, which the generator controls.
export function isReferenceTest(name) {
  return /numpy reference|SCF converged|band-path point converged/.test(name);
}
