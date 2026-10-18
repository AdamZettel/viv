// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Putting an engine together with the reader's input, and reading what it
// printed. Shared by the pages, the worker's callers and the tests, so all
// three assemble and read a run the same way.
import { deckText } from './deck.js';
import { pyRepr } from './pyrepr.js';

/**
 * The engine's program with its input block replaced. The block runs from a
 * line starting "# ---- input:" to the line starting "# ---- end of input"
 * (engine-build.mjs refuses an engine without one).
 */
export function withInput(program, block) {
  const a = program.search(/^# ---- input:/m);
  const endAt = program.search(/^# ---- end of input.*$/m);
  if (a < 0 || endAt < 0) throw new Error('this engine program has no input block');
  const b = program.indexOf('\n', endAt);
  return program.slice(0, a) + block.replace(/\n?$/, '\n') + program.slice(b + 1);
}

/**
 * dft/'s reference for this deck, if the deck is exactly one of dft/'s decks:
 * the same text from the pseudopotentials down, which means the same crystal,
 * pseudopotentials and settings. Then dft/<m>_it_check.viv's pinned numbers
 * apply to this run as they do to that deck.
 */
export function dftReferenceFor(text, dftRefs) {
  const body = text.slice(text.indexOf('# GTH/HGH'));
  for (const [id, r] of Object.entries(dftRefs)) if (r.deckBody === body) return { id, ...r };
  return null;
}

/** The dft-bands input block: the deck, then the reference it is held to (or none). */
export function dftInputBlock(built, sourceName, dftRefs) {
  const text = deckText(built, sourceName);
  const ref = dftReferenceFor(text, dftRefs);
  const lines = ['# ---- input: the reader\'s deck', text.trimEnd(), '# the reference'];
  if (ref) {
    lines.push(`# dft/${ref.id}.viv is this same deck, so dft/${ref.id}_it_check.viv's numpy reference applies`,
      'HAS_REF = 1.0', `REF_ETOT = ${pyRepr(ref.etot)}`, `REF_NB = ${ref.nb}.0`,
      `REFPATH = [${ref.refpath.map((r) => `[${r.map(pyRepr).join(', ')}]`).join(',\n           ')}]`);
  } else {
    lines.push('HAS_REF = 0.0', 'REF_ETOT = 0.0', 'REF_NB = 0.0', 'REFPATH = zeros(1, 1)');
  }
  lines.push('# ---- end of input');
  return { block: lines.join('\n'), text, ref };
}

/** The reference variables as deck entries, for the compiled engine's loader. */
export function dftReferenceVariables(ref) {
  if (!ref) return { HAS_REF: 0, REF_ETOT: 0, REF_NB: 0, REFPATH: { dims: [1, 1], data: [0] } };
  return {
    HAS_REF: 1, REF_ETOT: ref.etot, REF_NB: ref.nb,
    REFPATH: { dims: [ref.refpath.length, ref.nb], data: ref.refpath.flat() },
  };
}

/**
 * What a run printed:
 *   NAME: number          a `show` of a number
 *   NAME: [a, b, ...]     a `show` of an array
 *   PASS / FAIL <test>    test results; "error: ..." is a failure too
 *   narrate: <stage>      where the program is
 *   figure / plot         the band structure, one plot line per band
 */
export function parseReport(text) {
  const shown = {};
  const series = {};
  const pass = [];
  const fail = [];
  const stages = [];
  const bands = [];
  let x = null;
  const nums = (s) => s.split(',').map((v) => Number(v));
  for (const line of text.split('\n')) {
    let m;
    if ((m = line.match(/^PASS (.*)$/))) pass.push(m[1]);
    else if ((m = line.match(/^FAIL (.*)$/))) fail.push(m[1]);
    else if ((m = line.match(/^error: (.*)$/))) fail.push(m[1]);
    else if ((m = line.match(/^narrate: (.*)$/))) stages.push(m[1]);
    else if ((m = line.match(/^plot: .*? ; \[([^\]]*)\] ; \[([^\]]*)\]\s*$/))) { x = nums(m[1]); bands.push(nums(m[2])); }
    else if ((m = line.match(/^([A-Za-z_][A-Za-z0-9_]*): \[([^\]]*)\]$/))) shown[m[1]] = nums(m[2]);
    else if ((m = line.match(/^([A-Za-z_][A-Za-z0-9_]*): (-?[0-9.]+(?:e[-+]?[0-9]+)?|-?inf|nan)$/))) {
      const v = Number(m[2]);
      (series[m[1]] = series[m[1]] || []).push(v);
      shown[m[1]] = v;
    }
  }
  return { shown, series, pass, fail, stages, x, bands };
}

/** A number the engine printed in three parts (engines/precise.viv), put back together. */
export function precise(shown, name) {
  const hi = shown[`${name}_HI`], mid = shown[`${name}_MID`], lo = shown[`${name}_LO`];
  if ([hi, mid, lo].some((v) => typeof v !== 'number')) return null;
  return hi + mid * 1e-6 + lo * 1e-12;
}
