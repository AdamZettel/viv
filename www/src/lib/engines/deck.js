// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// From a parsed structure file to the DFT engine's input deck: the same deck
// dft/ref/make_deck.py writes for the materials in dft/ref/materials.py, with
// the crystal read from a file instead of that table.
//
//   buildDeck(parsed, overrides)  ->  { deck, crystal, settings, pseudos, notes, refusals }
//   deckText(built)               ->  the .viv deck, as make_deck.py prints it
//   deckBinary(built)             ->  the same values as bytes, for the compiled engine
//
// The settings follow materials.py wherever it has something to say:
//   - a crystal with the composition of a material in that table (silicon,
//     gallium arsenide, ...) takes that material's settings whole: its measured
//     cutoff, k-mesh, smearing and band count;
//   - otherwise each element brings the cutoff measured for it there (the
//     largest wins), and an element that appears in no material there has no
//     measured cutoff, so the reader must give one: the page says the run is
//     then unvalidated;
//   - the pseudopotential of an element is the one materials.py uses for it
//     (Ga GTH-PADE-q3, Cu q11), and make_pseudo.py's default (the most valence
//     electrons) for any other element.
// A setting the input file states (a pw.x ecutwfc, say) or the reader types
// replaces the default, and every setting carries where it came from.
import TABLES from './dft-tables.json' with { type: 'json' };
import { canonicalFcc, Refusal } from './crystal.js';
import { pyRepr } from './pyrepr.js';

export { TABLES };

// Fixed by make_deck.py for every material; the reader does not set these.
const FIXED = { MIX: 0.6, SCF_TOL: 1.0e-9, MAXIT: 80, NITER: 3, NITER_BANDS: 60, EIG_TOL: 1.0e-5 };
export const CORNER = [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.5]];
// The GTH table's layout, as dft/pseudo/make_pseudo.py names it in the deck:
// local coefficients C1..C4, channels l = 0..3, projectors per channel.
const GTH_NLOC = 4, GTH_NL = 4, GTH_NPMAX = 3;
export const MAX_ATOMS = 8;

/** Every element the materials table uses, with its GTH entry and measured cutoff. */
function measured() {
  const out = {};
  for (const [name, m] of Object.entries(TABLES.materials)) {
    for (const a of m.atoms) {
      const el = a.entry.split(' ')[0];
      const cur = out[el] || { entry: a.entry, ecut: 0, materials: [] };
      cur.ecut = Math.max(cur.ecut, m.ecut);
      if (!cur.materials.includes(name)) cur.materials.push(name);
      out[el] = cur;
    }
  }
  return out;
}
export const MEASURED = measured();

/**
 * The GTH entry the engine uses for an element, or a Refusal saying why there
 * is none it can use. Mirrors make_pseudo.py: an explicit valence (from
 * materials.py) picks that entry; otherwise the one with the most electrons.
 */
export function pseudoFor(el) {
  const cands = TABLES.gth[el];
  if (!cands || !cands.length) {
    return { refusal: `No pseudopotential for ${el}: CP2K's GTH-PADE table, which the engine's pseudopotentials come from, has no entry for it.` };
  }
  const want = MEASURED[el] && MEASURED[el].entry;
  const e = want ? cands.find((c) => c.name === want) : cands.reduce((a, b) => (b.q > a.q ? b : a));
  // the iterative engine applies projectors up to l = 2 (real_ylm in dft/pwdft_it.viv)
  if (e.shells.some((s, l) => l >= 3 && s.nprj > 0)) {
    return { refusal: `${e.name} has f-channel projectors (l = 3); the engine applies projectors only up to l = 2.` };
  }
  return { entry: e };
}

/** Composition as a sorted "El:n" list, for matching against materials.py. */
const composition = (atoms) => {
  const c = {};
  for (const a of atoms) c[a.el] = (c[a.el] || 0) + 1;
  return Object.keys(c).sort().map((k) => `${k}:${c[k]}`).join(' ');
};

/** The material in materials.py with this composition, if any. */
export function matchMaterial(atoms) {
  const want = composition(atoms);
  for (const [name, m] of Object.entries(TABLES.materials)) {
    if (composition(m.atoms.map((a) => ({ el: a.entry.split(' ')[0] }))) === want) return name;
  }
  return null;
}

/**
 * Build the deck. `parsed` is an adapter's output; `overrides` holds values
 * the reader typed ({ ecut, nkmesh, smear, nbands }, any subset). Refusals are
 * collected rather than thrown where the page can still show what it
 * understood; a structure that is not fcc throws, since there is nothing to show.
 */
export function buildDeck(parsed, overrides = {}) {
  const refusals = [];
  const notes = [...(parsed.notes || [])];
  const crystal = canonicalFcc(parsed.lattice, parsed.atoms);
  if (crystal.nPrimitive > MAX_ATOMS) {
    refusals.push(`${crystal.nPrimitive} atoms in the primitive cell: more than the ${MAX_ATOMS} a browser run allows. Run it on your own machine with dft/run.sh.`);
  }
  if (crystal.nInput !== crystal.nPrimitive) {
    notes.push(`The file's cell holds ${crystal.nInput} atoms; the primitive fcc cell holds ${crystal.nPrimitive}, and that is what the engine computes.`);
  }
  // species in order of first appearance, as make_deck.py orders them
  const species = [];
  for (const a of crystal.atoms) if (!species.includes(a.el)) species.push(a.el);
  const pseudos = species.map((el) => {
    const p = pseudoFor(el);
    if (p.refusal) refusals.push(p.refusal);
    return p.entry || null;
  });
  const nelec = pseudos.every(Boolean)
    ? crystal.atoms.reduce((s, a) => s + pseudos[species.indexOf(a.el)].zion, 0) : null;

  // ---- settings, each with where it came from
  const material = matchMaterial(crystal.atoms);
  const settings = {};
  const set = (k, value, from) => { settings[k] = { value, from }; };
  if (material) {
    const m = TABLES.materials[material];
    const why = `dft/ref/materials.py, ${material}`;
    set('ecut', m.ecut, `${why} (a measured cutoff)`);
    set('nkmesh', m.nkmesh, why);
    set('smear', m.smear, why);
    set('nbands', m.nbands, why);
  } else {
    const unmeasured = species.filter((el) => !MEASURED[el]);
    if (!unmeasured.length) {
      const ecut = Math.max(...species.map((el) => MEASURED[el].ecut));
      const who = species.filter((el) => MEASURED[el].ecut === ecut)[0];
      set('ecut', ecut, `the cutoff measured for ${who} in dft/ref/materials.py (${MEASURED[who].materials.join(', ')})`);
    }
    if (nelec !== null) {
      const metal = nelec % 2 === 1;
      set('smear', metal ? 0.01 : 0.0, metal
        ? `an odd electron count (${nelec}) makes it a metal: materials.py's metal smearing`
        : `an even electron count (${nelec}): treated as an insulator; give a smearing if it is a metal`);
      set('nkmesh', metal ? 4 : 2, metal ? 'materials.py\'s metal mesh' : 'materials.py\'s semiconductor mesh');
      set('nbands', Math.ceil(nelec / 2) + 4, 'half the electrons, plus four empty bands');
    }
  }
  for (const [k, v] of Object.entries(parsed.settings || {})) settings[k] = v;
  for (const [k, v] of Object.entries(overrides)) {
    if (v !== undefined && v !== null && v !== '' && Number.isFinite(Number(v))) settings[k] = { value: Number(v), from: 'set by you', user: true };
  }
  // (an element with no pseudopotential at all has been refused above; its cutoff is moot)
  const noCutoff = species.filter((el, i) => !MEASURED[el] && pseudos[i]);
  if (!settings.ecut && noCutoff.length) {
    refusals.push(`No measured cutoff for ${noCutoff.join(', ')}: dft/ref/materials.py measures cutoffs for ${Object.keys(MEASURED).join(', ')} only. Enter a cutoff to run anyway; the result is marked unvalidated, because no cutoff has been measured for these elements.`);
  }
  const unvalidated = !material && (species.some((el) => !MEASURED[el]) || Object.values(settings).some((s) => s.user));
  // sanity of the numbers themselves
  const v = (k) => (settings[k] ? settings[k].value : null);
  if (v('ecut') !== null && !(v('ecut') > 0)) refusals.push('The cutoff must be positive.');
  if (v('nkmesh') !== null && !(Number.isInteger(v('nkmesh')) && v('nkmesh') >= 1 && v('nkmesh') <= 8)) refusals.push('The k-mesh size must be a whole number from 1 to 8.');
  if (v('smear') !== null && !(v('smear') >= 0)) refusals.push('The smearing width cannot be negative.');
  if (nelec !== null && v('nbands') !== null) {
    if (!Number.isInteger(v('nbands'))) refusals.push('The band count must be a whole number.');
    else if (v('nbands') < Math.ceil(nelec / 2) + 1) refusals.push(`${v('nbands')} bands cannot hold ${nelec} electrons with an empty band above them; use at least ${Math.ceil(nelec / 2) + 1}.`);
  }
  if (nelec !== null && nelec % 2 === 1 && v('smear') === 0) {
    refusals.push(`${nelec} electrons with no smearing: a spin-restricted calculation needs an even count to fill bands exactly. Give a smearing width; the engine will then treat the crystal as a metal.`);
  }

  let deck = null;
  if (!refusals.length) {
    const ns = species.length;
    const z = (...dims) => ({ dims, data: new Array(dims.reduce((a, b) => a * b, 1)).fill(0) });
    const ZION = z(ns), RLOC = z(ns), CLOC = z(ns, GTH_NLOC), NPRJ = z(ns, GTH_NL), RPRJ = z(ns, GTH_NL), HPRJ = z(ns, GTH_NL, GTH_NPMAX, GTH_NPMAX);
    pseudos.forEach((e, s) => {
      ZION.data[s] = e.zion;
      RLOC.data[s] = e.rloc;
      e.cloc.forEach((c, i) => { CLOC.data[s * GTH_NLOC + i] = c; });
      e.shells.forEach((sh, l) => {
        NPRJ.data[s * GTH_NL + l] = sh.nprj;
        RPRJ.data[s * GTH_NL + l] = sh.r;
        for (let a = 0; a < sh.nprj; a++) for (let b = 0; b < sh.nprj; b++) HPRJ.data[((s * GTH_NL + l) * GTH_NPMAX + a) * GTH_NPMAX + b] = sh.h[a][b];
      });
    });
    const a = crystal.alat;
    deck = {
      NSPEC: ns, ZION, RLOC, CLOC, NPRJ, RPRJ, HPRJ,
      ALAT: a,
      A: { dims: [3, 3], data: [0.0, a / 2.0, a / 2.0, a / 2.0, 0.0, a / 2.0, a / 2.0, a / 2.0, 0.0] },
      NAT: crystal.atoms.length,
      ATSPEC: { dims: [crystal.atoms.length], data: crystal.atoms.map((x) => species.indexOf(x.el)) },
      ATFRAC: { dims: [crystal.atoms.length, 3], data: crystal.atoms.flatMap((x) => x.frac) },
      ECUT: v('ecut'), NBANDS: v('nbands'), SMEAR: v('smear'), NKMESH: v('nkmesh'),
      ...FIXED,
      NCORNER: 3, CORNER: { dims: [3, 3], data: CORNER.flat() }, NSEG: TABLES.nseg,
    };
  }
  return { format: parsed.format, formatName: parsed.formatName, title: parsed.title, crystal, species, pseudos, nelec, material, settings, unvalidated, notes, refusals, deck };
}

// ---- the deck as text: make_deck.py's deck(), line for line ------------------

/** make_pseudo.py's output for the species, in order. */
function pseudoText(pseudos) {
  const ns = pseudos.length;
  const L = [
    '# GTH/HGH pseudopotentials, LDA (Pade). GENERATED by dft/pseudo/make_pseudo.py',
    "# from CP2K's GTH_POTENTIALS table -- do not edit by hand.",
    '#   S. Goedecker, M. Teter, J. Hutter, PRB 54, 1703 (1996)',
    '#   C. Hartwigsen, S. Goedecker, J. Hutter, PRB 58, 3641 (1998)',
  ];
  pseudos.forEach((e, s) => L.push(`#   species ${s}: ${e.name}`));
  L.push(`NSPEC = ${ns}.0`,
    `GTH_NLOC = ${GTH_NLOC}.0         #: GTH_NLOC  coefficients C1..C${GTH_NLOC} of the local part`,
    `GTH_NL = ${GTH_NL}.0           #: GTH_NL  angular momentum channels, l = 0 .. ${GTH_NL - 1}`,
    `GTH_NPMAX = ${GTH_NPMAX}.0        #: GTH_NPMAX  at most this many projectors in one channel`,
    'ZION = zeros(NSPEC)', 'RLOC = zeros(NSPEC)', 'CLOC = zeros(NSPEC, GTH_NLOC)',
    'NPRJ = zeros(NSPEC, GTH_NL)       #: NPRJ  projectors per (species, l)',
    'RPRJ = zeros(NSPEC, GTH_NL)       #: RPRJ  projector radius r_l per (species, l)',
    'HPRJ = zeros(NSPEC, GTH_NL, GTH_NPMAX, GTH_NPMAX) #: HPRJ  the coupling matrix h^l_ij per (species, l)');
  pseudos.forEach((e, s) => {
    L.push(`ZION[${s}] = ${e.zion}.0`, `RLOC[${s}] = ${pyRepr(e.rloc)}`);
    e.cloc.forEach((c, i) => { if (c) L.push(`CLOC[${s}, ${i}] = ${pyRepr(c)}`); });
    e.shells.forEach((sh, l) => {
      L.push(`NPRJ[${s}, ${l}] = ${sh.nprj}.0`, `RPRJ[${s}, ${l}] = ${pyRepr(sh.r)}`);
      for (let a = 0; a < sh.nprj; a++) for (let b = 0; b < sh.nprj; b++) if (sh.h[a][b]) L.push(`HPRJ[${s}, ${l}, ${a}, ${b}] = ${pyRepr(sh.h[a][b])}`);
    });
  });
  return L.join('\n');
}

const intF = (x) => `${x}.0`; // make_deck.py's f"{n}.0" for a count

/**
 * The deck as Viv source. Everything from the pseudopotential block down is
 * make_deck.py's text; only the comment header says where it came from.
 */
export function deckText(built, sourceName = '') {
  const { deck: d, pseudos, species, crystal } = built;
  if (!d) throw new Error('deckText: the deck was refused');
  const formula = species.map((el) => { const n = crystal.atoms.filter((a) => a.el === el).length; return n > 1 ? `${el}${n}` : el; }).join('');
  const what = built.material ? TABLES.notes[built.material] : `${formula}, on an fcc lattice.`;
  const L = [`# ${what}`, '#',
    `# GENERATED by the dft-bands engine (engines/dft-bands.viv) from a ${built.formatName}${sourceName ? ` (${sourceName})` : ''},`,
    '# written the way dft/ref/make_deck.py writes a deck. Run it with the iterative engine:',
    '#',
    '#     viv run --use <this deck> --use dft/kmesh.viv --use dft/pw_common.viv --use dft/pwdft_it.viv engines/dft-bands.viv',
    '', pseudoText(pseudos), '',
    '# --- the crystal: an fcc lattice ---------------------------------------------',
    `ALAT = ${pyRepr(d.ALAT)}                          #: ALAT  lattice constant, bohr`,
    'A = [[0.0, ALAT / 2.0, ALAT / 2.0],',
    '     [ALAT / 2.0, 0.0, ALAT / 2.0],',
    '     [ALAT / 2.0, ALAT / 2.0, 0.0]]   #: A  primitive vectors, one per ROW',
    `NAT = ${intF(d.NAT)}`,
    `ATSPEC = [${d.ATSPEC.data.map(intF).join(', ')}]${' '.repeat(8)}#: ATSPEC  species index of each atom`,
    `ATFRAC = [${crystal.atoms.map((a) => `[${a.frac.map(pyRepr).join(', ')}]`).join(',\n          ')}]   #: ATFRAC  positions, fractions of A`,
    '', '# --- the calculation --------------------------------------------------------',
    `ECUT = ${pyRepr(d.ECUT)}                           #: ECUT  plane-wave cutoff, hartree`,
    `NBANDS = ${intF(d.NBANDS)}                         #: NBANDS  bands computed at every k-point`,
    `SMEAR = ${pyRepr(d.SMEAR)}                          #: SMEAR  Fermi-Dirac width, hartree; 0 for an insulator`,
    `NKMESH = ${intF(d.NKMESH)}                         #: NKMESH  the k-mesh is NKMESH^3, shifted (dft/kmesh.viv)`,
    'MIX = 0.6                             #: MIX  the damping in the Anderson mixer (dft/pw_common.viv)',
    'SCF_TOL = 1.0e-9', 'MAXIT = 80.0',
    'NITER = 3.0                           #: NITER  LOBPCG steps per k-point per SCF iteration (iterative engine)',
    'NITER_BANDS = 60.0                    #: NITER_BANDS  at most this many LOBPCG steps per band-path point, from a cold start',
    'EIG_TOL = 1.0e-5                      #: EIG_TOL  residual norm |H psi - eps psi| every band that matters must reach', '',
    '# --- the band path: L -> Gamma -> X ------------------------------------------',
    'NCORNER = 3.0',
    'CORNER = [[0.5, 0.5, 0.5],', '          [0.0, 0.0, 0.0],', '          [0.5, 0.0, 0.5]]',
    `NSEG = ${intF(d.NSEG)}                           #: NSEG  points per path segment; the figure has 2 NSEG + 1`, ''];
  return L.join('\n');
}

// ---- the deck as bytes: what the compiled engine loads ------------------------
//
// The layout of the compiler's checkpoint files (lang/viv/src/lower.rs, the
// "ckpt" runtime), which the compiled engine's loader reads (engines/build.mjs):
//   "VCK2", then per variable: u32 name length, the name, a tag byte;
//   tag 0: one f64; tag 1: i32 rank, i32 dims[8], i32 size, size f64s (row-major).

const ORDER = ['NSPEC', 'ZION', 'RLOC', 'CLOC', 'NPRJ', 'RPRJ', 'HPRJ', 'ALAT', 'A', 'NAT', 'ATSPEC', 'ATFRAC',
  'ECUT', 'NBANDS', 'SMEAR', 'NKMESH', 'MIX', 'SCF_TOL', 'MAXIT', 'NITER', 'NITER_BANDS', 'EIG_TOL', 'NCORNER', 'CORNER', 'NSEG'];
export const DECK_VARIABLES = ORDER;

export function encodeVariables(vars, order = Object.keys(vars)) {
  const enc = new TextEncoder();
  let size = 4;
  for (const k of order) {
    const v = vars[k];
    size += 4 + enc.encode(k).length + 1 + (typeof v === 'number' ? 8 : 4 + 32 + 4 + 8 * v.data.length);
  }
  const buf = new ArrayBuffer(size);
  const dv = new DataView(buf);
  const u8 = new Uint8Array(buf);
  u8.set(enc.encode('VCK2'), 0);
  let o = 4;
  for (const k of order) {
    const v = vars[k];
    const name = enc.encode(k);
    dv.setUint32(o, name.length, true); o += 4;
    u8.set(name, o); o += name.length;
    if (typeof v === 'number') {
      u8[o++] = 0;
      dv.setFloat64(o, v, true); o += 8;
    } else {
      u8[o++] = 1;
      dv.setInt32(o, v.dims.length, true); o += 4;
      for (let i = 0; i < 8; i++) { dv.setInt32(o, v.dims[i] || 0, true); o += 4; }
      dv.setInt32(o, v.data.length, true); o += 4;
      for (const x of v.data) { dv.setFloat64(o, x, true); o += 8; }
    }
  }
  return u8;
}

export function deckBinary(built) {
  if (!built.deck) throw new Error('deckBinary: the deck was refused');
  return encodeVariables(built.deck, ORDER);
}

export { Refusal };
