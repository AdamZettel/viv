// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// A Quantum ESPRESSO pw.x input: Fortran namelists (&CONTROL, &SYSTEM,
// &ELECTRONS, ...) and then cards (ATOMIC_SPECIES, ATOMIC_POSITIONS,
// CELL_PARAMETERS, K_POINTS).
//
// Besides the crystal, a pw.x input carries settings, and those that mean the
// same thing here are kept: the cutoff (ecutwfc, in rydberg), the band count
// (nbnd), smearing (occupations, degauss) and the k-mesh size. What does not
// carry over is said, not dropped: the pseudopotential files are QE's and the
// engine uses GTH ones, and a magnetic or spin-orbit input is refused.
import { BOHR_ANGSTROM, Refusal, toFrac } from '../crystal.js';
import { elementOf } from '../elements.js';

/** A Fortran value: 'text', .true., 1.0d-8, 12 */
function fortranValue(raw) {
  const s = raw.trim();
  if (/^'.*'$|^".*"$/.test(s)) return s.slice(1, -1);
  if (/^\.?t(rue)?\.?$/i.test(s)) return true;
  if (/^\.?f(alse)?\.?$/i.test(s)) return false;
  const n = Number(s.replace(/[dD]/, 'e'));
  return Number.isFinite(n) ? n : s;
}

/** Namelists as { system: { ibrav: 2, 'celldm(1)': 10.26, ... }, ... } and the card lines after them. */
function split(text) {
  const lines = text.replace(/\r/g, '').split('\n').map((l) => l.replace(/[!#].*$/, ''));
  const nml = {};
  const cards = [];
  let cur = null;
  for (const line of lines) {
    const t = line.trim();
    if (!t) continue;
    if (t.startsWith('&')) { cur = t.slice(1).toLowerCase(); nml[cur] = nml[cur] || {}; continue; }
    if (cur && t === '/') { cur = null; continue; }
    if (cur) {
      // key = value pairs, several to a line, separated by commas outside quotes
      const re = /([A-Za-z_][A-Za-z0-9_]*(?:\(\s*\d+\s*(?:,\s*\d+\s*)?\))?)\s*=\s*('[^']*'|"[^"]*"|[^,\s]+)/g;
      let m;
      while ((m = re.exec(t))) nml[cur][m[1].toLowerCase().replace(/\s+/g, '')] = fortranValue(m[2]);
      continue;
    }
    cards.push(t);
  }
  return { nml, cards };
}

/** The lines of one card, and the option in its header (the word in braces or after the name). */
function card(cards, name) {
  const i = cards.findIndex((l) => l.toUpperCase().startsWith(name));
  if (i < 0) return null;
  const opt = (cards[i].slice(name.length).match(/[A-Za-z_]+/) || [''])[0].toLowerCase();
  const body = [];
  for (let j = i + 1; j < cards.length; j++) {
    if (/^(ATOMIC_SPECIES|ATOMIC_POSITIONS|K_POINTS|CELL_PARAMETERS|OCCUPATIONS|CONSTRAINTS|ATOMIC_FORCES|ADDITIONAL_K_POINTS|SOLVENTS|HUBBARD)\b/i.test(cards[j])) break;
    body.push(cards[j]);
  }
  return { opt, body };
}

/** QE's Bravais lattices that are cubic, from celldm(1) = a in bohr (Modules/latgen.f90). */
function ibravCell(ibrav, a) {
  if (ibrav === 1) return [[a, 0, 0], [0, a, 0], [0, 0, a]];
  if (ibrav === 2) return [[-a / 2, 0, a / 2], [0, a / 2, a / 2], [-a / 2, a / 2, 0]];
  if (ibrav === 3) return [[a / 2, a / 2, a / 2], [-a / 2, a / 2, a / 2], [-a / 2, -a / 2, a / 2]];
  return null;
}

export function parseQe(text) {
  const { nml, cards } = split(text);
  const sys = nml.system;
  if (!sys) throw new Refusal('pw.x input: no &SYSTEM namelist.');
  const notes = [];
  const settings = {};

  // ---- what the engine cannot do: say so before anything else
  const nspin = sys.nspin || 1;
  const magnetized = Object.entries(sys).some(([k, v]) => k.startsWith('starting_magnetization') && Number(v) !== 0);
  if (nspin === 2 || sys.noncolin === true || sys.lspinorb === true || magnetized) {
    throw new Refusal('pw.x input: this is a spin-polarized (nspin = 2, noncolin, lspinorb or starting_magnetization) calculation. The engine is spin-restricted LDA and cannot do it.');
  }
  if (sys.tot_charge && Number(sys.tot_charge) !== 0) throw new Refusal(`pw.x input: tot_charge = ${sys.tot_charge}. The engine computes neutral cells only.`);
  if (sys.lda_plus_u === true || cards.some((l) => /^HUBBARD/i.test(l))) throw new Refusal('pw.x input: the file asks for DFT+U (a Hubbard U); the engine does plain LDA and cannot run it.');
  if (sys.input_dft && !/^(pz|lda|sla\+pz)$/i.test(String(sys.input_dft))) {
    notes.push(`input_dft = '${sys.input_dft}' is overridden: the engine is LDA (the Teter-93 Pade fit), the functional its GTH pseudopotentials were made with.`);
  }

  // ---- the cell, in bohr
  const ibrav = Number(sys.ibrav);
  let alat = null;
  if (sys['celldm(1)']) alat = Number(sys['celldm(1)']);
  else if (sys.a) alat = Number(sys.a) / BOHR_ANGSTROM;
  let L;
  if (ibrav === 0) {
    const cp = card(cards, 'CELL_PARAMETERS');
    if (!cp) throw new Refusal('pw.x input: ibrav = 0 but there is no CELL_PARAMETERS card.');
    const v = cp.body.slice(0, 3).map((l) => l.split(/\s+/).map(Number));
    let unit = cp.opt || (alat ? 'alat' : 'bohr');
    const f = unit === 'angstrom' ? 1 / BOHR_ANGSTROM : unit === 'bohr' ? 1 : unit === 'alat' ? alat : NaN;
    if (!Number.isFinite(f)) throw new Refusal(unit === 'alat' ? 'pw.x input: CELL_PARAMETERS is in alat units, but the file gives no celldm(1) or A to scale it.' : `pw.x input: CELL_PARAMETERS is in "${unit}"; this page reads alat, bohr and angstrom.`);
    L = v.map((r) => r.map((x) => x * f));
    if (!alat) alat = Math.hypot(...L[0]);
  } else {
    if (!alat) throw new Refusal('pw.x input: no lattice parameter (celldm(1) or A).');
    L = ibravCell(ibrav, alat);
    if (!L) throw new Refusal(`pw.x input: ibrav = ${sys.ibrav}. This page reads ibrav = 0 (CELL_PARAMETERS), 1, 2 and 3; the engine itself needs an fcc lattice.`);
  }

  // ---- species and atoms
  const spc = card(cards, 'ATOMIC_SPECIES');
  if (!spc) throw new Refusal('pw.x input: no ATOMIC_SPECIES card.');
  const labels = new Map();
  const pseudoFiles = [];
  for (const l of spc.body) {
    const [label, , file] = l.split(/\s+/);
    const el = elementOf(label);
    if (!el) throw new Refusal(`pw.x input: species "${label}" does not name an element.`);
    labels.set(label, el);
    if (file) pseudoFiles.push(`${label}: ${file}`);
  }
  if (pseudoFiles.length) notes.push(`The pseudopotential files named (${pseudoFiles.join(', ')}) are Quantum ESPRESSO's; the engine uses its own GTH-PADE ones, listed below.`);
  const pos = card(cards, 'ATOMIC_POSITIONS');
  if (!pos) throw new Refusal('pw.x input: no ATOMIC_POSITIONS card.');
  const unit = pos.opt || 'alat';
  const nat = Number(sys.nat);
  const atoms = [];
  for (const l of pos.body.slice(0, nat)) {
    const t = l.split(/\s+/);
    const el = labels.get(t[0]) || elementOf(t[0]);
    if (!el) throw new Refusal(`pw.x input: atom label "${t[0]}" is not in ATOMIC_SPECIES.`);
    const p = t.slice(1, 4).map(Number);
    if (p.length < 3 || p.some((x) => !Number.isFinite(x))) throw new Refusal(`pw.x input: cannot read the position "${l}".`);
    let frac;
    if (unit === 'crystal') frac = p;
    else if (unit === 'alat') frac = toFrac(p.map((x) => x * alat), L);
    else if (unit === 'bohr') frac = toFrac(p, L);
    else if (unit === 'angstrom') frac = toFrac(p.map((x) => x / BOHR_ANGSTROM), L);
    else throw new Refusal(`pw.x input: ATOMIC_POSITIONS in "${unit}" is not read here (crystal, alat, bohr and angstrom are).`);
    atoms.push({ el, frac, label: t[0] });
  }
  if (Number.isFinite(nat) && atoms.length !== nat) throw new Refusal(`pw.x input: nat = ${nat} but ${atoms.length} positions were found.`);

  // ---- settings that carry over (QE's energies are in rydberg: half a hartree)
  if (sys.ecutwfc) settings.ecut = { value: Number(sys.ecutwfc) / 2, from: `ecutwfc = ${sys.ecutwfc} Ry in the input` };
  if (sys.nbnd) settings.nbands = { value: Number(sys.nbnd), from: `nbnd = ${sys.nbnd} in the input` };
  const occ = String(sys.occupations || 'fixed').toLowerCase();
  if (occ === 'smearing') {
    const deg = Number(sys.degauss || 0);
    if (!(deg > 0)) throw new Refusal('pw.x input: occupations = \'smearing\' needs a degauss width, and the file gives none.');
    settings.smear = { value: deg / 2, from: `degauss = ${sys.degauss} Ry in the input` };
    const kind = String(sys.smearing || 'gaussian').toLowerCase();
    if (!/^(fd|f-d|fermi-dirac)$/.test(kind)) {
      notes.push(`smearing = '${kind}' becomes Fermi-Dirac smearing of the same width: the only kind the engine has.`);
    }
  } else if (occ === 'fixed') {
    settings.smear = { value: 0, from: 'occupations = \'fixed\' in the input' };
  } else {
    throw new Refusal(`pw.x input: occupations = '${occ}'; the engine handles only 'fixed' and 'smearing'.`);
  }
  const kp = card(cards, 'K_POINTS');
  if (kp && kp.opt === 'automatic' && kp.body.length) {
    const n = kp.body[0].split(/\s+/).map(Number);
    const m = Math.max(n[0], n[1], n[2]);
    settings.nkmesh = { value: m, from: `K_POINTS automatic ${n.slice(0, 3).join(' ')} in the input` };
    if (!(n[0] === n[1] && n[1] === n[2])) notes.push(`The k-mesh ${n.slice(0, 3).join('x')} is not the same in every direction; the engine's fcc mesh is ${m}x${m}x${m} (the largest).`);
    notes.push(`K_POINTS automatic becomes the engine's own mesh: the four-shift fcc mesh (dft/kmesh.viv) of size ${m}, whatever shifts the input asked for.`);
  } else if (kp) {
    notes.push(`K_POINTS ${kp.opt || 'tpiba'} is not a mesh the engine reads; it uses its default mesh, and its band path is L - Gamma - X.`);
  }
  const calc = String((nml.control && nml.control.calculation) || 'scf').toLowerCase();
  if (calc !== 'scf' && calc !== 'bands' && calc !== 'nscf') notes.push(`calculation = '${calc}' is ignored: the engine runs an SCF with the atoms held fixed, then computes bands along the path.`);
  const title = (nml.control && nml.control.prefix) || '';
  return { format: 'qe-pw', formatName: 'Quantum ESPRESSO pw.x input', title: String(title), lattice: L, atoms, settings, notes };
}

export const looksLikeQe = (text) => /&system/i.test(text) && /ATOMIC_POSITIONS/i.test(text);
