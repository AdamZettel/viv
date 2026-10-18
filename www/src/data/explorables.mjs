// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The programs /explore can open. One entry per program; adding a program is
// adding a line here (scripts/explore-build.mjs builds its bundle from the
// repository, and /explore/<slug> opens it).
//
//   slug    the page's URL: /explore/<slug>
//   file    the program, a repository path; its imported modules, and each
//           module's .intent and .debug files beside it, come along
//   title   what the index and the page call it
//   blurb   one or two sentences: what the program computes
//   posts   feed posts whose card gets an Explain link to this page (slugs
//           without the date)
export const EXPLORABLES = [
  {
    slug: 'minpack-rosenbrock',
    file: 'lib/minpack/story_rosenbrock.viv',
    title: 'Levenberg-Marquardt on Rosenbrock\'s function',
    blurb:
      'MINPACK\'s lmder, ported to Viv, walks Rosenbrock\'s curved valley from (-1.2, 1) to (1, 1) in 16 iterations. Each iteration factors the Jacobian, lmpar finds the step that fits the trust region, and the step is kept or rejected by how well the linear model predicted it; the first trial step is rejected.',
    posts: [],
  },
  {
    slug: 'minpack-thurber',
    file: 'posts/2026-09-25-fitting-electron-mobility-with-minpack.viv',
    title: "Electron mobility fitted to NIST's certified digits",
    blurb:
      'The NIST Thurber problem: a seven-parameter rational model fitted to 37 electron-mobility measurements by lmder, checked against NIST\'s certified values.',
    posts: ['fitting-electron-mobility-with-minpack'],
  },
  {
    slug: 'water-rhf',
    file: 'scf/story_water.viv',
    title: 'Water, restricted Hartree-Fock',
    blurb:
      'The Hartree-Fock energy of water in the 6-31G basis, from the core-Hamiltonian guess, using lib/hf. The self-consistent field loop builds J and K, forms the Fock matrix, extrapolates it with DIIS and solves for new orbitals until the density stops changing.',
    posts: ['hartree-fock-water'],
  },
  {
    slug: 'pyscf-port-water',
    file: 'study/ports/pyscf/viv/water_631g.viv',
    title: 'Water against PySCF, restricted Hartree-Fock',
    blurb:
      'The PySCF port test (study/ports/pyscf) at its parity size: water in 6-31G through lib/hf\'s rhf path, from the core-Hamiltonian guess with DIIS from the second iteration, as PySCF 2.5.0 runs it. The run ends in two checks that are steps of the program: the energy against PySCF\'s -75.98394849810563 within 1e-10, and the orbital energies within 1e-8.',
    posts: [],
  },
  {
    slug: 'skyrme-o16',
    file: 'lib/skyrme/story_o16.viv',
    title: 'Oxygen-16, Skyrme Hartree-Fock',
    blurb:
      'The ground state of oxygen-16 from the SLy5 Skyrme functional on a radial mesh, using lib/skyrme. Each pass computes the fields from the densities, solves each (l, j) block for its lowest orbital by inverse iteration, builds new densities and mixes half of them in, until the energy changes by less than 1e-9 MeV. The post solves on two meshes and extrapolates; this runs its 0.1 fm mesh only.',
    posts: ['skyrme-hartree-fock-of-oxygen-16'],
  },
  {
    slug: 'casida-water',
    file: 'scf/story_casida.viv',
    title: 'Water, Casida excitation energies',
    blurb:
      'The singlet and triplet excitation energies of water in 6-31G, at the post\'s size, using lib/hf and lib/hf/casida. From the RHF ground state it moves the integrals to the orbital basis, builds the Casida matrices A and B, and solves the Tamm-Dancoff problem (A alone) and time-dependent Hartree-Fock (A and B, folded into a symmetric problem).',
    posts: ['casida-tdhf-and-tda-for-water'],
  },
  {
    slug: 'silicon-dense',
    file: 'dft/explore_si.viv',
    title: 'Silicon, plane-wave DFT',
    blurb:
      'Silicon in the dense plane-wave engine (dft/pwdft.viv) at a 5 Ha cutoff on four k-points: the bases, the self-consistent loop from a uniform density, and the bands along L, Gamma, X.',
    posts: ['silicon-bands-from-the-crystal-up'],
  },
  {
    slug: 'tiny-gpt',
    file: 'lib/gpt/story_gpt.viv',
    title: 'A small GPT trained on one repeated line',
    blurb:
      'A GPT with two blocks of two attention heads, 6,560 parameters, trained in your browser on the line "hello world!" repeated: token and position embeddings, RMSNorm, causal self-attention and a ReLU-squared MLP in each block, cross-entropy, the gradient from value_and_grad (the explorer shows its forward pass as the defs it was written in), and 30 steps of Muon for the block matrices and Adam for the rest. The loss falls from log 9 to 0.20, and from the prompt "hello" the model writes the line on.',
    posts: ['a-transformer-trained-by-muon-in-this-tab'],
  },
];

/** The explorable program for a post's slug (with or without its date), or null. */
export function explorableForPost(slug) {
  const s = String(slug || '').replace(/^\d{4}-\d{2}-\d{2}-/, '');
  return EXPLORABLES.find((e) => (e.posts || []).includes(s)) || null;
}
