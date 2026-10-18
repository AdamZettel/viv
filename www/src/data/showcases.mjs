// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The showcase pages (/show/<slug>), in reading order. Each one is a program
// that already carries its own page: the `###` header gives the title, the
// equation and the problem prose, the tests name the reference, and the
// figure() calls draw the figure. scripts/show-build.mjs runs each one
// through viv.wasm and refuses a program that fails a test, draws nothing,
// or has no tests.
//
//   slug       the page's address
//   file       the program, from the repository root
//   reference  the named reference its tests check against, in a few words
//   card       one plain sentence for the home page's demo scroll
//   gallery    the /gallery entry it belongs to (its vipyr examples come from there)
//   explore    the recorded run on /explore, when there is one
export const SHOWCASES = [
  {
    slug: 'ising',
    file: 'posts/2026-09-25-metropolis-ising-against-onsager.viv',
    reference: "Onsager and Yang's exact magnetization",
    card: "A 16 by 16 grid of spins, flipped one at a time by the Metropolis rule, lands on Onsager's exact magnetization.",
    gallery: 'ising',
  },
  {
    slug: 'cartpole-lqr',
    file: 'show/cartpole_lqr.viv',
    reference: "SciPy's solve_continuous_are",
    card: 'Gains from the Riccati equation catch a pole released 25 degrees from upright.',
    gallery: 'cartpole-lqr',
  },
  {
    slug: 'landau-damping',
    file: 'posts/2026-09-25-landau-damping-from-the-dispersion-relation.viv',
    reference: 'a 30-digit mpmath root',
    card: 'The damping rate of a plasma wave, from Newton’s method on the kinetic dispersion relation.',
    gallery: 'landau-damping',
  },
  {
    slug: 'oxygen16-skyrme',
    file: 'posts/2026-09-25-skyrme-hartree-fock-of-oxygen-16.viv',
    reference: 'skyrme_rpa, Colò et al. (2013)',
    card: 'The ground state of oxygen-16 from the SLy5 Skyrme functional, against a published code.',
    gallery: 'oxygen16-skyrme',
    explore: 'skyrme-o16',
  },
  {
    slug: 'sd-shell',
    file: 'show/sd_shell.viv',
    reference: 'BIGSTICK',
    card: 'The five lowest states of neon-20 in the nuclear shell model, against BIGSTICK.',
    gallery: 'sd-shell-model',
  },
  {
    slug: 'water-hartree-fock',
    file: 'posts/2026-09-24-hartree-fock-water.viv',
    reference: 'PySCF 2.5.0',
    card: "Water's Hartree-Fock energy along a bond stretch, every point PySCF's to 1e-9 hartree.",
    gallery: 'water-pyscf',
    explore: 'water-rhf',
  },
  {
    slug: 'robertson',
    file: 'show/robertson.viv',
    reference: 'the Bari IVP test set',
    card: 'Three reactions with rates nine orders of magnitude apart, carried to t = 1e11 by Radau IIA in under a thousand steps.',
    gallery: 'ode-solve',
  },
  {
    slug: 'thurber-fit',
    file: 'posts/2026-09-25-fitting-electron-mobility-with-minpack.viv',
    reference: "NIST's certified values",
    card: "MINPACK's Levenberg-Marquardt, in Viv, fits NIST's electron-mobility data to the certified digits.",
    gallery: 'thurber-fit',
    explore: 'minpack-thurber',
  },
];
