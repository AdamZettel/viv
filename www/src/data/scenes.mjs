// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The scenes: one picture per program, drawn from the program's own records
// by a module in src/components/scenes/. The home page shows the first eight
// in order, the walk first; /gallery shows them all as cards; a /show page
// uses its program's scene as the picture at the top.
//
//   id       the scene's key, and the /show slug when it has a page
//   kind     the scene module (src/components/SceneViz.jsx), or 'walk'
//   slug     the staged records and program (public/programs/show/<slug>.*)
//   live     'auto' (programs well under a second) or 'click': a Run button
//            runs it again in the reader's browser; null: no button. Nothing
//            runs on arrival, so scrolling the page never loads viv.wasm
//   title    the physics, plainly
//   line     one sentence: what the picture shows
//   ref      the reference its tests check against, in a few words
//   check    the line under the readout, when "Checked against <ref>" would claim more than the picture shows
//   href     the program's page
export const SCENES = [
  {
    id: 'walk', kind: 'walk', slug: 'skyrme-o16', home: true,
    title: 'Solving for the ground state of oxygen-16',
    line: 'Neutrons and protons settle into shells in the mean field they make together.',
    ref: 'a published Skyrme Hartree-Fock code', href: '/explore/walk/skyrme-o16', cta: 'The whole walk',
  },
  {
    id: 'water-hartree-fock', kind: 'density', slug: 'water-hartree-fock', live: 'click', home: true,
    title: "Water's electron density settles",
    line: 'Each frame is one Roothaan step from the core guess, and the contours are the density in the plane of the molecule.',
    ref: 'PySCF, to 1e-9 hartree', href: '/show/water-hartree-fock', cta: 'Hartree-Fock for water',
  },
  {
    id: 'ising', kind: 'spins', slug: 'ising', live: 'click', home: true,
    title: 'The Ising model at its critical temperature',
    line: 'Spins flip by the Metropolis rule at T_c, where patches of every size form and dissolve.',
    ref: "Onsager and Yang's exact magnetization", href: '/show/ising', cta: 'The Ising model',
    check: "Its page checks the sampled magnetization at T = 1.8, below T_c, against Onsager and Yang's exact value, and the run at every temperature against a reference Metropolis code",
  },
  {
    id: 'kdv', kind: 'waterfall', slug: 'pde-kdv_soliton', live: 'auto', home: true,
    title: 'Two solitons pass through each other',
    line: 'The taller, faster soliton overtakes the shorter one, and both come out with their shapes intact, shifted from where they would have been.',
    ref: "Hirota's exact two-soliton solution", href: '/pde/kdv_soliton', cta: 'The KdV equation',
  },
  {
    id: 'silicon', kind: 'bands', home: true,
    title: 'The band structure of silicon',
    line: 'Plane-wave density functional theory in the local density approximation, along the path from L through Γ to X.',
    ref: 'a NumPy plane-wave code', href: '/dft', cta: 'Density functional theory',
  },
  {
    id: 'landau-damping', kind: 'wave', slug: 'landau-damping', live: 'auto', home: true,
    title: 'Landau damping of a Langmuir wave',
    line: "Electrons moving near the wave's phase speed take its energy, and the field decays at the rate of the dispersion relation's root.",
    ref: 'a 30-digit mpmath root', href: '/show/landau-damping', cta: 'Landau damping',
  },
  {
    id: 'cartpole-lqr', kind: 'cartpole', slug: 'cartpole-lqr', live: 'auto', home: true,
    title: 'A pole balanced on a moving cart',
    line: 'Released 25 degrees from upright, the pole is caught by gains from the Riccati equation.',
    ref: "SciPy's solve_continuous_are", href: '/show/cartpole-lqr', cta: 'The cart-pole',
  },
  {
    id: 'robertson', kind: 'glowplot', slug: 'robertson', live: 'auto', home: true,
    title: "Robertson's stiff chemical kinetics",
    line: 'Three species react at rates nine orders of magnitude apart, followed from 10⁻⁵ to 10¹¹ seconds.',
    ref: 'the Bari test set', href: '/show/robertson', cta: "Robertson's kinetics",
  },
  {
    id: 'oxygen16-skyrme', kind: 'glowplot', slug: 'oxygen16-skyrme', live: 'click',
    title: 'The densities of oxygen-16',
    line: 'Neutron and proton densities from the SLy5 Skyrme functional.',
    ref: 'a published Skyrme Hartree-Fock code', href: '/show/oxygen16-skyrme', cta: 'Oxygen-16',
  },
  {
    id: 'sd-shell', kind: 'glowplot', slug: 'sd-shell', live: 'click',
    title: 'The lowest states of neon-20',
    line: 'Five states of neon-20 in the sd shell with the USDB interaction, found by Lanczos.',
    ref: 'BIGSTICK', href: '/show/sd-shell', cta: 'Neon-20',
  },
  {
    id: 'thurber-fit', kind: 'glowplot', slug: 'thurber-fit', live: 'auto',
    title: 'Electron mobility fitted to NIST data',
    line: "MINPACK's Levenberg-Marquardt fits a rational model to NIST's Thurber measurements.",
    ref: "NIST's certified values", href: '/show/thurber-fit', cta: 'The fit',
  },
];

/** The scene of a /show page, by its slug. */
export const sceneOf = (slug) => SCENES.find((s) => s.slug === slug && s.kind !== 'walk') || null;
