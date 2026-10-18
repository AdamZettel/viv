// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The site's map: the two sides of the top bar, the pages each one holds, and
// the left-hand section trees of the documentation pages. Base.astro and
// DocPage.astro read this file and nothing else for navigation, and the search
// index (src/pages/search-index.json.js) lists every page named here.
import vipyr from '../data/vipyr.json';
import gallery from '../data/gallery.json';

export type Link = { href: string; label: string; blurb?: string; also?: string[] };
export type Side = { id: 'vipyr' | 'viv'; label: string; home: string; pages: Link[]; more?: Link[]; also?: string[] };

export { GITHUB, REPO_PUBLIC, repoLink, PUBLIC_SITE } from './repo.js';

// vipyr first: the side a Python user starts from.
export const SIDES: Side[] = [
  {
    id: 'vipyr',
    label: 'vipyr',
    home: '/vipyr',
    pages: [
      { href: '/vipyr/install', label: 'Install', blurb: 'pip install vipyr, and the domain extras' },
      { href: '/vipyr/getting-started', label: 'Getting started', blurb: 'A first call, and your own .viv file' },
      { href: '/vipyr', label: 'User guide', blurb: 'The topics and how results are checked' },
      { href: '/vipyr/api', label: 'API reference', blurb: 'Every function, its parameters and its checks', also: ['/vipyr/'] },
      { href: '/vipyr/gallery', label: 'Gallery', blurb: 'Worked examples with their output and figures' },
    ],
  },
  {
    id: 'viv',
    label: 'Viv language',
    home: '/tour',
    pages: [
      { href: '/tour', label: 'Tour', blurb: 'The language, one runnable feature at a time' },
      { href: '/docs', label: 'Docs', blurb: 'Topics by example, each one tested' },
      { href: '/reference', label: 'Reference', blurb: 'Every form, operator and builtin' },
      { href: '/workbench', label: 'Workbench', blurb: 'Edit, run and compile to C in the browser' },
      { href: '/explore', label: 'Explore', blurb: 'Step through a recorded run of a program', also: ['/explore/', '/atlas'] },
      { href: '/posts', label: 'Posts', blurb: 'Short programs that check their own claims', also: ['/p/', '/c/'] },
      { href: '/gallery', label: 'Gallery', blurb: 'Programs by topic, with the figures they drew', also: ['/applications', '/show'] },
    ],
    more: [
      { href: '/install', label: 'Install the compiler' },
      { href: '/ladders', label: 'Ladders', also: ['/ladders/'] },
      { href: '/c/claims', label: 'Claims' },
      { href: '/examples', label: 'Example programs' },
      { href: '/dft', label: 'Plane-wave DFT', also: ['/dft/'] },
      { href: '/pde', label: 'PDE solvers', also: ['/pde/'] },
      { href: '/engines', label: 'Engines', also: ['/engines/'] },
      { href: '/compare', label: 'Python and Viv', also: ['/compare/'] },
      { href: '/internals', label: 'Compiler internals' },
      { href: '/knotkit', label: 'knotkit' },
      { href: '/cli', label: 'Terminal' },
      { href: '/embed', label: 'Embed' },
    ],
  },
];

const norm = (p: string) => p.replace(/\/$/, '') || '/';
export const isUnder = (path: string, href: string) => {
  path = norm(path);
  return path === href || path.startsWith(href.endsWith('/') ? href : href + '/');
};
const linkCovers = (path: string, l: Link) =>
  norm(path) === l.href || (l.also || []).some((a) => (a.endsWith('/') ? norm(path).startsWith(a) : isUnder(path, a)));

/** Which side a path belongs to, if any (the home page belongs to neither). */
export function sideOf(path: string): Side | undefined {
  path = norm(path);
  if (path === '/') return undefined;
  if (isUnder(path, '/vipyr')) return SIDES[0];
  const viv = SIDES[1];
  if ([...viv.pages, ...(viv.more || [])].some((l) => linkCovers(path, l) || isUnder(path, l.href))) return viv;
  return undefined;
}

/** The entry of a side's page list that is current on a path: the most specific match. */
export function currentLink(side: Side, path: string): string | undefined {
  path = norm(path);
  const all = [...side.pages, ...(side.more || [])];
  const exact = all.find((l) => l.href === path);
  if (exact) return exact.href;
  const covering = all.filter((l) => linkCovers(path, l) || isUnder(path, l.href));
  covering.sort((a, b) => b.href.length - a.href.length);
  return covering[0]?.href;
}

// ---- the topics ----------------------------------------------------------------
// Every family of vipyr and every gallery entry sits in one topic, and every
// gallery entry in one of its topic's sections. The ids must match TOPICS in
// python/vipyr/_gallery.py, which checks gallery/index.txt against them.
export const TOPICS = [
  {
    id: 'data',
    title: 'Data',
    pitch: 'Check feature code for future leakage, get the same numbers in training and serving, and compile custom per-group logic to C.',
    families: ['features'],
    sections: [
      { id: 'features', title: 'Dataframes and features' },
    ],
  },
  {
    id: 'stats',
    title: 'Statistics',
    pitch: 'Probability distributions, summary statistics and regression, with the names and results of SciPy, statsmodels and R.',
    families: ['stats'],
    sections: [
      { id: 'dist', title: 'Distributions' },
      { id: 'regression', title: 'Regression and GLMs' },
      { id: 'tests', title: 'Tests and ANOVA' },
      { id: 'bayes', title: 'Bayesian inference' },
      { id: 'timeseries', title: 'Time series' },
      { id: 'survival', title: 'Survival analysis' },
    ],
  },
  {
    id: 'ml',
    title: 'Machine learning',
    pitch: 'A small GPT trained in compiled Viv and returned to Python as named arrays, matching a NumPy program and PyTorch step for step; scikit-learn pipelines compiled to one C file.',
    families: ['gpt'],
    sections: [
      { id: 'nn', title: 'Neural networks and transformers' },
      { id: 'optim', title: 'Optimizers' },
      { id: 'models', title: 'Models to C' },
    ],
  },
  {
    id: 'finance',
    title: 'Finance',
    pitch: 'Option pricing by closed form, binomial trees, Fourier inversion and finite differences, each checked against published values or SciPy.',
    families: ['finance'],
    sections: [
      { id: 'options', title: 'Options and volatility' },
      { id: 'sde', title: 'SDE simulation and Monte Carlo' },
      { id: 'rates', title: 'Rates and term structure' },
      { id: 'risk', title: 'Risk and Greeks' },
    ],
  },
  {
    id: 'science',
    title: 'Science',
    pitch: 'Classic research codes, callable from Python and checked against their originals: PySCF Hartree-Fock, BIGSTICK shell model, plane-wave DFT, Vlasov plasma, Monte Carlo.',
    families: ['scf', 'dft', 'nuclear', 'plasma', 'statmech', 'xc', 'symreg'],
    sections: [
      { id: 'qchem', title: 'Quantum chemistry' },
      { id: 'solids', title: 'Atoms and solids' },
      { id: 'nuclear', title: 'Nuclear' },
      { id: 'plasma', title: 'Plasma' },
      { id: 'statmech', title: 'Statistical mechanics' },
    ],
  },
  {
    id: 'math',
    title: 'Math',
    pitch: 'Differential equations, linear algebra, special functions and randomized sketches written as readable algorithms, checked against SciPy and mpmath.',
    families: ['linalg', 'special', 'randla', 'ode'],
    sections: [
      { id: 'ode', title: 'Differential equations' },
      { id: 'pde', title: 'Partial differential equations' },
      { id: 'linalg', title: 'Linear algebra' },
      { id: 'fitting', title: 'Fitting and optimization' },
      { id: 'accuracy', title: 'Floating-point accuracy' },
      { id: 'autodiff', title: 'Automatic differentiation' },
      { id: 'symbolic', title: 'Computer algebra' },
    ],
  },
  {
    id: 'control',
    title: 'Dynamics and control',
    pitch: 'Riccati and Lyapunov solvers on the Schur form, LQR design and closed-loop simulation; every solution is checked by the residual of its equation.',
    families: ['control'],
    sections: [
      { id: 'linear', title: 'Linear control and stability' },
      { id: 'estimation', title: 'State estimation' },
      { id: 'robotics', title: 'Robots and mechanisms' },
    ],
  },
  {
    id: 'hardware',
    title: 'Hardware',
    pitch: 'Logic built from NAND gates up, simulated gate by gate; a C compiler, an assembler and an RV32IM simulator, each written in Viv.',
    families: ['hardware'],
    sections: [
      { id: 'logic', title: 'Digital logic' },
      { id: 'processors', title: 'Processors' },
      { id: 'memory', title: 'Memory and caches' },
      { id: 'analog', title: 'Physical and analog computing' },
    ],
  },
  {
    id: 'power',
    title: 'Power systems',
    pitch: 'AC and DC power flow on MATPOWER cases, Newton-Raphson and fast-decoupled, with a sparse LU written in Viv, checked against MATPOWER\'s own solution of case9 and against an independent SciPy power flow on every case.',
    families: ['power'],
    sections: [
      { id: 'flow', title: 'Power flow' },
      { id: 'stability', title: 'Transient stability' },
    ],
  },
] as const;
export type TopicId = (typeof TOPICS)[number]['id'];

export const topicOfFamily = (fam: string): TopicId =>
  (TOPICS.find((t) => (t.families as readonly string[]).includes(fam))?.id ?? 'math') as TopicId;

// ---- section trees ------------------------------------------------------------
export type TreeItem = { href: string; label: string; code?: boolean; badge?: string; children?: TreeItem[] };
export type Tree = { title: string; items: TreeItem[] }[];

// A function or family marked "coming" is not built yet: the navigation leaves
// it out (its page still exists for anyone who has the address).
const fams = () => {
  const fns = (Object.values(vipyr.functions) as any[]).filter((x) => x.kind !== 'coming');
  return vipyr.fields
    .map((f: any) => ({ ...f, fns: fns.filter((x) => x.family === f.id).sort((a, b) => a.name.localeCompare(b.name)) }))
    .filter((f: any) => f.fns.length);
};

/** The left tree of every /vipyr page. The current family opens to its functions. */
export function vipyrTree(path: string): Tree {
  const families = fams();
  const open = (id: string) => isUnder(path, `/vipyr/${id}`);
  return [
    {
      title: 'Start',
      items: [
        { href: '/vipyr', label: 'User guide' },
        { href: '/vipyr/install', label: 'Install' },
        { href: '/vipyr/getting-started', label: 'Getting started' },
      ],
    },
    {
      title: 'Gallery',
      items: [
        { href: '/vipyr/gallery', label: 'All examples' },
        ...TOPICS.filter((t) => (gallery as any).entries.some((e: any) => e.topic === t.id)).map((t) => ({
          href: `/vipyr/gallery#${t.id}`,
          label: t.title,
          children: isUnder(path, '/vipyr/gallery')
            ? (gallery as any).entries
                .filter((e: any) => e.topic === t.id)
                .map((e: any) => ({ href: `/vipyr/gallery/${e.slug}`, label: e.title }))
            : undefined,
        })),
      ],
    },
    {
      title: 'API reference',
      items: [
        { href: '/vipyr/api', label: 'Overview' },
        ...TOPICS.flatMap((t) =>
          families
            .filter((f: any) => (t.families as readonly string[]).includes(f.id))
            .map((f: any) => ({
              href: `/vipyr/${f.id}`,
              label: `vp.${f.id}`,
              code: true,
              children: open(f.id)
                ? f.fns.map((x: any) => ({
                    href: `/vipyr/${x.family}/${x.name}`,
                    label: x.name,
                    code: true,
                  }))
                : undefined,
            })),
        ),
        ...families
          .filter((f: any) => !TOPICS.some((t) => (t.families as readonly string[]).includes(f.id)))
          .map((f: any) => ({ href: `/vipyr/${f.id}`, label: `vp.${f.id}`, code: true })),
        { href: '/vipyr/ports', label: 'Ports of classic codes' },
      ],
    },
  ];
}

/** The left tree of the Viv language's documentation pages. */
export function vivTree(docTopics: { slug: string; title: string }[] = []): Tree {
  return [
    {
      title: 'Learn',
      items: [
        { href: '/tour', label: 'Tour' },
        { href: '/docs', label: 'Docs', children: docTopics.map((t) => ({ href: `/docs/${t.slug}`, label: t.title })) },
        { href: '/reference', label: 'Reference' },
        { href: '/compare', label: 'Python and Viv' },
      ],
    },
    {
      title: 'Run',
      items: [
        { href: '/workbench', label: 'Workbench' },
        { href: '/explore', label: 'Explore a run' },
        { href: '/cli', label: 'Terminal' },
        { href: '/install', label: 'Install the compiler' },
      ],
    },
    {
      title: 'Programs',
      items: [
        { href: '/posts', label: 'Posts' },
        { href: '/ladders', label: 'Ladders' },
        { href: '/examples', label: 'Example programs' },
        { href: '/gallery', label: 'Gallery' },
      ],
    },
    {
      title: 'Inside',
      items: [
        { href: '/internals', label: 'Compiler internals' },
        { href: '/knotkit', label: 'knotkit' },
      ],
    },
  ];
}
