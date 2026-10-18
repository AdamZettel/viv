// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The upstream code or data a vipyr function or gallery example follows, and
// its license, as THIRD_PARTY.md and REUSE.toml record them. A function page
// matches its source paths; a gallery page matches the calls in its code.
const CREDITS = [
  {
    src: /^study\/ports\/pyscf\/|^lib\/hf\//,
    code: /vp\.scf\./,
    text: 'Checked against PySCF 2.5.0 (Apache-2.0), which runs as an external oracle; no PySCF code is copied. The Gaussian basis sets come from the Basis Set Exchange.',
  },
  {
    src: /^lib\/(are|schur|lyap)\.viv$/,
    code: /vp\.control\.(lqr|care|lyapunov|schur)\b/,
    text: "The Schur, Riccati and Lyapunov solvers use algorithms after SLICOT; no SLICOT code is copied.",
  },
  {
    src: /^lib\/minpack\//,
    code: /vp\.optimize\./,
    text: "A port of MINPACK (Argonne National Laboratory), under MINPACK's BSD-style copyright notice (1999, University of Chicago), which the port and the Fortran in lib/minpack/ref both carry. This product includes software developed by the University of Chicago, as Operator of Argonne National Laboratory.",
  },
  {
    src: /^lib\/power\//,
    code: /vp\.power\./,
    text: "The methods follow MATPOWER's (BSD-3-Clause). The test cases are MATPOWER's case files, each with its source named in its header.",
  },
  {
    src: /^ports\/liquids\//,
    code: /vp\.liquids\./,
    text: "A port of the examples of Allen and Tildesley's Computer Simulation of Liquids, which are CC0-1.0.",
  },
  {
    src: /^ports\/dftatom\//,
    code: /vp\.dftatom\./,
    text: 'A port of dftatom by Ondřej Čertík, Jiří Vackář and John Pask (MIT).',
  },
  {
    src: /^lib\/nuclear\/shell\//,
    code: /vp\.nuclear\.shell_model\b/,
    text: 'The shell model follows BIGSTICK 8.0.0 (MIT), whose notice heads lib/nuclear/shell/shell.viv. The USDB interaction is from Brown and Richter, Physical Review C 74, 034315 (2006).',
  },
  {
    src: /^dft\//,
    code: /vp\.dft\./,
    text: "The GTH pseudopotential parameters are CP2K's GTH_POTENTIALS table as PySCF ships it (Goedecker, Teter and Hutter 1996; Hartwigsen, Goedecker and Hutter 1998); only the numbers are used, no CP2K or PySCF code.",
  },
  {
    src: /^lib\/gguf\//,
    code: /vp\.gguf\./,
    text: "The GGUF reader is a rewrite, from the GGUF specification (docs/gguf.md), of the reader in ggml's src/gguf.cpp and gguf-py's gguf_reader.py; the dequantizers are a rewrite of dequantize_row_* in ggml's src/ggml-quants.c, with the block structs of src/ggml-common.h (ggml at commit 3612c0af93f2, MIT, The ggml authors), and are tested number for number against gguf-py 0.19.0. The test model is llama2.c's stories260K (MIT).",
  },
  {
    src: /^lib\/cc\/|^lib\/rv32\//,
    code: /lib\/cc|rv32/,
    text: 'The C compiler follows chibicc by Rui Ueyama (MIT); the simulator is tested with the riscv-tests suites (BSD-3-Clause).',
  },
];

export const creditsForSources = (sources) =>
  CREDITS.filter((c) => sources.some((s) => c.src.test(s))).map((c) => c.text);

export const creditsForCode = (code) => CREDITS.filter((c) => c.code.test(code)).map((c) => c.text);
