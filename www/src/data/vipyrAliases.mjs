// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// vp.linalg's control solvers are the same kernels as vp.control's, under
// SciPy's names. vp.control is their home page; the linalg names are aliases.
export const ALIAS_OF = {
  'linalg.lqr': 'control.lqr',
  'linalg.schur': 'control.schur',
  'linalg.solve_care': 'control.care',
  'linalg.solve_lyapunov': 'control.lyapunov',
};
