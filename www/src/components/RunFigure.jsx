// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
import { parseOwnOutput, Chart } from './VivCell.jsx';

// The figure a program drew, from the figure records of its build-time run.
// Title, axes and legend are the program's own figure() and plot() arguments;
// this component adds nothing to them.
export default function RunFigure({ text }) {
  const { charts } = parseOwnOutput(text);
  return <>{charts.map((c, i) => <Chart key={i} chart={c} />)}</>;
}
