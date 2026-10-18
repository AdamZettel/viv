// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// The sweep's inputs, deterministic (xorshift64), written as raw f64:
// in.bin, five families of 200 000 (see `fams`); pow.bin, 200 000 (base, exponent) pairs.
import { writeFileSync } from 'node:fs';
let s = 0x9e3779b97f4a7c15n;
const rnd = () => {
  s ^= s << 13n; s &= (1n << 64n) - 1n; s ^= s >> 7n; s ^= s << 17n; s &= (1n << 64n) - 1n;
  return Number(s >> 11n) / 2 ** 53;
};
const fams = {
  unit: () => rnd() * 2 - 1,
  ten: () => (rnd() * 2 - 1) * 10,
  thousand: () => (rnd() * 2 - 1) * 1000,
  wide: () => (rnd() < 0.5 ? -1 : 1) * 10 ** (rnd() * 40 - 20),
  exparg: () => (rnd() * 2 - 1) * 700,
};
const N = 200000;
const all = [];
for (const f of Object.values(fams)) for (let i = 0; i < N; i++) all.push(f());
writeFileSync('in.bin', Buffer.from(new Float64Array(all).buffer));
const p = [];
for (let i = 0; i < N; i++) p.push(rnd() * 10, (rnd() * 2 - 1) * 20);
writeFileSync('pow.bin', Buffer.from(new Float64Array(p).buffer));
