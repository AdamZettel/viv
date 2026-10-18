// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
// Count, per function, the inputs where V8's Math, Viv's libm in viv.wasm
// (its runtime's vrt_* exports) and the platform's C libm (c_<fn>.bin from
// ref.c) differ in any bit, with the worst difference in units in the last
// place; and, last column, where viv.wasm and Viv's libm compiled natively
// from vmath.c (v_<fn>.bin) differ -- the engines' agreement, which must be 0.
import { readFileSync } from 'node:fs';
const rd = (f) => new Float64Array(readFileSync(f).buffer.slice(0));
const x = rd('in.bin');
const p = rd('pow.bin');
const { instance } = await WebAssembly.instantiate(readFileSync(process.env.VIV_WASM), {});
const ex = instance.exports;
const bits = new Float64Array(2);
const u = new BigUint64Array(bits.buffer);
const same = (a, b) => {
  if (Number.isNaN(a) && Number.isNaN(b)) return true;
  bits[0] = a; bits[1] = b;
  return u[0] === u[1];
};
const ulps = (a, b) => { bits[0] = a; bits[1] = b; return u[0] > u[1] ? u[0] - u[1] : u[1] - u[0]; };
const fns = {
  sin: [Math.sin, ex.vrt_sin], cos: [Math.cos, ex.vrt_cos], tan: [Math.tan, ex.vrt_tan],
  exp: [Math.exp, ex.vrt_exp], log: [Math.log, ex.vrt_ln], atan: [Math.atan, ex.vrt_atan],
  asin: [Math.asin, ex.vrt_asin], acos: [Math.acos, ex.vrt_acos], sinh: [Math.sinh, ex.vrt_sinh],
  cosh: [Math.cosh, ex.vrt_cosh], tanh: [Math.tanh, ex.vrt_tanh],
};
const row = (name, n, js, rs, c, v) => {
  let dj = 0, dr = 0, djr = 0, dv = 0, mj = 0n, mr = 0n;
  for (let i = 0; i < n; i++) {
    const a = js(i), b = rs(i);
    if (!same(a, c[i])) { dj++; const d = ulps(a, c[i]); if (d > mj) mj = d; }
    if (!same(b, c[i])) { dr++; const d = ulps(b, c[i]); if (d > mr) mr = d; }
    if (!same(a, b)) djr++;
    if (!same(b, v[i])) dv++;
  }
  if (dv) failed = true;
  console.log(`${name.padEnd(6)} ${String(dj).padStart(8)} (${mj} ulp) ${String(dr).padStart(8)} (${mr} ulp) ${String(djr).padStart(8)} ${String(dv).padStart(10)}   of ${n}`);
};
let failed = false;
console.log('fn      V8 vs C libm      Viv vs C libm     V8 vs Viv  Viv wasm vs Viv C');
for (const [name, [js, rs]] of Object.entries(fns)) row(name, x.length, (i) => js(x[i]), (i) => rs(x[i]), rd(`c_${name}.bin`), rd(`v_${name}.bin`));
row('pow', p.length / 2, (i) => Math.pow(p[2 * i], p[2 * i + 1]), (i) => ex.vrt_pow(p[2 * i], p[2 * i + 1]), rd('c_pow.bin'), rd('v_pow.bin'));
if (failed) { console.log('FAIL: Viv\'s libm differs between wasm and native C'); process.exitCode = 1; }
