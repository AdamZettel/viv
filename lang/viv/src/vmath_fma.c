/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* ---- fma(x, y, z): x*y + z rounded once (IEEE 754 fusedMultiplyAdd), in
   every mode. musl 1.2's src/math/fma.c (MIT), with integer arithmetic only,
   so a program needs no libm and no fused instruction; the interpreter and
   the wasm backend use Rust's f64::mul_add, which rounds once too, so the
   engines agree on every input. scikit-learn's compiled gradient boosting
   accumulates `out += scale * value` with the fused instruction on arm64,
   and a port that must give its bits needs this. A runtime fragment of its
   own (lower.rs, "fma"): only a program that calls fma carries it. ---- */
#if defined(__FP_FAST_FMA)
/* The target has the fused instruction (arm64; x86-64 built with FMA3), so
   the builtin compiles to that one instruction, never a call into libm.
   IEEE 754 defines fma's result exactly, so it gives the bits the integer
   code below gives, and costs a cycle instead of tens of nanoseconds. */
static double viv_fma(double x, double y, double z) { return __builtin_fma(x, y, z); }
#else
typedef struct { uint64_t m; int e; int sign; } viv_fma_num;
static uint64_t viv_fma_bits(double x) { union { double f; uint64_t i; } u; u.f = x; return u.i; }
static viv_fma_num viv_fma_normalize(double x)
{
	uint64_t ix = viv_fma_bits(x);
	int e = (int)(ix >> 52);
	int sign = e & 0x800;
	viv_fma_num r;
	e &= 0x7ff;
	if (!e) {
		ix = viv_fma_bits(x * 0x1p63);
		e = (int)(ix >> 52) & 0x7ff;
		e = e ? e - 63 : 0x800;
	}
	ix &= (1ull << 52) - 1;
	ix |= 1ull << 52;
	ix <<= 1;
	e -= 0x3ff + 52 + 1;
	r.m = ix; r.e = e; r.sign = sign;
	return r;
}
static void viv_fma_mul(uint64_t* hi, uint64_t* lo, uint64_t x, uint64_t y)
{
	uint64_t t1, t2, t3;
	uint64_t xlo = (uint32_t)x, xhi = x >> 32;
	uint64_t ylo = (uint32_t)y, yhi = y >> 32;
	t1 = xlo * ylo;
	t2 = xlo * yhi + xhi * ylo;
	t3 = xhi * yhi;
	*lo = t1 + (t2 << 32);
	*hi = t3 + (t2 >> 32) + (t1 > *lo);
}
static int viv_fma_clz64(uint64_t x)
{
	int n = 0;
	while (!(x >> 63)) { x <<= 1; n++; }
	return n;
}
/* x * 2^n, as musl's scalbn (multiplications by powers of two, rounding once at the end) */
static double viv_fma_scalbn(double x, int n)
{
	union { double f; uint64_t i; } u;
	double y = x;
	if (n > 1023) {
		y *= 0x1p1023;
		n -= 1023;
		if (n > 1023) {
			y *= 0x1p1023;
			n -= 1023;
			if (n > 1023) n = 1023;
		}
	} else if (n < -1022) {
		y *= 0x1p-1022 * 0x1p53;
		n += 1022 - 53;
		if (n < -1022) {
			y *= 0x1p-1022 * 0x1p53;
			n += 1022 - 53;
			if (n < -1022) n = -1022;
		}
	}
	u.i = (uint64_t)(0x3ff + n) << 52;
	return y * u.f;
}
static double viv_fma_soft(double x, double y, double z);
/* fma is one correctly rounded operation, so a fused instruction gives the
   same bits as the integer code below on every finite input; the
   instruction is used where every build has it (arm64, or x86-64 built with
   FMA), and an infinity or NaN among the inputs still takes the integer
   code, so those cases are as they were. */
static double viv_fma(double x, double y, double z)
{
#if defined(__aarch64__) || defined(__FMA__)
	if (((viv_fma_bits(x) >> 52 & 0x7ff) != 0x7ff) && ((viv_fma_bits(y) >> 52 & 0x7ff) != 0x7ff) && ((viv_fma_bits(z) >> 52 & 0x7ff) != 0x7ff))
		return __builtin_fma(x, y, z);
#endif
	return viv_fma_soft(x, y, z);
}
static double viv_fma_soft(double x, double y, double z)
{
	viv_fma_num nx = viv_fma_normalize(x), ny = viv_fma_normalize(y), nz = viv_fma_normalize(z);
	const int zeroinfnan = 0x7ff - 0x3ff - 52 - 1;
	uint64_t rhi, rlo, zhi, zlo;
	int e, d, sign, samesign, nonzero;
	int64_t i;
	double r;
	if (nx.e >= zeroinfnan || ny.e >= zeroinfnan)
		return x * y + z;
	if (nz.e >= zeroinfnan) {
		if (nz.e > zeroinfnan) { /* z == 0: x*y rounded, unless it is an exact zero */
			double p = x * y;
			return p == 0 && x != 0 && y != 0 ? p : p + z;
		}
		return z;
	}
	viv_fma_mul(&rhi, &rlo, nx.m, ny.m);
	e = nx.e + ny.e;
	d = nz.e - e;
	if (d > 0) {
		if (d < 64) {
			zlo = nz.m << d;
			zhi = nz.m >> (64 - d);
		} else {
			zlo = 0;
			zhi = nz.m;
			e = nz.e - 64;
			d -= 64;
			if (d == 0) {
			} else if (d < 64) {
				rlo = rhi << (64 - d) | rlo >> d | !!(rlo << (64 - d));
				rhi = rhi >> d;
			} else {
				rlo = 1;
				rhi = 0;
			}
		}
	} else {
		zhi = 0;
		d = -d;
		if (d == 0) {
			zlo = nz.m;
		} else if (d < 64) {
			zlo = nz.m >> d | !!(nz.m << (64 - d));
		} else {
			zlo = 1;
		}
	}
	sign = nx.sign ^ ny.sign;
	samesign = !(sign ^ nz.sign);
	nonzero = 1;
	if (samesign) {
		rlo += zlo;
		rhi += zhi + (rlo < zlo);
	} else {
		uint64_t t = rlo;
		rlo -= zlo;
		rhi = rhi - zhi - (t < rlo);
		if (rhi >> 63) {
			rlo = -rlo;
			rhi = -rhi - !!rlo;
			sign = !sign;
		}
		nonzero = !!rhi;
	}
	if (nonzero) {
		e += 64;
		d = viv_fma_clz64(rhi) - 1;
		rhi = rhi << d | rlo >> (64 - d) | !!(rlo << d);
	} else if (rlo) {
		d = viv_fma_clz64(rlo) - 1;
		if (d < 0)
			rhi = rlo >> 1 | (rlo & 1);
		else
			rhi = rlo << d;
	} else {
		return x * y + z; /* an exact zero, with IEEE's sign */
	}
	e -= d;
	i = (int64_t)rhi; /* in [2^62, 2^63) */
	if (sign)
		i = -i;
	r = (double)i;
	if (e < -1022 - 62) {
		/* subnormal before rounding: round only once */
		if (e == -1022 - 63) {
			double c = 0x1p63;
			if (sign)
				c = -c;
			if (r == c) {
				float fltmin = (float)(0x0.ffffff8p-63 * FLT_MIN * r);
				return DBL_MIN / FLT_MIN * fltmin;
			}
			if (rhi << 53) {
				i = (int64_t)(rhi >> 1 | (rhi & 1) | 1ull << 62);
				if (sign)
					i = -i;
				r = (double)i;
				r = 2 * r - c;
			}
		} else {
			d = 10;
			i = (int64_t)((rhi >> d | !!(rhi << (64 - d))) << d);
			if (sign)
				i = -i;
			r = (double)i;
		}
	}
	return viv_fma_scalbn(r, e);
}
#endif
#define fma viv_fma
