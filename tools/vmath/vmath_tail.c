/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* ---- Viv's own: exp, correctly rounded (vmath/exp_cr.rs, line for line) ----
   x = k ln2/128 + r and e^x = 2^(k/128) e^r, 2^(j/128) from a table; a fast
   path good to 2^-66 that keeps the last sum's rounding error, and a
   double-double slow path when the result is that close to a midpoint.
   Below |x| = 2^-54 it is 1 + x; at or below -708 the slow path rounded
   once to the grid of 2^-1074 (viv_exp_tiny); past -746 and 709.79 (0 and
   inf) it is musl's viv_exp.
   The constants come from tools/vmath/exp_table.py. A program's `exp` is
   this function (the #define at the end), and so is the exp inside erf,
   gamma and the complex functions below; musl's viv_exp stays inside the
   musl code (cosh, sinh through __expo2) and past the two range ends. */
#define VIV_EXP_L1 0x3f762e42fefa0000ULL
#define VIV_EXP_L2 0x3d0cf79abc9e3b3aULL
#define VIV_EXP_L3 0xb9aff0342542fc33ULL
#define VIV_EXP_INVL 0x40671547652b82feULL
static const uint64_t viv_exp_T[256] = {
	0x3ff0000000000000ULL, 0x0000000000000000ULL, 0x3ff0163da9fb3335ULL, 0x3c9b61299ab8cdb7ULL,
	0x3ff02c9a3e778061ULL, 0xbc719083535b085dULL, 0x3ff04315e86e7f85ULL, 0xbc90a31c1977c96eULL,
	0x3ff059b0d3158574ULL, 0x3c8d73e2a475b465ULL, 0x3ff0706b29ddf6deULL, 0xbc8c91dfe2b13c27ULL,
	0x3ff0874518759bc8ULL, 0x3c6186be4bb284ffULL, 0x3ff09e3ecac6f383ULL, 0x3c91487818316136ULL,
	0x3ff0b5586cf9890fULL, 0x3c98a62e4adc610bULL, 0x3ff0cc922b7247f7ULL, 0x3c901edc16e24f71ULL,
	0x3ff0e3ec32d3d1a2ULL, 0x3c403a1727c57b53ULL, 0x3ff0fb66affed31bULL, 0xbc6b9bedc44ebd7bULL,
	0x3ff11301d0125b51ULL, 0xbc96c51039449b3aULL, 0x3ff12abdc06c31ccULL, 0xbc51b514b36ca5c7ULL,
	0x3ff1429aaea92de0ULL, 0xbc932fbf9af1369eULL, 0x3ff15a98c8a58e51ULL, 0x3c82406ab9eeab0aULL,
	0x3ff172b83c7d517bULL, 0xbc819041b9d78a76ULL, 0x3ff18af9388c8deaULL, 0xbc911023d1970f6cULL,
	0x3ff1a35beb6fcb75ULL, 0x3c8e5b4c7b4968e4ULL, 0x3ff1bbe084045cd4ULL, 0xbc995386352ef607ULL,
	0x3ff1d4873168b9aaULL, 0x3c9e016e00a2643cULL, 0x3ff1ed5022fcd91dULL, 0xbc91df98027bb78cULL,
	0x3ff2063b88628cd6ULL, 0x3c8dc775814a8495ULL, 0x3ff21f49917ddc96ULL, 0x3c82a97e9494a5eeULL,
	0x3ff2387a6e756238ULL, 0x3c99b07eb6c70573ULL, 0x3ff251ce4fb2a63fULL, 0x3c8ac155bef4f4a4ULL,
	0x3ff26b4565e27cddULL, 0x3c82bd339940e9d9ULL, 0x3ff284dfe1f56381ULL, 0xbc9a4c3a8c3f0d7eULL,
	0x3ff29e9df51fdee1ULL, 0x3c8612e8afad1255ULL, 0x3ff2b87fd0dad990ULL, 0xbc410adcd6381aa4ULL,
	0x3ff2d285a6e4030bULL, 0x3c90024754db41d5ULL, 0x3ff2ecafa93e2f56ULL, 0x3c71ca0f45d52383ULL,
	0x3ff306fe0a31b715ULL, 0x3c86f46ad23182e4ULL, 0x3ff32170fc4cd831ULL, 0x3c8a9ce78e18047cULL,
	0x3ff33c08b26416ffULL, 0x3c932721843659a6ULL, 0x3ff356c55f929ff1ULL, 0xbc8b5cee5c4e4628ULL,
	0x3ff371a7373aa9cbULL, 0xbc963aeabf42eae2ULL, 0x3ff38cae6d05d866ULL, 0xbc9e958d3c9904bdULL,
	0x3ff3a7db34e59ff7ULL, 0xbc75e436d661f5e3ULL, 0x3ff3c32dc313a8e5ULL, 0xbc9efff8375d29c3ULL,
	0x3ff3dea64c123422ULL, 0x3c8ada0911f09ebcULL, 0x3ff3fa4504ac801cULL, 0xbc97d023f956f9f3ULL,
	0x3ff4160a21f72e2aULL, 0xbc5ef3691c309278ULL, 0x3ff431f5d950a897ULL, 0xbc81c7dde35f7999ULL,
	0x3ff44e086061892dULL, 0x3c489b7a04ef80d0ULL, 0x3ff46a41ed1d0057ULL, 0x3c9c944bd1648a76ULL,
	0x3ff486a2b5c13cd0ULL, 0x3c73c1a3b69062f0ULL, 0x3ff4a32af0d7d3deULL, 0x3c99cb62f3d1be56ULL,
	0x3ff4bfdad5362a27ULL, 0x3c7d4397afec42e2ULL, 0x3ff4dcb299fddd0dULL, 0x3c98ecdbbc6a7833ULL,
	0x3ff4f9b2769d2ca7ULL, 0xbc94b309d25957e3ULL, 0x3ff516daa2cf6642ULL, 0xbc8f768569bd93efULL,
	0x3ff5342b569d4f82ULL, 0xbc807abe1db13cadULL, 0x3ff551a4ca5d920fULL, 0xbc8d689cefede59bULL,
	0x3ff56f4736b527daULL, 0x3c99bb2c011d93adULL, 0x3ff58d12d497c7fdULL, 0x3c8295e15b9a1de8ULL,
	0x3ff5ab07dd485429ULL, 0x3c96324c054647adULL, 0x3ff5c9268a5946b7ULL, 0x3c3c4b1b816986a2ULL,
	0x3ff5e76f15ad2148ULL, 0x3c9ba6f93080e65eULL, 0x3ff605e1b976dc09ULL, 0xbc93e2429b56de47ULL,
	0x3ff6247eb03a5585ULL, 0xbc9383c17e40b497ULL, 0x3ff6434634ccc320ULL, 0xbc8c483c759d8933ULL,
	0x3ff6623882552225ULL, 0xbc9bb60987591c34ULL, 0x3ff68155d44ca973ULL, 0x3c6038ae44f73e65ULL,
	0x3ff6a09e667f3bcdULL, 0xbc9bdd3413b26456ULL, 0x3ff6c012750bdabfULL, 0xbc72895667ff0b0dULL,
	0x3ff6dfb23c651a2fULL, 0xbc6bbe3a683c88abULL, 0x3ff6ff7df9519484ULL, 0xbc883c0f25860ef6ULL,
	0x3ff71f75e8ec5f74ULL, 0xbc816e4786887a99ULL, 0x3ff73f9a48a58174ULL, 0xbc90a8d96c65d53cULL,
	0x3ff75feb564267c9ULL, 0xbc90245957316dd3ULL, 0x3ff780694fde5d3fULL, 0x3c9866b80a02162dULL,
	0x3ff7a11473eb0187ULL, 0xbc841577ee04992fULL, 0x3ff7c1ed0130c132ULL, 0x3c9f124cd1164dd6ULL,
	0x3ff7e2f336cf4e62ULL, 0x3c705d02ba15797eULL, 0x3ff80427543e1a12ULL, 0xbc927c86626d972bULL,
	0x3ff82589994cce13ULL, 0xbc9d4c1dd41532d8ULL, 0x3ff8471a4623c7adULL, 0xbc88d684a341cdfbULL,
	0x3ff868d99b4492edULL, 0xbc9fc6f89bd4f6baULL, 0x3ff88ac7d98a6699ULL, 0x3c9994c2f37cb53aULL,
	0x3ff8ace5422aa0dbULL, 0x3c96e9f156864b27ULL, 0x3ff8cf3216b5448cULL, 0xbc70d55e32e9e3aaULL,
	0x3ff8f1ae99157736ULL, 0x3c85cc13a2e3976cULL, 0x3ff9145b0b91ffc6ULL, 0xbc9dd6792e582524ULL,
	0x3ff93737b0cdc5e5ULL, 0xbc675fc781b57ebcULL, 0x3ff95a44cbc8520fULL, 0xbc764b7c96a5f039ULL,
	0x3ff97d829fde4e50ULL, 0xbc9d185b7c1b85d1ULL, 0x3ff9a0f170ca07baULL, 0xbc9173bd91cee632ULL,
	0x3ff9c49182a3f090ULL, 0x3c7c7c46b071f2beULL, 0x3ff9e86319e32323ULL, 0x3c7824ca78e64c6eULL,
	0x3ffa0c667b5de565ULL, 0xbc9359495d1cd533ULL, 0x3ffa309bec4a2d33ULL, 0x3c96305c7ddc36abULL,
	0x3ffa5503b23e255dULL, 0xbc9d2f6edb8d41e1ULL, 0x3ffa799e1330b358ULL, 0x3c9bcb7ecac563c7ULL,
	0x3ffa9e6b5579fdbfULL, 0x3c90fac90ef7fd31ULL, 0x3ffac36bbfd3f37aULL, 0xbc8f9234cae76cd0ULL,
	0x3ffae89f995ad3adULL, 0x3c97a1cd345dcc81ULL, 0x3ffb0e07298db666ULL, 0xbc9bdef54c80e425ULL,
	0x3ffb33a2b84f15fbULL, 0xbc62805e3084d708ULL, 0x3ffb59728de5593aULL, 0xbc9c71dfbbba6de3ULL,
	0x3ffb7f76f2fb5e47ULL, 0xbc75584f7e54ac3bULL, 0x3ffba5b030a1064aULL, 0xbc9efcd30e54292eULL,
	0x3ffbcc1e904bc1d2ULL, 0x3c823dd07a2d9e84ULL, 0x3ffbf2c25bd71e09ULL, 0xbc9efdca3f6b9c73ULL,
	0x3ffc199bdd85529cULL, 0x3c811065895048ddULL, 0x3ffc40ab5fffd07aULL, 0x3c9b4537e083c60aULL,
	0x3ffc67f12e57d14bULL, 0x3c92884dff483cadULL, 0x3ffc8f6d9406e7b5ULL, 0x3c71acbc48805c44ULL,
	0x3ffcb720dcef9069ULL, 0x3c7503cbd1e949dbULL, 0x3ffcdf0b555dc3faULL, 0xbc8dd83b53829d72ULL,
	0x3ffd072d4a07897cULL, 0xbc9cbc3743797a9cULL, 0x3ffd2f87080d89f2ULL, 0xbc9d487b719d8578ULL,
	0x3ffd5818dcfba487ULL, 0x3c82ed02d75b3707ULL, 0x3ffd80e316c98398ULL, 0xbc911ec18beddfe8ULL,
	0x3ffda9e603db3285ULL, 0x3c9c2300696db532ULL, 0x3ffdd321f301b460ULL, 0x3c92da5778f018c3ULL,
	0x3ffdfc97337b9b5fULL, 0xbc91a5cd4f184b5cULL, 0x3ffe264614f5a129ULL, 0xbc97b627817a1496ULL,
	0x3ffe502ee78b3ff6ULL, 0x3c839e8980a9cc8fULL, 0x3ffe7a51fbc74c83ULL, 0x3c92d522ca0c8de2ULL,
	0x3ffea4afa2a490daULL, 0xbc9e9c23179c2893ULL, 0x3ffecf482d8e67f1ULL, 0xbc9c93f3b411ad8cULL,
	0x3ffefa1bee615a27ULL, 0x3c9dc7f486a4b6b0ULL, 0x3fff252b376bba97ULL, 0x3c93a1a5bf0d8e43ULL,
	0x3fff50765b6e4540ULL, 0x3c99d3e12dd8a18bULL, 0x3fff7bfdad9cbe14ULL, 0xbc9dbb12d006350aULL,
	0x3fffa7c1819e90d8ULL, 0x3c874853f3a5931eULL, 0x3fffd3c22b8f71f1ULL, 0x3c62eb74966579e7ULL,
};
static const uint64_t viv_exp_C[26] = {
	0x3ff0000000000000ULL, 0x0000000000000000ULL, 0x3ff0000000000000ULL, 0x0000000000000000ULL,
	0x3fe0000000000000ULL, 0x0000000000000000ULL, 0x3fc5555555555555ULL, 0x3c65555555555555ULL,
	0x3fa5555555555555ULL, 0x3c45555555555555ULL, 0x3f81111111111111ULL, 0x3c01111111111111ULL,
	0x3f56c16c16c16c17ULL, 0xbbef49f49f49f49fULL, 0x3f2a01a01a01a01aULL, 0x3b6a01a01a01a01aULL,
	0x3efa01a01a01a01aULL, 0x3b3a01a01a01a01aULL, 0x3ec71de3a556c734ULL, 0xbb6c154f8ddc6c00ULL,
	0x3e927e4fb7789f5cULL, 0x3b3cbbc05b4fa99aULL, 0x3e5ae64567f544e4ULL, 0xbafc062e06d1f209ULL,
	0x3e21eed8eff8d898ULL, 0xbac2aec959e14c06ULL,
};
static double viv_exp_bits(uint64_t b) { union {double f; uint64_t i;} u; u.i = b; return u.f; }
static uint64_t viv_exp_tobits(double x) { union {double f; uint64_t i;} u; u.f = x; return u.i; }
static double viv_exp_t(int i) { return viv_exp_bits(viv_exp_T[i]); }
static double viv_exp_c(int i) { return viv_exp_bits(viv_exp_C[i]); }
static void viv_exp_two_sum(double a, double b, double* s, double* e)
{
	double t = a + b;
	double bb = t - a;
	*s = t;
	*e = (a - (t - bb)) + (b - bb);
}
static void viv_exp_fast_two_sum(double a, double b, double* s, double* e)
{
	double t = a + b;
	*s = t;
	*e = b - (t - a);
}
static void viv_exp_split(double a, double* hi, double* lo)
{
	double t = 134217729.0 * a;
	*hi = t - (t - a);
	*lo = a - *hi;
}
static void viv_exp_two_prod(double a, double b, double* p, double* e)
{
	double ah, al, bh, bl;
	double q = a * b;
	viv_exp_split(a, &ah, &al);
	viv_exp_split(b, &bh, &bl);
	*p = q;
	*e = ((ah * bh - q) + ah * bl + al * bh) + al * bl;
}
static void viv_exp_dd_mul(double ah, double al, double bh, double bl, double* h, double* l)
{
	double p, e;
	viv_exp_two_prod(ah, bh, &p, &e);
	viv_exp_fast_two_sum(p, e + (ah * bl + al * bh), h, l);
}
static void viv_exp_dd_add(double ah, double al, double bh, double bl, double* h, double* l)
{
	double s, e;
	viv_exp_two_sum(ah, bh, &s, &e);
	viv_exp_fast_two_sum(s, e + (al + bl), h, l);
}
/* x - k L1, exactly: k L1 is exact for |k| < 2^17; past that k = 2m + b,
   and x - 2m L1, then - b L1, are each exact (Sterbenz) */
static double viv_exp_reduce(double x, double k)
{
	double l1 = viv_exp_bits(VIV_EXP_L1);
	if (fabs(k) < 131072.0)
		return x - k * l1;
	double m = floor(k * 0.5);
	double b = k - 2.0 * m;
	return (x - (2.0 * m) * l1) - b * l1;
}
/* 2^(j/128) e^r in double-double (about 2^-100 relative): the slow path */
static void viv_exp_slow_dd(double x, double k, int j, double* vh, double* vl)
{
	double t0 = viv_exp_reduce(x, k);
	double p2h, p2l, rh, rl, ah, al;
	viv_exp_two_prod(k, viv_exp_bits(VIV_EXP_L2), &p2h, &p2l);
	double p3 = k * viv_exp_bits(VIV_EXP_L3);
	viv_exp_two_sum(t0, -p2h, &rh, &rl);
	viv_exp_fast_two_sum(rh, (rl - p2l) - p3, &rh, &rl);
	ah = viv_exp_c(24);
	al = viv_exp_c(25);
	for (int n = 11; n >= 0; n--) {
		double mh, ml;
		viv_exp_dd_mul(ah, al, rh, rl, &mh, &ml);
		viv_exp_dd_add(mh, ml, viv_exp_c(2 * n), viv_exp_c(2 * n + 1), &ah, &al);
	}
	viv_exp_dd_mul(viv_exp_t(2 * j), viv_exp_t(2 * j + 1), ah, al, vh, vl);
}
static double viv_exp_slow(double x, double k, int j)
{
	double vh, vl;
	viv_exp_slow_dd(x, k, j, &vh, &vl);
	return vh + vl;
}
/* e^x below 2^-1021 (x <= -708): the slow path's double-double times 2^e,
   rounded once to the grid of 2^-1074 (exp_cr.rs exp_tiny) */
static double viv_exp_tiny(double x, double k, int j, int64_t e)
{
	double vh, vl, fh, fl;
	viv_exp_slow_dd(x, k, j, &vh, &vl);
	double sc = viv_exp_bits((uint64_t)(e + 1074 + 1023) << 52);
	double uh = vh * sc;
	double ul = vl * sc;
	double big = 4503599627370496.0; /* 2^52: above it a double is a whole number */
	double n0 = uh >= big ? uh : (uh + big) - big;
	viv_exp_two_sum(uh - n0, ul, &fh, &fl);
	double n = n0;
	if (fh > 0.5 || (fh == 0.5 && fl > 0.0))
		n = n0 + 1.0;
	else if (fh < -0.5 || (fh == -0.5 && fl < 0.0))
		n = n0 - 1.0;
	return n * viv_exp_bits(1);
}
static double viv_exp_cr(double x)
{
	if (!(fabs(x) >= 5.551115123125783e-17))
		return 1.0 + x; /* |x| < 2^-54, or NaN */
	if (!(x > -746.0 && x < 709.79))
		return viv_exp(x); /* 0, or an overflow to inf */
	double k = floor(x * viv_exp_bits(VIV_EXP_INVL) + 0.5);
	int64_t ki = (int64_t)k;
	int j = (int)(ki & 127);
	int64_t e = (ki - j) / 128;
	if (x <= -708.0)
		return viv_exp_tiny(x, k, j, e);
	double t0 = viv_exp_reduce(x, k);
	double kl2 = k * viv_exp_bits(VIV_EXP_L2);
	double r = t0 - kl2;
	double dr = (t0 - r) - kl2;
	double th = viv_exp_t(2 * j);
	double tl = viv_exp_t(2 * j + 1);
	double q = r * r * (viv_exp_c(4) + r * (viv_exp_c(6) + r * (viv_exp_c(8) + r * (viv_exp_c(10) + r * viv_exp_c(12)))));
	double a, a_lo;
	viv_exp_two_prod(th, r, &a, &a_lo);
	double s = th + a;
	double s_lo = (th - s) + a;
	double lo = s_lo + (a_lo + (th * (q + dr) + tl * (1.0 + r)));
	double y = s + lo;
	double d = lo - (y - s);
	/* the gap below y, which is never wider than the one above (y > 0) */
	double gap = y - viv_exp_bits(viv_exp_tobits(y) - 1);
	if (!(fabs(d) <= 0.5 * gap - th * 1.3552527156068805e-20))
		y = viv_exp_slow(x, k, j); /* within 2^-66 of a midpoint */
	if (e > 1023) /* 2^e overflows: two exact steps, the second to inf when e^x rounds there */
		return y * viv_exp_bits((uint64_t)(e - 1 + 1023) << 52) * 2.0;
	return y * viv_exp_bits((uint64_t)(e + 1023) << 52);
}

/* ---- Viv's own: erf and gamma (interp.rs erf_impl and gamma_impl) ---- */
/* erf through the incomplete gamma function P(1/2, x^2): its series below
   x^2 = 1.5, its continued fraction above. */
static double viv_erf(double x)
{
	if (x == 0.0)
		return 0.0;
	double sign = x < 0.0 ? -1.0 : 1.0;
	double a = 0.5;
	double xx = x * x;
	/* erf(+-inf) is +-1 (and so is erf of any x whose square overflows):
	   the formulas below would take -inf + inf */
	if (xx > DBL_MAX)
		return sign;
	double gln = 0.5723649429247001; /* ln Gamma(1/2) = ln sqrt(pi) */
	double p;
	if (xx < 1.5) {
		double ap = a;
		double sum = 1.0 / a;
		double del = sum;
		for (int k = 0; k < 300; k++) {
			ap += 1.0;
			del *= xx / ap;
			sum += del;
			if (fabs(del) < fabs(sum) * 1e-17)
				break;
		}
		p = sum * viv_exp_cr(-xx + a * viv_log(xx) - gln);
	} else {
		double b = xx + 1.0 - a;
		double c = 1e300;
		double d = 1.0 / b;
		double h = d;
		for (int i = 1; i < 300; i++) {
			double an = -(double)i * ((double)i - a);
			b += 2.0;
			d = an * d + b;
			if (fabs(d) < 1e-300)
				d = 1e-300;
			c = b + an / c;
			if (fabs(c) < 1e-300)
				c = 1e-300;
			d = 1.0 / d;
			double del = d * c;
			h *= del;
			if (fabs(del - 1.0) < 1e-17)
				break;
		}
		p = 1.0 - viv_exp_cr(-xx + a * viv_log(xx) - gln) * h;
	}
	return sign * p;
}

/* Gamma by the Lanczos approximation (g = 7, 9 terms), reflection below 1/2. */
static double viv_tgamma(double x)
{
	static const double G[9] = {
		0.99999999999980993, 676.5203681218851, -1259.1392167224028,
		771.32342877765313, -176.61502916214059, 12.507343278686905,
		-0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
	};
	const double pi = 3.141592653589793;
	if (x < 0.5)
		return pi / (viv_sin(pi * x) * viv_tgamma(1.0 - x));
	/* Gamma overflows past 171.62; the formula would take inf * 0 (gamma(inf)) */
	if (x > 172.0)
		return INFINITY;
	x = x - 1.0;
	double a = G[0];
	double t = x + 7.5;
	for (int i = 1; i < 9; i++)
		a += G[i] / (x + (double)i);
	return sqrt(2.0 * pi) * viv_pow(t, x + 0.5) * viv_exp_cr(-t) * a;
}

/* max and min, the same in every engine and on every platform: a NaN gives
   the other argument, and +0 is larger than -0 (C's fmax leaves the sign of
   a tie of zeros to the platform, and Rust's f64::max compiled to wasm
   answered max(-0.0, 0.0) = -0.0). */
static double viv_fmax(double x, double y)
{
	if (x != x)
		return y;
	if (y != y)
		return x;
	/* selects, not branches: which of two samples is larger is a coin
	   toss in a Monte Carlo loop, and a mispredicted branch per call made
	   NPB EP's `int(max(g1, g2))` cost more than its log */
	double m = x > y ? x : y;
	double tie = signbit(x) ? y : x;
	return x == y ? tie : m;
}
static double viv_fmin(double x, double y)
{
	if (x != x)
		return y;
	if (y != y)
		return x;
	double m = x < y ? x : y;
	double tie = signbit(x) ? x : y;
	return x == y ? tie : m;
}

/* ---- complex functions, by the interpreter's formulas (interp.rs, "1-argument
   math builtins on a COMPLEX scalar"), not the platform's <complex.h>. No
   header: `_Complex double` is a C99 keyword, and the parts are read and
   written through a union (C99 6.2.5: an array of two doubles). ---- */
typedef union { _Complex double z; double p[2]; } viv_cparts;
static _Complex double viv_cmake(double re, double im)
{
	viv_cparts u;
	u.p[0] = re;
	u.p[1] = im;
	return u.z;
}
static double viv_cabs(_Complex double z)
{
	viv_cparts u;
	u.z = z;
	return viv_hypot(u.p[0], u.p[1]);
}
static _Complex double viv_cexp(_Complex double z)
{
	viv_cparts u;
	u.z = z;
	double m = viv_exp_cr(u.p[0]);
	return viv_cmake(m * viv_cos(u.p[1]), m * viv_sin(u.p[1]));
}
static _Complex double viv_clog(_Complex double z)
{
	viv_cparts u;
	u.z = z;
	return viv_cmake(viv_log(viv_hypot(u.p[0], u.p[1])), viv_atan2(u.p[1], u.p[0]));
}
/* The principal square root by the half-angle formulas (Kahan 1987; C99
   Annex G): t = sqrt((|re| + |z|) / 2), then t + i im/(2t) for re >= 0 and
   |im|/(2t) + i copysign(t, im) for re < 0. No angle is formed, so
   sqrt(-1 + 0i) is exactly 0 + 1i. The twin of vmath::csqrt, operation for
   operation; a part above DBL_MAX/4 is scaled by 1/4 first. */
static _Complex double viv_csqrt(_Complex double z)
{
	viv_cparts u;
	u.z = z;
	double re = u.p[0], im = u.p[1];
	if (re == 0.0 && im == 0.0)
		return viv_cmake(0.0, im);
	if (isinf(im))
		return viv_cmake(INFINITY, im);
	int big = fabs(re) > DBL_MAX / 4.0 || fabs(im) > DBL_MAX / 4.0;
	double a = big ? re * 0.25 : re, b = big ? im * 0.25 : im, s = big ? 2.0 : 1.0;
	double t = sqrt((fabs(a) + viv_hypot(a, b)) * 0.5);
	if (a >= 0.0)
		return viv_cmake(s * t, s * (b / (2.0 * t)));
	return viv_cmake(s * (fabs(b) / (2.0 * t)), s * copysign(t, b));
}
static _Complex double viv_csin(_Complex double z)
{
	viv_cparts u;
	u.z = z;
	double re = u.p[0], im = u.p[1];
	return viv_cmake(viv_sin(re) * viv_cosh(im), viv_cos(re) * viv_sinh(im));
}
static _Complex double viv_ccos(_Complex double z)
{
	viv_cparts u;
	u.z = z;
	double re = u.p[0], im = u.p[1];
	return viv_cmake(viv_cos(re) * viv_cosh(im), -viv_sin(re) * viv_sinh(im));
}
/* ---- complex arithmetic, in real arithmetic written out, as vmath/mod.rs
   has it. The generated C calls these for `+ - * /` on complex scalars in
   place of C's own operators: `*` and `/` on _Complex double follow Annex G
   for infinities and NaNs, and `/` scales against overflow by whatever
   method the platform's compiler and runtime pick, so the last bit of a
   quotient depended on the machine. A real operand arrives converted by C's
   rules, with imaginary part +0.0, which is how the interpreter lifts it. ---- */
static _Complex double viv_cadd(_Complex double a, _Complex double b)
{
	viv_cparts x, y;
	x.z = a;
	y.z = b;
	return viv_cmake(x.p[0] + y.p[0], x.p[1] + y.p[1]);
}
static _Complex double viv_csub(_Complex double a, _Complex double b)
{
	viv_cparts x, y;
	x.z = a;
	y.z = b;
	return viv_cmake(x.p[0] - y.p[0], x.p[1] - y.p[1]);
}
/* (ar br - ai bi) + i (ar bi + ai br): four products, rounded one at a time
   (FP_CONTRACT is off for the whole file, so no fused multiply-add). */
static _Complex double viv_cmul(_Complex double a, _Complex double b)
{
	viv_cparts x, y;
	x.z = a;
	y.z = b;
	double ar = x.p[0], ai = x.p[1], br = y.p[0], bi = y.p[1];
	return viv_cmake(ar * br - ai * bi, ar * bi + ai * br);
}
/* Smith's algorithm (R. L. Smith, "Algorithm 116: Complex division", CACM
   5(8), 1962): divide numerator and denominator through by the larger
   component of the divisor, so the divisor is never squared. The textbook
   (a conj b) / |b|^2 overflows once |b| passes 1e154 and underflows below
   1e-154; this overflows only near where the quotient itself does. A real
   divisor (bi = 0) gives r = 0 and d = br, so each part is one correctly
   rounded division. A zero divisor gives NaN + NaN i. */
static void viv_cdiv_parts(double ar, double ai, double br, double bi, double* cr, double* ci)
{
	if (fabs(br) >= fabs(bi)) {
		double r = bi / br;
		double d = br + bi * r;
		*cr = (ar + ai * r) / d;
		*ci = (ai - ar * r) / d;
	} else {
		double r = br / bi;
		double d = bi + br * r;
		*cr = (ar * r + ai) / d;
		*ci = (ai * r - ar) / d;
	}
}
static _Complex double viv_cdiv(_Complex double a, _Complex double b)
{
	viv_cparts x, y;
	x.z = a;
	y.z = b;
	double cr, ci;
	viv_cdiv_parts(x.p[0], x.p[1], y.p[0], y.p[1], &cr, &ci);
	return viv_cmake(cr, ci);
}
/* sin z / cos z, by the division above */
static _Complex double viv_ctan(_Complex double z)
{
	viv_cparts u;
	u.z = z;
	double re = u.p[0], im = u.p[1];
	double sr = viv_sin(re) * viv_cosh(im), si = viv_cos(re) * viv_sinh(im);
	double cr = viv_cos(re) * viv_cosh(im), ci = -viv_sin(re) * viv_sinh(im);
	double tr, ti;
	viv_cdiv_parts(sr, si, cr, ci, &tr, &ti);
	return viv_cmake(tr, ti);
}

#pragma GCC diagnostic pop

/* ---- the generated C and the runtime say `sin(x)`; from here on that is
   viv_sin. Object-like, so `sin` passed as a function pointer (an array
   map) is Viv's too. A header included after this point that declares, say,
   `double sin(double)` redeclares viv_sin, which C allows after a static
   definition. ---- */
/* Fast mode (-DVIV_FAST) takes the platform's libm for
   the real functions below: each is within an ulp or two of Viv's, and keeps
   C99 Annex F's special values (NaN in, NaN out; the infinities), which
   Viv's follow too (-DVIV_FAST_KEEP_LIBM keeps Viv's, to tell what the
   libm changes from what the rest of fast mode does). fmax and fmin stay
   Viv's in every mode, because C's
   propagate a NaN differently, and so do the complex functions. */
/* x ^ 2 is the commonest power in numerical code. viv_pow answers it with
   x*x after its special cases, none of which changes x*x (1, NaN, +-0,
   +-inf all square to what pow gives), so this test gives the same bits and
   lets the C compiler fold a literal exponent into one multiply. */
static inline double viv_pow_sq(double x, double y) { return y == 2.0 ? x * x : viv_pow(x, y); }
#if !defined(VIV_FAST) || defined(VIV_FAST_KEEP_LIBM)
#define sin viv_sin
#define cos viv_cos
#define tan viv_tan
#define exp viv_exp_cr
#define expm1 viv_expm1
#define log viv_log
#define log10 viv_log10
#define pow viv_pow_sq
#define atan viv_atan
#define atan2 viv_atan2
#define asin viv_asin
#define acos viv_acos
#define sinh viv_sinh
#define cosh viv_cosh
#define tanh viv_tanh
#define hypot viv_hypot
#define erf viv_erf
#define tgamma viv_tgamma
#endif
#define fmax viv_fmax
#define fmin viv_fmin
#define cabs viv_cabs
#define cexp viv_cexp
#define clog viv_clog
#define csqrt viv_csqrt
#define csin viv_csin
#define ccos viv_ccos
#define ctan viv_ctan
