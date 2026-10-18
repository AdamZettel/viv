/* SPDX-License-Identifier: (MIT OR Apache-2.0) AND (MIT AND SunPro)
   SPDX-FileCopyrightText: 2026 The Viv authors
   SPDX-FileCopyrightText: 1993 Sun Microsystems, Inc.
   SPDX-FileCopyrightText: 2005-2020 Rich Felker et al. (musl)

   This file is compiled from these components; NOTICES, beside it, has
   the full text of each license.

     churn.viv
         NOASSERTION
     features.viv
         MIT OR Apache-2.0; 2026 The Viv authors
     the Viv runtime (lang/viv/src)
         MIT OR Apache-2.0; 2026 The Viv authors
     Viv's libm (lang/viv/src/vmath.c: fdlibm as musl carries it)
         MIT AND SunPro; 1993 Sun Microsystems, Inc.; 2005-2020 Rich Felker et al. (musl); 2026 The Viv authors
*/
/* churn.c -- written by `viv export`; do not edit.
   viv 0.0.1; program 63dec3d40467e2a7

   A kernel collection: one file, C99 with GNU attributes, the C library
   only. Build it:

       cc -O2 -ffp-contract=off -fvisibility=hidden -shared -fPIC churn.c -o libchurn.so -lm

   (exact: the interpreter's bits; README.md has the fast build). The
   interface is churn.h, pasted below. */
#ifndef VIV_EMBED
#define VIV_EMBED 1
#endif
/* churn.h -- written by `viv export`; do not edit.
   viv 0.0.1; program 63dec3d40467e2a7
   A kernel collection. */
#ifndef CHURN_H
#define CHURN_H
#include <stddef.h>
#include <stdint.h>

#define CHURN_ABI 1

/* an array argument, borrowed for the call: row-major, `dims[0..rank)` */
typedef struct { const double* data; int64_t rank; const int64_t* dims; } churn_arg;
/* an array result, allocated by the library and freed by the kernel's _free */
typedef struct { double* data; int64_t rank; int64_t dims[8]; } churn_array;
/* each printed line, without its newline; stream 1 is output, 2 a diagnostic */
typedef void (*churn_line_fn)(const char* line, long n, int stream, void* ctx);

enum { CHURN_OK = 0, CHURN_ERROR = 1, CHURN_INTERRUPTED = 2, CHURN_BAD_PARAM = 3, CHURN_BUSY = 4 };
enum { CHURN_E_ERROR = 1, CHURN_E_INDEX = 2, CHURN_E_VALUE = 3, CHURN_E_OVERFLOW = 4, CHURN_E_ZERODIV = 5,
       CHURN_E_RECURSION = 6, CHURN_E_MEMORY = 7, CHURN_E_INTERRUPT = 8, CHURN_E_STOP = 9 };

/* score(xf, xi, rows) -> result
     xf: float64 array, NaN and infinity allowed
     xi: int64 array
     rows: int
     -> result: Arr
   0 on success; otherwise msg (cap bytes) holds the error and r->error_kind its kind. */
typedef struct { churn_arg xf; churn_arg xi; int64_t rows; } churn_score_args;
typedef struct { churn_array result; int error_kind; } churn_score_results;
int churn_score(const churn_score_args* a, churn_score_results* r, char* msg, size_t cap);
void churn_score_free(churn_score_results* r);   /* the arrays of r; safe on a zeroed r and twice */

/* Run the program's top level once: its modules' constants, in import
   order, and its test blocks. The first call of a kernel does it if this
   was not called. Its status is kept: a failed init fails every call. */
int churn_init(churn_line_fn on_line, void* ctx, char* msg, size_t cap);
/* where the lines a kernel prints go; NULL (the default) drops them */
void churn_set_output(churn_line_fn on_line, void* ctx);
/* the test blocks init ran: their count, and each one's name, result and first failure */
int churn_ntests(void);
const char* churn_test_name(int i);
int churn_test_passed(int i);
const char* churn_test_detail(int i);
/* stop the call in progress at its next loop pass; from any thread */
void churn_interrupt(void);
int churn_abi(void);
const char* churn_version(void);
#endif
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#pragma STDC FP_CONTRACT OFF

/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* Errors and output.

   Every run-time error of the runtime writes its message with viv_errf and
   ends in viv_stop(kind), or in viv_stop_exit(kind) where the site used to
   call exit rather than _exit. Every byte a program prints goes through
   viv_printf, viv_putchar, viv_puts, viv_fputs and viv_fwrite (to stdout),
   and viv_fflush.

   In a program build these are macros for the C library's own calls, so a
   program prints and fails exactly as it always has: the message to stderr,
   then _exit(1), or exit(1), which runs the failed-assert summary. This part
   includes no header: each runtime fragment includes what its calls need,
   as before.

   Under -DVIV_EMBED (a program exported as a library, `viv export`) an error
   unwinds with longjmp to viv_embed_run, which returns the error's kind and
   message, and output goes line by line to the caller's callback. The C
   library's exit, _exit, printf, puts, putchar, atexit and perror are then
   poisoned (Viv's after_includes), so a runtime site that does not come
   through here stops the build. */
#if defined(__GNUC__)
#define VIV_NORETURN __attribute__((noreturn))
#define VIV_UNUSED __attribute__((unused))
#else
#define VIV_NORETURN
#define VIV_UNUSED
#endif
/* B251: GCC's loop vectorizer turns a complex multiply written in real
   arithmetic into FCMLA on aarch64, a fused multiply-add, even under
   -ffp-contract=off (GCC 15), and the products no longer round as the
   interpreter's do. The runtime's complex loops (the complex array
   arithmetic, the FFT, the complex matrix product) are kept scalar in an
   exact GCC build. Clang does not fuse them, and a fast build may. */
#if defined(__GNUC__) && !defined(__clang__) && !defined(VIV_FAST)
#define VIV_SCALAR_LOOPS __attribute__((optimize("no-tree-loop-vectorize")))
#else
#define VIV_SCALAR_LOOPS
#endif
/* What kind of error stopped a run, for a caller that tells them apart (the
   exported wrapper raises IndexError, ZeroDivisionError, ...). Every stop
   names its kind; the list is crate::errkind::ErrKind, which checks these
   lines against itself. */
#define VIV_E_ERROR 1
#define VIV_E_INDEX 2
#define VIV_E_VALUE 3
#define VIV_E_OVERFLOW 4
#define VIV_E_ZERODIV 5
#define VIV_E_RECURSION 6
#define VIV_E_MEMORY 7
#define VIV_E_INTERRUPT 8
#define VIV_E_STOP 9

#ifndef VIV_EMBED
#define viv_printf printf
#define viv_putchar putchar
#define viv_puts puts
#define viv_fputs fputs
#define viv_fwrite fwrite
#define viv_fflush fflush
#define viv_errf(...) fprintf(stderr, __VA_ARGS__)
#define viv_stop(kind) _exit(1)
#define viv_stop_exit(kind) exit(1)
/* an interrupt poll: nothing outside an exported library */
#define VIV_POLL() ((void)0)
#else
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <setjmp.h>
#include <signal.h>
/* One printed line, without its newline; `stream` is 1 for the program's
   output and 2 for a diagnostic that did not stop the run. */
typedef void (*viv_line_fn)(const char* line, long n, int stream, void* ctx);
static jmp_buf* viv_embed_jmp = 0;
static viv_line_fn viv_embed_line = 0;
static void* viv_embed_ctx = 0;
/* the output line being assembled */
static char* viv_embed_buf = 0;
static size_t viv_embed_n = 0, viv_embed_cap = 0;
/* the error message being assembled (viv_errf), and its kind */
static char viv_embed_msg[4096];
static size_t viv_embed_msg_n = 0;
static int viv_embed_kind = 0;
/* set by the exported interrupt function, from any thread */
static volatile sig_atomic_t viv_embed_stop_flag = 0;
/* the program's test blocks as they end: name, passed, first failure */
typedef struct { char* name; int passed; char* detail; } viv_embed_test_row;
static viv_embed_test_row* viv_embed_tests = 0;
static int viv_embed_ntests = 0, viv_embed_tests_cap = 0;

static void viv_embed_deliver(const char* s, size_t n, int stream) {
    if (viv_embed_line) viv_embed_line(s, (long)n, stream, viv_embed_ctx);
}
static void viv_embed_push(char c) {
    if (viv_embed_n + 1 >= viv_embed_cap) {
        size_t cap = viv_embed_cap ? 2 * viv_embed_cap : 256;
        char* b = (char*)realloc(viv_embed_buf, cap);
        if (!b) return;     /* out of memory: the line is cut short, not the run */
        viv_embed_buf = b;
        viv_embed_cap = cap;
    }
    viv_embed_buf[viv_embed_n++] = c;
}
static void viv_embed_flush_line(void) {
    if (!viv_embed_buf) {
        viv_embed_deliver("", 0, 1);
        return;
    }
    viv_embed_buf[viv_embed_n] = 0;
    viv_embed_deliver(viv_embed_buf, viv_embed_n, 1);
    viv_embed_n = 0;
}
static void viv_out(const char* s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '\n') viv_embed_flush_line();
        else viv_embed_push(s[i]);
    }
}
VIV_UNUSED static int viv_printf(const char* fmt, ...) {
    char small[1024];
    va_list ap;
    va_start(ap, fmt);
    int r = vsnprintf(small, sizeof small, fmt, ap);
    va_end(ap);
    if (r < 0) return r;
    if ((size_t)r < sizeof small) {
        viv_out(small, (size_t)r);
        return r;
    }
    char* big = (char*)malloc((size_t)r + 1);
    if (!big) {
        viv_out(small, sizeof small - 1);
        return r;
    }
    va_start(ap, fmt);
    vsnprintf(big, (size_t)r + 1, fmt, ap);
    va_end(ap);
    viv_out(big, (size_t)r);
    free(big);
    return r;
}
VIV_UNUSED static int viv_putchar(int c) {
    char ch = (char)c;
    viv_out(&ch, 1);
    return c;
}
VIV_UNUSED static int viv_puts(const char* s) {
    viv_out(s, strlen(s));
    viv_out("\n", 1);
    return 0;
}
/* the queued diagnostic's complete lines, as stream 2: for text that did not
   end in a stop, or when the buffer is full */
static void viv_embed_flush_msg(void) {
    size_t a = 0;
    for (size_t i = 0; i < viv_embed_msg_n; i++) {
        if (viv_embed_msg[i] == '\n') {
            viv_embed_msg[i] = 0;
            viv_embed_deliver(viv_embed_msg + a, i - a, 2);
            a = i + 1;
        }
    }
    if (a < viv_embed_msg_n) {
        viv_embed_msg[viv_embed_msg_n] = 0;
        viv_embed_deliver(viv_embed_msg + a, viv_embed_msg_n - a, 2);
    }
    viv_embed_msg_n = 0;
    viv_embed_msg[0] = 0;
}
static void viv_embed_vmsg(const char* fmt, va_list ap) {
    char part[2048];
    int r = vsnprintf(part, sizeof part, fmt, ap);
    if (r <= 0) return;
    size_t n = (size_t)r < sizeof part ? (size_t)r : sizeof part - 1;
    if (viv_embed_msg_n + n >= sizeof viv_embed_msg) viv_embed_flush_msg();
    if (n >= sizeof viv_embed_msg) n = sizeof viv_embed_msg - 1;
    memcpy(viv_embed_msg + viv_embed_msg_n, part, n);
    viv_embed_msg_n += n;
    viv_embed_msg[viv_embed_msg_n] = 0;
}
VIV_UNUSED static void viv_errf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    viv_embed_vmsg(fmt, ap);
    va_end(ap);
}
VIV_UNUSED static int viv_fputs(const char* s, FILE* f) {
    if (f == stdout) {
        viv_out(s, strlen(s));
        return 0;
    }
    if (f == stderr) {
        viv_errf("%s", s);
        return 0;
    }
    return fputs(s, f);
}
VIV_UNUSED static size_t viv_fwrite(const void* p, size_t size, size_t n, FILE* f) {
    if (f == stdout) {
        viv_out((const char*)p, size * n);
        return n;
    }
    return fwrite(p, size, n, f);
}
VIV_UNUSED static int viv_fflush(FILE* f) {
    return f == stdout || f == stderr ? 0 : fflush(f);
}
VIV_NORETURN VIV_UNUSED static void viv_stop(int kind) {
    viv_embed_kind = kind;
    if (!viv_embed_jmp) abort();     /* a stop outside a run: there is nowhere to return to */
    longjmp(*viv_embed_jmp, 1);
}
/* a site that called exit: in a library, the same as any other stop */
#define viv_stop_exit(kind) viv_stop(kind)
/* Polled once per iteration of the loops that can run long (lower.rs
   `poll_loop`) and at each loop iteration's arena release. */
#define VIV_POLL() do { if (viv_embed_stop_flag) { viv_errf("error: the run was interrupted\n"); viv_stop(VIV_E_INTERRUPT); } } while (0)
static char* viv_embed_dup(const char* s) {
    size_t n = strlen(s);
    char* d = (char*)malloc(n + 1);
    if (d) memcpy(d, s, n + 1);
    return d;
}
VIV_UNUSED static void viv_embed_test(const char* name, int passed, const char* detail) {
    if (viv_embed_ntests == viv_embed_tests_cap) {
        int cap = viv_embed_tests_cap ? 2 * viv_embed_tests_cap : 16;
        viv_embed_test_row* t = (viv_embed_test_row*)realloc(viv_embed_tests, (size_t)cap * sizeof *t);
        if (!t) return;
        viv_embed_tests = t;
        viv_embed_tests_cap = cap;
    }
    viv_embed_test_row* r = &viv_embed_tests[viv_embed_ntests++];
    r->name = viv_embed_dup(name ? name : "");
    r->passed = passed;
    r->detail = viv_embed_dup(detail ? detail : "");
}
/* Hand the test table to the caller, who frees each string and the array. */
VIV_UNUSED static viv_embed_test_row* viv_embed_take_tests(int* n) {
    viv_embed_test_row* t = viv_embed_tests;
    *n = viv_embed_ntests;
    viv_embed_tests = 0;
    viv_embed_ntests = viv_embed_tests_cap = 0;
    return t;
}
VIV_UNUSED static void viv_embed_free_tests(viv_embed_test_row* t, int n) {
    for (int i = 0; i < n; i++) {
        free(t[i].name);
        free(t[i].detail);
    }
    free(t);
}
/* Run `body` (the program's main) once. Returns 0, or the kind of the error
   that stopped it, with its message in msg (cap bytes, always terminated).
   Output goes to on_line as it is printed. The caller resets the program's
   state before and releases its arrays after. */
VIV_UNUSED static int viv_embed_run(int (*body)(void), viv_line_fn on_line, void* ctx, char* msg, size_t cap) {
    jmp_buf here;
    volatile int code = 0;
    viv_embed_line = on_line;
    viv_embed_ctx = ctx;
    viv_embed_n = 0;
    viv_embed_msg_n = 0;
    viv_embed_msg[0] = 0;
    viv_embed_kind = 0;
    viv_embed_free_tests(viv_embed_tests, viv_embed_ntests);
    viv_embed_tests = 0;
    viv_embed_ntests = viv_embed_tests_cap = 0;
    viv_embed_jmp = &here;
    if (setjmp(here) == 0) {
        body();
    } else {
        code = viv_embed_kind ? viv_embed_kind : VIV_E_ERROR;
    }
    viv_embed_jmp = 0;
    if (viv_embed_n) viv_embed_flush_line();
    if (msg && cap) msg[0] = 0;
    if (code) {
        /* the message without its trailing newline */
        size_t n = viv_embed_msg_n;
        while (n && viv_embed_msg[n - 1] == '\n') n--;
        if (msg && cap) {
            size_t k = n < cap - 1 ? n : cap - 1;
            memcpy(msg, viv_embed_msg, k);
            msg[k] = 0;
        }
        viv_embed_msg_n = 0;
    } else if (viv_embed_msg_n) {
        viv_embed_flush_msg();
    }
    viv_embed_line = 0;
    viv_embed_ctx = 0;
    return code;
}
#endif

/* Viv's libm (lang/viv/src/vmath.c, generated by tools/vmath/gen_c.py; do
   not edit by hand). Every engine computes sin, exp, pow and the rest with
   this one algorithm: the interpreter and the direct wasm backend with the
   Rust twin in lang/viv/src/vmath/, a compiled program with this file, which
   the C backend puts at the top of every program. Platform libms round
   differently in the last bit (tools/libm-sweep/run.sh counts how often), and
   a program near a tolerance then prints different digits in different
   engines. The code is fdlibm as musl 1.1.19 carries it (MIT), except exp,
   which is Viv's own and correctly rounded (viv_exp_cr); the Rust twin
   is the `libm` crate 0.2.8, a port of the same files. `cargo test -p viv
   --test vmath` holds the two to the same bits. */
#pragma STDC FP_CONTRACT OFF
#include <math.h>
#include <stdint.h>
#include <float.h>

/* musl's code compares a uint32_t with an int32_t once (pow), on purpose, and
   a program uses a few of these functions, not all: neither is worth a
   warning in a program compiled with -Wall -Wextra. Popped at the end. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-function"

#define FORCE_EVAL(x) do { volatile double __x; __x = (x); (void)__x; } while (0)
#define EXTRACT_WORDS(hi,lo,d) do { union {double f; uint64_t i;} __u; __u.f = (d); (hi) = __u.i >> 32; (lo) = (uint32_t)__u.i; } while (0)
#define GET_HIGH_WORD(hi,d) do { union {double f; uint64_t i;} __u; __u.f = (d); (hi) = __u.i >> 32; } while (0)
#define GET_LOW_WORD(lo,d) do { union {double f; uint64_t i;} __u; __u.f = (d); (lo) = (uint32_t)__u.i; } while (0)
#define INSERT_WORDS(d,hi,lo) do { union {double f; uint64_t i;} __u; __u.i = ((uint64_t)(hi)<<32) | (uint32_t)(lo); (d) = __u.f; } while (0)
#define SET_HIGH_WORD(d,hi) do { union {double f; uint64_t i;} __u; __u.f = (d); __u.i &= 0xffffffff; __u.i |= (uint64_t)(hi) << 32; (d) = __u.f; } while (0)
#define SET_LOW_WORD(d,lo) do { union {double f; uint64_t i;} __u; __u.f = (d); __u.i &= 0xffffffff00000000ull; __u.i |= (uint32_t)(lo); (d) = __u.f; } while (0)
#define STRICT_ASSIGN(type, lval, rval) ((lval) = (type)(rval))

/* ---- musl src/math/scalbn.c ---- */
static double viv_scalbn(double x, int n)
{
	union {double f; uint64_t i;} u;
	double y = x;

	if (n > 1023) {
		y *= 0x1p1023;
		n -= 1023;
		if (n > 1023) {
			y *= 0x1p1023;
			n -= 1023;
			if (n > 1023)
				n = 1023;
		}
	} else if (n < -1022) {
		/* make sure final n < -53 to avoid double
		   rounding in the subnormal range */
		y *= 0x1p-1022 * 0x1p53;
		n += 1022 - 53;
		if (n < -1022) {
			y *= 0x1p-1022 * 0x1p53;
			n += 1022 - 53;
			if (n < -1022)
				n = -1022;
		}
	}
	u.i = (uint64_t)(0x3ff+n)<<52;
	x = y * u.f;
	return x;
}

/* ---- musl src/math/__rem_pio2_large.c ---- */
/* origin: FreeBSD /usr/src/lib/msun/src/k_rem_pio2.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/*
 * __rem_pio2_large(x,y,e0,nx,prec)
 * double x[],y[]; int e0,nx,prec;
 *
 * __rem_pio2_large return the last three digits of N with
 *              y = x - N*pi/2
 * so that |y| < pi/2.
 *
 * The method is to compute the integer (mod 8) and fraction parts of
 * (2/pi)*x without doing the full multiplication. In general we
 * skip the part of the product that are known to be a huge integer (
 * more accurately, = 0 mod 8 ). Thus the number of operations are
 * independent of the exponent of the input.
 *
 * (2/pi) is represented by an array of 24-bit integers in ipio2[].
 *
 * Input parameters:
 *      x[]     The input value (must be positive) is broken into nx
 *              pieces of 24-bit integers in double precision format.
 *              x[i] will be the i-th 24 bit of x. The scaled exponent
 *              of x[0] is given in input parameter e0 (i.e., x[0]*2^e0
 *              match x's up to 24 bits.
 *
 *              Example of breaking a double positive z into x[0]+x[1]+x[2]:
 *                      e0 = ilogb(z)-23
 *                      z  = scalbn(z,-e0)
 *              for i = 0,1,2
 *                      x[i] = floor(z)
 *                      z    = (z-x[i])*2**24
 *
 *
 *      y[]     ouput result in an array of double precision numbers.
 *              The dimension of y[] is:
 *                      24-bit  precision       1
 *                      53-bit  precision       2
 *                      64-bit  precision       2
 *                      113-bit precision       3
 *              The actual value is the sum of them. Thus for 113-bit
 *              precison, one may have to do something like:
 *
 *              long double t,w,r_head, r_tail;
 *              t = (long double)y[2] + (long double)y[1];
 *              w = (long double)y[0];
 *              r_head = t+w;
 *              r_tail = w - (r_head - t);
 *
 *      e0      The exponent of x[0]. Must be <= 16360 or you need to
 *              expand the ipio2 table.
 *
 *      nx      dimension of x[]
 *
 *      prec    an integer indicating the precision:
 *                      0       24  bits (single)
 *                      1       53  bits (double)
 *                      2       64  bits (extended)
 *                      3       113 bits (quad)
 *
 * External function:
 *      double scalbn(), floor();
 *
 *
 * Here is the description of some local variables:
 *
 *      jk      jk+1 is the initial number of terms of ipio2[] needed
 *              in the computation. The minimum and recommended value
 *              for jk is 3,4,4,6 for single, double, extended, and quad.
 *              jk+1 must be 2 larger than you might expect so that our
 *              recomputation test works. (Up to 24 bits in the integer
 *              part (the 24 bits of it that we compute) and 23 bits in
 *              the fraction part may be lost to cancelation before we
 *              recompute.)
 *
 *      jz      local integer variable indicating the number of
 *              terms of ipio2[] used.
 *
 *      jx      nx - 1
 *
 *      jv      index for pointing to the suitable ipio2[] for the
 *              computation. In general, we want
 *                      ( 2^e0*x[0] * ipio2[jv-1]*2^(-24jv) )/8
 *              is an integer. Thus
 *                      e0-3-24*jv >= 0 or (e0-3)/24 >= jv
 *              Hence jv = max(0,(e0-3)/24).
 *
 *      jp      jp+1 is the number of terms in PIo2[] needed, jp = jk.
 *
 *      q[]     double array with integral value, representing the
 *              24-bits chunk of the product of x and 2/pi.
 *
 *      q0      the corresponding exponent of q[0]. Note that the
 *              exponent for q[i] would be q0-24*i.
 *
 *      PIo2[]  double precision array, obtained by cutting pi/2
 *              into 24 bits chunks.
 *
 *      f[]     ipio2[] in floating point
 *
 *      iq[]    integer array by breaking up q[] in 24-bits chunk.
 *
 *      fq[]    final product of x*(2/pi) in fq[0],..,fq[jk]
 *
 *      ih      integer. If >0 it indicates q[] is >= 0.5, hence
 *              it also indicates the *sign* of the result.
 *
 */
/*
 * Constants:
 * The hexadecimal values are the intended ones for the following
 * constants. The decimal values may be used, provided that the
 * compiler will convert from decimal to binary accurately enough
 * to produce the hexadecimal values shown.
 */


static const int viv__rem_pio2_large_init_jk[] = {3,4,4,6}; /* initial value for jk */

/*
 * Table of constants for 2/pi, 396 Hex digits (476 decimal) of 2/pi
 *
 *              integer array, contains the (24*i)-th to (24*i+23)-th
 *              bit of 2/pi after binary point. The corresponding
 *              floating value is
 *
 *                      ipio2[i] * 2^(-24(i+1)).
 *
 * NB: This table must have at least (e0-3)/24 + jk terms.
 *     For quad precision (e0 <= 16360, jk = 6), this is 686.
 */
static const int32_t viv__rem_pio2_large_ipio2[] = {
0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62,
0x95993C, 0x439041, 0xFE5163, 0xABDEBB, 0xC561B7, 0x246E3A,
0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129,
0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41,
0x3991D6, 0x398353, 0x39F49C, 0x845F8B, 0xBDF928, 0x3B1FF8,
0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF,
0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5,
0xF17B3D, 0x0739F7, 0x8A5292, 0xEA6BFB, 0x5FB11F, 0x8D5D08,
0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
0x91615E, 0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880,
0x4D7327, 0x310606, 0x1556CA, 0x73A8C9, 0x60E27B, 0xC08C6B,

#if LDBL_MAX_EXP > 1024
0x47C419, 0xC367CD, 0xDCE809, 0x2A8359, 0xC4768B, 0x961CA6,
0xDDAF44, 0xD15719, 0x053EA5, 0xFF0705, 0x3F7E33, 0xE832C2,
0xDE4F98, 0x327DBB, 0xC33D26, 0xEF6B1E, 0x5EF89F, 0x3A1F35,
0xCAF27F, 0x1D87F1, 0x21907C, 0x7C246A, 0xFA6ED5, 0x772D30,
0x433B15, 0xC614B5, 0x9D19C3, 0xC2C4AD, 0x414D2C, 0x5D000C,
0x467D86, 0x2D71E3, 0x9AC69B, 0x006233, 0x7CD2B4, 0x97A7B4,
0xD55537, 0xF63ED7, 0x1810A3, 0xFC764D, 0x2A9D64, 0xABD770,
0xF87C63, 0x57B07A, 0xE71517, 0x5649C0, 0xD9D63B, 0x3884A7,
0xCB2324, 0x778AD6, 0x23545A, 0xB91F00, 0x1B0AF1, 0xDFCE19,
0xFF319F, 0x6A1E66, 0x615799, 0x47FBAC, 0xD87F7E, 0xB76522,
0x89E832, 0x60BFE6, 0xCDC4EF, 0x09366C, 0xD43F5D, 0xD7DE16,
0xDE3B58, 0x929BDE, 0x2822D2, 0xE88628, 0x4D58E2, 0x32CAC6,
0x16E308, 0xCB7DE0, 0x50C017, 0xA71DF3, 0x5BE018, 0x34132E,
0x621283, 0x014883, 0x5B8EF5, 0x7FB0AD, 0xF2E91E, 0x434A48,
0xD36710, 0xD8DDAA, 0x425FAE, 0xCE616A, 0xA4280A, 0xB499D3,
0xF2A606, 0x7F775C, 0x83C2A3, 0x883C61, 0x78738A, 0x5A8CAF,
0xBDD76F, 0x63A62D, 0xCBBFF4, 0xEF818D, 0x67C126, 0x45CA55,
0x36D9CA, 0xD2A828, 0x8D61C2, 0x77C912, 0x142604, 0x9B4612,
0xC459C4, 0x44C5C8, 0x91B24D, 0xF31700, 0xAD43D4, 0xE54929,
0x10D5FD, 0xFCBE00, 0xCC941E, 0xEECE70, 0xF53E13, 0x80F1EC,
0xC3E7B3, 0x28F8C7, 0x940593, 0x3E71C1, 0xB3092E, 0xF3450B,
0x9C1288, 0x7B20AB, 0x9FB52E, 0xC29247, 0x2F327B, 0x6D550C,
0x90A772, 0x1FE76B, 0x96CB31, 0x4A1679, 0xE27941, 0x89DFF4,
0x9794E8, 0x84E6E2, 0x973199, 0x6BED88, 0x365F5F, 0x0EFDBB,
0xB49A48, 0x6CA467, 0x427271, 0x325D8D, 0xB8159F, 0x09E5BC,
0x25318D, 0x3974F7, 0x1C0530, 0x010C0D, 0x68084B, 0x58EE2C,
0x90AA47, 0x02E774, 0x24D6BD, 0xA67DF7, 0x72486E, 0xEF169F,
0xA6948E, 0xF691B4, 0x5153D1, 0xF20ACF, 0x339820, 0x7E4BF5,
0x6863B2, 0x5F3EDD, 0x035D40, 0x7F8985, 0x295255, 0xC06437,
0x10D86D, 0x324832, 0x754C5B, 0xD4714E, 0x6E5445, 0xC1090B,
0x69F52A, 0xD56614, 0x9D0727, 0x50045D, 0xDB3BB4, 0xC576EA,
0x17F987, 0x7D6B49, 0xBA271D, 0x296996, 0xACCCC6, 0x5414AD,
0x6AE290, 0x89D988, 0x50722C, 0xBEA404, 0x940777, 0x7030F3,
0x27FC00, 0xA871EA, 0x49C266, 0x3DE064, 0x83DD97, 0x973FA3,
0xFD9443, 0x8C860D, 0xDE4131, 0x9D3992, 0x8C70DD, 0xE7B717,
0x3BDF08, 0x2B3715, 0xA0805C, 0x93805A, 0x921110, 0xD8E80F,
0xAF806C, 0x4BFFDB, 0x0F9038, 0x761859, 0x15A562, 0xBBCB61,
0xB989C7, 0xBD4010, 0x04F2D2, 0x277549, 0xF6B6EB, 0xBB22DB,
0xAA140A, 0x2F2689, 0x768364, 0x333B09, 0x1A940E, 0xAA3A51,
0xC2A31D, 0xAEEDAF, 0x12265C, 0x4DC26D, 0x9C7A2D, 0x9756C0,
0x833F03, 0xF6F009, 0x8C402B, 0x99316D, 0x07B439, 0x15200C,
0x5BC3D8, 0xC492F5, 0x4BADC6, 0xA5CA4E, 0xCD37A7, 0x36A9E6,
0x9492AB, 0x6842DD, 0xDE6319, 0xEF8C76, 0x528B68, 0x37DBFC,
0xABA1AE, 0x3115DF, 0xA1AE00, 0xDAFB0C, 0x664D64, 0xB705ED,
0x306529, 0xBF5657, 0x3AFF47, 0xB9F96A, 0xF3BE75, 0xDF9328,
0x3080AB, 0xF68C66, 0x15CB04, 0x0622FA, 0x1DE4D9, 0xA4B33D,
0x8F1B57, 0x09CD36, 0xE9424E, 0xA4BE13, 0xB52333, 0x1AAAF0,
0xA8654F, 0xA5C1D2, 0x0F3F0B, 0xCD785B, 0x76F923, 0x048B7B,
0x721789, 0x53A6C6, 0xE26E6F, 0x00EBEF, 0x584A9B, 0xB7DAC4,
0xBA66AA, 0xCFCF76, 0x1D02D1, 0x2DF1B1, 0xC1998C, 0x77ADC3,
0xDA4886, 0xA05DF7, 0xF480C6, 0x2FF0AC, 0x9AECDD, 0xBC5C3F,
0x6DDED0, 0x1FC790, 0xB6DB2A, 0x3A25A3, 0x9AAF00, 0x9353AD,
0x0457B6, 0xB42D29, 0x7E804B, 0xA707DA, 0x0EAA76, 0xA1597B,
0x2A1216, 0x2DB7DC, 0xFDE5FA, 0xFEDB89, 0xFDBE89, 0x6C76E4,
0xFCA906, 0x70803E, 0x156E85, 0xFF87FD, 0x073E28, 0x336761,
0x86182A, 0xEABD4D, 0xAFE7B3, 0x6E6D8F, 0x396795, 0x5BBF31,
0x48D784, 0x16DF30, 0x432DC7, 0x356125, 0xCE70C9, 0xB8CB30,
0xFD6CBF, 0xA200A4, 0xE46C05, 0xA0DD5A, 0x476F21, 0xD21262,
0x845CB9, 0x496170, 0xE0566B, 0x015299, 0x375550, 0xB7D51E,
0xC4F133, 0x5F6E13, 0xE4305D, 0xA92E85, 0xC3B21D, 0x3632A1,
0xA4B708, 0xD4B1EA, 0x21F716, 0xE4698F, 0x77FF27, 0x80030C,
0x2D408D, 0xA0CD4F, 0x99A520, 0xD3A2B3, 0x0A5D2F, 0x42F9B4,
0xCBDA11, 0xD0BE7D, 0xC1DB9B, 0xBD17AB, 0x81A2CA, 0x5C6A08,
0x17552E, 0x550027, 0xF0147F, 0x8607E1, 0x640B14, 0x8D4196,
0xDEBE87, 0x2AFDDA, 0xB6256B, 0x34897B, 0xFEF305, 0x9EBFB9,
0x4F6A68, 0xA82A4A, 0x5AC44F, 0xBCF82D, 0x985AD7, 0x95C7F4,
0x8D4D0D, 0xA63A20, 0x5F57A4, 0xB13F14, 0x953880, 0x0120CC,
0x86DD71, 0xB6DEC9, 0xF560BF, 0x11654D, 0x6B0701, 0xACB08C,
0xD0C0B2, 0x485551, 0x0EFB1E, 0xC37295, 0x3B06A3, 0x3540C0,
0x7BDC06, 0xCC45E0, 0xFA294E, 0xC8CAD6, 0x41F3E8, 0xDE647C,
0xD8649B, 0x31BED9, 0xC397A4, 0xD45877, 0xC5E369, 0x13DAF0,
0x3C3ABA, 0x461846, 0x5F7555, 0xF5BDD2, 0xC6926E, 0x5D2EAC,
0xED440E, 0x423E1C, 0x87C461, 0xE9FD29, 0xF3D6E7, 0xCA7C22,
0x35916F, 0xC5E008, 0x8DD7FF, 0xE26A6E, 0xC6FDB0, 0xC10893,
0x745D7C, 0xB2AD6B, 0x9D6ECD, 0x7B723E, 0x6A11C6, 0xA9CFF7,
0xDF7329, 0xBAC9B5, 0x5100B7, 0x0DB2E2, 0x24BA74, 0x607DE5,
0x8AD874, 0x2C150D, 0x0C1881, 0x94667E, 0x162901, 0x767A9F,
0xBEFDFD, 0xEF4556, 0x367ED9, 0x13D9EC, 0xB9BA8B, 0xFC97C4,
0x27A831, 0xC36EF1, 0x36C594, 0x56A8D8, 0xB5A8B4, 0x0ECCCF,
0x2D8912, 0x34576F, 0x89562C, 0xE3CE99, 0xB920D6, 0xAA5E6B,
0x9C2A3E, 0xCC5F11, 0x4A0BFD, 0xFBF4E1, 0x6D3B8E, 0x2C86E2,
0x84D4E9, 0xA9B4FC, 0xD1EEEF, 0xC9352E, 0x61392F, 0x442138,
0xC8D91B, 0x0AFC81, 0x6A4AFB, 0xD81C2F, 0x84B453, 0x8C994E,
0xCC2254, 0xDC552A, 0xD6C6C0, 0x96190B, 0xB8701A, 0x649569,
0x605A26, 0xEE523F, 0x0F117F, 0x11B5F4, 0xF5CBFC, 0x2DBC34,
0xEEBC34, 0xCC5DE8, 0x605EDD, 0x9B8E67, 0xEF3392, 0xB817C9,
0x9B5861, 0xBC57E1, 0xC68351, 0x103ED8, 0x4871DD, 0xDD1C2D,
0xA118AF, 0x462C21, 0xD7F359, 0x987AD9, 0xC0549E, 0xFA864F,
0xFC0656, 0xAE79E5, 0x362289, 0x22AD38, 0xDC9367, 0xAAE855,
0x382682, 0x9BE7CA, 0xA40D51, 0xB13399, 0x0ED7A9, 0x480569,
0xF0B265, 0xA7887F, 0x974C88, 0x36D1F9, 0xB39221, 0x4A827B,
0x21CF98, 0xDC9F40, 0x5547DC, 0x3A74E1, 0x42EB67, 0xDF9DFE,
0x5FD45E, 0xA4677B, 0x7AACBA, 0xA2F655, 0x23882B, 0x55BA41,
0x086E59, 0x862A21, 0x834739, 0xE6E389, 0xD49EE5, 0x40FB49,
0xE956FF, 0xCA0F1C, 0x8A59C5, 0x2BFA94, 0xC5C1D3, 0xCFC50F,
0xAE5ADB, 0x86C547, 0x624385, 0x3B8621, 0x94792C, 0x876110,
0x7B4C2A, 0x1A2C80, 0x12BF43, 0x902688, 0x893C78, 0xE4C4A8,
0x7BDBE5, 0xC23AC4, 0xEAF426, 0x8A67F7, 0xBF920D, 0x2BA365,
0xB1933D, 0x0B7CBD, 0xDC51A4, 0x63DD27, 0xDDE169, 0x19949A,
0x9529A8, 0x28CE68, 0xB4ED09, 0x209F44, 0xCA984E, 0x638270,
0x237C7E, 0x32B90F, 0x8EF5A7, 0xE75614, 0x08F121, 0x2A9DB5,
0x4D7E6F, 0x5119A5, 0xABF9B5, 0xD6DF82, 0x61DD96, 0x023616,
0x9F3AC4, 0xA1A283, 0x6DED72, 0x7A8D39, 0xA9B882, 0x5C326B,
0x5B2746, 0xED3400, 0x7700D2, 0x55F4FC, 0x4D5901, 0x8071E0,
#endif
};

static const double viv__rem_pio2_large_PIo2[] = {
  1.57079625129699707031e+00, /* 0x3FF921FB, 0x40000000 */
  7.54978941586159635335e-08, /* 0x3E74442D, 0x00000000 */
  5.39030252995776476554e-15, /* 0x3CF84698, 0x80000000 */
  3.28200341580791294123e-22, /* 0x3B78CC51, 0x60000000 */
  1.27065575308067607349e-29, /* 0x39F01B83, 0x80000000 */
  1.22933308981111328932e-36, /* 0x387A2520, 0x40000000 */
  2.73370053816464559624e-44, /* 0x36E38222, 0x80000000 */
  2.16741683877804819444e-51, /* 0x3569F31D, 0x00000000 */
};

static int viv__rem_pio2_large(double *x, double *y, int e0, int nx, int prec)
{
	int32_t jz,jx,jv,jp,jk,carry,n,iq[20],i,j,k,m,q0,ih;
	double z,fw,f[20],fq[20],q[20];

	/* initialize jk*/
	jk = viv__rem_pio2_large_init_jk[prec];
	jp = jk;

	/* determine jx,jv,q0, note that 3>q0 */
	jx = nx-1;
	jv = (e0-3)/24;  if(jv<0) jv=0;
	q0 = e0-24*(jv+1);

	/* set up f[0] to f[jx+jk] where f[jx+jk] = ipio2[jv+jk] */
	j = jv-jx; m = jx+jk;
	for (i=0; i<=m; i++,j++)
		f[i] = j<0 ? 0.0 : (double)viv__rem_pio2_large_ipio2[j];

	/* compute q[0],q[1],...q[jk] */
	for (i=0; i<=jk; i++) {
		for (j=0,fw=0.0; j<=jx; j++)
			fw += x[j]*f[jx+i-j];
		q[i] = fw;
	}

	jz = jk;
recompute:
	/* distill q[] into iq[] reversingly */
	for (i=0,j=jz,z=q[jz]; j>0; i++,j--) {
		fw    = (double)(int32_t)(0x1p-24*z);
		iq[i] = (int32_t)(z - 0x1p24*fw);
		z     = q[j-1]+fw;
	}

	/* compute n */
	z  = viv_scalbn(z,q0);       /* actual value of z */
	z -= 8.0*floor(z*0.125); /* trim off integer >= 8 */
	n  = (int32_t)z;
	z -= (double)n;
	ih = 0;
	if (q0 > 0) {  /* need iq[jz-1] to determine n */
		i  = iq[jz-1]>>(24-q0); n += i;
		iq[jz-1] -= i<<(24-q0);
		ih = iq[jz-1]>>(23-q0);
	}
	else if (q0 == 0) ih = iq[jz-1]>>23;
	else if (z >= 0.5) ih = 2;

	if (ih > 0) {  /* q > 0.5 */
		n += 1; carry = 0;
		for (i=0; i<jz; i++) {  /* compute 1-q */
			j = iq[i];
			if (carry == 0) {
				if (j != 0) {
					carry = 1;
					iq[i] = 0x1000000 - j;
				}
			} else
				iq[i] = 0xffffff - j;
		}
		if (q0 > 0) {  /* rare case: chance is 1 in 12 */
			switch(q0) {
			case 1:
				iq[jz-1] &= 0x7fffff; break;
			case 2:
				iq[jz-1] &= 0x3fffff; break;
			}
		}
		if (ih == 2) {
			z = 1.0 - z;
			if (carry != 0)
				z -= viv_scalbn(1.0,q0);
		}
	}

	/* check if recomputation is needed */
	if (z == 0.0) {
		j = 0;
		for (i=jz-1; i>=jk; i--) j |= iq[i];
		if (j == 0) {  /* need recomputation */
			for (k=1; iq[jk-k]==0; k++);  /* k = no. of terms needed */

			for (i=jz+1; i<=jz+k; i++) {  /* add q[jz+1] to q[jz+k] */
				f[jx+i] = (double)viv__rem_pio2_large_ipio2[jv+i];
				for (j=0,fw=0.0; j<=jx; j++)
					fw += x[j]*f[jx+i-j];
				q[i] = fw;
			}
			jz += k;
			goto recompute;
		}
	}

	/* chop off zero terms */
	if (z == 0.0) {
		jz -= 1;
		q0 -= 24;
		while (iq[jz] == 0) {
			jz--;
			q0 -= 24;
		}
	} else { /* break z into 24-bit if necessary */
		z = viv_scalbn(z,-q0);
		if (z >= 0x1p24) {
			fw = (double)(int32_t)(0x1p-24*z);
			iq[jz] = (int32_t)(z - 0x1p24*fw);
			jz += 1;
			q0 += 24;
			iq[jz] = (int32_t)fw;
		} else
			iq[jz] = (int32_t)z;
	}

	/* convert integer "bit" chunk to floating-point value */
	fw = viv_scalbn(1.0,q0);
	for (i=jz; i>=0; i--) {
		q[i] = fw*(double)iq[i];
		fw *= 0x1p-24;
	}

	/* compute PIo2[0,...,jp]*q[jz,...,0] */
	for(i=jz; i>=0; i--) {
		for (fw=0.0,k=0; k<=jp && k<=jz-i; k++)
			fw += viv__rem_pio2_large_PIo2[k]*q[i+k];
		fq[jz-i] = fw;
	}

	/* compress fq[] into y[] */
	switch(prec) {
	case 0:
		fw = 0.0;
		for (i=jz; i>=0; i--)
			fw += fq[i];
		y[0] = ih==0 ? fw : -fw;
		break;
	case 1:
	case 2:
		fw = 0.0;
		for (i=jz; i>=0; i--)
			fw += fq[i];
		// TODO: drop excess precision here once double is used
		fw = (double)fw;
		y[0] = ih==0 ? fw : -fw;
		fw = fq[0]-fw;
		for (i=1; i<=jz; i++)
			fw += fq[i];
		y[1] = ih==0 ? fw : -fw;
		break;
	case 3:  /* painful */
		for (i=jz; i>0; i--) {
			fw      = fq[i-1]+fq[i];
			fq[i]  += fq[i-1]-fw;
			fq[i-1] = fw;
		}
		for (i=jz; i>1; i--) {
			fw      = fq[i-1]+fq[i];
			fq[i]  += fq[i-1]-fw;
			fq[i-1] = fw;
		}
		for (fw=0.0,i=jz; i>=2; i--)
			fw += fq[i];
		if (ih==0) {
			y[0] =  fq[0]; y[1] =  fq[1]; y[2] =  fw;
		} else {
			y[0] = -fq[0]; y[1] = -fq[1]; y[2] = -fw;
		}
	}
	return n&7;
}

/* ---- musl src/math/__rem_pio2.c ---- */
/* origin: FreeBSD /usr/src/lib/msun/src/e_rem_pio2.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 *
 * Optimized by Bruce D. Evans.
 */
/* __rem_pio2(x,y)
 *
 * return the remainder of x rem pi/2 in y[0]+y[1]
 * use __rem_pio2_large() for large x
 */


#if FLT_EVAL_METHOD==0 || FLT_EVAL_METHOD==1
#define EPS DBL_EPSILON
#elif FLT_EVAL_METHOD==2
#define EPS LDBL_EPSILON
#endif

/*
 * invpio2:  53 bits of 2/pi
 * pio2_1:   first  33 bit of pi/2
 * pio2_1t:  pi/2 - pio2_1
 * pio2_2:   second 33 bit of pi/2
 * pio2_2t:  pi/2 - (pio2_1+pio2_2)
 * pio2_3:   third  33 bit of pi/2
 * pio2_3t:  pi/2 - (pio2_1+pio2_2+pio2_3)
 */
static const double
viv__rem_pio2_toint   = 1.5/EPS,
viv__rem_pio2_invpio2 = 6.36619772367581382433e-01, /* 0x3FE45F30, 0x6DC9C883 */
viv__rem_pio2_pio2_1  = 1.57079632673412561417e+00, /* 0x3FF921FB, 0x54400000 */
viv__rem_pio2_pio2_1t = 6.07710050650619224932e-11, /* 0x3DD0B461, 0x1A626331 */
viv__rem_pio2_pio2_2  = 6.07710050630396597660e-11, /* 0x3DD0B461, 0x1A600000 */
viv__rem_pio2_pio2_2t = 2.02226624879595063154e-21, /* 0x3BA3198A, 0x2E037073 */
viv__rem_pio2_pio2_3  = 2.02226624871116645580e-21, /* 0x3BA3198A, 0x2E000000 */
viv__rem_pio2_pio2_3t = 8.47842766036889956997e-32; /* 0x397B839A, 0x252049C1 */

/* caller must handle the case when reduction is not needed: |x| ~<= pi/4 */
static int viv__rem_pio2(double x, double *y)
{
	union {double f; uint64_t i;} u = {x};
	double z,w,t,r,fn;
	double tx[3],ty[2];
	uint32_t ix;
	int sign, n, ex, ey, i;

	sign = u.i>>63;
	ix = u.i>>32 & 0x7fffffff;
	if (ix <= 0x400f6a7a) {  /* |x| ~<= 5pi/4 */
		if ((ix & 0xfffff) == 0x921fb)  /* |x| ~= pi/2 or 2pi/2 */
			goto medium;  /* cancellation -- use medium case */
		if (ix <= 0x4002d97c) {  /* |x| ~<= 3pi/4 */
			if (!sign) {
				z = x - viv__rem_pio2_pio2_1;  /* one round good to 85 bits */
				y[0] = z - viv__rem_pio2_pio2_1t;
				y[1] = (z-y[0]) - viv__rem_pio2_pio2_1t;
				return 1;
			} else {
				z = x + viv__rem_pio2_pio2_1;
				y[0] = z + viv__rem_pio2_pio2_1t;
				y[1] = (z-y[0]) + viv__rem_pio2_pio2_1t;
				return -1;
			}
		} else {
			if (!sign) {
				z = x - 2*viv__rem_pio2_pio2_1;
				y[0] = z - 2*viv__rem_pio2_pio2_1t;
				y[1] = (z-y[0]) - 2*viv__rem_pio2_pio2_1t;
				return 2;
			} else {
				z = x + 2*viv__rem_pio2_pio2_1;
				y[0] = z + 2*viv__rem_pio2_pio2_1t;
				y[1] = (z-y[0]) + 2*viv__rem_pio2_pio2_1t;
				return -2;
			}
		}
	}
	if (ix <= 0x401c463b) {  /* |x| ~<= 9pi/4 */
		if (ix <= 0x4015fdbc) {  /* |x| ~<= 7pi/4 */
			if (ix == 0x4012d97c)  /* |x| ~= 3pi/2 */
				goto medium;
			if (!sign) {
				z = x - 3*viv__rem_pio2_pio2_1;
				y[0] = z - 3*viv__rem_pio2_pio2_1t;
				y[1] = (z-y[0]) - 3*viv__rem_pio2_pio2_1t;
				return 3;
			} else {
				z = x + 3*viv__rem_pio2_pio2_1;
				y[0] = z + 3*viv__rem_pio2_pio2_1t;
				y[1] = (z-y[0]) + 3*viv__rem_pio2_pio2_1t;
				return -3;
			}
		} else {
			if (ix == 0x401921fb)  /* |x| ~= 4pi/2 */
				goto medium;
			if (!sign) {
				z = x - 4*viv__rem_pio2_pio2_1;
				y[0] = z - 4*viv__rem_pio2_pio2_1t;
				y[1] = (z-y[0]) - 4*viv__rem_pio2_pio2_1t;
				return 4;
			} else {
				z = x + 4*viv__rem_pio2_pio2_1;
				y[0] = z + 4*viv__rem_pio2_pio2_1t;
				y[1] = (z-y[0]) + 4*viv__rem_pio2_pio2_1t;
				return -4;
			}
		}
	}
	if (ix < 0x413921fb) {  /* |x| ~< 2^20*(pi/2), medium size */
medium:
		/* rint(x/(pi/2)), Assume round-to-nearest. */
		fn = (double)x*viv__rem_pio2_invpio2 + viv__rem_pio2_toint - viv__rem_pio2_toint;
		n = (int32_t)fn;
		r = x - fn*viv__rem_pio2_pio2_1;
		w = fn*viv__rem_pio2_pio2_1t;  /* 1st round, good to 85 bits */
		y[0] = r - w;
		u.f = y[0];
		ey = u.i>>52 & 0x7ff;
		ex = ix>>20;
		if (ex - ey > 16) { /* 2nd round, good to 118 bits */
			t = r;
			w = fn*viv__rem_pio2_pio2_2;
			r = t - w;
			w = fn*viv__rem_pio2_pio2_2t - ((t-r)-w);
			y[0] = r - w;
			u.f = y[0];
			ey = u.i>>52 & 0x7ff;
			if (ex - ey > 49) {  /* 3rd round, good to 151 bits, covers all cases */
				t = r;
				w = fn*viv__rem_pio2_pio2_3;
				r = t - w;
				w = fn*viv__rem_pio2_pio2_3t - ((t-r)-w);
				y[0] = r - w;
			}
		}
		y[1] = (r - y[0]) - w;
		return n;
	}
	/*
	 * all other (large) arguments
	 */
	if (ix >= 0x7ff00000) {  /* x is inf or NaN */
		y[0] = y[1] = x - x;
		return 0;
	}
	/* set z = scalbn(|x|,-ilogb(x)+23) */
	u.f = x;
	u.i &= (uint64_t)-1>>12;
	u.i |= (uint64_t)(0x3ff + 23)<<52;
	z = u.f;
	for (i=0; i < 2; i++) {
		tx[i] = (double)(int32_t)z;
		z     = (z-tx[i])*0x1p24;
	}
	tx[i] = z;
	/* skip zero terms, first term is non-zero */
	while (tx[i] == 0.0)
		i--;
	n = viv__rem_pio2_large(tx,ty,(int)(ix>>20)-(0x3ff+23),i+1,1);
	if (sign) {
		y[0] = -ty[0];
		y[1] = -ty[1];
		return -n;
	}
	y[0] = ty[0];
	y[1] = ty[1];
	return n;
}
#undef EPS

/* ---- musl src/math/__sin.c ---- */
/* origin: FreeBSD /usr/src/lib/msun/src/k_sin.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* __sin( x, y, iy)
 * kernel sin function on ~[-pi/4, pi/4] (except on -0), pi/4 ~ 0.7854
 * Input x is assumed to be bounded by ~pi/4 in magnitude.
 * Input y is the tail of x.
 * Input iy indicates whether y is 0. (if iy=0, y assume to be 0).
 *
 * Algorithm
 *      1. Since sin(-x) = -sin(x), we need only to consider positive x.
 *      2. Callers must return sin(-0) = -0 without calling here since our
 *         odd polynomial is not evaluated in a way that preserves -0.
 *         Callers may do the optimization sin(x) ~ x for tiny x.
 *      3. sin(x) is approximated by a polynomial of degree 13 on
 *         [0,pi/4]
 *                               3            13
 *              sin(x) ~ x + S1*x + ... + S6*x
 *         where
 *
 *      |sin(x)         2     4     6     8     10     12  |     -58
 *      |----- - (1+S1*x +S2*x +S3*x +S4*x +S5*x  +S6*x   )| <= 2
 *      |  x                                               |
 *
 *      4. sin(x+y) = sin(x) + sin'(x')*y
 *                  ~ sin(x) + (1-x*x/2)*y
 *         For better accuracy, let
 *                   3      2      2      2      2
 *              r = x *(S2+x *(S3+x *(S4+x *(S5+x *S6))))
 *         then                   3    2
 *              sin(x) = x + (S1*x + (x *(r-y/2)+y))
 */


static const double
viv__sin_S1  = -1.66666666666666324348e-01, /* 0xBFC55555, 0x55555549 */
viv__sin_S2  =  8.33333333332248946124e-03, /* 0x3F811111, 0x1110F8A6 */
viv__sin_S3  = -1.98412698298579493134e-04, /* 0xBF2A01A0, 0x19C161D5 */
viv__sin_S4  =  2.75573137070700676789e-06, /* 0x3EC71DE3, 0x57B1FE7D */
viv__sin_S5  = -2.50507602534068634195e-08, /* 0xBE5AE5E6, 0x8A2B9CEB */
viv__sin_S6  =  1.58969099521155010221e-10; /* 0x3DE5D93A, 0x5ACFD57C */

static double viv__sin(double x, double y, int iy)
{
	double z,r,v,w;

	z = x*x;
	w = z*z;
	r = viv__sin_S2 + z*(viv__sin_S3 + z*viv__sin_S4) + z*w*(viv__sin_S5 + z*viv__sin_S6);
	v = z*x;
	if (iy == 0)
		return x + v*(viv__sin_S1 + z*r);
	else
		return x - ((z*(0.5*y - v*r) - y) - v*viv__sin_S1);
}

/* ---- musl src/math/__cos.c ---- */
/* origin: FreeBSD /usr/src/lib/msun/src/k_cos.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/*
 * __cos( x,  y )
 * kernel cos function on [-pi/4, pi/4], pi/4 ~ 0.785398164
 * Input x is assumed to be bounded by ~pi/4 in magnitude.
 * Input y is the tail of x.
 *
 * Algorithm
 *      1. Since cos(-x) = cos(x), we need only to consider positive x.
 *      2. if x < 2^-27 (hx<0x3e400000 0), return 1 with inexact if x!=0.
 *      3. cos(x) is approximated by a polynomial of degree 14 on
 *         [0,pi/4]
 *                                       4            14
 *              cos(x) ~ 1 - x*x/2 + C1*x + ... + C6*x
 *         where the remez error is
 *
 *      |              2     4     6     8     10    12     14 |     -58
 *      |cos(x)-(1-.5*x +C1*x +C2*x +C3*x +C4*x +C5*x  +C6*x  )| <= 2
 *      |                                                      |
 *
 *                     4     6     8     10    12     14
 *      4. let r = C1*x +C2*x +C3*x +C4*x +C5*x  +C6*x  , then
 *             cos(x) ~ 1 - x*x/2 + r
 *         since cos(x+y) ~ cos(x) - sin(x)*y
 *                        ~ cos(x) - x*y,
 *         a correction term is necessary in cos(x) and hence
 *              cos(x+y) = 1 - (x*x/2 - (r - x*y))
 *         For better accuracy, rearrange to
 *              cos(x+y) ~ w + (tmp + (r-x*y))
 *         where w = 1 - x*x/2 and tmp is a tiny correction term
 *         (1 - x*x/2 == w + tmp exactly in infinite precision).
 *         The exactness of w + tmp in infinite precision depends on w
 *         and tmp having the same precision as x.  If they have extra
 *         precision due to compiler bugs, then the extra precision is
 *         only good provided it is retained in all terms of the final
 *         expression for cos().  Retention happens in all cases tested
 *         under FreeBSD, so don't pessimize things by forcibly clipping
 *         any extra precision in w.
 */


static const double
viv__cos_C1  =  4.16666666666666019037e-02, /* 0x3FA55555, 0x5555554C */
viv__cos_C2  = -1.38888888888741095749e-03, /* 0xBF56C16C, 0x16C15177 */
viv__cos_C3  =  2.48015872894767294178e-05, /* 0x3EFA01A0, 0x19CB1590 */
viv__cos_C4  = -2.75573143513906633035e-07, /* 0xBE927E4F, 0x809C52AD */
viv__cos_C5  =  2.08757232129817482790e-09, /* 0x3E21EE9E, 0xBDB4B1C4 */
viv__cos_C6  = -1.13596475577881948265e-11; /* 0xBDA8FAE9, 0xBE8838D4 */

static double viv__cos(double x, double y)
{
	double hz,z,r,w;

	z  = x*x;
	w  = z*z;
	r  = z*(viv__cos_C1+z*(viv__cos_C2+z*viv__cos_C3)) + w*w*(viv__cos_C4+z*(viv__cos_C5+z*viv__cos_C6));
	hz = 0.5*z;
	w  = 1.0-hz;
	return w + (((1.0-w)-hz) + (z*r-x*y));
}

/* ---- musl src/math/__tan.c ---- */
/* origin: FreeBSD /usr/src/lib/msun/src/k_tan.c */
/*
 * ====================================================
 * Copyright 2004 Sun Microsystems, Inc.  All Rights Reserved.
 *
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* __tan( x, y, k )
 * kernel tan function on ~[-pi/4, pi/4] (except on -0), pi/4 ~ 0.7854
 * Input x is assumed to be bounded by ~pi/4 in magnitude.
 * Input y is the tail of x.
 * Input odd indicates whether tan (if odd = 0) or -1/tan (if odd = 1) is returned.
 *
 * Algorithm
 *      1. Since tan(-x) = -tan(x), we need only to consider positive x.
 *      2. Callers must return tan(-0) = -0 without calling here since our
 *         odd polynomial is not evaluated in a way that preserves -0.
 *         Callers may do the optimization tan(x) ~ x for tiny x.
 *      3. tan(x) is approximated by a odd polynomial of degree 27 on
 *         [0,0.67434]
 *                               3             27
 *              tan(x) ~ x + T1*x + ... + T13*x
 *         where
 *
 *              |tan(x)         2     4            26   |     -59.2
 *              |----- - (1+T1*x +T2*x +.... +T13*x    )| <= 2
 *              |  x                                    |
 *
 *         Note: tan(x+y) = tan(x) + tan'(x)*y
 *                        ~ tan(x) + (1+x*x)*y
 *         Therefore, for better accuracy in computing tan(x+y), let
 *                   3      2      2       2       2
 *              r = x *(T2+x *(T3+x *(...+x *(T12+x *T13))))
 *         then
 *                                  3    2
 *              tan(x+y) = x + (T1*x + (x *(r+y)+y))
 *
 *      4. For x in [0.67434,pi/4],  let y = pi/4 - x, then
 *              tan(x) = tan(pi/4-y) = (1-tan(y))/(1+tan(y))
 *                     = 1 - 2*(tan(y) - (tan(y)^2)/(1+tan(y)))
 */


static const double viv__tan_T[] = {
             3.33333333333334091986e-01, /* 3FD55555, 55555563 */
             1.33333333333201242699e-01, /* 3FC11111, 1110FE7A */
             5.39682539762260521377e-02, /* 3FABA1BA, 1BB341FE */
             2.18694882948595424599e-02, /* 3F9664F4, 8406D637 */
             8.86323982359930005737e-03, /* 3F8226E3, E96E8493 */
             3.59207910759131235356e-03, /* 3F6D6D22, C9560328 */
             1.45620945432529025516e-03, /* 3F57DBC8, FEE08315 */
             5.88041240820264096874e-04, /* 3F4344D8, F2F26501 */
             2.46463134818469906812e-04, /* 3F3026F7, 1A8D1068 */
             7.81794442939557092300e-05, /* 3F147E88, A03792A6 */
             7.14072491382608190305e-05, /* 3F12B80F, 32F0A7E9 */
            -1.85586374855275456654e-05, /* BEF375CB, DB605373 */
             2.59073051863633712884e-05, /* 3EFB2A70, 74BF7AD4 */
},
viv__tan_pio4 =       7.85398163397448278999e-01, /* 3FE921FB, 54442D18 */
viv__tan_pio4lo =     3.06161699786838301793e-17; /* 3C81A626, 33145C07 */

static double viv__tan(double x, double y, int odd)
{
	double z, r, v, w, s, a;
	double w0, a0;
	uint32_t hx;
	int big, sign;

	GET_HIGH_WORD(hx,x);
	big = (hx&0x7fffffff) >= 0x3FE59428; /* |x| >= 0.6744 */
	if (big) {
		sign = hx>>31;
		if (sign) {
			x = -x;
			y = -y;
		}
		x = (viv__tan_pio4 - x) + (viv__tan_pio4lo - y);
		y = 0.0;
	}
	z = x * x;
	w = z * z;
	/*
	 * Break x^5*(T[1]+x^2*T[2]+...) into
	 * x^5(T[1]+x^4*T[3]+...+x^20*T[11]) +
	 * x^5(x^2*(T[2]+x^4*T[4]+...+x^22*[T12]))
	 */
	r = viv__tan_T[1] + w*(viv__tan_T[3] + w*(viv__tan_T[5] + w*(viv__tan_T[7] + w*(viv__tan_T[9] + w*viv__tan_T[11]))));
	v = z*(viv__tan_T[2] + w*(viv__tan_T[4] + w*(viv__tan_T[6] + w*(viv__tan_T[8] + w*(viv__tan_T[10] + w*viv__tan_T[12])))));
	s = z * x;
	r = y + z*(s*(r + v) + y) + s*viv__tan_T[0];
	w = x + r;
	if (big) {
		s = 1 - 2*odd;
		v = s - 2.0 * (x + (r - w*w/(w + s)));
		return sign ? -v : v;
	}
	if (!odd)
		return w;
	/* -1.0/(x+r) has up to 2ulp error, so compute it accurately */
	w0 = w;
	SET_LOW_WORD(w0, 0);
	v = r - (w0 - x);       /* w0+v = r+x */
	a0 = a = -1.0 / w;
	SET_LOW_WORD(a0, 0);
	return a0 + a*(1.0 + a0*w0 + a0*v);
}

/* ---- musl src/math/sin.c ---- */
/* origin: FreeBSD /usr/src/lib/msun/src/s_sin.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* sin(x)
 * Return sine function of x.
 *
 * kernel function:
 *      __sin            ... sine function on [-pi/4,pi/4]
 *      __cos            ... cose function on [-pi/4,pi/4]
 *      __rem_pio2       ... argument reduction routine
 *
 * Method.
 *      Let S,C and T denote the sin, cos and tan respectively on
 *      [-PI/4, +PI/4]. Reduce the argument x to y1+y2 = x-k*pi/2
 *      in [-pi/4 , +pi/4], and let n = k mod 4.
 *      We have
 *
 *          n        sin(x)      cos(x)        tan(x)
 *     ----------------------------------------------------------
 *          0          S           C             T
 *          1          C          -S            -1/T
 *          2         -S          -C             T
 *          3         -C           S            -1/T
 *     ----------------------------------------------------------
 *
 * Special cases:
 *      Let trig be any of sin, cos, or tan.
 *      trig(+-INF)  is NaN, with signals;
 *      trig(NaN)    is that NaN;
 *
 * Accuracy:
 *      TRIG(x) returns trig(x) nearly rounded
 */


static double viv_sin(double x)
{
	double y[2];
	uint32_t ix;
	unsigned n;

	/* High word of x. */
	GET_HIGH_WORD(ix, x);
	ix &= 0x7fffffff;

	/* |x| ~< pi/4 */
	if (ix <= 0x3fe921fb) {
		if (ix < 0x3e500000) {  /* |x| < 2**-26 */
			/* raise inexact if x != 0 and underflow if subnormal*/
			FORCE_EVAL(ix < 0x00100000 ? x/0x1p120f : x+0x1p120f);
			return x;
		}
		return viv__sin(x, 0.0, 0);
	}

	/* sin(Inf or NaN) is NaN */
	if (ix >= 0x7ff00000)
		return x - x;

	/* argument reduction needed */
	n = viv__rem_pio2(x, y);
	switch (n&3) {
	case 0: return  viv__sin(y[0], y[1], 1);
	case 1: return  viv__cos(y[0], y[1]);
	case 2: return -viv__sin(y[0], y[1], 1);
	default:
		return -viv__cos(y[0], y[1]);
	}
}

/* ---- musl src/math/cos.c ---- */
/* origin: FreeBSD /usr/src/lib/msun/src/s_cos.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* cos(x)
 * Return cosine function of x.
 *
 * kernel function:
 *      __sin           ... sine function on [-pi/4,pi/4]
 *      __cos           ... cosine function on [-pi/4,pi/4]
 *      __rem_pio2      ... argument reduction routine
 *
 * Method.
 *      Let S,C and T denote the sin, cos and tan respectively on
 *      [-PI/4, +PI/4]. Reduce the argument x to y1+y2 = x-k*pi/2
 *      in [-pi/4 , +pi/4], and let n = k mod 4.
 *      We have
 *
 *          n        sin(x)      cos(x)        tan(x)
 *     ----------------------------------------------------------
 *          0          S           C             T
 *          1          C          -S            -1/T
 *          2         -S          -C             T
 *          3         -C           S            -1/T
 *     ----------------------------------------------------------
 *
 * Special cases:
 *      Let trig be any of sin, cos, or tan.
 *      trig(+-INF)  is NaN, with signals;
 *      trig(NaN)    is that NaN;
 *
 * Accuracy:
 *      TRIG(x) returns trig(x) nearly rounded
 */


static double viv_cos(double x)
{
	double y[2];
	uint32_t ix;
	unsigned n;

	GET_HIGH_WORD(ix, x);
	ix &= 0x7fffffff;

	/* |x| ~< pi/4 */
	if (ix <= 0x3fe921fb) {
		if (ix < 0x3e46a09e) {  /* |x| < 2**-27 * sqrt(2) */
			/* raise inexact if x!=0 */
			FORCE_EVAL(x + 0x1p120f);
			return 1.0;
		}
		return viv__cos(x, 0);
	}

	/* cos(Inf or NaN) is NaN */
	if (ix >= 0x7ff00000)
		return x-x;

	/* argument reduction */
	n = viv__rem_pio2(x, y);
	switch (n&3) {
	case 0: return  viv__cos(y[0], y[1]);
	case 1: return -viv__sin(y[0], y[1], 1);
	case 2: return -viv__cos(y[0], y[1]);
	default:
		return  viv__sin(y[0], y[1], 1);
	}
}

/* ---- musl src/math/tan.c ---- */
/* origin: FreeBSD /usr/src/lib/msun/src/s_tan.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* tan(x)
 * Return tangent function of x.
 *
 * kernel function:
 *      __tan           ... tangent function on [-pi/4,pi/4]
 *      __rem_pio2      ... argument reduction routine
 *
 * Method.
 *      Let S,C and T denote the sin, cos and tan respectively on
 *      [-PI/4, +PI/4]. Reduce the argument x to y1+y2 = x-k*pi/2
 *      in [-pi/4 , +pi/4], and let n = k mod 4.
 *      We have
 *
 *          n        sin(x)      cos(x)        tan(x)
 *     ----------------------------------------------------------
 *          0          S           C             T
 *          1          C          -S            -1/T
 *          2         -S          -C             T
 *          3         -C           S            -1/T
 *     ----------------------------------------------------------
 *
 * Special cases:
 *      Let trig be any of sin, cos, or tan.
 *      trig(+-INF)  is NaN, with signals;
 *      trig(NaN)    is that NaN;
 *
 * Accuracy:
 *      TRIG(x) returns trig(x) nearly rounded
 */


static double viv_tan(double x)
{
	double y[2];
	uint32_t ix;
	unsigned n;

	GET_HIGH_WORD(ix, x);
	ix &= 0x7fffffff;

	/* |x| ~< pi/4 */
	if (ix <= 0x3fe921fb) {
		if (ix < 0x3e400000) { /* |x| < 2**-27 */
			/* raise inexact if x!=0 and underflow if subnormal */
			FORCE_EVAL(ix < 0x00100000 ? x/0x1p120f : x+0x1p120f);
			return x;
		}
		return viv__tan(x, 0.0, 0);
	}

	/* tan(Inf or NaN) is NaN */
	if (ix >= 0x7ff00000)
		return x - x;

	/* argument reduction */
	n = viv__rem_pio2(x, y);
	return viv__tan(y[0], y[1], n&1);
}

/* ---- musl src/math/exp.c ---- */
/* origin: FreeBSD /usr/src/lib/msun/src/e_exp.c */
/*
 * ====================================================
 * Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
 *
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* exp(x)
 * Returns the exponential of x.
 *
 * Method
 *   1. Argument reduction:
 *      Reduce x to an r so that |r| <= 0.5*ln2 ~ 0.34658.
 *      Given x, find r and integer k such that
 *
 *               x = k*ln2 + r,  |r| <= 0.5*ln2.
 *
 *      Here r will be represented as r = hi-lo for better
 *      accuracy.
 *
 *   2. Approximation of exp(r) by a special rational function on
 *      the interval [0,0.34658]:
 *      Write
 *          R(r**2) = r*(exp(r)+1)/(exp(r)-1) = 2 + r*r/6 - r**4/360 + ...
 *      We use a special Remez algorithm on [0,0.34658] to generate
 *      a polynomial of degree 5 to approximate R. The maximum error
 *      of this polynomial approximation is bounded by 2**-59. In
 *      other words,
 *          R(z) ~ 2.0 + P1*z + P2*z**2 + P3*z**3 + P4*z**4 + P5*z**5
 *      (where z=r*r, and the values of P1 to P5 are listed below)
 *      and
 *          |                  5          |     -59
 *          | 2.0+P1*z+...+P5*z   -  R(z) | <= 2
 *          |                             |
 *      The computation of exp(r) thus becomes
 *                              2*r
 *              exp(r) = 1 + ----------
 *                            R(r) - r
 *                                 r*c(r)
 *                     = 1 + r + ----------- (for better accuracy)
 *                                2 - c(r)
 *      where
 *                              2       4             10
 *              c(r) = r - (P1*r  + P2*r  + ... + P5*r   ).
 *
 *   3. Scale back to obtain exp(x):
 *      From step 1, we have
 *         exp(x) = 2^k * exp(r)
 *
 * Special cases:
 *      exp(INF) is INF, exp(NaN) is NaN;
 *      exp(-INF) is 0, and
 *      for finite argument, only exp(0)=1 is exact.
 *
 * Accuracy:
 *      according to an error analysis, the error is always less than
 *      1 ulp (unit in the last place).
 *
 * Misc. info.
 *      For IEEE double
 *          if x >  709.782712893383973096 then exp(x) overflows
 *          if x < -745.133219101941108420 then exp(x) underflows
 */


static const double
viv_exp_half[2] = {0.5,-0.5},
viv_exp_ln2hi = 6.93147180369123816490e-01, /* 0x3fe62e42, 0xfee00000 */
viv_exp_ln2lo = 1.90821492927058770002e-10, /* 0x3dea39ef, 0x35793c76 */
viv_exp_invln2 = 1.44269504088896338700e+00, /* 0x3ff71547, 0x652b82fe */
viv_exp_P1   =  1.66666666666666019037e-01, /* 0x3FC55555, 0x5555553E */
viv_exp_P2   = -2.77777777770155933842e-03, /* 0xBF66C16C, 0x16BEBD93 */
viv_exp_P3   =  6.61375632143793436117e-05, /* 0x3F11566A, 0xAF25DE2C */
viv_exp_P4   = -1.65339022054652515390e-06, /* 0xBEBBBD41, 0xC5D26BF1 */
viv_exp_P5   =  4.13813679705723846039e-08; /* 0x3E663769, 0x72BEA4D0 */

static double viv_exp(double x)
{
	double hi, lo, c, xx, y;
	int k, sign;
	uint32_t hx;

	GET_HIGH_WORD(hx, x);
	sign = hx>>31;
	hx &= 0x7fffffff;  /* high word of |x| */

	/* special cases */
	if (hx >= 0x4086232b) {  /* if |x| >= 708.39... */
		if (isnan(x))
			return x;
		if (x > 709.782712893383973096) {
			/* overflow if x!=inf */
			x *= 0x1p1023;
			return x;
		}
		if (x < -708.39641853226410622) {
			/* underflow if x!=-inf */
			FORCE_EVAL((float)(-0x1p-149/x));
			if (x < -745.13321910194110842)
				return 0;
		}
	}

	/* argument reduction */
	if (hx > 0x3fd62e42) {  /* if |x| > 0.5 ln2 */
		if (hx >= 0x3ff0a2b2)  /* if |x| >= 1.5 ln2 */
			k = (int)(viv_exp_invln2*x + viv_exp_half[sign]);
		else
			k = 1 - sign - sign;
		hi = x - k*viv_exp_ln2hi;  /* k*ln2hi is exact here */
		lo = k*viv_exp_ln2lo;
		x = hi - lo;
	} else if (hx > 0x3e300000)  {  /* if |x| > 2**-28 */
		k = 0;
		hi = x;
		lo = 0;
	} else {
		/* inexact if x!=0 */
		FORCE_EVAL(0x1p1023 + x);
		return 1 + x;
	}

	/* x is now in primary range */
	xx = x*x;
	c = x - xx*(viv_exp_P1+xx*(viv_exp_P2+xx*(viv_exp_P3+xx*(viv_exp_P4+xx*viv_exp_P5))));
	y = 1 + (x*c/(2-c) - lo + hi);
	if (k == 0)
		return y;
	return viv_scalbn(y, k);
}

/* ---- musl src/math/expm1.c ---- */
/* origin: FreeBSD /usr/src/lib/msun/src/s_expm1.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* expm1(x)
 * Returns exp(x)-1, the exponential of x minus 1.
 *
 * Method
 *   1. Argument reduction:
 *      Given x, find r and integer k such that
 *
 *               x = k*ln2 + r,  |r| <= 0.5*ln2 ~ 0.34658
 *
 *      Here a correction term c will be computed to compensate
 *      the error in r when rounded to a floating-point number.
 *
 *   2. Approximating expm1(r) by a special rational function on
 *      the interval [0,0.34658]:
 *      Since
 *          r*(exp(r)+1)/(exp(r)-1) = 2+ r^2/6 - r^4/360 + ...
 *      we define R1(r*r) by
 *          r*(exp(r)+1)/(exp(r)-1) = 2+ r^2/6 * R1(r*r)
 *      That is,
 *          R1(r**2) = 6/r *((exp(r)+1)/(exp(r)-1) - 2/r)
 *                   = 6/r * ( 1 + 2.0*(1/(exp(r)-1) - 1/r))
 *                   = 1 - r^2/60 + r^4/2520 - r^6/100800 + ...
 *      We use a special Remez algorithm on [0,0.347] to generate
 *      a polynomial of degree 5 in r*r to approximate R1. The
 *      maximum error of this polynomial approximation is bounded
 *      by 2**-61. In other words,
 *          R1(z) ~ 1.0 + Q1*z + Q2*z**2 + Q3*z**3 + Q4*z**4 + Q5*z**5
 *      where   Q1  =  -1.6666666666666567384E-2,
 *              Q2  =   3.9682539681370365873E-4,
 *              Q3  =  -9.9206344733435987357E-6,
 *              Q4  =   2.5051361420808517002E-7,
 *              Q5  =  -6.2843505682382617102E-9;
 *              z   =  r*r,
 *      with error bounded by
 *          |                  5           |     -61
 *          | 1.0+Q1*z+...+Q5*z   -  R1(z) | <= 2
 *          |                              |
 *
 *      expm1(r) = exp(r)-1 is then computed by the following
 *      specific way which minimize the accumulation rounding error:
 *                             2     3
 *                            r     r    [ 3 - (R1 + R1*r/2)  ]
 *            expm1(r) = r + --- + --- * [--------------------]
 *                            2     2    [ 6 - r*(3 - R1*r/2) ]
 *
 *      To compensate the error in the argument reduction, we use
 *              expm1(r+c) = expm1(r) + c + expm1(r)*c
 *                         ~ expm1(r) + c + r*c
 *      Thus c+r*c will be added in as the correction terms for
 *      expm1(r+c). Now rearrange the term to avoid optimization
 *      screw up:
 *                      (      2                                    2 )
 *                      ({  ( r    [ R1 -  (3 - R1*r/2) ]  )  }    r  )
 *       expm1(r+c)~r - ({r*(--- * [--------------------]-c)-c} - --- )
 *                      ({  ( 2    [ 6 - r*(3 - R1*r/2) ]  )  }    2  )
 *                      (                                             )
 *
 *                 = r - E
 *   3. Scale back to obtain expm1(x):
 *      From step 1, we have
 *         expm1(x) = either 2^k*[expm1(r)+1] - 1
 *                  = or     2^k*[expm1(r) + (1-2^-k)]
 *   4. Implementation notes:
 *      (A). To save one multiplication, we scale the coefficient Qi
 *           to Qi*2^i, and replace z by (x^2)/2.
 *      (B). To achieve maximum accuracy, we compute expm1(x) by
 *        (i)   if x < -56*ln2, return -1.0, (raise inexact if x!=inf)
 *        (ii)  if k=0, return r-E
 *        (iii) if k=-1, return 0.5*(r-E)-0.5
 *        (iv)  if k=1 if r < -0.25, return 2*((r+0.5)- E)
 *                     else          return  1.0+2.0*(r-E);
 *        (v)   if (k<-2||k>56) return 2^k(1-(E-r)) - 1 (or exp(x)-1)
 *        (vi)  if k <= 20, return 2^k((1-2^-k)-(E-r)), else
 *        (vii) return 2^k(1-((E+2^-k)-r))
 *
 * Special cases:
 *      expm1(INF) is INF, expm1(NaN) is NaN;
 *      expm1(-INF) is -1, and
 *      for finite argument, only expm1(0)=0 is exact.
 *
 * Accuracy:
 *      according to an error analysis, the error is always less than
 *      1 ulp (unit in the last place).
 *
 * Misc. info.
 *      For IEEE double
 *          if x >  7.09782712893383973096e+02 then expm1(x) overflow
 *
 * Constants:
 * The hexadecimal values are the intended ones for the following
 * constants. The decimal values may be used, provided that the
 * compiler will convert from decimal to binary accurately enough
 * to produce the hexadecimal values shown.
 */


static const double
viv_expm1_o_threshold = 7.09782712893383973096e+02, /* 0x40862E42, 0xFEFA39EF */
viv_expm1_ln2_hi      = 6.93147180369123816490e-01, /* 0x3fe62e42, 0xfee00000 */
viv_expm1_ln2_lo      = 1.90821492927058770002e-10, /* 0x3dea39ef, 0x35793c76 */
viv_expm1_invln2      = 1.44269504088896338700e+00, /* 0x3ff71547, 0x652b82fe */
/* Scaled Q's: Qn_here = 2**n * Qn_above, for R(2*z) where z = hxs = x*x/2: */
viv_expm1_Q1 = -3.33333333333331316428e-02, /* BFA11111 111110F4 */
viv_expm1_Q2 =  1.58730158725481460165e-03, /* 3F5A01A0 19FE5585 */
viv_expm1_Q3 = -7.93650757867487942473e-05, /* BF14CE19 9EAADBB7 */
viv_expm1_Q4 =  4.00821782732936239552e-06, /* 3ED0CFCA 86E65239 */
viv_expm1_Q5 = -2.01099218183624371326e-07; /* BE8AFDB7 6E09C32D */

static double viv_expm1(double x)
{
	double y,hi,lo,c,t,e,hxs,hfx,r1,twopk;
	union {double f; uint64_t i;} u = {x};
	uint32_t hx = u.i>>32 & 0x7fffffff;
	int k, sign = u.i>>63;

	/* filter out huge and non-finite argument */
	if (hx >= 0x4043687A) {  /* if |x|>=56*ln2 */
		if (isnan(x))
			return x;
		if (sign)
			return -1;
		if (x > viv_expm1_o_threshold) {
			x *= 0x1p1023;
			return x;
		}
	}

	/* argument reduction */
	if (hx > 0x3fd62e42) {  /* if  |x| > 0.5 ln2 */
		if (hx < 0x3FF0A2B2) {  /* and |x| < 1.5 ln2 */
			if (!sign) {
				hi = x - viv_expm1_ln2_hi;
				lo = viv_expm1_ln2_lo;
				k =  1;
			} else {
				hi = x + viv_expm1_ln2_hi;
				lo = -viv_expm1_ln2_lo;
				k = -1;
			}
		} else {
			k  = viv_expm1_invln2*x + (sign ? -0.5 : 0.5);
			t  = k;
			hi = x - t*viv_expm1_ln2_hi;  /* t*ln2_hi is exact here */
			lo = t*viv_expm1_ln2_lo;
		}
		x = hi-lo;
		c = (hi-x)-lo;
	} else if (hx < 0x3c900000) {  /* |x| < 2**-54, return x */
		if (hx < 0x00100000)
			FORCE_EVAL((float)x);
		return x;
	} else
		k = 0;

	/* x is now in primary range */
	hfx = 0.5*x;
	hxs = x*hfx;
	r1 = 1.0+hxs*(viv_expm1_Q1+hxs*(viv_expm1_Q2+hxs*(viv_expm1_Q3+hxs*(viv_expm1_Q4+hxs*viv_expm1_Q5))));
	t  = 3.0-r1*hfx;
	e  = hxs*((r1-t)/(6.0 - x*t));
	if (k == 0)   /* c is 0 */
		return x - (x*e-hxs);
	e  = x*(e-c) - c;
	e -= hxs;
	/* exp(x) ~ 2^k (x_reduced - e + 1) */
	if (k == -1)
		return 0.5*(x-e) - 0.5;
	if (k == 1) {
		if (x < -0.25)
			return -2.0*(e-(x+0.5));
		return 1.0+2.0*(x-e);
	}
	u.i = (uint64_t)(0x3ff + k)<<52;  /* 2^k */
	twopk = u.f;
	if (k < 0 || k > 56) {  /* suffice to return exp(x)-1 */
		y = x - e + 1.0;
		if (k == 1024)
			y = y*2.0*0x1p1023;
		else
			y = y*twopk;
		return y - 1.0;
	}
	u.i = (uint64_t)(0x3ff - k)<<52;  /* 2^-k */
	if (k < 20)
		y = (x-e+(1-u.f))*twopk;
	else
		y = (x-(e+u.f)+1)*twopk;
	return y;
}

/* ---- musl src/math/__expo2.c ---- */
/* k is such that k*ln2 has minimal relative error and x - kln2 > log(DBL_MIN) */
static const int viv__expo2_k = 2043;
static const double viv__expo2_kln2 = 0x1.62066151add8bp+10;

/* exp(x)/2 for x >= log(DBL_MAX), slightly better than 0.5*exp(x/2)*exp(x/2) */
static double viv__expo2(double x)
{
	double scale;

	/* note that k is odd and scale*scale overflows */
	INSERT_WORDS(scale, (uint32_t)(0x3ff + viv__expo2_k/2) << 20, 0);
	/* exp(x - k ln2) * 2**(k-1) */
	return viv_exp(x - viv__expo2_kln2) * scale * scale;
}

/* ---- musl src/math/log.c ---- */
/* origin: FreeBSD /usr/src/lib/msun/src/e_log.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* log(x)
 * Return the logarithm of x
 *
 * Method :
 *   1. Argument Reduction: find k and f such that
 *                      x = 2^k * (1+f),
 *         where  sqrt(2)/2 < 1+f < sqrt(2) .
 *
 *   2. Approximation of log(1+f).
 *      Let s = f/(2+f) ; based on log(1+f) = log(1+s) - log(1-s)
 *               = 2s + 2/3 s**3 + 2/5 s**5 + .....,
 *               = 2s + s*R
 *      We use a special Remez algorithm on [0,0.1716] to generate
 *      a polynomial of degree 14 to approximate R The maximum error
 *      of this polynomial approximation is bounded by 2**-58.45. In
 *      other words,
 *                      2      4      6      8      10      12      14
 *          R(z) ~ Lg1*s +Lg2*s +Lg3*s +Lg4*s +Lg5*s  +Lg6*s  +Lg7*s
 *      (the values of Lg1 to Lg7 are listed in the program)
 *      and
 *          |      2          14          |     -58.45
 *          | Lg1*s +...+Lg7*s    -  R(z) | <= 2
 *          |                             |
 *      Note that 2s = f - s*f = f - hfsq + s*hfsq, where hfsq = f*f/2.
 *      In order to guarantee error in log below 1ulp, we compute log
 *      by
 *              log(1+f) = f - s*(f - R)        (if f is not too large)
 *              log(1+f) = f - (hfsq - s*(hfsq+R)).     (better accuracy)
 *
 *      3. Finally,  log(x) = k*ln2 + log(1+f).
 *                          = k*ln2_hi+(f-(hfsq-(s*(hfsq+R)+k*ln2_lo)))
 *         Here ln2 is split into two floating point number:
 *                      ln2_hi + ln2_lo,
 *         where n*ln2_hi is always exact for |n| < 2000.
 *
 * Special cases:
 *      log(x) is NaN with signal if x < 0 (including -INF) ;
 *      log(+INF) is +INF; log(0) is -INF with signal;
 *      log(NaN) is that NaN with no signal.
 *
 * Accuracy:
 *      according to an error analysis, the error is always less than
 *      1 ulp (unit in the last place).
 *
 * Constants:
 * The hexadecimal values are the intended ones for the following
 * constants. The decimal values may be used, provided that the
 * compiler will convert from decimal to binary accurately enough
 * to produce the hexadecimal values shown.
 */


static const double
viv_log_ln2_hi = 6.93147180369123816490e-01,  /* 3fe62e42 fee00000 */
viv_log_ln2_lo = 1.90821492927058770002e-10,  /* 3dea39ef 35793c76 */
viv_log_Lg1 = 6.666666666666735130e-01,  /* 3FE55555 55555593 */
viv_log_Lg2 = 3.999999999940941908e-01,  /* 3FD99999 9997FA04 */
viv_log_Lg3 = 2.857142874366239149e-01,  /* 3FD24924 94229359 */
viv_log_Lg4 = 2.222219843214978396e-01,  /* 3FCC71C5 1D8E78AF */
viv_log_Lg5 = 1.818357216161805012e-01,  /* 3FC74664 96CB03DE */
viv_log_Lg6 = 1.531383769920937332e-01,  /* 3FC39A09 D078C69F */
viv_log_Lg7 = 1.479819860511658591e-01;  /* 3FC2F112 DF3E5244 */

static double viv_log(double x)
{
	union {double f; uint64_t i;} u = {x};
	double hfsq,f,s,z,R,w,t1,t2,dk;
	uint32_t hx;
	int k;

	hx = u.i>>32;
	k = 0;
	if (hx < 0x00100000 || hx>>31) {
		if (u.i<<1 == 0)
			return -1/(x*x);  /* log(+-0)=-inf */
		if (hx>>31)
			return (x-x)/0.0; /* log(-#) = NaN */
		/* subnormal number, scale x up */
		k -= 54;
		x *= 0x1p54;
		u.f = x;
		hx = u.i>>32;
	} else if (hx >= 0x7ff00000) {
		return x;
	} else if (hx == 0x3ff00000 && u.i<<32 == 0)
		return 0;

	/* reduce x into [sqrt(2)/2, sqrt(2)] */
	hx += 0x3ff00000 - 0x3fe6a09e;
	k += (int)(hx>>20) - 0x3ff;
	hx = (hx&0x000fffff) + 0x3fe6a09e;
	u.i = (uint64_t)hx<<32 | (u.i&0xffffffff);
	x = u.f;

	f = x - 1.0;
	hfsq = 0.5*f*f;
	s = f/(2.0+f);
	z = s*s;
	w = z*z;
	t1 = w*(viv_log_Lg2+w*(viv_log_Lg4+w*viv_log_Lg6));
	t2 = z*(viv_log_Lg1+w*(viv_log_Lg3+w*(viv_log_Lg5+w*viv_log_Lg7)));
	R = t2 + t1;
	dk = k;
	return s*(hfsq+R) + dk*viv_log_ln2_lo - hfsq + f + dk*viv_log_ln2_hi;
}

/* ---- musl src/math/log10.c ---- */
/* origin: FreeBSD /usr/src/lib/msun/src/e_log10.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/*
 * Return the base 10 logarithm of x.  See log.c for most comments.
 *
 * Reduce x to 2^k (1+f) and calculate r = log(1+f) - f + f*f/2
 * as in log.c, then combine and scale in extra precision:
 *    log10(x) = (f - f*f/2 + r)/log(10) + k*log10(2)
 */


static const double
viv_log10_ivln10hi  = 4.34294481878168880939e-01, /* 0x3fdbcb7b, 0x15200000 */
viv_log10_ivln10lo  = 2.50829467116452752298e-11, /* 0x3dbb9438, 0xca9aadd5 */
viv_log10_log10_2hi = 3.01029995663611771306e-01, /* 0x3FD34413, 0x509F6000 */
viv_log10_log10_2lo = 3.69423907715893078616e-13, /* 0x3D59FEF3, 0x11F12B36 */
viv_log10_Lg1 = 6.666666666666735130e-01,  /* 3FE55555 55555593 */
viv_log10_Lg2 = 3.999999999940941908e-01,  /* 3FD99999 9997FA04 */
viv_log10_Lg3 = 2.857142874366239149e-01,  /* 3FD24924 94229359 */
viv_log10_Lg4 = 2.222219843214978396e-01,  /* 3FCC71C5 1D8E78AF */
viv_log10_Lg5 = 1.818357216161805012e-01,  /* 3FC74664 96CB03DE */
viv_log10_Lg6 = 1.531383769920937332e-01,  /* 3FC39A09 D078C69F */
viv_log10_Lg7 = 1.479819860511658591e-01;  /* 3FC2F112 DF3E5244 */

static double viv_log10(double x)
{
	union {double f; uint64_t i;} u = {x};
	double hfsq,f,s,z,R,w,t1,t2,dk,y,hi,lo,val_hi,val_lo;
	uint32_t hx;
	int k;

	hx = u.i>>32;
	k = 0;
	if (hx < 0x00100000 || hx>>31) {
		if (u.i<<1 == 0)
			return -1/(x*x);  /* log(+-0)=-inf */
		if (hx>>31)
			return (x-x)/0.0; /* log(-#) = NaN */
		/* subnormal number, scale x up */
		k -= 54;
		x *= 0x1p54;
		u.f = x;
		hx = u.i>>32;
	} else if (hx >= 0x7ff00000) {
		return x;
	} else if (hx == 0x3ff00000 && u.i<<32 == 0)
		return 0;

	/* reduce x into [sqrt(2)/2, sqrt(2)] */
	hx += 0x3ff00000 - 0x3fe6a09e;
	k += (int)(hx>>20) - 0x3ff;
	hx = (hx&0x000fffff) + 0x3fe6a09e;
	u.i = (uint64_t)hx<<32 | (u.i&0xffffffff);
	x = u.f;

	f = x - 1.0;
	hfsq = 0.5*f*f;
	s = f/(2.0+f);
	z = s*s;
	w = z*z;
	t1 = w*(viv_log10_Lg2+w*(viv_log10_Lg4+w*viv_log10_Lg6));
	t2 = z*(viv_log10_Lg1+w*(viv_log10_Lg3+w*(viv_log10_Lg5+w*viv_log10_Lg7)));
	R = t2 + t1;

	/* See log2.c for details. */
	/* hi+lo = f - hfsq + s*(hfsq+R) ~ log(1+f) */
	hi = f - hfsq;
	u.f = hi;
	u.i &= (uint64_t)-1<<32;
	hi = u.f;
	lo = f - hi - hfsq + s*(hfsq+R);

	/* val_hi+val_lo ~ log10(1+f) + k*log10(2) */
	val_hi = hi*viv_log10_ivln10hi;
	dk = k;
	y = dk*viv_log10_log10_2hi;
	val_lo = dk*viv_log10_log10_2lo + (lo+hi)*viv_log10_ivln10lo + lo*viv_log10_ivln10hi;

	/*
	 * Extra precision in for adding y is not strictly needed
	 * since there is no very large cancellation near x = sqrt(2) or
	 * x = 1/sqrt(2), but we do it anyway since it costs little on CPUs
	 * with some parallelism and it reduces the error for many args.
	 */
	w = y + val_hi;
	val_lo += (y - w) + val_hi;
	val_hi = w;

	return val_lo + val_hi;
}

/* ---- musl src/math/pow.c ---- */
/* origin: FreeBSD /usr/src/lib/msun/src/e_pow.c */
/*
 * ====================================================
 * Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
 *
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* pow(x,y) return x**y
 *
 *                    n
 * Method:  Let x =  2   * (1+f)
 *      1. Compute and return log2(x) in two pieces:
 *              log2(x) = w1 + w2,
 *         where w1 has 53-24 = 29 bit trailing zeros.
 *      2. Perform y*log2(x) = n+y' by simulating muti-precision
 *         arithmetic, where |y'|<=0.5.
 *      3. Return x**y = 2**n*exp(y'*log2)
 *
 * Special cases:
 *      1.  (anything) ** 0  is 1
 *      2.  1 ** (anything)  is 1
 *      3.  (anything except 1) ** NAN is NAN
 *      4.  NAN ** (anything except 0) is NAN
 *      5.  +-(|x| > 1) **  +INF is +INF
 *      6.  +-(|x| > 1) **  -INF is +0
 *      7.  +-(|x| < 1) **  +INF is +0
 *      8.  +-(|x| < 1) **  -INF is +INF
 *      9.  -1          ** +-INF is 1
 *      10. +0 ** (+anything except 0, NAN)               is +0
 *      11. -0 ** (+anything except 0, NAN, odd integer)  is +0
 *      12. +0 ** (-anything except 0, NAN)               is +INF, raise divbyzero
 *      13. -0 ** (-anything except 0, NAN, odd integer)  is +INF, raise divbyzero
 *      14. -0 ** (+odd integer) is -0
 *      15. -0 ** (-odd integer) is -INF, raise divbyzero
 *      16. +INF ** (+anything except 0,NAN) is +INF
 *      17. +INF ** (-anything except 0,NAN) is +0
 *      18. -INF ** (+odd integer) is -INF
 *      19. -INF ** (anything) = -0 ** (-anything), (anything except odd integer)
 *      20. (anything) ** 1 is (anything)
 *      21. (anything) ** -1 is 1/(anything)
 *      22. (-anything) ** (integer) is (-1)**(integer)*(+anything**integer)
 *      23. (-anything except 0 and inf) ** (non-integer) is NAN
 *
 * Accuracy:
 *      pow(x,y) returns x**y nearly rounded. In particular
 *                      pow(integer,integer)
 *      always returns the correct integer provided it is
 *      representable.
 *
 * Constants :
 * The hexadecimal values are the intended ones for the following
 * constants. The decimal values may be used, provided that the
 * compiler will convert from decimal to binary accurately enough
 * to produce the hexadecimal values shown.
 */


static const double
viv_pow_bp[]   = {1.0, 1.5,},
viv_pow_dp_h[] = { 0.0, 5.84962487220764160156e-01,}, /* 0x3FE2B803, 0x40000000 */
viv_pow_dp_l[] = { 0.0, 1.35003920212974897128e-08,}, /* 0x3E4CFDEB, 0x43CFD006 */
viv_pow_two53  =  9007199254740992.0, /* 0x43400000, 0x00000000 */
viv_pow_huge   =  1.0e300,
viv_pow_tiny   =  1.0e-300,
/* poly coefs for (3/2)*(log(x)-2s-2/3*s**3 */
viv_pow_L1 =  5.99999999999994648725e-01, /* 0x3FE33333, 0x33333303 */
viv_pow_L2 =  4.28571428578550184252e-01, /* 0x3FDB6DB6, 0xDB6FABFF */
viv_pow_L3 =  3.33333329818377432918e-01, /* 0x3FD55555, 0x518F264D */
viv_pow_L4 =  2.72728123808534006489e-01, /* 0x3FD17460, 0xA91D4101 */
viv_pow_L5 =  2.30660745775561754067e-01, /* 0x3FCD864A, 0x93C9DB65 */
viv_pow_L6 =  2.06975017800338417784e-01, /* 0x3FCA7E28, 0x4A454EEF */
viv_pow_P1 =  1.66666666666666019037e-01, /* 0x3FC55555, 0x5555553E */
viv_pow_P2 = -2.77777777770155933842e-03, /* 0xBF66C16C, 0x16BEBD93 */
viv_pow_P3 =  6.61375632143793436117e-05, /* 0x3F11566A, 0xAF25DE2C */
viv_pow_P4 = -1.65339022054652515390e-06, /* 0xBEBBBD41, 0xC5D26BF1 */
viv_pow_P5 =  4.13813679705723846039e-08, /* 0x3E663769, 0x72BEA4D0 */
viv_pow_lg2     =  6.93147180559945286227e-01, /* 0x3FE62E42, 0xFEFA39EF */
viv_pow_lg2_h   =  6.93147182464599609375e-01, /* 0x3FE62E43, 0x00000000 */
viv_pow_lg2_l   = -1.90465429995776804525e-09, /* 0xBE205C61, 0x0CA86C39 */
viv_pow_ovt     =  8.0085662595372944372e-017, /* -(1024-log2(ovfl+.5ulp)) */
viv_pow_cp      =  9.61796693925975554329e-01, /* 0x3FEEC709, 0xDC3A03FD =2/(3ln2) */
viv_pow_cp_h    =  9.61796700954437255859e-01, /* 0x3FEEC709, 0xE0000000 =(float)cp */
viv_pow_cp_l    = -7.02846165095275826516e-09, /* 0xBE3E2FE0, 0x145B01F5 =tail of cp_h*/
viv_pow_ivln2   =  1.44269504088896338700e+00, /* 0x3FF71547, 0x652B82FE =1/ln2 */
viv_pow_ivln2_h =  1.44269502162933349609e+00, /* 0x3FF71547, 0x60000000 =24b 1/ln2*/
viv_pow_ivln2_l =  1.92596299112661746887e-08; /* 0x3E54AE0B, 0xF85DDF44 =1/ln2 tail*/

static double viv_pow(double x, double y)
{
	double z,ax,z_h,z_l,p_h,p_l;
	double y1,t1,t2,r,s,t,u,v,w;
	int32_t i,j,k,yisint,n;
	int32_t hx,hy,ix,iy;
	uint32_t lx,ly;

	EXTRACT_WORDS(hx, lx, x);
	EXTRACT_WORDS(hy, ly, y);
	ix = hx & 0x7fffffff;
	iy = hy & 0x7fffffff;

	/* x**0 = 1, even if x is NaN */
	if ((iy|ly) == 0)
		return 1.0;
	/* 1**y = 1, even if y is NaN */
	if (hx == 0x3ff00000 && lx == 0)
		return 1.0;
	/* NaN if either arg is NaN */
	if (ix > 0x7ff00000 || (ix == 0x7ff00000 && lx != 0) ||
	    iy > 0x7ff00000 || (iy == 0x7ff00000 && ly != 0))
		return x + y;

	/* determine if y is an odd int when x < 0
	 * yisint = 0       ... y is not an integer
	 * yisint = 1       ... y is an odd int
	 * yisint = 2       ... y is an even int
	 */
	yisint = 0;
	if (hx < 0) {
		if (iy >= 0x43400000)
			yisint = 2; /* even integer y */
		else if (iy >= 0x3ff00000) {
			k = (iy>>20) - 0x3ff;  /* exponent */
			if (k > 20) {
				uint32_t j = ly>>(52-k);
				if ((j<<(52-k)) == ly)
					yisint = 2 - (j&1);
			} else if (ly == 0) {
				uint32_t j = iy>>(20-k);
				if ((j<<(20-k)) == iy)
					yisint = 2 - (j&1);
			}
		}
	}

	/* special value of y */
	if (ly == 0) {
		if (iy == 0x7ff00000) {  /* y is +-inf */
			if (((ix-0x3ff00000)|lx) == 0)  /* (-1)**+-inf is 1 */
				return 1.0;
			else if (ix >= 0x3ff00000) /* (|x|>1)**+-inf = inf,0 */
				return hy >= 0 ? y : 0.0;
			else                       /* (|x|<1)**+-inf = 0,inf */
				return hy >= 0 ? 0.0 : -y;
		}
		if (iy == 0x3ff00000) {    /* y is +-1 */
			if (hy >= 0)
				return x;
			y = 1/x;
#if FLT_EVAL_METHOD!=0
			{
				union {double f; uint64_t i;} u = {y};
				uint64_t i = u.i & -1ULL/2;
				if (i>>52 == 0 && (i&(i-1)))
					FORCE_EVAL((float)y);
			}
#endif
			return y;
		}
		if (hy == 0x40000000)    /* y is 2 */
			return x*x;
		if (hy == 0x3fe00000) {  /* y is 0.5 */
			if (hx >= 0)     /* x >= +0 */
				return sqrt(x);
		}
	}

	ax = fabs(x);
	/* special value of x */
	if (lx == 0) {
		if (ix == 0x7ff00000 || ix == 0 || ix == 0x3ff00000) { /* x is +-0,+-inf,+-1 */
			z = ax;
			if (hy < 0)   /* z = (1/|x|) */
				z = 1.0/z;
			if (hx < 0) {
				if (((ix-0x3ff00000)|yisint) == 0) {
					z = (z-z)/(z-z); /* (-1)**non-int is NaN */
				} else if (yisint == 1)
					z = -z;          /* (x<0)**odd = -(|x|**odd) */
			}
			return z;
		}
	}

	s = 1.0; /* sign of result */
	if (hx < 0) {
		if (yisint == 0) /* (x<0)**(non-int) is NaN */
			return (x-x)/(x-x);
		if (yisint == 1) /* (x<0)**(odd int) */
			s = -1.0;
	}

	/* |y| is huge */
	if (iy > 0x41e00000) { /* if |y| > 2**31 */
		if (iy > 0x43f00000) {  /* if |y| > 2**64, must o/uflow */
			if (ix <= 0x3fefffff)
				return hy < 0 ? viv_pow_huge*viv_pow_huge : viv_pow_tiny*viv_pow_tiny;
			if (ix >= 0x3ff00000)
				return hy > 0 ? viv_pow_huge*viv_pow_huge : viv_pow_tiny*viv_pow_tiny;
		}
		/* over/underflow if x is not close to one */
		if (ix < 0x3fefffff)
			return hy < 0 ? s*viv_pow_huge*viv_pow_huge : s*viv_pow_tiny*viv_pow_tiny;
		if (ix > 0x3ff00000)
			return hy > 0 ? s*viv_pow_huge*viv_pow_huge : s*viv_pow_tiny*viv_pow_tiny;
		/* now |1-x| is tiny <= 2**-20, suffice to compute
		   log(x) by x-x^2/2+x^3/3-x^4/4 */
		t = ax - 1.0;       /* t has 20 trailing zeros */
		w = (t*t)*(0.5 - t*(0.3333333333333333333333-t*0.25));
		u = viv_pow_ivln2_h*t;      /* ivln2_h has 21 sig. bits */
		v = t*viv_pow_ivln2_l - w*viv_pow_ivln2;
		t1 = u + v;
		SET_LOW_WORD(t1, 0);
		t2 = v - (t1-u);
	} else {
		double ss,s2,s_h,s_l,t_h,t_l;
		n = 0;
		/* take care subnormal number */
		if (ix < 0x00100000) {
			ax *= viv_pow_two53;
			n -= 53;
			GET_HIGH_WORD(ix,ax);
		}
		n += ((ix)>>20) - 0x3ff;
		j = ix & 0x000fffff;
		/* determine interval */
		ix = j | 0x3ff00000;   /* normalize ix */
		if (j <= 0x3988E)      /* |x|<sqrt(3/2) */
			k = 0;
		else if (j < 0xBB67A)  /* |x|<sqrt(3)   */
			k = 1;
		else {
			k = 0;
			n += 1;
			ix -= 0x00100000;
		}
		SET_HIGH_WORD(ax, ix);

		/* compute ss = s_h+s_l = (x-1)/(x+1) or (x-1.5)/(x+1.5) */
		u = ax - viv_pow_bp[k];        /* bp[0]=1.0, bp[1]=1.5 */
		v = 1.0/(ax+viv_pow_bp[k]);
		ss = u*v;
		s_h = ss;
		SET_LOW_WORD(s_h, 0);
		/* t_h=ax+bp[k] High */
		t_h = 0.0;
		SET_HIGH_WORD(t_h, ((ix>>1)|0x20000000) + 0x00080000 + (k<<18));
		t_l = ax - (t_h-viv_pow_bp[k]);
		s_l = v*((u-s_h*t_h)-s_h*t_l);
		/* compute log(ax) */
		s2 = ss*ss;
		r = s2*s2*(viv_pow_L1+s2*(viv_pow_L2+s2*(viv_pow_L3+s2*(viv_pow_L4+s2*(viv_pow_L5+s2*viv_pow_L6)))));
		r += s_l*(s_h+ss);
		s2 = s_h*s_h;
		t_h = 3.0 + s2 + r;
		SET_LOW_WORD(t_h, 0);
		t_l = r - ((t_h-3.0)-s2);
		/* u+v = ss*(1+...) */
		u = s_h*t_h;
		v = s_l*t_h + t_l*ss;
		/* 2/(3log2)*(ss+...) */
		p_h = u + v;
		SET_LOW_WORD(p_h, 0);
		p_l = v - (p_h-u);
		z_h = viv_pow_cp_h*p_h;        /* cp_h+cp_l = 2/(3*log2) */
		z_l = viv_pow_cp_l*p_h+p_l*viv_pow_cp + viv_pow_dp_l[k];
		/* log2(ax) = (ss+..)*2/(3*log2) = n + dp_h + z_h + z_l */
		t = (double)n;
		t1 = ((z_h + z_l) + viv_pow_dp_h[k]) + t;
		SET_LOW_WORD(t1, 0);
		t2 = z_l - (((t1 - t) - viv_pow_dp_h[k]) - z_h);
	}

	/* split up y into y1+y2 and compute (y1+y2)*(t1+t2) */
	y1 = y;
	SET_LOW_WORD(y1, 0);
	p_l = (y-y1)*t1 + y*t2;
	p_h = y1*t1;
	z = p_l + p_h;
	EXTRACT_WORDS(j, i, z);
	if (j >= 0x40900000) {                      /* z >= 1024 */
		if (((j-0x40900000)|i) != 0)        /* if z > 1024 */
			return s*viv_pow_huge*viv_pow_huge;         /* overflow */
		if (p_l + viv_pow_ovt > z - p_h)
			return s*viv_pow_huge*viv_pow_huge;         /* overflow */
	} else if ((j&0x7fffffff) >= 0x4090cc00) {  /* z <= -1075 */  // FIXME: instead of abs(j) use unsigned j
		if (((j-0xc090cc00)|i) != 0)        /* z < -1075 */
			return s*viv_pow_tiny*viv_pow_tiny;         /* underflow */
		if (p_l <= z - p_h)
			return s*viv_pow_tiny*viv_pow_tiny;         /* underflow */
	}
	/*
	 * compute 2**(p_h+p_l)
	 */
	i = j & 0x7fffffff;
	k = (i>>20) - 0x3ff;
	n = 0;
	if (i > 0x3fe00000) {  /* if |z| > 0.5, set n = [z+0.5] */
		n = j + (0x00100000>>(k+1));
		k = ((n&0x7fffffff)>>20) - 0x3ff;  /* new k for n */
		t = 0.0;
		SET_HIGH_WORD(t, n & ~(0x000fffff>>k));
		n = ((n&0x000fffff)|0x00100000)>>(20-k);
		if (j < 0)
			n = -n;
		p_h -= t;
	}
	t = p_l + p_h;
	SET_LOW_WORD(t, 0);
	u = t*viv_pow_lg2_h;
	v = (p_l-(t-p_h))*viv_pow_lg2 + t*viv_pow_lg2_l;
	z = u + v;
	w = v - (z-u);
	t = z*z;
	t1 = z - t*(viv_pow_P1+t*(viv_pow_P2+t*(viv_pow_P3+t*(viv_pow_P4+t*viv_pow_P5))));
	r = (z*t1)/(t1-2.0) - (w + z*w);
	z = 1.0 - (r-z);
	GET_HIGH_WORD(j, z);
	/* n<<20 with n negative is undefined in C (UBSan reports it); n is at
	   most about 1100 in magnitude, so the product fits an int32_t and is
	   the same number the shift meant */
	j += n * (1 << 20);
	if ((j>>20) <= 0)  /* subnormal output */
		z = viv_scalbn(z,n);
	else
		SET_HIGH_WORD(z, j);
	return s*z;
}

/* ---- musl src/math/atan.c ---- */
/* origin: FreeBSD /usr/src/lib/msun/src/s_atan.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* atan(x)
 * Method
 *   1. Reduce x to positive by atan(x) = -atan(-x).
 *   2. According to the integer k=4t+0.25 chopped, t=x, the argument
 *      is further reduced to one of the following intervals and the
 *      arctangent of t is evaluated by the corresponding formula:
 *
 *      [0,7/16]      atan(x) = t-t^3*(a1+t^2*(a2+...(a10+t^2*a11)...)
 *      [7/16,11/16]  atan(x) = atan(1/2) + atan( (t-0.5)/(1+t/2) )
 *      [11/16.19/16] atan(x) = atan( 1 ) + atan( (t-1)/(1+t) )
 *      [19/16,39/16] atan(x) = atan(3/2) + atan( (t-1.5)/(1+1.5t) )
 *      [39/16,INF]   atan(x) = atan(INF) + atan( -1/t )
 *
 * Constants:
 * The hexadecimal values are the intended ones for the following
 * constants. The decimal values may be used, provided that the
 * compiler will convert from decimal to binary accurately enough
 * to produce the hexadecimal values shown.
 */



static const double viv_atan_atanhi[] = {
  4.63647609000806093515e-01, /* atan(0.5)hi 0x3FDDAC67, 0x0561BB4F */
  7.85398163397448278999e-01, /* atan(1.0)hi 0x3FE921FB, 0x54442D18 */
  9.82793723247329054082e-01, /* atan(1.5)hi 0x3FEF730B, 0xD281F69B */
  1.57079632679489655800e+00, /* atan(inf)hi 0x3FF921FB, 0x54442D18 */
};

static const double viv_atan_atanlo[] = {
  2.26987774529616870924e-17, /* atan(0.5)lo 0x3C7A2B7F, 0x222F65E2 */
  3.06161699786838301793e-17, /* atan(1.0)lo 0x3C81A626, 0x33145C07 */
  1.39033110312309984516e-17, /* atan(1.5)lo 0x3C700788, 0x7AF0CBBD */
  6.12323399573676603587e-17, /* atan(inf)lo 0x3C91A626, 0x33145C07 */
};

static const double viv_atan_aT[] = {
  3.33333333333329318027e-01, /* 0x3FD55555, 0x5555550D */
 -1.99999999998764832476e-01, /* 0xBFC99999, 0x9998EBC4 */
  1.42857142725034663711e-01, /* 0x3FC24924, 0x920083FF */
 -1.11111104054623557880e-01, /* 0xBFBC71C6, 0xFE231671 */
  9.09088713343650656196e-02, /* 0x3FB745CD, 0xC54C206E */
 -7.69187620504482999495e-02, /* 0xBFB3B0F2, 0xAF749A6D */
  6.66107313738753120669e-02, /* 0x3FB10D66, 0xA0D03D51 */
 -5.83357013379057348645e-02, /* 0xBFADDE2D, 0x52DEFD9A */
  4.97687799461593236017e-02, /* 0x3FA97B4B, 0x24760DEB */
 -3.65315727442169155270e-02, /* 0xBFA2B444, 0x2C6A6C2F */
  1.62858201153657823623e-02, /* 0x3F90AD3A, 0xE322DA11 */
};

static double viv_atan(double x)
{
	double w,s1,s2,z;
	uint32_t ix,sign;
	int id;

	GET_HIGH_WORD(ix, x);
	sign = ix >> 31;
	ix &= 0x7fffffff;
	if (ix >= 0x44100000) {   /* if |x| >= 2^66 */
		if (isnan(x))
			return x;
		z = viv_atan_atanhi[3] + 0x1p-120f;
		return sign ? -z : z;
	}
	if (ix < 0x3fdc0000) {    /* |x| < 0.4375 */
		if (ix < 0x3e400000) {  /* |x| < 2^-27 */
			if (ix < 0x00100000)
				/* raise underflow for subnormal x */
				FORCE_EVAL((float)x);
			return x;
		}
		id = -1;
	} else {
		x = fabs(x);
		if (ix < 0x3ff30000) {  /* |x| < 1.1875 */
			if (ix < 0x3fe60000) {  /*  7/16 <= |x| < 11/16 */
				id = 0;
				x = (2.0*x-1.0)/(2.0+x);
			} else {                /* 11/16 <= |x| < 19/16 */
				id = 1;
				x = (x-1.0)/(x+1.0);
			}
		} else {
			if (ix < 0x40038000) {  /* |x| < 2.4375 */
				id = 2;
				x = (x-1.5)/(1.0+1.5*x);
			} else {                /* 2.4375 <= |x| < 2^66 */
				id = 3;
				x = -1.0/x;
			}
		}
	}
	/* end of argument reduction */
	z = x*x;
	w = z*z;
	/* break sum from i=0 to 10 aT[i]z**(i+1) into odd and even poly */
	s1 = z*(viv_atan_aT[0]+w*(viv_atan_aT[2]+w*(viv_atan_aT[4]+w*(viv_atan_aT[6]+w*(viv_atan_aT[8]+w*viv_atan_aT[10])))));
	s2 = w*(viv_atan_aT[1]+w*(viv_atan_aT[3]+w*(viv_atan_aT[5]+w*(viv_atan_aT[7]+w*viv_atan_aT[9]))));
	if (id < 0)
		return x - x*(s1+s2);
	z = viv_atan_atanhi[id] - (x*(s1+s2) - viv_atan_atanlo[id] - x);
	return sign ? -z : z;
}

/* ---- musl src/math/atan2.c ---- */
/* origin: FreeBSD /usr/src/lib/msun/src/e_atan2.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 *
 */
/* atan2(y,x)
 * Method :
 *      1. Reduce y to positive by atan2(y,x)=-atan2(-y,x).
 *      2. Reduce x to positive by (if x and y are unexceptional):
 *              ARG (x+iy) = arctan(y/x)           ... if x > 0,
 *              ARG (x+iy) = pi - arctan[y/(-x)]   ... if x < 0,
 *
 * Special cases:
 *
 *      ATAN2((anything), NaN ) is NaN;
 *      ATAN2(NAN , (anything) ) is NaN;
 *      ATAN2(+-0, +(anything but NaN)) is +-0  ;
 *      ATAN2(+-0, -(anything but NaN)) is +-pi ;
 *      ATAN2(+-(anything but 0 and NaN), 0) is +-pi/2;
 *      ATAN2(+-(anything but INF and NaN), +INF) is +-0 ;
 *      ATAN2(+-(anything but INF and NaN), -INF) is +-pi;
 *      ATAN2(+-INF,+INF ) is +-pi/4 ;
 *      ATAN2(+-INF,-INF ) is +-3pi/4;
 *      ATAN2(+-INF, (anything but,0,NaN, and INF)) is +-pi/2;
 *
 * Constants:
 * The hexadecimal values are the intended ones for the following
 * constants. The decimal values may be used, provided that the
 * compiler will convert from decimal to binary accurately enough
 * to produce the hexadecimal values shown.
 */


static const double
viv_atan2_pi     = 3.1415926535897931160E+00, /* 0x400921FB, 0x54442D18 */
viv_atan2_pi_lo  = 1.2246467991473531772E-16; /* 0x3CA1A626, 0x33145C07 */

static double viv_atan2(double y, double x)
{
	double z;
	uint32_t m,lx,ly,ix,iy;

	if (isnan(x) || isnan(y))
		return x+y;
	EXTRACT_WORDS(ix, lx, x);
	EXTRACT_WORDS(iy, ly, y);
	if ((ix-0x3ff00000 | lx) == 0)  /* x = 1.0 */
		return viv_atan(y);
	m = ((iy>>31)&1) | ((ix>>30)&2);  /* 2*sign(x)+sign(y) */
	ix = ix & 0x7fffffff;
	iy = iy & 0x7fffffff;

	/* when y = 0 */
	if ((iy|ly) == 0) {
		switch(m) {
		case 0:
		case 1: return y;   /* atan(+-0,+anything)=+-0 */
		case 2: return  viv_atan2_pi; /* atan(+0,-anything) = pi */
		case 3: return -viv_atan2_pi; /* atan(-0,-anything) =-pi */
		}
	}
	/* when x = 0 */
	if ((ix|lx) == 0)
		return m&1 ? -viv_atan2_pi/2 : viv_atan2_pi/2;
	/* when x is INF */
	if (ix == 0x7ff00000) {
		if (iy == 0x7ff00000) {
			switch(m) {
			case 0: return  viv_atan2_pi/4;   /* atan(+INF,+INF) */
			case 1: return -viv_atan2_pi/4;   /* atan(-INF,+INF) */
			case 2: return  3*viv_atan2_pi/4; /* atan(+INF,-INF) */
			case 3: return -3*viv_atan2_pi/4; /* atan(-INF,-INF) */
			}
		} else {
			switch(m) {
			case 0: return  0.0; /* atan(+...,+INF) */
			case 1: return -0.0; /* atan(-...,+INF) */
			case 2: return  viv_atan2_pi;  /* atan(+...,-INF) */
			case 3: return -viv_atan2_pi;  /* atan(-...,-INF) */
			}
		}
	}
	/* |y/x| > 0x1p64 */
	if (ix+(64<<20) < iy || iy == 0x7ff00000)
		return m&1 ? -viv_atan2_pi/2 : viv_atan2_pi/2;

	/* z = atan(|y/x|) without spurious underflow */
	if ((m&2) && iy+(64<<20) < ix)  /* |y/x| < 0x1p-64, x<0 */
		z = 0;
	else
		z = viv_atan(fabs(y/x));
	switch (m) {
	case 0: return z;              /* atan(+,+) */
	case 1: return -z;             /* atan(-,+) */
	case 2: return viv_atan2_pi - (z-viv_atan2_pi_lo); /* atan(+,-) */
	default: /* case 3 */
		return (z-viv_atan2_pi_lo) - viv_atan2_pi; /* atan(-,-) */
	}
}

/* ---- musl src/math/asin.c ---- */
/* origin: FreeBSD /usr/src/lib/msun/src/e_asin.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* asin(x)
 * Method :
 *      Since  asin(x) = x + x^3/6 + x^5*3/40 + x^7*15/336 + ...
 *      we approximate asin(x) on [0,0.5] by
 *              asin(x) = x + x*x^2*R(x^2)
 *      where
 *              R(x^2) is a rational approximation of (asin(x)-x)/x^3
 *      and its remez error is bounded by
 *              |(asin(x)-x)/x^3 - R(x^2)| < 2^(-58.75)
 *
 *      For x in [0.5,1]
 *              asin(x) = pi/2-2*asin(sqrt((1-x)/2))
 *      Let y = (1-x), z = y/2, s := sqrt(z), and pio2_hi+pio2_lo=pi/2;
 *      then for x>0.98
 *              asin(x) = pi/2 - 2*(s+s*z*R(z))
 *                      = pio2_hi - (2*(s+s*z*R(z)) - pio2_lo)
 *      For x<=0.98, let pio4_hi = pio2_hi/2, then
 *              f = hi part of s;
 *              c = sqrt(z) - f = (z-f*f)/(s+f)         ...f+c=sqrt(z)
 *      and
 *              asin(x) = pi/2 - 2*(s+s*z*R(z))
 *                      = pio4_hi+(pio4-2s)-(2s*z*R(z)-pio2_lo)
 *                      = pio4_hi+(pio4-2f)-(2s*z*R(z)-(pio2_lo+2c))
 *
 * Special cases:
 *      if x is NaN, return x itself;
 *      if |x|>1, return NaN with invalid signal.
 *
 */


static const double
viv_asin_pio2_hi = 1.57079632679489655800e+00, /* 0x3FF921FB, 0x54442D18 */
viv_asin_pio2_lo = 6.12323399573676603587e-17, /* 0x3C91A626, 0x33145C07 */
/* coefficients for R(x^2) */
viv_asin_pS0 =  1.66666666666666657415e-01, /* 0x3FC55555, 0x55555555 */
viv_asin_pS1 = -3.25565818622400915405e-01, /* 0xBFD4D612, 0x03EB6F7D */
viv_asin_pS2 =  2.01212532134862925881e-01, /* 0x3FC9C155, 0x0E884455 */
viv_asin_pS3 = -4.00555345006794114027e-02, /* 0xBFA48228, 0xB5688F3B */
viv_asin_pS4 =  7.91534994289814532176e-04, /* 0x3F49EFE0, 0x7501B288 */
viv_asin_pS5 =  3.47933107596021167570e-05, /* 0x3F023DE1, 0x0DFDF709 */
viv_asin_qS1 = -2.40339491173441421878e+00, /* 0xC0033A27, 0x1C8A2D4B */
viv_asin_qS2 =  2.02094576023350569471e+00, /* 0x40002AE5, 0x9C598AC8 */
viv_asin_qS3 = -6.88283971605453293030e-01, /* 0xBFE6066C, 0x1B8D0159 */
viv_asin_qS4 =  7.70381505559019352791e-02; /* 0x3FB3B8C5, 0xB12E9282 */

static double viv_asin_R(double z)
{
	double p, q;
	p = z*(viv_asin_pS0+z*(viv_asin_pS1+z*(viv_asin_pS2+z*(viv_asin_pS3+z*(viv_asin_pS4+z*viv_asin_pS5)))));
	q = 1.0+z*(viv_asin_qS1+z*(viv_asin_qS2+z*(viv_asin_qS3+z*viv_asin_qS4)));
	return p/q;
}

static double viv_asin(double x)
{
	double z,r,s;
	uint32_t hx,ix;

	GET_HIGH_WORD(hx, x);
	ix = hx & 0x7fffffff;
	/* |x| >= 1 or nan */
	if (ix >= 0x3ff00000) {
		uint32_t lx;
		GET_LOW_WORD(lx, x);
		if ((ix-0x3ff00000 | lx) == 0)
			/* asin(1) = +-pi/2 with inexact */
			return x*viv_asin_pio2_hi + 0x1p-120f;
		return 0/(x-x);
	}
	/* |x| < 0.5 */
	if (ix < 0x3fe00000) {
		/* if 0x1p-1022 <= |x| < 0x1p-26, avoid raising underflow */
		if (ix < 0x3e500000 && ix >= 0x00100000)
			return x;
		return x + x*viv_asin_R(x*x);
	}
	/* 1 > |x| >= 0.5 */
	z = (1 - fabs(x))*0.5;
	s = sqrt(z);
	r = viv_asin_R(z);
	if (ix >= 0x3fef3333) {  /* if |x| > 0.975 */
		x = viv_asin_pio2_hi-(2*(s+s*r)-viv_asin_pio2_lo);
	} else {
		double f,c;
		/* f+c = sqrt(z) */
		f = s;
		SET_LOW_WORD(f,0);
		c = (z-f*f)/(s+f);
		x = 0.5*viv_asin_pio2_hi - (2*s*r - (viv_asin_pio2_lo-2*c) - (0.5*viv_asin_pio2_hi-2*f));
	}
	if (hx >> 31)
		return -x;
	return x;
}

/* ---- musl src/math/acos.c ---- */
/* origin: FreeBSD /usr/src/lib/msun/src/e_acos.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* acos(x)
 * Method :
 *      acos(x)  = pi/2 - asin(x)
 *      acos(-x) = pi/2 + asin(x)
 * For |x|<=0.5
 *      acos(x) = pi/2 - (x + x*x^2*R(x^2))     (see asin.c)
 * For x>0.5
 *      acos(x) = pi/2 - (pi/2 - 2asin(sqrt((1-x)/2)))
 *              = 2asin(sqrt((1-x)/2))
 *              = 2s + 2s*z*R(z)        ...z=(1-x)/2, s=sqrt(z)
 *              = 2f + (2c + 2s*z*R(z))
 *     where f=hi part of s, and c = (z-f*f)/(s+f) is the correction term
 *     for f so that f+c ~ sqrt(z).
 * For x<-0.5
 *      acos(x) = pi - 2asin(sqrt((1-|x|)/2))
 *              = pi - 0.5*(s+s*z*R(z)), where z=(1-|x|)/2,s=sqrt(z)
 *
 * Special cases:
 *      if x is NaN, return x itself;
 *      if |x|>1, return NaN with invalid signal.
 *
 * Function needed: sqrt
 */


static const double
viv_acos_pio2_hi = 1.57079632679489655800e+00, /* 0x3FF921FB, 0x54442D18 */
viv_acos_pio2_lo = 6.12323399573676603587e-17, /* 0x3C91A626, 0x33145C07 */
viv_acos_pS0 =  1.66666666666666657415e-01, /* 0x3FC55555, 0x55555555 */
viv_acos_pS1 = -3.25565818622400915405e-01, /* 0xBFD4D612, 0x03EB6F7D */
viv_acos_pS2 =  2.01212532134862925881e-01, /* 0x3FC9C155, 0x0E884455 */
viv_acos_pS3 = -4.00555345006794114027e-02, /* 0xBFA48228, 0xB5688F3B */
viv_acos_pS4 =  7.91534994289814532176e-04, /* 0x3F49EFE0, 0x7501B288 */
viv_acos_pS5 =  3.47933107596021167570e-05, /* 0x3F023DE1, 0x0DFDF709 */
viv_acos_qS1 = -2.40339491173441421878e+00, /* 0xC0033A27, 0x1C8A2D4B */
viv_acos_qS2 =  2.02094576023350569471e+00, /* 0x40002AE5, 0x9C598AC8 */
viv_acos_qS3 = -6.88283971605453293030e-01, /* 0xBFE6066C, 0x1B8D0159 */
viv_acos_qS4 =  7.70381505559019352791e-02; /* 0x3FB3B8C5, 0xB12E9282 */

static double viv_acos_R(double z)
{
	double p, q;
	p = z*(viv_acos_pS0+z*(viv_acos_pS1+z*(viv_acos_pS2+z*(viv_acos_pS3+z*(viv_acos_pS4+z*viv_acos_pS5)))));
	q = 1.0+z*(viv_acos_qS1+z*(viv_acos_qS2+z*(viv_acos_qS3+z*viv_acos_qS4)));
	return p/q;
}

static double viv_acos(double x)
{
	double z,w,s,c,df;
	uint32_t hx,ix;

	GET_HIGH_WORD(hx, x);
	ix = hx & 0x7fffffff;
	/* |x| >= 1 or nan */
	if (ix >= 0x3ff00000) {
		uint32_t lx;

		GET_LOW_WORD(lx,x);
		if ((ix-0x3ff00000 | lx) == 0) {
			/* acos(1)=0, acos(-1)=pi */
			if (hx >> 31)
				return 2*viv_acos_pio2_hi + 0x1p-120f;
			return 0;
		}
		return 0/(x-x);
	}
	/* |x| < 0.5 */
	if (ix < 0x3fe00000) {
		if (ix <= 0x3c600000)  /* |x| < 2**-57 */
			return viv_acos_pio2_hi + 0x1p-120f;
		return viv_acos_pio2_hi - (x - (viv_acos_pio2_lo-x*viv_acos_R(x*x)));
	}
	/* x < -0.5 */
	if (hx >> 31) {
		z = (1.0+x)*0.5;
		s = sqrt(z);
		w = viv_acos_R(z)*s-viv_acos_pio2_lo;
		return 2*(viv_acos_pio2_hi - (s+w));
	}
	/* x > 0.5 */
	z = (1.0-x)*0.5;
	s = sqrt(z);
	df = s;
	SET_LOW_WORD(df,0);
	c = (z-df*df)/(s+df);
	w = viv_acos_R(z)*s+c;
	return 2*(df+w);
}

/* ---- musl src/math/sinh.c ---- */
/* sinh(x) = (exp(x) - 1/exp(x))/2
 *         = (exp(x)-1 + (exp(x)-1)/exp(x))/2
 *         = x + x^3/6 + o(x^5)
 */
static double viv_sinh(double x)
{
	union {double f; uint64_t i;} u = {.f = x};
	uint32_t w;
	double t, h, absx;

	h = 0.5;
	if (u.i >> 63)
		h = -h;
	/* |x| */
	u.i &= (uint64_t)-1/2;
	absx = u.f;
	w = u.i >> 32;

	/* |x| < log(DBL_MAX) */
	if (w < 0x40862e42) {
		t = viv_expm1(absx);
		if (w < 0x3ff00000) {
			if (w < 0x3ff00000 - (26<<20))
				/* note: inexact and underflow are raised by expm1 */
				/* note: this branch avoids spurious underflow */
				return x;
			return h*(2*t - t*t/(t+1));
		}
		/* note: |x|>log(0x1p26)+eps could be just h*exp(x) */
		return h*(t + t/(t+1));
	}

	/* |x| > log(DBL_MAX) or nan */
	/* note: the result is stored to handle overflow */
	t = 2*h*viv__expo2(absx);
	return t;
}

/* ---- musl src/math/cosh.c ---- */
/* cosh(x) = (exp(x) + 1/exp(x))/2
 *         = 1 + 0.5*(exp(x)-1)*(exp(x)-1)/exp(x)
 *         = 1 + x*x/2 + o(x^4)
 */
static double viv_cosh(double x)
{
	union {double f; uint64_t i;} u = {.f = x};
	uint32_t w;
	double t;

	/* |x| */
	u.i &= (uint64_t)-1/2;
	x = u.f;
	w = u.i >> 32;

	/* |x| < log(2) */
	if (w < 0x3fe62e42) {
		if (w < 0x3ff00000 - (26<<20)) {
			/* raise inexact if x!=0 */
			FORCE_EVAL(x + 0x1p120f);
			return 1;
		}
		t = viv_expm1(x);
		return 1 + t*t/(2*(1+t));
	}

	/* |x| < log(DBL_MAX) */
	if (w < 0x40862e42) {
		t = viv_exp(x);
		/* note: if x>log(0x1p26) then the 1/t is not needed */
		return 0.5*(t + 1/t);
	}

	/* |x| > log(DBL_MAX) or nan */
	/* note: the result is stored to handle overflow */
	t = viv__expo2(x);
	return t;
}

/* ---- musl src/math/tanh.c ---- */
/* tanh(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))
 *         = (exp(2*x) - 1)/(exp(2*x) - 1 + 2)
 *         = (1 - exp(-2*x))/(exp(-2*x) - 1 + 2)
 */
static double viv_tanh(double x)
{
	union {double f; uint64_t i;} u = {.f = x};
	uint32_t w;
	int sign;
	double t;

	/* x = |x| */
	sign = u.i >> 63;
	u.i &= (uint64_t)-1/2;
	x = u.f;
	w = u.i >> 32;

	if (w > 0x3fe193ea) {
		/* |x| > log(3)/2 ~= 0.5493 or nan */
		if (w > 0x40340000) {
			/* |x| > 20 or nan */
			/* note: this branch avoids raising overflow */
			t = 1 - 0/x;
		} else {
			t = viv_expm1(2*x);
			t = 1 - 2/(t+2);
		}
	} else if (w > 0x3fd058ae) {
		/* |x| > log(5/3)/2 ~= 0.2554 */
		t = viv_expm1(2*x);
		t = t/(t+2);
	} else if (w >= 0x00100000) {
		/* |x| >= 0x1p-1022, up to 2ulp error in [0.1,0.2554] */
		t = viv_expm1(-2*x);
		t = -t/(t+2);
	} else {
		/* |x| is subnormal */
		/* note: the branch above would not raise underflow in [0x1p-1023,0x1p-1022) */
		FORCE_EVAL((float)x);
		t = x;
	}
	return sign ? -t : t;
}

/* ---- musl src/math/hypot.c ---- */
#if FLT_EVAL_METHOD > 1U && LDBL_MANT_DIG == 64
#define SPLIT (0x1p32 + 1)
#else
#define SPLIT (0x1p27 + 1)
#endif

static void viv_hypot_sq(double *hi, double *lo, double x)
{
	double xh, xl, xc;

	xc = (double)x*SPLIT;
	xh = x - xc + xc;
	xl = x - xh;
	*hi = (double)x*x;
	*lo = xh*xh - *hi + 2*xh*xl + xl*xl;
}

static double viv_hypot(double x, double y)
{
	union {double f; uint64_t i;} ux = {x}, uy = {y}, ut;
	int ex, ey;
	double hx, lx, hy, ly, z;

	/* arrange |x| >= |y| */
	ux.i &= -1ULL>>1;
	uy.i &= -1ULL>>1;
	if (ux.i < uy.i) {
		ut = ux;
		ux = uy;
		uy = ut;
	}

	/* special cases */
	ex = ux.i>>52;
	ey = uy.i>>52;
	x = ux.f;
	y = uy.f;
	/* note: hypot(inf,nan) == inf */
	if (ey == 0x7ff)
		return y;
	if (ex == 0x7ff || uy.i == 0)
		return x;
	/* note: hypot(x,y) ~= x + y*y/x/2 with inexact for small y/x */
	/* 64 difference is enough for ld80 double */
	if (ex - ey > 64)
		return x + y;

	/* precise sqrt argument in nearest rounding mode without overflow */
	/* xh*xh must not overflow and xl*xl must not underflow in sq */
	z = 1;
	if (ex > 0x3ff+510) {
		z = 0x1p700;
		x *= 0x1p-700;
		y *= 0x1p-700;
	} else if (ey < 0x3ff-450) {
		z = 0x1p-700;
		x *= 0x1p700;
		y *= 0x1p700;
	}
	viv_hypot_sq(&hx, &lx, x);
	viv_hypot_sq(&hy, &ly, y);
	return z*sqrt(ly+lx+hy+hx);
}
#undef SPLIT

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
static _Complex double viv_csqrt(_Complex double z)
{
	viv_cparts u;
	u.z = z;
	double r = sqrt(viv_hypot(u.p[0], u.p[1]));
	double th = 0.5 * viv_atan2(u.p[1], u.p[0]);
	return viv_cmake(r * viv_cos(th), r * viv_sin(th));
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

#if defined(VIV_FAST) && !defined(VIV_FAST_NO_CONTRACT) && defined(__clang__)
#pragma clang fp contract(fast)
#elif defined(VIV_FAST) && !defined(VIV_FAST_NO_CONTRACT)
#pragma STDC FP_CONTRACT ON
#endif
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <stdarg.h>
#include <errno.h>

#ifdef VIV_EMBED
#define main viv_program_main
#pragma GCC poison exit _exit printf puts putchar vprintf perror atexit
#endif
typedef struct { int rank; int dims[8]; int size; double* data; } viv_arr;

/* The array arena: every array allocation is pushed on a stack. A function
   that can allocate notes the height on entry and frees, on return, everything
   above it except the array it returns (which is re-filed under the caller).
   Arrays made at the top level sit below every mark and live for the run.
   With OpenMP each thread has its own stack, so a function called inside a
   parallel loop frees only what that thread allocated. */
#ifdef _OPENMP
#define VIV_TLS __thread
#else
#define VIV_TLS
#endif
static VIV_TLS void** viv_arena = 0;
static VIV_TLS size_t viv_arena_top = 0, viv_arena_cap = 0;
static void viv_arena_push(void* p) {
    if (viv_arena_top == viv_arena_cap) {
        viv_arena_cap = viv_arena_cap ? 2 * viv_arena_cap : 256;
        viv_arena = (void**)realloc(viv_arena, viv_arena_cap * sizeof(void*));
    }
    viv_arena[viv_arena_top++] = p;
}
/* A map is not one block: it is filed with a tag in its pointer's low bits
   (1 a map of numbers, 2 a map with strings; every block malloc gives is
   aligned past them) and freed by the function its runtime puts in
   viv_arena_drop. */
static void (*viv_arena_drop[4])(void*);
/* An enum node is filed with tag 3 and
   begins with this head: its place on the arena, and the number of the last
   release pass that kept it. A pass keeps a node by marking it, so a release
   asks the node instead of searching the kept list, and a walk stops at a
   node it has marked, or at one below the mark it releases to. */
typedef struct { int tag; unsigned keep; long pos; } viv_en_head;
static void viv_arena_free(void* p) {
    size_t tag = (size_t)p & 3;
    if (tag == 3) free((char*)p - 3);
    else if (tag) viv_arena_drop[tag]((char*)p - tag);
    else free(p);
}
static void viv_arena_release(size_t mark, const void* keep) {
    size_t w = mark;
    for (size_t i = mark; i < viv_arena_top; i++) {
        if (viv_arena[i] == keep) viv_arena[w++] = viv_arena[i];
        else viv_arena_free(viv_arena[i]);
    }
    viv_arena_top = w;
}

/* The end of a loop iteration: the arrays the loop's
   variables hold are kept, everything else filed since the loop began is
   freed. So a loop allocating each time round runs in the memory of one
   iteration, as it does in the direct wasm runtime (vrt_release_kept). A def
   returning a record keeps the record's arrays the same way. */
static VIV_TLS const void** viv_kept = 0;
static VIV_TLS size_t viv_kept_n = 0, viv_kept_cap = 0;
static void viv_arena_keep(const void* p) {
    if (viv_kept_n == viv_kept_cap) {
        viv_kept_cap = viv_kept_cap ? 2 * viv_kept_cap : 16;
        viv_kept = (const void**)realloc((void*)viv_kept, viv_kept_cap * sizeof(void*));
    }
    viv_kept[viv_kept_n++] = p;
}
/* the number of this release pass; an enum node marked with it is kept */
static VIV_TLS unsigned viv_en_pass = 1;
static void viv_arena_release_kept(size_t mark) {
    VIV_POLL();
    size_t w = mark;
    for (size_t i = mark; i < viv_arena_top; i++) {
        int keep = 0;
        if (((size_t)viv_arena[i] & 3) == 3) {
            viv_en_head* h = (viv_en_head*)((char*)viv_arena[i] - 3);
            if (h->keep == viv_en_pass) { keep = 1; h->pos = (long)w; }
        } else {
            for (size_t k = 0; k < viv_kept_n; k++) {
                if (viv_kept[k] == viv_arena[i]) { keep = 1; break; }
            }
        }
        if (keep) viv_arena[w++] = viv_arena[i];
        else viv_arena_free(viv_arena[i]);
    }
    viv_arena_top = w;
    viv_kept_n = 0;
    viv_en_pass++;
}

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
static double viv_fma(double x, double y, double z)
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

#if !defined(VIV_RELEASE) && !defined(VIV_NO_BOUNDS)
#define VIV_BOUNDS 1
/* Where each checked array access is in the source: the lines under the
   error, as the interpreter prints them. Row 0 is an access with no source. */
static const char* const viv_sites[] = {
    "",
    "  --> features.viv:36:31\n  36 |         while stop_at < n and card[stop_at] == card[start] {\n                                     ^^^^^^^^^^^^^ here\n",
    "  --> features.viv:36:48\n  36 |         while stop_at < n and card[stop_at] == card[start] {\n                                                      ^^^^^^^^^^^ here\n",
    "  --> features.viv:37:16\n  37 |             if ts[stop_at] < ts[stop_at - 1] {\n                      ^^^^^^^^^^^ here\n",
    "  --> features.viv:37:30\n  37 |             if ts[stop_at] < ts[stop_at - 1] {\n                                    ^^^^^^^^^^^^^^^ here\n",
    "  --> features.viv:38:22\n  38 |                 stop(f\"card_features: rows must be sorted by (card, ts); row {stop_at} goes back in time\")\n                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ here\n",
    "  --> features.viv:38:17\n  38 |                 stop(f\"card_features: rows must be sorted by (card, ts); row {stop_at} goes back in time\")\n                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ here\n",
    "  --> features.viv:43:18\n  43 |             stop(f\"card_features: rows must be sorted by (card, ts); card falls at row {stop_at}\")\n                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ here\n",
    "  --> features.viv:43:13\n  43 |             stop(f\"card_features: rows must be sorted by (card, ts); card falls at row {stop_at}\")\n                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ here\n",
    "  --> features.viv:54:17\n  54 |             t = ts[i]\n                       ^^^^^ here\n",
    "  --> features.viv:56:38\n  56 |             while same < stop_at and ts[same] == t {\n                                            ^^^^^^^^ here\n",
    "  --> features.viv:59:19\n  59 |             while ts[left_1h] < t - HOUR {\n                         ^^^^^^^^^^^ here\n",
    "  --> features.viv:59:33\n  59 |             while ts[left_1h] < t - HOUR {\n                                       ^^^^^^^^ here\n",
    "  --> features.viv:62:19\n  62 |             while ts[left_24h] < t - DAY {\n                         ^^^^^^^^^^^^ here\n",
    "  --> features.viv:62:34\n  62 |             while ts[left_24h] < t - DAY {\n                                        ^^^^^^^ here\n",
    "  --> features.viv:63:41\n  63 |                 declines_before_left += declined[left_24h]\n                                               ^^^^^^^^^^^^^^^^^^ here\n",
    "  --> features.viv:63:17\n  63 |                 declines_before_left += declined[left_24h]\n                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ here\n",
    "  --> features.viv:66:27\n  66 |             if locked and t - last_bad >= CLEAN {\n                                 ^^^^^^^^^^^^ here\n",
    "  --> features.viv:73:21\n  73 |             else if declines_seen - declines_before_left >= WATCH_DECLINES {\n                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ here\n",
    "  --> features.viv:77:17\n  77 |                 n_prev[k] = i - left_1h\n                       ^^^^^^^^^^^^^^^^^^^^^^^ here\n",
    "  --> features.viv:78:17\n  78 |                 status[k] = state\n                       ^^^^^^^^^^^^^^^^^ here\n",
    "  --> features.viv:82:34\n  82 |                 declines_seen += declined[k]\n                                        ^^^^^^^^^^^ here\n",
    "  --> features.viv:82:17\n  82 |                 declines_seen += declined[k]\n                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ here\n",
    "  --> features.viv:83:20\n  83 |                 if chargeback[k] != 0 {\n                          ^^^^^^^^^^^^^ here\n",
    "  --> churn.viv:5648:21\n5648 |     if numel(xf) != rows * NF or len(xi) != rows * NI {\n                           ^^^^^^^^^ here\n",
    "  --> churn.viv:5648:45\n5648 |     if numel(xf) != rows * NF or len(xi) != rows * NI {\n                                                   ^^^^^^^^^ here\n",
    "  --> churn.viv:5649:14\n5649 |         stop(f\"score: {rows} rows need {rows * NF} floats and {rows * NI} integers; got {numel(xf)} and {len(xi)}\")\n                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ here\n",
    "  --> churn.viv:5649:9\n5649 |         stop(f\"score: {rows} rows need {rows * NF} floats and {rows * NI} integers; got {numel(xf)} and {len(xi)}\")\n               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ here\n",
    "  --> churn.viv:5651:22\n5651 |     card = int_zeros(rows)\n                            ^^^^ here\n",
    "  --> churn.viv:5656:19\n5656 |         card[i] = xi[i * NI + 0]\n                         ^^^^^^^^^^^^^^ here\n",
    "  --> churn.viv:5656:9\n5656 |         card[i] = xi[i * NI + 0]\n               ^^^^^^^^^^^^^^^^^^^^^^^^ here\n",
    "  --> churn.viv:5657:17\n5657 |         ts[i] = xi[i * NI + 1]\n                       ^^^^^^^^^^^^^^ here\n",
    "  --> churn.viv:5657:9\n5657 |         ts[i] = xi[i * NI + 1]\n               ^^^^^^^^^^^^^^^^^^^^^^ here\n",
    "  --> churn.viv:5658:23\n5658 |         declined[i] = xi[i * NI + 3]\n                             ^^^^^^^^^^^^^^ here\n",
    "  --> churn.viv:5658:9\n5658 |         declined[i] = xi[i * NI + 3]\n               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ here\n",
    "  --> churn.viv:5659:25\n5659 |         chargeback[i] = xi[i * NI + 4]\n                               ^^^^^^^^^^^^^^ here\n",
    "  --> churn.viv:5659:9\n5659 |         chargeback[i] = xi[i * NI + 4]\n               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ here\n",
    "  --> churn.viv:5668:20\n5668 |         r_amount = xf[i, 0]\n                          ^^^^^^^^ here\n",
    "  --> churn.viv:5670:18\n5670 |             stop(f\"row {i}: column amount is NaN, which this pipeline refuses (scikit-learn raises ValueError)\")\n                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ here\n",
    "  --> churn.viv:5670:13\n5670 |             stop(f\"row {i}: column amount is NaN, which this pipeline refuses (scikit-learn raises ValueError)\")\n                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ here\n",
    "  --> churn.viv:5673:18\n5673 |             stop(f\"row {i}: column amount is infinite, which this pipeline refuses (scikit-learn raises ValueError)\")\n                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ here\n",
    "  --> churn.viv:5673:13\n5673 |             stop(f\"row {i}: column amount is infinite, which this pipeline refuses (scikit-learn raises ValueError)\")\n                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ here\n",
    "  --> churn.viv:5675:27\n5675 |         r_merchant: Int = xi[i * NI + 2]\n                                 ^^^^^^^^^^^^^^ here\n",
    "  --> churn.viv:5678:28\n5678 |         r_n_prev_1h: Int = f0.n_prev_1h[i]\n                                  ^^^^^^^^^^^^^^^ here\n",
    "  --> churn.viv:5679:25\n5679 |         r_status: Int = f0.status[i]\n                               ^^^^^^^^^^^^ here\n",
    "  --> churn.viv:5719:17\n5719 |         raw13 = INIT_RAW[0]\n                       ^^^^^^^^^^^ here\n",
    "  --> churn.viv:5877:9\n5877 |         out[i, 0] = p15\n               ^^^^^^^^^^^^^^^ here\n",
    "  --> churn.viv:5878:9\n5878 |         out[i, 1] = p14\n               ^^^^^^^^^^^^^^^ here\n",
    "  --> churn.viv:5879:9\n5879 |         out[i, 2] = float(cls16)\n               ^^^^^^^^^^^^^^^^^^^^^^^^ here\n",
};
#endif

/* C's %g without the C library's %g (printfmt.rs, line for line) */
/* %.<p>g from %.<p-1>e and %.<p>f, not from the C library's %g: macOS's
   viv_printf("%g", 7890305.0) writes 7.89030e+06, keeping a trailing zero when the
   rounding is a tie, where C (and every other engine) says 7.8903e+06 */
static void viv_trim(char* m) {
    if (!strchr(m, '.')) return;
    size_t n = strlen(m);
    while (m[n - 1] == '0') m[--n] = 0;
    if (m[n - 1] == '.') m[--n] = 0;
}
/* the exponent %e wrote: atoi without <stdlib.h>, whose declarations (wait,
   exit, abs, ...) a program's own defs may be named like */
static int viv_exp10(const char* c) {
    int neg = *c == '-', v = 0;
    if (*c == '-' || *c == '+') c++;
    while (*c >= '0' && *c <= '9') v = v * 10 + (*c++ - '0');
    return neg ? -v : v;
}
static void viv_fmt_g(char* b, int sz, double x, int p) {
    char s[64];
    snprintf(s, sizeof s, "%.*e", p - 1, x);
    char* e = strchr(s, 'e');
    int ex = viv_exp10(e + 1);
    if (ex < -4 || ex >= p) {
        *e = 0;
        viv_trim(s);
        snprintf(b, sz, "%se%c%02d", s, ex < 0 ? '-' : '+', ex < 0 ? -ex : ex);
    } else {
        snprintf(b, sz, "%.*f", p - 1 - ex, x);
        viv_trim(b);
    }
}
static const char* viv_gp(double x, int p) {
    static char bufs[8][64];
    static int k = 0;
    char* b = bufs[k = (k + 1) & 7];
    if (x != x) return "nan";
    if (x > DBL_MAX || x < -DBL_MAX) return x < 0 ? "-inf" : "inf";
    viv_fmt_g(b, 64, x, p < 1 ? 1 : p);
    return b;
}
/* a whole number below 10^16, which print and str write as its digits
   (printfmt.rs, whole) */
static int viv_whole(double x) { return x < 1e16 && x > -1e16 && x == (double)(long long)x; }
static const char* viv_g(double x) {
    static char bufs[8][32];
    static int k = 0;
    if (x == 0.0) return "0";
    if (viv_whole(x)) {
        char* b = bufs[k = (k + 1) & 7];
        snprintf(b, sizeof bufs[0], "%.0f", x);
        return b;
    }
    return viv_gp(x, 6);
}
/* a number at full precision: the shortest of %.15g, %.16g, %.17g that reads
   back as the same double (the interpreter's format_rt); how an index that is
   not an integer is shown in its error */
static void viv_rt_text(char* b, int sz, double x) {
    x += 0.0;
    if (x != x) { snprintf(b, sz, "nan"); return; }
    if (x > DBL_MAX || x < -DBL_MAX) { snprintf(b, sz, x < 0 ? "-inf" : "inf"); return; }
    if (x == 0.0) { snprintf(b, sz, "0"); return; }
    for (int p = 15; p <= 16; p++) {
        double y;
        viv_fmt_g(b, sz, x, p);
        if (sscanf(b, "%lf", &y) == 1 && y == x) return;
    }
    viv_fmt_g(b, sz, x, 17);
}

/* Bounds checks: every array read and write goes through viv_ix, which
   checks one index against its dimension, so an index out of bounds stops
   the program instead of reading or writing memory it does not own. The
   words and the location are the interpreter's, byte for byte.
   An index is a double that must be exactly an integer: a fraction, NaN or
   an infinity stops the program ("index 2.5 is not an integer"), as in
   the interpreter (interp.rs, index_int). An integer is then read as Rust's
   `as i64` reads it, saturating at the ends of the 64-bit range. A negative
   index counts from the end, as in Python: a[-1] is the last element.
   -DVIV_RELEASE compiles the checks out: viv_ix is then an (int) cast and
   the step from the end for a negative index. -DVIV_NO_BOUNDS
   drops only these checks and keeps the rest of a debug build (the solver
   certificates, the story), which is how their cost is measured
   (tools/bounds-cost.sh). */
#if defined(__GNUC__)
/* The accessors are always inlined: an accessor that is called takes its
   viv_arr (48 bytes) by value, and with the checks in it the compiler
   stopped inlining some of them, which alone made scf/coverage.viv take
   1.8 times as long. The failing path is cold and never returns. */
#define VIV_ACCESS static inline __attribute__((always_inline, unused))
#define VIV_COLD __attribute__((cold, noinline, unused))
#define VIV_LIKELY(c) __builtin_expect(!!(c), 1)
#else
#define VIV_ACCESS static inline
#define VIV_COLD
#define VIV_LIKELY(c) (c)
#endif
#ifndef VIV_BOUNDS
/* (void)(site): the site of an accessor is still passed, and read, so a
   release build has no unused parameter to warn about */
static inline int viv_ix_wrap(int i, int n) { return i < 0 ? i + n : i; }
/* unchecked, but never undefined: the double is clamped into int range
   before it is converted (fmax sends NaN to INT_MIN) */
#define viv_ix(x, n, k, site) ((void)(site), viv_ix_wrap((int)fmin(fmax((x), -2147483648.0), 2147483647.0), (n)))
#define viv_rank(have, want, site) ((void)(site))
/* an index computed as a 64-bit integer (lower/intidx.rs) */
static inline int viv_ixl_wrap(long long i, int n) { return (int)(i < 0 ? i + n : i); }
#define viv_ixl(i, n, k, site) ((void)(site), viv_ixl_wrap((i), (n)))
#define viv_ixp(m, k, i, n, site) (((m) >> (k)) & 1 ? (int)(i) : viv_ixl((i), (n), (k), (site)))
#else
/* An array indexed with the wrong number of indices: the interpreter checks
   the count before it reads any index, so this comes first in every
   accessor, and a[5] on a 2 x 3 array stops with the rank, not an index. */
VIV_COLD static void viv_rank_slow(int have, int want, int site) {
    viv_fflush(stdout);
    viv_errf("error: array has rank %d but was indexed with %d index/indices\n%s", have, want, viv_sites[site]);
    viv_stop(VIV_E_INDEX);
}
#define viv_rank(have, want, site) do { if (!VIV_LIKELY((have) == (want))) viv_rank_slow((have), (want), (site)); } while (0)
/* An index that failed the quick test: NaN, or a number no int holds, or an
   int out of range. Read it as `as i64` does and stop if it is out. */
VIV_COLD static int viv_ix_slow(double x, int n, int k, int site) {
    if (x - floor(x) != 0.0) {
        char b[40];
        viv_rt_text(b, sizeof b, x);
        viv_fflush(stdout);
        viv_errf("error: index %s is not an integer\n%s", b, viv_sites[site]);
        viv_stop(VIV_E_INDEX);
    }
    long long i = x != x ? 0
                : x >= 9223372036854775807.0 ? LLONG_MAX
                : x < -9223372036854775807.0 ? LLONG_MIN
                : (long long)x;
    if (i >= 0 && i < n) return (int)i;
    /* a negative index counts from the end */
    if (i < 0 && i >= -(long long)n) return (int)(i + n);
    /* what the program printed comes first, as in the interpreter */
    viv_fflush(stdout);
    viv_errf("error: index %lld out of bounds for dimension %d (size %d)\n%s", i, k, n, viv_sites[site]);
    /* _exit, not exit: the run stopped, so no "N assertions FAILED" summary
       or story from the atexit handlers -- the interpreter prints neither */
    viv_stop(VIV_E_INDEX);
}
VIV_ACCESS int viv_ix(double x, int n, int k, int site) {
    /* The quick test is on the int, not the double: clamped into int range
       first (fmax also sends NaN to INT_MIN), the conversion is defined for
       every x and truncates toward zero as `as i64` does, and one unsigned
       compare then takes both ends. Written as two double compares instead,
       scf/coverage.viv took 1.5 times as long: the compiler turns a loop's
       double counter into an int only when it is indexed through an int. */
    int t = (int)fmin(fmax(x, -2147483648.0), 2147483647.0);
    if (VIV_LIKELY((unsigned)t < (unsigned)n && (double)t == x)) return t;
    return viv_ix_slow(x, n, k, site);
}
/* An index computed as a 64-bit integer (lower/intidx.rs): a loop counter,
   an Int, or a number the compiler proved is always an integer, so there is
   no fraction to test for, and one unsigned compare takes both ends. An Int
   past 2^53 stops as the interpreter stops converting it to a number. */
VIV_COLD static int viv_ixl_slow(long long i, int n, int k, int site) {
    viv_fflush(stdout);
    if (i > 9007199254740992LL || i < -9007199254740992LL) {
        viv_errf("error: converting %lld to a number would change it; write float(n) to round it\n%s", i, viv_sites[site]);
        viv_stop(VIV_E_OVERFLOW);
    }
    if (i < 0 && i >= -(long long)n) return (int)(i + n);
    viv_errf("error: index %lld out of bounds for dimension %d (size %d)\n%s", i, k, n, viv_sites[site]);
    viv_stop(VIV_E_INDEX);
}
VIV_ACCESS int viv_ixl(long long i, int n, int k, int site) {
    if (VIV_LIKELY((unsigned long long)i < (unsigned long long)(long long)n)) return (int)i;
    return viv_ixl_slow(i, n, k, site);
}
/* bit k of m: the compiler proved index k inside its dimension (0 <= i <
   n, lower/intidx.rs), so it is not tested */
#define viv_ixp(m, k, i, n, site) (((m) >> (k)) & 1 ? (int)(i) : viv_ixl((i), (n), (k), (site)))
#endif

/* Every array's block is made here. The element count is multiplied out in
   64 bits and may not pass INT_MAX (the struct's `size`): an index that
   passed its dimension's check then gives an offset below `size`, so no
   offset overflows. `zeros` and the array vocabulary refuse an
   array past that first, with their own words (2147483647 elements, as the
   interpreter does natively; the wasm engines stop at 16,000,000); this is
   the last check. A failed calloc stops the run instead of handing out a null
   block. `width` is 1 for numbers, 2 for complex numbers. */
static void viv_arr_fail(int kind, const char* m, long long n) {
    viv_fflush(stdout);
    viv_errf(m, n);
    viv_stop(kind);
}
static viv_arr viv_arr_alloc(int rank, const int* dims, int width) {
    viv_arr a; a.rank = rank; a.dims[7] = 0;
    long long n = 1;
    for (int i = 0; i < rank; i++) {
        a.dims[i] = dims[i] < 0 ? 0 : dims[i];
        n *= a.dims[i];
        if (n > 2147483647LL) viv_arr_fail(VIV_E_OVERFLOW, "error: an array of more than %lld elements cannot be made\n", 2147483647LL);
    }
    a.size = (int)n;
    a.data = (double*)calloc((size_t)n * (size_t)width, sizeof(double));
    if (!a.data && n > 0) viv_arr_fail(VIV_E_MEMORY, "error: out of memory making an array of %lld elements\n", n);
    viv_arena_push(a.data);
    return a;
}
static viv_arr viv_arr_new(int rank, const int* dims) { return viv_arr_alloc(rank, dims, 1); }
/* the dimensions `zeros` and `czeros` are given, read as the interpreter
   reads them (interp.rs, zeros_shape): a negative or NaN one is 0, a
   fraction is cut, a huge one saturates; the element count, a saturating
   64-bit product, is checked against the limit before anything is made */
static void viv_zeros_dims(int rank, const double* x, int* d, int width) {
    unsigned long long total = 1;
    for (int i = 0; i < rank; i++) {
        double v = x[i];
        unsigned long long u = !(v > 0) ? 0 : v >= 18446744073709551615.0 ? 18446744073709551615ULL : (unsigned long long)v;
        total = (u != 0 && total > 18446744073709551615ULL / u) ? 18446744073709551615ULL : total * u;
        d[i] = u > 2147483647ULL ? 2147483647 : (int)u;
    }
    if (width == 1 && total > 2147483647ULL) {
        viv_fflush(stdout);
        viv_errf("error: zeros: array of %llu elements exceeds the 2147483647-element limit\n", total);
        viv_stop(VIV_E_OVERFLOW);
    }
    if (width == 2 && total > 1073741823ULL) {
        viv_fflush(stdout);
        viv_errf("error: czeros: complex array of %llu elements exceeds the 1073741823-element limit\n", total);
        viv_stop(VIV_E_OVERFLOW);
    }
}

static viv_arr viv_ivec(int n, ...) {
    int dims[1]; dims[0] = n;
    viv_arr a = viv_arr_new(1, dims);
    va_list ap; va_start(ap, n);
    for (int i = 0; i < n; i++) ((long long*)a.data)[i] = va_arg(ap, long long);
    va_end(ap);
    return a;
}
VIV_ACCESS long long viv_iget_at(viv_arr a, double i, int site) {
    viv_rank(a.rank, 1, site);
    return ((long long*)a.data)[viv_ix(i, a.dims[0], 0, site)];
}
VIV_ACCESS long long viv_iget(viv_arr a, double i) { return viv_iget_at(a, i, 0); }
VIV_ACCESS void viv_iset_at(viv_arr a, double i, long long v, int site) {
    viv_rank(a.rank, 1, site);
    ((long long*)a.data)[viv_ix(i, a.dims[0], 0, site)] = v;
}
VIV_ACCESS void viv_iset(viv_arr a, double i, long long v) { viv_iset_at(a, i, v, 0); }
/* the same with an index that is an Int: no trip through a double */
VIV_ACCESS long long viv_iget_l_at(viv_arr a, long long i, int site) {
    viv_rank(a.rank, 1, site);
    return ((long long*)a.data)[viv_ixl(i, a.dims[0], 0, site)];
}
VIV_ACCESS void viv_iset_l_at(viv_arr a, long long i, long long v, int site) {
    viv_rank(a.rank, 1, site);
    ((long long*)a.data)[viv_ixl(i, a.dims[0], 0, site)] = v;
}
VIV_ACCESS long long viv_iget_l(viv_arr a, long long i) { return viv_iget_l_at(a, i, 0); }
/* a number index with an exact integer form (lower/intidx.rs): as an Int,
   with bit 0 of m set when the compiler proved it inside the array */
VIV_ACCESS long long viv_iget_m_at(viv_arr a, long long i, int m, int site) {
    viv_rank(a.rank, 1, site);
    return ((long long*)a.data)[viv_ixp(m, 0, i, a.dims[0], site)];
}
VIV_ACCESS void viv_iset_m_at(viv_arr a, long long i, long long v, int m, int site) {
    viv_rank(a.rank, 1, site);
    ((long long*)a.data)[viv_ixp(m, 0, i, a.dims[0], site)] = v;
}
VIV_ACCESS long long viv_iget_m(viv_arr a, long long i, int m) { return viv_iget_m_at(a, i, m, 0); }
VIV_ACCESS void viv_iset_m(viv_arr a, long long i, long long v, int m) { viv_iset_m_at(a, i, v, m, 0); }
VIV_ACCESS void viv_iset_l(viv_arr a, long long i, long long v) { viv_iset_l_at(a, i, v, 0); }

/* Int: 64-bit, checked; every message is
   interp/ints.rs's, and in a checked build the location follows it */
#if defined(__GNUC__)
#define VIV_I_UNUSED __attribute__((unused))
#define VIV_I_COLD __attribute__((noinline, cold))
#else
#define VIV_I_UNUSED
#define VIV_I_COLD
#endif
VIV_I_UNUSED static void viv_i_fail(int kind, const char* m, int site) {
    viv_fflush(stdout);
#ifdef VIV_BOUNDS
    viv_errf("error: %s\n%s", m, viv_sites[site]);
#else
    (void)site;
    viv_errf("error: %s\n", m);
#endif
    viv_stop(kind);
}
VIV_I_UNUSED static void viv_i_ovf(long long a, const char* op, long long b, int site) {
    char m[160];
    snprintf(m, sizeof m, "integer overflow: %lld %s %lld does not fit in an Int", a, op, b);
    viv_i_fail(VIV_E_OVERFLOW, m, site);
}
VIV_I_UNUSED static void viv_i_ovf1(const char* what, long long a, int site) {
    char m[160];
    snprintf(m, sizeof m, "integer overflow: %s(%lld) does not fit in an Int", what, a);
    viv_i_fail(VIV_E_OVERFLOW, m, site);
}
#if defined(__GNUC__) || defined(__clang__)
VIV_I_UNUSED static long long viv_i_add(long long a, long long b, int site) { long long r; if (__builtin_add_overflow(a, b, &r)) viv_i_ovf(a, "+", b, site); return r; }
VIV_I_UNUSED static long long viv_i_sub(long long a, long long b, int site) { long long r; if (__builtin_sub_overflow(a, b, &r)) viv_i_ovf(a, "-", b, site); return r; }
VIV_I_UNUSED static int viv_i_mul_ovf(long long a, long long b, long long* r) { return __builtin_mul_overflow(a, b, r); }
#else
VIV_I_UNUSED static long long viv_i_add(long long a, long long b, int site) { if ((b > 0 && a > LLONG_MAX - b) || (b < 0 && a < LLONG_MIN - b)) viv_i_ovf(a, "+", b, site); return a + b; }
VIV_I_UNUSED static long long viv_i_sub(long long a, long long b, int site) { if ((b < 0 && a > LLONG_MAX + b) || (b > 0 && a < LLONG_MIN + b)) viv_i_ovf(a, "-", b, site); return a - b; }
VIV_I_UNUSED static int viv_i_mul_ovf(long long a, long long b, long long* r) {
    if (a == 0 || b == 0) { *r = 0; return 0; }
    if (a == -1) { if (b == LLONG_MIN) return 1; *r = -b; return 0; }
    if (b == -1) { if (a == LLONG_MIN) return 1; *r = -a; return 0; }
    if (a > 0 ? (b > 0 ? a > LLONG_MAX / b : b < LLONG_MIN / a) : (b > 0 ? a < LLONG_MIN / b : a < LLONG_MAX / b)) return 1;
    *r = a * b;
    return 0;
}
#endif
VIV_I_UNUSED static long long viv_i_mul(long long a, long long b, int site) { long long r; if (viv_i_mul_ovf(a, b, &r)) viv_i_ovf(a, "*", b, site); return r; }
VIV_I_UNUSED static long long viv_i_fdiv(long long a, long long b, int site) {
    if (b == 0) viv_i_fail(VIV_E_ZERODIV, "division by zero in `a // b`", site);
    if (a == LLONG_MIN && b == -1) viv_i_ovf(a, "//", b, site);
    long long q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}
/* the ceiling of a / b, exactly: -((-a) // b) without forming -a */
VIV_I_UNUSED static long long viv_i_cdiv(long long a, long long b, int site) {
    if (b == 0) viv_i_fail(VIV_E_ZERODIV, "division by zero in `a / b`", site);
    if (a == LLONG_MIN && b == -1) viv_i_ovf(a, "/", b, site);
    long long q = a / b;
    return (a % b != 0 && ((a < 0) == (b < 0))) ? q + 1 : q;
}
VIV_I_UNUSED static long long viv_i_fmod(long long a, long long b, int site) {
    if (b == 0) viv_i_fail(VIV_E_ZERODIV, "division by zero in `a % b`", site);
    if (b == -1) return 0;
    long long r = a % b;
    return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r;
}
VIV_I_UNUSED static long long viv_i_pow(long long a, long long k, int site) {
    if (k < 0) viv_i_fail(VIV_E_VALUE, "an Int to a negative power is not an Int; write float(n) ^ k", site);
    long long r = 1, b = a, e = k;
    while (e > 0) {
        if (e & 1) { if (viv_i_mul_ovf(r, b, &r)) viv_i_ovf(a, "^", k, site); }
        e >>= 1;
        if (e > 0) { if (viv_i_mul_ovf(b, b, &b)) viv_i_ovf(a, "^", k, site); }
    }
    return r;
}
VIV_I_UNUSED static long long viv_i_neg(long long a, int site) { if (a == LLONG_MIN) viv_i_ovf1("-", a, site); return -a; }
VIV_I_UNUSED static long long viv_i_abs(long long a, int site) { if (a == LLONG_MIN) viv_i_ovf1("abs", a, site); return a < 0 ? -a : a; }
VIV_I_UNUSED static double viv_i_num(long long a, int site) {
    if (a > 9007199254740992LL || a < -9007199254740992LL) {
        char m[160];
        snprintf(m, sizeof m, "converting %lld to a number would change it; write float(n) to round it", a);
        viv_i_fail(VIV_E_OVERFLOW, m, site);
    }
    return (double)a;
}
/* the messages of viv_i_of, out of line so that the check itself inlines
   into a hot loop (`l: Int = int(max(g1, g2))` once per sample in NPB EP) */
VIV_I_UNUSED VIV_I_COLD static void viv_i_of_fail(double x, int site) {
    char b[40], m[96];
    if (x != x) viv_i_fail(VIV_E_VALUE, "cannot convert float NaN to integer", site);
    if (x > DBL_MAX || x < -DBL_MAX) viv_i_fail(VIV_E_OVERFLOW, "cannot convert float infinity to integer", site);
    viv_rt_text(b, sizeof b, x);
    if (x - floor(x) != 0.0) { snprintf(m, sizeof m, "%s is not an integer", b); viv_i_fail(VIV_E_VALUE, m, site); }
    snprintf(m, sizeof m, "%s does not fit in an Int", b);
    viv_i_fail(VIV_E_OVERFLOW, m, site);
}
VIV_I_UNUSED static inline long long viv_i_of(double x, int site) {
    /* NaN fails every comparison and an infinity the range check */
    if (!(x - floor(x) == 0.0 && x >= -9223372036854775808.0 && x < 9223372036854775808.0)) viv_i_of_fail(x, site);
    return (long long)x;
}
VIV_I_UNUSED static double viv_i_cmp(long long a, double x) {
    if (x != x) return x;
    if (a <= 9007199254740992LL && a >= -9007199254740992LL) { double y = (double)a; return y < x ? -1.0 : y > x ? 1.0 : 0.0; }
    if (x >= 9223372036854775808.0) return -1.0;
    if (x < -9223372036854775808.0) return 1.0;
    long long b = (long long)x;
    return a < b ? -1.0 : a > b ? 1.0 : 0.0;
}
VIV_I_UNUSED static double viv_i_float(long long a) { return (double)a; }
VIV_I_UNUSED static long long viv_i_min(long long a, long long b) { return b < a ? b : a; }
VIV_I_UNUSED static long long viv_i_max(long long a, long long b) { return b > a ? b : a; }
VIV_I_UNUSED static long long viv_i_band(long long a, long long b) { return a & b; }
VIV_I_UNUSED static long long viv_i_bor(long long a, long long b) { return a | b; }
VIV_I_UNUSED static long long viv_i_bxor(long long a, long long b) { return a ^ b; }
VIV_I_UNUSED static long long viv_i_bnot(long long a) { return ~a; }
VIV_I_UNUSED static void viv_i_shift_fail(const char* name, long long k, int site) {
    char m[96];
    snprintf(m, sizeof m, "`%s` takes a shift count of 0 or more, not %lld", name, k);
    viv_i_fail(VIV_E_VALUE, m, site);
}
VIV_I_UNUSED static long long viv_i_shl(long long a, long long k, int site) { if (k < 0) viv_i_shift_fail("shl", k, site); return k >= 64 ? 0 : (long long)((unsigned long long)a << k); }
VIV_I_UNUSED static long long viv_i_shr(long long a, long long k, int site) { if (k < 0) viv_i_shift_fail("shr", k, site); return k >= 64 ? 0 : (long long)((unsigned long long)a >> k); }
VIV_I_UNUSED static long long viv_i_sar(long long a, long long k, int site) { if (k < 0) viv_i_shift_fail("sar", k, site); return k >= 64 ? (a < 0 ? -1 : 0) : (a < 0 ? ~(~a >> k) : a >> k); }
VIV_I_UNUSED static long long viv_i_wrap_add(long long a, long long b) { return (long long)((unsigned long long)a + (unsigned long long)b); }
VIV_I_UNUSED static long long viv_i_wrap_sub(long long a, long long b) { return (long long)((unsigned long long)a - (unsigned long long)b); }
VIV_I_UNUSED static long long viv_i_wrap_mul(long long a, long long b) { return (long long)((unsigned long long)a * (unsigned long long)b); }
/* the high half of the 128-bit product of two unsigned 64-bit patterns, from
   32-bit halves (no __int128 needed) */
VIV_I_UNUSED static long long viv_i_umulhi(long long a, long long b) {
    unsigned long long x = (unsigned long long)a, y = (unsigned long long)b;
    unsigned long long x0 = x & 0xffffffffULL, x1 = x >> 32, y0 = y & 0xffffffffULL, y1 = y >> 32;
    unsigned long long p00 = x0 * y0, p01 = x0 * y1, p10 = x1 * y0, p11 = x1 * y1;
    unsigned long long mid = (p00 >> 32) + (p01 & 0xffffffffULL) + (p10 & 0xffffffffULL);
    return (long long)(p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32));
}
VIV_I_UNUSED static double viv_i_ult(long long a, long long b) { return (unsigned long long)a < (unsigned long long)b ? 1.0 : 0.0; }

static double viv_len(viv_arr a) { return (double)a.dims[0]; }

VIV_ACCESS double viv_lget1_at(viv_arr a, long long i, int m, int site) {
    viv_rank(a.rank, 1, site);
    return a.data[viv_ixp(m, 0, i, a.dims[0], site)];
}
VIV_ACCESS double viv_lget1(viv_arr a, long long i, int m) { return viv_lget1_at(a, i, m, 0); }

VIV_ACCESS double viv_lget2_at(viv_arr a, long long i, long long j, int m, int site) {
    viv_rank(a.rank, 2, site);
    int ii = viv_ixp(m, 0, i, a.dims[0], site); int jj = viv_ixp(m, 1, j, a.dims[1], site);
    return a.data[ii * a.dims[1] + jj];
}
VIV_ACCESS double viv_lget2(viv_arr a, long long i, long long j, int m) { return viv_lget2_at(a, i, j, m, 0); }

VIV_ACCESS void viv_lset2_at(viv_arr a, long long i, long long j, double v, int m, int site) {
    viv_rank(a.rank, 2, site);
    int ii = viv_ixp(m, 0, i, a.dims[0], site); int jj = viv_ixp(m, 1, j, a.dims[1], site);
    a.data[ii * a.dims[1] + jj] = v;
}
VIV_ACCESS void viv_lset2(viv_arr a, long long i, long long j, double v, int m) { viv_lset2_at(a, i, j, v, m, 0); }

static double viv_numel(viv_arr a) { return (double)a.size; }

#if defined(__GNUC__)
#define VIV_S_UNUSED __attribute__((unused))
#define VIV_S_NORETURN __attribute__((noreturn, unused))
#else
#define VIV_S_UNUSED
#define VIV_S_NORETURN
#endif
/* A string: immutable UTF-8. `owner` is the arena block it lives in (itself,
   a list's chunk, or 0 for a literal); `len` is bytes, `nch` code points;
   (ci, cb) caches the code point last found and its byte offset. */
typedef struct viv_strb {
    void* owner;
    int len, nch;
    int ci, cb;
    char s[];
} viv_strb;
typedef viv_strb* viv_str;
/* the bytes of the strings a list holds, never moved; each twice the last */
typedef struct viv_chunk {
    struct viv_chunk* prev;
    long cap, used;
    long pad;
    char data[];
} viv_chunk;
/* A list of strings: `items` point into its chunks (or at literals).
   `unsealed`: an append made it and nothing else holds it (crate::growable). */
typedef struct viv_strsb {
    int n, cap;
    int unsealed, pad;
    viv_chunk* chunk;
    viv_str items[];
} viv_strsb;
typedef viv_strsb* viv_strs;
#define VIV_S_MAX 1073741823L
VIV_S_NORETURN static void viv_s_fail(int kind, const char* msg, int site) {
    viv_fflush(stdout);
#ifdef VIV_BOUNDS
    viv_errf("error: %s\n%s", msg, viv_sites[site]);
#else
    (void)site;
    viv_errf("error: %s\n", msg);
#endif
    viv_stop(kind);
}
/* a format's width and precision, which may be values ({x:.{p}f}): whole
   numbers, the width from 0 to 100 and the precision to maxp (40 for a
   number, 2^30 for a string); a precision of -1 is a string format's none
   (printfmt.rs Fmt::with_limits) */
VIV_S_UNUSED static void viv_s_fmtwp(double prec, double width, double maxp, int site) {
    char msg[160];
    if (!(width >= 0.0 && width <= 100.0 && width == (double)(int)width)) {
        snprintf(msg, sizeof msg, "a format's width is %s; it must be a whole number from 0 to 100", viv_g(width));
        viv_s_fail(VIV_E_VALUE, msg, site);
    }
    if (prec != -1.0 && !(prec >= 0.0 && prec <= maxp && prec == (double)(long)prec)) {
        snprintf(msg, sizeof msg, "a format's precision is %s; it must be a whole number from 0 to %.0f", viv_g(prec), maxp);
        viv_s_fail(VIV_E_VALUE, msg, site);
    }
}
VIV_S_NORETURN static void viv_s_nomem(void) {
    viv_fflush(stdout);
    viv_errf("error: out of memory making a string\n");
    viv_stop(VIV_E_MEMORY);
}
VIV_S_UNUSED static void viv_s_too_long(unsigned long long n, int site) {
    char msg[160];
    (void)n;
    snprintf(msg, sizeof msg, "the string would be longer than Viv's limit of %ld bytes", VIV_S_MAX);
    viv_s_fail(VIV_E_OVERFLOW, msg, site);
}
VIV_S_UNUSED static int viv_s_nchars(const char* b, long n) {
    int k = 0;
    for (long i = 0; i < n; i++) k += ((unsigned char)b[i] & 0xC0) != 0x80;
    return k;
}
/* a new string of `len` bytes, on the arena; the caller fills s[0..len) */
VIV_S_UNUSED static viv_str viv_s_alloc(long len, int site) {
    if (len > VIV_S_MAX) viv_s_too_long((unsigned long long)len, site);
    viv_str r = (viv_str)malloc(sizeof(viv_strb) + (size_t)len + 1);
    if (!r) viv_s_nomem();
    r->owner = r; r->len = (int)len; r->nch = 0; r->ci = 0; r->cb = 0; r->s[len] = 0;
    viv_arena_push(r);
    return r;
}
/* the string whose bytes are b[0..len) */
VIV_S_UNUSED static viv_str viv_s_make(const char* b, long len, int site) {
    viv_str r = viv_s_alloc(len, site);
    if (len) memcpy(r->s, b, (size_t)len);
    r->nch = viv_s_nchars(r->s, len);
    return r;
}
/* a string for the rest of the run: a literal, a one-character string */
VIV_S_UNUSED static viv_str viv_s_static(const char* b, int len, int nch) {
    viv_str r = (viv_str)malloc(sizeof(viv_strb) + (size_t)len + 1);
    if (!r) viv_s_nomem();
    r->owner = 0; r->len = len; r->nch = nch; r->ci = 0; r->cb = 0;
    if (len) memcpy(r->s, b, (size_t)len);
    r->s[len] = 0;
    return r;
}
/* a literal, made when the run first reaches it (viv_lits is the str_lits fragment) */
#define VIV_LIT(k, text, len, nch) (viv_lits[k] ? viv_lits[k] : (viv_lits[k] = viv_s_static(text, len, nch)))
VIV_S_UNUSED static viv_str viv_s_empty(void) {
    static viv_str e = 0;
    return e ? e : (e = viv_s_static("", 0, 0));
}
/* the one-character string of an ASCII byte: made once */
VIV_S_UNUSED static viv_str viv_s_ascii(unsigned char c) {
    static viv_str t[128];
    if (!t[c]) { char b = (char)c; t[c] = viv_s_static(&b, 1, 1); }
    return t[c];
}
/* a string read as it is, or "" for a variable never assigned (a function,
   so an argument with a side effect, such as va_arg, is evaluated once) */
VIV_S_UNUSED static viv_str viv_s_or_empty(viv_str s) { return s ? s : viv_s_empty(); }
#define VIV_S(s) viv_s_or_empty(s)
/* the byte offset of code point ci (0 <= ci <= nch), from the nearest of
   the start, the cached point and the end; the point found is cached */
VIV_S_UNUSED static int viv_s_byteof(viv_str s, int ci) {
    if (s->nch == s->len) return ci;
    const unsigned char* b = (const unsigned char*)s->s;
    int cc = s->ci, cb = s->cb;
    if (cc > s->nch || cb > s->len) { cc = 0; cb = 0; }
    int from_start = ci, from_cache = ci > cc ? ci - cc : cc - ci, from_end = s->nch - ci;
    int k, p;
    if (from_start <= from_cache && from_start <= from_end) { k = 0; p = 0; }
    else if (from_cache <= from_end) { k = cc; p = cb; }
    else { k = s->nch; p = s->len; }
    while (k < ci) { p++; while (p < s->len && (b[p] & 0xC0) == 0x80) p++; k++; }
    while (k > ci) { p--; while (p > 0 && (b[p] & 0xC0) == 0x80) p--; k--; }
    s->ci = ci; s->cb = p;
    return p;
}
/* the code point at b[*i], and *i moved past it (valid UTF-8) */
VIV_S_UNUSED static unsigned viv_s_dec(const char* s, long* i) {
    const unsigned char* b = (const unsigned char*)s + *i;
    unsigned c = b[0];
    if (c < 0x80) { *i += 1; return c; }
    if (c < 0xE0) { *i += 2; return ((c & 0x1F) << 6) | (b[1] & 0x3F); }
    if (c < 0xF0) { *i += 3; return ((c & 0x0F) << 12) | ((b[1] & 0x3F) << 6) | (b[2] & 0x3F); }
    *i += 4;
    return ((c & 0x07) << 18) | ((b[1] & 0x3F) << 12) | ((b[2] & 0x3F) << 6) | (b[3] & 0x3F);
}
/* the code point that ends at b[*i] (exclusive), and *i moved to its start */
VIV_S_UNUSED static unsigned viv_s_dec_back(const char* s, long* i) {
    long j = *i - 1;
    while (j > 0 && ((unsigned char)s[j] & 0xC0) == 0x80) j--;
    long k = j;
    unsigned c = viv_s_dec(s, &k);
    *i = j;
    return c;
}
VIV_S_UNUSED static int viv_s_enc(unsigned c, char* o) {
    if (c < 0x80) { o[0] = (char)c; return 1; }
    if (c < 0x800) { o[0] = (char)(0xC0 | (c >> 6)); o[1] = (char)(0x80 | (c & 0x3F)); return 2; }
    if (c < 0x10000) { o[0] = (char)(0xE0 | (c >> 12)); o[1] = (char)(0x80 | ((c >> 6) & 0x3F)); o[2] = (char)(0x80 | (c & 0x3F)); return 3; }
    o[0] = (char)(0xF0 | (c >> 18)); o[1] = (char)(0x80 | ((c >> 12) & 0x3F)); o[2] = (char)(0x80 | ((c >> 6) & 0x3F)); o[3] = (char)(0x80 | (c & 0x3F));
    return 4;
}
/* whether c is in the sorted inclusive ranges t[0..n) */
VIV_S_UNUSED static int viv_s_inr(unsigned c, const unsigned (*t)[2], int n) {
    int lo = 0, hi = n;
    while (lo < hi) { int m = (lo + hi) / 2; if (t[m][0] <= c) lo = m + 1; else hi = m; }
    return lo > 0 && c <= t[lo - 1][1];
}
/* a string being built: bytes that grow by doubling, then one copy out */
typedef struct { char* p; long n, cap; int site; } viv_sb;
VIV_S_UNUSED static void viv_sb_room(viv_sb* b, long more) {
    if (b->n + more > VIV_S_MAX) viv_s_too_long((unsigned long long)(b->n + more), b->site);
    if (b->n + more <= b->cap) return;
    long cap = b->cap ? 2 * b->cap : 64;
    while (cap < b->n + more) cap *= 2;
    b->p = (char*)realloc(b->p, (size_t)cap);
    if (!b->p) viv_s_nomem();
    b->cap = cap;
}
VIV_S_UNUSED static void viv_sb_put(viv_sb* b, const char* s, long n) {
    viv_sb_room(b, n);
    if (n) memcpy(b->p + b->n, s, (size_t)n);
    b->n += n;
}
VIV_S_UNUSED static void viv_sb_cp(viv_sb* b, unsigned c) {
    char e[4];
    viv_sb_put(b, e, viv_s_enc(c, e));
}
VIV_S_UNUSED static void viv_sb_cstr(viv_sb* b, const char* s) { viv_sb_put(b, s, (long)strlen(s)); }
/* the string built, and the builder freed */
VIV_S_UNUSED static viv_str viv_sb_done(viv_sb* b) {
    viv_str r = viv_s_make(b->p ? b->p : "", b->n, b->site);
    free(b->p);
    b->p = 0; b->n = b->cap = 0;
    return r;
}
/* a number read as a position or a count: Rust's `as i64` */
VIV_S_UNUSED static long long viv_s_i64(double x) {
    return x != x ? 0 : x >= 9223372036854775807.0 ? LLONG_MAX : x < -9223372036854775807.0 ? LLONG_MIN : (long long)x;
}
/* Python's index into n items: negative from the end; -1 when out of range */
/* an index must be exactly an integer, as an array's must (interp.rs,
   index_int; the bounds fragment's viv_ix) */
VIV_S_UNUSED static void viv_s_whole(double x, int site) {
    if (x - floor(x) != 0.0) {
        char b[40], m[80];
        viv_rt_text(b, sizeof b, x);
        snprintf(m, sizeof m, "index %s is not an integer", b);
        viv_s_fail(VIV_E_INDEX, m, site);
    }
}
VIV_S_UNUSED static long viv_s_pyindex(long n, double x) {
    long long i = viv_s_i64(x);
    if (i < 0) i = i < LLONG_MIN + n ? LLONG_MIN : i + n;
    return (i >= 0 && i < n) ? (long)i : -1;
}
/* a slice: kind 1 both bounds, 2 no end, 3 no start, 4 neither
   (PySlice_Unpack and PySlice_AdjustIndices); the step 0 stops */
VIV_S_UNUSED static long viv_s_pyslice(long n, double kindf, double lof, double hif, double stepf, long* start, long long* step, int site) {
    long long st = viv_s_i64(stepf);
    if (st == 0) viv_s_fail(VIV_E_VALUE, "slice step cannot be zero", site);
    int kind = (int)viv_s_i64(kindf);
    int lo_given = kind == 1 || kind == 2, hi_given = kind == 1 || kind == 3;
    long long b[2];
    for (int e = 0; e < 2; e++) {
        int given = e == 0 ? lo_given : hi_given;
        long long v;
        if (!given) v = e == 0 ? (st < 0 ? n - 1 : 0) : (st < 0 ? -1 : n);
        else {
            v = viv_s_i64(e == 0 ? lof : hif);
            if (v < 0) { v = v < LLONG_MIN + n ? LLONG_MIN : v + n; if (v < 0) v = st < 0 ? -1 : 0; }
            else if (v >= n) v = st < 0 ? n - 1 : n;
        }
        b[e] = v;
    }
    long long count;
    if (st < 0) count = b[1] < b[0] ? (b[0] - b[1] - 1) / (-st) + 1 : 0;
    else count = b[0] < b[1] ? (b[1] - b[0] - 1) / st + 1 : 0;
    *start = (long)b[0];
    *step = st;
    return (long)count;
}
VIV_S_UNUSED static double viv_s_len(viv_str s) { return (double)VIV_S(s)->nch; }
/* s[i]: one character; an ASCII one is made once */
VIV_S_UNUSED static viv_str viv_s_at(viv_str s, double i, int site) {
    s = VIV_S(s);
    viv_s_whole(i, site);
    long k = viv_s_pyindex(s->nch, i);
    if (k < 0) viv_s_fail(VIV_E_INDEX, "string index out of range", site);
    if (s->nch == s->len) return viv_s_ascii((unsigned char)s->s[k]);
    int a = viv_s_byteof(s, (int)k), b = viv_s_byteof(s, (int)k + 1);
    if (b - a == 1) return viv_s_ascii((unsigned char)s->s[a]);
    return viv_s_make(s->s + a, b - a, site);
}
VIV_S_UNUSED static viv_str viv_s_slice(viv_str s, double kind, double lo, double hi, double step, int site) {
    s = VIV_S(s);
    long start;
    long long st;
    long count = viv_s_pyslice(s->nch, kind, lo, hi, step, &start, &st, site);
    if (count == 0) return viv_s_empty();
    if (st == 1) {
        int a = viv_s_byteof(s, (int)start), b = viv_s_byteof(s, (int)(start + count));
        return viv_s_make(s->s + a, b - a, site);
    }
    if (s->nch == s->len) {
        viv_str r = viv_s_alloc(count, site);
        for (long k = 0; k < count; k++) r->s[k] = s->s[start + k * st];
        r->nch = (int)count;
        return r;
    }
    /* the code points' offsets, then the chosen ones */
    long* off = (long*)malloc(sizeof(long) * ((size_t)s->nch + 1));
    if (!off) viv_s_nomem();
    long i = 0;
    for (int k = 0; k < s->nch; k++) { off[k] = i; viv_s_dec(s->s, &i); }
    off[s->nch] = s->len;
    viv_sb b = {0, 0, 0, site};
    for (long k = 0; k < count; k++) { long c = start + k * st; viv_sb_put(&b, s->s + off[c], off[c + 1] - off[c]); }
    free(off);
    return viv_sb_done(&b);
}
VIV_S_UNUSED static viv_str viv_s_cat(viv_str a, viv_str b, int site) {
    a = VIV_S(a); b = VIV_S(b);
    if (b->len == 0) return a;
    if (a->len == 0) return b;
    viv_str r = viv_s_alloc((long)a->len + b->len, site);
    memcpy(r->s, a->s, (size_t)a->len);
    memcpy(r->s + a->len, b->s, (size_t)b->len);
    r->nch = a->nch + b->nch;
    return r;
}
/* an f-string: its n parts, one after another */
VIV_S_UNUSED static viv_str viv_s_catn(int n, ...) {
    va_list ap;
    long total = 0, nch = 0;
    va_start(ap, n);
    for (int k = 0; k < n; k++) { viv_str p = VIV_S(va_arg(ap, viv_str)); total += p->len; nch += p->nch; }
    int site = va_arg(ap, int);
    va_end(ap);
    viv_str r = viv_s_alloc(total, site);
    long at = 0;
    va_start(ap, n);
    for (int k = 0; k < n; k++) { viv_str p = VIV_S(va_arg(ap, viv_str)); memcpy(r->s + at, p->s, (size_t)p->len); at += p->len; }
    va_end(ap);
    r->nch = (int)nch;
    return r;
}
VIV_S_UNUSED static viv_str viv_s_rep(viv_str s, double nf, int site) {
    s = VIV_S(s);
    long long n = viv_s_i64(nf);
    if (n <= 0 || s->len == 0) return viv_s_empty();
    if (n > VIV_S_MAX || (unsigned long long)s->len * (unsigned long long)n > (unsigned long long)VIV_S_MAX) viv_s_too_long(0, site);
    unsigned long long total = (unsigned long long)s->len * (unsigned long long)n;
    viv_str r = viv_s_alloc((long)total, site);
    for (long long k = 0; k < n; k++) memcpy(r->s + k * s->len, s->s, (size_t)s->len);
    r->nch = (int)((long long)s->nch * n);
    return r;
}
VIV_S_UNUSED static double viv_s_eq(viv_str a, viv_str b) {
    a = VIV_S(a); b = VIV_S(b);
    return a->len == b->len && memcmp(a->s, b->s, (size_t)a->len) == 0;
}
/* the order of two strings: their UTF-8 bytes, which is code point order */
VIV_S_UNUSED static int viv_s_order(viv_str a, viv_str b) {
    a = VIV_S(a); b = VIV_S(b);
    int m = a->len < b->len ? a->len : b->len;
    int c = m ? memcmp(a->s, b->s, (size_t)m) : 0;
    if (c) return c < 0 ? -1 : 1;
    return a->len < b->len ? -1 : a->len > b->len;
}
VIV_S_UNUSED static double viv_s_cmp(viv_str a, viv_str b) { return (double)viv_s_order(a, b); }
/* for a map with string keys (the map branch): FNV-1a, 64 bits, of the bytes */
VIV_S_UNUSED static unsigned long long viv_str_hash(viv_str s) {
    s = VIV_S(s);
    unsigned long long h = 0xcbf29ce484222325ULL;
    for (int i = 0; i < s->len; i++) { h ^= (unsigned char)s->s[i]; h *= 0x100000001b3ULL; }
    return h;
}
VIV_S_UNUSED static int viv_str_eq(viv_str a, viv_str b) { return viv_s_eq(a, b) != 0.0; }
VIV_S_UNUSED static int viv_str_cmp(viv_str a, viv_str b) { return viv_s_order(a, b); }
VIV_S_UNUSED static double viv_s_stop(viv_str s, int site) { viv_s_fail(VIV_E_STOP, VIV_S(s)->s, site); }
/* the first byte offset >= from where `nd` starts in `h`, or -1 */
VIV_S_UNUSED static long viv_s_find_bytes(const char* h, long hn, const char* nd, long m, long from) {
    if (m == 0) return from <= hn ? from : -1;
    for (long i = from; i + m <= hn; i++) {
        const char* p = (const char*)memchr(h + i, nd[0], (size_t)(hn - m + 1 - i));
        if (!p) return -1;
        i = p - h;
        if (memcmp(p, nd, (size_t)m) == 0) return i;
    }
    return -1;
}
/* the last byte offset where `nd` starts in h[0..hn), or -1 */
VIV_S_UNUSED static long viv_s_rfind_bytes(const char* h, long hn, const char* nd, long m) {
    for (long i = hn - m; i >= 0; i--) if (memcmp(h + i, nd, (size_t)m) == 0) return i;
    return -1;
}
/* What loops and returns keep.
   A loop that allocates frees, at the top of each pass, what the passes
   before made, keeping what its variables hold; a def that returns a string
   keeps its block, one that returns a list keeps the list and its chunks.
   The keep list is the arena_keep fragment's, shared with arrays and records. */
VIV_S_UNUSED static void viv_keep_ptr(const void* p) { if (p) viv_arena_keep(p); }
VIV_S_UNUSED static void viv_keep_str(viv_str s) { if (s) viv_keep_ptr(s->owner); }
VIV_S_UNUSED static void viv_keep_strs(viv_strs l) {
    if (!l) return;
    viv_keep_ptr(l);
    for (viv_chunk* c = l->chunk; c; c = c->prev) viv_keep_ptr(c);
}

/* print with text and numbers, and fmt (printfmt.rs, line for line) */
static void viv_fmt_e(char* b, int sz, double x, int p) { snprintf(b, sz, "%.*e", p, x); }
static void viv_repr(char* out, double x) {
    if (x != x) { strcpy(out, "nan"); return; }
    if (x > DBL_MAX || x < -DBL_MAX) { strcpy(out, x < 0 ? "-inf" : "inf"); return; }
    if (x == 0.0) { strcpy(out, signbit(x) ? "-0.0" : "0.0"); return; }
    double a = x < 0 ? -x : x;
    char b[48], t[48], d[24];
    unsigned long long n = 0;
    int ex = 0, p;
    for (p = 1; p <= 17; p++) {
        snprintf(b, sizeof b, "%.*e", p - 1, a);
        const char* c = b;
        n = 0;
        for (; *c != 'e'; c++) if (*c >= '0' && *c <= '9') n = n * 10 + (unsigned long long)(*c - '0');
        ex = atoi(c + 1);
        double r = strtod(b, 0);
        if (r != a) {
            unsigned long long lo = 1, hi;
            for (int k = 1; k < p; k++) lo *= 10;
            hi = lo * 10;
            if (r < a) { n++; if (n == hi) { n = lo; ex++; } }
            else { n--; if (n < lo) { n = hi - 1; ex--; } }
            snprintf(t, sizeof t, "%llue%d", n, ex - (p - 1));
            if (strtod(t, 0) != a) continue;
        }
        break;
    }
    snprintf(d, sizeof d, "%llu", n);
    int nd = (int)strlen(d);
    while (nd > 1 && d[nd - 1] == '0') nd--;
    d[nd] = 0;
    char* s = out;
    if (x < 0) *s++ = '-';
    int decpt = ex + 1;
    if (-4 < decpt && decpt <= 16) {
        if (decpt <= 0) {
            *s++ = '0'; *s++ = '.';
            for (int k = 0; k < -decpt; k++) *s++ = '0';
            strcpy(s, d);
        } else if (decpt >= nd) {
            strcpy(s, d); s += nd;
            for (int k = 0; k < decpt - nd; k++) *s++ = '0';
            strcpy(s, ".0");
        } else {
            memcpy(s, d, (size_t)decpt); s += decpt;
            *s++ = '.';
            strcpy(s, d + decpt);
        }
    } else {
        *s++ = d[0];
        if (nd > 1) { *s++ = '.'; strcpy(s, d + 1); s += nd - 1; }
        sprintf(s, "e%c%02d", ex < 0 ? '-' : '+', ex < 0 ? -ex : ex);
    }
}
/* a format's flags and width on a number's text s, into out (printfmt.rs
   `finish` and `group`, line for line). word is the kind word: the kind's
   letter, then the flags from bit 8 -- bit 0 upper case, 1-2 the sign
   (- + space), 3-5 the alignment (none < > ^ =), 6-7 grouping (none , _),
   8-14 the fill (0 for a space). out holds 1024 bytes: s is at most 360
   (%.40f of 1e308), its grouping a third more, the width 100. */
static void viv_numfin(char* out, const char* s, int word, int width) {
    int fl = (word >> 8) & 32767;
    int n = (int)strlen(s);
    if (fl == 0) {
        int k = 0;
        for (; k < width - n; k++) out[k] = ' ';
        memcpy(out + k, s, (size_t)n + 1);
        return;
    }
    int neg = s[0] == '-';
    if (neg) { s++; n--; }
    char body[600];
    for (int i = 0; i <= n; i++) body[i] = (fl & 1) && s[i] >= 'a' && s[i] <= 'z' ? (char)(s[i] - 32) : s[i];
    int sg = (fl >> 1) & 3, al = (fl >> 3) & 7, gr = (fl >> 6) & 3, fc = (fl >> 8) & 127;
    const char* sign = neg ? "-" : sg == 1 ? "+" : sg == 2 ? " " : "";
    int sl = (int)strlen(sign);
    char fill = fc ? (char)fc : ' ';
    char align = al == 1 ? '<' : al == 2 ? '>' : al == 3 ? '^' : al == 4 ? '=' : 0;
    if (gr == 1 || gr == 2) {
        char sep = gr == 1 ? ',' : '_';
        int nd = 0;
        while (body[nd] >= '0' && body[nd] <= '9') nd++;
        if (nd > 0) {
            int min = 0;
            if (fill == '0' && align == '=') { min = width - (sl + n - nd); if (min < 0) min = 0; }
            char g[600];
            int m = 0, i = 0;
            while (i < nd || m < min) {
                if (i > 0 && i % 3 == 0) g[m++] = sep;
                g[m++] = i < nd ? body[nd - 1 - i] : '0';
                i++;
            }
            char t[600];
            for (int k = 0; k < m; k++) t[k] = g[m - 1 - k];
            memcpy(t + m, body + nd, (size_t)(n - nd) + 1);
            n = m + n - nd;
            memcpy(body, t, (size_t)n + 1);
        }
    }
    int len = sl + n, k = width - len, o = 0;
    if (k <= 0) { memcpy(out, sign, (size_t)sl); memcpy(out + sl, body, (size_t)n + 1); return; }
    int left = align == '<' ? 0 : align == '^' ? k / 2 : align == '=' ? 0 : k;
    int mid = align == '=' ? k : 0;
    for (int q = 0; q < left; q++) out[o++] = fill;
    memcpy(out + o, sign, (size_t)sl); o += sl;
    for (int q = 0; q < mid; q++) out[o++] = fill;
    memcpy(out + o, body, (size_t)n); o += n;
    for (int q = left + mid; q < k; q++) out[o++] = fill;
    out[o] = 0;
}
/* a number's text in a format, before the flags: kind 'b' bare %g, 'r'
   repr, 'g' 'e' 'f' 'd', '%' (100 x in %f, then %) */
static void viv_numtext(char* b, int sz, double x, int kind, int prec) {
    if ((kind == 's' || kind == 'b') && x == 0.0) strcpy(b, "0");
    else if ((kind == 's' || kind == 'b') && viv_whole(x)) snprintf(b, (size_t)sz, "%.0f", x);
    else if (kind == 'r' || kind == 's') viv_repr(b, x);
    else if (kind == '%') {
        double y = x * 100.0;
        if (y != y) strcpy(b, "nan%");
        else if (y > DBL_MAX || y < -DBL_MAX) strcpy(b, y < 0 ? "-inf%" : "inf%");
        else { snprintf(b, (size_t)sz - 1, "%.*f", prec, y); strcat(b, "%"); }
    }
    else if (kind == 'b' && x == 0.0) strcpy(b, "0");
    else if (x != x) strcpy(b, "nan");
    else if (x > DBL_MAX || x < -DBL_MAX) strcpy(b, x < 0 ? "-inf" : "inf");
    else if (kind == 'b') viv_fmt_g(b, sz, x, 6);
    else if (kind == 'g') viv_fmt_g(b, sz, x, prec < 1 ? 1 : prec);
    else if (kind == 'e') viv_fmt_e(b, sz, x, prec);
    else if (kind == 'f') snprintf(b, (size_t)sz, "%.*f", prec, x);
    else snprintf(b, (size_t)sz, "%.0f", x + 0.0); /* d: a negative zero is 0 */
}
/* one number in a format (the kind word: see viv_numfin) */
static void viv_pnum(double x, int kind, int prec, int width) {
    static char b[512], o[1024];
    viv_numtext(b, sizeof b, x, kind & 255, prec);
    viv_numfin(o, b, kind, width);
    viv_fputs(o, stdout);
}
static void viv_parr(const double* data, const int* dims, int rank, int kind, int prec, int width) {
    viv_putchar('[');
    if (rank <= 1) {
        for (int i = 0; i < dims[0]; i++) { if (i) viv_fputs(", ", stdout); viv_pnum(data[i], kind, prec, width); }
    } else {
        int block = 1;
        for (int k = 1; k < rank; k++) block *= dims[k];
        for (int i = 0; i < dims[0]; i++) { if (i) viv_fputs(", ", stdout); viv_parr(data + (size_t)i * block, dims + 1, rank - 1, kind, prec, width); }
    }
    viv_putchar(']');
}
static double viv_print(int n, ...) {
    va_list ap;
    va_start(ap, n);
    for (int k = 0; k < n; k++) {
        int tag = va_arg(ap, int);
        if (tag == 0) viv_fputs(va_arg(ap, const char*), stdout);
        else if (tag == 1) { double x = va_arg(ap, double); int kd = va_arg(ap, int), p = va_arg(ap, int), w = va_arg(ap, int); viv_pnum(x, kd, p, w); }
        else if (tag == 2) {
            viv_arr a = va_arg(ap, viv_arr);
            int kd = va_arg(ap, int), p = va_arg(ap, int), w = va_arg(ap, int);
            int one = a.size;
            viv_parr(a.data, a.rank <= 1 ? &one : a.dims, a.rank, kd, p, w);
        }
        else if (tag == 3) viv_fputs(va_arg(ap, double) != 0.0 ? "true" : "false", stdout);
        else {
            _Complex double z = va_arg(ap, _Complex double);
            double re = ((double*)&z)[0] + 0.0, im = ((double*)&z)[1] + 0.0;
            viv_pnum(re, 'b', 6, 0);
            viv_fputs(im < 0.0 ? "-" : "+", stdout);
            viv_pnum(im < 0.0 ? -im : im, 'b', 6, 0);
            viv_putchar('i');
        }
    }
    va_end(ap);
    viv_putchar('\n');
    return 0.0;
}

/* a number in a format, as viv_pnum writes it, onto sb (the kind word and
   viv_numfin: printfmt.rs) */
VIV_S_UNUSED static void viv_s_numput(viv_sb* sb, double x, int kind, int prec, int width) {
    static char b[512], o[1024];
    viv_numtext(b, sizeof b, x, kind & 255, prec);
    viv_numfin(o, b, kind, width);
    viv_sb_put(sb, o, (long)strlen(o));
}
VIV_S_UNUSED static viv_str viv_s_fmt_num(double x, double kind, double prec, double width) {
    viv_s_fmtwp(prec, width, 40.0, 0);
    viv_sb b = {0, 0, 0, 0};
    viv_s_numput(&b, x, (int)kind, (int)prec, (int)width);
    return viv_sb_done(&b);
}
VIV_S_UNUSED static void viv_s_arrput(viv_sb* sb, const double* data, const int* dims, int rank, int kind, int prec, int width) {
    viv_sb_put(sb, "[", 1);
    if (rank <= 1) {
        for (int i = 0; i < dims[0]; i++) { if (i) viv_sb_put(sb, ", ", 2); viv_s_numput(sb, data[i], kind, prec, width); }
    } else {
        int block = 1;
        for (int k = 1; k < rank; k++) block *= dims[k];
        for (int i = 0; i < dims[0]; i++) { if (i) viv_sb_put(sb, ", ", 2); viv_s_arrput(sb, data + (size_t)i * block, dims + 1, rank - 1, kind, prec, width); }
    }
    viv_sb_put(sb, "]", 1);
}
VIV_S_UNUSED static viv_str viv_s_fmt_arr(viv_arr a, double kind, double prec, double width) {
    viv_s_fmtwp(prec, width, 40.0, 0);
    viv_sb b = {0, 0, 0, 0};
    int one = a.size;
    viv_s_arrput(&b, a.data, a.rank <= 1 ? &one : a.dims, a.rank, (int)kind, (int)prec, (int)width);
    return viv_sb_done(&b);
}
VIV_S_UNUSED static viv_str viv_s_str_bool(double b) {
    static viv_str t = 0, f = 0;
    if (b != 0.0) return t ? t : (t = viv_s_static("true", 4, 4));
    return f ? f : (f = viv_s_static("false", 5, 5));
}
VIV_S_UNUSED static viv_str viv_s_str_cplx(_Complex double z) {
    viv_sb b = {0, 0, 0, 0};
    double re = ((double*)&z)[0] + 0.0, im = ((double*)&z)[1] + 0.0;
    viv_s_numput(&b, re, 'b', 6, 0);
    viv_sb_put(&b, im < 0.0 ? "-" : "+", 1);
    viv_s_numput(&b, im < 0.0 ? -im : im, 'b', 6, 0);
    viv_sb_put(&b, "i", 1);
    return viv_sb_done(&b);
}

/* an Int as print, str and fmt(n, "d") write it: digits, right aligned in
   `width` columns (interp/ints.rs, text) */
VIV_S_UNUSED static viv_str viv_i_str(long long a, double width) {
    char b[160];
    int w = (int)width;
    int n = snprintf(b, sizeof b, "%*lld", w < 0 ? 0 : w > 100 ? 100 : w, a);
    return viv_s_make(b, n, 0);
}
/* an Int in a spec with flags (`+05d`, `,`): its digits, then viv_numfin
   (interp's printfmt::int_text) */
VIV_S_UNUSED static viv_str viv_i_fmt(long long a, double word, double width) {
    char b[32], o[1024];
    viv_s_fmtwp(-1.0, width, 40.0, 0);
    snprintf(b, sizeof b, "%lld", a);
    viv_numfin(o, b, (int)word, (int)width);
    return viv_s_make(o, (long)strlen(o), 0);
}
VIV_S_UNUSED static viv_str viv_i_ustr(long long a) {
    char b[32];
    int n = snprintf(b, sizeof b, "%llu", (unsigned long long)a);
    return viv_s_make(b, n, 0);
}
/* an Int array as print and show write it: [1, 2, 3] */
VIV_S_UNUSED static viv_str viv_iarr_str(viv_arr a) {
    long cap = 2 + 22 * (long)a.size, n = 0;
    char* b = (char*)malloc((size_t)cap);
    if (!b) viv_s_nomem();
    b[n++] = '[';
    for (int i = 0; i < a.size; i++) n += snprintf(b + n, (size_t)(cap - n), i ? ", %lld" : "%lld", ((long long*)a.data)[i]);
    b[n++] = ']';
    viv_str r = viv_s_make(b, n, 0);
    free(b);
    return r;
}

static viv_str viv_lits[11];

static viv_arr viv_vec(int n, ...) {
    int dims[1]; dims[0] = n;
    viv_arr a = viv_arr_new(1, dims);
    va_list ap; va_start(ap, n);
    for (int i = 0; i < n; i++) a.data[i] = va_arg(ap, double);
    va_end(ap);
    return a;
}

static viv_arr viv_zeros1(double n) { double x[1] = {n}; int d[1]; viv_zeros_dims(1, x, d, 1); return viv_arr_new(1, d); }

static viv_arr viv_zeros2(double r, double c) { double x[2] = {r, c}; int d[2]; viv_zeros_dims(2, x, d, 1); return viv_arr_new(2, d); }

typedef struct {
    viv_arr n_prev_1h;
    viv_arr status;
} viv_rec_Features;

long long CLEAN;
long long DAY;
long long HOUR;
viv_arr INIT_RAW;
double LEARNING_RATE;
long long LOCKED;
long long NF;
long long NI;
long long NOUT;
long long NS_PER_S;
long long NZ;
long long OK;
long long WATCH;
long long WATCH_DECLINES;

viv_rec_Features card_features(viv_arr card_viv_in, viv_arr ts_viv_in, viv_arr declined_viv_in, viv_arr chargeback_viv_in);
double _tree0(double z0, double z1, double z10);
double _tree1(double z0, double z1, double z10);
double _tree2(double z0, double z1, double z7, double z8);
double _tree3(double z0, double z1, double z10);
double _tree4(double z0, double z1, double z7, double z8);
double _tree5(double z0, double z1, double z7, double z8);
double _tree6(double z0, double z1, double z10);
double _tree7(double z0, double z1, double z7, double z8);
double _tree8(double z0, double z1, double z5, double z8);
double _tree9(double z0, double z1, double z7, double z10);
double _tree10(double z0, double z1, double z7, double z8);
double _tree11(double z0, double z1, double z7, double z8);
double _tree12(double z0, double z1, double z10);
double _tree13(double z0, double z7, double z8);
double _tree14(double z0, double z1, double z5, double z8);
double _tree15(double z0, double z1, double z10);
double _tree16(double z0, double z7, double z8);
double _tree17(double z0, double z1, double z7, double z10);
double _tree18(double z0, double z1, double z8);
double _tree19(double z0, double z1, double z7, double z8);
double _tree20(double z0, double z1, double z6, double z7, double z10);
double _tree21(double z0, double z1, double z7, double z8);
double _tree22(double z0, double z1, double z7, double z10);
double _tree23(double z0, double z1, double z5, double z8);
double _tree24(double z0, double z1, double z7);
double _tree25(double z0, double z1, double z3, double z7, double z8);
double _tree26(double z0, double z1, double z8, double z10);
double _tree27(double z0, double z1, double z2, double z7);
double _tree28(double z0, double z1, double z7);
double _tree29(double z0, double z1, double z5, double z7, double z10);
double _tree30(double z0, double z7, double z8);
double _tree31(double z0, double z1, double z10);
double _tree32(double z0, double z1, double z7, double z10);
double _tree33(double z0, double z1, double z8);
double _tree34(double z0, double z1, double z7);
double _tree35(double z0, double z1, double z9);
double _tree36(double z0, double z5, double z7, double z10);
double _tree37(double z0, double z1);
double _tree38(double z0, double z1, double z7);
double _tree39(double z0, double z4, double z10);
double _tree40(double z0, double z1, double z4);
double _tree41(double z0, double z4, double z7, double z10);
double _tree42(double z0, double z1, double z10);
double _tree43(double z0, double z1, double z4, double z9);
double _tree44(double z0, double z1, double z7);
double _tree45(double z0, double z1, double z8);
double _tree46(double z0, double z1);
double _tree47(double z0, double z1, double z2, double z7);
double _tree48(double z0, double z7);
double _tree49(double z0, double z4, double z8);
double _tree50(double z0, double z1, double z10);
double _tree51(double z0, double z1, double z4, double z8);
double _tree52(double z0, double z1, double z8);
double _tree53(double z0, double z1, double z9);
double _tree54(double z0, double z7, double z9);
double _tree55(double z0, double z4);
double _tree56(double z0, double z1);
double _tree57(double z0, double z7);
double _tree58(double z0, double z4, double z8);
double _tree59(double z0, double z1);
double _tree60(double z0, double z1, double z7, double z8);
double _tree61(double z0, double z1, double z7);
double _tree62(double z0, double z7);
double _tree63(double z0, double z1);
double _tree64(double z0, double z1, double z5);
double _tree65(double z0, double z5, double z8);
double _tree66(double z0, double z1, double z7);
double _tree67(double z0, double z1);
double _tree68(double z0, double z3);
double _tree69(double z0, double z1, double z7);
double _tree70(double z0, double z7, double z8);
double _tree71(double z0, double z7);
double _tree72(double z0, double z1, double z3, double z4, double z10);
double _tree73(double z0, double z4, double z10);
double _tree74(double z0, double z8);
double _tree75(double z0, double z3);
double _tree76(double z0, double z1);
double _tree77(double z0, double z4, double z8);
double _tree78(double z0, double z2, double z4);
double _tree79(double z0, double z1, double z4, double z10);
double _tree80(double z0, double z4);
double _tree81(double z0, double z7);
double _tree82(double z0, double z1, double z7);
double _tree83(double z0, double z1, double z7, double z9);
double _tree84(double z0, double z9);
double _tree85(double z0, double z1);
double _tree86(double z0, double z1, double z6);
double _tree87(double z0, double z3);
double _tree88(double z0, double z4, double z8);
double _tree89(double z0, double z1, double z2, double z7);
double _tree90(double z0);
double _tree91(double z0, double z4, double z6, double z10);
double _tree92(double z0, double z1);
double _tree93(double z0, double z2);
double _tree94(double z0, double z1, double z4, double z9);
double _tree95(double z0, double z1);
double _tree96(double z0, double z7, double z10);
double _tree97(double z0, double z10);
double _tree98(double z0, double z1, double z7);
double _tree99(double z0, double z1);
double _tree100(double z0, double z7);
double _tree101(double z0, double z4, double z5, double z10);
double _tree102(double z0, double z10);
double _tree103(double z0);
double _tree104(double z0, double z1);
double _tree105(double z0, double z1);
double _tree106(double z0, double z5);
double _tree107(double z0, double z1, double z7);
double _tree108(double z0, double z1, double z3);
double _tree109(double z0, double z10);
double _tree110(double z0, double z1, double z7);
double _tree111(double z0, double z1);
double _tree112(double z0, double z5);
double _tree113(double z0, double z6, double z10);
double _tree114(double z0, double z1, double z8);
double _tree115(double z0, double z1);
double _tree116(double z0, double z1);
double _tree117(double z0, double z1, double z7, double z9);
double _tree118(double z0, double z1, double z5, double z10);
double _tree119(double z0, double z2, double z7);
double _tree120(double z0);
double _tree121(double z0, double z1);
double _tree122(double z0, double z1, double z5, double z8);
double _tree123(double z0, double z1, double z6, double z7);
double _tree124(double z0, double z2, double z3);
double _tree125(double z0);
double _tree126(double z0, double z2, double z10);
double _tree127(double z0, double z1);
double _tree128(double z0, double z1);
double _tree129(double z0, double z7);
double _tree130(double z0, double z1);
double _tree131(double z0, double z1);
double _tree132(double z0, double z1);
double _tree133(double z0, double z5, double z8);
double _tree134(double z0, double z1, double z2);
double _tree135(double z0, double z1, double z4, double z9);
double _tree136(double z0, double z1);
double _tree137(double z0, double z4, double z5);
double _tree138(double z0, double z1);
double _tree139(double z0, double z8);
double _tree140(double z0, double z8);
double _tree141(double z0, double z5, double z10);
double _tree142(double z0, double z4);
double _tree143(double z0, double z1);
double _tree144(double z0, double z7);
double _tree145(double z0, double z3);
double _tree146(double z0, double z1);
double _tree147(double z0, double z1);
double _tree148(double z0, double z2, double z5, double z10);
double _tree149(double z0, double z7);
viv_arr score(viv_arr xf_viv_in, viv_arr xi_viv_in, long long rows);

viv_rec_Features card_features(viv_arr card_viv_in, viv_arr ts_viv_in, viv_arr declined_viv_in, viv_arr chargeback_viv_in) {
    double n = 0;
    viv_arr n_prev = {0};
    viv_arr status = {0};
    long long start = {0};
    long long stop_at = {0};
    long long left_1h = {0};
    long long left_24h = {0};
    long long declines_before_left = {0};
    long long declines_seen = {0};
    double locked = 0;
    long long last_bad = {0};
    long long i = {0};
    long long t = {0};
    long long same = {0};
    long long state = {0};
    long long viv__lo1 = {0};
    long long viv__hi2 = {0};
    long long viv__k3 = {0};
    long long k = {0};
    long long viv__lo4 = {0};
    long long viv__hi5 = {0};
    long long viv__k6 = {0};
    viv_arr chargeback = chargeback_viv_in;
    (void)(chargeback);
    viv_arr declined = declined_viv_in;
    (void)(declined);
    viv_arr ts = ts_viv_in;
    (void)(ts);
    viv_arr card = card_viv_in;
    (void)(card);
    size_t viv_mark = viv_arena_top;
    n = viv_len(ts);
    n_prev = viv_zeros1(n);
    status = viv_zeros1(n);
    start = 0LL;
    {
        size_t viv_lmark = viv_arena_top;
        while (viv_i_cmp(start, n) < 0.0) {
            if (viv_arena_top > viv_lmark) {
                viv_arena_release_kept(viv_lmark);
            } else {
                VIV_POLL();
            }
            VIV_POLL();
            stop_at = (start + 1LL);
            {
                size_t viv_lmark = viv_arena_top;
                while ((viv_i_cmp(stop_at, n) < 0.0) && (viv_iget_l_at(card, stop_at, 1) == viv_iget_l_at(card, start, 2))) {
                    if (viv_arena_top > viv_lmark) {
                        viv_arena_release_kept(viv_lmark);
                    } else {
                        VIV_POLL();
                    }
                    if (viv_iget_l_at(ts, stop_at, 3) < viv_iget_l_at(ts, (stop_at - 1LL), 4)) {
                        viv_s_stop(viv_s_catn(3, VIV_LIT(0, "card_features: rows must be sorted by (card, ts); row ", 54, 54), viv_i_str(stop_at, 0.0), VIV_LIT(1, " goes back in time", 18, 18), 5), 6);
                    }
                    stop_at = (stop_at + 1LL);
                }
                if (viv_arena_top > viv_lmark) {
                    viv_arena_release_kept(viv_lmark);
                } else {
                    VIV_POLL();
                }
            }
            if ((viv_i_cmp(stop_at, n) < 0.0) && (viv_iget_l_at(card, stop_at, 1) < viv_iget_l_at(card, start, 2))) {
                viv_s_stop(viv_s_catn(2, VIV_LIT(2, "card_features: rows must be sorted by (card, ts); card falls at row ", 68, 68), viv_i_str(stop_at, 0.0), 7), 8);
            }
            left_1h = start;
            left_24h = start;
            declines_before_left = 0LL;
            declines_seen = 0LL;
            locked = 0.0;
            last_bad = 0LL;
            i = start;
            while (i < stop_at) {
                VIV_POLL();
                t = viv_iget_l_at(ts, i, 9);
                same = i;
                while ((same < stop_at) && (viv_iget_l_at(ts, same, 10) == t)) {
                    same = (same + 1LL);
                }
                while (viv_iget_l_at(ts, left_1h, 11) < viv_i_sub(t, HOUR, 12)) {
                    left_1h = (left_1h + 1LL);
                }
                while (viv_iget_l_at(ts, left_24h, 13) < viv_i_sub(t, DAY, 14)) {
                    declines_before_left = viv_i_add(declines_before_left, viv_iget_l_at(declined, left_24h, 15), 16);
                    left_24h = (left_24h + 1LL);
                }
                if (locked && (viv_i_sub(t, last_bad, 17) >= CLEAN)) {
                    locked = 0.0;
                }
                state = OK;
                if (locked) {
                    state = LOCKED;
                } else {
                    if (viv_i_sub(declines_seen, declines_before_left, 18) >= WATCH_DECLINES) {
                        state = WATCH;
                    }
                }
                viv__lo1 = i;
                viv__hi2 = same;
                {
                    long long viv_n_viv__k3 = (viv__hi2 - viv__lo1);
                    for (long long viv_k_viv__k3 = 0; viv_k_viv__k3 < viv_n_viv__k3; viv_k_viv__k3 = (viv_k_viv__k3 + 1)) {
                        viv__k3 = viv_k_viv__k3;
                        k = (viv__lo1 + viv__k3);
                        viv_iset_l_at(n_prev, k, (i - left_1h), 19);
                        viv_iset_l_at(status, k, state, 20);
                    }
                }
                viv__lo4 = i;
                viv__hi5 = same;
                {
                    long long viv_n_viv__k6 = (viv__hi5 - viv__lo4);
                    for (long long viv_k_viv__k6 = 0; viv_k_viv__k6 < viv_n_viv__k6; viv_k_viv__k6 = (viv_k_viv__k6 + 1)) {
                        viv__k6 = viv_k_viv__k6;
                        k = (viv__lo4 + viv__k6);
                        declines_seen = viv_i_add(declines_seen, viv_iget_l_at(declined, k, 21), 22);
                        if (viv_iget_l_at(chargeback, k, 23) != 0LL) {
                            locked = 1.0;
                            last_bad = t;
                        } else {
                            if ((viv_iget_l_at(declined, k, 21) != 0LL) && locked) {
                                last_bad = t;
                            }
                        }
                    }
                }
                i = same;
            }
            start = stop_at;
        }
        if (viv_arena_top > viv_lmark) {
            viv_arena_release_kept(viv_lmark);
        } else {
            VIV_POLL();
        }
    }
    {
        viv_rec_Features viv_ret = ((viv_rec_Features){ .n_prev_1h = n_prev, .status = status });
        viv_arena_keep(viv_ret.n_prev_1h.data);
        viv_arena_keep(viv_ret.status.data);
        viv_arena_release_kept(viv_mark);
        return viv_ret;
    }
}

double _tree0(double z0, double z1, double z10) {
    if (z10 <= 0.5000000298023224) {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= 0.6615031063556671) {
                return -(0.2989893528092728);
            } else {
                return 0.04868715450212998;
            }
        } else {
            if (z0 <= 0.7070446908473967) {
                return 0.06757213261358312;
            } else {
                return 0.515725900553473;
            }
        }
    } else {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= -(0.8956183493137361)) {
                return 1.4421340899188053;
            } else {
                return 2.0997330747798753;
            }
        } else {
            if (z0 <= 0.2957584410905838) {
                return 2.640175943049786;
            } else {
                return 3.6908103096090885;
            }
        }
    }
    return 0.0;
}

double _tree1(double z0, double z1, double z10) {
    if (z10 <= 0.5000000298023224) {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= 0.6615031063556671) {
                return -(0.2750049646824661);
            } else {
                return 0.04366451934218726;
            }
        } else {
            if (z0 <= 0.7070446908473967) {
                return 0.06051873706159079;
            } else {
                return 0.4473108644355323;
            }
        }
    } else {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= 1.0326610207557678) {
                return 1.4751256475429413;
            } else {
                return 2.050320459424935;
            }
        } else {
            if (z0 <= 0.2957584410905838) {
                return 1.9818541815219304;
            } else {
                return 2.5924676400127424;
            }
        }
    }
    return 0.0;
}

double _tree2(double z0, double z1, double z7, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= -(0.5451718866825105)) {
                return 1.0313936172473797;
            } else {
                return 1.3863868304487879;
            }
        } else {
            if (z0 <= 0.30423130095005035) {
                return 1.5457670265353283;
            } else {
                return 1.9825638421552296;
            }
        }
    } else {
        if (z1 <= 0.4003589302301407) {
            if (z7 <= 0.5000000298023224) {
                return -(0.22078005344747226);
            } else {
                return 0.3584463100878202;
            }
        } else {
            if (z0 <= 0.9223962724208831) {
                return 0.06121907452368419;
            } else {
                return 0.43338340303182593;
            }
        }
    }
    return 0.0;
}

double _tree3(double z0, double z1, double z10) {
    if (z10 <= 0.5000000298023224) {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= 0.8380208313465118) {
                return -(0.23026957414414445);
            } else {
                return 0.08875419391121542;
            }
        } else {
            if (z1 <= 2.3318938016891475) {
                return 0.06946572174346803;
            } else {
                return 0.5438138263404367;
            }
        }
    } else {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= 1.5415027737617493) {
                return 1.0194918489463665;
            } else {
                return 1.5061552033166894;
            }
        } else {
            if (z0 <= 1.985032856464386) {
                return 1.376160855716597;
            } else {
                return 2.376270597169685;
            }
        }
    }
    return 0.0;
}

double _tree4(double z0, double z1, double z7, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= -(0.9151529967784882)) {
                return 0.5978108444559296;
            } else {
                return 0.9534427054192777;
            }
        } else {
            if (z0 <= 0.30423130095005035) {
                return 1.0722294062873476;
            } else {
                return 1.3949470883222939;
            }
        }
    } else {
        if (z7 <= 0.5000000298023224) {
            if (z1 <= 0.4003589302301407) {
                return -(0.18709892046046916);
            } else {
                return 0.07106590138014475;
            }
        } else {
            if (z0 <= -(0.6209568083286287)) {
                return 0.06946745302219466;
            } else {
                return 0.5457003564955536;
            }
        }
    }
    return 0.0;
}

double _tree5(double z0, double z1, double z7, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= -(0.5451718866825105)) {
                return 0.6225432724800125;
            } else {
                return 0.8518785247325855;
            }
        } else {
            if (z0 <= 1.985032856464386) {
                return 0.9759938187398828;
            } else {
                return 1.690321843563419;
            }
        }
    } else {
        if (z0 <= 0.704102724790573) {
            if (z7 <= 0.5000000298023224) {
                return -(0.1620209672555981);
            } else {
                return 0.2917253939072962;
            }
        } else {
            if (z0 <= 2.955880284309387) {
                return 0.10601959467750457;
            } else {
                return 0.5830366308492818;
            }
        }
    }
    return 0.0;
}

double _tree6(double z0, double z1, double z10) {
    if (z10 <= 0.5000000298023224) {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= 0.555592507123947) {
                return -(0.19730630761206633);
            } else {
                return 0.045703610541901016;
            }
        } else {
            if (z1 <= 4.2634289264678955) {
                return 0.08254046314640308;
            } else {
                return 1.0749361225110694;
            }
        }
    } else {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= 1.0326610207557678) {
                return 0.632254123108573;
            } else {
                return 0.9034384914216472;
            }
        } else {
            if (z0 <= 1.985032856464386) {
                return 0.8707936125661077;
            } else {
                return 1.5272309017707046;
            }
        }
    }
    return 0.0;
}

double _tree7(double z0, double z1, double z7, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z1 <= 0.4003589302301407) {
            if (z7 <= 0.5000000298023224) {
                return 0.563512430998252;
            } else {
                return 1.1133900489653754;
            }
        } else {
            if (z0 <= 0.13889304548501968) {
                return 0.6724736031278618;
            } else {
                return 0.9345710531487534;
            }
        }
    } else {
        if (z7 <= 0.5000000298023224) {
            if (z0 <= 0.704102724790573) {
                return -(0.13821450113053432);
            } else {
                return 0.09613216812681537;
            }
        } else {
            if (z0 <= -(0.6209568083286287)) {
                return 0.04402803880328627;
            } else {
                return 0.44726764952696807;
            }
        }
    }
    return 0.0;
}

double _tree8(double z0, double z1, double z5, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z1 <= 2.3318938016891475) {
            if (z0 <= 0.13912840932607648) {
                return 0.4812685932952441;
            } else {
                return 0.6679382144880407;
            }
        } else {
            if (z5 <= 0.5000000298023224) {
                return 1.352568233559763;
            } else {
                return 0.534119442541777;
            }
        }
    } else {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= 1.5113770365715025) {
                return -(0.14924211865298131);
            } else {
                return 0.16327498743241997;
            }
        } else {
            if (z1 <= 2.3318938016891475) {
                return 0.0586734296521097;
            } else {
                return 0.4441430226688814;
            }
        }
    }
    return 0.0;
}

double _tree9(double z0, double z1, double z7, double z10) {
    if (z10 <= 0.5000000298023224) {
        if (z0 <= -(0.01561878202483058)) {
            if (z1 <= 2.3318938016891475) {
                return -(0.14560398561049306);
            } else {
                return 0.3931047917120793;
            }
        } else {
            if (z7 <= 0.5000000298023224) {
                return 0.03056888196533577;
            } else {
                return 0.4487994886557709;
            }
        }
    } else {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= -(1.1003788113594057)) {
                return 0.1266019755120037;
            } else {
                return 0.47779104648842347;
            }
        } else {
            if (z1 <= 2.3318938016891475) {
                return 0.5898481655999133;
            } else {
                return 1.0257937110662918;
            }
        }
    }
    return 0.0;
}

double _tree10(double z0, double z1, double z7, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z7 <= 0.5000000298023224) {
            if (z0 <= 1.575041115283966) {
                return 0.39402704138482925;
            } else {
                return 0.7342306546476749;
            }
        } else {
            if (z0 <= 0.2669272273778915) {
                return 0.7258788417118125;
            } else {
                return 1.2081445775181343;
            }
        }
    } else {
        if (z1 <= 0.4003589302301407) {
            if (z7 <= 0.5000000298023224) {
                return -(0.125074219986847);
            } else {
                return 0.23403047539180305;
            }
        } else {
            if (z0 <= 2.807016968727112) {
                return 0.07242388600956524;
            } else {
                return 0.74660979463999;
            }
        }
    }
    return 0.0;
}

double _tree11(double z0, double z1, double z7, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z7 <= 0.5000000298023224) {
            if (z0 <= -(0.5282261669635774)) {
                return 0.25824013727677153;
            } else {
                return 0.436740781503452;
            }
        } else {
            if (z1 <= 2.3318938016891475) {
                return 0.7289492877278004;
            } else {
                return 1.7933821688579525;
            }
        }
    } else {
        if (z0 <= 0.9568760693073273) {
            if (z7 <= 0.5000000298023224) {
                return -(0.09803883953684421);
            } else {
                return 0.21708991424366164;
            }
        } else {
            if (z0 <= 3.703844666481018) {
                return 0.1094904011429005;
            } else {
                return 0.684994855247519;
            }
        }
    }
    return 0.0;
}

double _tree12(double z0, double z1, double z10) {
    if (z10 <= 0.5000000298023224) {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= -(0.013971283566206694)) {
                return -(0.15759201906334033);
            } else {
                return 0.00040071560307273867;
            }
        } else {
            if (z1 <= 4.2634289264678955) {
                return 0.0703263275611383;
            } else {
                return 0.7901714119334958;
            }
        }
    } else {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= -(1.1003788113594057)) {
                return 0.04338060585739418;
            } else {
                return 0.3283113466386859;
            }
        } else {
            if (z0 <= 1.985032856464386) {
                return 0.44640852350557586;
            } else {
                return 1.005536661356735;
            }
        }
    }
    return 0.0;
}

double _tree13(double z0, double z7, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z7 <= 0.5000000298023224) {
            if (z0 <= 1.575041115283966) {
                return 0.2740556616727274;
            } else {
                return 0.5585343256856364;
            }
        } else {
            if (z0 <= 2.4062041044235225) {
                return 0.6806141799788794;
            } else {
                return -(0.5629730797437084);
            }
        }
    } else {
        if (z7 <= 0.5000000298023224) {
            if (z0 <= 1.7443804144859312) {
                return -(0.0725757434543679);
            } else {
                return 0.1925692753069596;
            }
        } else {
            if (z0 <= -(0.6209568083286287)) {
                return 0.02901369721790245;
            } else {
                return 0.32460457162062184;
            }
        }
    }
    return 0.0;
}

double _tree14(double z0, double z1, double z5, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z1 <= 2.3318938016891475) {
            if (z0 <= 0.13912840932607648) {
                return 0.22220699375370007;
            } else {
                return 0.3510151705818997;
            }
        } else {
            if (z5 <= 0.5000000298023224) {
                return 0.8794167875658693;
            } else {
                return 0.213095937802335;
            }
        }
    } else {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= 1.6951908469200132) {
                return -(0.09802400701578233);
            } else {
                return 0.14279512603331823;
            }
        } else {
            if (z1 <= 4.2634289264678955) {
                return 0.06717860645571626;
            } else {
                return 0.6951114046667957;
            }
        }
    }
    return 0.0;
}

double _tree15(double z0, double z1, double z10) {
    if (z10 <= 0.5000000298023224) {
        if (z0 <= -(0.01561878202483058)) {
            if (z1 <= 2.3318938016891475) {
                return -(0.10196912462515398);
            } else {
                return 0.30764691762960583;
            }
        } else {
            if (z1 <= 0.4003589302301407) {
                return 0.005978263428199951;
            } else {
                return 0.15762148919556482;
            }
        }
    } else {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= -(1.1003788113594057)) {
                return -(0.012240182954485663);
            } else {
                return 0.22984645228587378;
            }
        } else {
            if (z0 <= -(1.3589184880256655)) {
                return 2.3470657494559166;
            } else {
                return 0.3517815084378513;
            }
        }
    }
    return 0.0;
}

double _tree16(double z0, double z7, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z7 <= 0.5000000298023224) {
            if (z0 <= 1.7630912661552427) {
                return 0.19276279990162754;
            } else {
                return 0.46687695712422167;
            }
        } else {
            if (z0 <= -(0.7399297058582307)) {
                return 0.8924066040763492;
            } else {
                return 0.4007585410237596;
            }
        }
    } else {
        if (z7 <= 0.5000000298023224) {
            if (z0 <= 0.42402796447277064) {
                return -(0.0822120255554116);
            } else {
                return 0.05336559409622479;
            }
        } else {
            if (z0 <= -(0.6209568083286287)) {
                return 0.040360768979874645;
            } else {
                return 0.29023034958907473;
            }
        }
    }
    return 0.0;
}

double _tree17(double z0, double z1, double z7, double z10) {
    if (z10 <= 0.5000000298023224) {
        if (z7 <= 0.5000000298023224) {
            if (z1 <= 0.4003589302301407) {
                return -(0.07811339966842204);
            } else {
                return 0.05428665077056031;
            }
        } else {
            if (z0 <= -(0.6209568083286287)) {
                return 0.03818142153904773;
            } else {
                return 0.2583777502211326;
            }
        }
    } else {
        if (z1 <= 2.3318938016891475) {
            if (z0 <= -(1.355388104915619)) {
                return 1.2499561609951253;
            } else {
                return 0.18927601447732792;
            }
        } else {
            if (z0 <= -(0.666851431131363)) {
                return 0.05894190264562868;
            } else {
                return 0.7662938107971978;
            }
        }
    }
    return 0.0;
}

double _tree18(double z0, double z1, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z0 <= -(0.5282261669635774)) {
            if (z0 <= -(1.355388104915619)) {
                return 1.0107395028466633;
            } else {
                return 0.0916252820436101;
            }
        } else {
            if (z1 <= 2.3318938016891475) {
                return 0.2105540927965586;
            } else {
                return 0.6611715575220746;
            }
        }
    } else {
        if (z0 <= 1.512083113193512) {
            if (z1 <= 2.3318938016891475) {
                return -(0.04933620642444041);
            } else {
                return 0.2883094371480609;
            }
        } else {
            if (z0 <= 3.703844666481018) {
                return 0.11251823122870333;
            } else {
                return 0.5290328687291292;
            }
        }
    }
    return 0.0;
}

double _tree19(double z0, double z1, double z7, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z7 <= 0.5000000298023224) {
            if (z0 <= 1.575041115283966) {
                return 0.13202701958745838;
            } else {
                return 0.3518916072766719;
            }
        } else {
            if (z0 <= -(1.255596697330475)) {
                return -(1.6732857157430368);
            } else {
                return 0.44081336965695966;
            }
        }
    } else {
        if (z0 <= -(0.21072968095541)) {
            if (z1 <= 2.3318938016891475) {
                return -(0.08866024203222322);
            } else {
                return 0.25274948130287234;
            }
        } else {
            if (z1 <= 0.4003589302301407) {
                return -(0.002543814957499949);
            } else {
                return 0.12284550336902124;
            }
        }
    }
    return 0.0;
}

double _tree20(double z0, double z1, double z6, double z7, double z10) {
    if (z10 <= 0.5000000298023224) {
        if (z7 <= 0.5000000298023224) {
            if (z0 <= -(0.01561878202483058)) {
                return -(0.07691575218816923);
            } else {
                return 0.028704368175246447;
            }
        } else {
            if (z0 <= 4.209744215011596) {
                return 0.17623565734484795;
            } else {
                return 1.751411323067934;
            }
        }
    } else {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= 5.949738264083861) {
                return 0.11388964052028454;
            } else {
                return -(1.863905729109498);
            }
        } else {
            if (z6 <= 0.5000000298023224) {
                return 0.27791526625070273;
            } else {
                return 0.028467514482624735;
            }
        }
    }
    return 0.0;
}

double _tree21(double z0, double z1, double z7, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z7 <= 0.5000000298023224) {
            if (z0 <= -(0.5282261669635774)) {
                return 0.04298388780007577;
            } else {
                return 0.16087785184444495;
            }
        } else {
            if (z0 <= 0.2669272273778915) {
                return 0.23170768788522517;
            } else {
                return 0.6575570016282638;
            }
        }
    } else {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= 1.7309650778770445) {
                return -(0.06317272555867458);
            } else {
                return 0.10807802711174248;
            }
        } else {
            if (z0 <= 3.1536976099014282) {
                return 0.05254427754651569;
            } else {
                return 0.5539241734137683;
            }
        }
    }
    return 0.0;
}

double _tree22(double z0, double z1, double z7, double z10) {
    if (z10 <= 0.5000000298023224) {
        if (z7 <= 0.5000000298023224) {
            if (z0 <= 2.955880284309387) {
                return -(0.03177294122817942);
            } else {
                return 0.2609788685399151;
            }
        } else {
            if (z0 <= -(0.6209568083286287)) {
                return 0.03121934541512372;
            } else {
                return 0.21098820943632998;
            }
        }
    } else {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= 5.949738264083861) {
                return 0.08888912744673162;
            } else {
                return -(1.7286262160265613);
            }
        } else {
            if (z0 <= -(1.2884290814399721)) {
                return 0.9901606527067057;
            } else {
                return 0.1916789021313709;
            }
        }
    }
    return 0.0;
}

double _tree23(double z0, double z1, double z5, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z1 <= 2.3318938016891475) {
            if (z0 <= 1.9353725314140318) {
                return 0.08776549447267208;
            } else {
                return 0.29372615645112715;
            }
        } else {
            if (z5 <= 0.5000000298023224) {
                return 0.5669040009888892;
            } else {
                return -(0.03907187508336262);
            }
        }
    } else {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= -(0.5953029096126556)) {
                return -(0.11624505585108043);
            } else {
                return -(0.012036537013386795);
            }
        } else {
            if (z1 <= 4.2634289264678955) {
                return 0.0479806576013228;
            } else {
                return 0.4988861750336321;
            }
        }
    }
    return 0.0;
}

double _tree24(double z0, double z1, double z7) {
    if (z7 <= 0.5000000298023224) {
        if (z1 <= 2.3318938016891475) {
            if (z0 <= 1.7443804144859312) {
                return -(0.02930844564094346);
            } else {
                return 0.11897941283534749;
            }
        } else {
            if (z1 <= 4.2634289264678955) {
                return 0.15419125311785228;
            } else {
                return 0.45407562020327574;
            }
        }
    } else {
        if (z0 <= 4.209744215011596) {
            if (z0 <= -(1.3103172183036804)) {
                return 1.0230072354594149;
            } else {
                return 0.14510508575635378;
            }
        } else {
            if (z0 <= 4.353782892227173) {
                return 3.4049325082694626;
            } else {
                return 0.8815847751923757;
            }
        }
    }
    return 0.0;
}

double _tree25(double z0, double z1, double z3, double z7, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z0 <= -(1.355388104915619)) {
            if (z3 <= 0.5000000298023224) {
                return 1.005097944906219;
            } else {
                return -(1.5164207447793874);
            }
        } else {
            if (z0 <= -(1.1134411692619324)) {
                return -(0.12072080384342959);
            } else {
                return 0.1075645771507678;
            }
        }
    } else {
        if (z0 <= -(0.5948322117328645)) {
            if (z1 <= 2.3318938016891475) {
                return -(0.08699621782308814);
            } else {
                return 0.23388430579619118;
            }
        } else {
            if (z7 <= 0.5000000298023224) {
                return 0.006088829318411046;
            } else {
                return 0.16419779586684063;
            }
        }
    }
    return 0.0;
}

double _tree26(double z0, double z1, double z8, double z10) {
    if (z1 <= 0.4003589302301407) {
        if (z8 <= 0.5000000298023224) {
            if (z0 <= -(1.166867196559906)) {
                return -(0.18414568064087505);
            } else {
                return 0.07521830475928863;
            }
        } else {
            if (z0 <= 3.682074189186096) {
                return -(0.04040261344005161);
            } else {
                return 0.32104839409807795;
            }
        }
    } else {
        if (z0 <= 3.1536976099014282) {
            if (z10 <= 0.5000000298023224) {
                return 0.041875240618746645;
            } else {
                return 0.15613145918044874;
            }
        } else {
            if (z0 <= 3.17170250415802) {
                return 3.256450901034727;
            } else {
                return 0.38759064531713394;
            }
        }
    }
    return 0.0;
}

double _tree27(double z0, double z1, double z2, double z7) {
    if (z0 <= -(0.013265212532132864)) {
        if (z1 <= 2.3318938016891475) {
            if (z7 <= 0.5000000298023224) {
                return -(0.0497647830317234);
            } else {
                return 0.09912723123460641;
            }
        } else {
            if (z0 <= -(1.2998438477516177)) {
                return -(1.0284698737048263);
            } else {
                return 0.18064321242105744;
            }
        }
    } else {
        if (z1 <= 4.2634289264678955) {
            if (z0 <= 3.6448875665664673) {
                return 0.0336728284309833;
            } else {
                return 0.2978257178971228;
            }
        } else {
            if (z2 <= 0.5000000298023224) {
                return 0.68087006517031;
            } else {
                return -(0.6558027768066731);
            }
        }
    }
    return 0.0;
}

double _tree28(double z0, double z1, double z7) {
    if (z1 <= 0.4003589302301407) {
        if (z7 <= 0.5000000298023224) {
            if (z0 <= 5.282736539840697) {
                return -(0.031080213665535695);
            } else {
                return -(0.8652023661874098);
            }
        } else {
            if (z0 <= 1.0341908335685728) {
                return 0.055789006916935224;
            } else {
                return 0.31021543362276477;
            }
        }
    } else {
        if (z0 <= 2.013393521308899) {
            if (z1 <= 2.3318938016891475) {
                return 0.027459091856122674;
            } else {
                return 0.1680125317082478;
            }
        } else {
            if (z0 <= 2.021983981132507) {
                return 1.7726006854541447;
            } else {
                return 0.20660503615942621;
            }
        }
    }
    return 0.0;
}

double _tree29(double z0, double z1, double z5, double z7, double z10) {
    if (z10 <= 0.5000000298023224) {
        if (z0 <= -(0.21072968095541)) {
            if (z1 <= 2.3318938016891475) {
                return -(0.05362245980427168);
            } else {
                return 0.13195752134349736;
            }
        } else {
            if (z1 <= 4.2634289264678955) {
                return 0.02259292877911025;
            } else {
                return 0.4626905967538288;
            }
        }
    } else {
        if (z0 <= -(1.355388104915619)) {
            if (z7 <= 0.5000000298023224) {
                return 0.9371091183831278;
            } else {
                return -(1.8311838562395353);
            }
        } else {
            if (z5 <= 0.5000000298023224) {
                return 0.10767982354435288;
            } else {
                return 0.006219005426302117;
            }
        }
    }
    return 0.0;
}

double _tree30(double z0, double z7, double z8) {
    if (z7 <= 0.5000000298023224) {
        if (z0 <= -(0.5976564586162567)) {
            if (z0 <= -(0.5988332331180574)) {
                return -(0.05667872525202796);
            } else {
                return -(0.9734646857794239);
            }
        } else {
            if (z8 <= 0.5000000298023224) {
                return 0.08739025120542372;
            } else {
                return 0.004834133799599501;
            }
        }
    } else {
        if (z0 <= -(1.3103172183036804)) {
            if (z0 <= -(1.3152597546577456)) {
                return 0.6219015316274795;
            } else {
                return 2.87714889051091;
            }
        } else {
            if (z0 <= 4.209744215011596) {
                return 0.10298144123414195;
            } else {
                return 1.0397197791762556;
            }
        }
    }
    return 0.0;
}

double _tree31(double z0, double z1, double z10) {
    if (z1 <= 0.4003589302301407) {
        if (z0 <= 1.5113770365715025) {
            if (z0 <= 1.4986677765846252) {
                return -(0.029770601878924354);
            } else {
                return -(0.6606582250265566);
            }
        } else {
            if (z0 <= 5.417596101760864) {
                return 0.07217479461842184;
            } else {
                return -(0.7262695365299754);
            }
        }
    } else {
        if (z0 <= 3.1536976099014282) {
            if (z10 <= 0.5000000298023224) {
                return 0.032595389366971456;
            } else {
                return 0.12350124485375291;
            }
        } else {
            if (z0 <= 3.1881774663925166) {
                return 1.8528584823571932;
            } else {
                return 0.26913067105084104;
            }
        }
    }
    return 0.0;
}

double _tree32(double z0, double z1, double z7, double z10) {
    if (z7 <= 0.5000000298023224) {
        if (z0 <= 0.704102724790573) {
            if (z10 <= 0.5000000298023224) {
                return -(0.031059137158566622);
            } else {
                return 0.04702678550078772;
            }
        } else {
            if (z1 <= 4.2634289264678955) {
                return 0.040238842245555635;
            } else {
                return 0.632050999420663;
            }
        }
    } else {
        if (z0 <= -(1.3103172183036804)) {
            if (z1 <= 0.4003589302301407) {
                return 0.3778982728731581;
            } else {
                return 1.6065752963481073;
            }
        } else {
            if (z0 <= 0.08652612939476967) {
                return 0.047795433753676;
            } else {
                return 0.16344526931348624;
            }
        }
    }
    return 0.0;
}

double _tree33(double z0, double z1, double z8) {
    if (z0 <= 3.7479740381240845) {
        if (z1 <= 0.4003589302301407) {
            if (z8 <= 0.5000000298023224) {
                return 0.045480109159027435;
            } else {
                return -(0.02879055163143736);
            }
        } else {
            if (z1 <= 4.2634289264678955) {
                return 0.03449037195458293;
            } else {
                return 0.24623328286605675;
            }
        }
    } else {
        if (z0 <= 4.152905702590942) {
            if (z0 <= 4.082416296005248) {
                return 0.5485223232206569;
            } else {
                return 1.4172623297321494;
            }
        } else {
            if (z0 <= 5.417596101760864) {
                return 0.15428761099895658;
            } else {
                return -(0.5467276226563346);
            }
        }
    }
    return 0.0;
}

double _tree34(double z0, double z1, double z7) {
    if (z0 <= -(0.5948322117328645)) {
        if (z1 <= 2.3318938016891475) {
            if (z0 <= -(1.1669848561286928)) {
                return -(0.12266344481557874);
            } else {
                return -(0.03677770801758548);
            }
        } else {
            if (z0 <= -(0.5970681011676789)) {
                return 0.11784927246418958;
            } else {
                return 3.104323722763245;
            }
        }
    } else {
        if (z1 <= 0.4003589302301407) {
            if (z7 <= 0.5000000298023224) {
                return -(0.006779981928698163);
            } else {
                return 0.10076866631277492;
            }
        } else {
            if (z0 <= 3.8520017862319946) {
                return 0.054504097493322405;
            } else {
                return 0.40486142883613907;
            }
        }
    }
    return 0.0;
}

double _tree35(double z0, double z1, double z9) {
    if (z0 <= 1.512083113193512) {
        if (z1 <= 2.3318938016891475) {
            if (z0 <= 1.4570096135139465) {
                return -(0.010399347326720687);
            } else {
                return -(0.29621851330159615);
            }
        } else {
            if (z0 <= 0.8140144050121306) {
                return 0.07202719706427413;
            } else {
                return 0.46955831770172085;
            }
        }
    } else {
        if (z0 <= 1.60151869058609) {
            if (z9 <= 0.5000000298023224) {
                return 0.30845024680105065;
            } else {
                return 3.090562740167174;
            }
        } else {
            if (z0 <= 1.6834229826927183) {
                return -(0.2527969564016484);
            } else {
                return 0.06837752517351704;
            }
        }
    }
    return 0.0;
}

double _tree36(double z0, double z5, double z7, double z10) {
    if (z10 <= 0.5000000298023224) {
        if (z0 <= 3.7780996561050415) {
            if (z7 <= 0.5000000298023224) {
                return -(0.01451428900676051);
            } else {
                return 0.07471288033316534;
            }
        } else {
            if (z0 <= 3.7982226610183716) {
                return 2.1021288517840007;
            } else {
                return 0.19321251522904537;
            }
        }
    } else {
        if (z0 <= -(1.355388104915619)) {
            if (z0 <= -(1.362684190273285)) {
                return 0.4392523100833913;
            } else {
                return 1.6815784665966793;
            }
        } else {
            if (z5 <= 0.5000000298023224) {
                return 0.08155097094534315;
            } else {
                return -(0.008673549781003606);
            }
        }
    }
    return 0.0;
}

double _tree37(double z0, double z1) {
    if (z0 <= -(0.5948322117328645)) {
        if (z1 <= 2.3318938016891475) {
            if (z0 <= -(1.1669848561286928)) {
                return -(0.11082436378344092);
            } else {
                return -(0.03198174906912538);
            }
        } else {
            if (z0 <= -(0.5970681011676789)) {
                return 0.0983079004389486;
            } else {
                return 2.357439399476421;
            }
        }
    } else {
        if (z1 <= 0.4003589302301407) {
            if (z0 <= 5.417596101760864) {
                return 0.0014135084125756362;
            } else {
                return -(0.6635338053595864);
            }
        } else {
            if (z0 <= 5.894546747207642) {
                return 0.04981452411401834;
            } else {
                return 1.252226687253009;
            }
        }
    }
    return 0.0;
}

double _tree38(double z0, double z1, double z7) {
    if (z7 <= 0.5000000298023224) {
        if (z0 <= -(0.013265212532132864)) {
            if (z0 <= -(1.165337383747101)) {
                return -(0.10955776133313722);
            } else {
                return -(0.019034340748885752);
            }
        } else {
            if (z1 <= 4.2634289264678955) {
                return 0.018066046560674117;
            } else {
                return 0.373655941887425;
            }
        }
    } else {
        if (z0 <= 4.209744215011596) {
            if (z0 <= -(1.3103172183036804)) {
                return 0.6483928690481808;
            } else {
                return 0.0697286552367813;
            }
        } else {
            if (z0 <= 4.353782892227173) {
                return 2.2639507238215306;
            } else {
                return 0.48614497481136215;
            }
        }
    }
    return 0.0;
}

double _tree39(double z0, double z4, double z10) {
    if (z10 <= 0.5000000298023224) {
        if (z0 <= 3.7780996561050415) {
            if (z0 <= -(0.5948322117328645)) {
                return -(0.04058735354314758);
            } else {
                return 0.0061385578939539415;
            }
        } else {
            if (z0 <= 4.128663778305053) {
                return 0.5564746485978885;
            } else {
                return -(0.022236559761233468);
            }
        }
    } else {
        if (z0 <= -(1.2884290814399721)) {
            if (z4 <= 0.5000000298023224) {
                return 0.1567321641535223;
            } else {
                return 1.0568762667158178;
            }
        } else {
            if (z0 <= -(1.166867196559906)) {
                return -(0.18724974587697485);
            } else {
                return 0.05537502770229449;
            }
        }
    }
    return 0.0;
}

double _tree40(double z0, double z1, double z4) {
    if (z1 <= 0.4003589302301407) {
        if (z0 <= 1.0343084931373596) {
            if (z0 <= 0.9895906746387482) {
                return -(0.020779315015517094);
            } else {
                return -(0.2559900940009265);
            }
        } else {
            if (z0 <= 1.0957366824150083) {
                return 0.329368916228102;
            } else {
                return 0.009581832112080762;
            }
        }
    } else {
        if (z4 <= 0.5000000298023224) {
            if (z0 <= 2.806781649589538) {
                return 0.04770396798698521;
            } else {
                return 0.21452102032872117;
            }
        } else {
            if (z0 <= 1.085380971431732) {
                return -(0.062487698642168375);
            } else {
                return 0.19056263838877213;
            }
        }
    }
    return 0.0;
}

double _tree41(double z0, double z4, double z7, double z10) {
    if (z10 <= 0.5000000298023224) {
        if (z0 <= 3.7780996561050415) {
            if (z7 <= 0.5000000298023224) {
                return -(0.011690987085145623);
            } else {
                return 0.0591727395861264;
            }
        } else {
            if (z0 <= 3.785160422325134) {
                return 2.4281617388851915;
            } else {
                return 0.1511671710164082;
            }
        }
    } else {
        if (z4 <= 0.5000000298023224) {
            if (z0 <= -(1.166867196559906)) {
                return -(0.24175216056581825);
            } else {
                return 0.03393867296429562;
            }
        } else {
            if (z0 <= 1.9165440201759338) {
                return 0.07367653450465969;
            } else {
                return 0.7544190008176586;
            }
        }
    }
    return 0.0;
}

double _tree42(double z0, double z1, double z10) {
    if (z0 <= 5.417596101760864) {
        if (z0 <= 3.7780996561050415) {
            if (z10 <= 0.5000000298023224) {
                return -(0.006259604891497525);
            } else {
                return 0.042336659469991204;
            }
        } else {
            if (z0 <= 3.785160422325134) {
                return 2.018311817861139;
            } else {
                return 0.21729708846781046;
            }
        }
    } else {
        if (z0 <= 5.9016077518463135) {
            if (z1 <= 0.4003589302301407) {
                return -(1.0501422174629154);
            } else {
                return -(1.4425268697144997);
            }
        } else {
            if (z0 <= 5.948443651199341) {
                return 2.6505585255367197;
            } else {
                return -(0.10489962370236623);
            }
        }
    }
    return 0.0;
}

double _tree43(double z0, double z1, double z4, double z9) {
    if (z1 <= 0.4003589302301407) {
        if (z0 <= 4.797077417373657) {
            if (z0 <= 3.7001966238021846) {
                return -(0.013926883893492571);
            } else {
                return 0.27740115574160834;
            }
        } else {
            if (z0 <= 4.939939260482787) {
                return -(1.1570300913399634);
            } else {
                return -(0.2715384033084613);
            }
        }
    } else {
        if (z4 <= 0.5000000298023224) {
            if (z9 <= 0.5000000298023224) {
                return 0.04818679523787094;
            } else {
                return -(0.3477850998849619);
            }
        } else {
            if (z0 <= 0.42449869215488434) {
                return -(0.07882520674201453);
            } else {
                return 0.1147282058014982;
            }
        }
    }
    return 0.0;
}

double _tree44(double z0, double z1, double z7) {
    if (z7 <= 0.5000000298023224) {
        if (z1 <= 4.2634289264678955) {
            if (z0 <= 5.376173257827759) {
                return -(0.005241819632056442);
            } else {
                return -(0.5792850927953657);
            }
        } else {
            if (z0 <= 0.4053171128034591) {
                return 0.04516935396495307;
            } else {
                return 0.507389660743424;
            }
        }
    } else {
        if (z0 <= 6.5514280796051025) {
            if (z0 <= -(1.3103172183036804)) {
                return 0.5587642645838706;
            } else {
                return 0.05775021301234636;
            }
        } else {
            return 3.1396529290586717;
        }
    }
    return 0.0;
}

double _tree45(double z0, double z1, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z0 <= -(1.2884290814399721)) {
            if (z0 <= -(1.2901942133903503)) {
                return 0.29993876997690866;
            } else {
                return 2.2729060906839496;
            }
        } else {
            if (z0 <= -(1.1662787795066833)) {
                return -(0.19049663910109624);
            } else {
                return 0.04230408484776601;
            }
        }
    } else {
        if (z1 <= 8.12649869918823) {
            if (z0 <= 0.9568760693073273) {
                return -(0.012669831729897055);
            } else {
                return 0.030662927931677547;
            }
        } else {
            return 3.6530594727316403;
        }
    }
    return 0.0;
}

double _tree46(double z0, double z1) {
    if (z1 <= 0.4003589302301407) {
        if (z0 <= -(1.167220175266266)) {
            if (z0 <= -(1.205112636089325)) {
                return -(0.041883683531717106);
            } else {
                return -(0.28281134234111793);
            }
        } else {
            if (z0 <= 4.797077417373657) {
                return -(0.006366826079857673);
            } else {
                return -(0.36217001976963614);
            }
        }
    } else {
        if (z0 <= 5.894546747207642) {
            if (z0 <= 5.412771463394165) {
                return 0.024761185034952712;
            } else {
                return -(1.3703186415625106);
            }
        } else {
            if (z0 <= 7.858600854873657) {
                return 1.3243583019985379;
            } else {
                return -(1.4342890811550695);
            }
        }
    }
    return 0.0;
}

double _tree47(double z0, double z1, double z2, double z7) {
    if (z7 <= 0.5000000298023224) {
        if (z1 <= 4.2634289264678955) {
            if (z0 <= 5.376173257827759) {
                return -(0.004724791990086838);
            } else {
                return -(0.5199337689713474);
            }
        } else {
            if (z2 <= 0.5000000298023224) {
                return 0.22892178083013667;
            } else {
                return -(0.6377134456587554);
            }
        }
    } else {
        if (z0 <= 4.209744215011596) {
            if (z1 <= 6.194963693618774) {
                return 0.05429356442913605;
            } else {
                return 2.8014784209273342;
            }
        } else {
            if (z0 <= 5.498441457748412) {
                return 1.1426714188916902;
            } else {
                return -(0.3761787054921777);
            }
        }
    }
    return 0.0;
}

double _tree48(double z0, double z7) {
    if (z0 <= -(0.5948322117328645)) {
        if (z0 <= -(1.3988113999366762)) {
            if (z7 <= 0.5000000298023224) {
                return -(1.1361106181295362);
            } else {
                return -(1.334218664280326);
            }
        } else {
            if (z0 <= -(1.3983407616615298)) {
                return 6.8879094922596185;
            } else {
                return -(0.02542450169382767);
            }
        }
    } else {
        if (z0 <= -(0.5823582708835603)) {
            if (z0 <= -(0.5837704241275788)) {
                return 0.1629609120850347;
            } else {
                return 1.1533475459799185;
            }
        } else {
            if (z7 <= 0.5000000298023224) {
                return 0.004557405704113802;
            } else {
                return 0.06436370641570348;
            }
        }
    }
    return 0.0;
}

double _tree49(double z0, double z4, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z4 <= 0.5000000298023224) {
            if (z0 <= 2.2219196557998657) {
                return 0.022364403121548176;
            } else {
                return -(0.2626980321528087);
            }
        } else {
            if (z0 <= 1.918309152126312) {
                return 0.06108380460299857;
            } else {
                return 0.6492250188558936;
            }
        }
    } else {
        if (z0 <= 2.6420317888259888) {
            if (z0 <= 2.527413010597229) {
                return -(0.005525653859460751);
            } else {
                return -(0.3041273071560346);
            }
        } else {
            if (z0 <= 2.6423848867416377) {
                return 5.851997329878703;
            } else {
                return 0.079390513294569;
            }
        }
    }
    return 0.0;
}

double _tree50(double z0, double z1, double z10) {
    if (z1 <= 2.3318938016891475) {
        if (z0 <= 1.5246747136116028) {
            if (z0 <= 1.4570096135139465) {
                return -(0.005942544164285761);
            } else {
                return -(0.24172148465975712);
            }
        } else {
            if (z0 <= 1.600341975688934) {
                return 0.3186431185226239;
            } else {
                return 0.018259143566489434;
            }
        }
    } else {
        if (z0 <= -(1.2998438477516177)) {
            if (z10 <= 0.5000000298023224) {
                return -(0.9577326280892693);
            } else {
                return -(1.962526354579682);
            }
        } else {
            if (z0 <= -(0.9759927690029144)) {
                return 0.33721738190577266;
            } else {
                return 0.048037797036685356;
            }
        }
    }
    return 0.0;
}

double _tree51(double z0, double z1, double z4, double z8) {
    if (z1 <= 0.4003589302301407) {
        if (z0 <= -(1.3961048722267153)) {
            if (z8 <= 0.5000000298023224) {
                return -(1.5880154944745015);
            } else {
                return -(1.120393269506284);
            }
        } else {
            if (z0 <= -(1.395634114742279)) {
                return 9.186703893673304;
            } else {
                return -(0.00966161535324723);
            }
        }
    } else {
        if (z4 <= 0.5000000298023224) {
            if (z0 <= -(1.3902209401130678)) {
                return 1.231516872648926;
            } else {
                return 0.03774037675147284;
            }
        } else {
            if (z0 <= 0.42449869215488434) {
                return -(0.07256832723679894);
            } else {
                return 0.09619583257200716;
            }
        }
    }
    return 0.0;
}

double _tree52(double z0, double z1, double z8) {
    if (z1 <= 8.12649869918823) {
        if (z8 <= 0.5000000298023224) {
            if (z0 <= -(1.2884290814399721)) {
                return 0.34392714654981643;
            } else {
                return 0.02798272703030598;
            }
        } else {
            if (z0 <= 2.6420317888259888) {
                return -(0.0060951220228797984);
            } else {
                return 0.06979655058126055;
            }
        }
    } else {
        return 2.799644465169881;
    }
    return 0.0;
}

double _tree53(double z0, double z1, double z9) {
    if (z0 <= -(1.165337383747101)) {
        if (z0 <= -(1.2884290814399721)) {
            if (z0 <= -(1.2893704771995544)) {
                return 0.0507446147757191;
            } else {
                return 2.506174532844196;
            }
        } else {
            if (z9 <= 0.5000000298023224) {
                return -(0.12739880934544548);
            } else {
                return -(1.2817365084812846);
            }
        }
    } else {
        if (z0 <= -(1.1535695195198061)) {
            if (z0 <= -(1.1549816727638247)) {
                return 0.16974211777460696;
            } else {
                return 0.9438428879228519;
            }
        } else {
            if (z1 <= 6.194963693618774) {
                return 0.001923833619594072;
            } else {
                return 0.6252634670025599;
            }
        }
    }
    return 0.0;
}

double _tree54(double z0, double z7, double z9) {
    if (z7 <= 0.5000000298023224) {
        if (z0 <= -(1.0215342640876772)) {
            if (z9 <= 0.5000000298023224) {
                return -(0.04813242336499244);
            } else {
                return -(1.2700107709817836);
            }
        } else {
            if (z0 <= -(1.0208281874656677)) {
                return 1.807137374224227;
            } else {
                return 0.0012732450476109543;
            }
        }
    } else {
        if (z0 <= 6.5514280796051025) {
            if (z0 <= -(1.3103172183036804)) {
                return 0.4822918567581992;
            } else {
                return 0.04210494992300678;
            }
        } else {
            return 2.640563379306394;
        }
    }
    return 0.0;
}

double _tree55(double z0, double z4) {
    if (z0 <= 5.417596101760864) {
        if (z0 <= 5.1661174297332755) {
            if (z0 <= 5.131166696548462) {
                return 0.00012192677381381612;
            } else {
                return -(1.7993917624796802);
            }
        } else {
            if (z0 <= 5.181533098220825) {
                return 4.102356122900095;
            } else {
                return 0.5336879844802788;
            }
        }
    } else {
        if (z0 <= 6.06341528892517) {
            if (z4 <= 0.5000000298023224) {
                return -(1.2877402478045417);
            } else {
                return -(0.3068933002399679);
            }
        } else {
            if (z0 <= 6.117900609970092) {
                return 3.5774258720460272;
            } else {
                return 0.07015868899667975;
            }
        }
    }
    return 0.0;
}

double _tree56(double z0, double z1) {
    if (z1 <= 0.4003589302301407) {
        if (z0 <= 4.797077417373657) {
            if (z0 <= 4.537596464157104) {
                return -(0.008914043595012239);
            } else {
                return 0.8338346362303736;
            }
        } else {
            if (z0 <= 4.888866662979126) {
                return -(1.3289411900591603);
            } else {
                return -(0.21434873514285419);
            }
        }
    } else {
        if (z0 <= 6.1340224742889395) {
            if (z0 <= 5.412771463394165) {
                return 0.019468970827297948;
            } else {
                return -(0.823093499285106);
            }
        } else {
            if (z0 <= 6.611444234848022) {
                return 2.5321870482793103;
            } else {
                return 0.09050790656435183;
            }
        }
    }
    return 0.0;
}

double _tree57(double z0, double z7) {
    if (z0 <= -(0.013265212532132864)) {
        if (z0 <= -(0.02338556107133627)) {
            if (z0 <= -(0.02691591437906027)) {
                return -(0.01145771142446704);
            } else {
                return 0.5197270644797795;
            }
        } else {
            if (z0 <= -(0.022679490037262443)) {
                return -(0.7773695201026415);
            } else {
                return -(0.1992978035673308);
            }
        }
    } else {
        if (z0 <= -(0.005263077327981592)) {
            if (z7 <= 0.5000000298023224) {
                return 0.4312832958082026;
            } else {
                return -(0.7003233989298911);
            }
        } else {
            if (z0 <= -(0.0031448648078367114)) {
                return -(0.5197196560226351);
            } else {
                return 0.013134769521853119;
            }
        }
    }
    return 0.0;
}

double _tree58(double z0, double z4, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z4 <= 0.5000000298023224) {
            if (z0 <= -(1.166867196559906)) {
                return -(0.23140865283849044);
            } else {
                return 0.018810029845997975;
            }
        } else {
            if (z0 <= 1.918309152126312) {
                return 0.05442159893307811;
            } else {
                return 0.5835221472312883;
            }
        }
    } else {
        if (z0 <= -(1.3740989565849304)) {
            if (z0 <= -(1.3908093571662905)) {
                return -(0.04977290727454066);
            } else {
                return -(0.641137930635828);
            }
        } else {
            if (z0 <= -(1.3731575608253481)) {
                return 2.7573489992756386;
            } else {
                return -(0.0031381072239552842);
            }
        }
    }
    return 0.0;
}

double _tree59(double z0, double z1) {
    if (z1 <= 2.3318938016891475) {
        if (z0 <= 1.7443804144859312) {
            if (z0 <= 1.601989448070526) {
                return -(0.0035269933628598008);
            } else {
                return -(0.1846317355589332);
            }
        } else {
            if (z0 <= 1.7486168742179868) {
                return 1.3225237561844774;
            } else {
                return 0.036085464377123004;
            }
        }
    } else {
        if (z0 <= 3.8930715322494502) {
            if (z0 <= 3.0374313592910767) {
                return 0.06694947633790187;
            } else {
                return -(0.7541310801961785);
            }
        } else {
            if (z0 <= 4.259169340133666) {
                return 2.4252075241714977;
            } else {
                return -(0.29267115049001047);
            }
        }
    }
    return 0.0;
}

double _tree60(double z0, double z1, double z7, double z8) {
    if (z7 <= 0.5000000298023224) {
        if (z1 <= 8.12649869918823) {
            if (z0 <= -(1.1740456223487856)) {
                return -(0.0693797882855755);
            } else {
                return -(0.00031056792294329095);
            }
        } else {
            return 2.2653101939099645;
        }
    } else {
        if (z8 <= 0.5000000298023224) {
            if (z0 <= -(0.7399297058582307)) {
                return 0.540747058933782;
            } else {
                return 0.03935773890273352;
            }
        } else {
            if (z0 <= -(1.3243209719657898)) {
                return 0.5928597332517757;
            } else {
                return 0.029858669210072134;
            }
        }
    }
    return 0.0;
}

double _tree61(double z0, double z1, double z7) {
    if (z0 <= -(1.3988113999366762)) {
        if (z7 <= 0.5000000298023224) {
            if (z1 <= 0.4003589302301407) {
                return -(1.0963025437915956);
            } else {
                return -(1.1574438117684693);
            }
        } else {
            return -(1.3679113086814105);
        }
    } else {
        if (z0 <= -(1.3983407616615298)) {
            return 4.0171620058098965;
        } else {
            if (z0 <= 3.6448875665664673) {
                return -(0.0009038341070203351);
            } else {
                return 0.10232305641246522;
            }
        }
    }
    return 0.0;
}

double _tree62(double z0, double z7) {
    if (z0 <= 5.417596101760864) {
        if (z0 <= 5.1661174297332755) {
            if (z0 <= 5.131166696548462) {
                return 0.00012520794144723753;
            } else {
                return -(1.7306035798189248);
            }
        } else {
            if (z0 <= 5.181533098220825) {
                return 3.080940682827158;
            } else {
                return 0.4701828743506987;
            }
        }
    } else {
        if (z0 <= 5.770984411239623) {
            if (z7 <= 0.5000000298023224) {
                return -(1.2245297282434746);
            } else {
                return -(1.3981035125897963);
            }
        } else {
            if (z0 <= 5.779810667037963) {
                return 2.4369850864334106;
            } else {
                return -(0.06966458562136231);
            }
        }
    }
    return 0.0;
}

double _tree63(double z0, double z1) {
    if (z1 <= 0.4003589302301407) {
        if (z0 <= 4.797077417373657) {
            if (z0 <= 4.724117040634154) {
                return -(0.007627622803892523);
            } else {
                return 1.3732215125250575;
            }
        } else {
            if (z0 <= 4.939939260482787) {
                return -(1.060041191787846);
            } else {
                return -(0.18846983290586758);
            }
        }
    } else {
        if (z0 <= 4.803079366683959) {
            if (z0 <= 4.719291925430298) {
                return 0.016586442117992463;
            } else {
                return -(1.4726463084023071);
            }
        } else {
            if (z0 <= 4.87674593925476) {
                return 3.1211502368272734;
            } else {
                return 0.22066266336139154;
            }
        }
    }
    return 0.0;
}

double _tree64(double z0, double z1, double z5) {
    if (z0 <= -(0.5948322117328645)) {
        if (z1 <= 6.194963693618774) {
            if (z0 <= -(1.2884290814399721)) {
                return 0.08816480119635564;
            } else {
                return -(0.024081008586238482);
            }
        } else {
            if (z5 <= 0.5000000298023224) {
                return 0.5796517403061904;
            } else {
                return 3.452631786059461;
            }
        }
    } else {
        if (z0 <= -(0.5927139818668367)) {
            if (z0 <= -(0.5929493010044099)) {
                return 0.3630545795327223;
            } else {
                return 2.560299911819292;
            }
        } else {
            if (z0 <= -(0.574356108903885)) {
                return 0.13950803476100107;
            } else {
                return 0.00540329545753912;
            }
        }
    }
    return 0.0;
}

double _tree65(double z0, double z5, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z0 <= -(1.355388104915619)) {
            if (z5 <= 0.5000000298023224) {
                return 0.12625678157734932;
            } else {
                return 1.2693570315270557;
            }
        } else {
            if (z0 <= -(1.3456208109855652)) {
                return -(1.6244130021944634);
            } else {
                return 0.02518575656780694;
            }
        }
    } else {
        if (z0 <= -(1.3651553988456726)) {
            if (z0 <= -(1.372216045856476)) {
                return -(0.18380830775462959);
            } else {
                return -(0.8263794388072853);
            }
        } else {
            if (z0 <= -(1.3364418148994448)) {
                return 0.28139715562621187;
            } else {
                return -(0.003268670106405854);
            }
        }
    }
    return 0.0;
}

double _tree66(double z0, double z1, double z7) {
    if (z7 <= 0.5000000298023224) {
        if (z1 <= 4.2634289264678955) {
            if (z0 <= 5.282736539840697) {
                return -(0.0032261218843610234);
            } else {
                return -(0.42697482486849303);
            }
        } else {
            if (z0 <= 0.4053171128034591) {
                return -(0.0032376633973874316);
            } else {
                return 0.3907098183276499;
            }
        }
    } else {
        if (z0 <= 5.0513808727264395) {
            if (z1 <= 6.194963693618774) {
                return 0.03713964144125219;
            } else {
                return 2.237925585546086;
            }
        } else {
            if (z0 <= 5.498441457748412) {
                return 2.072104363089245;
            } else {
                return -(0.2459630202510693);
            }
        }
    }
    return 0.0;
}

double _tree67(double z0, double z1) {
    if (z1 <= 8.12649869918823) {
        if (z0 <= 3.6448875665664673) {
            if (z0 <= 3.567455172538757) {
                return -(0.0003709511136064329);
            } else {
                return -(0.6485335219973304);
            }
        } else {
            if (z0 <= 3.6579500436782837) {
                return 2.526472117558188;
            } else {
                return 0.06428542968655847;
            }
        }
    } else {
        return 2.007342633412827;
    }
    return 0.0;
}

double _tree68(double z0, double z3) {
    if (z0 <= 0.9568760693073273) {
        if (z0 <= 0.9408718049526213) {
            if (z0 <= 0.9406364262104034) {
                return -(0.003694969764056777);
            } else {
                return 3.676280591462709;
            }
        } else {
            if (z0 <= 0.9559346735477448) {
                return -(0.30572140601070796);
            } else {
                return -(1.2520330582571175);
            }
        }
    } else {
        if (z0 <= 0.9895906746387482) {
            if (z3 <= 0.5000000298023224) {
                return 0.20237311346969292;
            } else {
                return 1.0592420715712203;
            }
        } else {
            if (z0 <= 1.0227760672569273) {
                return -(0.26652405774925414);
            } else {
                return 0.020580502968363047;
            }
        }
    }
    return 0.0;
}

double _tree69(double z0, double z1, double z7) {
    if (z0 <= -(1.3988113999366762)) {
        if (z7 <= 0.5000000298023224) {
            if (z1 <= 0.4003589302301407) {
                return -(1.0853551411010145);
            } else {
                return -(1.139053661036543);
            }
        } else {
            return -(1.3194066179921993);
        }
    } else {
        if (z0 <= -(1.3983407616615298)) {
            return 3.036480037559057;
        } else {
            if (z1 <= 2.3318938016891475) {
                return -(0.0016236689217722046);
            } else {
                return 0.05139639692777122;
            }
        }
    }
    return 0.0;
}

double _tree70(double z0, double z7, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z0 <= -(1.2981963753700259)) {
            if (z7 <= 0.5000000298023224) {
                return 0.411947026727757;
            } else {
                return -(1.910713374937189);
            }
        } else {
            if (z0 <= -(1.2954897284507751)) {
                return -(1.6730447155364254);
            } else {
                return 0.02162706240915579;
            }
        }
    } else {
        if (z0 <= -(1.3740989565849304)) {
            if (z0 <= -(1.3766879439353943)) {
                return -(0.25630376024696505);
            } else {
                return -(1.1620635779594686);
            }
        } else {
            if (z0 <= -(1.3731575608253481)) {
                return 2.040200229167786;
            } else {
                return -(0.002560998025671669);
            }
        }
    }
    return 0.0;
}

double _tree71(double z0, double z7) {
    if (z7 <= 0.5000000298023224) {
        if (z0 <= -(1.1740456223487856)) {
            if (z0 <= -(1.1759284138679507)) {
                return -(0.050596915972104264);
            } else {
                return -(0.855313008028192);
            }
        } else {
            if (z0 <= -(1.172633469104767)) {
                return 1.0989588344074688;
            } else {
                return -(0.00044472096720763615);
            }
        }
    } else {
        if (z0 <= 6.5514280796051025) {
            if (z0 <= -(1.3103172183036804)) {
                return 0.38583786616353744;
            } else {
                return 0.031218459215818353;
            }
        } else {
            return 2.3095769459573448;
        }
    }
    return 0.0;
}

double _tree72(double z0, double z1, double z3, double z4, double z10) {
    if (z1 <= 0.4003589302301407) {
        if (z0 <= 4.797077417373657) {
            if (z0 <= 4.747299432754517) {
                return -(0.0065989761663441);
            } else {
                return 1.4996487959311409;
            }
        } else {
            if (z3 <= 0.5000000298023224) {
                return -(0.17307100453684085);
            } else {
                return -(1.23200395660317);
            }
        }
    } else {
        if (z4 <= 0.5000000298023224) {
            if (z0 <= 0.6132549345493317) {
                return 0.046793397001824104;
            } else {
                return -(0.02221582685348772);
            }
        } else {
            if (z10 <= 0.5000000298023224) {
                return -(0.0394204373718167);
            } else {
                return 0.17346231474776905;
            }
        }
    }
    return 0.0;
}

double _tree73(double z0, double z4, double z10) {
    if (z0 <= -(0.013265212532132864)) {
        if (z0 <= -(0.122941542416811)) {
            if (z0 <= -(0.12317689880728722)) {
                return -(0.004295817406749876);
            } else {
                return 1.8886910881261414;
            }
        } else {
            if (z10 <= 0.5000000298023224) {
                return -(0.047917591112745975);
            } else {
                return -(0.33168967909310104);
            }
        }
    } else {
        if (z0 <= -(0.009028788190335035)) {
            if (z4 <= 0.5000000298023224) {
                return 0.21293357204387822;
            } else {
                return 1.04300918180212;
            }
        } else {
            if (z0 <= 0.1223003827035427) {
                return 0.07039716281654743;
            } else {
                return 0.003186925442398566;
            }
        }
    }
    return 0.0;
}

double _tree74(double z0, double z8) {
    if (z0 <= 3.7780996561050415) {
        if (z0 <= 3.4678992033004756) {
            if (z0 <= 3.4669579267501827) {
                return -(0.00021649458133467519);
            } else {
                return 5.403283039927042;
            }
        } else {
            if (z8 <= 0.5000000298023224) {
                return -(1.2562512991957862);
            } else {
                return -(0.1330808734016543);
            }
        }
    } else {
        if (z0 <= 4.152905702590942) {
            if (z0 <= 3.785160422325134) {
                return 1.6704992669109333;
            } else {
                return 0.29172410826269024;
            }
        } else {
            if (z0 <= 4.1923277378082275) {
                return -(0.8583143176430347);
            } else {
                return -(0.010328647313611229);
            }
        }
    }
    return 0.0;
}

double _tree75(double z0, double z3) {
    if (z0 <= 0.9568760693073273) {
        if (z0 <= 0.9408718049526213) {
            if (z0 <= 0.9349878728389739) {
                return -(0.004135589226602855);
            } else {
                return 0.5384337808696212;
            }
        } else {
            if (z0 <= 0.9559346735477448) {
                return -(0.2802093523912734);
            } else {
                return -(1.2247585375651993);
            }
        }
    } else {
        if (z0 <= 0.9851188957691193) {
            if (z3 <= 0.5000000298023224) {
                return 0.19129591465799575;
            } else {
                return 0.9811583909621193;
            }
        } else {
            if (z0 <= 1.0227760672569273) {
                return -(0.19901995001289244);
            } else {
                return 0.01927458997520852;
            }
        }
    }
    return 0.0;
}

double _tree76(double z0, double z1) {
    if (z1 <= 8.12649869918823) {
        if (z0 <= -(0.5948322117328645)) {
            if (z1 <= 6.194963693618774) {
                return -(0.016225600685729407);
            } else {
                return 0.9098588502826972;
            }
        } else {
            if (z0 <= -(0.5927139818668367)) {
                return 0.49868626461921534;
            } else {
                return 0.005764104558687666;
            }
        }
    } else {
        return 1.8182801733565288;
    }
    return 0.0;
}

double _tree77(double z0, double z4, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z4 <= 0.5000000298023224) {
            if (z0 <= 2.2219196557998657) {
                return 0.012525879348201053;
            } else {
                return -(0.2463866136014755);
            }
        } else {
            if (z0 <= -(1.257361948490143)) {
                return 0.7912533283331799;
            } else {
                return 0.05079542551370854;
            }
        }
    } else {
        if (z0 <= -(1.3651553988456726)) {
            if (z0 <= -(1.372216045856476)) {
                return -(0.14543076527149237);
            } else {
                return -(0.7926496374648483);
            }
        } else {
            if (z0 <= -(1.3364418148994448)) {
                return 0.25314127853542834;
            } else {
                return -(0.0027843649274749594);
            }
        }
    }
    return 0.0;
}

double _tree78(double z0, double z2, double z4) {
    if (z0 <= 1.9700877070426939) {
        if (z0 <= 1.9623209834098814) {
            if (z0 <= 1.9616149067878723) {
                return -(0.0019832301793481807);
            } else {
                return 1.5950350492882726;
            }
        } else {
            if (z2 <= 0.5000000298023224) {
                return -(0.8565725081832521);
            } else {
                return 5.822074806399401;
            }
        }
    } else {
        if (z0 <= 1.9755008816719055) {
            if (z0 <= 1.9725589156150816) {
                return 0.21346032185810407;
            } else {
                return 1.428798221209851;
            }
        } else {
            if (z4 <= 0.5000000298023224) {
                return -(0.007337425837475501);
            } else {
                return 0.11261005292075212;
            }
        }
    }
    return 0.0;
}

double _tree79(double z0, double z1, double z4, double z10) {
    if (z0 <= 7.334460496902466) {
        if (z0 <= 6.253230810165404) {
            if (z0 <= 6.233107805252075) {
                return -(6.204949769678864e-5);
            } else {
                return 4.133409300788471;
            }
        } else {
            if (z10 <= 0.5000000298023224) {
                return -(1.202732791894083);
            } else {
                return -(1.5147699095365048);
            }
        }
    } else {
        if (z0 <= 7.757279634475708) {
            if (z4 <= 0.5000000298023224) {
                return 3.4248981557229006;
            } else {
                return -(1.1626058547183424);
            }
        } else {
            if (z1 <= 0.4003589302301407) {
                return -(1.1606670892342188);
            } else {
                return -(1.3685785014628642);
            }
        }
    }
    return 0.0;
}

double _tree80(double z0, double z4) {
    if (z0 <= 5.417596101760864) {
        if (z0 <= 5.1661174297332755) {
            if (z0 <= 5.131166696548462) {
                return 9.542580396460483e-5;
            } else {
                return -(1.6718505695883572);
            }
        } else {
            if (z0 <= 5.181533098220825) {
                return 2.5647614293344483;
            } else {
                return 0.4222330271283069;
            }
        }
    } else {
        if (z0 <= 6.06341528892517) {
            if (z4 <= 0.5000000298023224) {
                return -(1.2170975273216678);
            } else {
                return -(0.20108404140532096);
            }
        } else {
            if (z0 <= 6.117900609970092) {
                return 3.1166668852651833;
            } else {
                return 0.06834693392402569;
            }
        }
    }
    return 0.0;
}

double _tree81(double z0, double z7) {
    if (z7 <= 0.5000000298023224) {
        if (z0 <= -(1.1740456223487856)) {
            if (z0 <= -(1.1759284138679507)) {
                return -(0.045311670293544194);
            } else {
                return -(0.8211686740558628);
            }
        } else {
            if (z0 <= -(1.172633469104767)) {
                return 0.9144265586734031;
            } else {
                return -(0.00041847979363881973);
            }
        }
    } else {
        if (z0 <= 4.209744215011596) {
            if (z0 <= 3.3481026887893677) {
                return 0.0345570898131527;
            } else {
                return -(0.4369833850274986);
            }
        } else {
            if (z0 <= 4.353782892227173) {
                return 1.7906885298915616;
            } else {
                return 0.2565820709251532;
            }
        }
    }
    return 0.0;
}

double _tree82(double z0, double z1, double z7) {
    if (z0 <= -(1.3988113999366762)) {
        if (z7 <= 0.5000000298023224) {
            if (z1 <= 0.4003589302301407) {
                return -(1.0726155164685358);
            } else {
                return -(1.1177361889603972);
            }
        } else {
            return -(1.2810091808914128);
        }
    } else {
        if (z0 <= -(1.3983407616615298)) {
            return 2.5903649726959417;
        } else {
            if (z0 <= -(1.2884290814399721)) {
                return 0.09214927921703678;
            } else {
                return -(0.0010694198355429528);
            }
        }
    }
    return 0.0;
}

double _tree83(double z0, double z1, double z7, double z9) {
    if (z1 <= 0.4003589302301407) {
        if (z9 <= 0.5000000298023224) {
            if (z0 <= -(1.3961048722267153)) {
                return -(1.13806200252719);
            } else {
                return -(0.006521269798123185);
            }
        } else {
            if (z7 <= 0.5000000298023224) {
                return 0.11372836375329425;
            } else {
                return 1.6721908446015976;
            }
        }
    } else {
        if (z0 <= 4.803079366683959) {
            if (z0 <= 4.719291925430298) {
                return 0.01300522903341731;
            } else {
                return -(1.4126575282289793);
            }
        } else {
            if (z0 <= 4.87674593925476) {
                return 2.538748992484189;
            } else {
                return 0.19289541866215026;
            }
        }
    }
    return 0.0;
}

double _tree84(double z0, double z9) {
    if (z0 <= -(1.165337383747101)) {
        if (z0 <= -(1.2804269194602966)) {
            if (z0 <= -(1.2806622385978699)) {
                return 0.055883241401467894;
            } else {
                return 4.554394569948917;
            }
        } else {
            if (z9 <= 0.5000000298023224) {
                return -(0.09691805600621144);
            } else {
                return -(1.2175272129846157);
            }
        }
    } else {
        if (z0 <= -(1.1535695195198061)) {
            if (z0 <= -(1.1549816727638247)) {
                return 0.163295324308317;
            } else {
                return 0.8095505707684945;
            }
        } else {
            if (z0 <= -(1.146744191646576)) {
                return -(0.32130403701351257);
            } else {
                return 0.0017840433409307213;
            }
        }
    }
    return 0.0;
}

double _tree85(double z0, double z1) {
    if (z1 <= 8.12649869918823) {
        if (z0 <= 5.417596101760864) {
            if (z0 <= 5.1661174297332755) {
                return -(1.2717455603937146e-5);
            } else {
                return 0.682748551183756;
            }
        } else {
            if (z0 <= 5.770984411239623) {
                return -(1.201400091413587);
            } else {
                return 0.043483812058765364;
            }
        }
    } else {
        return 1.6816525397052078;
    }
    return 0.0;
}

double _tree86(double z0, double z1, double z6) {
    if (z1 <= 2.3318938016891475) {
        if (z0 <= 1.5246747136116028) {
            if (z0 <= 1.4040542244911194) {
                return -(0.002351707941244946);
            } else {
                return -(0.1738718927168651);
            }
        } else {
            if (z0 <= 1.600341975688934) {
                return 0.27416970031943033;
            } else {
                return 0.004189113079306442;
            }
        }
    } else {
        if (z0 <= -(1.2960781455039978)) {
            if (z6 <= 0.5000000298023224) {
                return -(1.019505167616387);
            } else {
                return -(0.16473368947923925);
            }
        } else {
            if (z0 <= -(0.9759927690029144)) {
                return 0.3044149093236573;
            } else {
                return 0.016691310491436212;
            }
        }
    }
    return 0.0;
}

double _tree87(double z0, double z3) {
    if (z0 <= 3.7479740381240845) {
        if (z0 <= 3.7354999780654907) {
            if (z0 <= 3.726909518241882) {
                return -(0.0006926238404129824);
            } else {
                return 0.9281971652490218;
            }
        } else {
            if (z3 <= 0.5000000298023224) {
                return -(1.3504116727145572);
            } else {
                return -(0.18360924050771227);
            }
        }
    } else {
        if (z0 <= 4.152905702590942) {
            if (z0 <= 4.082416296005248) {
                return 0.18497960160060364;
            } else {
                return 0.8499139592702871;
            }
        } else {
            if (z0 <= 4.1923277378082275) {
                return -(0.8051431367285319);
            } else {
                return -(0.017321182555726775);
            }
        }
    }
    return 0.0;
}

double _tree88(double z0, double z4, double z8) {
    if (z8 <= 0.5000000298023224) {
        if (z4 <= 0.5000000298023224) {
            if (z0 <= 2.580721259117126) {
                return 0.009549084971730955;
            } else {
                return -(0.2978660855332001);
            }
        } else {
            if (z0 <= 1.918309152126312) {
                return 0.03731416111347151;
            } else {
                return 0.5088768312744418;
            }
        }
    } else {
        if (z0 <= -(1.3584477305412292)) {
            if (z0 <= -(1.3599775433540346)) {
                return -(0.19274541122502548);
            } else {
                return -(1.1710778834107738);
            }
        } else {
            if (z0 <= -(1.3364418148994448)) {
                return 0.2823768110492659;
            } else {
                return -(0.0025156607654429042);
            }
        }
    }
    return 0.0;
}

double _tree89(double z0, double z1, double z2, double z7) {
    if (z7 <= 0.5000000298023224) {
        if (z1 <= 4.2634289264678955) {
            if (z0 <= 5.282736539840697) {
                return -(0.002395381292260102);
            } else {
                return -(0.36109975002723066);
            }
        } else {
            if (z2 <= 0.5000000298023224) {
                return 0.14672495037543057;
            } else {
                return -(0.6269684851249259);
            }
        }
    } else {
        if (z1 <= 6.194963693618774) {
            if (z0 <= -(1.0317723155021667)) {
                return 0.15924172070979448;
            } else {
                return 0.016174540424712464;
            }
        } else {
            return 1.9651074374612325;
        }
    }
    return 0.0;
}

double _tree90(double z0) {
    if (z0 <= 0.704102724790573) {
        if (z0 <= 0.6760952770709991) {
            if (z0 <= 0.6751538217067717) {
                return -(0.0028995207840471463);
            } else {
                return 1.1018394602263584;
            }
        } else {
            if (z0 <= 0.7022198736667633) {
                return -(0.21045571936555574);
            } else {
                return -(1.032691518558845);
            }
        }
    } else {
        if (z0 <= 0.7045734226703644) {
            if (z0 <= 0.7043381035327911) {
                return 1.6045536898944919;
            } else {
                return 7.013776551712192;
            }
        } else {
            if (z0 <= 0.7391709387302399) {
                return 0.20954843552942914;
            } else {
                return 0.00764703703063199;
            }
        }
    }
    return 0.0;
}

double _tree91(double z0, double z4, double z6, double z10) {
    if (z0 <= 7.334460496902466) {
        if (z0 <= 6.253230810165404) {
            if (z0 <= 6.233107805252075) {
                return -(5.6382786839088205e-5);
            } else {
                return 3.2661863565861;
            }
        } else {
            if (z10 <= 0.5000000298023224) {
                return -(1.1783541211243804);
            } else {
                return -(1.4535170199888068);
            }
        }
    } else {
        if (z0 <= 7.757279634475708) {
            if (z4 <= 0.5000000298023224) {
                return 2.862498270094974;
            } else {
                return -(1.1409942536845354);
            }
        } else {
            if (z6 <= 0.5000000298023224) {
                return -(1.1393401776039822);
            } else {
                return -(1.3193820563809204);
            }
        }
    }
    return 0.0;
}

double _tree92(double z0, double z1) {
    if (z0 <= -(1.165337383747101)) {
        if (z0 <= -(1.2884290814399721)) {
            if (z0 <= -(1.2886644005775454)) {
                return 0.05302407433830981;
            } else {
                return 4.107023925328052;
            }
        } else {
            if (z0 <= -(1.2879583239555359)) {
                return -(1.2845840519109517);
            } else {
                return -(0.08407700486925712);
            }
        }
    } else {
        if (z0 <= -(1.1535695195198061)) {
            if (z1 <= 4.2634289264678955) {
                return 0.2051783032267198;
            } else {
                return 1.7015809158250537;
            }
        } else {
            if (z0 <= -(1.146744191646576)) {
                return -(0.2970847092618013);
            } else {
                return 0.0016646928067416099;
            }
        }
    }
    return 0.0;
}

double _tree93(double z0, double z2) {
    if (z0 <= 1.9700877070426939) {
        if (z0 <= 1.9623209834098814) {
            if (z0 <= 1.9616149067878723) {
                return -(0.0016373557922338222);
            } else {
                return 1.2964341943568016;
            }
        } else {
            if (z2 <= 0.5000000298023224) {
                return -(0.813668274870882);
            } else {
                return 3.6934776890867886;
            }
        }
    } else {
        if (z0 <= 2.0698789358139034) {
            if (z0 <= 2.0644658803939815) {
                return 0.1594960932214389;
            } else {
                return 1.2051465480021504;
            }
        } else {
            if (z0 <= 2.0803524255752563) {
                return -(0.8990341156511096);
            } else {
                return 0.016167954982202828;
            }
        }
    }
    return 0.0;
}

double _tree94(double z0, double z1, double z4, double z9) {
    if (z1 <= 0.4003589302301407) {
        if (z9 <= 0.5000000298023224) {
            if (z0 <= 4.797077417373657) {
                return -(0.005489063387718346);
            } else {
                return -(0.2596528438124458);
            }
        } else {
            if (z0 <= -(0.8664340674877167)) {
                return -(0.9153800950407718);
            } else {
                return 0.3262231274102304;
            }
        }
    } else {
        if (z4 <= 0.5000000298023224) {
            if (z0 <= 0.6132549345493317) {
                return 0.0419380146300608;
            } else {
                return -(0.024614225783664335);
            }
        } else {
            if (z0 <= 1.085380971431732) {
                return -(0.050067770594475286);
            } else {
                return 0.11511147327544485;
            }
        }
    }
    return 0.0;
}

double _tree95(double z0, double z1) {
    if (z0 <= 3.7780996561050415) {
        if (z0 <= 3.4678992033004756) {
            if (z0 <= 3.4643689393997192) {
                return -(0.0001349458435289387);
            } else {
                return 2.284096083028729;
            }
        } else {
            if (z0 <= 3.5059093236923213) {
                return -(0.9195308488557788);
            } else {
                return -(0.07410557289979981);
            }
        }
    } else {
        if (z0 <= 3.785160422325134) {
            if (z1 <= 2.3318938016891475) {
                return 2.171833453429365;
            } else {
                return -(2.1303550762397867);
            }
        } else {
            if (z0 <= 3.793515563011169) {
                return -(1.7291824084144864);
            } else {
                return 0.07487421596911839;
            }
        }
    }
    return 0.0;
}

double _tree96(double z0, double z7, double z10) {
    if (z10 <= 0.5000000298023224) {
        if (z0 <= -(1.3584477305412292)) {
            if (z0 <= -(1.3599775433540346)) {
                return -(0.1815869808921542);
            } else {
                return -(1.1716092592219682);
            }
        } else {
            if (z0 <= -(1.3364418148994448)) {
                return 0.2415683948527871;
            } else {
                return -(0.002220270221449021);
            }
        }
    } else {
        if (z0 <= -(1.355388104915619)) {
            if (z7 <= 0.5000000298023224) {
                return 0.5658251457653557;
            } else {
                return -(2.044486489470843);
            }
        } else {
            if (z0 <= -(1.3456208109855652)) {
                return -(1.5627472519797576);
            } else {
                return 0.017673109542004385;
            }
        }
    }
    return 0.0;
}

double _tree97(double z0, double z10) {
    if (z0 <= -(0.5948322117328645)) {
        if (z0 <= -(0.6371964514255525)) {
            if (z0 <= -(0.6388439238071443)) {
                return -(0.00889249300127307);
            } else {
                return 0.49766171548215327;
            }
        } else {
            if (z10 <= 0.5000000298023224) {
                return -(0.054536343776262104);
            } else {
                return -(0.3941071188712464);
            }
        }
    } else {
        if (z0 <= -(0.5823582708835603)) {
            if (z0 <= -(0.5837704241275788)) {
                return 0.11246845121268197;
            } else {
                return 0.9371895070747872;
            }
        } else {
            if (z0 <= -(0.5814168155193329)) {
                return -(0.6572646041785511);
            } else {
                return 0.00422181356601414;
            }
        }
    }
    return 0.0;
}

double _tree98(double z0, double z1, double z7) {
    if (z1 <= 8.12649869918823) {
        if (z0 <= -(1.3988113999366762)) {
            if (z7 <= 0.5000000298023224) {
                return -(1.0717464097995801);
            } else {
                return -(1.2455485376293969);
            }
        } else {
            if (z0 <= -(1.3983407616615298)) {
                return 2.2675464590132877;
            } else {
                return 2.6887804619830757e-5;
            }
        }
    } else {
        return 1.569892061902576;
    }
    return 0.0;
}

double _tree99(double z0, double z1) {
    if (z1 <= 6.194963693618774) {
        if (z0 <= 3.6448875665664673) {
            if (z0 <= 3.567455172538757) {
                return -(0.0002678989836655438);
            } else {
                return -(0.5860387018582744);
            }
        } else {
            if (z0 <= 3.6579500436782837) {
                return 2.0008673728439526;
            } else {
                return 0.040200481114758456;
            }
        }
    } else {
        if (z0 <= 1.5000799298286438) {
            if (z0 <= 0.3791924864053726) {
                return 0.12988651672161405;
            } else {
                return 2.7895778123960153;
            }
        } else {
            if (z0 <= 3.2806729078292842) {
                return -(1.6371058799549858);
            } else {
                return -(2.0045321602449118);
            }
        }
    }
    return 0.0;
}

double _tree100(double z0, double z7) {
    if (z7 <= 0.5000000298023224) {
        if (z0 <= -(1.0215342640876772)) {
            if (z0 <= -(1.056837856769562)) {
                return -(0.00276685182768301);
            } else {
                return -(0.19084759280507674);
            }
        } else {
            if (z0 <= -(1.0208281874656677)) {
                return 1.4344307547708735;
            } else {
                return 0.0008425338968312256;
            }
        }
    } else {
        if (z0 <= 0.08652612939476967) {
            if (z0 <= -(0.05633552931249142)) {
                return 0.03521745797660047;
            } else {
                return -(0.35637554936532845);
            }
        } else {
            if (z0 <= 0.14230572432279584) {
                return 1.0601960524375815;
            } else {
                return 0.015164647489864298;
            }
        }
    }
    return 0.0;
}

double _tree101(double z0, double z4, double z5, double z10) {
    if (z0 <= 7.334460496902466) {
        if (z0 <= 6.253230810165404) {
            if (z0 <= 6.233107805252075) {
                return -(5.540790744142899e-5);
            } else {
                return 2.750585912011734;
            }
        } else {
            if (z10 <= 0.5000000298023224) {
                return -(1.1596636245512277);
            } else {
                return -(1.3881077400949045);
            }
        }
    } else {
        if (z0 <= 7.757279634475708) {
            if (z4 <= 0.5000000298023224) {
                return 2.5044325076074045;
            } else {
                return -(1.1242421217812677);
            }
        } else {
            if (z5 <= 0.5000000298023224) {
                return -(1.2830337809496306);
            } else {
                return -(1.1228048847553749);
            }
        }
    }
    return 0.0;
}

double _tree102(double z0, double z10) {
    if (z10 <= 0.5000000298023224) {
        if (z0 <= 5.417596101760864) {
            if (z0 <= 5.070327043533325) {
                return -(0.0020224077444504183);
            } else {
                return 0.5670305608078703;
            }
        } else {
            if (z0 <= 5.910668611526489) {
                return -(1.1630467765141526);
            } else {
                return 0.2506127526563049;
            }
        }
    } else {
        if (z0 <= -(0.12670724838972092)) {
            if (z0 <= -(0.12964921444654465)) {
                return 0.04056584079706866;
            } else {
                return 1.6288329279065314;
            }
        } else {
            if (z0 <= -(0.0831662230193615)) {
                return -(0.5750450101427882);
            } else {
                return 0.013877072974274382;
            }
        }
    }
    return 0.0;
}

double _tree103(double z0) {
    if (z0 <= -(1.165337383747101)) {
        if (z0 <= -(1.2804269194602966)) {
            if (z0 <= -(1.2808975577354431)) {
                return 0.040364784713664614;
            } else {
                return 2.18860184603455;
            }
        } else {
            if (z0 <= -(1.2759551405906677)) {
                return -(0.542387227018095);
            } else {
                return -(0.07206774390892394);
            }
        }
    } else {
        if (z0 <= -(1.1535695195198061)) {
            if (z0 <= -(1.1547463536262514)) {
                return 0.13074663931168548;
            } else {
                return 0.7346241236560309;
            }
        } else {
            if (z0 <= -(1.146744191646576)) {
                return -(0.27268257248890676);
            } else {
                return 0.0014688482972995414;
            }
        }
    }
    return 0.0;
}

double _tree104(double z0, double z1) {
    if (z1 <= 0.4003589302301407) {
        if (z0 <= 4.797077417373657) {
            if (z0 <= 4.537596464157104) {
                return -(0.004937674286239178);
            } else {
                return 0.5762343896555604;
            }
        } else {
            if (z0 <= 4.939939260482787) {
                return -(0.9946879039774568);
            } else {
                return -(0.12329801624809372);
            }
        }
    } else {
        if (z0 <= -(1.3902209401130678)) {
            if (z0 <= -(1.3915154337882996)) {
                return 0.29984632693799196;
            } else {
                return 6.519991242772815;
            }
        } else {
            if (z0 <= -(1.3578593134880066)) {
                return -(0.5648780119665521);
            } else {
                return 0.01114086022418377;
            }
        }
    }
    return 0.0;
}

double _tree105(double z0, double z1) {
    if (z0 <= 0.9149825274944304) {
        if (z0 <= 0.9126289784908294) {
            if (z0 <= 0.5972506701946257) {
                return -(1.4687724702529876e-5);
            } else {
                return -(0.03425782409738329);
            }
        } else {
            if (z1 <= 0.4003589302301407) {
                return -(0.6390256966064352);
            } else {
                return -(1.3863075360275356);
            }
        }
    } else {
        if (z0 <= 0.9248675405979156) {
            if (z0 <= 0.9225139915943146) {
                return 0.1820149028867428;
            } else {
                return 1.195561740078885;
            }
        } else {
            if (z0 <= 0.930516093969345) {
                return -(0.46427831121557994);
            } else {
                return 0.012774635220947515;
            }
        }
    }
    return 0.0;
}

double _tree106(double z0, double z5) {
    if (z0 <= -(1.3740989565849304)) {
        if (z0 <= -(1.3896325230598452)) {
            if (z0 <= -(1.391279995441437)) {
                return -(0.0882915980055724);
            } else {
                return 2.6411381781257828;
            }
        } else {
            if (z5 <= 0.5000000298023224) {
                return -(0.3165977331398906);
            } else {
                return -(1.1995322037168108);
            }
        }
    } else {
        if (z0 <= -(1.3731575608253481)) {
            if (z0 <= -(1.3736281991004946)) {
                return 0.7158273801447501;
            } else {
                return 3.1636688458567366;
            }
        } else {
            if (z0 <= -(1.3670383095741274)) {
                return -(0.6149551798386355);
            } else {
                return 0.00027899646814140975;
            }
        }
    }
    return 0.0;
}

double _tree107(double z0, double z1, double z7) {
    if (z1 <= 8.12649869918823) {
        if (z0 <= -(1.3988113999366762)) {
            if (z7 <= 0.5000000298023224) {
                return -(1.06535284565334);
            } else {
                return -(1.223054790845688);
            }
        } else {
            if (z0 <= -(1.3983407616615298)) {
                return 1.9857517193886338;
            } else {
                return 2.0637075786877085e-5;
            }
        }
    } else {
        return 1.4802492530924856;
    }
    return 0.0;
}

double _tree108(double z0, double z1, double z3) {
    if (z1 <= 2.3318938016891475) {
        if (z0 <= 1.8378171324729917) {
            if (z0 <= 1.8314624428749084) {
                return -(0.003024052849144888);
            } else {
                return -(0.6007547991080379);
            }
        } else {
            if (z0 <= 1.8491142392158508) {
                return 0.793841887964567;
            } else {
                return 0.018859781387863345;
            }
        }
    } else {
        if (z3 <= 0.5000000298023224) {
            if (z0 <= 1.8107510209083557) {
                return 0.09480263374542101;
            } else {
                return -(0.33252505862439674);
            }
        } else {
            if (z0 <= 2.2631071805953975) {
                return -(0.19501063588064874);
            } else {
                return 1.1662048207244453;
            }
        }
    }
    return 0.0;
}

double _tree109(double z0, double z10) {
    if (z10 <= 0.5000000298023224) {
        if (z0 <= -(0.21072968095541)) {
            if (z0 <= -(0.21637824922800067)) {
                return -(0.009137175512584975);
            } else {
                return -(0.45532984267762533);
            }
        } else {
            if (z0 <= -(0.207199327647686)) {
                return 0.47847352112114044;
            } else {
                return 0.005258653691347807;
            }
        }
    } else {
        if (z0 <= 1.9693816304206848) {
            if (z0 <= 1.9617325663566587) {
                return 0.01056951412863124;
            } else {
                return -(2.127727491140032);
            }
        } else {
            if (z0 <= 2.2169772386550903) {
                return 0.6139302503924922;
            } else {
                return -(0.006260582169634533);
            }
        }
    }
    return 0.0;
}

double _tree110(double z0, double z1, double z7) {
    if (z7 <= 0.5000000298023224) {
        if (z1 <= 4.2634289264678955) {
            if (z0 <= 5.282736539840697) {
                return -(0.0019451661370633196);
            } else {
                return -(0.3209240398794734);
            }
        } else {
            if (z0 <= 0.4053171128034591) {
                return -(0.03906078912751169);
            } else {
                return 0.3164635549845002;
            }
        }
    } else {
        if (z0 <= -(1.0317723155021667)) {
            if (z0 <= -(1.0375385880470278)) {
                return 0.1035200572843243;
            } else {
                return 2.0552698473314837;
            }
        } else {
            if (z0 <= -(1.0125907063484194)) {
                return -(0.7404727623826224);
            } else {
                return 0.01803198443069072;
            }
        }
    }
    return 0.0;
}

double _tree111(double z0, double z1) {
    if (z0 <= 7.334460496902466) {
        if (z0 <= 6.253230810165404) {
            if (z0 <= 6.233107805252075) {
                return -(5.226697839216964e-5);
            } else {
                return 2.4262694848833517;
            }
        } else {
            if (z1 <= 0.4003589302301407) {
                return -(1.1387652945791613);
            } else {
                return -(1.3417573938705074);
            }
        }
    } else {
        if (z0 <= 7.351170778274536) {
            return 3.727729671452209;
        } else {
            if (z1 <= 2.3318938016891475) {
                return -(0.40883611867818076);
            } else {
                return 1.826444364321808;
            }
        }
    }
    return 0.0;
}

double _tree112(double z0, double z5) {
    if (z0 <= 3.7479740381240845) {
        if (z0 <= 3.7320874929428096) {
            if (z0 <= 3.726909518241882) {
                return -(0.0005353320700636872);
            } else {
                return 2.6167758720185863;
            }
        } else {
            if (z5 <= 0.5000000298023224) {
                return -(0.9612000575523593);
            } else {
                return 0.7681272850110462;
            }
        }
    } else {
        if (z0 <= 3.7495037317276) {
            return 4.35044071204197;
        } else {
            if (z0 <= 4.128663778305053) {
                return 0.2211465015593533;
            } else {
                return -(0.06439726835457953);
            }
        }
    }
    return 0.0;
}

double _tree113(double z0, double z6, double z10) {
    if (z0 <= -(1.3973992466926577)) {
        if (z10 <= 0.5000000298023224) {
            if (z0 <= -(1.3988113999366762)) {
                return -(1.072654248097858);
            } else {
                return 0.22995327970189733;
            }
        } else {
            return -(1.5076131560518036);
        }
    } else {
        if (z0 <= -(1.3969286084175112)) {
            if (z6 <= 0.5000000298023224) {
                return -(1.0876460402986423);
            } else {
                return 6.4123836976190285;
            }
        } else {
            if (z0 <= -(1.3896325230598452)) {
                return 0.43421441981762826;
            } else {
                return -(0.0001121075610144917);
            }
        }
    }
    return 0.0;
}

double _tree114(double z0, double z1, double z8) {
    if (z0 <= -(1.3740989565849304)) {
        if (z0 <= -(1.3757464289665222)) {
            if (z8 <= 0.5000000298023224) {
                return 0.44306141902769464;
            } else {
                return -(0.21955665790391357);
            }
        } else {
            if (z8 <= 0.5000000298023224) {
                return -(1.8592292250842706);
            } else {
                return -(1.149711589556734);
            }
        }
    } else {
        if (z0 <= -(1.3731575608253481)) {
            if (z1 <= 0.4003589302301407) {
                return 1.069866943059773;
            } else {
                return 3.8834637590738175;
            }
        } else {
            if (z0 <= -(1.3670383095741274)) {
                return -(0.5752257995278486);
            } else {
                return 0.00026124401233737786;
            }
        }
    }
    return 0.0;
}

double _tree115(double z0, double z1) {
    if (z0 <= -(1.2884290814399721)) {
        if (z0 <= -(1.2893704771995544)) {
            if (z1 <= 2.3318938016891475) {
                return 0.05119630798188507;
            } else {
                return -(0.6482600761697839);
            }
        } else {
            if (z0 <= -(1.2886644005775454)) {
                return 1.1873564313791771;
            } else {
                return 2.796645418796557;
            }
        }
    } else {
        if (z0 <= -(1.2201755642890932)) {
            if (z0 <= -(1.228883683681488)) {
                return -(0.057715808029838285);
            } else {
                return -(0.4222160294949862);
            }
        } else {
            if (z0 <= -(1.218998730182648)) {
                return 0.735484468913179;
            } else {
                return 0.0006563958716353526;
            }
        }
    }
    return 0.0;
}

double _tree116(double z0, double z1) {
    if (z1 <= 6.194963693618774) {
        if (z0 <= 3.780806422233581) {
            if (z0 <= 3.567455172538757) {
                return -(0.0001499783063523856);
            } else {
                return -(0.19624293236203927);
            }
        } else {
            if (z0 <= 3.785160422325134) {
                return 1.8848399468730337;
            } else {
                return 0.04694753349811718;
            }
        }
    } else {
        if (z0 <= 1.5000799298286438) {
            if (z0 <= 0.3791924864053726) {
                return 0.10008242987241374;
            } else {
                return 2.2951083289815672;
            }
        } else {
            if (z0 <= 3.2806729078292842) {
                return -(1.5534946423029323);
            } else {
                return -(1.8179687787993644);
            }
        }
    }
    return 0.0;
}

double _tree117(double z0, double z1, double z7, double z9) {
    if (z1 <= 0.4003589302301407) {
        if (z9 <= 0.5000000298023224) {
            if (z0 <= 4.797077417373657) {
                return -(0.004384233568703705);
            } else {
                return -(0.22371104437336867);
            }
        } else {
            if (z7 <= 0.5000000298023224) {
                return 0.08344845868195502;
            } else {
                return 1.40649468333843;
            }
        }
    } else {
        if (z0 <= 4.803079366683959) {
            if (z0 <= 4.719291925430298) {
                return 0.008875743079406839;
            } else {
                return -(1.365652923890706);
            }
        } else {
            if (z0 <= 4.87674593925476) {
                return 2.174116483616136;
            } else {
                return 0.19937184105694403;
            }
        }
    }
    return 0.0;
}

double _tree118(double z0, double z1, double z5, double z10) {
    if (z1 <= 8.12649869918823) {
        if (z10 <= 0.5000000298023224) {
            if (z5 <= 0.5000000298023224) {
                return -(0.007228539140723666);
            } else {
                return 0.01158900660154923;
            }
        } else {
            if (z0 <= -(0.12670724838972092)) {
                return 0.03908586767824944;
            } else {
                return -(0.014270082774411125);
            }
        }
    } else {
        return 1.4071829775231148;
    }
    return 0.0;
}

double _tree119(double z0, double z2, double z7) {
    if (z7 <= 0.5000000298023224) {
        if (z0 <= -(1.3446792960166931)) {
            if (z2 <= 0.5000000298023224) {
                return -(0.20465235060329617);
            } else {
                return 1.3495269011285917;
            }
        } else {
            if (z0 <= -(1.3416196703910828)) {
                return 0.9300794949803518;
            } else {
                return -(0.001436409865562335);
            }
        }
    } else {
        if (z0 <= 1.222947061061859) {
            if (z0 <= 1.1242148280143736) {
                return 0.019279603240279004;
            } else {
                return -(0.5657421619051292);
            }
        } else {
            if (z0 <= 1.2283603549003599) {
                return 3.1589469266917587;
            } else {
                return 0.08133693653843453;
            }
        }
    }
    return 0.0;
}

double _tree120(double z0) {
    if (z0 <= -(0.5948322117328645)) {
        if (z0 <= -(0.7141581475734711)) {
            if (z0 <= -(0.7235724031925203)) {
                return -(0.006813473479127882);
            } else {
                return 0.3329687669825778;
            }
        } else {
            if (z0 <= -(0.7054499685764314)) {
                return -(0.3978551284429419);
            } else {
                return -(0.02407764596583526);
            }
        }
    } else {
        if (z0 <= -(0.574356108903885)) {
            if (z0 <= -(0.5745914876461029)) {
                return 0.12868944916943015;
            } else {
                return 5.093755909242328;
            }
        } else {
            if (z0 <= -(0.5637650787830354)) {
                return -(0.2381000013430168);
            } else {
                return 0.004016513690763909;
            }
        }
    }
    return 0.0;
}

double _tree121(double z0, double z1) {
    if (z0 <= 7.334460496902466) {
        if (z0 <= 6.253230810165404) {
            if (z0 <= 6.233107805252075) {
                return -(5.2419363927505414e-5);
            } else {
                return 2.196984692232802;
            }
        } else {
            if (z1 <= 0.4003589302301407) {
                return -(1.1217852436685012);
            } else {
                return -(1.3047875712364168);
            }
        }
    } else {
        if (z0 <= 7.508624792098998) {
            if (z0 <= 7.351170778274536) {
                return 2.9219019362637204;
            } else {
                return 1.4442695508556274;
            }
        } else {
            if (z1 <= 2.3318938016891475) {
                return -(1.1467277659898787);
            } else {
                return 1.6750579161348167;
            }
        }
    }
    return 0.0;
}

double _tree122(double z0, double z1, double z5, double z8) {
    if (z0 <= -(1.2945483326911928)) {
        if (z0 <= -(1.2954897284507751)) {
            if (z1 <= 2.3318938016891475) {
                return 0.0556885731744515;
            } else {
                return -(0.7989093828228279);
            }
        } else {
            if (z5 <= 0.5000000298023224) {
                return 1.4337038123740016;
            } else {
                return 4.535625891185657;
            }
        }
    } else {
        if (z0 <= -(1.2938422560691833)) {
            if (z8 <= 0.5000000298023224) {
                return -(1.9029807907609686);
            } else {
                return -(1.141243690183662);
            }
        } else {
            if (z0 <= -(1.1669848561286928)) {
                return -(0.06004633041315534);
            } else {
                return 0.0013573849814135788;
            }
        }
    }
    return 0.0;
}

double _tree123(double z0, double z1, double z6, double z7) {
    if (z1 <= 2.3318938016891475) {
        if (z0 <= -(1.2860754132270815)) {
            if (z6 <= 0.5000000298023224) {
                return 0.015785918305494976;
            } else {
                return 0.38336665529421965;
            }
        } else {
            if (z0 <= -(1.2837218642234802)) {
                return -(0.7374639093000377);
            } else {
                return -(0.0015500691260078737);
            }
        }
    } else {
        if (z0 <= 1.5495049357414243) {
            if (z0 <= 1.5089057087898254) {
                return 0.04251823725529579;
            } else {
                return 2.4356708365261976;
            }
        } else {
            if (z7 <= 0.5000000298023224) {
                return -(0.12481582047731066);
            } else {
                return -(1.4492370873164164);
            }
        }
    }
    return 0.0;
}

double _tree124(double z0, double z2, double z3) {
    if (z0 <= -(1.3740989565849304)) {
        if (z0 <= -(1.3896325230598452)) {
            if (z0 <= -(1.391279995441437)) {
                return -(0.07958070970668808);
            } else {
                return 2.123357254348918;
            }
        } else {
            if (z3 <= 0.5000000298023224) {
                return -(0.592211141991473);
            } else {
                return 0.48072876174260915;
            }
        }
    } else {
        if (z0 <= -(1.3731575608253481)) {
            if (z2 <= 0.5000000298023224) {
                return 1.506733839692394;
            } else {
                return -(1.2548411855532253);
            }
        } else {
            if (z0 <= -(1.3670383095741274)) {
                return -(0.5388120012030949);
            } else {
                return 0.00023313395189020813;
            }
        }
    }
    return 0.0;
}

double _tree125(double z0) {
    if (z0 <= 0.704102724790573) {
        if (z0 <= 0.6760952770709991) {
            if (z0 <= 0.6751538217067717) {
                return -(0.001671878746211736);
            } else {
                return 0.9309845603502027;
            }
        } else {
            if (z0 <= 0.7022198736667633) {
                return -(0.18931885971825185);
            } else {
                return -(1.0000736117079665);
            }
        }
    } else {
        if (z0 <= 0.7045734226703644) {
            if (z0 <= 0.7043381035327911) {
                return 1.2972487283583969;
            } else {
                return 3.9947435573239116;
            }
        } else {
            if (z0 <= 0.7391709387302399) {
                return 0.18967940483633955;
            } else {
                return 0.0035801177572464815;
            }
        }
    }
    return 0.0;
}

double _tree126(double z0, double z2, double z10) {
    if (z10 <= 0.5000000298023224) {
        if (z0 <= -(1.3446792960166931)) {
            if (z2 <= 0.5000000298023224) {
                return -(0.19615562259796426);
            } else {
                return 1.0645577475855332;
            }
        } else {
            if (z0 <= -(1.3364418148994448)) {
                return 0.5245793688907496;
            } else {
                return -(0.0016625537180412107);
            }
        }
    } else {
        if (z0 <= -(1.355388104915619)) {
            if (z0 <= -(1.362684190273285)) {
                return 0.15621073235510147;
            } else {
                return 1.2596662782644907;
            }
        } else {
            if (z0 <= -(1.3456208109855652)) {
                return -(1.4864002744563567);
            } else {
                return 0.012662715135626432;
            }
        }
    }
    return 0.0;
}

double _tree127(double z0, double z1) {
    if (z0 <= 5.417596101760864) {
        if (z0 <= 5.361698865890502) {
            if (z0 <= 5.282736539840697) {
                return 9.063092337311112e-5;
            } else {
                return -(1.2616209469508066);
            }
        } else {
            if (z0 <= 5.379703760147095) {
                return 2.076468496367952;
            } else {
                return 0.17888940691543612;
            }
        }
    } else {
        if (z0 <= 5.770984411239623) {
            if (z1 <= 0.4003589302301407) {
                return -(1.1283362191030168);
            } else {
                return -(1.1812428056629891);
            }
        } else {
            if (z0 <= 5.779810667037963) {
                return 2.274487880116585;
            } else {
                return 0.026537594043393228;
            }
        }
    }
    return 0.0;
}

double _tree128(double z0, double z1) {
    if (z0 <= 3.703844666481018) {
        if (z0 <= 3.6858397722244263) {
            if (z0 <= 3.682074189186096) {
                return -(0.0004013900997918332);
            } else {
                return 4.371639240404073;
            }
        } else {
            if (z1 <= 0.4003589302301407) {
                return -(1.2614027519471904);
            } else {
                return -(1.3891797017215606);
            }
        }
    } else {
        if (z0 <= 3.70513904094696) {
            return 4.690239350584063;
        } else {
            if (z0 <= 4.128663778305053) {
                return 0.15534515893968573;
            } else {
                return -(0.058739187214389185);
            }
        }
    }
    return 0.0;
}

double _tree129(double z0, double z7) {
    if (z7 <= 0.5000000298023224) {
        if (z0 <= -(1.040127456188202)) {
            if (z0 <= -(1.056837856769562)) {
                return 0.0029251631244973926;
            } else {
                return -(0.35841327463854394);
            }
        } else {
            if (z0 <= -(0.9452786743640901)) {
                return 0.07230884362288724;
            } else {
                return -(0.002465415562103945);
            }
        }
    } else {
        if (z0 <= -(1.3243209719657898)) {
            if (z0 <= -(1.3265568614006042)) {
                return 0.28995991429075907;
            } else {
                return 3.3555401264257085;
            }
        } else {
            if (z0 <= -(1.3152597546577456)) {
                return -(0.8092747293505995);
            } else {
                return 0.018504819974705156;
            }
        }
    }
    return 0.0;
}

double _tree130(double z0, double z1) {
    if (z0 <= -(0.013265212532132864)) {
        if (z0 <= -(0.03515340574085713)) {
            if (z0 <= -(0.03538876213133336)) {
                return -(0.003417045673340584);
            } else {
                return 3.570331160823425;
            }
        } else {
            if (z1 <= 4.2634289264678955) {
                return -(0.17844633021495027);
            } else {
                return 2.163314212174934;
            }
        }
    } else {
        if (z0 <= -(0.009028788190335035)) {
            if (z1 <= 0.4003589302301407) {
                return 0.2105789236263187;
            } else {
                return 0.7921818793522998;
            }
        } else {
            if (z0 <= 0.1223003827035427) {
                return 0.06138046219504212;
            } else {
                return -(0.0015857384521102207);
            }
        }
    }
    return 0.0;
}

double _tree131(double z0, double z1) {
    if (z1 <= 8.12649869918823) {
        if (z0 <= 2.015040755271911) {
            if (z0 <= 2.0093923807144165) {
                return -(0.0010720414131988106);
            } else {
                return -(0.7360385402810538);
            }
        } else {
            if (z0 <= 2.0263379812240596) {
                return 0.8048053751340644;
            } else {
                return 0.009424867399476688;
            }
        }
    } else {
        return 1.3501472335525666;
    }
    return 0.0;
}

double _tree132(double z0, double z1) {
    if (z0 <= 7.334460496902466) {
        if (z0 <= 6.253230810165404) {
            if (z0 <= 6.233107805252075) {
                return -(5.005166846530937e-5);
            } else {
                return 2.008341130242487;
            }
        } else {
            if (z1 <= 0.4003589302301407) {
                return -(1.1093496409003094);
            } else {
                return -(1.26684001991662);
            }
        }
    } else {
        if (z0 <= 7.508624792098998) {
            if (z0 <= 7.351170778274536) {
                return 2.4385160774700023;
            } else {
                return 1.384673330805168;
            }
        } else {
            if (z1 <= 2.3318938016891475) {
                return -(1.13114032518248);
            } else {
                return 1.5794655765254237;
            }
        }
    }
    return 0.0;
}

double _tree133(double z0, double z5, double z8) {
    if (z0 <= -(1.3973992466926577)) {
        if (z8 <= 0.5000000298023224) {
            return -(1.4572079142555692);
        } else {
            if (z0 <= -(1.3988113999366762)) {
                return -(1.0634010437987371);
            } else {
                return 0.26337383162684336;
            }
        }
    } else {
        if (z0 <= -(1.3969286084175112)) {
            if (z5 <= 0.5000000298023224) {
                return 3.9133893954700314;
            } else {
                return -(1.0741784446596732);
            }
        } else {
            if (z0 <= -(1.3896325230598452)) {
                return 0.3609998900463709;
            } else {
                return -(9.990844095330563e-5);
            }
        }
    }
    return 0.0;
}

double _tree134(double z0, double z1, double z2) {
    if (z0 <= -(1.165337383747101)) {
        if (z0 <= -(1.1719273924827576)) {
            if (z0 <= -(1.1740456223487856)) {
                return -(0.028660030422586892);
            } else {
                return 0.7583020444550905;
            }
        } else {
            if (z1 <= 4.2634289264678955) {
                return -(0.35489747982923486);
            } else {
                return 3.6680992030843824;
            }
        }
    } else {
        if (z0 <= -(1.1535695195198061)) {
            if (z2 <= 0.5000000298023224) {
                return 0.2329324026044322;
            } else {
                return -(0.5356608596776877);
            }
        } else {
            if (z0 <= -(1.146744191646576)) {
                return -(0.2500400248733783);
            } else {
                return 0.001073087622779154;
            }
        }
    }
    return 0.0;
}

double _tree135(double z0, double z1, double z4, double z9) {
    if (z1 <= 0.4003589302301407) {
        if (z9 <= 0.5000000298023224) {
            if (z0 <= -(1.3961048722267153)) {
                return -(1.1205980651477214);
            } else {
                return -(0.004014949102363969);
            }
        } else {
            if (z0 <= -(0.8664340674877167)) {
                return -(0.8840752536765354);
            } else {
                return 0.2723374242758894;
            }
        }
    } else {
        if (z4 <= 0.5000000298023224) {
            if (z0 <= 0.6132549345493317) {
                return 0.03556052226134232;
            } else {
                return -(0.027318187740919098);
            }
        } else {
            if (z0 <= 0.23103528469800946) {
                return -(0.0672420525913564);
            } else {
                return 0.05652504320841415;
            }
        }
    }
    return 0.0;
}

double _tree136(double z0, double z1) {
    if (z1 <= 6.194963693618774) {
        if (z0 <= 3.780806422233581) {
            if (z0 <= 3.347043395042419) {
                return 0.0001339241311212103;
            } else {
                return -(0.12444653643453042);
            }
        } else {
            if (z0 <= 3.785160422325134) {
                return 1.7238205468074668;
            } else {
                return 0.04104927697536012;
            }
        }
    } else {
        if (z0 <= 1.5000799298286438) {
            if (z0 <= 0.2666918784379959) {
                return 0.0028108935036836964;
            } else {
                return 1.917797416774454;
            }
        } else {
            if (z0 <= 3.2806729078292842) {
                return -(1.4675528931104005);
            } else {
                return -(1.6687076807214223);
            }
        }
    }
    return 0.0;
}

double _tree137(double z0, double z4, double z5) {
    if (z0 <= 2.6488569974899288) {
        if (z0 <= 2.527413010597229) {
            if (z0 <= 2.5268245935440063) {
                return 6.7369015178746e-5;
            } else {
                return 2.70949960459794;
            }
        } else {
            if (z4 <= 0.5000000298023224) {
                return -(0.41156441611119143);
            } else {
                return 0.1647918246796497;
            }
        }
    } else {
        if (z0 <= 2.6490923166275024) {
            return 5.673505753426329;
        } else {
            if (z5 <= 0.5000000298023224) {
                return 0.07246599170797671;
            } else {
                return -(0.0808479823707972);
            }
        }
    }
    return 0.0;
}

double _tree138(double z0, double z1) {
    if (z0 <= 0.9149825274944304) {
        if (z0 <= 0.9123935997486115) {
            if (z0 <= 0.5972506701946257) {
                return 0.000646901215954471;
            } else {
                return -(0.03152432374820806);
            }
        } else {
            if (z1 <= 0.4003589302301407) {
                return -(0.5476546569331914);
            } else {
                return -(1.3416310505693243);
            }
        }
    } else {
        if (z0 <= 0.9248675405979156) {
            if (z0 <= 0.9225139915943146) {
                return 0.1564858213914149;
            } else {
                return 0.993710610238837;
            }
        } else {
            if (z0 <= 0.930516093969345) {
                return -(0.4305627442980121);
            } else {
                return 0.009397751803869796;
            }
        }
    }
    return 0.0;
}

double _tree139(double z0, double z8) {
    if (z0 <= 5.417596101760864) {
        if (z0 <= 5.1661174297332755) {
            if (z0 <= 5.131166696548462) {
                return 3.558078551023759e-5;
            } else {
                return -(1.6196935139483544);
            }
        } else {
            if (z0 <= 5.193418741226196) {
                return 1.856979678768447;
            } else {
                return 0.1433155584717627;
            }
        }
    } else {
        if (z0 <= 5.9016077518463135) {
            if (z8 <= 0.5000000298023224) {
                return 0.7936350646783732;
            } else {
                return -(1.1321123985797323);
            }
        } else {
            if (z0 <= 5.948443651199341) {
                return 2.402040176731168;
            } else {
                return -(0.008281109726639141);
            }
        }
    }
    return 0.0;
}

double _tree140(double z0, double z8) {
    if (z0 <= 0.42402796447277064) {
        if (z0 <= 0.3875476568937301) {
            if (z0 <= 0.3854294270277023) {
                return -(0.001257199036190837);
            } else {
                return 0.7833359977432217;
            }
        } else {
            if (z8 <= 0.5000000298023224) {
                return 0.21385839714197588;
            } else {
                return -(0.26129849797723487);
            }
        }
    } else {
        if (z0 <= 0.43414832651615143) {
            if (z0 <= 0.43155939877033234) {
                return 0.255890339360476;
            } else {
                return 0.9776462231146476;
            }
        } else {
            if (z0 <= 0.4348544031381607) {
                return -(1.2424711041365095);
            } else {
                return 0.004403094603547195;
            }
        }
    }
    return 0.0;
}

double _tree141(double z0, double z5, double z10) {
    if (z10 <= 0.5000000298023224) {
        if (z5 <= 0.5000000298023224) {
            if (z0 <= 2.6246153116226196) {
                return -(0.00887362330471168);
            } else {
                return 0.06694761480630028;
            }
        } else {
            if (z0 <= 6.036937952041626) {
                return 0.010378590240300516;
            } else {
                return 1.3316629420356025;
            }
        }
    } else {
        if (z0 <= -(0.12670724838972092)) {
            if (z0 <= -(0.12964921444654465)) {
                return 0.03282778210018575;
            } else {
                return 1.4498530201466944;
            }
        } else {
            if (z0 <= -(0.0831662230193615)) {
                return -(0.522123085195506);
            } else {
                return 0.008337993881062216;
            }
        }
    }
    return 0.0;
}

double _tree142(double z0, double z4) {
    if (z0 <= 5.417596101760864) {
        if (z0 <= 5.361698865890502) {
            if (z0 <= 5.282736539840697) {
                return 9.085032431927867e-5;
            } else {
                return -(1.2333056438140078);
            }
        } else {
            if (z4 <= 0.5000000298023224) {
                return 1.3956446891593608;
            } else {
                return -(1.254352502913444);
            }
        }
    } else {
        if (z0 <= 6.06341528892517) {
            if (z0 <= 5.9645655155181885) {
                return -(0.5289643013100258);
            } else {
                return -(1.2651350027346384);
            }
        } else {
            if (z0 <= 6.086715936660767) {
                return 4.073441853892853;
            } else {
                return 0.09624971220142979;
            }
        }
    }
    return 0.0;
}

double _tree143(double z0, double z1) {
    if (z0 <= 4.169615983963013) {
        if (z0 <= 4.082416296005248) {
            if (z0 <= 4.068530321121216) {
                return 8.718341178776878e-5;
            } else {
                return -(1.04274054024153);
            }
        } else {
            if (z0 <= 4.085475683212279) {
                return 3.2980780803441068;
            } else {
                return 0.42021573613839813;
            }
        }
    } else {
        if (z0 <= 4.1923277378082275) {
            if (z1 <= 0.4003589302301407) {
                return -(1.2930186838960336);
            } else {
                return -(1.4113748281749972);
            }
        } else {
            if (z0 <= 4.197741270065307) {
                return 3.2853485393660686;
            } else {
                return -(0.038903293385565696);
            }
        }
    }
    return 0.0;
}

double _tree144(double z0, double z7) {
    if (z7 <= 0.5000000298023224) {
        if (z0 <= 4.169615983963013) {
            if (z0 <= 4.082416296005248) {
                return -(0.0013132049134218363);
            } else {
                return 0.532019020047567;
            }
        } else {
            if (z0 <= 4.1923277378082275) {
                return -(1.2770514915465732);
            } else {
                return -(0.0413072731871463);
            }
        }
    } else {
        if (z0 <= -(1.0317723155021667)) {
            if (z0 <= -(1.0375385880470278)) {
                return 0.09414400810689715;
            } else {
                return 1.6102812003924434;
            }
        } else {
            if (z0 <= -(1.0194160342216492)) {
                return -(0.994379286922555);
            } else {
                return 0.011122959212897202;
            }
        }
    }
    return 0.0;
}

double _tree145(double z0, double z3) {
    if (z0 <= -(0.5948322117328645)) {
        if (z0 <= -(0.6682635247707367)) {
            if (z0 <= -(0.669440358877182)) {
                return -(0.0036082812792391583);
            } else {
                return 0.7451795342799445;
            }
        } else {
            if (z0 <= -(0.6656745970249177)) {
                return -(0.6601986684011014);
            } else {
                return -(0.03786450350598695);
            }
        }
    } else {
        if (z0 <= -(0.5927139818668367)) {
            if (z0 <= -(0.5929493010044099)) {
                return 0.2343180110402962;
            } else {
                return 1.8181811813744915;
            }
        } else {
            if (z3 <= 0.5000000298023224) {
                return 0.008087566927229561;
            } else {
                return -(0.027407341683341083);
            }
        }
    }
    return 0.0;
}

double _tree146(double z0, double z1) {
    if (z1 <= 8.12649869918823) {
        if (z0 <= -(1.2884290814399721)) {
            if (z0 <= -(1.2893704771995544)) {
                return 0.02427942570462048;
            } else {
                return 1.3762586606492073;
            }
        } else {
            if (z0 <= -(1.2879583239555359)) {
                return -(1.2625010607580867);
            } else {
                return -(0.0005338331334926517);
            }
        }
    } else {
        return 1.3079485337141579;
    }
    return 0.0;
}

double _tree147(double z0, double z1) {
    if (z0 <= -(1.1669848561286928)) {
        if (z0 <= -(1.1719273924827576)) {
            if (z0 <= -(1.1740456223487856)) {
                return -(0.026816104317750965);
            } else {
                return 0.6459130256723772;
            }
        } else {
            if (z1 <= 4.2634289264678955) {
                return -(0.4766367603345105);
            } else {
                return 2.8640154934331283;
            }
        }
    } else {
        if (z0 <= -(1.0629571080207827)) {
            if (z0 <= -(1.0735481381416323)) {
                return 0.01868508819791872;
            } else {
                return 0.3797047144504942;
            }
        } else {
            if (z0 <= -(1.040127456188202)) {
                return -(0.2162044862174876);
            } else {
                return 0.001233309103566711;
            }
        }
    }
    return 0.0;
}

double _tree148(double z0, double z2, double z5, double z10) {
    if (z0 <= -(1.2945483326911928)) {
        if (z0 <= -(1.2950190901756287)) {
            if (z2 <= 0.5000000298023224) {
                return 0.007494019672902767;
            } else {
                return 0.5514533433505131;
            }
        } else {
            if (z5 <= 0.5000000298023224) {
                return 1.3923878458678574;
            } else {
                return 4.5496102376169105;
            }
        }
    } else {
        if (z0 <= -(1.2938422560691833)) {
            if (z10 <= 0.5000000298023224) {
                return -(1.1259580136517493);
            } else {
                return -(1.7573851672034047);
            }
        } else {
            if (z0 <= -(1.2380626797676089)) {
                return -(0.08565561447982595);
            } else {
                return 0.0005289332273502641;
            }
        }
    }
    return 0.0;
}

double _tree149(double z0, double z7) {
    if (z0 <= -(1.3446792960166931)) {
        if (z0 <= -(1.3498571515083315)) {
            if (z0 <= -(1.3507986664772036)) {
                return -(0.07194425699269465);
            } else {
                return 3.0266708694179933;
            }
        } else {
            if (z7 <= 0.5000000298023224) {
                return -(0.7016789526069359);
            } else {
                return -(1.3893330116134495);
            }
        }
    } else {
        if (z0 <= -(1.3416196703910828)) {
            if (z0 <= -(1.3423257470130923)) {
                return 0.39234697153778875;
            } else {
                return 2.063776347246711;
            }
        } else {
            if (z0 <= -(1.3385600447654724)) {
                return -(0.46397936150592406);
            } else {
                return 0.00015346297254867664;
            }
        }
    }
    return 0.0;
}

viv_arr score(viv_arr xf_viv_in, viv_arr xi_viv_in, long long rows) {
    viv_arr card = {0};
    viv_arr ts = {0};
    viv_arr declined = {0};
    viv_arr chargeback = {0};
    long long i = {0};
    viv_rec_Features f0 = {0};
    viv_arr out = {0};
    long long r_card = {0};
    long long r_ts = {0};
    double r_amount = 0;
    long long r_merchant = {0};
    long long r_declined = {0};
    long long r_chargeback = {0};
    long long r_n_prev_1h = {0};
    long long r_status = {0};
    double v1 = 0;
    double v2 = 0;
    double viv__if7 = 0;
    double v3 = 0;
    double viv__if8 = 0;
    double v4 = 0;
    double viv__if9 = 0;
    double v5 = 0;
    double viv__if10 = 0;
    double v6 = 0;
    double viv__if11 = 0;
    double v7 = 0;
    double viv__if12 = 0;
    double v8 = 0;
    long long code9 = {0};
    double viv__if13 = 0;
    double v10 = 0;
    double viv__if14 = 0;
    double v11 = 0;
    double viv__if15 = 0;
    double v12 = 0;
    double z0 = 0;
    double z1 = 0;
    double z2 = 0;
    double z3 = 0;
    double z4 = 0;
    double z5 = 0;
    double z6 = 0;
    double z7 = 0;
    double z8 = 0;
    double z9 = 0;
    double z10 = 0;
    double raw13 = 0;
    double p14 = 0;
    double p15 = 0;
    long long cls16 = {0};
    viv_arr xi = xi_viv_in;
    (void)(xi);
    viv_arr xf = xf_viv_in;
    (void)(xf);
    size_t viv_mark = viv_arena_top;
    if ((viv_i_cmp(viv_i_mul(rows, NF, 24), viv_numel(xf)) != 0.0) || ((long long)(viv_len(xi)) != viv_i_mul(rows, NI, 25))) {
        viv_s_stop(viv_s_catn(10, VIV_LIT(3, "score: ", 7, 7), viv_i_str(rows, 0.0), VIV_LIT(4, " rows need ", 11, 11), viv_i_str(viv_i_mul(rows, NF, 24), 0.0), VIV_LIT(5, " floats and ", 12, 12), viv_i_str(viv_i_mul(rows, NI, 25), 0.0), VIV_LIT(6, " integers; got ", 15, 15), viv_s_fmt_num(viv_numel(xf), 115.0, 0.0, 0.0), VIV_LIT(7, " and ", 5, 5), viv_s_fmt_num(viv_len(xi), 115.0, 0.0, 0.0), 26), 27);
    }
    card = viv_zeros1(viv_i_num(rows, 28));
    ts = viv_zeros1(viv_i_num(rows, 28));
    declined = viv_zeros1(viv_i_num(rows, 28));
    chargeback = viv_zeros1(viv_i_num(rows, 28));
    {
        long long viv_n_i = rows;
        if (((((card.rank == 1) && ((long long)(card.dims[0]) >= viv_n_i)) && ((ts.rank == 1) && ((long long)(ts.dims[0]) >= viv_n_i))) && ((declined.rank == 1) && ((long long)(declined.dims[0]) >= viv_n_i))) && ((chargeback.rank == 1) && ((long long)(chargeback.dims[0]) >= viv_n_i))) {
            for (long long viv_k_i = 0; viv_k_i < viv_n_i; viv_k_i = (viv_k_i + 1)) {
                i = viv_k_i;
                viv_iset_m_at(card, i, viv_iget_l_at(xi, ((i * NI) + 0LL), 29), 1, 30);
                viv_iset_m_at(ts, i, viv_iget_l_at(xi, ((i * NI) + 1LL), 31), 1, 32);
                viv_iset_m_at(declined, i, viv_iget_l_at(xi, ((i * NI) + 3LL), 33), 1, 34);
                viv_iset_m_at(chargeback, i, viv_iget_l_at(xi, ((i * NI) + 4LL), 35), 1, 36);
            }
        } else {
            for (long long viv_k_i = 0; viv_k_i < viv_n_i; viv_k_i = (viv_k_i + 1)) {
                i = viv_k_i;
                viv_iset_l_at(card, i, viv_iget_l_at(xi, ((i * NI) + 0LL), 29), 30);
                viv_iset_l_at(ts, i, viv_iget_l_at(xi, ((i * NI) + 1LL), 31), 32);
                viv_iset_l_at(declined, i, viv_iget_l_at(xi, ((i * NI) + 3LL), 33), 34);
                viv_iset_l_at(chargeback, i, viv_iget_l_at(xi, ((i * NI) + 4LL), 35), 36);
            }
        }
    }
    f0 = card_features(card, ts, declined, chargeback);
    out = viv_zeros2(viv_i_num(rows, 28), (double)(NOUT));
    {
        long long viv_n_i = rows;
        {
            size_t viv_lmark = viv_arena_top;
            for (long long viv_k_i = 0; viv_k_i < viv_n_i; viv_k_i = (viv_k_i + 1)) {
                i = viv_k_i;
                VIV_POLL();
                if (viv_arena_top > viv_lmark) {
                    viv_arena_release_kept(viv_lmark);
                } else {
                    VIV_POLL();
                }
                r_card = viv_iget_l_at(xi, ((i * NI) + 0LL), 29);
                r_ts = viv_iget_l_at(xi, ((i * NI) + 1LL), 31);
                r_amount = viv_lget2_at(xf, i, 0LL, 0, 37);
                if (r_amount != r_amount) {
                    viv_s_stop(viv_s_catn(3, VIV_LIT(8, "row ", 4, 4), viv_i_str(i, 0.0), VIV_LIT(9, ": column amount is NaN, which this pipeline refuses (scikit-learn raises ValueError)", 84, 84), 38), 39);
                }
                if (fabs(r_amount) == HUGE_VAL) {
                    viv_s_stop(viv_s_catn(3, VIV_LIT(8, "row ", 4, 4), viv_i_str(i, 0.0), VIV_LIT(10, ": column amount is infinite, which this pipeline refuses (scikit-learn raises ValueError)", 89, 89), 40), 41);
                }
                r_merchant = viv_iget_l_at(xi, ((i * NI) + 2LL), 42);
                r_declined = viv_iget_l_at(xi, ((i * NI) + 3LL), 33);
                r_chargeback = viv_iget_l_at(xi, ((i * NI) + 4LL), 35);
                r_n_prev_1h = viv_iget_l_at(f0.n_prev_1h, i, 43);
                r_status = viv_iget_l_at(f0.status, i, 44);
                v1 = ((r_amount - 60.068621083333326) / 42.48865694585739);
                v2 = ((viv_i_float(r_n_prev_1h) - 0.292725) / 0.5177229706850953);
                if (r_merchant == 0LL) {
                    viv__if7 = 1.0;
                } else {
                    viv__if7 = 0.0;
                }
                v3 = viv__if7;
                if (r_merchant == 1LL) {
                    viv__if8 = 1.0;
                } else {
                    viv__if8 = 0.0;
                }
                v4 = viv__if8;
                if (r_merchant == 2LL) {
                    viv__if9 = 1.0;
                } else {
                    viv__if9 = 0.0;
                }
                v5 = viv__if9;
                if (r_merchant == 3LL) {
                    viv__if10 = 1.0;
                } else {
                    viv__if10 = 0.0;
                }
                v6 = viv__if10;
                if (r_merchant == 4LL) {
                    viv__if11 = 1.0;
                } else {
                    viv__if11 = 0.0;
                }
                v7 = viv__if11;
                if (r_merchant == 5LL) {
                    viv__if12 = 1.0;
                } else {
                    viv__if12 = 0.0;
                }
                v8 = viv__if12;
                code9 = (-1LL);
                if (viv_i_float(r_status) == 0.0) {
                    code9 = 0LL;
                }
                if (viv_i_float(r_status) == 1.0) {
                    code9 = 1LL;
                }
                if (viv_i_float(r_status) == 2.0) {
                    code9 = 2LL;
                }
                if (code9 == 0LL) {
                    viv__if13 = 1.0;
                } else {
                    viv__if13 = 0.0;
                }
                v10 = viv__if13;
                if (code9 == 1LL) {
                    viv__if14 = 1.0;
                } else {
                    viv__if14 = 0.0;
                }
                v11 = viv__if14;
                if (code9 == 2LL) {
                    viv__if15 = 1.0;
                } else {
                    viv__if15 = 0.0;
                }
                v12 = viv__if15;
                z0 = v1;
                z1 = v2;
                z2 = v3;
                z3 = v4;
                z4 = v5;
                z5 = v6;
                z6 = v7;
                z7 = v8;
                z8 = v10;
                z9 = v11;
                z10 = v12;
                raw13 = viv_lget1_at(INIT_RAW, 0LL, 0, 45);
                raw13 = fma(LEARNING_RATE, _tree0(z0, z1, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree1(z0, z1, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree2(z0, z1, z7, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree3(z0, z1, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree4(z0, z1, z7, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree5(z0, z1, z7, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree6(z0, z1, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree7(z0, z1, z7, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree8(z0, z1, z5, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree9(z0, z1, z7, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree10(z0, z1, z7, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree11(z0, z1, z7, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree12(z0, z1, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree13(z0, z7, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree14(z0, z1, z5, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree15(z0, z1, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree16(z0, z7, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree17(z0, z1, z7, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree18(z0, z1, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree19(z0, z1, z7, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree20(z0, z1, z6, z7, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree21(z0, z1, z7, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree22(z0, z1, z7, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree23(z0, z1, z5, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree24(z0, z1, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree25(z0, z1, z3, z7, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree26(z0, z1, z8, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree27(z0, z1, z2, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree28(z0, z1, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree29(z0, z1, z5, z7, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree30(z0, z7, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree31(z0, z1, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree32(z0, z1, z7, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree33(z0, z1, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree34(z0, z1, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree35(z0, z1, z9), raw13);
                raw13 = fma(LEARNING_RATE, _tree36(z0, z5, z7, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree37(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree38(z0, z1, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree39(z0, z4, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree40(z0, z1, z4), raw13);
                raw13 = fma(LEARNING_RATE, _tree41(z0, z4, z7, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree42(z0, z1, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree43(z0, z1, z4, z9), raw13);
                raw13 = fma(LEARNING_RATE, _tree44(z0, z1, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree45(z0, z1, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree46(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree47(z0, z1, z2, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree48(z0, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree49(z0, z4, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree50(z0, z1, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree51(z0, z1, z4, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree52(z0, z1, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree53(z0, z1, z9), raw13);
                raw13 = fma(LEARNING_RATE, _tree54(z0, z7, z9), raw13);
                raw13 = fma(LEARNING_RATE, _tree55(z0, z4), raw13);
                raw13 = fma(LEARNING_RATE, _tree56(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree57(z0, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree58(z0, z4, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree59(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree60(z0, z1, z7, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree61(z0, z1, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree62(z0, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree63(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree64(z0, z1, z5), raw13);
                raw13 = fma(LEARNING_RATE, _tree65(z0, z5, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree66(z0, z1, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree67(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree68(z0, z3), raw13);
                raw13 = fma(LEARNING_RATE, _tree69(z0, z1, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree70(z0, z7, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree71(z0, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree72(z0, z1, z3, z4, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree73(z0, z4, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree74(z0, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree75(z0, z3), raw13);
                raw13 = fma(LEARNING_RATE, _tree76(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree77(z0, z4, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree78(z0, z2, z4), raw13);
                raw13 = fma(LEARNING_RATE, _tree79(z0, z1, z4, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree80(z0, z4), raw13);
                raw13 = fma(LEARNING_RATE, _tree81(z0, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree82(z0, z1, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree83(z0, z1, z7, z9), raw13);
                raw13 = fma(LEARNING_RATE, _tree84(z0, z9), raw13);
                raw13 = fma(LEARNING_RATE, _tree85(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree86(z0, z1, z6), raw13);
                raw13 = fma(LEARNING_RATE, _tree87(z0, z3), raw13);
                raw13 = fma(LEARNING_RATE, _tree88(z0, z4, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree89(z0, z1, z2, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree90(z0), raw13);
                raw13 = fma(LEARNING_RATE, _tree91(z0, z4, z6, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree92(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree93(z0, z2), raw13);
                raw13 = fma(LEARNING_RATE, _tree94(z0, z1, z4, z9), raw13);
                raw13 = fma(LEARNING_RATE, _tree95(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree96(z0, z7, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree97(z0, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree98(z0, z1, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree99(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree100(z0, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree101(z0, z4, z5, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree102(z0, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree103(z0), raw13);
                raw13 = fma(LEARNING_RATE, _tree104(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree105(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree106(z0, z5), raw13);
                raw13 = fma(LEARNING_RATE, _tree107(z0, z1, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree108(z0, z1, z3), raw13);
                raw13 = fma(LEARNING_RATE, _tree109(z0, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree110(z0, z1, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree111(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree112(z0, z5), raw13);
                raw13 = fma(LEARNING_RATE, _tree113(z0, z6, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree114(z0, z1, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree115(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree116(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree117(z0, z1, z7, z9), raw13);
                raw13 = fma(LEARNING_RATE, _tree118(z0, z1, z5, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree119(z0, z2, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree120(z0), raw13);
                raw13 = fma(LEARNING_RATE, _tree121(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree122(z0, z1, z5, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree123(z0, z1, z6, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree124(z0, z2, z3), raw13);
                raw13 = fma(LEARNING_RATE, _tree125(z0), raw13);
                raw13 = fma(LEARNING_RATE, _tree126(z0, z2, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree127(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree128(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree129(z0, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree130(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree131(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree132(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree133(z0, z5, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree134(z0, z1, z2), raw13);
                raw13 = fma(LEARNING_RATE, _tree135(z0, z1, z4, z9), raw13);
                raw13 = fma(LEARNING_RATE, _tree136(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree137(z0, z4, z5), raw13);
                raw13 = fma(LEARNING_RATE, _tree138(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree139(z0, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree140(z0, z8), raw13);
                raw13 = fma(LEARNING_RATE, _tree141(z0, z5, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree142(z0, z4), raw13);
                raw13 = fma(LEARNING_RATE, _tree143(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree144(z0, z7), raw13);
                raw13 = fma(LEARNING_RATE, _tree145(z0, z3), raw13);
                raw13 = fma(LEARNING_RATE, _tree146(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree147(z0, z1), raw13);
                raw13 = fma(LEARNING_RATE, _tree148(z0, z2, z5, z10), raw13);
                raw13 = fma(LEARNING_RATE, _tree149(z0, z7), raw13);
                p14 = (1.0 / (1.0 + exp(-(raw13))));
                p15 = (1.0 - p14);
                cls16 = 0LL;
                if (raw13 >= 0.0) {
                    cls16 = 1LL;
                }
                viv_lset2_at(out, i, 0LL, p15, 0, 46);
                viv_lset2_at(out, i, 1LL, p14, 0, 47);
                viv_lset2_at(out, i, 2LL, viv_i_float(cls16), 0, 48);
            }
            if (viv_arena_top > viv_lmark) {
                viv_arena_release_kept(viv_lmark);
            } else {
                VIV_POLL();
            }
        }
    }
    {
        viv_arr viv_ret = out;
        viv_arena_release(viv_mark, viv_ret.data);
        return viv_ret;
    }
}

int main(void) {
    NS_PER_S = 1000000000LL;
    HOUR = (3600LL * NS_PER_S);
    DAY = (24LL * HOUR);
    CLEAN = (7LL * DAY);
    WATCH_DECLINES = 3LL;
    OK = 0LL;
    WATCH = 1LL;
    LOCKED = 2LL;
    NF = 1LL;
    NI = 5LL;
    NZ = 11LL;
    NOUT = 3LL;
    LEARNING_RATE = 0.1;
    INIT_RAW = viv_vec(1, -(1.678079064411613));
    return 0;
}

/* ---- the exported entry points (`viv export`) ---- */
#undef main
#if defined(__GNUC__)
#define CHURN_EXPORT __attribute__((visibility("default")))
#else
#define CHURN_EXPORT
#endif
static int vivk_busy = 0;
static churn_line_fn vivk_on_line = 0;
static void* vivk_ctx = 0;

/* The program's state after init: restored before each call, so a call
   depends only on its arguments. Kept across calls instead: viv export's
   KEEP_ACROSS_RUNS (lang/viv/src/export.rs). */
static int vivk_loaded = 0;
static unsigned char vivk_s0[sizeof(viv_kept_n)];
static unsigned char vivk_s1[sizeof(viv_en_pass)];
static unsigned char vivk_s2[sizeof(CLEAN)];
static unsigned char vivk_s3[sizeof(DAY)];
static unsigned char vivk_s4[sizeof(HOUR)];
static unsigned char vivk_s5[sizeof(INIT_RAW)];
static unsigned char vivk_s6[sizeof(LEARNING_RATE)];
static unsigned char vivk_s7[sizeof(LOCKED)];
static unsigned char vivk_s8[sizeof(NF)];
static unsigned char vivk_s9[sizeof(NI)];
static unsigned char vivk_s10[sizeof(NOUT)];
static unsigned char vivk_s11[sizeof(NS_PER_S)];
static unsigned char vivk_s12[sizeof(NZ)];
static unsigned char vivk_s13[sizeof(OK)];
static unsigned char vivk_s14[sizeof(WATCH)];
static unsigned char vivk_s15[sizeof(WATCH_DECLINES)];
static void vivk_restore(void) {
    if (!vivk_loaded) {
        memcpy(vivk_s0, (const void*)&viv_kept_n, sizeof(viv_kept_n));
        memcpy(vivk_s1, (const void*)&viv_en_pass, sizeof(viv_en_pass));
        memcpy(vivk_s2, (const void*)&CLEAN, sizeof(CLEAN));
        memcpy(vivk_s3, (const void*)&DAY, sizeof(DAY));
        memcpy(vivk_s4, (const void*)&HOUR, sizeof(HOUR));
        memcpy(vivk_s5, (const void*)&INIT_RAW, sizeof(INIT_RAW));
        memcpy(vivk_s6, (const void*)&LEARNING_RATE, sizeof(LEARNING_RATE));
        memcpy(vivk_s7, (const void*)&LOCKED, sizeof(LOCKED));
        memcpy(vivk_s8, (const void*)&NF, sizeof(NF));
        memcpy(vivk_s9, (const void*)&NI, sizeof(NI));
        memcpy(vivk_s10, (const void*)&NOUT, sizeof(NOUT));
        memcpy(vivk_s11, (const void*)&NS_PER_S, sizeof(NS_PER_S));
        memcpy(vivk_s12, (const void*)&NZ, sizeof(NZ));
        memcpy(vivk_s13, (const void*)&OK, sizeof(OK));
        memcpy(vivk_s14, (const void*)&WATCH, sizeof(WATCH));
        memcpy(vivk_s15, (const void*)&WATCH_DECLINES, sizeof(WATCH_DECLINES));
        vivk_loaded = 1;
        return;
    }
    memcpy((void*)&viv_kept_n, vivk_s0, sizeof(viv_kept_n));
    memcpy((void*)&viv_en_pass, vivk_s1, sizeof(viv_en_pass));
    memcpy((void*)&CLEAN, vivk_s2, sizeof(CLEAN));
    memcpy((void*)&DAY, vivk_s3, sizeof(DAY));
    memcpy((void*)&HOUR, vivk_s4, sizeof(HOUR));
    memcpy((void*)&INIT_RAW, vivk_s5, sizeof(INIT_RAW));
    memcpy((void*)&LEARNING_RATE, vivk_s6, sizeof(LEARNING_RATE));
    memcpy((void*)&LOCKED, vivk_s7, sizeof(LOCKED));
    memcpy((void*)&NF, vivk_s8, sizeof(NF));
    memcpy((void*)&NI, vivk_s9, sizeof(NI));
    memcpy((void*)&NOUT, vivk_s10, sizeof(NOUT));
    memcpy((void*)&NS_PER_S, vivk_s11, sizeof(NS_PER_S));
    memcpy((void*)&NZ, vivk_s12, sizeof(NZ));
    memcpy((void*)&OK, vivk_s13, sizeof(OK));
    memcpy((void*)&WATCH, vivk_s14, sizeof(WATCH));
    memcpy((void*)&WATCH_DECLINES, vivk_s15, sizeof(WATCH_DECLINES));
}

static int vivk_bad(char* msg, size_t cap, const char* fmt, ...) {
    if (msg && cap) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(msg, cap, fmt, ap);
        va_end(ap);
    }
    vivk_busy = 0;
    return CHURN_BAD_PARAM;
}
VIV_UNUSED static int vivk_arg_ok(int64_t rank, const int64_t* dims, const void* data, long long dst[8], int want, char* why, size_t cap) {
    long long n = 1;
    if (rank < 1 || rank > 8 || (want && rank != want)) { snprintf(why, cap, "has rank %lld%s", (long long)rank, want ? "; it must have the manifest's rank" : ""); return 0; }
    if (!dims) { snprintf(why, cap, "has no dims"); return 0; }
    for (int i = 0; i < rank; i++) {
        if (dims[i] < 0 || dims[i] > 2147483647LL) { snprintf(why, cap, "has dimension %lld", (long long)dims[i]); return 0; }
        n *= dims[i];
        if (n > 2147483647LL) { snprintf(why, cap, "has more than 2147483647 elements"); return 0; }
        dst[i] = dims[i];
    }
    if (n > 0 && !data) { snprintf(why, cap, "has no data"); return 0; }
    return 1;
}
/* an array argument, copied into an array of the call's arena */
static viv_arr vivk_in_arr(const churn_arg* g) {
    int dims[8];
    for (int i = 0; i < g->rank; i++) dims[i] = (int)g->dims[i];
    viv_arr r = viv_arr_new((int)g->rank, dims);
    if (r.size) memcpy(r.data, g->data, (size_t)r.size * sizeof(double));
    return r;
}
/* an array result, copied out of the arena into a block the caller frees */
static void vivk_out_arr(churn_array* d, viv_arr s) {
    d->rank = s.rank;
    for (int i = 0; i < s.rank && i < 8; i++) d->dims[i] = s.dims[i];
    d->data = (double*)malloc((size_t)(s.size > 0 ? s.size : 1) * sizeof(double));
    if (!d->data) {
        viv_errf("error: out of memory copying a result of %d elements\n", s.size);
        viv_stop(VIV_E_MEMORY);
    }
    if (s.size > 0) memcpy(d->data, s.data, (size_t)s.size * sizeof(double));
}

static int vivk_inited = 0;
static int vivk_init_status = 0;
static int vivk_init_kind = 0;
static char vivk_init_msg[1024];
static viv_embed_test_row* vivk_tests = 0;
static int vivk_ntests = 0;
static int vivk_do_init(churn_line_fn on_line, void* ctx, char* msg, size_t cap) {
    if (!vivk_inited) {
        vivk_inited = 1;
        viv_embed_stop_flag = 0;
        int code = viv_embed_run(viv_program_main, on_line ? on_line : vivk_on_line, on_line ? ctx : vivk_ctx, vivk_init_msg, sizeof vivk_init_msg);
        vivk_tests = viv_embed_take_tests(&vivk_ntests);
        vivk_init_kind = code;
        vivk_init_status = code == 0 ? CHURN_OK : code == VIV_E_INTERRUPT ? CHURN_INTERRUPTED : CHURN_ERROR;
        /* the snapshot each call starts from */
        if (code == 0) vivk_restore();
    }
    if (msg && cap) snprintf(msg, cap, "%s", vivk_init_msg);
    return vivk_init_status;
}

CHURN_EXPORT int churn_init(churn_line_fn on_line, void* ctx, char* msg, size_t cap) {
    if (msg && cap) msg[0] = 0;
    if (vivk_busy) return CHURN_BUSY;
    vivk_busy = 1;
    int s = vivk_do_init(on_line, ctx, msg, cap);
    vivk_busy = 0;
    return s;
}
CHURN_EXPORT void churn_set_output(churn_line_fn on_line, void* ctx) { vivk_on_line = on_line; vivk_ctx = ctx; }
CHURN_EXPORT int churn_ntests(void) { return vivk_ntests; }
CHURN_EXPORT const char* churn_test_name(int i) { return i >= 0 && i < vivk_ntests ? vivk_tests[i].name : 0; }
CHURN_EXPORT int churn_test_passed(int i) { return i >= 0 && i < vivk_ntests ? vivk_tests[i].passed : 0; }
CHURN_EXPORT const char* churn_test_detail(int i) { return i >= 0 && i < vivk_ntests ? vivk_tests[i].detail : 0; }
CHURN_EXPORT int churn_abi(void) { return CHURN_ABI; }
CHURN_EXPORT const char* churn_version(void) { return "viv 0.0.1; program 63dec3d40467e2a7"; }
CHURN_EXPORT void churn_interrupt(void) { viv_embed_stop_flag = 1; }

/* score(xf, xi, rows) -> result */
static churn_score_args vivk_a_score;
static churn_score_results* vivk_r_score;
static int vivk_body_score(void) {
    const churn_score_args* a = &vivk_a_score;
    churn_score_results* r = vivk_r_score;
    (void)a;
    viv_arr x0 = vivk_in_arr(&a->xf);
    viv_arr x1 = vivk_in_arr(&a->xi);
    long long x2 = (long long)a->rows;
    viv_arr res = score(x0, x1, x2);
    vivk_out_arr(&r->result, res);
    return 0;
}
CHURN_EXPORT void churn_score_free(churn_score_results* r) {
    if (!r) return;
    free(r->result.data);
    r->result.data = 0;
}
CHURN_EXPORT int churn_score(const churn_score_args* a, churn_score_results* r, char* msg, size_t cap) {
    if (msg && cap) msg[0] = 0;
    if (!a || !r) return CHURN_BAD_PARAM;
    memset(r, 0, sizeof *r);
    if (vivk_busy) return CHURN_BUSY;
    vivk_busy = 1;
    if (!vivk_inited || vivk_init_status) {
        int s = vivk_do_init(0, 0, msg, cap);
        if (s) {
            r->error_kind = vivk_init_kind;
            vivk_busy = 0;
            return s;
        }
    }
    {
        {
            char why[96];
            long long dims[8];
            if (!vivk_arg_ok(a->xf.rank, a->xf.dims, (const void*)a->xf.data, dims, 0, why, sizeof why)) return vivk_bad(msg, cap, "xf %s", why);
        }
        {
            char why[96];
            long long dims[8];
            if (!vivk_arg_ok(a->xi.rank, a->xi.dims, (const void*)a->xi.data, dims, 0, why, sizeof why)) return vivk_bad(msg, cap, "xi %s", why);
        }
    }
    vivk_a_score = *a;
    vivk_r_score = r;
    vivk_restore();
    size_t mark = viv_arena_top;
    viv_embed_stop_flag = 0;
    int code = viv_embed_run(vivk_body_score, vivk_on_line, vivk_ctx, msg, cap);
    {
        int nt = 0;
        viv_embed_test_row* t = viv_embed_take_tests(&nt);
        viv_embed_free_tests(t, nt);
    }
    viv_arena_release(mark, 0);
    vivk_busy = 0;
    if (code == 0) return CHURN_OK;
    churn_score_free(r);
    r->error_kind = code;
    return code == VIV_E_INTERRUPT ? CHURN_INTERRUPTED : CHURN_ERROR;
}

/* ======================================================================
   churn: the model's schema, its row-major C entries, a CSV command line
   and a self-test. Written by `vipyr compile` after `viv export`; C99 and
   libc only. Build: cc -O2 -ffp-contract=off churn.c -o churn
   ====================================================================== */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHURN_NCOLS 6
#define CHURN_NF 1
#define CHURN_NI 5
#define CHURN_NOUT 3
#define CHURN_NCLASSES 2

/* kind: 0 float (a double), 1 integer (int64), 2 category (a string, read as its code) */
static const char* const churn_col_name[CHURN_NCOLS] = { "card", "ts", "amount", "merchant", "declined", "chargeback" };
static const int churn_col_kind[CHURN_NCOLS] = { 1, 1, 0, 2, 1, 1 };
static const int churn_col_slot[CHURN_NCOLS] = { 0, 1, 0, 2, 3, 4 };
static const int churn_col_ncat[CHURN_NCOLS] = { 0, 0, 0, 6, 0, 0 };
static const char* const churn_cats[] = { "electronics", "fuel", "grocery", "online", "restaurant", "travel" };
static const int churn_col_cat0[CHURN_NCOLS] = { 0, 0, 0, 0, 6, 6 };
static const char* const churn_classes[] = { "0", "1" };
static const char* const churn_schema = "{\"columns\": [{\"name\": \"card\", \"kind\": \"int\"}, {\"name\": \"ts\", \"kind\": \"int\"}, {\"name\": \"amount\", \"kind\": \"float\"}, {\"name\": \"merchant\", \"kind\": \"category\", \"categories\": [\"electronics\", \"fuel\", \"grocery\", \"online\", \"restaurant\", \"travel\"]}, {\"name\": \"declined\", \"kind\": \"int\"}, {\"name\": \"chargeback\", \"kind\": \"int\"}], \"classes\": [\"0\", \"1\"], \"outputs\": 3}";

const char* churn_schema_json(void) { return churn_schema; }

/* the code of category `text` in column `col` (its position in the encoder's
   categories), or -1 when the encoder did not see it in training */
int64_t churn_category_code(int col, const char* text) {
    int k;
    if (col < 0 || col >= CHURN_NCOLS || churn_col_kind[col] != 2) return -1;
    for (k = 0; k < churn_col_ncat[col]; k++)
        if (strcmp(churn_cats[churn_col_cat0[col] + k], text) == 0) return k;
    return -1;
}

static int churn_run(const double* xf, const int64_t* xi, int64_t rows, churn_score_results* r, char* msg, size_t cap) {
    churn_score_args a;
    int64_t dims_f[2], dims_i[1];
    static const double no_f[1] = {0.0};
    static const int64_t no_i[1] = {0};
    dims_f[0] = rows; dims_f[1] = CHURN_NF;
    dims_i[0] = rows * CHURN_NI;
    a.xf.data = xf ? xf : no_f; a.xf.rank = 2; a.xf.dims = dims_f;
    a.xi.data = (const double*)(const void*)(xi ? xi : no_i); a.xi.rank = 1; a.xi.dims = dims_i;
    a.rows = rows;
    return churn_score(&a, r, msg, cap);
}

/* rows predictions into out: the value for a regressor, the class's index
   (into churn_classes) for a classifier. 0 on success; otherwise msg says why. */
int churn_predict(const double* xf, const int64_t* xi, int64_t rows, double* out, char* msg, size_t cap) {
    churn_score_results r;
    int64_t i;
    int s = churn_run(xf, xi, rows, &r, msg, cap);
    if (s) return s;
    for (i = 0; i < rows; i++) out[i] = r.result.data[i * CHURN_NOUT + (CHURN_NOUT - 1)];
    churn_score_free(&r);
    return 0;
}

/* rows x NCLASSES probabilities into out (a classifier only) */
int churn_predict_proba(const double* xf, const int64_t* xi, int64_t rows, double* out, char* msg, size_t cap) {
    churn_score_results r;
    int64_t i, k;
    int s;
    if (CHURN_NCLASSES == 0) {
        if (msg && cap) snprintf(msg, cap, "predict_proba: the model is a regressor");
        return 3;
    }
    s = churn_run(xf, xi, rows, &r, msg, cap);
    if (s) return s;
    for (i = 0; i < rows; i++)
        for (k = 0; k < CHURN_NCLASSES; k++) out[i * CHURN_NCLASSES + k] = r.result.data[i * CHURN_NOUT + k];
    churn_score_free(&r);
    return 0;
}

/* every output of the model: rows x NOUT (the probabilities, then the
   class's index, for a classifier) */
int churn_predict_all(const double* xf, const int64_t* xi, int64_t rows, double* out, char* msg, size_t cap) {
    churn_score_results r;
    int s = churn_run(xf, xi, rows, &r, msg, cap);
    if (s) return s;
    if (rows > 0) memcpy(out, r.result.data, (size_t)(rows * CHURN_NOUT) * sizeof(double));
    churn_score_free(&r);
    return 0;
}

#ifndef CHURN_NO_MAIN
/* ------------------------------------------------------------ CSV input */

typedef struct { char** cell; size_t n, cap; } churn_row;

/* split one line into cells: commas, double quotes with "" inside */
static int churn_split(char* line, churn_row* r) {
    char* s = line;
    r->n = 0;
    for (;;) {
        char* start;
        if (r->n == r->cap) {
            size_t nc = r->cap ? 2 * r->cap : 16;
            char** c = (char**)realloc(r->cell, nc * sizeof(char*));
            if (!c) return -1;
            r->cell = c; r->cap = nc;
        }
        if (*s == '"') {
            char* w;
            s++;
            start = w = s;
            for (;;) {
                if (*s == 0) return -2;
                if (*s == '"') {
                    if (s[1] == '"') { *w++ = '"'; s += 2; continue; }
                    s++;
                    break;
                }
                *w++ = *s++;
            }
            *w = 0;
            r->cell[r->n++] = start;
            if (*s == ',') { s++; continue; }
            if (*s == 0) return 0;
            return -2;
        }
        start = s;
        while (*s && *s != ',') s++;
        r->cell[r->n++] = start;
        if (*s == ',') { *s++ = 0; continue; }
        return 0;
    }
}

static int churn_parse_float(const char* c, double* v) {
    char* end;
    while (*c == ' ') c++;
    if (*c == 0) { *v = NAN; return 1; }            /* empty: missing, as pandas reads it */
    errno = 0;
    *v = strtod(c, &end);
    while (*end == ' ') end++;
    return end != c && *end == 0;
}

static int churn_parse_int(const char* c, int64_t* v) {
    char* end;
    while (*c == ' ') c++;
    if (strcmp(c, "True") == 0 || strcmp(c, "true") == 0) { *v = 1; return 1; }
    if (strcmp(c, "False") == 0 || strcmp(c, "false") == 0) { *v = 0; return 1; }
    errno = 0;
    *v = (int64_t)strtoll(c, &end, 10);
    while (*end == ' ') end++;
    return end != c && *end == 0 && errno == 0;
}

/* A table read from lines: xf, xi and the row count, or an exit status. */
typedef struct { double* xf; int64_t* xi; int64_t rows; } churn_table;

static int churn_read(char** lines, size_t nlines, int ignore_extra, churn_table* t, const char* src) {
    churn_row r = {0, 0, 0};
    int where[CHURN_NCOLS + 1];
    int* pos = 0;
    size_t ncell, i;
    int c, status = 0;
    size_t cap = 0;
    t->xf = 0; t->xi = 0; t->rows = 0;
    if (nlines == 0) { fprintf(stderr, "churn: %s: no header line; the columns are: %s\n", src, churn_schema); return 2; }
    if (churn_split(lines[0], &r) != 0) { fprintf(stderr, "churn: %s: the header line is not CSV\n", src); return 2; }
    ncell = r.n;
    pos = (int*)malloc((ncell ? ncell : 1) * sizeof(int));
    for (c = 0; c < CHURN_NCOLS; c++) where[c] = -1;
    for (i = 0; i < ncell; i++) {
        pos[i] = -1;
        for (c = 0; c < CHURN_NCOLS; c++)
            if (strcmp(r.cell[i], churn_col_name[c]) == 0) {
                if (where[c] >= 0) { fprintf(stderr, "churn: %s: column %s appears twice\n", src, churn_col_name[c]); status = 2; }
                where[c] = (int)i; pos[i] = c;
            }
        if (pos[i] < 0 && !ignore_extra) {
            fprintf(stderr, "churn: %s: column '%s' is not one the model reads (pass --ignore-extra to skip it)\n", src, r.cell[i]);
            status = 2;
        }
    }
    for (c = 0; c < CHURN_NCOLS; c++)
        if (where[c] < 0) {
            fprintf(stderr, "churn: %s: missing column '%s' (%s)\n", src, churn_col_name[c],
                    churn_col_kind[c] == 0 ? "float" : churn_col_kind[c] == 1 ? "integer" : "category");
            status = 2;
        }
    if (status) { free(pos); free(r.cell); return status; }
    for (i = 1; i < nlines; i++) {
        size_t k;
        if (lines[i][0] == 0) continue;
        if (churn_split(lines[i], &r) != 0 || r.n != ncell) {
            fprintf(stderr, "churn: %s: line %lu has %lu cells; the header has %lu\n", src, (unsigned long)(i + 1),
                    (unsigned long)r.n, (unsigned long)ncell);
            status = 3; break;
        }
        if ((size_t)t->rows == cap) {
            size_t nc = cap ? 2 * cap : 1024;
            double* f = (double*)realloc(t->xf, (nc * CHURN_NF + 1) * sizeof(double));
            int64_t* g = (int64_t*)realloc(t->xi, (nc * CHURN_NI + 1) * sizeof(int64_t));
            if (f) t->xf = f;
            if (g) t->xi = g;
            if (!f || !g) { fprintf(stderr, "churn: out of memory\n"); status = 3; break; }
            cap = nc;
        }
        for (k = 0; k < ncell; k++) {
            int col = pos[k];
            const char* cell = r.cell[k];
            if (col < 0) continue;
            if (churn_col_kind[col] == 0) {
                if (!churn_parse_float(cell, &t->xf[t->rows * CHURN_NF + churn_col_slot[col]])) {
                    fprintf(stderr, "churn: %s: line %lu, column %s: '%s' is not a number\n", src, (unsigned long)(i + 1), churn_col_name[col], cell);
                    status = 3;
                }
            } else if (churn_col_kind[col] == 1) {
                if (!churn_parse_int(cell, &t->xi[t->rows * CHURN_NI + churn_col_slot[col]])) {
                    fprintf(stderr, "churn: %s: line %lu, column %s: '%s' is not an integer\n", src, (unsigned long)(i + 1), churn_col_name[col], cell);
                    status = 3;
                }
            } else {
                t->xi[t->rows * CHURN_NI + churn_col_slot[col]] = churn_category_code(col, cell);
            }
        }
        if (status) break;
        t->rows++;
    }
    free(pos);
    free(r.cell);
    return status;
}

/* the lines of a stream, in one buffer */
static char** churn_lines(FILE* f, size_t* n, char** buf) {
    size_t len = 0, cap = 1 << 16, nl = 0, lcap = 1024, i;
    char* b = (char*)malloc(cap);
    char** lines;
    size_t got;
    if (!b) return 0;
    while ((got = fread(b + len, 1, cap - len - 1, f)) > 0) {
        len += got;
        if (len + 1 == cap) {
            char* nb = (char*)realloc(b, cap * 2);
            if (!nb) { free(b); return 0; }
            b = nb; cap *= 2;
        }
    }
    b[len] = 0;
    lines = (char**)malloc(lcap * sizeof(char*));
    if (!lines) { free(b); return 0; }
    if (len > 0) lines[nl++] = b;
    for (i = 0; i < len; i++)
        if (b[i] == '\n' || b[i] == '\r') {
            b[i] = 0;
            if (b[i + 1] == '\n' && i + 1 < len) { i++; b[i] = 0; }
            if (i + 1 < len) {
                if (nl == lcap) {
                    char** nl2 = (char**)realloc(lines, 2 * lcap * sizeof(char*));
                    if (!nl2) { free(lines); free(b); return 0; }
                    lines = nl2; lcap *= 2;
                }
                lines[nl++] = b + i + 1;
            }
        }
    *n = nl;
    *buf = b;
    return lines;
}

/* ------------------------------------------------------------ self-test */

#define CHURN_ST_ROWS 300
static const char* const churn_st_csv[] = {
    "card,ts,amount,merchant,declined,chargeback",
    "20000000,1767270348000000000,0x1.9ce147ae147aep+5,online,0,0",
    "20000000,1767423666000000000,0x1.b170a3d70a3d7p+4,grocery,0,0",
    "20000000,1767423934000000000,0x1.ef851eb851eb8p+5,restaurant,0,0",
    "20000000,1767481178000000000,0x1.9d8f5c28f5c29p+6,online,0,0",
    "20000000,1767500291000000000,0x1.5428f5c28f5c3p+3,online,0,0",
    "20000000,1767678265000000000,0x1.16e147ae147aep+4,fuel,0,0",
    "20000000,1767684225000000000,0x1.92e6666666666p+7,online,0,0",
    "20000000,1767710083000000000,0x1.8eb851eb851ecp+4,online,0,0",
    "20000000,1767725881000000000,0x1.bc7ae147ae148p+3,fuel,0,0",
    "20000000,1767732637000000000,0x1.675c28f5c28f6p+5,online,0,0",
    "20000000,1767732920000000000,0x1.47eb851eb851fp+5,grocery,1,0",
    "20000000,1767745946000000000,0x1.0c6147ae147aep+7,electronics,0,0",
    "20000000,1767816796000000000,0x1.5828f5c28f5c3p+4,grocery,0,0",
    "20000000,1767951493000000000,0x1.39eb851eb851fp+3,grocery,0,0",
    "20000000,1767952162000000000,0x1.70d70a3d70a3dp+6,grocery,1,0",
    "20000000,1768037687000000000,0x1.55ae147ae147bp+5,travel,0,0",
    "20000000,1768037721000000000,0x1.07c28f5c28f5cp+5,grocery,1,0",
    "20000000,1768041354000000000,0x1.cbd70a3d70a3dp+5,grocery,1,0",
    "20000000,1768041520000000000,0x1.8270a3d70a3d7p+6,restaurant,1,0",
    "20000000,1768057849000000000,0x1.a600000000000p+5,online,0,0",
    "20000000,1768076788000000000,0x1.028f5c28f5c29p+4,online,0,0",
    "20000000,1768076940000000000,0x1.9a5c28f5c28f6p+6,grocery,1,0",
    "20000000,1768192426000000000,0x1.58d70a3d70a3dp+6,grocery,0,0",
    "20000000,1768225984000000000,0x1.547ae147ae148p+2,grocery,0,0",
    "20000000,1768252115000000000,0x1.568f5c28f5c29p+4,fuel,0,0",
    "20000000,1768252887000000000,0x1.ef0a3d70a3d71p+5,travel,0,0",
    "20000000,1768319471000000000,0x1.4cf5c28f5c28fp+5,grocery,0,0",
    "20000000,1768442909000000000,0x1.464cccccccccdp+7,restaurant,1,0",
    "20000000,1768453866000000000,0x1.01851eb851eb8p+6,restaurant,0,0",
    "20000000,1768526413000000000,0x1.0cccccccccccdp+6,grocery,0,0",
    "20000000,1768555313000000000,0x1.c47ae147ae148p+3,grocery,0,0",
    "20000000,1768556256000000000,0x1.11c28f5c28f5cp+5,online,0,0",
    "20000000,1768631896000000000,0x1.2f5c28f5c28f6p+5,fuel,0,0",
    "20000000,1768641218000000000,0x1.59d70a3d70a3dp+5,online,0,0",
    "20000000,1768641555000000000,0x1.42b851eb851ecp+6,online,1,0",
    "20000000,1768647058000000000,0x1.a83d70a3d70a4p+6,restaurant,0,0",
    "20000000,1768690501000000000,0x1.fc51eb851eb85p+4,restaurant,0,0",
    "20000000,1768693561000000000,0x1.87d70a3d70a3dp+5,restaurant,1,0",
    "20000000,1768773916000000000,0x1.b028f5c28f5c3p+5,online,0,0",
    "20000000,1768900700000000000,0x1.13851eb851eb8p+3,restaurant,0,0",
    "20000000,1768905452000000000,0x1.9d70a3d70a3d7p+6,grocery,0,0",
    "20000000,1768938001000000000,0x1.e1eb851eb851fp+4,grocery,0,0",
    "20000000,1768974994000000000,0x1.587ae147ae148p+5,restaurant,0,0",
    "20000000,1768975192000000000,0x1.4000000000000p+5,grocery,0,0",
    "20000000,1769065807000000000,0x1.473d70a3d70a4p+6,online,0,0",
    "20000000,1769134744000000000,0x1.919999999999ap+5,online,0,0",
    "20000000,1769134870000000000,0x1.10947ae147ae1p+7,travel,0,0",
    "20000000,1769387790000000000,0x1.c4f5c28f5c28fp+5,travel,0,0",
    "20000000,1769387876000000000,0x1.7dd70a3d70a3dp+5,grocery,1,0",
    "20000000,1769391450000000000,0x1.3147ae147ae14p+4,online,0,0",
    "20000000,1769485408000000000,0x1.0b33333333333p+5,restaurant,0,0",
    "20000000,1769573168000000000,0x1.19d70a3d70a3dp+5,online,0,0",
    "20000000,1769573647000000000,0x1.4147ae147ae14p+3,fuel,0,0",
    "20000000,1769634318000000000,0x1.0beb851eb851fp+7,grocery,0,0",
    "20000000,1769734070000000000,0x1.5b33333333333p+5,online,0,0",
    "20000000,1769776676000000000,0x1.44d1eb851eb85p+7,restaurant,0,0",
    "20000001,1767254449000000000,0x1.57ae147ae147bp+2,restaurant,0,0",
    "20000001,1767254943000000000,0x1.89dc28f5c28f6p+7,online,0,0",
    "20000001,1767363781000000000,0x1.ae66666666666p+4,online,0,0",
    "20000001,1767364742000000000,0x1.fb851eb851eb8p+5,online,1,0",
    "20000001,1767380840000000000,0x1.2e5c28f5c28f6p+7,fuel,0,0",
    "20000001,1767387522000000000,0x1.107ae147ae148p+6,grocery,0,0",
    "20000001,1767482474000000000,0x1.5d33333333333p+5,fuel,0,0",
    "20000001,1767482728000000000,0x1.f370a3d70a3d7p+6,fuel,0,0",
    "20000001,1767530870000000000,0x1.8a8f5c28f5c29p+5,online,0,0",
    "20000001,1767531597000000000,0x1.351eb851eb852p+4,fuel,0,0",
    "20000001,1767578545000000000,0x1.2133333333333p+5,grocery,0,0",
    "20000001,1767721693000000000,0x1.155c28f5c28f6p+5,grocery,0,0",
    "20000001,1767745645000000000,0x1.8ef5c28f5c28fp+5,online,0,0",
    "20000001,1767832061000000000,0x1.7fae147ae147bp+3,fuel,0,0",
    "20000001,1767950533000000000,0x1.935c28f5c28f6p+5,fuel,0,0",
    "20000001,1767991233000000000,0x1.185c28f5c28f6p+6,grocery,0,0",
    "20000001,1768037310000000000,0x1.29e147ae147aep+7,fuel,0,0",
    "20000001,1768147048000000000,0x1.8933333333333p+5,grocery,0,0",
    "20000001,1768166715000000000,0x1.4bae147ae147bp+5,travel,0,0",
    "20000001,1768172699000000000,0x1.60f5c28f5c28fp+3,grocery,0,0",
    "20000001,1768232930000000000,0x1.03c28f5c28f5cp+5,grocery,0,0",
    "20000001,1768299035000000000,0x1.f7eb851eb851fp+5,grocery,0,0",
    "20000001,1768360078000000000,0x1.c5ae147ae147bp+5,travel,1,0",
    "20000001,1768360853000000000,0x1.14851eb851eb8p+7,online,0,0",
    "20000001,1768390261000000000,0x1.bb0a3d70a3d71p+6,travel,0,0",
    "20000001,1768504288000000000,0x1.c23d70a3d70a4p+5,travel,0,0",
    "20000001,1768515667000000000,0x1.435c28f5c28f6p+5,online,0,0",
    "20000001,1768545402000000000,0x1.13ae147ae147bp+4,travel,0,0",
    "20000001,1768592271000000000,0x1.f70a3d70a3d71p+4,online,0,0",
    "20000001,1768628029000000000,0x1.0c8f5c28f5c29p+5,online,0,0",
    "20000001,1768662378000000000,0x1.de8f5c28f5c29p+5,restaurant,0,0",
    "20000001,1768778185000000000,0x1.e200000000000p+5,online,0,0",
    "20000001,1768778908000000000,0x1.60147ae147ae1p+5,grocery,0,0",
    "20000001,1768797533000000000,0x1.847ae147ae148p+6,grocery,0,0",
    "20000001,1768808560000000000,0x1.6570a3d70a3d7p+6,grocery,0,0",
    "20000001,1768809337000000000,0x1.468f5c28f5c29p+4,grocery,0,0",
    "20000001,1768811006000000000,0x1.9b33333333333p+5,fuel,0,0",
    "20000001,1768847928000000000,0x1.0e147ae147ae1p+2,grocery,0,0",
    "20000001,1768874871000000000,0x1.a828f5c28f5c3p+5,fuel,0,0",
    "20000001,1768894645000000000,0x1.c6ccccccccccdp+5,online,0,0",
    "20000001,1768904904000000000,0x1.f70a3d70a3d71p+5,grocery,0,0",
    "20000001,1768905044000000000,0x1.e1eb851eb851fp+5,online,1,0",
    "20000001,1768938565000000000,0x1.38ccccccccccdp+4,grocery,0,0",
    "20000001,1768939745000000000,0x1.d033333333333p+6,restaurant,0,0",
    "20000001,1768946486000000000,0x1.8a00000000000p+6,restaurant,0,0",
    "20000001,1769045903000000000,0x1.6733333333333p+4,online,0,0",
    "20000001,1769098677000000000,0x1.921eb851eb852p+6,online,0,0",
    "20000001,1769103506000000000,0x1.24851eb851eb8p+6,grocery,0,0",
    "20000001,1769231032000000000,0x1.43ae147ae147bp+4,grocery,0,0",
    "20000001,1769243731000000000,0x1.55eb851eb851fp+4,grocery,0,0",
    "20000001,1769248898000000000,0x1.2e147ae147ae1p+3,online,0,0",
    "20000001,1769350781000000000,0x1.6f0a3d70a3d71p+5,fuel,0,0",
    "20000001,1769351143000000000,0x1.50ccccccccccdp+4,fuel,0,0",
    "20000001,1769408560000000000,0x1.00f5c28f5c28fp+5,restaurant,0,0",
    "20000001,1769467296000000000,0x1.32147ae147ae1p+7,online,0,1",
    "20000001,1769513566000000000,0x1.7c8f5c28f5c29p+6,restaurant,0,0",
    "20000001,1769526475000000000,0x1.e451eb851eb85p+4,online,0,0",
    "20000001,1769538780000000000,0x1.d47ae147ae148p+2,online,0,0",
    "20000001,1769575198000000000,0x1.7ad70a3d70a3dp+6,grocery,0,0",
    "20000001,1769599284000000000,0x1.d59999999999ap+4,online,0,0",
    "20000001,1769748385000000000,0x1.ace6666666666p+7,electronics,0,0",
    "20000001,1769786887000000000,0x1.d5eb851eb851fp+4,restaurant,0,0",
    "20000002,1767357896000000000,0x1.5e851eb851eb8p+6,grocery,0,0",
    "20000002,1767415283000000000,0x1.eb70a3d70a3d7p+5,grocery,0,0",
    "20000002,1767551632000000000,0x1.bb5c28f5c28f6p+6,electronics,0,0",
    "20000002,1767552336000000000,0x1.1d66666666666p+6,fuel,0,0",
    "20000002,1767554354000000000,0x1.747ae147ae148p+3,restaurant,0,0",
    "20000002,1767703240000000000,0x1.c19999999999ap+3,grocery,0,0",
    "20000002,1767706562000000000,0x1.d4ccccccccccdp+3,online,1,0",
    "20000002,1767830670000000000,0x1.4947ae147ae14p+4,grocery,0,0",
    "20000002,1767893508000000000,0x1.e9851eb851eb8p+5,restaurant,0,0",
    "20000002,1767894457000000000,0x1.8ab851eb851ecp+5,grocery,0,0",
    "20000002,1767899618000000000,0x1.77eb851eb851fp+5,grocery,0,0",
    "20000002,1767900651000000000,0x1.2600000000000p+5,fuel,1,0",
    "20000002,1767916765000000000,0x1.34ccccccccccdp+5,grocery,0,0",
    "20000002,1767925397000000000,0x1.087ae147ae148p+6,travel,0,0",
    "20000002,1767935929000000000,0x1.16147ae147ae1p+5,grocery,0,0",
    "20000002,1768070753000000000,0x1.f70a3d70a3d71p+5,online,0,0",
    "20000002,1768097747000000000,0x1.af5c28f5c28f6p+5,grocery,0,0",
    "20000002,1768098757000000000,0x1.c833333333333p+6,electronics,0,0",
    "20000002,1768219091000000000,0x1.d70a3d70a3d71p+3,grocery,0,0",
    "20000002,1768370465000000000,0x1.3ea8f5c28f5c3p+7,grocery,0,0",
    "20000002,1768391569000000000,0x1.d19999999999ap+5,online,0,0",
    "20000002,1768409093000000000,0x1.607ae147ae148p+4,fuel,0,0",
    "20000002,1768455049000000000,0x1.1147ae147ae14p+5,grocery,0,0",
    "20000002,1768456233000000000,0x1.4e147ae147ae1p+4,online,0,0",
    "20000002,1768575008000000000,0x1.6c51eb851eb85p+5,grocery,0,0",
    "20000002,1768710405000000000,0x1.d9eb851eb851fp+4,electronics,0,0",
    "20000002,1768711559000000000,0x1.05ae147ae147bp+6,online,0,0",
    "20000002,1768838236000000000,0x1.4f28f5c28f5c3p+6,restaurant,0,0",
    "20000002,1768844270000000000,0x1.6a00000000000p+5,travel,0,0",
    "20000002,1768875514000000000,0x1.1d851eb851eb8p+7,fuel,0,0",
    "20000002,1769003021000000000,0x1.1b0a3d70a3d71p+4,grocery,0,0",
    "20000002,1769162859000000000,0x1.8d47ae147ae14p+4,online,0,0",
    "20000002,1769167670000000000,0x1.6870a3d70a3d7p+6,grocery,0,0",
    "20000002,1769226562000000000,0x1.20a3d70a3d70ap+4,fuel,0,0",
    "20000002,1769296902000000000,0x1.9247ae147ae14p+6,online,0,0",
    "20000002,1769434404000000000,0x1.a570a3d70a3d7p+4,grocery,0,0",
    "20000002,1769533972000000000,0x1.3cb851eb851ecp+5,travel,0,0",
    "20000002,1769590736000000000,0x1.d99999999999ap+4,online,0,0",
    "20000002,1769590743000000000,0x1.14e147ae147aep+5,online,1,0",
    "20000002,1769625421000000000,0x1.0500000000000p+6,online,0,0",
    "20000002,1769752473000000000,0x1.09d1eb851eb85p+7,online,0,0",
    "20000003,1767235094000000000,0x1.c733333333333p+4,grocery,0,0",
    "20000003,1767259880000000000,0x1.309999999999ap+6,restaurant,0,0",
    "20000003,1767259890000000000,0x1.18ae147ae147bp+6,grocery,0,0",
    "20000003,1767261059000000000,0x1.5c7ae147ae148p+3,online,0,0",
    "20000003,1767290110000000000,0x1.ee147ae147ae1p+5,grocery,0,0",
    "20000003,1767293306000000000,0x1.620a3d70a3d71p+6,online,0,0",
    "20000003,1767293729000000000,0x1.d7c28f5c28f5cp+5,electronics,0,0",
    "20000003,1767442446000000000,0x1.bc8f5c28f5c29p+6,grocery,0,0",
    "20000003,1767456283000000000,0x1.f23d70a3d70a4p+6,online,1,0",
    "20000003,1767485844000000000,0x1.d3c28f5c28f5cp+5,fuel,0,0",
    "20000003,1767506264000000000,0x1.79851eb851eb8p+6,restaurant,0,0",
    "20000003,1767506436000000000,0x1.d347ae147ae14p+5,restaurant,0,0",
    "20000003,1767511533000000000,0x1.1266666666666p+5,fuel,0,0",
    "20000003,1767605365000000000,0x1.268f5c28f5c29p+6,fuel,0,0",
    "20000003,1767625205000000000,0x1.f90a3d70a3d71p+5,travel,0,0",
    "20000003,1767632649000000000,0x1.0ae147ae147aep+5,travel,0,0",
    "20000003,1767710109000000000,0x1.18a3d70a3d70ap+4,grocery,0,0",
    "20000003,1767710414000000000,0x1.08c7ae147ae14p+7,online,0,0",
    "20000003,1767719622000000000,0x1.155c28f5c28f6p+5,travel,0,0",
    "20000003,1767776757000000000,0x1.0e3d70a3d70a4p+4,online,0,0",
    "20000003,1767800332000000000,0x1.a47ae147ae148p+2,online,0,0",
    "20000003,1767809237000000000,0x1.90ccccccccccdp+5,online,0,0",
    "20000003,1767895840000000000,0x1.8b851eb851eb8p+3,electronics,0,0",
    "20000003,1767900142000000000,0x1.45eb851eb851fp+6,grocery,0,0",
    "20000003,1767901156000000000,0x1.80147ae147ae1p+5,grocery,0,0",
    "20000003,1767931803000000000,0x1.24ccccccccccdp+5,restaurant,0,0",
    "20000003,1767932405000000000,0x1.03d1eb851eb85p+7,fuel,1,0",
    "20000003,1768002856000000000,0x1.af5c28f5c28f6p+1,grocery,0,0",
    "20000003,1768107768000000000,0x1.c8b851eb851ecp+5,online,0,0",
    "20000003,1768383469000000000,0x1.7dd1eb851eb85p+7,grocery,0,0",
    "20000003,1768395565000000000,0x1.2aa8f5c28f5c3p+7,grocery,0,0",
    "20000003,1768404411000000000,0x1.bee147ae147aep+4,travel,0,0",
    "20000003,1768405241000000000,0x1.30d70a3d70a3dp+7,restaurant,0,0",
    "20000003,1768442066000000000,0x1.f5eb851eb851fp+4,online,0,0",
    "20000003,1768453263000000000,0x1.15eb851eb851fp+5,fuel,0,0",
    "20000003,1768502287000000000,0x1.d2ccccccccccdp+5,restaurant,0,0",
    "20000003,1768502537000000000,0x1.f11eb851eb852p+5,fuel,0,0",
    "20000003,1768521183000000000,0x1.1851eb851eb85p+2,grocery,0,0",
    "20000003,1768521743000000000,0x1.383d70a3d70a4p+6,online,0,0",
    "20000003,1768647183000000000,0x1.cdccccccccccdp+6,online,0,0",
    "20000003,1768796665000000000,0x1.0eb851eb851ecp+6,travel,0,0",
    "20000003,1768803541000000000,0x1.d2147ae147ae1p+4,online,0,0",
    "20000003,1768804155000000000,0x1.e15c28f5c28f6p+5,online,1,0",
    "20000003,1768943248000000000,0x1.7d5c28f5c28f6p+5,restaurant,0,0",
    "20000003,1768943973000000000,0x1.33a3d70a3d70ap+7,online,0,0",
    "20000003,1769109345000000000,0x1.1866666666666p+6,grocery,0,0",
    "20000003,1769145640000000000,0x1.b4a3d70a3d70ap+4,grocery,0,0",
    "20000003,1769217373000000000,0x1.d3ae147ae147bp+5,online,0,0",
    "20000003,1769223266000000000,0x1.77ae147ae147bp+5,fuel,0,0",
    "20000003,1769237975000000000,0x1.e4ccccccccccdp+4,online,0,0",
    "20000003,1769254913000000000,0x1.2ee147ae147aep+4,online,1,0",
    "20000003,1769349926000000000,0x1.daccccccccccdp+5,online,0,0",
    "20000003,1769397426000000000,0x1.10f5c28f5c28fp+5,fuel,1,0",
    "20000003,1769543586000000000,0x1.13ccccccccccdp+7,grocery,0,0",
    "20000003,1769544334000000000,0x1.573d70a3d70a4p+6,restaurant,1,0",
    "20000003,1769668601000000000,0x1.30ccccccccccdp+6,online,0,0",
    "20000003,1769719355000000000,0x1.34b851eb851ecp+6,restaurant,0,0",
    "20000003,1769719546000000000,0x1.6b851eb851eb8p+3,restaurant,0,0",
    "20000003,1769748346000000000,0x1.15ae147ae147bp+5,restaurant,0,0",
    "20000003,1769786969000000000,0x1.491eb851eb852p+5,fuel,1,0",
    "20000004,1767406004000000000,0x1.6dc28f5c28f5cp+4,restaurant,0,0",
    "20000004,1767406572000000000,0x1.7ea3d70a3d70ap+5,grocery,0,0",
    "20000004,1767425201000000000,0x1.f2e147ae147aep+5,grocery,0,0",
    "20000004,1767425690000000000,0x1.00d999999999ap+8,restaurant,0,0",
    "20000004,1767426122000000000,0x1.7c51eb851eb85p+5,online,0,0",
    "20000004,1767622845000000000,0x1.0b70a3d70a3d7p+6,fuel,0,0",
    "20000004,1767682123000000000,0x1.40f5c28f5c28fp+4,restaurant,0,0",
    "20000004,1767682884000000000,0x1.94a3d70a3d70ap+6,online,1,0",
    "20000004,1767718221000000000,0x1.a99999999999ap+5,online,0,0",
    "20000004,1767802124000000000,0x1.2100000000000p+6,electronics,0,0",
    "20000004,1767853613000000000,0x1.2e8f5c28f5c29p+4,fuel,1,0",
    "20000004,1767877530000000000,0x1.4a47ae147ae14p+6,fuel,0,0",
    "20000004,1768147544000000000,0x1.359999999999ap+4,grocery,0,0",
    "20000004,1768175304000000000,0x1.860a3d70a3d71p+6,fuel,0,0",
    "20000004,1768176246000000000,0x1.fd1eb851eb852p+3,grocery,1,0",
    "20000004,1768318149000000000,0x1.3accccccccccdp+5,fuel,0,0",
    "20000004,1768336961000000000,0x1.8c147ae147ae1p+5,fuel,0,0",
    "20000004,1768337086000000000,0x1.09c28f5c28f5cp+4,grocery,0,0",
    "20000004,1768426674000000000,0x1.4ee147ae147aep+6,online,0,0",
    "20000004,1768465099000000000,0x1.7733333333333p+5,online,0,0",
    "20000004,1768466064000000000,0x1.e0ccccccccccdp+4,online,0,0",
    "20000004,1768490335000000000,0x1.1cccccccccccdp+4,grocery,0,0",
    "20000004,1768491156000000000,0x1.99eb851eb851fp+5,online,0,0",
    "20000004,1768508392000000000,0x1.3c51eb851eb85p+6,online,0,0",
    "20000004,1768558992000000000,0x1.679999999999ap+5,fuel,0,0",
    "20000004,1768559279000000000,0x1.f0a3d70a3d70ap+1,electronics,0,0",
    "20000004,1768579984000000000,0x1.0b9999999999ap+5,online,0,0",
    "20000004,1768580683000000000,0x1.c90a3d70a3d71p+5,fuel,1,0",
    "20000004,1768657687000000000,0x1.116b851eb851fp+7,online,1,0",
    "20000004,1768658767000000000,0x1.5f00000000000p+6,grocery,0,0",
    "20000004,1768717421000000000,0x1.c27ae147ae148p+5,grocery,0,0",
    "20000004,1768756847000000000,0x1.3d1eb851eb852p+6,online,1,0",
    "20000004,1768760951000000000,0x1.8f33333333333p+4,restaurant,0,0",
    "20000004,1768761592000000000,0x1.26e147ae147aep+6,online,1,0",
    "20000004,1768804900000000000,0x1.128f5c28f5c29p+5,online,0,0",
    "20000004,1768806016000000000,0x1.5a3d70a3d70a4p+6,grocery,0,0",
    "20000004,1768932367000000000,0x1.b10a3d70a3d71p+5,fuel,0,0",
    "20000004,1768940346000000000,0x1.1c147ae147ae1p+6,grocery,0,0",
    "20000004,1769156743000000000,0x1.1875c28f5c28fp+7,restaurant,1,0",
    "20000004,1769160430000000000,0x1.41d70a3d70a3dp+5,online,0,0",
    "20000004,1769160968000000000,0x1.373d70a3d70a4p+6,grocery,0,0",
    "20000004,1769293945000000000,0x1.73eb851eb851fp+6,restaurant,0,0",
    "20000004,1769462250000000000,0x1.c8b851eb851ecp+5,online,0,0",
    "20000004,1769548619000000000,0x1.9a1eb851eb852p+6,travel,0,0",
    "20000004,1769549257000000000,0x1.0ae147ae147aep+5,online,0,0",
    "20000004,1769566053000000000,0x1.008f5c28f5c29p+5,grocery,0,0",
    "20000004,1769642353000000000,0x1.8328f5c28f5c3p+6,online,0,0",
    "20000004,1769643502000000000,0x1.05851eb851eb8p+5,restaurant,1,0",
    "20000004,1769650306000000000,0x1.f800000000000p+4,electronics,0,0",
    "20000004,1769802384000000000,0x1.273d70a3d70a4p+6,online,0,0",
    "20000005,1767255767000000000,0x1.4e7ae147ae148p+5,grocery,0,0",
    "20000005,1767256515000000000,0x1.030a3d70a3d71p+4,online,0,0",
    "20000005,1767299684000000000,0x1.b333333333333p+4,fuel,0,0",
    "20000005,1767347028000000000,0x1.18051eb851eb8p+7,travel,0,0",
    "20000005,1767477932000000000,0x1.5d70a3d70a3d7p+5,fuel,0,0",
    "20000005,1767478647000000000,0x1.5d8f5c28f5c29p+6,grocery,0,0",
    "20000005,1767490311000000000,0x1.3947ae147ae14p+6,grocery,0,0",
    "20000005,1767490455000000000,0x1.aa3d70a3d70a4p+3,restaurant,0,0",
    "20000005,1767593666000000000,0x1.8f851eb851eb8p+5,grocery,0,0",
    "20000005,1767688393000000000,0x1.9628f5c28f5c3p+6,fuel,0,0",
    "20000005,1767689476000000000,0x1.fbd70a3d70a3dp+4,travel,0,0",
    "20000005,1767848055000000000,0x1.111eb851eb852p+6,grocery,0,0",
    "20000005,1767848955000000000,0x1.0bccccccccccdp+7,online,0,0",
    "20000005,1767896170000000000,0x1.062e147ae147bp+7,travel,0,0",
    "20000005,1767939915000000000,0x1.17d70a3d70a3dp+6,fuel,0,0",
    "20000005,1768104767000000000,0x1.ba28f5c28f5c3p+5,online,1,0",
    "20000005,1768226427000000000,0x1.e451eb851eb85p+4,grocery,0,0",
    "20000005,1768227345000000000,0x1.21d70a3d70a3dp+6,fuel,1,0",
    "20000005,1768245042000000000,0x1.7d5c28f5c28f6p+6,online,0,0",
    "20000005,1768298538000000000,0x1.9d0a3d70a3d71p+6,restaurant,0,0",
    "20000005,1768299389000000000,0x1.16b851eb851ecp+5,restaurant,0,0",
    "20000005,1768571468000000000,0x1.c1eb851eb851fp+3,restaurant,1,0",
    "20000005,1768655016000000000,0x1.c2e147ae147aep+5,grocery,0,0",
    "20000005,1768656027000000000,0x1.b270a3d70a3d7p+6,fuel,0,0",
    "20000005,1768658353000000000,0x1.1170a3d70a3d7p+4,fuel,0,0",
    "20000005,1768748404000000000,0x1.0e66666666666p+5,restaurant,0,0",
    "20000005,1768769944000000000,0x1.5c1eb851eb852p+7,grocery,1,0",
    "20000005,1768785355000000000,0x1.7c9999999999ap+6,grocery,0,0",
    "20000005,1768804203000000000,0x1.e970a3d70a3d7p+5,grocery,0,0",
    "20000005,1768856100000000000,0x1.e1851eb851eb8p+6,grocery,0,0",
    "20000005,1768901114000000000,0x1.3f0a3d70a3d71p+6,grocery,0,0"
};
/* scikit-learn's outputs for those rows (rows x NOUT), as hex floats */
static const double churn_st_expect[] = {
    0x1.c4549dd34b820p-1, 0x1.dd5b1165a3f04p-4, 0x0.0p+0, 0x1.c7ed43d2ca207p-1,
    0x1.c095e169aefc7p-4, 0x0.0p+0, 0x1.a6ff44fd3ad7fp-1, 0x1.6402ec0b14a03p-3,
    0x0.0p+0, 0x1.b7777d2ac0894p-1, 0x1.22220b54fddafp-3, 0x0.0p+0,
    0x1.c3432f95365b0p-1, 0x1.e5e683564d284p-4, 0x0.0p+0, 0x1.c78d677f70dc3p-1,
    0x1.c394c404791e9p-4, 0x0.0p+0, 0x1.a10d31138018fp-1, 0x1.7bcb3bb1ff9c3p-3,
    0x0.0p+0, 0x1.c7bc878480791p-1, 0x1.c21bc3dbfc379p-4, 0x0.0p+0,
    0x1.c813c9c5a2f7dp-1, 0x1.bf61b1d2e8417p-4, 0x0.0p+0, 0x1.c541807782ce3p-1,
    0x1.d5f3fc43e98e8p-4, 0x0.0p+0, 0x1.b32f37017f5bcp-1, 0x1.334323fa02910p-3,
    0x0.0p+0, 0x1.9e48bc123ddcfp-1, 0x1.86dd0fb7088c5p-3, 0x0.0p+0,
    0x1.c7ed43d2ca207p-1, 0x1.c095e169aefc7p-4, 0x0.0p+0, 0x1.cddec1eeeb491p-1,
    0x1.9109f088a5b78p-4, 0x0.0p+0, 0x1.9d9d2d54b156fp-1, 0x1.898b4aad3aa43p-3,
    0x0.0p+0, 0x1.9cc56fc083fa8p-1, 0x1.8cea40fdf0162p-3, 0x0.0p+0,
    0x1.b36c15c270a99p-1, 0x1.324fa8f63d59bp-3, 0x0.0p+0, 0x1.c4c28402f0d92p-1,
    0x1.d9ebdfe879373p-4, 0x0.0p+0, 0x1.9e0ee75985021p-1, 0x1.87c46299ebf7bp-3,
    0x0.0p+0, 0x1.8081f4a2fe432p-1, 0x1.fdf82d7406f38p-3, 0x0.0p+0,
    0x1.ae7e498aab300p-1, 0x1.4606d9d5533ffp-3, 0x0.0p+0, 0x1.51510345ef870p-1,
    0x1.5d5df97420f1fp-2, 0x0.0p+0, 0x1.bef4c560f6919p-1, 0x1.042cea7c25b9dp-3,
    0x0.0p+0, 0x1.7bedcca65c7d8p-1, 0x1.082466b34704fp-2, 0x0.0p+0,
    0x1.c7ed43d2ca207p-1, 0x1.c095e169aefc7p-4, 0x0.0p+0, 0x1.81a77ca887223p-1,
    0x1.f9620d5de3775p-3, 0x0.0p+0, 0x1.c5741c2b9d9a8p-1, 0x1.d45f1ea3132c0p-4,
    0x0.0p+0, 0x1.a7b9fd74600d8p-1, 0x1.61180a2e7fca0p-3, 0x0.0p+0,
    0x1.c003a4a98c306p-1, 0x1.ffe2dab39e7ccp-4, 0x0.0p+0, 0x1.c0bd51ca4e3cap-1,
    0x1.fa1571ad8e1b3p-4, 0x0.0p+0, 0x1.c813c9c5a2f7dp-1, 0x1.bf61b1d2e8417p-4,
    0x0.0p+0, 0x1.b2ac9544184c6p-1, 0x1.354daaef9ece6p-3, 0x0.0p+0,
    0x1.c5a329649fa36p-1, 0x1.d2e6b4db02e4dp-4, 0x0.0p+0, 0x1.c541807782ce3p-1,
    0x1.d5f3fc43e98e8p-4, 0x0.0p+0, 0x1.a6a2a2ff7ee9fp-1, 0x1.6575740204584p-3,
    0x0.0p+0, 0x1.b0a99e9351b10p-1, 0x1.3d5985b2b93c0p-3, 0x0.0p+0,
    0x1.cb343c463372cp-1, 0x1.a65e1dce6469dp-4, 0x0.0p+0, 0x1.aefd62a76d84ep-1,
    0x1.440a756249ec9p-3, 0x0.0p+0, 0x1.c4549dd34b820p-1, 0x1.dd5b1165a3f04p-4,
    0x0.0p+0, 0x1.ccc5208e525f0p-1, 0x1.99d6fb8d6d082p-4, 0x0.0p+0,
    0x1.b7b414e99afa8p-1, 0x1.212fac5994161p-3, 0x0.0p+0, 0x1.c80351770bdaep-1,
    0x1.bfe57447a128dp-4, 0x0.0p+0, 0x1.c5741c2b9d9a8p-1, 0x1.d45f1ea3132c0p-4,
    0x0.0p+0, 0x1.b32f37017f5bcp-1, 0x1.334323fa02910p-3, 0x0.0p+0,
    0x1.bfbd179d8112bp-1, 0x1.010ba189fbb54p-3, 0x0.0p+0, 0x1.c541807782ce3p-1,
    0x1.d5f3fc43e98e8p-4, 0x0.0p+0, 0x1.7207db7762592p-1, 0x1.1bf049113b4dcp-2,
    0x0.0p+0, 0x1.9bb3bb7bc715ap-1, 0x1.91311210e3a99p-3, 0x0.0p+0,
    0x1.b32f37017f5bcp-1, 0x1.334323fa02910p-3, 0x0.0p+0, 0x1.b189e3db02fadp-1,
    0x1.39d87093f414cp-3, 0x0.0p+0, 0x1.c868fefad00c0p-1, 0x1.bcb808297f9ffp-4,
    0x0.0p+0, 0x1.c277b627eef3cp-1, 0x1.ec424ec08861fp-4, 0x0.0p+0,
    0x1.b87cc1540090bp-1, 0x1.1e0cfaaffdbd3p-3, 0x0.0p+0, 0x1.ac93b81a85f0bp-1,
    0x1.4db11f95e83d4p-3, 0x0.0p+0, 0x1.c541807782ce3p-1, 0x1.d5f3fc43e98e8p-4,
    0x0.0p+0, 0x1.a7b9fd74600d8p-1, 0x1.61180a2e7fca0p-3, 0x0.0p+0,
    0x1.ca643608f93fap-1, 0x1.acde4fb83602ep-4, 0x0.0p+0, 0x1.7542ffeb03376p-1,
    0x1.157a0029f9913p-2, 0x0.0p+0, 0x1.c7bc878480791p-1, 0x1.c21bc3dbfc379p-4,
    0x0.0p+0, 0x1.a6b7849f4decap-1, 0x1.6521ed82c84d8p-3, 0x0.0p+0,
    0x1.a7fc4b8e04b25p-1, 0x1.600ed1c7ed36cp-3, 0x0.0p+0, 0x1.c0bd51ca4e3cap-1,
    0x1.fa1571ad8e1b3p-4, 0x0.0p+0, 0x1.c5a329649fa36p-1, 0x1.d2e6b4db02e4dp-4,
    0x0.0p+0, 0x1.8ef61ce2f01ffp-1, 0x1.c4278c743f804p-3, 0x0.0p+0,
    0x1.c541807782ce3p-1, 0x1.d5f3fc43e98e8p-4, 0x0.0p+0, 0x1.b1ca8dc15b0d4p-1,
    0x1.38d5c8fa93cb0p-3, 0x0.0p+0, 0x1.c5741c2b9d9a8p-1, 0x1.d45f1ea3132c0p-4,
    0x0.0p+0, 0x1.cccaf8dbe4bf8p-1, 0x1.99a83920da03ep-4, 0x0.0p+0,
    0x1.c541807782ce3p-1, 0x1.d5f3fc43e98e8p-4, 0x0.0p+0, 0x1.c813c9c5a2f7dp-1,
    0x1.bf61b1d2e8417p-4, 0x0.0p+0, 0x1.c5a329649fa36p-1, 0x1.d2e6b4db02e4dp-4,
    0x0.0p+0, 0x1.c0bd51ca4e3cap-1, 0x1.fa1571ad8e1b3p-4, 0x0.0p+0,
    0x1.a7fc4b8e04b25p-1, 0x1.600ed1c7ed36cp-3, 0x0.0p+0, 0x1.c5741c2b9d9a8p-1,
    0x1.d45f1ea3132c0p-4, 0x0.0p+0, 0x1.9cc56fc083fa8p-1, 0x1.8cea40fdf0162p-3,
    0x0.0p+0, 0x1.b7cca613dd2a8p-1, 0x1.20cd67b08b560p-3, 0x0.0p+0,
    0x1.c82efd3c5f2d7p-1, 0x1.be88161d06948p-4, 0x0.0p+0, 0x1.c003a4a98c306p-1,
    0x1.ffe2dab39e7ccp-4, 0x0.0p+0, 0x1.9bb3bb7bc715ap-1, 0x1.91311210e3a99p-3,
    0x0.0p+0, 0x1.8a60b58225652p-1, 0x1.d67d29f76a6b7p-3, 0x0.0p+0,
    0x1.8cf3c13582da0p-1, 0x1.cc30fb29f497ep-3, 0x0.0p+0, 0x1.9bb3bb7bc715ap-1,
    0x1.91311210e3a99p-3, 0x0.0p+0, 0x1.c541807782ce3p-1, 0x1.d5f3fc43e98e8p-4,
    0x0.0p+0, 0x1.afc460a7aae20p-1, 0x1.40ee7d615477fp-3, 0x0.0p+0,
    0x1.c7d2a5f705e7ep-1, 0x1.c16ad047d0c0fp-4, 0x0.0p+0, 0x1.c838a10ea6885p-1,
    0x1.be3af78acbbd5p-4, 0x0.0p+0, 0x1.bda2d764c193ap-1, 0x1.0974a26cf9b17p-3,
    0x0.0p+0, 0x1.bfccffb514ee4p-1, 0x1.00cc012bac46fp-3, 0x0.0p+0,
    0x1.b32f37017f5bcp-1, 0x1.334323fa02910p-3, 0x0.0p+0, 0x1.b57a0fcf0264cp-1,
    0x1.2a17c0c3f66cep-3, 0x0.0p+0, 0x1.bd66a3b08bf58p-1, 0x1.0a65713dd02a1p-3,
    0x0.0p+0, 0x1.b656ad0e0352dp-1, 0x1.26a54bc7f2b4bp-3, 0x0.0p+0,
    0x1.97926b87a3df6p-1, 0x1.a1b651e170826p-3, 0x0.0p+0, 0x1.c7f70f555c684p-1,
    0x1.c04785551cbe4p-4, 0x0.0p+0, 0x1.c4b79d0e6c173p-1, 0x1.da43178c9f466p-4,
    0x0.0p+0, 0x1.c48f62ea9c800p-1, 0x1.db84e8ab1c003p-4, 0x0.0p+0,
    0x1.c003a4a98c306p-1, 0x1.ffe2dab39e7ccp-4, 0x0.0p+0, 0x1.a6b7849f4decap-1,
    0x1.6521ed82c84d8p-3, 0x0.0p+0, 0x1.c78d677f70dc3p-1, 0x1.c394c404791e9p-4,
    0x0.0p+0, 0x1.97e8738715042p-1, 0x1.a05e31e3abef7p-3, 0x0.0p+0,
    0x1.b57a0fcf0264cp-1, 0x1.2a17c0c3f66cep-3, 0x0.0p+0, 0x1.c7bc878480791p-1,
    0x1.c21bc3dbfc379p-4, 0x0.0p+0, 0x1.b84f40355cdbcp-1, 0x1.1ec2ff2a8c912p-3,
    0x0.0p+0, 0x1.c0bd51ca4e3cap-1, 0x1.fa1571ad8e1b3p-4, 0x0.0p+0,
    0x1.c7ed43d2ca207p-1, 0x1.c095e169aefc7p-4, 0x0.0p+0, 0x1.c7ed43d2ca207p-1,
    0x1.c095e169aefc7p-4, 0x0.0p+0, 0x1.cdb29e1f2116ap-1, 0x1.926b0f06f74b1p-4,
    0x0.0p+0, 0x1.c5a329649fa36p-1, 0x1.d2e6b4db02e4dp-4, 0x0.0p+0,
    0x1.b24910b5dc7b8p-1, 0x1.36dbbd288e122p-3, 0x0.0p+0, 0x1.c82efd3c5f2d7p-1,
    0x1.be88161d06948p-4, 0x0.0p+0, 0x1.a772b41a9da24p-1, 0x1.62352f9589770p-3,
    0x0.0p+0, 0x1.2316b27d925e2p-1, 0x1.b9d29b04db43bp-2, 0x0.0p+0,
    0x1.39edff0ed6d2fp-1, 0x1.8c2401e2525a2p-2, 0x0.0p+0, 0x1.6d49d1675415cp-1,
    0x1.256c5d3157d49p-2, 0x0.0p+0, 0x1.20d2ebecdb732p-1, 0x1.be5a28264919bp-2,
    0x0.0p+0, 0x1.35938d76d42d0p-1, 0x1.94d8e51257a5fp-2, 0x0.0p+0,
    0x1.2841491592240p-1, 0x1.af7d6dd4dbb7fp-2, 0x0.0p+0, 0x1.3339c2a7460ccp-1,
    0x1.998c7ab173e67p-2, 0x0.0p+0, 0x1.bef4c560f6919p-1, 0x1.042cea7c25b9dp-3,
    0x0.0p+0, 0x1.c003a4a98c306p-1, 0x1.ffe2dab39e7ccp-4, 0x0.0p+0,
    0x1.b2c870887085cp-1, 0x1.34de3dde3de91p-3, 0x0.0p+0, 0x1.a83572174c979p-1,
    0x1.5f2a37a2cda1cp-3, 0x0.0p+0, 0x1.8d245eefb2ee4p-1, 0x1.cb6e844134472p-3,
    0x0.0p+0, 0x1.c813c9c5a2f7dp-1, 0x1.bf61b1d2e8417p-4, 0x0.0p+0,
    0x1.afd27bca0d753p-1, 0x1.40b610d7ca2b4p-3, 0x0.0p+0, 0x1.c7ed43d2ca207p-1,
    0x1.c095e169aefc7p-4, 0x0.0p+0, 0x1.c003a4a98c306p-1, 0x1.ffe2dab39e7ccp-4,
    0x0.0p+0, 0x1.b32f37017f5bcp-1, 0x1.334323fa02910p-3, 0x0.0p+0,
    0x1.c5741c2b9d9a8p-1, 0x1.d45f1ea3132c0p-4, 0x0.0p+0, 0x1.af3b4652fee99p-1,
    0x1.4312e6b40459dp-3, 0x0.0p+0, 0x1.c5741c2b9d9a8p-1, 0x1.d45f1ea3132c0p-4,
    0x0.0p+0, 0x1.93aa29f2b27adp-1, 0x1.b15758353614cp-3, 0x0.0p+0,
    0x1.c81ac5fd4d93fp-1, 0x1.bf29d01593607p-4, 0x0.0p+0, 0x1.bfccffb514ee4p-1,
    0x1.00cc012bac46fp-3, 0x0.0p+0, 0x1.c487ead494ed8p-1, 0x1.dbc0a95b58944p-4,
    0x0.0p+0, 0x1.97e8738715042p-1, 0x1.a05e31e3abef7p-3, 0x0.0p+0,
    0x1.c640dc2483d07p-1, 0x1.cdf91edbe17c6p-4, 0x0.0p+0, 0x1.a6d8c87e8fa0ep-1,
    0x1.649cde05c17cap-3, 0x0.0p+0, 0x1.c48f62ea9c800p-1, 0x1.db84e8ab1c003p-4,
    0x0.0p+0, 0x1.c7ed43d2ca207p-1, 0x1.c095e169aefc7p-4, 0x0.0p+0,
    0x1.c868fefad00c0p-1, 0x1.bcb808297f9ffp-4, 0x0.0p+0, 0x1.b208bc9c834acp-1,
    0x1.37dd0d8df2d51p-3, 0x0.0p+0, 0x1.c5741c2b9d9a8p-1, 0x1.d45f1ea3132c0p-4,
    0x0.0p+0, 0x1.c63522f1a4467p-1, 0x1.ce56e872ddcc6p-4, 0x0.0p+0,
    0x1.a7ac215eca44ep-1, 0x1.614f7a84d6ec7p-3, 0x0.0p+0, 0x1.be93fe17a4be4p-1,
    0x1.05b007a16d06fp-3, 0x0.0p+0, 0x1.9cc56fc083fa8p-1, 0x1.8cea40fdf0162p-3,
    0x0.0p+0, 0x1.a827148f9530ap-1, 0x1.5f63adc1ab3d9p-3, 0x0.0p+0,
    0x1.c78d677f70dc3p-1, 0x1.c394c404791e9p-4, 0x0.0p+0, 0x1.c7bc878480791p-1,
    0x1.c21bc3dbfc379p-4, 0x0.0p+0, 0x1.b50958c5c6edfp-1, 0x1.2bda9ce8e4484p-3,
    0x0.0p+0, 0x1.c78d677f70dc3p-1, 0x1.c394c404791e9p-4, 0x0.0p+0,
    0x1.b84f40355cdbcp-1, 0x1.1ec2ff2a8c912p-3, 0x0.0p+0, 0x1.c7ed43d2ca207p-1,
    0x1.c095e169aefc7p-4, 0x0.0p+0, 0x1.9cc56fc083fa8p-1, 0x1.8cea40fdf0162p-3,
    0x0.0p+0, 0x1.c60318c1da3e9p-1, 0x1.cfe739f12e0bbp-4, 0x0.0p+0,
    0x1.b2ac9544184c6p-1, 0x1.354daaef9ece6p-3, 0x0.0p+0, 0x1.bfccffb514ee4p-1,
    0x1.00cc012bac46fp-3, 0x0.0p+0, 0x1.ad80bbdaf4f4cp-1, 0x1.49fd10942c2cfp-3,
    0x0.0p+0, 0x1.c7ed43d2ca207p-1, 0x1.c095e169aefc7p-4, 0x0.0p+0,
    0x1.c0bd51ca4e3cap-1, 0x1.fa1571ad8e1b3p-4, 0x0.0p+0, 0x1.ab91d9ca9c7d4p-1,
    0x1.51b898d58e0b1p-3, 0x0.0p+0, 0x1.84a029897c908p-1, 0x1.ed7f59da0dbe2p-3,
    0x0.0p+0, 0x1.c003a4a98c306p-1, 0x1.ffe2dab39e7ccp-4, 0x0.0p+0,
    0x1.a8927e6562a78p-1, 0x1.5db6066a7561fp-3, 0x0.0p+0, 0x1.aab5b3126c365p-1,
    0x1.552933b64f26dp-3, 0x0.0p+0, 0x1.b2c870887085cp-1, 0x1.34de3dde3de91p-3,
    0x0.0p+0, 0x1.af45474e26155p-1, 0x1.42eae2c767aabp-3, 0x0.0p+0,
    0x1.c4f20d5f69ba3p-1, 0x1.d86f9504b22eap-4, 0x0.0p+0, 0x1.b77c3f4b27e9ep-1,
    0x1.220f02d360587p-3, 0x0.0p+0, 0x1.ad362049f0c5bp-1, 0x1.4b277ed83ce95p-3,
    0x0.0p+0, 0x1.c868fefad00c0p-1, 0x1.bcb808297f9ffp-4, 0x0.0p+0,
    0x1.c0efa1fc09192p-1, 0x1.f882f01fb7374p-4, 0x0.0p+0, 0x1.96aec1db5b9d7p-1,
    0x1.a544f892918a5p-3, 0x0.0p+0, 0x1.b06c1d545580dp-1, 0x1.3e4f8aaea9fcdp-3,
    0x0.0p+0, 0x1.c78d677f70dc3p-1, 0x1.c394c404791e9p-4, 0x0.0p+0,
    0x1.96d6f35701f21p-1, 0x1.a4a432a3f837cp-3, 0x0.0p+0, 0x1.a35f7900772c9p-1,
    0x1.72821bfe234dbp-3, 0x0.0p+0, 0x1.c75c6239cbae3p-1, 0x1.c51cee31a28e5p-4,
    0x0.0p+0, 0x1.cd424688f76dep-1, 0x1.95edcbb84490dp-4, 0x0.0p+0,
    0x1.c541807782ce3p-1, 0x1.d5f3fc43e98e8p-4, 0x0.0p+0, 0x1.c813c9c5a2f7dp-1,
    0x1.bf61b1d2e8417p-4, 0x0.0p+0, 0x1.bff3c82c44cb3p-1, 0x1.0030df4eecd35p-3,
    0x0.0p+0, 0x1.b32f37017f5bcp-1, 0x1.334323fa02910p-3, 0x0.0p+0,
    0x1.c5741c2b9d9a8p-1, 0x1.d45f1ea3132c0p-4, 0x0.0p+0, 0x1.9a198cead5599p-1,
    0x1.9799cc54aa99cp-3, 0x0.0p+0, 0x1.c7f70f555c684p-1, 0x1.c04785551cbe4p-4,
    0x0.0p+0, 0x1.c48f62ea9c800p-1, 0x1.db84e8ab1c003p-4, 0x0.0p+0,
    0x1.9e9f60b7ec600p-1, 0x1.85827d204e802p-3, 0x0.0p+0, 0x1.a6d8c87e8fa0ep-1,
    0x1.649cde05c17cap-3, 0x0.0p+0, 0x1.afc460a7aae20p-1, 0x1.40ee7d615477fp-3,
    0x0.0p+0, 0x1.8fd24eae53d5ep-1, 0x1.c0b6c546b0a87p-3, 0x0.0p+0,
    0x1.c7d2a5f705e7ep-1, 0x1.c16ad047d0c0fp-4, 0x0.0p+0, 0x1.c81ac5fd4d93fp-1,
    0x1.bf29d01593607p-4, 0x0.0p+0, 0x1.c4c28402f0d92p-1, 0x1.d9ebdfe879373p-4,
    0x0.0p+0, 0x1.a74201f7c96a1p-1, 0x1.62f7f820da57bp-3, 0x0.0p+0,
    0x1.c7f70f555c684p-1, 0x1.c04785551cbe4p-4, 0x0.0p+0, 0x1.a98e8ec7e1092p-1,
    0x1.59c5c4e07bdb6p-3, 0x0.0p+0, 0x1.b288730fae6e3p-1, 0x1.35de33c146475p-3,
    0x0.0p+0, 0x1.93aa29f2b27adp-1, 0x1.b15758353614cp-3, 0x0.0p+0,
    0x1.c7bc878480791p-1, 0x1.c21bc3dbfc379p-4, 0x0.0p+0, 0x1.a6b7849f4decap-1,
    0x1.6521ed82c84d8p-3, 0x0.0p+0, 0x1.c5741c2b9d9a8p-1, 0x1.d45f1ea3132c0p-4,
    0x0.0p+0, 0x1.8f7cd78dd47f8p-1, 0x1.c20ca1c8ae022p-3, 0x0.0p+0,
    0x1.c0bd51ca4e3cap-1, 0x1.fa1571ad8e1b3p-4, 0x0.0p+0, 0x1.c7ed43d2ca207p-1,
    0x1.c095e169aefc7p-4, 0x0.0p+0, 0x1.c48f62ea9c800p-1, 0x1.db84e8ab1c003p-4,
    0x0.0p+0, 0x1.c5a329649fa36p-1, 0x1.d2e6b4db02e4dp-4, 0x0.0p+0,
    0x1.c7d2a5f705e7ep-1, 0x1.c16ad047d0c0fp-4, 0x0.0p+0, 0x1.c75c6239cbae3p-1,
    0x1.c51cee31a28e5p-4, 0x0.0p+0, 0x1.c670b0b327d35p-1, 0x1.cc7a7a66c1656p-4,
    0x0.0p+0, 0x1.c868fefad00c0p-1, 0x1.bcb808297f9ffp-4, 0x0.0p+0,
    0x1.ac5288ac7f12cp-1, 0x1.4eb5dd4e03b52p-3, 0x0.0p+0, 0x1.a767470ff9aafp-1,
    0x1.6262e3c019543p-3, 0x0.0p+0, 0x1.c08734f073cd3p-1, 0x1.fbc6587c61966p-4,
    0x0.0p+0, 0x1.c22ad118b40c8p-1, 0x1.eea9773a5f9bdp-4, 0x0.0p+0,
    0x1.b27bedbecb043p-1, 0x1.36104904d3ef3p-3, 0x0.0p+0, 0x1.c81ac5fd4d93fp-1,
    0x1.bf29d01593607p-4, 0x0.0p+0, 0x1.c5a329649fa36p-1, 0x1.d2e6b4db02e4dp-4,
    0x0.0p+0, 0x1.c7ed43d2ca207p-1, 0x1.c095e169aefc7p-4, 0x0.0p+0,
    0x1.b32f37017f5bcp-1, 0x1.334323fa02910p-3, 0x0.0p+0, 0x1.c003a4a98c306p-1,
    0x1.ffe2dab39e7ccp-4, 0x0.0p+0, 0x1.592a99e684adap-1, 0x1.4daacc32f6a4cp-2,
    0x0.0p+0, 0x1.9298c09dab604p-1, 0x1.b59cfd89527f2p-3, 0x0.0p+0,
    0x1.c0efa1fc09192p-1, 0x1.f882f01fb7374p-4, 0x0.0p+0, 0x1.c7ed43d2ca207p-1,
    0x1.c095e169aefc7p-4, 0x0.0p+0, 0x1.949ece1105c67p-1, 0x1.ad84c7bbe8e63p-3,
    0x0.0p+0, 0x1.c4549dd34b820p-1, 0x1.dd5b1165a3f04p-4, 0x0.0p+0,
    0x1.c0bd51ca4e3cap-1, 0x1.fa1571ad8e1b3p-4, 0x0.0p+0, 0x1.c78d677f70dc3p-1,
    0x1.c394c404791e9p-4, 0x0.0p+0, 0x1.c026a1d9a214fp-1, 0x1.fecaf132ef58bp-4,
    0x0.0p+0, 0x1.c78d677f70dc3p-1, 0x1.c394c404791e9p-4, 0x0.0p+0,
    0x1.b5b3d97b79f36p-1, 0x1.29309a1218326p-3, 0x0.0p+0, 0x1.b7619a860c7a3p-1,
    0x1.227995e7ce173p-3, 0x0.0p+0, 0x1.c5a329649fa36p-1, 0x1.d2e6b4db02e4dp-4,
    0x0.0p+0, 0x1.c5a329649fa36p-1, 0x1.d2e6b4db02e4dp-4, 0x0.0p+0,
    0x1.b7619a860c7a3p-1, 0x1.227995e7ce173p-3, 0x0.0p+0, 0x1.be5c4d2982bf4p-1,
    0x1.068ecb59f5031p-3, 0x0.0p+0, 0x1.c541807782ce3p-1, 0x1.d5f3fc43e98e8p-4,
    0x0.0p+0, 0x1.b4956b18c7a81p-1, 0x1.2daa539ce15fcp-3, 0x0.0p+0,
    0x1.c78d677f70dc3p-1, 0x1.c394c404791e9p-4, 0x0.0p+0, 0x1.a949a14f93dafp-1,
    0x1.5ad97ac1b0944p-3, 0x0.0p+0, 0x1.bfbd179d8112bp-1, 0x1.010ba189fbb54p-3,
    0x0.0p+0, 0x1.c5a329649fa36p-1, 0x1.d2e6b4db02e4dp-4, 0x0.0p+0,
    0x1.ae70cb8c8c67cp-1, 0x1.463cd1cdce60fp-3, 0x0.0p+0, 0x1.c838a10ea6885p-1,
    0x1.be3af78acbbd5p-4, 0x0.0p+0, 0x1.ad751d26fb4c2p-1, 0x1.4a2b8b6412cf9p-3,
    0x0.0p+0, 0x1.a7c6a23830802p-1, 0x1.60e5771f3dffap-3, 0x0.0p+0,
    0x1.a87136d3b6ffep-1, 0x1.5e3b24b124008p-3, 0x0.0p+0, 0x1.c4c28402f0d92p-1,
    0x1.d9ebdfe879373p-4, 0x0.0p+0, 0x1.bfbd179d8112bp-1, 0x1.010ba189fbb54p-3,
    0x0.0p+0, 0x1.c7ed43d2ca207p-1, 0x1.c095e169aefc7p-4, 0x0.0p+0,
    0x1.a7ac215eca44ep-1, 0x1.614f7a84d6ec7p-3, 0x0.0p+0, 0x1.c838a10ea6885p-1,
    0x1.be3af78acbbd5p-4, 0x0.0p+0, 0x1.a87136d3b6ffep-1, 0x1.5e3b24b124008p-3,
    0x0.0p+0, 0x1.c4b79d0e6c173p-1, 0x1.da43178c9f466p-4, 0x0.0p+0,
    0x1.c0bd51ca4e3cap-1, 0x1.fa1571ad8e1b3p-4, 0x0.0p+0, 0x1.a7e4dffea3e65p-1,
    0x1.606c80057066cp-3, 0x0.0p+0, 0x1.c541807782ce3p-1, 0x1.d5f3fc43e98e8p-4,
    0x0.0p+0, 0x1.ad63246f8bd2cp-1, 0x1.4a736e41d0b51p-3, 0x0.0p+0,
    0x1.b77c3f4b27e9ep-1, 0x1.220f02d360587p-3, 0x0.0p+0, 0x1.c48f62ea9c800p-1,
    0x1.db84e8ab1c003p-4, 0x0.0p+0, 0x1.90d51da671117p-1, 0x1.bcab89663bba5p-3,
    0x0.0p+0, 0x1.b2ac9544184c6p-1, 0x1.354daaef9ece6p-3, 0x0.0p+0,
    0x1.c82efd3c5f2d7p-1, 0x1.be88161d06948p-4, 0x0.0p+0, 0x1.b53beb8e26448p-1,
    0x1.2b1051c766ee0p-3, 0x0.0p+0, 0x1.b29fd9d1e3cc3p-1, 0x1.358098b870cf4p-3,
    0x0.0p+0, 0x1.c80351770bdaep-1, 0x1.bfe57447a128dp-4, 0x0.0p+0,
    0x1.c08734f073cd3p-1, 0x1.fbc6587c61966p-4, 0x0.0p+0, 0x1.c5741c2b9d9a8p-1,
    0x1.d45f1ea3132c0p-4, 0x0.0p+0, 0x1.b32086e7355dap-1, 0x1.337de4632a896p-3,
    0x0.0p+0, 0x1.c7ed43d2ca207p-1, 0x1.c095e169aefc7p-4, 0x0.0p+0,
    0x1.7c0b4aec7731dp-1, 0x1.07e96a27119c6p-2, 0x0.0p+0, 0x1.c5a329649fa36p-1,
    0x1.d2e6b4db02e4dp-4, 0x0.0p+0, 0x1.a87136d3b6ffep-1, 0x1.5e3b24b124008p-3,
    0x0.0p+0, 0x1.be879adf0cd08p-1, 0x1.05e19483ccbe1p-3, 0x0.0p+0,
    0x1.b27bedbecb043p-1, 0x1.36104904d3ef3p-3, 0x0.0p+0, 0x1.c5741c2b9d9a8p-1,
    0x1.d45f1ea3132c0p-4, 0x0.0p+0, 0x1.a5391e1acac95p-1, 0x1.6b1b8794d4dabp-3,
    0x0.0p+0, 0x1.a3e1b50d34ca1p-1, 0x1.70792bcb2cd7dp-3, 0x0.0p+0,
    0x1.c0bd51ca4e3cap-1, 0x1.fa1571ad8e1b3p-4, 0x0.0p+0, 0x1.96d6f35701f21p-1,
    0x1.a4a432a3f837cp-3, 0x0.0p+0, 0x1.855fd069bae4ep-1, 0x1.ea80be59146c9p-3,
    0x0.0p+0, 0x1.c0efa1fc09192p-1, 0x1.f882f01fb7374p-4, 0x0.0p+0,
    0x1.c48f62ea9c800p-1, 0x1.db84e8ab1c003p-4, 0x0.0p+0, 0x1.c80351770bdaep-1,
    0x1.bfe57447a128dp-4, 0x0.0p+0, 0x1.a83572174c979p-1, 0x1.5f2a37a2cda1cp-3,
    0x0.0p+0, 0x1.b73f807ebda97p-1, 0x1.2301fe05095a4p-3, 0x0.0p+0,
    0x1.b7b414e99afa8p-1, 0x1.212fac5994161p-3, 0x0.0p+0, 0x1.a4782135c61c4p-1,
    0x1.6e1f7b28e78efp-3, 0x0.0p+0, 0x1.c813c9c5a2f7dp-1, 0x1.bf61b1d2e8417p-4,
    0x0.0p+0, 0x1.c4c28402f0d92p-1, 0x1.d9ebdfe879373p-4, 0x0.0p+0,
    0x1.9833ba3ff55aap-1, 0x1.9f3117002a958p-3, 0x0.0p+0, 0x1.8ec7daacc193cp-1,
    0x1.c4e0954cf9b11p-3, 0x0.0p+0, 0x1.c868fefad00c0p-1, 0x1.bcb808297f9ffp-4,
    0x0.0p+0, 0x1.a48c6a9ab8805p-1, 0x1.6dce55951dfebp-3, 0x0.0p+0,
    0x1.b77c3f4b27e9ep-1, 0x1.220f02d360587p-3, 0x0.0p+0, 0x1.c003a4a98c306p-1,
    0x1.ffe2dab39e7ccp-4, 0x0.0p+0, 0x1.b3e6a3481c546p-1, 0x1.306572df8eae7p-3,
    0x0.0p+0, 0x1.bff3c82c44cb3p-1, 0x1.0030df4eecd35p-3, 0x0.0p+0,
};
/* the largest |difference| allowed from scikit-learn's value: 0 means the bits */
static const double churn_st_tol = 4.440892098500626e-16;
/* this file's own outputs at compile time, which the self-test requires to the bit */
static const double churn_st_own[] = {
    0x1.c4549dd34b820p-1, 0x1.dd5b1165a3f04p-4, 0x0.0p+0, 0x1.c7ed43d2ca207p-1,
    0x1.c095e169aefc7p-4, 0x0.0p+0, 0x1.a6ff44fd3ad7fp-1, 0x1.6402ec0b14a03p-3,
    0x0.0p+0, 0x1.b7777d2ac0894p-1, 0x1.22220b54fddafp-3, 0x0.0p+0,
    0x1.c3432f95365b0p-1, 0x1.e5e683564d284p-4, 0x0.0p+0, 0x1.c78d677f70dc3p-1,
    0x1.c394c404791e9p-4, 0x0.0p+0, 0x1.a10d31138018fp-1, 0x1.7bcb3bb1ff9c3p-3,
    0x0.0p+0, 0x1.c7bc878480791p-1, 0x1.c21bc3dbfc379p-4, 0x0.0p+0,
    0x1.c813c9c5a2f7dp-1, 0x1.bf61b1d2e8417p-4, 0x0.0p+0, 0x1.c541807782ce3p-1,
    0x1.d5f3fc43e98e8p-4, 0x0.0p+0, 0x1.b32f37017f5bcp-1, 0x1.334323fa02910p-3,
    0x0.0p+0, 0x1.9e48bc123ddcfp-1, 0x1.86dd0fb7088c5p-3, 0x0.0p+0,
    0x1.c7ed43d2ca207p-1, 0x1.c095e169aefc7p-4, 0x0.0p+0, 0x1.cddec1eeeb491p-1,
    0x1.9109f088a5b78p-4, 0x0.0p+0, 0x1.9d9d2d54b156fp-1, 0x1.898b4aad3aa43p-3,
    0x0.0p+0, 0x1.9cc56fc083fa8p-1, 0x1.8cea40fdf0162p-3, 0x0.0p+0,
    0x1.b36c15c270a99p-1, 0x1.324fa8f63d59bp-3, 0x0.0p+0, 0x1.c4c28402f0d92p-1,
    0x1.d9ebdfe879373p-4, 0x0.0p+0, 0x1.9e0ee75985021p-1, 0x1.87c46299ebf7bp-3,
    0x0.0p+0, 0x1.8081f4a2fe432p-1, 0x1.fdf82d7406f38p-3, 0x0.0p+0,
    0x1.ae7e498aab300p-1, 0x1.4606d9d5533ffp-3, 0x0.0p+0, 0x1.51510345ef870p-1,
    0x1.5d5df97420f1fp-2, 0x0.0p+0, 0x1.bef4c560f6919p-1, 0x1.042cea7c25b9dp-3,
    0x0.0p+0, 0x1.7bedcca65c7d8p-1, 0x1.082466b34704fp-2, 0x0.0p+0,
    0x1.c7ed43d2ca207p-1, 0x1.c095e169aefc7p-4, 0x0.0p+0, 0x1.81a77ca887223p-1,
    0x1.f9620d5de3775p-3, 0x0.0p+0, 0x1.c5741c2b9d9a8p-1, 0x1.d45f1ea3132c0p-4,
    0x0.0p+0, 0x1.a7b9fd74600d8p-1, 0x1.61180a2e7fca0p-3, 0x0.0p+0,
    0x1.c003a4a98c306p-1, 0x1.ffe2dab39e7ccp-4, 0x0.0p+0, 0x1.c0bd51ca4e3cap-1,
    0x1.fa1571ad8e1b3p-4, 0x0.0p+0, 0x1.c813c9c5a2f7dp-1, 0x1.bf61b1d2e8417p-4,
    0x0.0p+0, 0x1.b2ac9544184c6p-1, 0x1.354daaef9ece6p-3, 0x0.0p+0,
    0x1.c5a329649fa36p-1, 0x1.d2e6b4db02e4dp-4, 0x0.0p+0, 0x1.c541807782ce3p-1,
    0x1.d5f3fc43e98e8p-4, 0x0.0p+0, 0x1.a6a2a2ff7ee9fp-1, 0x1.6575740204584p-3,
    0x0.0p+0, 0x1.b0a99e9351b10p-1, 0x1.3d5985b2b93c0p-3, 0x0.0p+0,
    0x1.cb343c463372cp-1, 0x1.a65e1dce6469dp-4, 0x0.0p+0, 0x1.aefd62a76d84ep-1,
    0x1.440a756249ec9p-3, 0x0.0p+0, 0x1.c4549dd34b820p-1, 0x1.dd5b1165a3f04p-4,
    0x0.0p+0, 0x1.ccc5208e525f0p-1, 0x1.99d6fb8d6d082p-4, 0x0.0p+0,
    0x1.b7b414e99afa8p-1, 0x1.212fac5994161p-3, 0x0.0p+0, 0x1.c80351770bdaep-1,
    0x1.bfe57447a128dp-4, 0x0.0p+0, 0x1.c5741c2b9d9a8p-1, 0x1.d45f1ea3132c0p-4,
    0x0.0p+0, 0x1.b32f37017f5bcp-1, 0x1.334323fa02910p-3, 0x0.0p+0,
    0x1.bfbd179d8112bp-1, 0x1.010ba189fbb54p-3, 0x0.0p+0, 0x1.c541807782ce3p-1,
    0x1.d5f3fc43e98e8p-4, 0x0.0p+0, 0x1.7207db7762592p-1, 0x1.1bf049113b4dcp-2,
    0x0.0p+0, 0x1.9bb3bb7bc715ap-1, 0x1.91311210e3a99p-3, 0x0.0p+0,
    0x1.b32f37017f5bcp-1, 0x1.334323fa02910p-3, 0x0.0p+0, 0x1.b189e3db02fadp-1,
    0x1.39d87093f414cp-3, 0x0.0p+0, 0x1.c868fefad00c0p-1, 0x1.bcb808297f9ffp-4,
    0x0.0p+0, 0x1.c277b627eef3cp-1, 0x1.ec424ec08861fp-4, 0x0.0p+0,
    0x1.b87cc1540090bp-1, 0x1.1e0cfaaffdbd3p-3, 0x0.0p+0, 0x1.ac93b81a85f0bp-1,
    0x1.4db11f95e83d4p-3, 0x0.0p+0, 0x1.c541807782ce3p-1, 0x1.d5f3fc43e98e8p-4,
    0x0.0p+0, 0x1.a7b9fd74600d8p-1, 0x1.61180a2e7fca0p-3, 0x0.0p+0,
    0x1.ca643608f93fap-1, 0x1.acde4fb83602ep-4, 0x0.0p+0, 0x1.7542ffeb03376p-1,
    0x1.157a0029f9913p-2, 0x0.0p+0, 0x1.c7bc878480791p-1, 0x1.c21bc3dbfc379p-4,
    0x0.0p+0, 0x1.a6b7849f4decap-1, 0x1.6521ed82c84d8p-3, 0x0.0p+0,
    0x1.a7fc4b8e04b25p-1, 0x1.600ed1c7ed36cp-3, 0x0.0p+0, 0x1.c0bd51ca4e3cap-1,
    0x1.fa1571ad8e1b3p-4, 0x0.0p+0, 0x1.c5a329649fa36p-1, 0x1.d2e6b4db02e4dp-4,
    0x0.0p+0, 0x1.8ef61ce2f01ffp-1, 0x1.c4278c743f804p-3, 0x0.0p+0,
    0x1.c541807782ce3p-1, 0x1.d5f3fc43e98e8p-4, 0x0.0p+0, 0x1.b1ca8dc15b0d4p-1,
    0x1.38d5c8fa93cb0p-3, 0x0.0p+0, 0x1.c5741c2b9d9a8p-1, 0x1.d45f1ea3132c0p-4,
    0x0.0p+0, 0x1.cccaf8dbe4bf8p-1, 0x1.99a83920da03ep-4, 0x0.0p+0,
    0x1.c541807782ce3p-1, 0x1.d5f3fc43e98e8p-4, 0x0.0p+0, 0x1.c813c9c5a2f7dp-1,
    0x1.bf61b1d2e8417p-4, 0x0.0p+0, 0x1.c5a329649fa36p-1, 0x1.d2e6b4db02e4dp-4,
    0x0.0p+0, 0x1.c0bd51ca4e3cap-1, 0x1.fa1571ad8e1b3p-4, 0x0.0p+0,
    0x1.a7fc4b8e04b25p-1, 0x1.600ed1c7ed36cp-3, 0x0.0p+0, 0x1.c5741c2b9d9a8p-1,
    0x1.d45f1ea3132c0p-4, 0x0.0p+0, 0x1.9cc56fc083fa8p-1, 0x1.8cea40fdf0162p-3,
    0x0.0p+0, 0x1.b7cca613dd2a8p-1, 0x1.20cd67b08b560p-3, 0x0.0p+0,
    0x1.c82efd3c5f2d7p-1, 0x1.be88161d06948p-4, 0x0.0p+0, 0x1.c003a4a98c306p-1,
    0x1.ffe2dab39e7ccp-4, 0x0.0p+0, 0x1.9bb3bb7bc715ap-1, 0x1.91311210e3a99p-3,
    0x0.0p+0, 0x1.8a60b58225652p-1, 0x1.d67d29f76a6b7p-3, 0x0.0p+0,
    0x1.8cf3c13582da0p-1, 0x1.cc30fb29f497ep-3, 0x0.0p+0, 0x1.9bb3bb7bc715ap-1,
    0x1.91311210e3a99p-3, 0x0.0p+0, 0x1.c541807782ce3p-1, 0x1.d5f3fc43e98e8p-4,
    0x0.0p+0, 0x1.afc460a7aae20p-1, 0x1.40ee7d615477fp-3, 0x0.0p+0,
    0x1.c7d2a5f705e7ep-1, 0x1.c16ad047d0c0fp-4, 0x0.0p+0, 0x1.c838a10ea6885p-1,
    0x1.be3af78acbbd5p-4, 0x0.0p+0, 0x1.bda2d764c193ap-1, 0x1.0974a26cf9b17p-3,
    0x0.0p+0, 0x1.bfccffb514ee4p-1, 0x1.00cc012bac46fp-3, 0x0.0p+0,
    0x1.b32f37017f5bcp-1, 0x1.334323fa02910p-3, 0x0.0p+0, 0x1.b57a0fcf0264cp-1,
    0x1.2a17c0c3f66cep-3, 0x0.0p+0, 0x1.bd66a3b08bf58p-1, 0x1.0a65713dd02a1p-3,
    0x0.0p+0, 0x1.b656ad0e0352dp-1, 0x1.26a54bc7f2b4bp-3, 0x0.0p+0,
    0x1.97926b87a3df6p-1, 0x1.a1b651e170826p-3, 0x0.0p+0, 0x1.c7f70f555c684p-1,
    0x1.c04785551cbe4p-4, 0x0.0p+0, 0x1.c4b79d0e6c173p-1, 0x1.da43178c9f466p-4,
    0x0.0p+0, 0x1.c48f62ea9c800p-1, 0x1.db84e8ab1c003p-4, 0x0.0p+0,
    0x1.c003a4a98c306p-1, 0x1.ffe2dab39e7ccp-4, 0x0.0p+0, 0x1.a6b7849f4decap-1,
    0x1.6521ed82c84d8p-3, 0x0.0p+0, 0x1.c78d677f70dc3p-1, 0x1.c394c404791e9p-4,
    0x0.0p+0, 0x1.97e8738715042p-1, 0x1.a05e31e3abef7p-3, 0x0.0p+0,
    0x1.b57a0fcf0264cp-1, 0x1.2a17c0c3f66cep-3, 0x0.0p+0, 0x1.c7bc878480791p-1,
    0x1.c21bc3dbfc379p-4, 0x0.0p+0, 0x1.b84f40355cdbcp-1, 0x1.1ec2ff2a8c912p-3,
    0x0.0p+0, 0x1.c0bd51ca4e3cap-1, 0x1.fa1571ad8e1b3p-4, 0x0.0p+0,
    0x1.c7ed43d2ca207p-1, 0x1.c095e169aefc7p-4, 0x0.0p+0, 0x1.c7ed43d2ca207p-1,
    0x1.c095e169aefc7p-4, 0x0.0p+0, 0x1.cdb29e1f2116ap-1, 0x1.926b0f06f74b1p-4,
    0x0.0p+0, 0x1.c5a329649fa36p-1, 0x1.d2e6b4db02e4dp-4, 0x0.0p+0,
    0x1.b24910b5dc7b8p-1, 0x1.36dbbd288e122p-3, 0x0.0p+0, 0x1.c82efd3c5f2d7p-1,
    0x1.be88161d06948p-4, 0x0.0p+0, 0x1.a772b41a9da24p-1, 0x1.62352f9589770p-3,
    0x0.0p+0, 0x1.2316b27d925e2p-1, 0x1.b9d29b04db43bp-2, 0x0.0p+0,
    0x1.39edff0ed6d2fp-1, 0x1.8c2401e2525a2p-2, 0x0.0p+0, 0x1.6d49d1675415cp-1,
    0x1.256c5d3157d49p-2, 0x0.0p+0, 0x1.20d2ebecdb732p-1, 0x1.be5a28264919bp-2,
    0x0.0p+0, 0x1.35938d76d42d0p-1, 0x1.94d8e51257a5fp-2, 0x0.0p+0,
    0x1.2841491592240p-1, 0x1.af7d6dd4dbb7fp-2, 0x0.0p+0, 0x1.3339c2a7460ccp-1,
    0x1.998c7ab173e67p-2, 0x0.0p+0, 0x1.bef4c560f6919p-1, 0x1.042cea7c25b9dp-3,
    0x0.0p+0, 0x1.c003a4a98c306p-1, 0x1.ffe2dab39e7ccp-4, 0x0.0p+0,
    0x1.b2c870887085cp-1, 0x1.34de3dde3de91p-3, 0x0.0p+0, 0x1.a83572174c979p-1,
    0x1.5f2a37a2cda1cp-3, 0x0.0p+0, 0x1.8d245eefb2ee4p-1, 0x1.cb6e844134472p-3,
    0x0.0p+0, 0x1.c813c9c5a2f7dp-1, 0x1.bf61b1d2e8417p-4, 0x0.0p+0,
    0x1.afd27bca0d753p-1, 0x1.40b610d7ca2b4p-3, 0x0.0p+0, 0x1.c7ed43d2ca207p-1,
    0x1.c095e169aefc7p-4, 0x0.0p+0, 0x1.c003a4a98c306p-1, 0x1.ffe2dab39e7ccp-4,
    0x0.0p+0, 0x1.b32f37017f5bcp-1, 0x1.334323fa02910p-3, 0x0.0p+0,
    0x1.c5741c2b9d9a8p-1, 0x1.d45f1ea3132c0p-4, 0x0.0p+0, 0x1.af3b4652fee99p-1,
    0x1.4312e6b40459dp-3, 0x0.0p+0, 0x1.c5741c2b9d9a8p-1, 0x1.d45f1ea3132c0p-4,
    0x0.0p+0, 0x1.93aa29f2b27adp-1, 0x1.b15758353614cp-3, 0x0.0p+0,
    0x1.c81ac5fd4d93fp-1, 0x1.bf29d01593607p-4, 0x0.0p+0, 0x1.bfccffb514ee4p-1,
    0x1.00cc012bac46fp-3, 0x0.0p+0, 0x1.c487ead494ed8p-1, 0x1.dbc0a95b58944p-4,
    0x0.0p+0, 0x1.97e8738715042p-1, 0x1.a05e31e3abef7p-3, 0x0.0p+0,
    0x1.c640dc2483d07p-1, 0x1.cdf91edbe17c6p-4, 0x0.0p+0, 0x1.a6d8c87e8fa0ep-1,
    0x1.649cde05c17cap-3, 0x0.0p+0, 0x1.c48f62ea9c800p-1, 0x1.db84e8ab1c003p-4,
    0x0.0p+0, 0x1.c7ed43d2ca207p-1, 0x1.c095e169aefc7p-4, 0x0.0p+0,
    0x1.c868fefad00c0p-1, 0x1.bcb808297f9ffp-4, 0x0.0p+0, 0x1.b208bc9c834acp-1,
    0x1.37dd0d8df2d51p-3, 0x0.0p+0, 0x1.c5741c2b9d9a8p-1, 0x1.d45f1ea3132c0p-4,
    0x0.0p+0, 0x1.c63522f1a4467p-1, 0x1.ce56e872ddcc6p-4, 0x0.0p+0,
    0x1.a7ac215eca44ep-1, 0x1.614f7a84d6ec7p-3, 0x0.0p+0, 0x1.be93fe17a4be4p-1,
    0x1.05b007a16d06fp-3, 0x0.0p+0, 0x1.9cc56fc083fa8p-1, 0x1.8cea40fdf0162p-3,
    0x0.0p+0, 0x1.a827148f9530ap-1, 0x1.5f63adc1ab3d9p-3, 0x0.0p+0,
    0x1.c78d677f70dc3p-1, 0x1.c394c404791e9p-4, 0x0.0p+0, 0x1.c7bc878480791p-1,
    0x1.c21bc3dbfc379p-4, 0x0.0p+0, 0x1.b50958c5c6edfp-1, 0x1.2bda9ce8e4484p-3,
    0x0.0p+0, 0x1.c78d677f70dc3p-1, 0x1.c394c404791e9p-4, 0x0.0p+0,
    0x1.b84f40355cdbcp-1, 0x1.1ec2ff2a8c912p-3, 0x0.0p+0, 0x1.c7ed43d2ca207p-1,
    0x1.c095e169aefc7p-4, 0x0.0p+0, 0x1.9cc56fc083fa8p-1, 0x1.8cea40fdf0162p-3,
    0x0.0p+0, 0x1.c60318c1da3e9p-1, 0x1.cfe739f12e0bbp-4, 0x0.0p+0,
    0x1.b2ac9544184c6p-1, 0x1.354daaef9ece6p-3, 0x0.0p+0, 0x1.bfccffb514ee4p-1,
    0x1.00cc012bac46fp-3, 0x0.0p+0, 0x1.ad80bbdaf4f4cp-1, 0x1.49fd10942c2cfp-3,
    0x0.0p+0, 0x1.c7ed43d2ca207p-1, 0x1.c095e169aefc7p-4, 0x0.0p+0,
    0x1.c0bd51ca4e3cap-1, 0x1.fa1571ad8e1b3p-4, 0x0.0p+0, 0x1.ab91d9ca9c7d4p-1,
    0x1.51b898d58e0b1p-3, 0x0.0p+0, 0x1.84a029897c908p-1, 0x1.ed7f59da0dbe2p-3,
    0x0.0p+0, 0x1.c003a4a98c306p-1, 0x1.ffe2dab39e7ccp-4, 0x0.0p+0,
    0x1.a8927e6562a78p-1, 0x1.5db6066a7561fp-3, 0x0.0p+0, 0x1.aab5b3126c365p-1,
    0x1.552933b64f26dp-3, 0x0.0p+0, 0x1.b2c870887085cp-1, 0x1.34de3dde3de91p-3,
    0x0.0p+0, 0x1.af45474e26155p-1, 0x1.42eae2c767aabp-3, 0x0.0p+0,
    0x1.c4f20d5f69ba3p-1, 0x1.d86f9504b22eap-4, 0x0.0p+0, 0x1.b77c3f4b27e9ep-1,
    0x1.220f02d360587p-3, 0x0.0p+0, 0x1.ad362049f0c5bp-1, 0x1.4b277ed83ce95p-3,
    0x0.0p+0, 0x1.c868fefad00c0p-1, 0x1.bcb808297f9ffp-4, 0x0.0p+0,
    0x1.c0efa1fc09192p-1, 0x1.f882f01fb7374p-4, 0x0.0p+0, 0x1.96aec1db5b9d7p-1,
    0x1.a544f892918a5p-3, 0x0.0p+0, 0x1.b06c1d545580dp-1, 0x1.3e4f8aaea9fcdp-3,
    0x0.0p+0, 0x1.c78d677f70dc3p-1, 0x1.c394c404791e9p-4, 0x0.0p+0,
    0x1.96d6f35701f21p-1, 0x1.a4a432a3f837cp-3, 0x0.0p+0, 0x1.a35f7900772c9p-1,
    0x1.72821bfe234dbp-3, 0x0.0p+0, 0x1.c75c6239cbae3p-1, 0x1.c51cee31a28e5p-4,
    0x0.0p+0, 0x1.cd424688f76dep-1, 0x1.95edcbb84490dp-4, 0x0.0p+0,
    0x1.c541807782ce3p-1, 0x1.d5f3fc43e98e8p-4, 0x0.0p+0, 0x1.c813c9c5a2f7dp-1,
    0x1.bf61b1d2e8417p-4, 0x0.0p+0, 0x1.bff3c82c44cb3p-1, 0x1.0030df4eecd35p-3,
    0x0.0p+0, 0x1.b32f37017f5bcp-1, 0x1.334323fa02910p-3, 0x0.0p+0,
    0x1.c5741c2b9d9a8p-1, 0x1.d45f1ea3132c0p-4, 0x0.0p+0, 0x1.9a198cead5599p-1,
    0x1.9799cc54aa99cp-3, 0x0.0p+0, 0x1.c7f70f555c684p-1, 0x1.c04785551cbe4p-4,
    0x0.0p+0, 0x1.c48f62ea9c800p-1, 0x1.db84e8ab1c003p-4, 0x0.0p+0,
    0x1.9e9f60b7ec600p-1, 0x1.85827d204e802p-3, 0x0.0p+0, 0x1.a6d8c87e8fa0ep-1,
    0x1.649cde05c17cap-3, 0x0.0p+0, 0x1.afc460a7aae20p-1, 0x1.40ee7d615477fp-3,
    0x0.0p+0, 0x1.8fd24eae53d5ep-1, 0x1.c0b6c546b0a87p-3, 0x0.0p+0,
    0x1.c7d2a5f705e7ep-1, 0x1.c16ad047d0c0fp-4, 0x0.0p+0, 0x1.c81ac5fd4d93fp-1,
    0x1.bf29d01593607p-4, 0x0.0p+0, 0x1.c4c28402f0d92p-1, 0x1.d9ebdfe879373p-4,
    0x0.0p+0, 0x1.a74201f7c96a1p-1, 0x1.62f7f820da57bp-3, 0x0.0p+0,
    0x1.c7f70f555c684p-1, 0x1.c04785551cbe4p-4, 0x0.0p+0, 0x1.a98e8ec7e1092p-1,
    0x1.59c5c4e07bdb6p-3, 0x0.0p+0, 0x1.b288730fae6e3p-1, 0x1.35de33c146475p-3,
    0x0.0p+0, 0x1.93aa29f2b27adp-1, 0x1.b15758353614cp-3, 0x0.0p+0,
    0x1.c7bc878480791p-1, 0x1.c21bc3dbfc379p-4, 0x0.0p+0, 0x1.a6b7849f4decap-1,
    0x1.6521ed82c84d8p-3, 0x0.0p+0, 0x1.c5741c2b9d9a8p-1, 0x1.d45f1ea3132c0p-4,
    0x0.0p+0, 0x1.8f7cd78dd47f8p-1, 0x1.c20ca1c8ae022p-3, 0x0.0p+0,
    0x1.c0bd51ca4e3cap-1, 0x1.fa1571ad8e1b3p-4, 0x0.0p+0, 0x1.c7ed43d2ca207p-1,
    0x1.c095e169aefc7p-4, 0x0.0p+0, 0x1.c48f62ea9c800p-1, 0x1.db84e8ab1c003p-4,
    0x0.0p+0, 0x1.c5a329649fa36p-1, 0x1.d2e6b4db02e4dp-4, 0x0.0p+0,
    0x1.c7d2a5f705e7ep-1, 0x1.c16ad047d0c0fp-4, 0x0.0p+0, 0x1.c75c6239cbae3p-1,
    0x1.c51cee31a28e5p-4, 0x0.0p+0, 0x1.c670b0b327d35p-1, 0x1.cc7a7a66c1656p-4,
    0x0.0p+0, 0x1.c868fefad00c0p-1, 0x1.bcb808297f9ffp-4, 0x0.0p+0,
    0x1.ac5288ac7f12cp-1, 0x1.4eb5dd4e03b52p-3, 0x0.0p+0, 0x1.a767470ff9aafp-1,
    0x1.6262e3c019543p-3, 0x0.0p+0, 0x1.c08734f073cd3p-1, 0x1.fbc6587c61966p-4,
    0x0.0p+0, 0x1.c22ad118b40c8p-1, 0x1.eea9773a5f9bdp-4, 0x0.0p+0,
    0x1.b27bedbecb043p-1, 0x1.36104904d3ef3p-3, 0x0.0p+0, 0x1.c81ac5fd4d93fp-1,
    0x1.bf29d01593607p-4, 0x0.0p+0, 0x1.c5a329649fa36p-1, 0x1.d2e6b4db02e4dp-4,
    0x0.0p+0, 0x1.c7ed43d2ca207p-1, 0x1.c095e169aefc7p-4, 0x0.0p+0,
    0x1.b32f37017f5bcp-1, 0x1.334323fa02910p-3, 0x0.0p+0, 0x1.c003a4a98c306p-1,
    0x1.ffe2dab39e7ccp-4, 0x0.0p+0, 0x1.592a99e684adap-1, 0x1.4daacc32f6a4cp-2,
    0x0.0p+0, 0x1.9298c09dab604p-1, 0x1.b59cfd89527f2p-3, 0x0.0p+0,
    0x1.c0efa1fc09192p-1, 0x1.f882f01fb7374p-4, 0x0.0p+0, 0x1.c7ed43d2ca207p-1,
    0x1.c095e169aefc7p-4, 0x0.0p+0, 0x1.949ece1105c67p-1, 0x1.ad84c7bbe8e63p-3,
    0x0.0p+0, 0x1.c4549dd34b820p-1, 0x1.dd5b1165a3f04p-4, 0x0.0p+0,
    0x1.c0bd51ca4e3cap-1, 0x1.fa1571ad8e1b3p-4, 0x0.0p+0, 0x1.c78d677f70dc3p-1,
    0x1.c394c404791e9p-4, 0x0.0p+0, 0x1.c026a1d9a214fp-1, 0x1.fecaf132ef58bp-4,
    0x0.0p+0, 0x1.c78d677f70dc3p-1, 0x1.c394c404791e9p-4, 0x0.0p+0,
    0x1.b5b3d97b79f36p-1, 0x1.29309a1218326p-3, 0x0.0p+0, 0x1.b7619a860c7a3p-1,
    0x1.227995e7ce173p-3, 0x0.0p+0, 0x1.c5a329649fa36p-1, 0x1.d2e6b4db02e4dp-4,
    0x0.0p+0, 0x1.c5a329649fa36p-1, 0x1.d2e6b4db02e4dp-4, 0x0.0p+0,
    0x1.b7619a860c7a3p-1, 0x1.227995e7ce173p-3, 0x0.0p+0, 0x1.be5c4d2982bf4p-1,
    0x1.068ecb59f5031p-3, 0x0.0p+0, 0x1.c541807782ce3p-1, 0x1.d5f3fc43e98e8p-4,
    0x0.0p+0, 0x1.b4956b18c7a81p-1, 0x1.2daa539ce15fcp-3, 0x0.0p+0,
    0x1.c78d677f70dc3p-1, 0x1.c394c404791e9p-4, 0x0.0p+0, 0x1.a949a14f93dafp-1,
    0x1.5ad97ac1b0944p-3, 0x0.0p+0, 0x1.bfbd179d8112bp-1, 0x1.010ba189fbb54p-3,
    0x0.0p+0, 0x1.c5a329649fa36p-1, 0x1.d2e6b4db02e4dp-4, 0x0.0p+0,
    0x1.ae70cb8c8c67cp-1, 0x1.463cd1cdce60fp-3, 0x0.0p+0, 0x1.c838a10ea6885p-1,
    0x1.be3af78acbbd5p-4, 0x0.0p+0, 0x1.ad751d26fb4c2p-1, 0x1.4a2b8b6412cf9p-3,
    0x0.0p+0, 0x1.a7c6a23830802p-1, 0x1.60e5771f3dffap-3, 0x0.0p+0,
    0x1.a87136d3b6ffep-1, 0x1.5e3b24b124008p-3, 0x0.0p+0, 0x1.c4c28402f0d92p-1,
    0x1.d9ebdfe879373p-4, 0x0.0p+0, 0x1.bfbd179d8112bp-1, 0x1.010ba189fbb54p-3,
    0x0.0p+0, 0x1.c7ed43d2ca207p-1, 0x1.c095e169aefc7p-4, 0x0.0p+0,
    0x1.a7ac215eca44ep-1, 0x1.614f7a84d6ec7p-3, 0x0.0p+0, 0x1.c838a10ea6885p-1,
    0x1.be3af78acbbd5p-4, 0x0.0p+0, 0x1.a87136d3b6ffep-1, 0x1.5e3b24b124008p-3,
    0x0.0p+0, 0x1.c4b79d0e6c173p-1, 0x1.da43178c9f466p-4, 0x0.0p+0,
    0x1.c0bd51ca4e3cap-1, 0x1.fa1571ad8e1b3p-4, 0x0.0p+0, 0x1.a7e4dffea3e65p-1,
    0x1.606c80057066cp-3, 0x0.0p+0, 0x1.c541807782ce3p-1, 0x1.d5f3fc43e98e8p-4,
    0x0.0p+0, 0x1.ad63246f8bd2cp-1, 0x1.4a736e41d0b51p-3, 0x0.0p+0,
    0x1.b77c3f4b27e9ep-1, 0x1.220f02d360587p-3, 0x0.0p+0, 0x1.c48f62ea9c800p-1,
    0x1.db84e8ab1c003p-4, 0x0.0p+0, 0x1.90d51da671117p-1, 0x1.bcab89663bba5p-3,
    0x0.0p+0, 0x1.b2ac9544184c6p-1, 0x1.354daaef9ece6p-3, 0x0.0p+0,
    0x1.c82efd3c5f2d7p-1, 0x1.be88161d06948p-4, 0x0.0p+0, 0x1.b53beb8e26448p-1,
    0x1.2b1051c766ee0p-3, 0x0.0p+0, 0x1.b29fd9d1e3cc3p-1, 0x1.358098b870cf4p-3,
    0x0.0p+0, 0x1.c80351770bdaep-1, 0x1.bfe57447a128dp-4, 0x0.0p+0,
    0x1.c08734f073cd3p-1, 0x1.fbc6587c61966p-4, 0x0.0p+0, 0x1.c5741c2b9d9a8p-1,
    0x1.d45f1ea3132c0p-4, 0x0.0p+0, 0x1.b32086e7355dap-1, 0x1.337de4632a896p-3,
    0x0.0p+0, 0x1.c7ed43d2ca207p-1, 0x1.c095e169aefc7p-4, 0x0.0p+0,
    0x1.7c0b4aec7731dp-1, 0x1.07e96a27119c6p-2, 0x0.0p+0, 0x1.c5a329649fa36p-1,
    0x1.d2e6b4db02e4dp-4, 0x0.0p+0, 0x1.a87136d3b6ffep-1, 0x1.5e3b24b124008p-3,
    0x0.0p+0, 0x1.be879adf0cd08p-1, 0x1.05e19483ccbe1p-3, 0x0.0p+0,
    0x1.b27bedbecb043p-1, 0x1.36104904d3ef3p-3, 0x0.0p+0, 0x1.c5741c2b9d9a8p-1,
    0x1.d45f1ea3132c0p-4, 0x0.0p+0, 0x1.a5391e1acac95p-1, 0x1.6b1b8794d4dabp-3,
    0x0.0p+0, 0x1.a3e1b50d34ca1p-1, 0x1.70792bcb2cd7dp-3, 0x0.0p+0,
    0x1.c0bd51ca4e3cap-1, 0x1.fa1571ad8e1b3p-4, 0x0.0p+0, 0x1.96d6f35701f21p-1,
    0x1.a4a432a3f837cp-3, 0x0.0p+0, 0x1.855fd069bae4ep-1, 0x1.ea80be59146c9p-3,
    0x0.0p+0, 0x1.c0efa1fc09192p-1, 0x1.f882f01fb7374p-4, 0x0.0p+0,
    0x1.c48f62ea9c800p-1, 0x1.db84e8ab1c003p-4, 0x0.0p+0, 0x1.c80351770bdaep-1,
    0x1.bfe57447a128dp-4, 0x0.0p+0, 0x1.a83572174c979p-1, 0x1.5f2a37a2cda1cp-3,
    0x0.0p+0, 0x1.b73f807ebda97p-1, 0x1.2301fe05095a4p-3, 0x0.0p+0,
    0x1.b7b414e99afa8p-1, 0x1.212fac5994161p-3, 0x0.0p+0, 0x1.a4782135c61c4p-1,
    0x1.6e1f7b28e78efp-3, 0x0.0p+0, 0x1.c813c9c5a2f7dp-1, 0x1.bf61b1d2e8417p-4,
    0x0.0p+0, 0x1.c4c28402f0d92p-1, 0x1.d9ebdfe879373p-4, 0x0.0p+0,
    0x1.9833ba3ff55aap-1, 0x1.9f3117002a958p-3, 0x0.0p+0, 0x1.8ec7daacc193cp-1,
    0x1.c4e0954cf9b11p-3, 0x0.0p+0, 0x1.c868fefad00c0p-1, 0x1.bcb808297f9ffp-4,
    0x0.0p+0, 0x1.a48c6a9ab8805p-1, 0x1.6dce55951dfebp-3, 0x0.0p+0,
    0x1.b77c3f4b27e9ep-1, 0x1.220f02d360587p-3, 0x0.0p+0, 0x1.c003a4a98c306p-1,
    0x1.ffe2dab39e7ccp-4, 0x0.0p+0, 0x1.b3e6a3481c546p-1, 0x1.306572df8eae7p-3,
    0x0.0p+0, 0x1.bff3c82c44cb3p-1, 0x1.0030df4eecd35p-3, 0x0.0p+0,
};

static int churn_selftest(void) {
    churn_table t;
    double* out;
    char msg[512];
    char* copy[CHURN_ST_ROWS + 1];
    int64_t i, bad = 0, badown = 0, n;
    int s, k;
    double worst = 0.0;
    for (i = 0; i <= CHURN_ST_ROWS; i++) {
        size_t m = strlen(churn_st_csv[i]) + 1;
        copy[i] = (char*)malloc(m);
        memcpy(copy[i], churn_st_csv[i], m);
    }
    s = churn_read(copy, CHURN_ST_ROWS + 1, 0, &t, "selftest");
    if (s) return s;
    n = t.rows * CHURN_NOUT;
    out = (double*)malloc((size_t)(n + 1) * sizeof(double));
    s = churn_predict_all(t.xf, t.xi, t.rows, out, msg, sizeof msg);
    if (s) { fprintf(stderr, "churn: selftest: the model stopped: %s\n", msg); return 4; }
    for (i = 0; i < n; i++) {
        double e = churn_st_expect[i], g = out[i], d = g > e ? g - e : e - g;
        int same = memcmp(&e, &g, sizeof e) == 0 || (e != e && g != g);
        if (!same && !(d <= churn_st_tol)) {
            if (bad < 5) fprintf(stderr, "churn: selftest: row %ld, output %ld: %.17g, scikit-learn gave %.17g\n",
                                 (long)(i / CHURN_NOUT), (long)(i % CHURN_NOUT), g, e);
            bad++;
        }
        if (!same && d > worst) worst = d;
        if (memcmp(&g, &churn_st_own[i], sizeof g) != 0 && !(g != g && churn_st_own[i] != churn_st_own[i])) badown++;
    }
    for (k = 0; k <= CHURN_ST_ROWS; k++) free(copy[k]);
    if (bad || badown) {
        fprintf(stdout, "selftest FAILED: %ld of %ld values differ from scikit-learn's beyond %g, %ld from this file's own\n",
               (long)bad, (long)n, churn_st_tol, (long)badown);
        return 1;
    }
    if (churn_st_tol == 0.0)
        fprintf(stdout, "selftest passed: %ld rows, %ld values, every one bit-identical to scikit-learn's\n", (long)t.rows, (long)n);
    else
        fprintf(stdout, "selftest passed: %ld rows, %ld values; largest difference from scikit-learn %.3g (allowed %g), "
               "and bit-identical to the values recorded at compile time\n", (long)t.rows, (long)n, worst, churn_st_tol);
    free(out); free(t.xf); free(t.xi);
    return 0;
}

static const char* churn_usage =
    "usage: churn [--ignore-extra] < input.csv > predictions.csv\n"
    "       churn --selftest     re-run the embedded check rows against scikit-learn's outputs\n"
    "       churn --schema       print the input columns as JSON\n"
    "exit status: 0 ok; 1 selftest failed; 2 wrong or missing columns; 3 a bad value; 4 the model refused a row\n";

/* the export renames a program's main and poisons printf: this file's own main is the command line */
#undef main
int main(int argc, char** argv) {
    int a, ignore_extra = 0, s;
    size_t nl = 0;
    char* buf = 0;
    char** lines;
    churn_table t;
    double* out;
    char msg[512];
    int64_t i;
    int k;
    for (a = 1; a < argc; a++) {
        if (strcmp(argv[a], "--selftest") == 0) return churn_selftest();
        if (strcmp(argv[a], "--schema") == 0) { fprintf(stdout, "%s\n", churn_schema); return 0; }
        if (strcmp(argv[a], "--ignore-extra") == 0) { ignore_extra = 1; continue; }
        fputs(churn_usage, strcmp(argv[a], "--help") == 0 ? stdout : stderr);
        return strcmp(argv[a], "--help") == 0 ? 0 : 2;
    }
    lines = churn_lines(stdin, &nl, &buf);
    if (!lines) { fprintf(stderr, "churn: cannot read stdin\n"); return 3; }
    s = churn_read(lines, nl, ignore_extra, &t, "stdin");
    if (s) return s;
    out = (double*)malloc((size_t)(t.rows * CHURN_NOUT + 1) * sizeof(double));
    s = churn_predict_all(t.xf, t.xi, t.rows, out, msg, sizeof msg);
    if (s) { fprintf(stderr, "churn: %s\n", msg); return 4; }
    if (CHURN_NCLASSES > 0) {
        fprintf(stdout, "prediction");
        for (k = 0; k < CHURN_NCLASSES; k++) fprintf(stdout, ",proba_%s", churn_classes[k]);
        fprintf(stdout, "\n");
        for (i = 0; i < t.rows; i++) {
            fprintf(stdout, "%s", churn_classes[(int)out[i * CHURN_NOUT + CHURN_NCLASSES]]);
            for (k = 0; k < CHURN_NCLASSES; k++) fprintf(stdout, ",%.17g", out[i * CHURN_NOUT + k]);
            fprintf(stdout, "\n");
        }
    } else {
        fprintf(stdout, "prediction\n");
        for (i = 0; i < t.rows; i++) fprintf(stdout, "%.17g\n", out[i]);
    }
    return 0;
}
#endif
