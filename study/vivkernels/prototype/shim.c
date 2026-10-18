/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* PROTOTYPE, hand-written, not generated: the error boundary and the exported
   entry proposed for `viv export` to emit.
   It wraps the unmodified C that main's `viv axpy.viv -o axpy.c` writes:
   the runtime's `_exit(1)` becomes a longjmp back to the entry, and what the
   runtime writes to stderr is captured as the message. */
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <setjmp.h>

static jmp_buf *vk_jmp = 0;
static char vk_msg[1024];
static size_t vk_msg_n = 0;

static void vk_bail(int code) {
    if (vk_jmp) longjmp(*vk_jmp, code ? code : 1);
    _exit(code);
}
static int vk_fprintf(FILE *f, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    int r;
    if (f == stderr && vk_jmp) {
        r = vsnprintf(vk_msg + vk_msg_n, sizeof vk_msg - vk_msg_n, fmt, ap);
        if (r > 0) vk_msg_n += (size_t)r < sizeof vk_msg - vk_msg_n ? (size_t)r : sizeof vk_msg - vk_msg_n - 1;
    } else r = vfprintf(f, fmt, ap);
    va_end(ap);
    return r;
}
#define _exit(c) vk_bail(c)
#define fprintf vk_fprintf
#define main viv_program_main
#include "axpy.c"
#undef main

/* The exported ABI: arrays as (data, rank, dims as int64), scalars by value,
   the scalar result through an out-pointer, and a status (0 = ok). */
typedef struct { double *data; int64_t rank; const int64_t *dims; } vk_f64;

static int vk_view(vk_f64 v, viv_arr *a) {
    long long n = 1;
    if (v.rank < 1 || v.rank > 8) return 0;
    a->rank = (int)v.rank;
    for (int k = 0; k < 8; k++) a->dims[k] = 0;
    for (int k = 0; k < v.rank; k++) {
        if (v.dims[k] < 0 || v.dims[k] > INT_MAX) return 0;
        a->dims[k] = (int)v.dims[k];
        n *= v.dims[k];
        if (n > INT_MAX) return 0;
    }
    a->size = (int)n;
    a->data = v.data;                 /* borrowed: no copy, never pushed on the arena */
    return 1;
}

int vk_axpy_v0(double a, vk_f64 x, vk_f64 y, double *out, char *msg, size_t cap) {
    viv_arr xa, ya;
    if (!vk_view(x, &xa) || !vk_view(y, &ya)) { snprintf(msg, cap, "error: shape not representable"); return 2; }
    jmp_buf here;
    size_t mark = viv_arena_top;
    vk_msg_n = 0; vk_msg[0] = 0;
    vk_jmp = &here;
    int code = setjmp(here);
    if (code == 0) {
        *out = axpy(a, xa, ya);
        vk_jmp = 0;
        viv_arena_release(mark, 0);
        return 0;
    }
    vk_jmp = 0;
    viv_arena_release(mark, 0);       /* free what the failed call allocated */
    snprintf(msg, cap, "%s", vk_msg);
    return 1;
}
