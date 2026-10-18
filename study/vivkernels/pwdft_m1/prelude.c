/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* PROTOTYPE prelude, hand-written: what `viv export` would put before the
   program. stdout lines go to a callback, the runtime's exit sites become a
   longjmp back to pwdft_run, and stderr text becomes the error message. */
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
#include "pwdft.h"

static pwdft_params vk_param;
static jmp_buf *vk_jmp = 0;
static char vk_msg[2048];
static size_t vk_msg_n = 0;
static char vk_line[1 << 16];
static size_t vk_line_n = 0;
static int vk_npass = 0, vk_nfail = 0;

__attribute__((noreturn)) static void vk_bail(int code) {
    if (vk_jmp) longjmp(*vk_jmp, code ? code : PWDFT_ERROR);
    _exit(code);
}
static void vk_emit_line(void) {
    vk_line[vk_line_n] = 0;
    if (!strncmp(vk_line, "PASS ", 5)) vk_npass++;
    if (!strncmp(vk_line, "FAIL ", 5)) vk_nfail++;
    vk_line_n = 0;
    if (vk_param.on_line && vk_param.on_line(vk_line, vk_param.ctx)) vk_bail(PWDFT_INTERRUPTED);
}
static void vk_out(const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '\n') { vk_emit_line(); continue; }
        if (vk_line_n < sizeof vk_line - 1) vk_line[vk_line_n++] = s[i];
    }
}
static int vk_printf(const char *fmt, ...) {
    char b[4096];
    va_list ap; va_start(ap, fmt);
    int r = vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    if (r > 0) vk_out(b, (size_t)r < sizeof b ? (size_t)r : sizeof b - 1);
    return r;
}
static int vk_fputs(const char *s, FILE *f) {
    if (f == stdout) { vk_out(s, strlen(s)); return 0; }
    return fputs(s, f);
}
static int vk_puts(const char *s) { vk_out(s, strlen(s)); vk_out("\n", 1); return 0; }
static int vk_putchar(int c) { char ch = (char)c; vk_out(&ch, 1); return c; }
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
#define exit(c) vk_bail(c)
#define printf vk_printf
#define fputs vk_fputs
#define puts vk_puts
#define putchar vk_putchar
#define fprintf vk_fprintf
#define main pwdft_program_main
