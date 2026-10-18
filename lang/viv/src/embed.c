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
   then _exit(4), or exit(4), which runs the failed-assert summary (4 is
   a run-time error in every engine; a failed test exits 1). This part
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
/* B344: an error inside a test block FAILs the test before its message.
   Every run-time error flushes stdout first, so the test runtime sets this
   hook while a test runs (viv_test_begin) and the flush calls it once. */
VIV_UNUSED static void (*viv_fatal_hook)(void) = 0;
VIV_UNUSED static int viv_fatal_run(void) {
    void (*h)(void) = viv_fatal_hook;
    viv_fatal_hook = 0;
    h();
    return 0;
}
#define viv_fflush(f) ((viv_fatal_hook && (f) == stdout ? viv_fatal_run() : 0), fflush(f))
#define viv_errf(...) fprintf(stderr, __VA_ARGS__)
/* exit 4: a run-time error, as `viv run` exits (a
   failed test exits 1) */
#define viv_stop(kind) _exit(4)
/* The failed-assert and failed-hold summaries run at exit and end it with
   1, a failed test, unless a run-time error began the exit: then 4 stays. */
VIV_UNUSED static int viv_exit_status = 1;
#define viv_stop_exit(kind) do { viv_exit_status = 4; exit(4); } while (0)
#define viv_stop_failed() _exit(viv_exit_status)
#define viv_stop_exit_failed() exit(viv_exit_status)
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
#define viv_stop_failed() viv_stop(VIV_E_ERROR)
#define viv_stop_exit_failed() viv_stop(VIV_E_ERROR)
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
