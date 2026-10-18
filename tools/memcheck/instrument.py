#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Append a memory report to a scratch copy of a Viv-generated C file.

    tools/memcheck/instrument.py prog.c > prog.mc.c

The generated C is not changed; a destructor is added at its end. When the
program returns from main (a run that stops with an error calls _exit and
reports nothing), the destructor writes one line to the file named by
$VIV_MEMSTAT:

    arena_n arena_bytes live_n live_bytes zone_in_use max_rss

- arena_n, arena_bytes: the blocks still on the array arena at exit and their
  malloc_size total (0 0 when the program has no arena). A map is filed with
  a tag in its pointer's low bits (interp/maps.rs); the tag is taken off, and
  the map counts as its header block;
- live_n, live_bytes: the part of those reachable from the program's globals
  (its top-level variables), found by a conservative mark: every
  pointer-sized word of a global, and then of every block marked so far, that
  equals the start of an arena block marks it. What is on the arena and not
  reachable is garbage the run kept until it ended;
- zone_in_use: the malloc zones' bytes in use at exit
  (malloc_zone_statistics), which includes blocks that are not on the arena:
  maps, strings stored in maps, the arena's own table, stdio's buffers;
- max_rss: the process's peak resident set (getrusage: bytes on macOS,
  kilobytes on Linux).

On Linux, malloc_usable_size and mallinfo2 stand in for macOS's malloc_size
and malloc_zone_statistics.

The globals are read from the C text: every line at column 0 of the form
`TYPE NAME;` (the compiler declares each top-level variable that way).
"""
import re
import sys

src = open(sys.argv[1]).read()
has_arena = "static VIV_TLS size_t viv_arena_top" in src
globs = re.findall(r"^([A-Za-z_][A-Za-z_0-9]*\*?) ([A-Za-z_][A-Za-z_0-9]*);$", src, re.M)
roots = ",\n    ".join(f"{{(const void*)&{name}, sizeof {name}}}" for _, name in globs)
if not roots:
    roots = "{0, 0}"

out = [src, r"""
/* ---- memcheck instrumentation (tools/memcheck/instrument.py) ---- */
#ifdef __APPLE__
#include <malloc/malloc.h>
#define VIV_MC_SIZE(p) malloc_size(p)
#else
/* glibc: the same two numbers through its own calls */
#include <malloc.h>
#define VIV_MC_SIZE(p) malloc_usable_size((void*)(p))
#endif
#include <stdint.h>
#include <sys/resource.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static const struct { const void* p; size_t n; } viv_mc_roots[] = {
    """ + roots + r"""
};
"""]
if has_arena:
    out.append(r"""
/* a map is filed with a tag in its low bits (interp/maps.rs) */
#define VIV_MC_UNTAG(p) ((uintptr_t)(p) & ~(uintptr_t)3)
/* the arena's blocks sorted by address, for a binary search per word */
static struct viv_mc_ent { uintptr_t a; size_t i; }* viv_mc_sorted;
static int viv_mc_cmp(const void* x, const void* y) {
    uintptr_t a = ((const struct viv_mc_ent*)x)->a, b = ((const struct viv_mc_ent*)y)->a;
    return a < b ? -1 : a > b;
}
static int viv_mc_find(uintptr_t w) {
    size_t lo = 0, hi = viv_arena_top;
    while (lo < hi) {
        size_t m = (lo + hi) / 2;
        if (viv_mc_sorted[m].a < w) lo = m + 1; else hi = m;
    }
    if (lo < viv_arena_top && viv_mc_sorted[lo].a == w) return (int)viv_mc_sorted[lo].i;
    return -1;
}
static void viv_mc_scan(const void* p, size_t n, char* mark, size_t* stack, size_t* sp) {
    const unsigned char* b = (const unsigned char*)p;
    for (size_t o = 0; o + sizeof(uintptr_t) <= n; o += sizeof(uintptr_t)) {
        uintptr_t w; memcpy(&w, b + o, sizeof w);
        if (w < 4096) continue;
        int k = viv_mc_find(w);
        if (k >= 0 && !mark[k]) { mark[k] = 1; stack[(*sp)++] = (size_t)k; }
    }
}
""")
out.append(r"""
__attribute__((destructor)) static void viv_mc_report(void) {
    const char* path = getenv("VIV_MEMSTAT");
    if (!path) return;
    size_t an = 0, ab = 0, ln = 0, lb = 0;
""")
if has_arena:
    out.append(r"""
    an = viv_arena_top;
    char* mark = (char*)calloc(an + 1, 1);
    size_t* stack = (size_t*)malloc((an + 1) * sizeof(size_t));
    size_t sp = 0;
    viv_mc_sorted = (struct viv_mc_ent*)malloc((an + 1) * sizeof *viv_mc_sorted);
    for (size_t i = 0; i < an; i++) { viv_mc_sorted[i].a = VIV_MC_UNTAG(viv_arena[i]); viv_mc_sorted[i].i = i; }
    qsort(viv_mc_sorted, an, sizeof *viv_mc_sorted, viv_mc_cmp);
    for (size_t r = 0; r < sizeof viv_mc_roots / sizeof viv_mc_roots[0]; r++)
        if (viv_mc_roots[r].p) viv_mc_scan(viv_mc_roots[r].p, viv_mc_roots[r].n, mark, stack, &sp);
    while (sp > 0) {
        size_t k = stack[--sp];
        viv_mc_scan((const void*)VIV_MC_UNTAG(viv_arena[k]), VIV_MC_SIZE((const void*)VIV_MC_UNTAG(viv_arena[k])), mark, stack, &sp);
    }
    for (size_t i = 0; i < an; i++) {
        size_t s = VIV_MC_SIZE((const void*)VIV_MC_UNTAG(viv_arena[i]));
        ab += s;
        if (mark[i]) { ln++; lb += s; }
    }
    free(mark); free(stack); free(viv_mc_sorted);
""")
out.append(r"""
#ifdef __APPLE__
    malloc_statistics_t st; memset(&st, 0, sizeof st);
    malloc_zone_statistics(NULL, &st);
    size_t in_use = (size_t)st.size_in_use;
#else
    struct mallinfo2 mi = mallinfo2();
    size_t in_use = mi.uordblks + mi.hblkhd;
#endif
    FILE* f = fopen(path, "w");
    if (!f) return;
    struct rusage ru; memset(&ru, 0, sizeof ru);
    getrusage(RUSAGE_SELF, &ru);
    fprintf(f, "%zu %zu %zu %zu %zu %ld\n", an, ab, ln, lb, in_use, (long)ru.ru_maxrss);
    fclose(f);
}
""")
sys.stdout.write("".join(out))
