/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* Strings: the C twin of lang/viv/src/interp/strings.rs, function for
   function. Cut into runtime fragments at the `//@ name:
   deps` lines by lower/strings.rs; the character tables (`viv_t_*`) are
   printed from interp/unicode_tables.rs. Everything is `static` and marked
   unused, so a program pays only for the fragments it calls. */
//@ str_core: arena arena_keep sites fmt_g
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
//@ str_list: str_core
/* a list with room for `cap` items, on the arena */
VIV_S_UNUSED static viv_strs viv_ss_alloc(long cap) {
    if (cap < 1) cap = 1;
    viv_strs l = (viv_strs)malloc(sizeof(viv_strsb) + (size_t)cap * sizeof(viv_str));
    if (!l) viv_s_nomem();
    l->n = 0; l->cap = (int)cap; l->unsealed = 0; l->pad = 0; l->chunk = 0;
    viv_arena_push(l);
    return l;
}
VIV_S_UNUSED static viv_strs viv_ss_new(void) { return viv_ss_alloc(1); }
/* file `nw` where the arena had `old` */
VIV_S_UNUSED static void viv_ss_refile(void* old, void* nw) {
    for (size_t i = viv_arena_top; i-- > 0; ) {
        if (viv_arena[i] == old) { viv_arena[i] = nw; return; }
    }
    viv_arena_push(nw);
}
/* a copy of `s` stored in l's chunks (a new chunk, twice the last, when it
   has no room); returns the copy */
VIV_S_UNUSED static viv_str viv_ss_store(viv_strs l, viv_str s) {
    long need = (long)(sizeof(viv_strb) + (size_t)s->len + 1);
    need = (need + 7) & ~7L;
    viv_chunk* c = l->chunk;
    if (!c || c->cap - c->used < need) {
        long cap = c ? 2 * c->cap : 256;
        if (cap < need) cap = need;
        viv_chunk* nc = (viv_chunk*)malloc(sizeof(viv_chunk) + (size_t)cap);
        if (!nc) viv_s_nomem();
        nc->prev = c; nc->cap = cap; nc->used = 0; nc->pad = 0;
        viv_arena_push(nc);
        l->chunk = c = nc;
    }
    viv_str r = (viv_str)(void*)(c->data + c->used);
    c->used += need;
    r->owner = c; r->len = s->len; r->nch = s->nch; r->ci = 0; r->cb = 0;
    memcpy(r->s, s->s, (size_t)s->len);
    r->s[s->len] = 0;
    return r;
}
/* a new list of the strings v[0..n), each copied into one chunk */
VIV_S_UNUSED static viv_strs viv_ss_copyof(viv_str* v, long n, long room) {
    viv_strs l = viv_ss_alloc(n + room);
    long bytes = 0;
    for (long k = 0; k < n; k++) bytes += ((long)(sizeof(viv_strb) + (size_t)v[k]->len + 1) + 7) & ~7L;
    if (bytes) {
        viv_chunk* c = (viv_chunk*)malloc(sizeof(viv_chunk) + (size_t)(room ? 2 * bytes : bytes));
        if (!c) viv_s_nomem();
        c->prev = 0; c->cap = room ? 2 * bytes : bytes; c->used = 0; c->pad = 0;
        viv_arena_push(c);
        l->chunk = c;
    }
    for (long k = 0; k < n; k++) l->items[l->n++] = viv_ss_store(l, v[k]);
    return l;
}
VIV_S_UNUSED static viv_strs viv_ss_or_new(viv_strs l) { return l ? l : viv_ss_new(); }
#define VIV_SS(l) viv_ss_or_new(l)
VIV_S_UNUSED static viv_strs viv_ss_of(int n, ...) {
    viv_str* v = (viv_str*)malloc(sizeof(viv_str) * (size_t)(n ? n : 1));
    if (!v) viv_s_nomem();
    va_list ap;
    va_start(ap, n);
    for (int k = 0; k < n; k++) v[k] = VIV_S(va_arg(ap, viv_str));
    va_end(ap);
    viv_strs l = viv_ss_copyof(v, n, 0);
    free(v);
    return l;
}
VIV_S_UNUSED static double viv_ss_len(viv_strs l) { return l ? (double)l->n : 0.0; }
VIV_S_UNUSED static viv_str viv_ss_at(viv_strs l, double i, int site) {
    viv_s_whole(i, site);
    long k = viv_s_pyindex(l ? l->n : 0, i);
    if (k < 0) viv_s_fail(VIV_E_INDEX, "list index out of range", site);
    return l->items[k];
}
VIV_S_UNUSED static viv_strs viv_ss_slice(viv_strs l, double kind, double lo, double hi, double step, int site) {
    l = VIV_SS(l);
    long start;
    long long st;
    long count = viv_s_pyslice(l->n, kind, lo, hi, step, &start, &st, site);
    viv_str* v = (viv_str*)malloc(sizeof(viv_str) * (size_t)(count ? count : 1));
    if (!v) viv_s_nomem();
    for (long k = 0; k < count; k++) v[k] = l->items[start + k * st];
    viv_strs r = viv_ss_copyof(v, count, 0);
    free(v);
    return r;
}
VIV_S_UNUSED static viv_strs viv_ss_cat(viv_strs a, viv_strs b) {
    a = VIV_SS(a); b = VIV_SS(b);
    long n = (long)a->n + b->n;
    viv_str* v = (viv_str*)malloc(sizeof(viv_str) * (size_t)(n ? n : 1));
    if (!v) viv_s_nomem();
    for (int k = 0; k < a->n; k++) v[k] = a->items[k];
    for (int k = 0; k < b->n; k++) v[a->n + k] = b->items[k];
    viv_strs r = viv_ss_copyof(v, n, 0);
    free(v);
    return r;
}
VIV_S_UNUSED static double viv_ss_eq(viv_strs a, viv_strs b) {
    a = VIV_SS(a); b = VIV_SS(b);
    if (a->n != b->n) return 0.0;
    for (int k = 0; k < a->n; k++) if (viv_s_eq(a->items[k], b->items[k]) == 0.0) return 0.0;
    return 1.0;
}
VIV_S_UNUSED static viv_strs viv_ss_seal(viv_strs l) { if (l) l->unsealed = 0; return l; }
/* append(l, s) (own: the compiler proved l's variable its only holder, and
   l is unsealed: grown in place, room doubling), else a copy with room */
VIV_S_UNUSED static viv_strs viv_ss_grow(viv_strs l, viv_str* add, long m, double own) {
    l = VIV_SS(l);
    if (own != 0.0 && l->unsealed) {
        if (l->n + m > l->cap) {
            long cap = 2L * l->cap;
            if (cap < l->n + m) cap = l->n + m;
            viv_strs q = (viv_strs)realloc(l, sizeof(viv_strsb) + (size_t)cap * sizeof(viv_str));
            if (!q) viv_s_nomem();
            viv_ss_refile(l, q);
            q->cap = (int)cap;
            l = q;
        }
        for (long k = 0; k < m; k++) l->items[l->n++] = viv_ss_store(l, add[k]);
        return l;
    }
    long n = l->n;
    viv_str* v = (viv_str*)malloc(sizeof(viv_str) * (size_t)(n + m ? n + m : 1));
    if (!v) viv_s_nomem();
    for (long k = 0; k < n; k++) v[k] = l->items[k];
    for (long k = 0; k < m; k++) v[n + k] = add[k];
    viv_strs r = viv_ss_copyof(v, n + m, n + m < 8 ? 8 : n + m);
    free(v);
    r->unsealed = 1;
    return r;
}
VIV_S_UNUSED static viv_strs viv_ss_append(viv_strs l, viv_str s, double own) {
    s = VIV_S(s);
    return viv_ss_grow(l, &s, 1, own);
}
VIV_S_UNUSED static viv_strs viv_ss_extend(viv_strs l, viv_strs m, double own) {
    m = VIV_SS(m);
    long n = m->n;
    /* m may be l itself, which the growth may move */
    viv_str* v = (viv_str*)malloc(sizeof(viv_str) * (size_t)(n ? n : 1));
    if (!v) viv_s_nomem();
    for (long k = 0; k < n; k++) v[k] = m->items[k];
    viv_strs r = viv_ss_grow(l, v, n, own);
    free(v);
    return r;
}
VIV_S_UNUSED static double viv_ss_contains(viv_strs l, viv_str s, int site) {
    (void)site;
    l = VIV_SS(l);
    for (int k = 0; k < l->n; k++) if (viv_s_eq(l->items[k], s) != 0.0) return 1.0;
    return 0.0;
}
VIV_S_UNUSED static double viv_ss_count(viv_strs l, viv_str s, int site) {
    (void)site;
    l = VIV_SS(l);
    double c = 0.0;
    for (int k = 0; k < l->n; k++) c += viv_s_eq(l->items[k], s) != 0.0;
    return c;
}
static int viv_ss_by(const void* a, const void* b) { return viv_s_order(*(const viv_str*)a, *(const viv_str*)b); }
VIV_S_UNUSED static viv_strs viv_ss_sorted(viv_strs l, int site) {
    (void)site;
    l = VIV_SS(l);
    viv_str* v = (viv_str*)malloc(sizeof(viv_str) * (size_t)(l->n ? l->n : 1));
    if (!v) viv_s_nomem();
    for (int k = 0; k < l->n; k++) v[k] = l->items[k];
    /* equal strings are the same value, so qsort's order among them is not seen */
    qsort(v, (size_t)l->n, sizeof(viv_str), viv_ss_by);
    viv_strs r = viv_ss_copyof(v, l->n, 0);
    free(v);
    return r;
}
/* sorted(set(words)): the distinct strings, in code point order */
VIV_S_UNUSED static viv_strs viv_ss_unique(viv_strs l, int site) {
    (void)site;
    l = VIV_SS(l);
    viv_str* v = (viv_str*)malloc(sizeof(viv_str) * (size_t)(l->n ? l->n : 1));
    if (!v) viv_s_nomem();
    for (int k = 0; k < l->n; k++) v[k] = l->items[k];
    qsort(v, (size_t)l->n, sizeof(viv_str), viv_ss_by);
    long m = 0;
    for (int k = 0; k < l->n; k++) if (m == 0 || viv_s_order(v[m - 1], v[k]) != 0) v[m++] = v[k];
    viv_strs r = viv_ss_copyof(v, m, 0);
    free(v);
    return r;
}
/* a list of pieces b[off[k] .. off[k] + len[k]) of one text, in one chunk */
VIV_S_UNUSED static viv_strs viv_ss_pieces(const char* b, const long* off, const long* len, long n) {
    viv_strs l = viv_ss_alloc(n);
    long bytes = 0;
    for (long k = 0; k < n; k++) bytes += ((long)(sizeof(viv_strb) + (size_t)len[k] + 1) + 7) & ~7L;
    if (bytes) {
        viv_chunk* c = (viv_chunk*)malloc(sizeof(viv_chunk) + (size_t)bytes);
        if (!c) viv_s_nomem();
        c->prev = 0; c->cap = bytes; c->used = 0; c->pad = 0;
        viv_arena_push(c);
        l->chunk = c;
    }
    for (long k = 0; k < n; k++) {
        viv_chunk* c = l->chunk;
        long need = ((long)(sizeof(viv_strb) + (size_t)len[k] + 1) + 7) & ~7L;
        viv_str r = (viv_str)(void*)(c->data + c->used);
        c->used += need;
        r->owner = c; r->len = (int)len[k]; r->ci = 0; r->cb = 0;
        memcpy(r->s, b + off[k], (size_t)len[k]);
        r->s[len[k]] = 0;
        r->nch = viv_s_nchars(r->s, len[k]);
        l->items[l->n++] = r;
    }
    return l;
}
/* pieces being collected for viv_ss_pieces */
typedef struct { long* off; long* len; long n, cap; } viv_pc;
VIV_S_UNUSED static void viv_pc_add(viv_pc* p, long off, long len) {
    if (p->n == p->cap) {
        p->cap = p->cap ? 2 * p->cap : 16;
        p->off = (long*)realloc(p->off, sizeof(long) * (size_t)p->cap);
        p->len = (long*)realloc(p->len, sizeof(long) * (size_t)p->cap);
        if (!p->off || !p->len) viv_s_nomem();
    }
    p->off[p->n] = off; p->len[p->n] = len; p->n++;
}
VIV_S_UNUSED static viv_strs viv_pc_done(viv_pc* p, const char* b, int reverse) {
    if (reverse) {
        for (long i = 0, j = p->n - 1; i < j; i++, j--) {
            long t = p->off[i]; p->off[i] = p->off[j]; p->off[j] = t;
            t = p->len[i]; p->len[i] = p->len[j]; p->len[j] = t;
        }
    }
    viv_strs l = viv_ss_pieces(b, p->off, p->len, p->n);
    free(p->off); free(p->len);
    return l;
}
/* one character of a text: its code point and its bytes */
typedef struct { unsigned cp; long off, len; } viv_s_ch;
VIV_S_UNUSED static int viv_s_ch_by(const void* a, const void* b) {
    unsigned x = ((const viv_s_ch*)a)->cp, y = ((const viv_s_ch*)b)->cp;
    return x < y ? -1 : x > y;
}
/* Python's sorted(s) of a string, and with `distinct` sorted(set(s)): its
   characters as one-character strings, in code point order */
VIV_S_UNUSED static viv_strs viv_s_chars_sorted(viv_str s, int distinct) {
    s = VIV_S(s);
    viv_s_ch* v = (viv_s_ch*)malloc(sizeof(viv_s_ch) * (size_t)(s->nch ? s->nch : 1));
    if (!v) viv_s_nomem();
    long n = 0, i = 0;
    while (i < s->len) {
        long at = i;
        unsigned cp = viv_s_dec(s->s, &i);
        v[n].cp = cp; v[n].off = at; v[n].len = i - at; n++;
    }
    qsort(v, (size_t)n, sizeof(viv_s_ch), viv_s_ch_by);
    long* off = (long*)malloc(sizeof(long) * (size_t)(n ? n : 1));
    long* len = (long*)malloc(sizeof(long) * (size_t)(n ? n : 1));
    if (!off || !len) viv_s_nomem();
    long m = 0;
    for (long k = 0; k < n; k++) {
        if (distinct && m > 0 && v[k].cp == v[k - 1].cp) continue;
        off[m] = v[k].off; len[m] = v[k].len; m++;
    }
    viv_strs r = viv_ss_pieces(s->s, off, len, m);
    free(v); free(off); free(len);
    return r;
}
VIV_S_UNUSED static viv_strs viv_s_sorted(viv_str s, int site) { (void)site; return viv_s_chars_sorted(s, 0); }
VIV_S_UNUSED static viv_strs viv_s_unique(viv_str s, int site) { (void)site; return viv_s_chars_sorted(s, 1); }
//@ str_repr: str_list str_t_printable
/* Python's repr(str) */
VIV_S_UNUSED static void viv_s_repr_into(viv_sb* b, viv_str s) {
    s = VIV_S(s);
    int has1 = memchr(s->s, '\'', (size_t)s->len) != 0, has2 = memchr(s->s, '"', (size_t)s->len) != 0;
    char q = (has1 && !has2) ? '"' : '\'';
    char e[16];
    viv_sb_put(b, &q, 1);
    long i = 0;
    while (i < s->len) {
        unsigned c = viv_s_dec(s->s, &i);
        if (c == '\\') viv_sb_cstr(b, "\\\\");
        else if (c == '\t') viv_sb_cstr(b, "\\t");
        else if (c == '\n') viv_sb_cstr(b, "\\n");
        else if (c == '\r') viv_sb_cstr(b, "\\r");
        else if (c == (unsigned)q) { viv_sb_put(b, "\\", 1); viv_sb_put(b, &q, 1); }
        else if (c < 0x20 || c == 0x7f) { snprintf(e, sizeof e, "\\x%02x", c); viv_sb_cstr(b, e); }
        else if (c < 0x80) viv_sb_cp(b, c);
        else if (viv_s_inr(c, viv_t_printable, VIV_T_PRINTABLE_N)) viv_sb_cp(b, c);
        else if (c < 0x100) { snprintf(e, sizeof e, "\\x%02x", c); viv_sb_cstr(b, e); }
        else if (c < 0x10000) { snprintf(e, sizeof e, "\\u%04x", c); viv_sb_cstr(b, e); }
        else { snprintf(e, sizeof e, "\\U%08x", c); viv_sb_cstr(b, e); }
    }
    viv_sb_put(b, &q, 1);
}
VIV_S_UNUSED static viv_str viv_s_repr(viv_str s, int site) {
    viv_sb b = {0, 0, 0, site};
    viv_s_repr_into(&b, s);
    return viv_sb_done(&b);
}
VIV_S_UNUSED static viv_str viv_ss_repr(viv_strs l, int site) {
    l = VIV_SS(l);
    viv_sb b = {0, 0, 0, site};
    viv_sb_put(&b, "[", 1);
    for (int k = 0; k < l->n; k++) {
        if (k) viv_sb_put(&b, ", ", 2);
        viv_s_repr_into(&b, l->items[k]);
    }
    viv_sb_put(&b, "]", 1);
    return viv_sb_done(&b);
}
VIV_S_UNUSED static double viv_ss_index(viv_strs l, viv_str s, int site) {
    l = VIV_SS(l);
    for (int k = 0; k < l->n; k++) if (viv_s_eq(l->items[k], s) != 0.0) return (double)k;
    viv_sb b = {0, 0, 0, site};
    viv_s_repr_into(&b, s);
    viv_sb_cstr(&b, " is not in list");
    viv_sb_put(&b, "", 1);
    viv_s_fail(VIV_E_VALUE, b.p, site);
}
//@ str_case: str_core str_t_case str_t_ignorable str_t_cased
/* push c's full mapping in table t (with its pool) */
VIV_S_UNUSED static void viv_s_mapcase(viv_sb* b, unsigned c, const unsigned (*t)[3], int n, const unsigned* pool) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int m = (lo + hi) / 2;
        if (t[m][0] == c) {
            if (t[m][2] == 1) viv_sb_cp(b, t[m][1]);
            else for (unsigned k = 0; k < t[m][2]; k++) viv_sb_cp(b, pool[t[m][1] + k]);
            return;
        }
        if (t[m][0] < c) lo = m + 1; else hi = m - 1;
    }
    viv_sb_cp(b, c);
}
VIV_S_UNUSED static viv_str viv_s_upper(viv_str s, int site) {
    s = VIV_S(s);
    viv_sb b = {0, 0, 0, site};
    long i = 0;
    while (i < s->len) viv_s_mapcase(&b, viv_s_dec(s->s, &i), viv_t_toupper, VIV_T_TOUPPER_N, viv_t_toupper_pool);
    return viv_sb_done(&b);
}
/* CPython's handle_capital_sigma, over the code points cs[0..n) */
VIV_S_UNUSED static int viv_s_final_sigma(const unsigned* cs, long n, long i) {
    long j = i;
    int found = 0;
    unsigned before = 0;
    while (j > 0) {
        j--;
        if (!viv_s_inr(cs[j], viv_t_ignorable, VIV_T_IGNORABLE_N)) { before = cs[j]; found = 1; break; }
    }
    if (!found || !viv_s_inr(before, viv_t_cased, VIV_T_CASED_N)) return 0;
    for (long k = i + 1; k < n; k++) {
        if (!viv_s_inr(cs[k], viv_t_ignorable, VIV_T_IGNORABLE_N)) return !viv_s_inr(cs[k], viv_t_cased, VIV_T_CASED_N);
    }
    return 1;
}
VIV_S_UNUSED static viv_str viv_s_lower(viv_str s, int site) {
    s = VIV_S(s);
    unsigned* cs = (unsigned*)malloc(sizeof(unsigned) * (size_t)(s->nch ? s->nch : 1));
    if (!cs) viv_s_nomem();
    long i = 0, n = 0;
    while (i < s->len) cs[n++] = viv_s_dec(s->s, &i);
    viv_sb b = {0, 0, 0, site};
    for (long k = 0; k < n; k++) {
        if (cs[k] == 0x3A3) viv_sb_cp(&b, viv_s_final_sigma(cs, n, k) ? 0x3C2 : 0x3C3);
        else viv_s_mapcase(&b, cs[k], viv_t_tolower, VIV_T_TOLOWER_N, viv_t_tolower_pool);
    }
    free(cs);
    return viv_sb_done(&b);
}
//@ str_pred: str_core str_t_space str_t_alpha str_t_digit str_t_numeric str_t_upper str_t_lower str_t_title str_t_printable str_t_decimal
VIV_S_UNUSED static int viv_s_decimal(unsigned c) {
    int lo = 0, hi = VIV_T_DECIMAL_N;
    while (lo < hi) { int m = (lo + hi) / 2; if (viv_t_decimal[m][0] <= c) lo = m + 1; else hi = m; }
    return (lo > 0 && c <= viv_t_decimal[lo - 1][1]) ? (int)(c - viv_t_decimal[lo - 1][2]) : -1;
}
/* which: 0 space, 1 alpha, 2 alnum, 3 digit, 4 decimal, 5 numeric, 6 printable */
VIV_S_UNUSED static int viv_s_is1(unsigned c, int which) {
    switch (which) {
    case 0: return viv_s_inr(c, viv_t_space, VIV_T_SPACE_N);
    case 1: return viv_s_inr(c, viv_t_alpha, VIV_T_ALPHA_N);
    case 2: return viv_s_inr(c, viv_t_alpha, VIV_T_ALPHA_N) || viv_s_decimal(c) >= 0 || viv_s_inr(c, viv_t_digit, VIV_T_DIGIT_N) || viv_s_inr(c, viv_t_numeric, VIV_T_NUMERIC_N);
    case 3: return viv_s_inr(c, viv_t_digit, VIV_T_DIGIT_N);
    case 4: return viv_s_decimal(c) >= 0;
    case 5: return viv_s_inr(c, viv_t_numeric, VIV_T_NUMERIC_N);
    default: return viv_s_inr(c, viv_t_printable, VIV_T_PRINTABLE_N);
    }
}
VIV_S_UNUSED static double viv_s_all(viv_str s, int which) {
    s = VIV_S(s);
    if (s->len == 0) return 0.0;
    long i = 0;
    while (i < s->len) if (!viv_s_is1(viv_s_dec(s->s, &i), which)) return 0.0;
    return 1.0;
}
VIV_S_UNUSED static double viv_s_isspace(viv_str s, int site) { (void)site; return viv_s_all(s, 0); }
VIV_S_UNUSED static double viv_s_isalpha(viv_str s, int site) { (void)site; return viv_s_all(s, 1); }
VIV_S_UNUSED static double viv_s_isalnum(viv_str s, int site) { (void)site; return viv_s_all(s, 2); }
VIV_S_UNUSED static double viv_s_isdigit(viv_str s, int site) { (void)site; return viv_s_all(s, 3); }
VIV_S_UNUSED static double viv_s_isdecimal(viv_str s, int site) { (void)site; return viv_s_all(s, 4); }
VIV_S_UNUSED static double viv_s_isnumeric(viv_str s, int site) { (void)site; return viv_s_all(s, 5); }
VIV_S_UNUSED static double viv_s_isprintable(viv_str s, int site) {
    (void)site;
    s = VIV_S(s);
    long i = 0;
    while (i < s->len) if (!viv_s_is1(viv_s_dec(s->s, &i), 6)) return 0.0;
    return 1.0;
}
VIV_S_UNUSED static double viv_s_isascii(viv_str s, int site) {
    (void)site;
    s = VIV_S(s);
    return s->len == s->nch;
}
/* CPython's unicode_isupper (upper 1) and unicode_islower (upper 0) */
VIV_S_UNUSED static double viv_s_iscase(viv_str s, int upper) {
    s = VIV_S(s);
    int cased = 0;
    long i = 0;
    while (i < s->len) {
        unsigned c = viv_s_dec(s->s, &i);
        int up = viv_s_inr(c, viv_t_upper, VIV_T_UPPER_N), lo = viv_s_inr(c, viv_t_lower, VIV_T_LOWER_N);
        if ((upper ? lo : up) || viv_s_inr(c, viv_t_title, VIV_T_TITLE_N)) return 0.0;
        if (upper ? up : lo) cased = 1;
    }
    return cased;
}
VIV_S_UNUSED static double viv_s_isupper(viv_str s, int site) { (void)site; return viv_s_iscase(s, 1); }
VIV_S_UNUSED static double viv_s_islower(viv_str s, int site) { (void)site; return viv_s_iscase(s, 0); }
//@ str_ws: str_list str_t_space
VIV_S_UNUSED static int viv_s_space(unsigned c) { return viv_s_inr(c, viv_t_space, VIV_T_SPACE_N); }
/* whether c is one of the code points of `set` */
VIV_S_UNUSED static int viv_s_inset(unsigned c, viv_str set) {
    long i = 0;
    while (i < set->len) if (viv_s_dec(set->s, &i) == c) return 1;
    return 0;
}
/* strip: which 0 both ends, 1 left, 2 right; set 0 is whitespace */
VIV_S_UNUSED static viv_str viv_s_strip3(viv_str s, viv_str set, int which, int site) {
    s = VIV_S(s);
    long a = 0, b = s->len;
    if (which != 2) {
        while (a < b) { long k = a; unsigned c = viv_s_dec(s->s, &k); if (set ? !viv_s_inset(c, set) : !viv_s_space(c)) break; a = k; }
    }
    if (which != 1) {
        while (b > a) { long k = b; unsigned c = viv_s_dec_back(s->s, &k); if (set ? !viv_s_inset(c, set) : !viv_s_space(c)) break; b = k; }
    }
    if (a == 0 && b == s->len) return s;
    return viv_s_make(s->s + a, b - a, site);
}
VIV_S_UNUSED static viv_str viv_s_strip(viv_str s, int site) { return viv_s_strip3(s, 0, 0, site); }
VIV_S_UNUSED static viv_str viv_s_lstrip(viv_str s, int site) { return viv_s_strip3(s, 0, 1, site); }
VIV_S_UNUSED static viv_str viv_s_rstrip(viv_str s, int site) { return viv_s_strip3(s, 0, 2, site); }
VIV_S_UNUSED static viv_str viv_s_strip_chars(viv_str s, viv_str c, int site) { return viv_s_strip3(s, VIV_S(c), 0, site); }
VIV_S_UNUSED static viv_str viv_s_lstrip_chars(viv_str s, viv_str c, int site) { return viv_s_strip3(s, VIV_S(c), 1, site); }
VIV_S_UNUSED static viv_str viv_s_rstrip_chars(viv_str s, viv_str c, int site) { return viv_s_strip3(s, VIV_S(c), 2, site); }
/* a maxsplit: the splits allowed; negative is no limit */
VIV_S_UNUSED static long viv_s_maxc(double n) {
    long long k = viv_s_i64(n);
    return k < 0 ? LONG_MAX : (k > LONG_MAX ? LONG_MAX : (long)k);
}
/* split and rsplit: CPython's split_whitespace, split, rsplit_whitespace, rsplit */
VIV_S_UNUSED static viv_strs viv_s_split4(viv_str s, viv_str sep, double maxsplit, int right, int site) {
    s = VIV_S(s);
    long max = viv_s_maxc(maxsplit);
    viv_pc p = {0, 0, 0, 0};
    const char* b = s->s;
    long n = s->len;
    if (sep && sep->len == 0) viv_s_fail(VIV_E_VALUE, "empty separator", site);
    if (sep && !right) {
        long at = 0;
        while (max > 0) {
            long k = viv_s_find_bytes(b, n, sep->s, sep->len, at);
            if (k < 0) break;
            viv_pc_add(&p, at, k - at);
            at = k + sep->len;
            max--;
        }
        viv_pc_add(&p, at, n - at);
        return viv_pc_done(&p, b, 0);
    }
    if (sep) {
        long end = n;
        while (max > 0) {
            long k = viv_s_rfind_bytes(b, end, sep->s, sep->len);
            if (k < 0) break;
            viv_pc_add(&p, k + sep->len, end - k - sep->len);
            end = k;
            max--;
        }
        viv_pc_add(&p, 0, end);
        return viv_pc_done(&p, b, 1);
    }
    if (!right) {
        long i = 0;
        while (max > 0) {
            while (i < n) { long k = i; if (!viv_s_space(viv_s_dec(b, &k))) break; i = k; }
            if (i == n) break;
            long j = i;
            viv_s_dec(b, &i);
            while (i < n) { long k = i; if (viv_s_space(viv_s_dec(b, &k))) break; i = k; }
            viv_pc_add(&p, j, i - j);
            max--;
        }
        if (i < n) {
            while (i < n) { long k = i; if (!viv_s_space(viv_s_dec(b, &k))) break; i = k; }
            if (i != n) viv_pc_add(&p, i, n - i);
        }
        return viv_pc_done(&p, b, 0);
    }
    long i = n;
    while (max > 0) {
        while (i > 0) { long k = i; if (!viv_s_space(viv_s_dec_back(b, &k))) break; i = k; }
        if (i == 0) break;
        long j = i;
        viv_s_dec_back(b, &i);
        while (i > 0) { long k = i; if (viv_s_space(viv_s_dec_back(b, &k))) break; i = k; }
        viv_pc_add(&p, i, j - i);
        max--;
    }
    if (i > 0) {
        while (i > 0) { long k = i; if (!viv_s_space(viv_s_dec_back(b, &k))) break; i = k; }
        if (i != 0) viv_pc_add(&p, 0, i);
    }
    return viv_pc_done(&p, b, 1);
}
VIV_S_UNUSED static viv_strs viv_s_split_ws(viv_str s, int site) { return viv_s_split4(s, 0, -1.0, 0, site); }
VIV_S_UNUSED static viv_strs viv_s_rsplit_ws(viv_str s, int site) { return viv_s_split4(s, 0, -1.0, 1, site); }
VIV_S_UNUSED static viv_strs viv_s_split(viv_str s, viv_str sep, double m, int site) { return viv_s_split4(s, VIV_S(sep), m, 0, site); }
VIV_S_UNUSED static viv_strs viv_s_rsplit(viv_str s, viv_str sep, double m, int site) { return viv_s_split4(s, VIV_S(sep), m, 1, site); }
VIV_S_UNUSED static int viv_s_linebreak(unsigned c) {
    return c == 0x0A || c == 0x0B || c == 0x0C || c == 0x0D || c == 0x1C || c == 0x1D || c == 0x1E || c == 0x85 || c == 0x2028 || c == 0x2029;
}
/* CPython's stringlib splitlines, without the line ends */
VIV_S_UNUSED static viv_strs viv_s_splitlines(viv_str s, int site) {
    (void)site;
    s = VIV_S(s);
    viv_pc p = {0, 0, 0, 0};
    const char* b = s->s;
    long n = s->len, i = 0, j = 0;
    while (i < n) {
        while (i < n) { long k = i; if (viv_s_linebreak(viv_s_dec(b, &k))) break; i = k; }
        long eol = i;
        if (i < n) {
            if (b[i] == '\r' && i + 1 < n && b[i + 1] == '\n') i += 2;
            else viv_s_dec(b, &i);
        }
        viv_pc_add(&p, j, eol - j);
        j = i;
    }
    return viv_pc_done(&p, b, 0);
}
VIV_S_UNUSED static viv_str viv_s_join(viv_str sep, viv_strs l, int site) {
    sep = VIV_S(sep); l = VIV_SS(l);
    long total = 0, nch = 0;
    for (int k = 0; k < l->n; k++) { total += l->items[k]->len; nch += l->items[k]->nch; }
    if (l->n > 1) { total += (long)sep->len * (l->n - 1); nch += (long)sep->nch * (l->n - 1); }
    if (total > VIV_S_MAX) viv_s_too_long((unsigned long long)total, site);
    viv_str r = viv_s_alloc(total, site);
    long at = 0;
    for (int k = 0; k < l->n; k++) {
        if (k) { memcpy(r->s + at, sep->s, (size_t)sep->len); at += sep->len; }
        memcpy(r->s + at, l->items[k]->s, (size_t)l->items[k]->len);
        at += l->items[k]->len;
    }
    r->nch = (int)nch;
    return r;
}
//@ str_search: str_core
/* CPython's ADJUST_INDICES; an end of NaN is "not given" */
VIV_S_UNUSED static void viv_s_adjust(long len, double start, double end, long long* s, long long* e) {
    long long n = len;
    *s = viv_s_i64(start);
    *e = end != end ? n : viv_s_i64(end);
    if (*e > n) *e = n;
    else if (*e < 0) { *e = *e < LLONG_MIN + n ? LLONG_MIN : *e + n; if (*e < 0) *e = 0; }
    if (*s < 0) { *s = *s < LLONG_MIN + n ? LLONG_MIN : *s + n; if (*s < 0) *s = 0; }
}
/* find / rfind: the code point where sub first (last) starts in s[start:end], or -1 */
VIV_S_UNUSED static double viv_s_find5(viv_str s, viv_str sub, double start, double end, int right) {
    s = VIV_S(s); sub = VIV_S(sub);
    long long a, e;
    viv_s_adjust(s->nch, start, end, &a, &e);
    if (e - a < sub->nch) return -1.0;
    int ba = viv_s_byteof(s, (int)a), bb = viv_s_byteof(s, (int)e);
    long k = right ? viv_s_rfind_bytes(s->s + ba, bb - ba, sub->s, sub->len) : viv_s_find_bytes(s->s + ba, bb - ba, sub->s, sub->len, 0);
    if (k < 0) return -1.0;
    return (double)(a + viv_s_nchars(s->s + ba, k));
}
VIV_S_UNUSED static double viv_s_find(viv_str s, viv_str sub, double a, double e, int site) { (void)site; return viv_s_find5(s, sub, a, e, 0); }
VIV_S_UNUSED static double viv_s_rfind(viv_str s, viv_str sub, double a, double e, int site) { (void)site; return viv_s_find5(s, sub, a, e, 1); }
VIV_S_UNUSED static double viv_s_index(viv_str s, viv_str sub, double a, double e, int site) {
    double k = viv_s_find5(s, sub, a, e, 0);
    if (k < 0) viv_s_fail(VIV_E_VALUE, "substring not found", site);
    return k;
}
VIV_S_UNUSED static double viv_s_rindex(viv_str s, viv_str sub, double a, double e, int site) {
    double k = viv_s_find5(s, sub, a, e, 1);
    if (k < 0) viv_s_fail(VIV_E_VALUE, "substring not found", site);
    return k;
}
VIV_S_UNUSED static double viv_s_count(viv_str s, viv_str sub, double start, double end, int site) {
    (void)site;
    s = VIV_S(s); sub = VIV_S(sub);
    long long a, e;
    viv_s_adjust(s->nch, start, end, &a, &e);
    if (e - a < sub->nch) return 0.0;
    if (sub->len == 0) return (double)(e - a + 1);
    int ba = viv_s_byteof(s, (int)a), bb = viv_s_byteof(s, (int)e);
    double c = 0.0;
    long at = 0;
    for (;;) {
        long k = viv_s_find_bytes(s->s + ba, bb - ba, sub->s, sub->len, at);
        if (k < 0) break;
        c += 1.0;
        at = k + sub->len;
    }
    return c;
}
VIV_S_UNUSED static viv_str viv_s_replace(viv_str s, viv_str old, viv_str nw, double count, int site) {
    s = VIV_S(s); old = VIV_S(old); nw = VIV_S(nw);
    long long mc = viv_s_i64(count);
    long max = mc < 0 ? LONG_MAX : (mc > LONG_MAX ? LONG_MAX : (long)mc);
    if (max == 0) return s;
    viv_sb b = {0, 0, 0, site};
    if (old->len == 0) {
        long left = max, i = 0;
        while (i < s->len) {
            if (left > 0) { viv_sb_put(&b, nw->s, nw->len); left--; }
            long k = i;
            viv_s_dec(s->s, &k);
            viv_sb_put(&b, s->s + i, k - i);
            i = k;
        }
        if (left > 0) viv_sb_put(&b, nw->s, nw->len);
        return viv_sb_done(&b);
    }
    long at = 0, left = max;
    while (left > 0) {
        long k = viv_s_find_bytes(s->s, s->len, old->s, old->len, at);
        if (k < 0) break;
        viv_sb_put(&b, s->s + at, k - at);
        viv_sb_put(&b, nw->s, nw->len);
        at = k + old->len;
        left--;
    }
    viv_sb_put(&b, s->s + at, s->len - at);
    return viv_sb_done(&b);
}
VIV_S_UNUSED static double viv_s_startswith(viv_str s, viv_str p, int site) {
    (void)site;
    s = VIV_S(s); p = VIV_S(p);
    return p->len <= s->len && memcmp(s->s, p->s, (size_t)p->len) == 0;
}
VIV_S_UNUSED static double viv_s_endswith(viv_str s, viv_str p, int site) {
    (void)site;
    s = VIV_S(s); p = VIV_S(p);
    return p->len <= s->len && memcmp(s->s + s->len - p->len, p->s, (size_t)p->len) == 0;
}
VIV_S_UNUSED static double viv_s_contains(viv_str s, viv_str sub, int site) {
    (void)site;
    s = VIV_S(s); sub = VIV_S(sub);
    return viv_s_find_bytes(s->s, s->len, sub->s, sub->len, 0) >= 0;
}
//@ str_pad: str_core
/* ljust (0), rjust (1), center (2): CPython's pad() */
VIV_S_UNUSED static viv_str viv_s_pad(viv_str s, double width, viv_str fill, int which, int site) {
    s = VIV_S(s); fill = VIV_S(fill);
    if (fill->nch != 1) viv_s_fail(VIV_E_VALUE, "The fill character must be exactly one character long", site);
    long long w = viv_s_i64(width), n = s->nch;
    if (w <= n) return s;
    long long marg = w - n, left, right;
    if (which == 0) { left = 0; right = marg; }
    else if (which == 1) { left = marg; right = 0; }
    else { left = marg / 2 + (marg & w & 1); right = marg - left; }
    if (w > VIV_S_MAX / 4) viv_s_too_long((unsigned long long)w, site);
    viv_sb b = {0, 0, 0, site};
    for (long long k = 0; k < left; k++) viv_sb_put(&b, fill->s, fill->len);
    viv_sb_put(&b, s->s, s->len);
    for (long long k = 0; k < right; k++) viv_sb_put(&b, fill->s, fill->len);
    return viv_sb_done(&b);
}
VIV_S_UNUSED static viv_str viv_s_ljust(viv_str s, double w, viv_str f, int site) { return viv_s_pad(s, w, f, 0, site); }
VIV_S_UNUSED static viv_str viv_s_rjust(viv_str s, double w, viv_str f, int site) { return viv_s_pad(s, w, f, 1, site); }
VIV_S_UNUSED static viv_str viv_s_center(viv_str s, double w, viv_str f, int site) { return viv_s_pad(s, w, f, 2, site); }
VIV_S_UNUSED static viv_str viv_s_zfill(viv_str s, double width, int site) {
    s = VIV_S(s);
    long long w = viv_s_i64(width), n = s->nch;
    if (w <= n) return s;
    if (w > VIV_S_MAX) viv_s_too_long((unsigned long long)w, site);
    long long fill = w - n;
    viv_sb b = {0, 0, 0, site};
    int sign = s->len > 0 && (s->s[0] == '+' || s->s[0] == '-');
    if (sign) viv_sb_put(&b, s->s, 1);
    for (long long k = 0; k < fill; k++) viv_sb_put(&b, "0", 1);
    viv_sb_put(&b, s->s + sign, s->len - sign);
    return viv_sb_done(&b);
}
/* fmt(s, spec): Python's format(str, spec), the spec checked when compiled:
   the fill's code point, the alignment ('<', '>', '^'), the width, the
   precision (-1: none) */
VIV_S_UNUSED static viv_str viv_s_fmt_str(viv_str s, double fillf, double alignf, double widthf, double precf, int site) {
    viv_s_fmtwp(precf, widthf, 1073741824.0, site);
    s = VIV_S(s);
    long body = s->len, n = s->nch;
    if (precf >= 0.0 && (long)precf < s->nch) { n = (long)precf; body = viv_s_byteof(s, (int)n); }
    long w = (long)widthf;
    if (n >= w) return body == s->len ? s : viv_s_make(s->s, body, site);
    long marg = w - n, left, right;
    int align = (int)alignf;
    if (align == '>') { left = marg; right = 0; }
    else if (align == '^') { left = marg / 2; right = marg - marg / 2; }
    else { left = 0; right = marg; }
    char f[4];
    int fl = viv_s_enc((unsigned)fillf, f);
    viv_sb b = {0, 0, 0, site};
    for (long k = 0; k < left; k++) viv_sb_put(&b, f, fl);
    viv_sb_put(&b, s->s, body);
    for (long k = 0; k < right; k++) viv_sb_put(&b, f, fl);
    return viv_sb_done(&b);
}
//@ str_num: str_repr str_t_space str_t_decimal
VIV_S_UNUSED static int viv_s_decimal_of(unsigned c) {
    int lo = 0, hi = VIV_T_DECIMAL_N;
    while (lo < hi) { int m = (lo + hi) / 2; if (viv_t_decimal[m][0] <= c) lo = m + 1; else hi = m; }
    return (lo > 0 && c <= viv_t_decimal[lo - 1][1]) ? (int)(c - viv_t_decimal[lo - 1][2]) : -1;
}
/* CPython's _PyUnicode_TransformDecimalAndSpaceToASCII: malloc'd, NUL-ended */
VIV_S_UNUSED static char* viv_s_numtext(viv_str s, long* outn) {
    char* t = (char*)malloc((size_t)s->len + 2);
    if (!t) viv_s_nomem();
    long i = 0, n = 0;
    while (i < s->len) {
        unsigned c = viv_s_dec(s->s, &i);
        int d;
        if (c < 127) t[n++] = (char)c;
        else if (viv_s_inr(c, viv_t_space, VIV_T_SPACE_N)) t[n++] = ' ';
        else if ((d = viv_s_decimal_of(c)) >= 0) t[n++] = (char)('0' + d);
        else { t[n++] = '?'; break; }
    }
    t[n] = 0;
    *outn = n;
    return t;
}
VIV_S_UNUSED static int viv_s_aspace(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r'; }
VIV_S_UNUSED static int viv_s_digit(char c) { return c >= '0' && c <= '9'; }
/* the message, and stop */
VIV_S_NORETURN static void viv_s_badnum(const char* head, viv_str s, int cut, int site) {
    viv_sb b = {0, 0, 0, site};
    viv_sb_cstr(&b, head);
    viv_sb r = {0, 0, 0, site};
    viv_s_repr_into(&r, s);
    long keep = r.n;
    if (cut) {
        /* the repr cut at 200 characters, as CPython's %.200R cuts it */
        long i = 0, k = 0;
        while (i < r.n && k < 200) { viv_s_dec(r.p, &i); k++; }
        keep = i;
    }
    viv_sb_put(&b, r.p, keep);
    viv_sb_put(&b, "", 1);
    free(r.p);
    viv_s_fail(VIV_E_VALUE, b.p, site);
}
/* CPython's float(str): the transform, the underscores, the whitespace, then
   the plain decimal (or inf, infinity, nan) handed to strtod */
VIV_S_UNUSED static double viv_s_float(viv_str s, int site) {
    s = VIV_S(s);
    long n;
    char* t = viv_s_numtext(s, &n);
    if (memchr(t, '_', (size_t)n)) {
        long w = 0;
        char prev = 0;
        for (long i = 0; i < n; i++) {
            char c = t[i];
            if (c == '_') { if (!viv_s_digit(prev)) goto bad; }
            else { t[w++] = c; if (prev == '_' && !viv_s_digit(c)) goto bad; }
            prev = c;
        }
        if (prev == '_') goto bad;
        n = w;
        t[n] = 0;
    }
    {
        long a = 0, e = n;
        while (a < e && viv_s_aspace(t[a])) a++;
        while (e > a && viv_s_aspace(t[e - 1])) e--;
        if (a == e) goto bad;
        char* p = t + a;
        long m = e - a;
        p[m] = 0;
        int i = 0, neg = 0;
        if (p[0] == '-') { neg = 1; i = 1; } else if (p[0] == '+') i = 1;
        {
            char low[16];
            long rest = m - i;
            if (rest == 3 || rest == 8) {
                for (long k = 0; k < rest; k++) low[k] = (char)(p[i + k] >= 'A' && p[i + k] <= 'Z' ? p[i + k] + 32 : p[i + k]);
                low[rest] = 0;
                if (strcmp(low, "inf") == 0 || strcmp(low, "infinity") == 0) { free(t); return neg ? -HUGE_VAL : HUGE_VAL; }
                if (strcmp(low, "nan") == 0) { free(t); return neg ? -NAN : NAN; }
            }
        }
        /* [-]digits[.digits][e[-]digits], as strings.rs's plain_decimal writes it */
        char* o = (char*)malloc((size_t)m + 16);
        if (!o) viv_s_nomem();
        long w = 0;
        int intd = 0, fracd = 0;
        if (neg) o[w++] = '-';
        while (i < m && viv_s_digit(p[i])) { o[w++] = p[i++]; intd++; }
        if (intd == 0) o[w++] = '0';
        if (i < m && p[i] == '.') {
            i++;
            long mark = w;
            o[w++] = '.';
            while (i < m && viv_s_digit(p[i])) { o[w++] = p[i++]; fracd++; }
            if (fracd == 0) w = mark;
        }
        if (intd + fracd == 0) { free(o); goto bad; }
        if (i < m && (p[i] == 'e' || p[i] == 'E')) {
            long j = i + 1;
            char exp[16];
            int en = 0;
            exp[en++] = 'e';
            if (j < m && (p[j] == '-' || p[j] == '+')) { if (p[j] == '-') exp[en++] = '-'; j++; }
            long d0 = j;
            while (j < m && p[j] == '0') j++;
            long from = j;
            while (j < m && viv_s_digit(p[j])) j++;
            if (j > d0) {
                if (j - from > 9) { memcpy(exp + en, "999999999", 9); en += 9; }
                else if (j == from) exp[en++] = '0';
                else { memcpy(exp + en, p + from, (size_t)(j - from)); en += (int)(j - from); }
                memcpy(o + w, exp, (size_t)en);
                w += en;
                i = (int)j;
            }
        }
        if (i != m) { free(o); goto bad; }
        o[w] = 0;
        double x = strtod(o, 0);
        free(o);
        free(t);
        return x;
    }
bad:
    free(t);
    viv_s_badnum("could not convert string to float: ", s, 0, site);
}
VIV_S_UNUSED static int viv_s_digitv(char b) {
    if (b >= '0' && b <= '9') return b - '0';
    if (b >= 'a' && b <= 'z') return b - 'a' + 10;
    if (b >= 'A' && b <= 'Z') return b - 'A' + 10;
    return 37;
}
/* the integer with digits d[0..n) in `base`, to the nearest double (ties to
   even); 0 in *ok when it is beyond the largest (strings.rs digits_to_f64) */
VIV_S_UNUSED static double viv_s_digits(const unsigned char* d, long n, unsigned base, int* ok) {
    unsigned* limbs = (unsigned*)calloc((size_t)(n / 4 + 2), sizeof(unsigned));
    if (!limbs) viv_s_nomem();
    long nl = 1;
    for (long k = 0; k < n; k++) {
        unsigned long long carry = d[k];
        for (long j = 0; j < nl; j++) {
            unsigned long long v = (unsigned long long)limbs[j] * base + carry;
            limbs[j] = (unsigned)v;
            carry = v >> 32;
        }
        if (carry) limbs[nl++] = (unsigned)carry;
    }
    while (nl > 1 && limbs[nl - 1] == 0) nl--;
    unsigned top = limbs[nl - 1];
    int lz = 0;
    if (top == 0) lz = 32; else while (!(top & 0x80000000u)) { top <<= 1; lz++; }
    unsigned long bits = 32UL * (unsigned long)(nl - 1) + (unsigned long)(32 - lz);
    *ok = 1;
    if (bits <= 64) {
        unsigned long long v = 0;
        for (long k = 0; k < nl; k++) v |= (unsigned long long)limbs[k] << (32 * k);
        free(limbs);
        return (double)v;
    }
#define VIV_BIT(k) ((limbs[(k) / 32] >> ((k) % 32)) & 1u)
    unsigned long long m = 0;
    for (unsigned long k = 0; k < 64; k++) m = (m << 1) | VIV_BIT(bits - 1 - k);
    unsigned long low = bits - 64;
    int sticky = 0;
    for (unsigned long k = 0; k < low && !sticky; k++) sticky = VIV_BIT(k);
#undef VIV_BIT
    free(limbs);
    unsigned long long keep = m >> 11, rest = m & 0x7ff;
    if (rest > 0x400 || (rest == 0x400 && (sticky || (keep & 1)))) keep += 1;
    unsigned long e = low + 11;
    if (e > 1100) { *ok = 0; return 0.0; }
    double x = (double)keep * pow(2.0, (double)e);
    if (!(x <= DBL_MAX)) { *ok = 0; return 0.0; }
    return x;
}
/* CPython's int(str, base) */
VIV_S_UNUSED static double viv_s_int(viv_str s, double basef, int site) {
    s = VIV_S(s);
    long long base_i = viv_s_i64(basef);
    if (base_i != 0 && !(base_i >= 2 && base_i <= 36)) viv_s_fail(VIV_E_VALUE, "int() base must be >= 2 and <= 36, or 0", site);
    unsigned base = (unsigned)base_i;
    long n;
    char* t = viv_s_numtext(s, &n);
    long i = 0;
    int neg = 0, error_if_nonzero = 0;
    char head[80];
    unsigned shown;
    unsigned char* dg = 0;
    long nd = 0;
    while (i < n && viv_s_aspace(t[i])) i++;
    if (i < n && (t[i] == '+' || t[i] == '-')) { neg = t[i] == '-'; i++; }
#define VIV_AT(k) ((k) < n ? t[k] : 0)
    if (base == 0) {
        if (VIV_AT(i) != '0') base = 10;
        else if (VIV_AT(i + 1) == 'x' || VIV_AT(i + 1) == 'X') base = 16;
        else if (VIV_AT(i + 1) == 'o' || VIV_AT(i + 1) == 'O') base = 8;
        else if (VIV_AT(i + 1) == 'b' || VIV_AT(i + 1) == 'B') base = 2;
        else { error_if_nonzero = 1; base = 10; }
    }
    shown = error_if_nonzero ? 0 : base;
    if (VIV_AT(i) == '0' && ((base == 16 && (VIV_AT(i + 1) == 'x' || VIV_AT(i + 1) == 'X')) || (base == 8 && (VIV_AT(i + 1) == 'o' || VIV_AT(i + 1) == 'O')) || (base == 2 && (VIV_AT(i + 1) == 'b' || VIV_AT(i + 1) == 'B')))) {
        i += 2;
        if (VIV_AT(i) == '_') i++;
    }
    if (VIV_AT(i) == '_') { shown = (unsigned)base_i; goto bad; }
#undef VIV_AT
    dg = (unsigned char*)malloc((size_t)n + 1);
    if (!dg) viv_s_nomem();
    {
        char prev = 0;
        while (i < n && ((unsigned)viv_s_digitv(t[i]) < base || t[i] == '_')) {
            if (t[i] == '_') { if (prev == '_') { shown = (unsigned)base_i; goto bad; } }
            else dg[nd++] = (unsigned char)viv_s_digitv(t[i]);
            prev = t[i];
            i++;
        }
        if (prev == '_' || nd == 0) { shown = (unsigned)base_i; goto bad; }
    }
    if ((base & (base - 1)) != 0 && nd > 4300) {
        char msg[240];
        snprintf(msg, sizeof msg, "Exceeds the limit (4300 digits) for integer string conversion: value has %ld digits; use sys.set_int_max_str_digits() to increase the limit", nd);
        viv_s_fail(VIV_E_VALUE, msg, site);
    }
    if (error_if_nonzero) {
        for (long k = 0; k < nd; k++) if (dg[k]) goto bad;
    }
    while (i < n && viv_s_aspace(t[i])) i++;
    if (i != n) goto bad;
    {
        int ok;
        double x = viv_s_digits(dg, nd, base, &ok);
        free(dg);
        free(t);
        if (!ok) viv_s_fail(VIV_E_OVERFLOW, "int too large to convert to float", site);
        return neg && x != 0.0 ? -x : x;
    }
bad:
    free(dg);
    free(t);
    snprintf(head, sizeof head, "invalid literal for int() with base %u: ", shown);
    viv_s_badnum(head, s, 1, site);
}
VIV_S_UNUSED static double viv_num_int(double x, int site) {
    if (x != x) viv_s_fail(VIV_E_VALUE, "cannot convert float NaN to integer", site);
    if (x > DBL_MAX || x < -DBL_MAX) viv_s_fail(VIV_E_OVERFLOW, "cannot convert float infinity to integer", site);
    return trunc(x) + 0.0;
}
VIV_S_UNUSED static viv_str viv_s_chr(double nf, int site) {
    long long k = viv_s_i64(nf);
    if (k < 0 || k >= 0x110000) viv_s_fail(VIV_E_VALUE, "chr() arg not in range(0x110000)", site);
    if (k >= 0xD800 && k <= 0xDFFF) {
        char msg[120];
        snprintf(msg, sizeof msg, "chr(0x%llx) is a surrogate, which a Viv string (UTF-8) cannot hold", k);
        viv_s_fail(VIV_E_VALUE, msg, site);
    }
    if (k < 0x80) return viv_s_ascii((unsigned char)k);
    char e[4];
    int n = viv_s_enc((unsigned)k, e);
    return viv_s_make(e, n, site);
}
VIV_S_UNUSED static double viv_s_ord(viv_str s, int site) {
    s = VIV_S(s);
    if (s->nch != 1) {
        char msg[120];
        snprintf(msg, sizeof msg, "ord() expected a character, but string of length %d found", s->nch);
        viv_s_fail(VIV_E_VALUE, msg, site);
    }
    long i = 0;
    return (double)viv_s_dec(s->s, &i);
}
//@ str_fmt: str_core print
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
//@ str_file: str_list str_ws str_repr arr_new
/* Rust's from_utf8: the length of the valid prefix (valid_up_to) */
VIV_S_UNUSED static long viv_s_valid(const unsigned char* b, long n) {
    long i = 0;
    while (i < n) {
        unsigned c = b[i];
        long k;
        unsigned lo = 0x80, hi = 0xBF;
        if (c < 0x80) { i++; continue; }
        else if (c >= 0xC2 && c <= 0xDF) k = 1;
        else if (c == 0xE0) { k = 2; lo = 0xA0; }
        else if (c >= 0xE1 && c <= 0xEC) k = 2;
        else if (c == 0xED) { k = 2; hi = 0x9F; }
        else if (c >= 0xEE && c <= 0xEF) k = 2;
        else if (c == 0xF0) { k = 3; lo = 0x90; }
        else if (c >= 0xF1 && c <= 0xF3) k = 3;
        else if (c == 0xF4) { k = 3; hi = 0x8F; }
        else return i;
        if (i + k >= n) return i;
        if (b[i + 1] < lo || b[i + 1] > hi) return i;
        for (long j = 2; j <= k; j++) if (b[i + j] < 0x80 || b[i + j] > 0xBF) return i;
        i += k + 1;
    }
    return n;
}
VIV_S_NORETURN static void viv_s_ioerr(viv_str path, int site) {
    int e = errno;
    viv_sb b = {0, 0, 0, site};
    if (e == ENOENT) { viv_sb_cstr(&b, "No such file or directory: "); viv_s_repr_into(&b, path); }
    else if (e == EACCES) { viv_sb_cstr(&b, "Permission denied: "); viv_s_repr_into(&b, path); }
    else {
        char tail[160];
        viv_s_repr_into(&b, path);
        snprintf(tail, sizeof tail, ": %s (os error %d)", strerror(e), e);
        viv_sb_cstr(&b, tail);
    }
    viv_sb_put(&b, "", 1);
    viv_s_fail(VIV_E_ERROR, b.p, site);
}
VIV_S_UNUSED static viv_str viv_s_read_text(viv_str path, int site) {
    path = VIV_S(path);
    errno = 0;
    FILE* f = fopen(path->s, "rb");
    if (!f) viv_s_ioerr(path, site);
    viv_sb b = {0, 0, 0, site};
    char buf[65536];
    size_t k;
    while ((k = fread(buf, 1, sizeof buf, f)) > 0) viv_sb_put(&b, buf, (long)k);
    if (ferror(f)) { fclose(f); viv_s_ioerr(path, site); }
    fclose(f);
    long ok = viv_s_valid((const unsigned char*)b.p, b.n);
    if (ok != b.n) {
        viv_sb m = {0, 0, 0, site};
        char tail[120];
        viv_s_repr_into(&m, path);
        snprintf(tail, sizeof tail, ": the file is not UTF-8 text (an invalid byte at offset %ld)", ok);
        viv_sb_cstr(&m, tail);
        viv_sb_put(&m, "", 1);
        viv_s_fail(VIV_E_VALUE, m.p, site);
    }
    /* universal newlines: \r\n and a lone \r are \n */
    long w = 0;
    for (long i = 0; i < b.n; i++) {
        if (b.p[i] == '\r') { b.p[w++] = '\n'; if (i + 1 < b.n && b.p[i + 1] == '\n') i++; }
        else b.p[w++] = b.p[i];
    }
    b.n = w;
    return viv_sb_done(&b);
}
VIV_S_UNUSED static viv_strs viv_s_read_lines(viv_str path, int site) {
    return viv_s_splitlines(viv_s_read_text(path, site), site);
}
/* read_bytes(path, offset, count): the bytes, each an Int (interp/bytes.rs,
   line for line). count NaN: to the end; fewer at the end */
VIV_S_UNUSED static viv_arr viv_s_read_bytes(viv_str path, double off, double count, int site) {
    path = VIV_S(path);
    errno = 0;
    FILE* f = fopen(path->s, "rb");
    if (!f) viv_s_ioerr(path, site);
    if (fseeko(f, 0, SEEK_END) != 0) { fclose(f); viv_s_ioerr(path, site); }
    long long n = (long long)ftello(f);
    if (n < 0) { fclose(f); viv_s_ioerr(path, site); }
    int whole_off = off >= 0.0 && floor(off) == off && off < 9.0e15;
    int whole_count = count != count || (count >= 0.0 && floor(count) == count && count < 9.0e15);
    if (!whole_off || !whole_count) { fclose(f); viv_s_fail(VIV_E_VALUE, "`read_bytes` takes a whole offset and count of 0 or more", site); }
    /* Python's f.seek(off); f.read(count): fewer bytes where the file ends */
    long long o = (long long)off < n ? (long long)off : n;
    long long c = count != count || (long long)count > n - o ? n - o : (long long)count;
    if (fseeko(f, (off_t)o, SEEK_SET) != 0) { fclose(f); viv_s_ioerr(path, site); }
    int dims[1];
    dims[0] = (int)c;
    viv_arr a = viv_arr_new(1, dims);
    long long* x = (long long*)a.data;
    unsigned char buf[65536];
    long long at = 0;
    while (at < c) {
        size_t want = (size_t)(c - at < (long long)sizeof buf ? c - at : (long long)sizeof buf);
        size_t k = fread(buf, 1, want, f);
        if (k == 0) { fclose(f); viv_s_ioerr(path, site); }
        for (size_t j = 0; j < k; j++) x[at + (long long)j] = buf[j];
        at += (long long)k;
    }
    fclose(f);
    return a;
}
VIV_S_UNUSED static double viv_s_write_text(viv_str path, viv_str text, int site) {
    path = VIV_S(path); text = VIV_S(text);
    errno = 0;
    FILE* f = fopen(path->s, "wb");
    if (!f) viv_s_ioerr(path, site);
    if (text->len && fwrite(text->s, 1, (size_t)text->len, f) != (size_t)text->len) { fclose(f); viv_s_ioerr(path, site); }
    if (fclose(f) != 0) viv_s_ioerr(path, site);
    return (double)text->nch;
}
//@ str_print: str_core print
/* print with string parts: viv_print's tags, and 6 for a viv_str (5 is a
   map's, in the maps fragment's viv_print_m) */
VIV_S_UNUSED static double viv_prints(int n, ...) {
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
        else if (tag == 6) { viv_str s = VIV_S(va_arg(ap, viv_str)); viv_fwrite(s->s, 1, (size_t)s->len, stdout); }
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
//@ str_map_print: str_core print maps
/* print with both string and map parts: viv_print's tags, 5 for a map and
   its two formats (as viv_print_m), 6 for a viv_str (as viv_prints) */
VIV_S_UNUSED static double viv_print_sm(int n, ...) {
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
        else if (tag == 5) {
            viv_map* m = va_arg(ap, viv_map*);
            int kk = va_arg(ap, int), kp = va_arg(ap, int), kw = va_arg(ap, int);
            int vk = va_arg(ap, int), vp = va_arg(ap, int), vw = va_arg(ap, int);
            viv_pmap(m, kk, kp, kw, vk, vp, vw);
        }
        else if (tag == 6) { viv_str s = VIV_S(va_arg(ap, viv_str)); viv_fwrite(s->s, 1, (size_t)s->len, stdout); }
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
//@ str_figure: str_core figure anim
/* a figure's title and labels, and an animation verb's text, as values */
VIV_S_UNUSED static double viv_figure_s(viv_str t, viv_str x, viv_str y) { return viv_figure(VIV_S(t)->s, VIV_S(x)->s, VIV_S(y)->s); }
VIV_S_UNUSED static double viv_figure_win_s(viv_str t, viv_str x, viv_str y, double a, double b, double c, double d, viv_str st) {
    return viv_figure_win(VIV_S(t)->s, VIV_S(x)->s, VIV_S(y)->s, a, b, c, d, VIV_S(st)->s);
}
/* up to three texts: a diagram's `note` has an id, its text and a style */
VIV_S_UNUSED static double viv_anim_s(int site, const char* verb, int nt, viv_str t0, viv_str t1, viv_str t2, int full, int na, viv_arr a0, viv_arr a1, viv_arr a2, int nn, double n0, double n1, double n2, double n3, double n4, double n5) {
    viv_sb b = {0, 0, 0, site};
    viv_sb_cstr(&b, verb);
    viv_sb_put(&b, ":", 1);
    viv_str ts[3];
    ts[0] = t0; ts[1] = t1; ts[2] = t2;
    for (int k = 0; k < nt; k++) {
        viv_str t = VIV_S(ts[k]);
        if (memchr(t->s, '\n', (size_t)t->len) || viv_s_find_bytes(t->s, t->len, " ; ", 3, 0) >= 0) {
            free(b.p);
            viv_s_fail(VIV_E_VALUE, "text in an animation cannot contain a newline or ` ; `", site);
        }
        viv_sb_put(&b, k ? " ; " : " ", k ? 3 : 1);
        viv_sb_put(&b, t->s, t->len);
    }
    viv_sb_put(&b, "", 1);
    double r = viv_anim(b.p, full, na, a0, a1, a2, nn, n0, n1, n2, n3, n4, n5);
    free(b.p);
    return r;
}
