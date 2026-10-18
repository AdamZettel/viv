/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
//@ str_db: str_list str_repr arr_new
/* VIV_USES_SQLITE: the db_* builtins (crate::db),
   the C twin of lang/viv/src/db.rs rule for rule and message for message.
   SQLite itself is third_party/sqlite/sqlite3.c, compiled beside this file
   (`viv export --with-sqlite`, or by hand with the flags in its README). */
typedef struct sqlite3 sqlite3;
typedef struct sqlite3_stmt sqlite3_stmt;
int sqlite3_open_v2(const char*, sqlite3**, int, const char*);
int sqlite3_close_v2(sqlite3*);
const char* sqlite3_errmsg(sqlite3*);
int sqlite3_exec(sqlite3*, const char*, int (*)(void*, int, char**, char**), void*, char**);
int sqlite3_prepare_v2(sqlite3*, const char*, int, sqlite3_stmt**, const char**);
int sqlite3_finalize(sqlite3_stmt*);
int sqlite3_reset(sqlite3_stmt*);
int sqlite3_step(sqlite3_stmt*);
int sqlite3_bind_parameter_count(sqlite3_stmt*);
int sqlite3_bind_text(sqlite3_stmt*, int, const char*, int, void (*)(void*));
int sqlite3_bind_double(sqlite3_stmt*, int, double);
int sqlite3_bind_int64(sqlite3_stmt*, int, long long);
int sqlite3_bind_null(sqlite3_stmt*, int);
int sqlite3_column_count(sqlite3_stmt*);
int sqlite3_column_type(sqlite3_stmt*, int);
double sqlite3_column_double(sqlite3_stmt*, int);
long long sqlite3_column_int64(sqlite3_stmt*, int);
const unsigned char* sqlite3_column_text(sqlite3_stmt*, int);
int sqlite3_column_bytes(sqlite3_stmt*, int);
int sqlite3_total_changes(sqlite3*);
#define VIV_DB_TRANSIENT ((void (*)(void*))-1)

/* a staged column: kind 0 numbers, 1 Ints, 2 strings, copied off the arena */
typedef struct { char* name; int kind; long n; double* f; long long* i; char** s; int* len; } viv_db_col;
typedef struct { sqlite3* db; int open; int ncol; viv_db_col cols[64]; } viv_db_conn;
static viv_db_conn* viv_db_conns;
static int viv_db_nconn;

VIV_S_NORETURN static void viv_db_fail(int kind, const char* who, const char* m, int site) {
    viv_sb b = {0, 0, 0, site};
    viv_sb_cstr(&b, who);
    viv_sb_cstr(&b, ": ");
    viv_sb_cstr(&b, m);
    viv_sb_put(&b, "", 1);
    viv_s_fail(kind, b.p, site);
}
static void viv_db_num(char* buf, size_t n, double x) {
    if (x == (double)(long long)x && fabs(x) < 1e15) snprintf(buf, n, "%lld", (long long)x);
    else snprintf(buf, n, "%.17g", x);
}
static viv_db_conn* viv_db_get(double h, int site) {
    long k = (long)h;
    if (!(h >= 1.0 && h == (double)k && k <= viv_db_nconn) || !viv_db_conns[k - 1].open) {
        char buf[64], m[128];
        viv_db_num(buf, sizeof buf, h);
        snprintf(m, sizeof m, "%s is not an open database", buf);
        viv_db_fail(VIV_E_VALUE, "db", m, site);
    }
    return &viv_db_conns[k - 1];
}
/* a string's bytes, NUL-terminated, on the heap (freed by the caller) */
static char* viv_db_cstr(viv_str s, const char* who, int site) {
    s = VIV_S(s);
    if (memchr(s->s, 0, (size_t)s->len)) viv_db_fail(VIV_E_VALUE, who, "a NUL character in the text", site);
    char* p = (char*)malloc((size_t)s->len + 1);
    if (!p) viv_s_nomem();
    memcpy(p, s->s, (size_t)s->len);
    p[s->len] = 0;
    return p;
}
VIV_S_UNUSED static double viv_s_db_open(viv_str path, int site) {
    char* p = viv_db_cstr(path, "db_open", site);
    sqlite3* db = 0;
    int rc = sqlite3_open_v2(p, &db, 0x2 | 0x4, 0);
    free(p);
    if (rc != 0) {
        viv_sb b = {0, 0, 0, site};
        viv_sb_cstr(&b, "db_open: ");
        viv_s_repr_into(&b, VIV_S(path));
        viv_sb_cstr(&b, ": ");
        viv_sb_cstr(&b, db ? sqlite3_errmsg(db) : "out of memory");
        viv_sb_put(&b, "", 1);
        sqlite3_close_v2(db);
        viv_s_fail(VIV_E_ERROR, b.p, site);
    }
    viv_db_conns = (viv_db_conn*)realloc(viv_db_conns, sizeof(viv_db_conn) * (size_t)(viv_db_nconn + 1));
    if (!viv_db_conns) viv_s_nomem();
    memset(&viv_db_conns[viv_db_nconn], 0, sizeof(viv_db_conn));
    viv_db_conns[viv_db_nconn].db = db;
    viv_db_conns[viv_db_nconn].open = 1;
    return (double)++viv_db_nconn;
}
static void viv_db_unstage(viv_db_conn* c) {
    for (int j = 0; j < c->ncol; j++) {
        viv_db_col* x = &c->cols[j];
        if (x->s) for (long r = 0; r < x->n; r++) free(x->s[r]);
        free(x->name); free(x->f); free(x->i); free(x->s); free(x->len);
    }
    c->ncol = 0;
}
VIV_S_UNUSED static double viv_s_db_close(double h, int site) {
    viv_db_conn* c = viv_db_get(h, site);
    viv_db_unstage(c);
    sqlite3_close_v2(c->db);
    c->open = 0;
    return 0.0;
}
/* parameters: kind 0 none, 1 strings, 2 numbers */
typedef struct { int kind; viv_strs s; viv_arr a; } viv_db_params;
static int viv_db_blank(const char* t) {
    for (; *t; t++) if (!(*t == ' ' || *t == '\t' || *t == '\n' || *t == '\r' || *t == '\f' || *t == '\v' || *t == ';')) return 0;
    return 1;
}
static void viv_db_bind(sqlite3_stmt* st, viv_db_params* p, const char* who, int site) {
    long want = sqlite3_bind_parameter_count(st);
    long n = p->kind == 1 ? VIV_SS(p->s)->n : p->kind == 2 ? p->a.size : 0;
    if (n != want) {
        char m[160];
        snprintf(m, sizeof m, "the statement takes %ld parameters and was given %ld", want, n);
        sqlite3_finalize(st);
        viv_db_fail(VIV_E_VALUE, who, m, site);
    }
    for (long k = 0; k < n; k++) {
        if (p->kind == 1) { viv_str s = VIV_S(VIV_SS(p->s)->items[k]); sqlite3_bind_text(st, (int)k + 1, s->s, s->len, VIV_DB_TRANSIENT); }
        else if (p->a.data[k] != p->a.data[k]) sqlite3_bind_null(st, (int)k + 1);
        else sqlite3_bind_double(st, (int)k + 1, p->a.data[k]);
    }
}
static void viv_db_plain(sqlite3* db, const char* sql) { sqlite3_exec(db, sql, 0, 0, 0); }
/* fail inside a savepoint: roll it back first, so the call changed nothing */
VIV_S_NORETURN static void viv_db_abort(sqlite3* db, sqlite3_stmt* st, int kind, const char* who, const char* m, int site) {
    viv_sb b = {0, 0, 0, site};
    viv_sb_cstr(&b, who);
    viv_sb_cstr(&b, ": ");
    viv_sb_cstr(&b, m);
    viv_sb_put(&b, "", 1);
    if (st) sqlite3_finalize(st);
    viv_db_plain(db, "ROLLBACK TO viv_db");
    viv_db_plain(db, "RELEASE viv_db");
    viv_s_fail(kind, b.p, site);
}
static double viv_db_exec_p(double h, viv_str sql, viv_db_params p, int site) {
    viv_db_conn* c = viv_db_get(h, site);
    sqlite3* db = c->db;
    char* text = viv_db_cstr(sql, "db_exec", site);
    int before = sqlite3_total_changes(db);
    viv_db_plain(db, "SAVEPOINT viv_db");
    const char* rest = text;
    while (!viv_db_blank(rest)) {
        sqlite3_stmt* st = 0;
        const char* tail = 0;
        if (sqlite3_prepare_v2(db, rest, -1, &st, &tail) != 0) { free(text); viv_db_abort(db, 0, VIV_E_ERROR, "db_exec", sqlite3_errmsg(db), site); }
        if (st) {
            long want = sqlite3_bind_parameter_count(st);
            long n = p.kind == 1 ? VIV_SS(p.s)->n : p.kind == 2 ? p.a.size : 0;
            if (n != want) {
                char m[160];
                snprintf(m, sizeof m, "the statement takes %ld parameters and was given %ld", want, n);
                free(text);
                viv_db_abort(db, st, VIV_E_VALUE, "db_exec", m, site);
            }
            viv_db_bind(st, &p, "db_exec", site);
            if (p.kind && !viv_db_blank(tail)) { free(text); viv_db_abort(db, st, VIV_E_VALUE, "db_exec", "SQL with parameters is one statement", site); }
            int rc;
            while ((rc = sqlite3_step(st)) == 100) {}
            if (rc != 101) { free(text); viv_db_abort(db, st, VIV_E_ERROR, "db_exec", sqlite3_errmsg(db), site); }
            sqlite3_finalize(st);
        }
        rest = tail;
    }
    free(text);
    viv_db_plain(db, "RELEASE viv_db");
    return (double)(sqlite3_total_changes(db) - before);
}
VIV_S_UNUSED static double viv_s_db_exec0(double h, viv_str sql, int site) { viv_db_params p = {0}; return viv_db_exec_p(h, sql, p, site); }
VIV_S_UNUSED static double viv_s_db_exec(double h, viv_str sql, viv_strs ps, int site) { viv_db_params p = {0}; p.kind = 1; p.s = ps; return viv_db_exec_p(h, sql, p, site); }
VIV_S_UNUSED static double viv_s_db_exec_n(double h, viv_str sql, viv_arr ps, int site) { viv_db_params p = {0}; p.kind = 2; p.a = ps; return viv_db_exec_p(h, sql, p, site); }

/* the one statement of a query giving one column, prepared and bound */
static sqlite3_stmt* viv_db_query(viv_db_conn* c, viv_str sql, viv_db_params* p, const char* who, int site) {
    char* text = viv_db_cstr(sql, who, site);
    sqlite3_stmt* st = 0;
    const char* tail = 0;
    if (sqlite3_prepare_v2(c->db, text, -1, &st, &tail) != 0) { free(text); viv_db_fail(VIV_E_ERROR, who, sqlite3_errmsg(c->db), site); }
    if (!st) { free(text); viv_db_fail(VIV_E_VALUE, who, "no statement", site); }
    int more = !viv_db_blank(tail);
    free(text);
    if (more) { sqlite3_finalize(st); viv_db_fail(VIV_E_VALUE, who, "a query is one statement", site); }
    viv_db_bind(st, p, who, site);
    int n = sqlite3_column_count(st);
    if (n != 1) {
        char m[96];
        snprintf(m, sizeof m, "the query gives %d columns; it reads one", n);
        sqlite3_finalize(st);
        viv_db_fail(VIV_E_VALUE, who, m, site);
    }
    return st;
}
static const char* viv_db_tname(int t) { return t == 3 ? "text" : t == 4 ? "a blob" : t == 5 ? "NULL" : "a number"; }
VIV_S_NORETURN static void viv_db_row_fail(sqlite3_stmt* st, const char* who, long row, const char* what, const char* tail, int site) {
    char m[200];
    snprintf(m, sizeof m, "row %ld holds %s, %s", row, what, tail);
    sqlite3_finalize(st);
    viv_db_fail(VIV_E_VALUE, who, m, site);
}
/* rows into a growing heap buffer of 8-byte cells */
static double viv_db_colx(double h, viv_str sql, viv_db_params p, int as_int, viv_arr* out, int site) {
    const char* who = as_int ? "db_col_int" : "db_col";
    viv_db_conn* c = viv_db_get(h, site);
    sqlite3_stmt* st = viv_db_query(c, sql, &p, who, site);
    long n = 0, cap = 16;
    double* v = (double*)malloc(sizeof(double) * (size_t)cap);
    if (!v) viv_s_nomem();
    int rc;
    while ((rc = sqlite3_step(st)) == 100) {
        if (n == cap) { cap *= 2; v = (double*)realloc(v, sizeof(double) * (size_t)cap); if (!v) viv_s_nomem(); }
        int t = sqlite3_column_type(st, 0);
        double x = sqlite3_column_double(st, 0);
        if (!as_int) {
            if (t == 5) v[n] = NAN;
            else if (t == 1 || t == 2) v[n] = x;
            else { free(v); viv_db_row_fail(st, who, n, viv_db_tname(t), "not a number", site); }
        }
        else {
            long long i;
            if (t == 1) i = sqlite3_column_int64(st, 0);
            else if (t == 2 && x == floor(x) && fabs(x) < 9.2e18) i = (long long)x;
            else if (t == 2) { char buf[64]; viv_db_num(buf, sizeof buf, x); free(v); viv_db_row_fail(st, who, n, buf, "not a whole number", site); }
            else { free(v); viv_db_row_fail(st, who, n, viv_db_tname(t), "not an Int", site); }
            memcpy(&v[n], &i, sizeof i);
        }
        n++;
    }
    if (rc != 101) { free(v); sqlite3_finalize(st); viv_db_fail(VIV_E_ERROR, who, sqlite3_errmsg(c->db), site); }
    sqlite3_finalize(st);
    int dims[1]; dims[0] = (int)n;
    *out = viv_arr_new(1, dims);
    if (n) memcpy(out->data, v, sizeof(double) * (size_t)n);
    free(v);
    return 0.0;
}
VIV_S_UNUSED static viv_arr viv_s_db_col0(double h, viv_str sql, int site) { viv_db_params p = {0}; viv_arr r; viv_db_colx(h, sql, p, 0, &r, site); return r; }
VIV_S_UNUSED static viv_arr viv_s_db_col(double h, viv_str sql, viv_strs ps, int site) { viv_db_params p = {0}; p.kind = 1; p.s = ps; viv_arr r; viv_db_colx(h, sql, p, 0, &r, site); return r; }
VIV_S_UNUSED static viv_arr viv_s_db_col_n(double h, viv_str sql, viv_arr ps, int site) { viv_db_params p = {0}; p.kind = 2; p.a = ps; viv_arr r; viv_db_colx(h, sql, p, 0, &r, site); return r; }
VIV_S_UNUSED static viv_arr viv_s_db_coli0(double h, viv_str sql, int site) { viv_db_params p = {0}; viv_arr r; viv_db_colx(h, sql, p, 1, &r, site); return r; }
VIV_S_UNUSED static viv_arr viv_s_db_coli(double h, viv_str sql, viv_strs ps, int site) { viv_db_params p = {0}; p.kind = 1; p.s = ps; viv_arr r; viv_db_colx(h, sql, p, 1, &r, site); return r; }
VIV_S_UNUSED static viv_arr viv_s_db_coli_n(double h, viv_str sql, viv_arr ps, int site) { viv_db_params p = {0}; p.kind = 2; p.a = ps; viv_arr r; viv_db_colx(h, sql, p, 1, &r, site); return r; }

static viv_strs viv_db_cols_p(double h, viv_str sql, viv_db_params p, int site) {
    const char* who = "db_col_str";
    viv_db_conn* c = viv_db_get(h, site);
    sqlite3_stmt* st = viv_db_query(c, sql, &p, who, site);
    long n = 0, cap = 16;
    viv_str* v = (viv_str*)malloc(sizeof(viv_str) * (size_t)cap);
    if (!v) viv_s_nomem();
    int rc;
    while ((rc = sqlite3_step(st)) == 100) {
        int t = sqlite3_column_type(st, 0);
        if (t == 5 || t == 4) { free(v); viv_db_row_fail(st, who, n, viv_db_tname(t), "not text", site); }
        if (n == cap) { cap *= 2; v = (viv_str*)realloc(v, sizeof(viv_str) * (size_t)cap); if (!v) viv_s_nomem(); }
        const unsigned char* s = sqlite3_column_text(st, 0);
        v[n++] = viv_s_make((const char*)s, sqlite3_column_bytes(st, 0), site);
    }
    if (rc != 101) { free(v); sqlite3_finalize(st); viv_db_fail(VIV_E_ERROR, who, sqlite3_errmsg(c->db), site); }
    sqlite3_finalize(st);
    viv_strs r = viv_ss_copyof(v, n, 0);
    free(v);
    return r;
}
VIV_S_UNUSED static viv_strs viv_s_db_cols0(double h, viv_str sql, int site) { viv_db_params p = {0}; return viv_db_cols_p(h, sql, p, site); }
VIV_S_UNUSED static viv_strs viv_s_db_cols(double h, viv_str sql, viv_strs ps, int site) { viv_db_params p = {0}; p.kind = 1; p.s = ps; return viv_db_cols_p(h, sql, p, site); }
VIV_S_UNUSED static viv_strs viv_s_db_cols_n(double h, viv_str sql, viv_arr ps, int site) { viv_db_params p = {0}; p.kind = 2; p.a = ps; return viv_db_cols_p(h, sql, p, site); }

static viv_db_col* viv_db_stage_new(double h, viv_str name, int kind, long n, int site) {
    viv_db_conn* c = viv_db_get(h, site);
    name = VIV_S(name);
    for (int j = 0; j < c->ncol; j++)
        if ((long)strlen(c->cols[j].name) == name->len && memcmp(c->cols[j].name, name->s, (size_t)name->len) == 0) {
            viv_sb b = {0, 0, 0, site};
            viv_sb_cstr(&b, "db_stage: column ");
            viv_s_repr_into(&b, name);
            viv_sb_cstr(&b, " is staged already");
            viv_sb_put(&b, "", 1);
            viv_s_fail(VIV_E_VALUE, b.p, site);
        }
    if (c->ncol == 64) viv_db_fail(VIV_E_VALUE, "db_stage", "at most 64 columns are staged", site);
    viv_db_col* x = &c->cols[c->ncol++];
    memset(x, 0, sizeof *x);
    x->name = viv_db_cstr(name, "db_stage", site);
    x->kind = kind;
    x->n = n;
    return x;
}
VIV_S_UNUSED static double viv_s_db_stage_f(double h, viv_str name, viv_arr a, int site) {
    viv_db_col* x = viv_db_stage_new(h, name, 0, a.size, site);
    x->f = (double*)malloc(sizeof(double) * (size_t)(a.size ? a.size : 1));
    if (!x->f) viv_s_nomem();
    if (a.size) memcpy(x->f, a.data, sizeof(double) * (size_t)a.size);
    return (double)a.size;
}
VIV_S_UNUSED static double viv_s_db_stage_i(double h, viv_str name, viv_arr a, int site) {
    viv_db_col* x = viv_db_stage_new(h, name, 1, a.size, site);
    x->i = (long long*)malloc(sizeof(long long) * (size_t)(a.size ? a.size : 1));
    if (!x->i) viv_s_nomem();
    if (a.size) memcpy(x->i, a.data, sizeof(long long) * (size_t)a.size);
    return (double)a.size;
}
VIV_S_UNUSED static double viv_s_db_stage_s(double h, viv_str name, viv_strs l, int site) {
    l = VIV_SS(l);
    viv_db_col* x = viv_db_stage_new(h, name, 2, l->n, site);
    x->s = (char**)malloc(sizeof(char*) * (size_t)(l->n ? l->n : 1));
    x->len = (int*)malloc(sizeof(int) * (size_t)(l->n ? l->n : 1));
    if (!x->s || !x->len) viv_s_nomem();
    for (long r = 0; r < l->n; r++) {
        viv_str s = VIV_S(l->items[r]);
        x->s[r] = (char*)malloc((size_t)s->len + 1);
        if (!x->s[r]) viv_s_nomem();
        memcpy(x->s[r], s->s, (size_t)s->len);
        x->len[r] = s->len;
    }
    return (double)l->n;
}
static void viv_db_quote(viv_sb* b, const char* s) {
    viv_sb_cstr(b, "\"");
    for (; *s; s++) { if (*s == '"') viv_sb_cstr(b, "\""); viv_sb_put(b, s, 1); }
    viv_sb_cstr(b, "\"");
}
VIV_S_UNUSED static double viv_s_db_insert(double h, viv_str table, int site) {
    viv_db_conn* c = viv_db_get(h, site);
    if (c->ncol == 0) viv_db_fail(VIV_E_VALUE, "db_insert", "no columns staged", site);
    long rows = c->cols[0].n;
    for (int j = 0; j < c->ncol; j++)
        if (c->cols[j].n != rows) {
            viv_sb b = {0, 0, 0, site};
            char m[96];
            viv_sb_cstr(&b, "db_insert: column ");
            viv_s_repr_into(&b, viv_s_make(c->cols[j].name, (long)strlen(c->cols[j].name), site));
            snprintf(m, sizeof m, " has %ld rows and column ", c->cols[j].n);
            viv_sb_cstr(&b, m);
            viv_s_repr_into(&b, viv_s_make(c->cols[0].name, (long)strlen(c->cols[0].name), site));
            snprintf(m, sizeof m, " has %ld", rows);
            viv_sb_cstr(&b, m);
            viv_sb_put(&b, "", 1);
            viv_db_unstage(c);
            viv_s_fail(VIV_E_VALUE, b.p, site);
        }
    char* tname = viv_db_cstr(table, "db_insert", site);
    viv_sb q = {0, 0, 0, site};
    viv_sb_cstr(&q, "INSERT INTO ");
    viv_db_quote(&q, tname);
    free(tname);
    viv_sb_cstr(&q, " (");
    for (int j = 0; j < c->ncol; j++) { if (j) viv_sb_cstr(&q, ", "); viv_db_quote(&q, c->cols[j].name); }
    viv_sb_cstr(&q, ") VALUES (");
    for (int j = 0; j < c->ncol; j++) viv_sb_cstr(&q, j ? ", ?" : "?");
    viv_sb_cstr(&q, ")");
    viv_sb_put(&q, "", 1);
    sqlite3* db = c->db;
    viv_db_plain(db, "SAVEPOINT viv_db");
    sqlite3_stmt* st = 0;
    if (sqlite3_prepare_v2(db, q.p, -1, &st, 0) != 0) { viv_db_unstage(c); viv_db_abort(db, 0, VIV_E_ERROR, "db_insert", sqlite3_errmsg(db), site); }
    for (long r = 0; r < rows; r++) {
        sqlite3_reset(st);
        for (int j = 0; j < c->ncol; j++) {
            viv_db_col* x = &c->cols[j];
            if (x->kind == 0) { if (x->f[r] != x->f[r]) sqlite3_bind_null(st, j + 1); else sqlite3_bind_double(st, j + 1, x->f[r]); }
            else if (x->kind == 1) sqlite3_bind_int64(st, j + 1, x->i[r]);
            else sqlite3_bind_text(st, j + 1, x->s[r], x->len[r], VIV_DB_TRANSIENT);
        }
        int rc;
        while ((rc = sqlite3_step(st)) == 100) {}
        if (rc != 101) { viv_db_unstage(c); viv_db_abort(db, st, VIV_E_ERROR, "db_insert", sqlite3_errmsg(db), site); }
    }
    sqlite3_finalize(st);
    viv_db_unstage(c);
    viv_db_plain(db, "RELEASE viv_db");
    return (double)rows;
}
