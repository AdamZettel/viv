# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The C written after `viv export`'s kernel: the schema, row-major entries,
a CSV command line and the self-test, all in C99 with libc only.

The kernel `<p>_score` takes the float columns as a rows x NF array and the
integer columns and category codes as rows * NI int64 values. This file adds:

    <p>_predict(xf, xi, rows, out, msg, cap)        rows predictions
    <p>_predict_proba(xf, xi, rows, out, msg, cap)  rows x K probabilities
    <p>_category_code(col, text)                    a category's code, -1 unknown
    <p>_schema_json()                               the schema, as JSON
    main: `./model < in.csv > out.csv`, `./model --selftest`, `./model --schema`

`-D<P>_NO_MAIN` leaves main out (the Python wrapper builds it that way).
"""
from __future__ import annotations

import json
import string


def c_str(s: str) -> str:
    out = []
    for ch in s.encode("utf-8"):
        c = chr(ch)
        if c in '"\\':
            out.append("\\" + c)
        elif 32 <= ch < 127 and c != "?":
            out.append(c)
        else:
            out.append("\\%03o" % ch)
    return '"' + "".join(out) + '"'


def hexf(x: float) -> str:
    x = float(x)
    if x != x:
        return "NAN"
    if x in (float("inf"), float("-inf")):
        return "-INFINITY" if x < 0 else "INFINITY"
    return x.hex()


TEMPLATE = r"""
/* ======================================================================
   $title: the model's schema, its row-major C entries, a CSV command line
   and a self-test. Written by `vipyr compile` after `viv export`; C99 and
   libc only. Build: cc -O2 -ffp-contract=off $file -o $p
   ====================================================================== */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ${P}_NCOLS $ncols
#define ${P}_NF $nf
#define ${P}_NI $ni
#define ${P}_NOUT $nout
#define ${P}_NCLASSES $nclasses

/* kind: 0 float (a double), 1 integer (int64), 2 category (a string, read as its code) */
static const char* const ${p}_col_name[${P}_NCOLS] = { $col_names };
static const int ${p}_col_kind[${P}_NCOLS] = { $col_kinds };
static const int ${p}_col_slot[${P}_NCOLS] = { $col_slots };
static const int ${p}_col_ncat[${P}_NCOLS] = { $col_ncat };
static const char* const ${p}_cats[] = { $cats };
static const int ${p}_col_cat0[${P}_NCOLS] = { $col_cat0 };
static const char* const ${p}_classes[] = { $classes };
static const char* const ${p}_schema = $schema_json;

const char* ${p}_schema_json(void) { return ${p}_schema; }

/* the code of category `text` in column `col` (its position in the encoder's
   categories), or -1 when the encoder did not see it in training */
int64_t ${p}_category_code(int col, const char* text) {
    int k;
    if (col < 0 || col >= ${P}_NCOLS || ${p}_col_kind[col] != 2) return -1;
    for (k = 0; k < ${p}_col_ncat[col]; k++)
        if (strcmp(${p}_cats[${p}_col_cat0[col] + k], text) == 0) return k;
    return -1;
}

static int ${p}_run(const double* xf, const int64_t* xi, int64_t rows, ${p}_score_results* r, char* msg, size_t cap) {
    ${p}_score_args a;
    int64_t dims_f[2], dims_i[1];
    static const double no_f[1] = {0.0};
    static const int64_t no_i[1] = {0};
    dims_f[0] = rows; dims_f[1] = ${P}_NF;
    dims_i[0] = rows * ${P}_NI;
    a.xf.data = xf ? xf : no_f; a.xf.rank = 2; a.xf.dims = dims_f;
    a.xi.data = (const double*)(const void*)(xi ? xi : no_i); a.xi.rank = 1; a.xi.dims = dims_i;
    a.rows = rows;
    return ${p}_score(&a, r, msg, cap);
}

/* rows predictions into out: the value for a regressor, the class's index
   (into ${p}_classes) for a classifier. 0 on success; otherwise msg says why. */
int ${p}_predict(const double* xf, const int64_t* xi, int64_t rows, double* out, char* msg, size_t cap) {
    ${p}_score_results r;
    int64_t i;
    int s = ${p}_run(xf, xi, rows, &r, msg, cap);
    if (s) return s;
    for (i = 0; i < rows; i++) out[i] = r.result.data[i * ${P}_NOUT + (${P}_NOUT - 1)];
    ${p}_score_free(&r);
    return 0;
}

/* rows x NCLASSES probabilities into out (a classifier only) */
int ${p}_predict_proba(const double* xf, const int64_t* xi, int64_t rows, double* out, char* msg, size_t cap) {
    ${p}_score_results r;
    int64_t i, k;
    int s;
    if (${P}_NCLASSES == 0) {
        if (msg && cap) snprintf(msg, cap, "predict_proba: the model is a regressor");
        return 3;
    }
    s = ${p}_run(xf, xi, rows, &r, msg, cap);
    if (s) return s;
    for (i = 0; i < rows; i++)
        for (k = 0; k < ${P}_NCLASSES; k++) out[i * ${P}_NCLASSES + k] = r.result.data[i * ${P}_NOUT + k];
    ${p}_score_free(&r);
    return 0;
}

/* every output of the model: rows x NOUT (the probabilities, then the
   class's index, for a classifier) */
int ${p}_predict_all(const double* xf, const int64_t* xi, int64_t rows, double* out, char* msg, size_t cap) {
    ${p}_score_results r;
    int s = ${p}_run(xf, xi, rows, &r, msg, cap);
    if (s) return s;
    if (rows > 0) memcpy(out, r.result.data, (size_t)(rows * ${P}_NOUT) * sizeof(double));
    ${p}_score_free(&r);
    return 0;
}

#ifndef ${P}_NO_MAIN
/* ------------------------------------------------------------ CSV input */

typedef struct { char** cell; size_t n, cap; } ${p}_row;

/* split one line into cells: commas, double quotes with "" inside */
static int ${p}_split(char* line, ${p}_row* r) {
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

static int ${p}_parse_float(const char* c, double* v) {
    char* end;
    while (*c == ' ') c++;
    if (*c == 0) { *v = NAN; return 1; }            /* empty: missing, as pandas reads it */
    errno = 0;
    *v = strtod(c, &end);
    while (*end == ' ') end++;
    return end != c && *end == 0;
}

static int ${p}_parse_int(const char* c, int64_t* v) {
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
typedef struct { double* xf; int64_t* xi; int64_t rows; } ${p}_table;

static int ${p}_read(char** lines, size_t nlines, int ignore_extra, ${p}_table* t, const char* src) {
    ${p}_row r = {0, 0, 0};
    int where[${P}_NCOLS + 1];
    int* pos = 0;
    size_t ncell, i;
    int c, status = 0;
    size_t cap = 0;
    t->xf = 0; t->xi = 0; t->rows = 0;
    if (nlines == 0) { fprintf(stderr, "$p: %s: no header line; the columns are: %s\n", src, ${p}_schema); return 2; }
    if (${p}_split(lines[0], &r) != 0) { fprintf(stderr, "$p: %s: the header line is not CSV\n", src); return 2; }
    ncell = r.n;
    pos = (int*)malloc((ncell ? ncell : 1) * sizeof(int));
    for (c = 0; c < ${P}_NCOLS; c++) where[c] = -1;
    for (i = 0; i < ncell; i++) {
        pos[i] = -1;
        for (c = 0; c < ${P}_NCOLS; c++)
            if (strcmp(r.cell[i], ${p}_col_name[c]) == 0) {
                if (where[c] >= 0) { fprintf(stderr, "$p: %s: column %s appears twice\n", src, ${p}_col_name[c]); status = 2; }
                where[c] = (int)i; pos[i] = c;
            }
        if (pos[i] < 0 && !ignore_extra) {
            fprintf(stderr, "$p: %s: column '%s' is not one the model reads (pass --ignore-extra to skip it)\n", src, r.cell[i]);
            status = 2;
        }
    }
    for (c = 0; c < ${P}_NCOLS; c++)
        if (where[c] < 0) {
            fprintf(stderr, "$p: %s: missing column '%s' (%s)\n", src, ${p}_col_name[c],
                    ${p}_col_kind[c] == 0 ? "float" : ${p}_col_kind[c] == 1 ? "integer" : "category");
            status = 2;
        }
    if (status) { free(pos); free(r.cell); return status; }
    for (i = 1; i < nlines; i++) {
        size_t k;
        if (lines[i][0] == 0) continue;
        if (${p}_split(lines[i], &r) != 0 || r.n != ncell) {
            fprintf(stderr, "$p: %s: line %lu has %lu cells; the header has %lu\n", src, (unsigned long)(i + 1),
                    (unsigned long)r.n, (unsigned long)ncell);
            status = 3; break;
        }
        if ((size_t)t->rows == cap) {
            size_t nc = cap ? 2 * cap : 1024;
            double* f = (double*)realloc(t->xf, (nc * ${P}_NF + 1) * sizeof(double));
            int64_t* g = (int64_t*)realloc(t->xi, (nc * ${P}_NI + 1) * sizeof(int64_t));
            if (f) t->xf = f;
            if (g) t->xi = g;
            if (!f || !g) { fprintf(stderr, "$p: out of memory\n"); status = 3; break; }
            cap = nc;
        }
        for (k = 0; k < ncell; k++) {
            int col = pos[k];
            const char* cell = r.cell[k];
            if (col < 0) continue;
            if (${p}_col_kind[col] == 0) {
                if (!${p}_parse_float(cell, &t->xf[t->rows * ${P}_NF + ${p}_col_slot[col]])) {
                    fprintf(stderr, "$p: %s: line %lu, column %s: '%s' is not a number\n", src, (unsigned long)(i + 1), ${p}_col_name[col], cell);
                    status = 3;
                }
            } else if (${p}_col_kind[col] == 1) {
                if (!${p}_parse_int(cell, &t->xi[t->rows * ${P}_NI + ${p}_col_slot[col]])) {
                    fprintf(stderr, "$p: %s: line %lu, column %s: '%s' is not an integer\n", src, (unsigned long)(i + 1), ${p}_col_name[col], cell);
                    status = 3;
                }
            } else {
                t->xi[t->rows * ${P}_NI + ${p}_col_slot[col]] = ${p}_category_code(col, cell);
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
static char** ${p}_lines(FILE* f, size_t* n, char** buf) {
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

#define ${P}_ST_ROWS $st_rows
static const char* const ${p}_st_csv[] = {
$st_csv
};
/* scikit-learn's outputs for those rows (rows x NOUT), as hex floats */
static const double ${p}_st_expect[] = {
$st_expect
};
/* the largest |difference| allowed from scikit-learn's value: 0 means the bits */
static const double ${p}_st_tol = $st_tol;
$st_own_decl

static int ${p}_selftest(void) {
    ${p}_table t;
    double* out;
    char msg[512];
    char* copy[${P}_ST_ROWS + 1];
    int64_t i, bad = 0, badown = 0, n;
    int s, k;
    double worst = 0.0;
    for (i = 0; i <= ${P}_ST_ROWS; i++) {
        size_t m = strlen(${p}_st_csv[i]) + 1;
        copy[i] = (char*)malloc(m);
        memcpy(copy[i], ${p}_st_csv[i], m);
    }
    s = ${p}_read(copy, ${P}_ST_ROWS + 1, 0, &t, "selftest");
    if (s) return s;
    n = t.rows * ${P}_NOUT;
    out = (double*)malloc((size_t)(n + 1) * sizeof(double));
    s = ${p}_predict_all(t.xf, t.xi, t.rows, out, msg, sizeof msg);
    if (s) { fprintf(stderr, "$p: selftest: the model stopped: %s\n", msg); return 4; }
    for (i = 0; i < n; i++) {
        double e = ${p}_st_expect[i], g = out[i], d = g > e ? g - e : e - g;
        int same = memcmp(&e, &g, sizeof e) == 0 || (e != e && g != g);
        if (!same && !(d <= ${p}_st_tol)) {
            if (bad < 5) fprintf(stderr, "$p: selftest: row %ld, output %ld: %.17g, scikit-learn gave %.17g\n",
                                 (long)(i / ${P}_NOUT), (long)(i % ${P}_NOUT), g, e);
            bad++;
        }
        if (!same && d > worst) worst = d;
$st_own_check
    }
    for (k = 0; k <= ${P}_ST_ROWS; k++) free(copy[k]);
    if (bad || badown) {
        fprintf(stdout, "selftest FAILED: %ld of %ld values differ from scikit-learn's beyond %g, %ld from this file's own\n",
               (long)bad, (long)n, ${p}_st_tol, (long)badown);
        return 1;
    }
    if (${p}_st_tol == 0.0)
        fprintf(stdout, "selftest passed: %ld rows, %ld values, every one bit-identical to scikit-learn's\n", (long)t.rows, (long)n);
    else
        fprintf(stdout, "selftest passed: %ld rows, %ld values; largest difference from scikit-learn %.3g (allowed %g), "
               "and bit-identical to the values recorded at compile time\n", (long)t.rows, (long)n, worst, ${p}_st_tol);
    free(out); free(t.xf); free(t.xi);
    return 0;
}

static const char* ${p}_usage =
    "usage: $p [--ignore-extra] < input.csv > predictions.csv\n"
    "       $p --selftest     re-run the embedded check rows against scikit-learn's outputs\n"
    "       $p --schema       print the input columns as JSON\n"
    "exit status: 0 ok; 1 selftest failed; 2 wrong or missing columns; 3 a bad value; 4 the model refused a row\n";

/* the export renames a program's main and poisons printf: this file's own main is the command line */
#undef main
int main(int argc, char** argv) {
    int a, ignore_extra = 0, s;
    size_t nl = 0;
    char* buf = 0;
    char** lines;
    ${p}_table t;
    double* out;
    char msg[512];
    int64_t i;
    int k;
    for (a = 1; a < argc; a++) {
        if (strcmp(argv[a], "--selftest") == 0) return ${p}_selftest();
        if (strcmp(argv[a], "--schema") == 0) { fprintf(stdout, "%s\n", ${p}_schema); return 0; }
        if (strcmp(argv[a], "--ignore-extra") == 0) { ignore_extra = 1; continue; }
        fputs(${p}_usage, strcmp(argv[a], "--help") == 0 ? stdout : stderr);
        return strcmp(argv[a], "--help") == 0 ? 0 : 2;
    }
    lines = ${p}_lines(stdin, &nl, &buf);
    if (!lines) { fprintf(stderr, "$p: cannot read stdin\n"); return 3; }
    s = ${p}_read(lines, nl, ignore_extra, &t, "stdin");
    if (s) return s;
    out = (double*)malloc((size_t)(t.rows * ${P}_NOUT + 1) * sizeof(double));
    s = ${p}_predict_all(t.xf, t.xi, t.rows, out, msg, sizeof msg);
    if (s) { fprintf(stderr, "$p: %s\n", msg); return 4; }
    if (${P}_NCLASSES > 0) {
        fprintf(stdout, "prediction");
        for (k = 0; k < ${P}_NCLASSES; k++) fprintf(stdout, ",proba_%s", ${p}_classes[k]);
        fprintf(stdout, "\n");
        for (i = 0; i < t.rows; i++) {
            fprintf(stdout, "%s", ${p}_classes[(int)out[i * ${P}_NOUT + ${P}_NCLASSES]]);
            for (k = 0; k < ${P}_NCLASSES; k++) fprintf(stdout, ",%.17g", out[i * ${P}_NOUT + k]);
            fprintf(stdout, "\n");
        }
    } else {
        fprintf(stdout, "prediction\n");
        for (i = 0; i < t.rows; i++) fprintf(stdout, "%.17g\n", out[i]);
    }
    return 0;
}
#endif
"""


def render(p, file, title, schema, classes, nout, st_csv_lines, st_expect, st_tol, st_own=None):
    cols = schema
    cats, cat0 = [], []
    for c in cols:
        cat0.append(str(len(cats)))
        if c.kind == "category":
            cats += [c_str(str(x)) for x in c.categories]
    sj = json.dumps({"columns": [{"name": c.name, "kind": c.kind, **({"categories": [str(x) for x in c.categories]}
                                                                       if c.kind == "category" else {})} for c in cols],
                     "classes": classes, "outputs": nout})
    own_decl = ""
    own_check = ""
    if st_own is not None:
        own_decl = ("/* this file's own outputs at compile time, which the self-test requires to the bit */\n"
                    "static const double %s_st_own[] = {\n%s\n};" % (p, _rows(st_own)))
        own_check = ("        if (memcmp(&g, &%s_st_own[i], sizeof g) != 0 && !(g != g && %s_st_own[i] != %s_st_own[i])) badown++;"
                     % (p, p, p))
    nf = sum(1 for c in cols if c.kind == "float")
    return string.Template(TEMPLATE).substitute(
        title=title, file=file, p=p, P=p.upper(), ncols=len(cols), nf=nf, ni=len(cols) - nf, nout=nout,
        nclasses=len(classes),
        col_names=", ".join(c_str(c.name) for c in cols),
        col_kinds=", ".join({"float": "0", "int": "1", "category": "2"}[c.kind] for c in cols),
        col_slots=", ".join(str(c.slot) for c in cols),
        col_ncat=", ".join(str(len(c.categories) if c.kind == "category" else 0) for c in cols),
        cats=", ".join(cats) or "0", col_cat0=", ".join(cat0),
        classes=", ".join(c_str(c) for c in classes) or "0",
        schema_json=c_str(sj),
        st_rows=len(st_csv_lines) - 1,
        st_csv=",\n".join("    " + c_str(l) for l in st_csv_lines),
        st_expect=_rows(st_expect), st_tol=repr(float(st_tol)) if st_tol else "0.0",
        st_own_decl=own_decl, st_own_check=own_check)


def _rows(vals, per=4):
    v = [hexf(x) for x in vals]
    return "\n".join("    " + ", ".join(v[i:i + per]) + "," for i in range(0, len(v), per)) or "    0.0"
