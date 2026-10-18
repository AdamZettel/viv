/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* PROTOTYPE pwdft.h: the C interface of pwdft.c.
   Hand-written here; `viv export` would write it. */
#ifndef PWDFT_H
#define PWDFT_H
#include <stddef.h>

#define PWDFT_ABI 0          /* prototype; the first real one is 1 */

typedef struct {
    double ecut;             /* plane-wave cutoff, hartree (deck: 10) */
    double nkmesh;           /* the k-mesh is nkmesh^3, shifted (deck: 2) */
    double alat;             /* lattice constant, bohr (deck: 10.26) */
    /* every line the program prints goes here; NULL drops them. A nonzero
       return stops the run (status PWDFT_INTERRUPTED). */
    int (*on_line)(const char *line, void *ctx);
    void *ctx;
} pwdft_params;

typedef struct {
    double etot;             /* total energy, hartree */
    double efermi;           /* Fermi level (valence-band top), hartree */
    double gap_ev;           /* smallest gap along the band path, eV */
    double iter;             /* SCF iterations */
    double drho;             /* last density change */
    int tests_passed, tests_failed;
    int npts, nbands;        /* shape of bands_ev */
    double *bands_ev;        /* npts x nbands, row-major, owned by the library */
} pwdft_results;

enum { PWDFT_OK = 0, PWDFT_ERROR = 1, PWDFT_INTERRUPTED = 2 };

/* Runs the whole program once. On PWDFT_ERROR, msg holds Viv's message with
   its source line. Not thread-safe in the prototype (file-scope state). */
int pwdft_run(const pwdft_params *p, pwdft_results *r, char *msg, size_t cap);
void pwdft_free(pwdft_results *r);
int pwdft_abi(void);
#endif
