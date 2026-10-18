/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* PROTOTYPE entry, hand-written: what `viv export` would put after the
   program. One run = the program's main, from its first top-level line to
   its last, with the parameters read from vk_param. */
#undef main
#undef printf
#undef fputs
#undef puts
#undef putchar
#undef fprintf
#undef exit
#undef _exit

#define VK_EXPORT __attribute__((visibility("default")))

static void vk_reset(void) {
    vk_msg_n = 0; vk_msg[0] = 0;
    vk_line_n = 0;
    vk_npass = vk_nfail = 0;
    viv_fails = 0;              /* else the runtime's atexit handler ends Python with status 1 */
    viv_test_depth = 0;
    viv_rng_state = 12345.0;    /* the runtime's initial seed */
}

VK_EXPORT int pwdft_abi(void) { return PWDFT_ABI; }

VK_EXPORT int pwdft_run(const pwdft_params *p, pwdft_results *r, char *msg, size_t cap) {
    jmp_buf here;
    size_t mark = viv_arena_top;
    memset(r, 0, sizeof *r);
    vk_param = *p;
    vk_reset();
    vk_jmp = &here;
    int code = setjmp(here);
    if (code == 0) {
        pwdft_program_main();
        if (vk_line_n) vk_emit_line();
        vk_jmp = 0;
        r->etot = ETOT; r->efermi = EFERMI; r->gap_ev = GAP;
        r->iter = ITER; r->drho = DRHO;
        r->tests_passed = vk_npass; r->tests_failed = vk_nfail;
        r->npts = EVB.dims[0]; r->nbands = EVB.dims[1];
        r->bands_ev = (double *)malloc(sizeof(double) * (size_t)EVB.size);
        if (r->bands_ev) memcpy(r->bands_ev, EVB.data, sizeof(double) * (size_t)EVB.size);
        viv_fails = 0;
        viv_arena_release(mark, 0);   /* every array the run made */
        return PWDFT_OK;
    }
    vk_jmp = 0;
    viv_fails = 0;
    viv_test_depth = 0;
    viv_arena_release(mark, 0);
    if (msg && cap) snprintf(msg, cap, "%s", code == PWDFT_INTERRUPTED ? "interrupted" : vk_msg);
    r->tests_passed = vk_npass; r->tests_failed = vk_nfail;
    return code == PWDFT_INTERRUPTED ? PWDFT_INTERRUPTED : PWDFT_ERROR;
}

VK_EXPORT void pwdft_free(pwdft_results *r) {
    free(r->bands_ev);
    r->bands_ev = 0;
}
