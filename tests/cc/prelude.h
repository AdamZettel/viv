/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2019 Rui Ueyama */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* Included before every program of tests/cc when clang builds it for the
 * host: the one runtime function lib/cc/codegen.viv provides. */
#include <stdio.h>

static int print_int(int x) {
    printf("%d\n", x);
    return x;
}
