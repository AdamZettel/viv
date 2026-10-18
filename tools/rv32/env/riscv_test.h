/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
// A minimal riscv-tests environment for Viv's RV32IM simulator
// (lib/rv32/sim.viv): user mode only, no traps, no CSRs. The suites'
// own environments (riscv-test-env, env/p) set up machine mode; this one
// starts at _start and reports through the exit ecall: a0 = 0 passes,
// otherwise a0 is the number of the failing test case.
#define RVTEST_RV32U
#define RVTEST_RV64U
#define RVTEST_RV32M
#define RVTEST_RV64M
#define TESTNUM gp
#define RVTEST_CODE_BEGIN .text; _start:
#define RVTEST_CODE_END
#define RVTEST_PASS li gp, 1; li a7, 93; li a0, 0; ecall
#define RVTEST_FAIL mv a0, gp; li a7, 93; ecall
#define RVTEST_DATA_BEGIN .data
#define RVTEST_DATA_END
