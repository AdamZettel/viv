# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""
A C compiler, an assembler and a RISC-V processor, all in Viv
=============================================================

Compile recursive Fibonacci from C to RV32IM assembly with `lib/cc`,
assemble it to machine code with `lib/rv32/asm.viv`, and run the machine
code on the instruction-set simulator `lib/rv32/sim.viv`, counting every
instruction it executes. The compiler follows the step order of Rui
Ueyama's chibicc; the simulator passes the riscv-tests rv32ui and rv32um
suites. `gallery/hardware/cc_fib.viv` ties the three together and is
loaded with `vp.load`.
"""

# %%
# ## The program and its result
#
# `fib_on_rv32(n)` compiles the C below with n filled in, runs it, and
# returns the exit code, the number of instructions executed and the count
# of each class of instruction. The exit code is main's return value,
# fib(n), compared here with Python's.

import numpy as np
import vipyr as vp
from vipyr.figure import figure

cc = vp.load("gallery/hardware/cc_fib.viv")
CLASSES = ["alu", "muldiv", "load", "store", "branch", "jump", "system"]
print(
    "int fib(int x) { if (x <= 1) return x; "
    "return fib(x - 1) + fib(x - 2); }"
)
print("int main() { return fib(n); }")
print("assembly lines for the program:", cc.asm_lines(10))


def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


ns = np.arange(0, 17)
runs = np.array([cc.fib_on_rv32(int(n)) for n in ns])
results, steps, mix = runs[:, 0], runs[:, 1], runs[:, 2:]
assert all(results[k] == fib(n) for k, n in enumerate(ns))
print("fib(n) from the simulator matches Python for n = 0 ..", ns[-1])

# %%
# ## Instructions grow with the number of calls
#
# fib(n) makes 2 fib(n+1) - 1 calls of fib: fib(n+1) of them are leaves
# (x <= 1) and the rest recurse. Per leaf and recursing call together the
# code runs 72 instructions, so the total is exactly 36 per call plus a
# constant 5 for the start-up code and main: the cost of one call in this
# compiler's stack-machine code.

calls = np.array([2 * fib(int(n) + 1) - 1 for n in ns])
assert np.array_equal(steps, 36 * calls + 5)
print("instructions = 36 * calls + 5 for every n")
print("  n   fib(n)   calls   instructions   per call")
for n, f, c, s in zip(ns, results, calls, steps):
    print("%3d %8d %7d %14d %10.2f" % (n, f, c, s, s / c))
fig = figure("Instructions executed for fib(n)", "n", "instructions")
fig.yscale("log")
fig.plot(ns, steps, "instructions")
fig.plot(ns, calls, "calls of fib")
fig.style("instructions", "s1 thick")
fig.style("calls of fib", "s2")

# %%
# ## What the processor spends them on
#
# The instruction mix of fib(16). The code generator keeps every
# intermediate value on the stack (push before a binary operator, pop
# after), so loads and stores are a large share; branches come from the
# `if`, jumps from calls and returns. There is no multiply or divide in
# this program, so the M-extension column is empty.

last = mix[-1]
for name, count in zip(CLASSES, last):
    print("%-7s %7d  %5.1f%%" % (name, count, 100.0 * count / steps[-1]))
fig = figure("Instruction mix of fib(16)", "", "instructions")
fig.bars(np.arange(len(CLASSES)), last, "count")
for k, name in enumerate(CLASSES):
    fig.xtick(name, k)
