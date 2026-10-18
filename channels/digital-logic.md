# Digital logic

Circuits written as programs: a gate is a function of its inputs, a wire is a variable holding a bit, and each circuit is tested against its full truth table.

The posts build up to a RISC-V processor in steps: gates, an adder, an arithmetic logic unit, a register file, an instruction decoder, then the loop that fetches and executes instructions. So far the gates and a four-bit adder are posted. A C compiler, an assembler and an RV32IM instruction-set simulator already exist as Viv library code, and the simulator passes the riscv-tests rv32ui and rv32um suites; the RISC-V example in the vipyr gallery runs a C program through all three.

## From a gate to an adder
- building-a-four-bit-adder-out-of-nand-gates — NAND into NOT, AND, OR and XOR, then a half adder, a full adder, and four in a row
