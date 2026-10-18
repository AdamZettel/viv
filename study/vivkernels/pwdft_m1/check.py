# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""PROTOTYPE check: libpwdft.so called from
Python against the same program built as an ordinary native executable and
run in the Viv interpreter.

    VIV=path/to/viv python3 check.py

  1. ecut 10 (the deck's value): the library's printed lines against the
     native executable's stdout, byte for byte, and its ETOT against the
     native and the interpreter's ETOT, bit for bit.
  2. ecut 8 and nkmesh 3, in the same process: against a native executable
     built from a copy of the deck with those two lines changed (`--set`
     reaches only the main file, and the deck is a `--use` file).
  3. Ctrl-C (a SIGINT sent after 1 s) raises KeyboardInterrupt; the next run
     still works.
"""
import os, signal, subprocess, sys, threading, time
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(here, "..", "..", ".."))
VIV = os.environ.get("VIV", os.path.join(root, "target", "release", "viv"))
FILES = ["--use", "dft/si.viv", "--use", "dft/kmesh.viv", "--use", "dft/pw_common.viv",
         "--use", "dft/pwdft_it.viv", "--use", "dft/si_it_check.viv",
         "study/vivkernels/pwdft_m1/report.viv"]
tmp = os.environ.get("TMPDIR", "/tmp")
sys.path.insert(0, here)
import pwdft  # builds libpwdft.so on first import if needed


def native(sets, tag):
    c = os.path.join(tmp, f"pwdft_native_{tag}.c")
    exe = c[:-2]
    files = list(FILES)
    if sets:
        deck = open(os.path.join(root, "dft/si.viv")).read()
        for name, value in sets.items():
            lines = deck.split("\n")
            hits = [i for i, l in enumerate(lines) if l.startswith(name + " = ")]
            assert len(hits) == 1, name
            lines[hits[0]] = f"{name} = {value}"
            deck = "\n".join(lines)
        files[1] = os.path.join(tmp, f"si_{tag}.viv")
        open(files[1], "w").write(deck)
    subprocess.run([VIV] + files + ["-o", c], cwd=root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["cc", "-O2", "-ffp-contract=off", "-Wno-unused-function", c, "-o", exe, "-lm"], check=True)
    t = time.time()
    out = subprocess.run([exe], capture_output=True, text=True).stdout
    return out, time.time() - t


def result(text, key):
    for line in text.splitlines():
        if line.startswith("RESULT " + key + " "):
            return float(line.split()[2])


def compare(label, lib_lines, r, ref_out):
    # the interpreter appends its call-graph records; they are not program output
    ref_out = "".join(l + "\n" for l in ref_out.splitlines() if not l.startswith(("__graph__ ", "__trace__ ")))
    same_text = "\n".join(lib_lines) + "\n" == ref_out
    e = result(ref_out, "ETOT")
    print(f"{label}: ETOT lib {r['etot']!r}  ref {e!r}  bit-equal {r['etot'] == e}  "
          f"transcript byte-equal {same_text}  tests {r['tests_passed']} pass {r['tests_failed']} fail")
    return same_text and r["etot"] == e


ok = True
# 1. the deck's parameters
log = []
t = time.time(); r = pwdft.run(log=log); t_lib = time.time() - t
out, t_nat = native({}, "deck")
ok &= compare("ecut 10, nkmesh 2 vs native", log, r, out)
print(f"   seconds: library {t_lib:.2f}, native executable {t_nat:.2f}; bands {r['bands_ev'].shape}, gap {r['gap_ev']:.4f} eV")
if os.environ.get("INTERP", "1") == "1":
    t = time.time()
    interp = subprocess.run([VIV, "run"] + FILES, cwd=root, capture_output=True, text=True).stdout
    print(f"   interpreter {time.time() - t:.1f} s")
    ok &= compare("ecut 10, nkmesh 2 vs interpreter", log, r, interp)

# 2. other parameters, same process (per-call state reset)
log2 = []
t = time.time(); r2 = pwdft.run(ecut=8.0, nkmesh=3, log=log2); t_lib2 = time.time() - t
out2, t_nat2 = native({"ECUT": "8.0", "NKMESH": "3.0"}, "e8k3")
ok &= compare("ecut 8, nkmesh 3 vs native, edited deck", log2, r2, out2)
print(f"   seconds: library {t_lib2:.2f}, native {t_nat2:.2f}")

# 3. Ctrl-C
threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
t = time.time()
try:
    pwdft.run()
    print("Ctrl-C: NOT interrupted")
    ok = False
except KeyboardInterrupt:
    print(f"Ctrl-C: KeyboardInterrupt after {time.time() - t:.2f} s")
r3 = pwdft.run()
print("after Ctrl-C, the next run: ETOT bit-equal to the first", r3["etot"] == r["etot"])
ok &= r3["etot"] == r["etot"]
print("ALL OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
