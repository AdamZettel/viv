# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Builds the upstream programs the port is compared with, twice.

    python3 ports/liquids/oracle/build.py <examples checkout> <build dir>

<examples checkout> is https://github.com/Allen-Tildesley/examples at the
commit UPSTREAM.md names. Into <build dir>/exact and <build dir>/stock it
writes the programs below, built with gfortran -O2 -fdefault-real-8
-fall-intrinsics (upstream's flags, SConstruct) and -ffp-contract=off.

  stock   upstream's sources unchanged: Fortran's RANDOM_NUMBER, the
          platform's libm, upstream's printed formats. For timings and
          for the tables of GUIDE.md.
  exact   upstream's sources with three changes, none of them to the
          arithmetic (patch below):
            - RANDOM_INIT and RANDOM_NUMBER are replaced by minstd.f90's
              minstd_seed and minstd_number, the generator Viv's rand() is;
            - the block averages and the configurations are written with 17
              significant digits (es25.16e3) instead of 6 and 10 decimals;
            - Viv's libm is linked in (vmath_shim.c), and the build adds
              -ffp-contract=off -fno-fast-math, as Viv's exact C build has.
          A run of one of these and the same run of the port must agree to
          the last bit.
"""
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
# upstream's flags and -O2; the exact build adds no-contraction (no fused
# multiply-adds), which Viv's exact C build uses too
FFLAGS = ["-O2", "-fdefault-real-8", "-fall-intrinsics"]
EXACT_FLAGS = ["-ffp-contract=off", "-fno-fast-math"]

CONAVGMAT = ["config_io_module.f90", "averages_module.f90", "maths_module.f90"]
CONAVG = ["config_io_module.f90", "averages_module.f90"]
CONMAT = ["config_io_module.f90", "maths_module.f90"]
# upstream's SConstruct `builds`, for the programs ported
PROGRAMS = {
    "initialize": CONMAT + ["initialize_module.f90", "initialize.f90"],
    "md_nve_lj": CONAVG + ["md_lj_module.f90", "lrc_lj_module.f90", "md_nve_lj.f90"],
    "md_nve_lj_ll": CONAVG + ["link_list_module.f90", "md_lj_ll_module.f90", "lrc_lj_module.f90", "md_nve_lj.f90"],
    "md_nve_lj_vl": CONAVG + ["verlet_list_module.f90", "md_lj_vl_module.f90", "lrc_lj_module.f90",
                              "md_nve_lj.f90"],
    "md_nvt_lj": CONAVGMAT + ["md_lj_module.f90", "lrc_lj_module.f90", "md_nvt_lj.f90"],
    "mc_nvt_lj": CONAVGMAT + ["mc_lj_module.f90", "lrc_lj_module.f90", "mc_nvt_lj.f90"],
    "mc_npt_lj": CONAVGMAT + ["mc_lj_module.f90", "lrc_lj_module.f90", "mc_npt_lj.f90"],
    "eos_lj": ["lrc_lj_module.f90", "eos_lj_module.f90", "eos_lj.f90"],
    "pair_distribution": ["config_io_module.f90", "pair_distribution.f90"],
    "diffusion": ["config_io_module.f90", "diffusion.f90"],
    "error_calc": ["maths_module.f90", "error_calc.f90"],
}
FULL = "es25.16e3"


def patch(name, text):
    """The exact build's changes to one upstream file."""
    if "RANDOM_NUMBER" in text or "RANDOM_INIT" in text:
        text = re.sub(r"CALL RANDOM_INIT\b", "CALL minstd_seed", text)
        text = re.sub(r"CALL RANDOM_NUMBER\b", "CALL minstd_number", text)
        text = text.replace("IMPLICIT NONE", "USE minstd_module\n  IMPLICIT NONE", 1)
    if name == "averages_module.f90":
        text = text.replace("colf_fmt  = 'f15.6'", "colf_fmt  = '%s'" % FULL)
        text = text.replace("cole_fmt  = 'es15.4'", "cole_fmt  = '%s'" % FULL)
        text = text.replace("sngl_fmt  = '(a,t40,f15.6)'", "sngl_fmt  = '(a,t40,%s)'" % FULL)
    if name == "config_io_module.f90":
        text = text.replace("fmt='(f15.8)') box", "fmt='(%s)') box" % FULL)
        text = text.replace("fmt='(*(f15.10))'", "fmt='(*(%s))'" % FULL)
    if name == "pair_distribution.f90":
        text = text.replace("fmt='(2f15.8)'", "fmt='(2%s)'" % FULL)
    if name == "diffusion.f90":
        text = text.replace("fmt='(f15.6,3f15.8)'", "fmt='(4%s)'" % FULL)
    if name == "error_calc.f90":
        text = text.replace("fmt='(a,t40,f15.6)'", "fmt='(a,t40,%s)'" % FULL)
        text = text.replace("fmt='(2i15,3f15.6)'", "fmt='(2i15,3%s)'" % FULL)
    if name == "eos_lj.f90":
        text = text.replace("t40,f15.6)", "t40,%s)" % FULL)
    return text


def build(src, out, exact):
    os.makedirs(out, exist_ok=True)
    work = os.path.join(out, "src")
    os.makedirs(work, exist_ok=True)
    extra = []
    if exact:
        shutil.copy(os.path.join(HERE, "minstd.f90"), work)
        subprocess.run(["cc", "-O2", "-ffp-contract=off", "-c", "-I", os.path.join(ROOT, "lang", "viv", "src"),
                        os.path.join(HERE, "vmath_shim.c"), "-o", os.path.join(work, "vmath_shim.o")], check=True)
        extra = [os.path.join(work, "vmath_shim.o")]
    for prog, files in PROGRAMS.items():
        paths = []
        if exact and any("RANDOM" in open(os.path.join(src, f)).read() for f in files):
            paths.append(os.path.join(work, "minstd.f90"))
        for f in files:
            text = open(os.path.join(src, f)).read()
            if exact:
                text = patch(f, text)
            with open(os.path.join(work, f), "w") as fh:
                fh.write(text)
            paths.append(os.path.join(work, f))
        mod = os.path.join(work, "mod_" + prog)
        os.makedirs(mod, exist_ok=True)
        flags = FFLAGS + (EXACT_FLAGS if exact else [])
        subprocess.run(["gfortran"] + flags + ["-J", mod] + paths + extra + ["-o", os.path.join(out, prog)],
                       check=True)
    print("built", ", ".join(PROGRAMS), "in", out)


def main():
    src, out = sys.argv[1], sys.argv[2]
    build(src, os.path.join(out, "exact"), True)
    build(src, os.path.join(out, "stock"), False)


if __name__ == "__main__":
    main()
