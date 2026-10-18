#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# Name and rule lint for the Hartree-Fock library. The interpreter and the C backend accept different programs, and
# several of the differences only show up once modules are combined, so this
# checks the combined set before either engine sees it.
#
#     tools/hf_names.sh
#
# It scans lib/hf (with lib/hf/tests), lib/hfref, lib/basis, lib/linalg, lang/viv/std/contracts.viv and the
# legacy lib/gto, lib/gto_md, lib/qcref and lib/qcfuzz, and fails on:
#   - a top-level def or global defined twice anywhere in that set (the
#     interpreter silently keeps the later one, while C refuses);
#   - a test local or top-level assignment that reuses a def or global name
#     from another module (top level and tests share one type environment);
#   - a lib/hf module that imports gto, gto_md, qcref or qcfuzz;
#   - erf( or gamma( in lib/hf code;
#   - a test header with anything after the `{` (the test would not be
#     stripped when another program uses the module);
#   - a lib/hf or lib/hfref top-level statement that is not a def, a record
#     declaration, a test or a scalar constant;
#   - a lib/hf def, global or test local without its module prefix.
# On success it prints `hf_names: N names, 0 collisions` and exits 0.
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
exec python3 - "$root" <<'PY'
import os, re, sys

root = sys.argv[1]

def rel(p):
    return os.path.relpath(p, root)

def listdir_viv(d):
    full = os.path.join(root, d)
    if not os.path.isdir(full):
        return []
    return sorted(os.path.join(full, f) for f in os.listdir(full) if f.endswith(".viv"))

hf_files = listdir_viv("lib/hf") + listdir_viv("lib/hf/tests")
basis_files = listdir_viv("lib/basis")
ref_files = listdir_viv("lib/hfref")
other = ["lib/linalg.viv", "lang/viv/std/contracts.viv", "lib/gto.viv",
         "lib/gto_md.viv", "lib/qcref.viv", "lib/qcfuzz.viv"]
other_files = [os.path.join(root, f) for f in other if os.path.exists(os.path.join(root, f))]
files = hf_files + ref_files + basis_files + other_files

PREFIXES = ["bas_", "opt_", "sw_", "es_", "boys_", "herm_", "md1_", "mderi_", "mdjk_",
            "os_", "hgp_", "gh_", "rys_", "c2s_", "tr2_", "tr4_", "eig_", "orth_", "den_",
            "gs_", "jk_", "accel_", "rhf_", "uhf_", "rohf_", "stab_", "int_", "hf_", "hfz_", "iv_",
            "scf_", "mol_", "casida_", "ao_", "shell_", "mdg_"]
CONST_PREFIXES = [p.upper() for p in PREFIXES] + ["AT_", "SH_", "PR_", "AO_", "OPT_"]
# the sweep drivers, the molecule constructors, the three entry points of
# lib/hf/scf (rhf, uhf, rohf: their own prefixes' names), and the two calls
# behind them, integrals(mol, o) and scf(setup)
DEF_EXTRA = ["sweep", "molecule", "rhf", "uhf", "rohf", "integrals", "scf"]
BANNED_USES = {"lib/gto", "lib/gto_md", "lib/qcref", "lib/qcfuzz"}

IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
re_def = re.compile(r"^def\s+(" + IDENT + r")\s*\(")
re_assign = re.compile(r"(?:^|[\s;{])(" + IDENT + r")\s*=(?!=)")
re_for = re.compile(r"\bfor\s+(" + IDENT + r")\s+(?:to|in)\b")
re_test_exact = re.compile(r'^test "[^"]*" \{$')
re_banned_call = re.compile(r"\b(erf|erfc|gamma|tgamma)\s*\(")
re_const_rhs = re.compile(r"^[-+*/().0-9eE\s]*$")

def strip_comment(line):
    """Code part of a line: everything before a # that is not inside a string."""
    quoted = False
    prev_bs = False
    for i, ch in enumerate(line):
        if quoted:
            if ch == '"' and not prev_bs:
                quoted = False
            prev_bs = ch == "\\" and not prev_bs
            continue
        if ch == '"':
            quoted = True
        elif ch == "#":
            return line[:i]
        prev_bs = False
    return line

def blank_strings(code):
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', code)

def brace_delta(code):
    code = blank_strings(code)
    return code.count("{") - code.count("}")

problems = []
top = {}          # name -> list of (file, line, kind)
locals_seen = []  # (name, file, line, where) for test-local and top-level assignments

def note(f, ln, msg):
    problems.append("%s:%d: %s" % (rel(f), ln, msg))

def has_prefix(name, prefixes):
    return any(name.startswith(p) for p in prefixes)

for f in files:
    is_hf = f in hf_files
    is_ref = f in ref_files
    lines = open(f, encoding="utf-8").read().split("\n")
    depth = 0
    in_fence = False
    in_test = False
    consts = set()
    in_import = False
    for n, raw in enumerate(lines, 1):
        if raw.strip() == "###":
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if raw.startswith("#!"):
            continue
        # an import line (and the rest of a parenthesized list) is not code
        if in_import:
            in_import = ")" not in raw
            continue
        m = re.match(r"^(?:from\s+([\w.]+)\s+import\b|import\s+([\w.]+))", raw)
        if m:
            in_import = "(" in raw and ")" not in raw
            # an import of a banned module: `import gto`, `from qcref import x`
            mod = "lib/" + (m.group(1) or m.group(2)).replace(".", "/")
            if is_hf and mod in BANNED_USES:
                note(f, n, "lib/hf may not use %s" % mod)
            continue
        code = strip_comment(raw)
        stripped = code.strip()
        if raw.lstrip().startswith("test ") and depth == 0:
            if not re_test_exact.match(raw.strip()) or raw != raw.rstrip():
                note(f, n, "test header must be exactly `test \"...\" {`: %s" % raw.strip())
        if is_hf and re_banned_call.search(blank_strings(code)):
            note(f, n, "erf/gamma are banned in lib/hf: %s" % stripped)
        if depth == 0 and stripped:
            m = re_def.match(stripped)
            if m:
                name = m.group(1)
                top.setdefault(name, []).append((f, n, "def"))
                if is_hf and not has_prefix(name, PREFIXES + DEF_EXTRA):
                    note(f, n, "def `%s` has no module prefix" % name)
            elif stripped.startswith("test "):
                in_test = True
            elif re.match(r"^record\s+" + IDENT, stripped):
                # a record declaration (Molecule, ScfOptions, ScfResult, ...):
                # its name is a type, collision-checked like a def
                name = re.match(r"^record\s+(" + IDENT + r")", stripped).group(1)
                top.setdefault(name, []).append((f, n, "record"))
            else:
                m = re.match(r"^(" + IDENT + r")\s*=(?!=)\s*(.*)$", stripped)
                if m and not stripped.startswith("for ") and ";" not in stripped:
                    name, rhs = m.group(1), m.group(2)
                    if is_hf or is_ref:
                        rhs_names = re.findall(IDENT, re.sub(r"\d+\.?\d*[eE][-+]?\d+", "0", rhs))
                        const = re_const_rhs.match(re.sub(IDENT, "0", rhs)) is not None \
                            and all(x in consts for x in rhs_names)
                        if not const:
                            note(f, n, "library top level holds scalar constants only: %s" % stripped)
                        consts.add(name)
                        if is_hf and not has_prefix(name, CONST_PREFIXES):
                            note(f, n, "global `%s` has no module prefix" % name)
                        top.setdefault(name, []).append((f, n, "global"))
                    else:
                        # legacy top-level code: a first assignment is a global
                        if not any(e[0] == f for e in top.get(name, [])):
                            top.setdefault(name, []).append((f, n, "global"))
                elif is_hf or is_ref:
                    note(f, n, "library top level holds defs, tests and constants only: %s" % stripped)
                else:
                    for m in re_for.finditer(stripped):
                        locals_seen.append((m.group(1), f, n, "top-level loop"))
        elif in_test and stripped:
            names = [m.group(1) for m in re_assign.finditer(blank_strings(code))]
            names += [m.group(1) for m in re_for.finditer(code)]
            for name in names:
                locals_seen.append((name, f, n, "test local"))
                if is_hf and not has_prefix(name, PREFIXES):
                    note(f, n, "test local `%s` has no module prefix" % name)
        depth += brace_delta(code)
        if depth <= 0:
            depth = 0
            in_test = False

collisions = 0
for name, where in sorted(top.items()):
    if len(where) > 1:
        collisions += 1
        places = ", ".join("%s:%d (%s)" % (rel(f), n, k) for f, n, k in where)
        problems.append("collision: `%s` is defined more than once: %s" % (name, places))

reported = set()
for name, f, n, what in locals_seen:
    for tf, tn, kind in top.get(name, []):
        if tf != f and (name, f) not in reported:
            reported.add((name, f))
            collisions += 1
            problems.append("collision: %s `%s` at %s:%d reuses the %s from %s:%d"
                            % (what, name, rel(f), n, kind, rel(tf), tn))

for p in problems:
    print(p)
others = len(problems) - collisions
tail = "" if others == 0 else ", %d other problems" % others
print("hf_names: %d names, %d collisions%s" % (len(top), collisions, tail))
sys.exit(1 if problems else 0)
PY
