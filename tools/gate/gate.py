#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Viv's gate: one fixed list of checks per tier, run the same way every time.

    tools/gate.sh quick|premerge|postmerge|full|nightly [--list] [--json] [--keep-going]
                  [--only a,b] [--ignore-lock] [--no-cache] [--sample F]
                  [--base REV] [--no-triggers] [--serial]
    tools/gate.sh cache [stats|prune [--days N]|clear]   the cache's store
    tools/gate.sh status [N]            the last N gate runs (target/gate/history.jsonl)
    tools/gate.sh bugs [--check [--run]] the bug ledger's trend; --check: every fixed
                                        bug names a test that exists (--run: and passes)
    tools/gate.sh ratchet [--update]    `viv features --usage` against its ratchet
    tools/gate.sh golden <base|nightly> hub-golden snapshot of this tree, compared
                                        with main's (base) or the last nightly's
    tools/gate.sh triggers [--base REV] the post-merge suites this branch's changes add
                                        to its pre-merge run

The tiers:

    quick      the language checks, before every commit you report
    premerge   quick, the library engine-agreement set, the wasm freshness
               checks and one small program per application family (the smoke
               set); plus the post-merge suites whose paths the branch touches
               (TRIGGERS below). tools/merge.sh runs it on the merged tree.
    postmerge  the application suites, py2viv, memcheck-quick, www-check,
               hub-golden, fast-identity: run on main after merges land
    full       premerge + postmerge, nothing left out
    nightly    full with the cache off, plus polybench, NPB, the timing
               checks, memcheck on every program, fuzz, port-verify

Each check has a name, a shell command run from the
repository root, and a pass rule (exit status 0, unless the check says
otherwise). The report is one line per check, in the order they finish:

    PASS  cargo-test              312.4s  target/gate/premerge/cargo-test.log

then a summary, and the exit status is nonzero when any check failed or was
blocked. A failed check prints the tail of its log and the command that
reruns it.

Lanes (tools/gate/lanes.py): independent checks run at once. Each check has
a weight (the cores it keeps busy, its peak memory, whether it builds under
a sanitizer); the running checks stay within GATE_CORES (default the core
count less two) and GATE_MEM_GB (default 10), two sanitizer-heavy checks
never run together, and a timing check runs alone. The ready check with the
longest chain of work behind it starts first. quick, premerge and full stop
at the first failure (the running checks are stopped, NOT RUN) unless
--keep-going; postmerge and nightly always run everything. --serial runs one
check at a time.

Timing lock: study/ports/harness/port.py holds study/ports/.timing.lock while it
times a port, in any worktree. While a live lock exists, a timing check is
refused (BLOCKED), and so is every heavy check (one that would load the
machine during somebody's timing run) unless --ignore-lock is given. BLOCKED
fails the gate: a gate that did not run its checks has not passed.

The cache (tools/gate/cache.py has the details): quick and full reuse a
passing result whose inputs have not changed since it passed, in any
worktree, and say so on its line:

    PASS  cargo-test                0.6s  target/gate/full/cargo-test.log  PASS (cached 3f9a0c21d4, from e09b6b0de 2026-10-06)
    PASS  3e-lib-hf                41.2s  target/gate/full/3e-lib-hf.log  [cache: 40 of 43 reused, 3 run]

A check says how in its `cache=`: per item ("items": a program of the
three-engines directories, scf, fast-identity, memcheck-quick, hub-golden;
a test binary of cargo-test and fast-check (cargo_tests.py); a script of
www-check (www_check.py); a bug's test), on the files under some paths plus
what viv traces reading (Trace), or on the whole working tree ("tree":
reused only for an identical tree). Rust tests and site scripts are traced
by tools/gate/fstrace.c (DYLD_INSERT_LIBRARIES): every file they and their
children open, stat, list or write. A failure is never stored, a timing check never cached.
--sample F (default 5%) of the reused results rerun; one that does not
reproduce fails the gate with CACHE MISMATCH and is entered in
bugs/ledger.jsonl. --no-cache runs everything; nightly always does.

Load (tools/gate/load.py): the checks' time limits are CPU time where a run
is one process (memcheck, hub-golden, the post budget), or wall time scaled
by max(1, load1/ncpu), with an absolute cap of 10 times the limit for a run
that hangs without working. A check that exits 75, or times out while
load1 > ncpu, is BLOCKED-LOAD, not FAIL: the rest of the tier runs, then the
gate waits (at most 30 minutes) for the load to fall and runs it again with
the checks that needed it. Still blocked, the gate ends BLOCKED-LOAD
(nonzero). tools/gate/test_load.py and tools/memcheck/test_limits.sh
demonstrate it.

Logs go to <tree>/target/gate/<tier>/. The run history and the hub-golden
snapshots are shared by all worktrees, in the main repository's
target/gate/.
"""
import datetime as dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import random
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd or ROOT, capture_output=True, text=True).stdout.strip()


ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=HERE,
                      capture_output=True, text=True).stdout.strip()
COMMON = os.path.dirname(os.path.abspath(os.path.join(ROOT, git("rev-parse", "--git-common-dir"))))
SHARED = os.path.join(COMMON, "target", "gate")
HISTORY = os.path.join(SHARED, "history.jsonl")
GOLDEN = os.path.join(SHARED, "golden")
LEDGER = os.path.join(ROOT, "bugs", "ledger.jsonl")
RATCHET = os.path.join(HERE, "features_ratchet.json")
TIERS = ["quick", "premerge", "postmerge", "full", "nightly"]
# a check's level -> the tiers it runs in (full = premerge + postmerge)
IN_TIERS = {"quick": {"quick", "premerge", "full", "nightly"},
            "premerge": {"premerge", "full", "nightly"},
            "postmerge": {"postmerge", "full", "nightly"},
            "nightly": {"nightly"}}
FAIL_FAST = {"quick", "premerge", "full"}   # postmerge and nightly report every failure
VIV = os.environ.get("GATE_VIV") or os.path.join(ROOT, "target", "release", "viv")
GATE = "tools/gate.sh"
SAMPLE = 0.05   # the share of cache hits rerun on every gate that uses the cache
PRUNE_DAYS = 14

os.environ.setdefault("GATE_ROOT", ROOT)
os.environ.setdefault("GATE_COMMON", COMMON)
sys.path.insert(0, HERE)
import cache as gcache  # noqa: E402  (tools/gate/cache.py)
import lanes as glanes  # noqa: E402  (tools/gate/lanes.py)
import load as gload  # noqa: E402  (tools/gate/load.py)


# ---------------------------------------------------------------- the checks

class Check:
    """`tier`: the check's level, quick, premerge, postmerge or nightly
    (IN_TIERS: the tiers each level runs in); `only`, when given, is the
    exact set of tiers instead.

    `cache`: how a passing result is reused (tools/gate/cache.py). None:
    never (builds, timing, the cheap checks). "tree": the whole working tree
    is the input, so it is reused only for an identical tree. Trace(paths):
    the files under `paths` plus every file viv reports reading during the
    run. "items": the check caches each program itself (its command goes
    through cache.py); the gate only reports the counts.

    The weight, for the lanes (tools/gate/lanes.py): `cores` kept busy,
    peak memory `mem` in GB, `san` for a sanitizer-heavy check, and `est`,
    the seconds it takes uncached, which orders
    the ready checks."""
    def __init__(self, name, cmd, tier="quick", only=None, needs=("build",), heavy=None,
                 timing=False, provisional=False, timeout=3600, what="", cache=None,
                 cores=1, mem=0.5, san=False, est=30):
        self.name, self.cmd, self.tier = name, cmd, tier
        self.cache = None if timing else cache   # a timing check is never cached
        self.only = only                  # the tiers it runs in, when not IN_TIERS[tier]
        self.needs = list(needs)
        self.heavy = (tier != "quick") if heavy is None else heavy
        self.timing, self.provisional = timing, provisional
        self.timeout, self.what = timeout, what
        self.cores, self.mem, self.san, self.est = cores, mem, san, est

    def runs_in(self, tier):
        if self.only is not None:
            return tier in self.only
        return tier in IN_TIERS[self.tier]


class Trace:
    """A check's inputs: the files under `paths`, and what viv traces (VIV_TRACE_INPUTS).
    io_ok: the check runs every program in the interpreter, which traces what
    read_text reads, so a program that reads files may be cached."""
    def __init__(self, *paths, io_ok=False):
        self.paths, self.io_ok = list(paths), io_ok


# the files tools/three-engines.sh reads besides the programs
TE = ["tools/three-engines.sh", "tools/run-wasm.mjs", "www/src/lib/vivDirect.js", "www/src/lib/uses.js"]


def three(d, tier, only=None, timeout=1800, est=30):
    name = "3e-" + d.replace("lang/viv/", "").replace("/", "-")
    return Check(name, f"python3 tools/gate/cache.py three-engines {d}", tier, only=only, needs=["build", "wasm"],
                 timeout=timeout, cache="items", cores=2, mem=1, est=est,
                 what=f"every program in {d}/ in three engines, identical output (tools/three-engines.sh, one program at a time)")


# three-engines, by level (seconds uncached). Not here:
# lang/viv/features/examples, whose 133 catalog examples tests/features.rs runs
# in three engines inside cargo-test.
QUICK_3E = [("lang/viv/tests/phrases", 10), ("lang/viv/tests/ints", 10), ("lib", 40),
            ("lib/cas/check", 20), ("lib/algo/tests", 20)]
# the library engine-agreement set
PRE_3E = [("lib/cas", 30), ("lib/algo", 20), ("lib/algo/legacy", 15), ("lib/algo/legacy/tests", 15),
          ("lib/gpt", 20), ("lib/randla", 30), ("lib/ode", 40), ("lib/stats", 30), ("engines", 42), ("pde", 12),
          ("lib/rv32", 5), ("lib/rv32/tests", 10), ("lib/cc", 15), ("lib/cc/tests", 40),
          ("lib/gguf", 5), ("lib/gguf/tests", 10)]
POST_3E = [("lib/hf", 186), ("lib/hf/tests", 414), ("scf", 1020), ("posts", 114),
           ("lib/rv32/tests/riscv", 120)]
# NPB suite 5 at class S: each port's test is NPB's own verification
NIGHTLY_3E = [("study/benchmarks/viv/nas", 300)]

# www/ `npm run check` parts that read only committed files (no wasm, no stage).
# claims:status is not one: it compares src/data/claims.json, which the staging
# regenerates, with the posts, so it runs in www-check after `npm run stage`.
WWW_STATIC = "npm run figuretest && npm run aispeak -- --strict && npm run claimstest"

NODE_MODULES = ("[ -e www/node_modules ] || { m=\"$GATE_COMMON/www/node_modules\"; "
                "[ -d \"$m\" ] || { echo \"no www/node_modules here or in $GATE_COMMON\"; exit 1; }; "
                "ln -s \"$m\" www/node_modules && trap 'rm -f \"$GATE_ROOT/www/node_modules\"' EXIT; }; ")

# cargo-test's application-sized test binaries: post-merge
CARGO_APP = ["viv:test:water_scf", "viv:test:molecule_db", "viv:test:md_polymer", "viv:test:geomopt",
             "viv:test:fem_poisson"]
# the smallest export manifest, for the smoke set
EXPORT_SMOKE = "lib/skyrme/skyrmehf.export.toml"

# The benchmark harness (study/benchmarks/harness) is in the working
# repository only; a checkout without it skips the checks that run it.
BENCH = "study/benchmarks/harness"


def with_bench(cmd):
    return f'if [ -d {BENCH} ]; then {cmd}; else echo "skipped: no {BENCH} in this checkout"; fi'


CHECKS = [
    # -- quick (and every tier above it)
    Check("build", "cargo build --release -p viv", needs=[], timeout=1800, cores=4, mem=3, est=40,
          what="the release compiler"),
    Check("wasm", "sh www/build-wasm.sh", needs=[], timeout=1800, cores=4, mem=2, est=40,
          what="viv.wasm with the runtime exports (three-engines runs engine 3 on it)"),
    Check("cargo-test", f"python3 tools/gate/cargo_tests.py --skip {','.join(CARGO_APP)}", needs=[],
          timeout=2700, cache="items", cores=6, mem=5, san=True, est=600,
          what="every Rust test in the workspace but the application-sized ones (cargo-test-app), "
               "two test binaries at a time (tools/gate/cargo_tests.py), and the doc tests"),
    *[three(d, "quick", est=e) for d, e in QUICK_3E],
    Check("lens-coverage", f"{VIV} lens coverage --check > /dev/null", cache="tree", est=60,
          what="intent coverage holds its ratchet (lens_coverage.json)"),
    Check("arch-check", f"{VIV} arch check --check > /dev/null",
          what="the architecture lens holds its ratchet (arch_ratchet.json)"),
    Check("features-ratchet", f"python3 tools/gate/gate.py ratchet",
          what="no `viv features --usage` pattern count rises above tools/gate/features_ratchet.json"),
    Check("bugs", "python3 tools/gate/gate.py bugs --check", needs=["build", "wasm"], cache="items", est=20,
          what="every fixed bug in bugs/ledger.jsonl names a test that exists; its bugs/tests programs pass "
               "in three engines (its cargo tests run in cargo-test)"),
    Check("www-static", NODE_MODULES + f"cd www && {WWW_STATIC}", needs=[],
          cores=2, mem=1, est=60, what="the parts of `npm run check` that read only committed files"),
    Check("py2viv-rules", "python3 tools/py2viv/check_rules.py --static", needs=[],
          what="py2viv's rule tables and ledger, static (postmerge runs the differential tests)"),
    Check("algo-spec", "python3 tools/algo/check.py", needs=["build"],
          what="lib/algo's specification (no runs)"),
    Check("licenses", "python3 tools/licenses/check.py", "premerge", needs=[], cores=1, mem=1, est=5,
          what="REUSE licensing: every tracked file has a license and a copyright, each license's text is in "
               "LICENSES/, hand-written sources carry an SPDX header, nothing is GPL (LICENSING.md)"),
    Check("gate-tests", "python3 tools/gate/test_lanes.py && python3 tools/gate/test_load.py", needs=[], est=15,
          what="the gate's own tests: lanes, fail-fast, tiers and path triggers, the load handling"),
    Check("fast-check", "python3 tools/gate/cargo_tests.py --only viv:test:fast_mode", needs=["build"],
          cache="items", cores=2, est=10,
          what="viv check --fast holds a program using every freedom within tolerance; FMA is on and off as asked"),

    # -- premerge: the library engine-agreement set, wasm freshness, the smoke set
    *[three(d, "premerge", est=e) for d, e in PRE_3E],
    Check("dft-wasm", "dft/build-wasm.sh --check", "premerge", needs=["wasm"],
          what="the committed DFT wasm modules are built from these sources"),
    Check("engines-wasm", "node engines/build-wasm.mjs --check", "premerge", needs=["wasm"],
          what="the committed engine wasm modules are built from these sources"),
    Check("xc-smoke", f"VIV={VIV} python3 xc/tests/oracle.py --smoke --check", "premerge", cores=1, mem=1, est=10,
          what="viv-xc: the smoke set of LDA/GGA functionals (xc/tests/oracle.py SMOKE) against libxc 6.1.0, "
               "the copy PySCF ships; zk, vrho, vsigma, unpolarized and polarized"),
    Check("xc-oracle", f"VIV={VIV} python3 xc/tests/oracle.py --check", "postmerge", cores=1, mem=1, est=240,
          what="viv-xc: every translated functional against libxc 6.1.0; fails when a status in "
               "xc/tests/expected.json changes"),
    Check("cc-corpus", "python3 tools/cc/corpus.py --check", "premerge", needs=[], cores=1, mem=1, est=10,
          what="lib/cc/tests/corpus.viv holds what clang -O0 on the host prints and returns for each tests/cc program"),
    Check("gguf", "tools/gguf/check.sh", "premerge", needs=["build", "wasm"], cores=1, mem=1, est=150,
          what="lib/gguf against gguf-py 0.19.0 on the fixture and the tinyllamas models when present "
               "(each skipped when absent), in three engines"),
    Check("minpack", "lib/minpack/check.sh", "premerge", est=12,
          what="the MINPACK port against the Fortran (the minpack family's smoke test: 12 seconds)"),
    Check("smoke-hf", "python3 tools/gate/cache.py three-engines lib/hf/tests hf", "premerge",
          needs=["build", "wasm"], cache="items", cores=2, mem=1, est=60,
          what="smoke: lib/hf/tests/hf.viv (RHF on H2 in STO-3G among others) in three engines: "
               "everything that loads lib/hf"),
    Check("smoke-dft", "dft/check.sh si", "premerge", cores=1, mem=1, est=70, cache=Trace("dft"),
          what="smoke: dft/ lint and lenses, silicon in both DFT engines"),
    Check("smoke-post", "python3 tools/gate/cache.py three-engines posts 2026-09-22-compound-interest", "premerge",
          needs=["build", "wasm"], cache="items", est=10, what="smoke: one post in three engines"),
    Check("smoke-export", f"EXPORT_CI_STRICT=1 EXPORT_CI_OUT=target/gate/export-smoke tools/export_ci.sh {EXPORT_SMOKE}",
          "premerge", cores=1, mem=1, est=30,
          what=f"smoke: `viv export` of {EXPORT_SMOKE}, built and tested through Python (tools/export_ci.sh)"),
    Check("vipyr", f"VIV={VIV} python3 -m pytest -q -p no:cacheprovider python/tests --ignore=python/tests/test_cuda_sim.py",
          "premerge", cores=1, mem=1, est=15,
          what="the Python package vipyr: load, the cache, inference, the import hook, stubs, "
               "a library wheel and vipyr's platform wheel in fresh venvs (python/tests; test_cuda_sim.py is cuda-sim's)"),
    Check("vivsmith-smoke", "python3 tools/fuzz/test_fuzz.py && python3 tools/fuzz/run.py --seed 1 --count 100 --jobs 2 --known tools/fuzz/known.json "
          "--out target/gate/vivsmith-smoke", "premerge", needs=["build", "wasm"], cores=2, mem=1, est=45,
          what="vivsmith: 100 random programs (fixed seeds) in the interpreter, native C and wasm, the same bytes "
               "(tools/fuzz/run.py; seeds failing on an open bug are listed in tools/fuzz/known.json)"),
    Check("claim-programs", "python3 tools/claims/run.py --check", "premerge", needs=["build"], cores=1, mem=1, est=40,
          what="the fast claims (claims/): each Viv program and its vipyr twin pass, agree on every value both "
               "print, and establish every number their statement gives (tools/claims/run.py)"),
    Check("cuda-sim", 'py="$HOME/.venvs/power/bin/python"; if [ -x "$py" ] && "$py" -c "import numba, pytest" 2> /dev/null; then '
          f'VIV={VIV} PYTHONPATH=python NUMBA_ENABLE_CUDASIM=1 nice "$py" -m pytest -q -p no:cacheprovider python/tests/test_cuda_sim.py; '
          'else echo "skipped: no Python with numba and pytest (numba\'s CUDA simulator)"; fi',
          "premerge", needs=["build"], cores=1, mem=1, est=60,
          what="examples/cuda's kernels against the same kernels in numba, run by numba's CUDA simulator, bit for bit "
               "(python/tests/test_cuda_sim.py, by a Python with numba; skipped when there is none)"),
    Check("vivdb", "python3 tools/vivdb/test_vivdb.py", "premerge", needs=["build"], cores=1, mem=1, est=5,
          what="the program database: build on tools/vivdb/fixture, byte spans, "
               "an idempotent rebuild, the impact, dead-code and tests-covering queries"),

    # -- postmerge: the application suites
    *[three(d, "postmerge", est=e) for d, e in POST_3E],
    Check("cargo-test-app", f"python3 tools/gate/cargo_tests.py --only {','.join(CARGO_APP)}", "postmerge",
          needs=[], cache="items", cores=4, mem=3, est=120,
          what="cargo-test's application-sized test binaries (water SCF, molecule db, MD, geomopt, FEM)"),
    Check("3e-ports", "python3 tools/gate/gate.py ports-parity", "postmerge", needs=["build", "wasm"], cache="items",
          cores=2, mem=1, est=30,
          what="each port's parity programs in three engines (not the production-size or _t timing programs)"),
    Check("compare", "study/compare/check.sh", "postmerge", needs=["build", "wasm"], cores=2, mem=1, est=30,
          cache=Trace("study/compare", *TE, io_ok=True),
          what="every Python/Viv pair of study/compare, and three-engines on each topic"),
    Check("scf", "scf/check.sh lint", "postmerge", est=30,
          what="lib/hf's name, size and architecture lints, the generated kernels, its lenses and coverage "
               "(its programs run in 3e-lib-hf, 3e-lib-hf-tests and 3e-scf)"),
    # after smoke-dft when both run: dft/run.sh builds into a fixed path per material
    Check("dft", "dft/check.sh", "postmerge", needs=["build", "smoke-dft"], cache=Trace("dft"), cores=1, mem=1,
          est=120, what="plane-wave DFT reproduces its references"),
    Check("fast-identity", "python3 tools/gate/cache.py fast-identity lib && python3 tools/gate/cache.py fast-identity lang/viv/features/examples",
          "postmerge", needs=["build"], cache="items", cores=2, mem=1, est=220,
          what="the fast build with its three freedoms taken away prints the exact bytes"),
    Check("memcheck-quick", "JOBS=2 tools/memcheck/ci.sh --quick target/gate/memcheck-quick", "postmerge",
          only={"postmerge", "full"}, cache="items", cores=4, mem=6, san=True, est=2180,
          what="ASan/UBSan on one program in four, short fuzz"),
    Check("algo-run", "python3 tools/algo/check.py --run", "postmerge", needs=["build", "wasm"], cores=2, mem=1,
          est=100, cache=Trace("tools/algo", "lib/algo", *TE, io_ok=True),
          what="lib/algo's specification, with the programs run"),
    Check("py2viv", "python3 tools/py2viv/check_rules.py && python3 -m pytest -q tools/py2viv/tests",
          "postmerge", needs=["build"], heavy=False, cache=Trace("tools/py2viv", io_ok=True), cores=2, mem=2,
          est=1180, what="py2viv's rules and its tests"),
    # after www-static when both run: each links and unlinks www/node_modules
    Check("www-check", "python3 tools/gate/www_check.py", "postmerge",
          needs=["build", "wasm", "www-static"], timeout=3600, cache="items", cores=4, mem=3, est=680,
          what="the site's staging and every `npm run check` test, one script at a time (tools/gate/www_check.py; py2viv:rules and algo:check are the py2viv and algo-spec checks)"),
    Check("export", "EXPORT_CI_STRICT=1 tools/export_ci.sh", "postmerge", needs=["build", "smoke-export"],
          cores=1, mem=2, est=430,
          what="every *.export.toml exported, built and tested through Python (tools/export_ci.sh)"),
    Check("hub-golden", "python3 tools/gate/gate.py golden base", "postmerge", only={"postmerge", "full"},
          needs=["build", "wasm"], timeout=5400, cache="items", cores=4, mem=3, est=600,
          what="hub-golden snapshot against main's; a run output that changed fails"),

    # -- nightly (with everything above, the cache off)
    *[three(d, "nightly", est=e) for d, e in NIGHTLY_3E],
    Check("vivsmith", "python3 tools/fuzz/run.py --seconds 1500 --jobs 4 --san --known tools/fuzz/known.json "
          "--open-kinds --reduce --out target/gate/vivsmith", "nightly", needs=["build", "wasm"], timeout=3600, cores=4, mem=2,
          san=True, est=1700,
          what="vivsmith for 25 minutes on new seeds: three engines and the C under ASan/UBSan, every failure "
               "kept and reduced in target/gate/vivsmith (tools/fuzz/run.py)"),
    Check("claim-programs-heavy", "python3 tools/claims/run.py --heavy --check", "nightly", needs=["build"],
          timeout=3600, cores=1, mem=2, est=300,
          what="the heavy claims (cost: heavy, such as the dense-LU timing on case2383wp), as the fast ones"),
    Check("polybench",
          with_bench("cd study/benchmarks/harness && sh check_upstream.sh && "
                     "python3 poly.py verify --sizes MINI --json \"$GATE_ROOT/target/gate/poly_verify.json\""),
          "nightly", needs=["build", "wasm"], timeout=3600, cores=2, mem=2, est=600,
          what="suite 4: all 30 PolyBench kernels at MINI, C vs NumPy vs Viv in every engine, "
               "the exact engines byte-identical, viv check --fast"),
    Check("memcheck", "JOBS=2 tools/memcheck/ci.sh target/gate/memcheck", "nightly", timeout=10800,
          cores=4, mem=6, san=True, est=7200,
          what="ASan/UBSan on every program, memory flat in loop length, races, fuzz"),
    Check("hub-golden-nightly", "python3 tools/gate/gate.py golden nightly", "nightly",
          only={"nightly"}, needs=["build", "wasm"], timeout=7200, cores=4, mem=3, est=900,
          what="hub-golden snapshot against the previous nightly's"),
    Check("fuzz", "FUZZ_SECONDS=${FUZZ_SECONDS:-600} sh tools/memcheck/fuzz/run.sh target/gate/fuzz",
          "nightly", timeout=3600, cores=2, mem=2, san=True, est=650,
          what="the C runtime fuzzed against the interpreter, new seeds"),
    Check("port-verify", "python3 tools/gate/gate.py ports-verify", "nightly", timeout=10800, cores=4, mem=3,
          est=3600, what="port.py verify for every port (study/ports/*/port.toml)"),
    Check("bench-smoke",
          # bench.py writes into results/: keep a copy and put it back, so the tree stays as it was
          with_bench("r=\"$GATE_ROOT/study/benchmarks/results\"; k=$(mktemp -d); cp -Rp \"$r\" $k/; trap 'rm -rf \"$r\"; mv $k/results \"$r\"; rmdir $k' EXIT; "
                     "cd study/benchmarks/harness && python3 bench.py time micro --quick --names fib,pi_sum --max-wait 0"),
          "nightly", timing=True, provisional=True, timeout=1800,
          what="two micro benchmarks (provisional: a failure warns, it does not fail the gate)"),
    Check("nas-timing",
          # nas.py writes into results/ and build/: keep results as it was
          with_bench("r=\"$GATE_ROOT/study/benchmarks/results\"; k=$(mktemp -d); cp -Rp \"$r\" $k/; trap 'rm -rf \"$r\"; mv $k/results \"$r\"; rmdir $k' EXIT; "
                     "cd study/benchmarks/harness && python3 nas.py build --classes S,W && python3 nas.py verify --classes S,W "
                     "&& python3 nas.py time --classes S,W --engines native,release,fast,ref-O2 --tag nightly --max-wait 0"),
          "nightly", timing=True, provisional=True, timeout=3600,
          what="NPB suite 5 at classes S and W: built, verified, timed (provisional: a failure warns)"),
    Check("polybench-smoke",
          with_bench("cd study/benchmarks/harness && o=\"$GATE_ROOT/target/gate/poly\"; mkdir -p \"$o\" && "
                     "python3 poly.py verify --kernels gemm,seidel-2d --sizes MINI,SMALL --json \"$o/verify.json\" && "
                     "python3 poly.py time --kernels gemm,seidel-2d --sizes SMALL --runs 3 "
                     "--columns c_O3,numpy,native,release,fast,wasm --verify-json \"$o/verify.json\" --json \"$o/time.json\""),
          "nightly", needs=["build", "wasm"], timing=True, provisional=True, timeout=1800,
          what="suite 4 timing smoke: two PolyBench kernels at SMALL (provisional: warns, does not fail)"),
]
POSTMERGE = [c.name for c in CHECKS if c.runs_in("postmerge") and not c.runs_in("premerge")]

# Path triggers: a branch whose changes against main
# touch one of these paths runs these post-merge suites in its pre-merge tier.
# A prefix ending in "/" is a directory; "*.x" matches a suffix; anything else
# is a prefix of the path. The one table: `tools/gate.sh triggers` prints it.
C_BACKEND = ["lang/viv/src/lower/", "lang/viv/src/lower.rs", "lang/viv/src/embed.c", "lang/viv/src/vmath",
             "lang/viv/src/wasm_backend.rs", "lang/viv/src/parallel.rs", "lang/viv/src/growable.rs"]
TRIGGERS = [
    (["lib/hf/", "lib/hfref/", "lib/basis/", "scf/", "tools/hf_names.sh", "tools/cas/gen-integrals.sh"],
     ["3e-lib-hf", "3e-lib-hf-tests", "3e-scf", "scf", "cargo-test-app"]),
    (["dft/"], ["dft"]),
    (["tools/py2viv/"], ["py2viv"]),
    (["xc/", "tools/vivxc/"], ["xc-oracle"]),
    (["www/"], ["www-check"]),
    (["posts/"], ["3e-posts", "www-check"]),
    (C_BACKEND + ["tools/memcheck/"], ["memcheck-quick"]),
    (["lang/viv/src/fast_cli.rs", "lang/viv/src/fastcmp.rs", "tools/fast-check.sh"], ["fast-identity"]),
    (["study/ports/"], ["3e-ports"]),
    (["study/compare/"], ["compare"]),
    (["lib/algo/", "tools/algo/"], ["algo-run"]),
    (["lang/viv/src/export", "tools/export_ci.sh", "tools/export/", "*.export.toml", "kernels/"], ["export"]),
    (["tools/hub-golden.sh"], ["hub-golden"]),
    # a change to the gate is checked by the whole post-merge tier
    (["tools/gate/", "tools/gate.sh", "tools/merge.sh", "tools/three-engines.sh", "tools/both-engines.sh",
      "tools/run-wasm.mjs"], POSTMERGE),
]


def path_matches(path, pat):
    if pat.startswith("*"):
        return path.endswith(pat[1:])
    return path.startswith(pat)


def changed_paths(base=None):
    """The paths this tree changes against `base` (default: its merge base with
    main): committed, uncommitted and untracked. None when there is no main."""
    if base is None:
        base = git("merge-base", "HEAD", "main")
        if not base:
            return None
    out = git("diff", "--name-only", base)
    out += "\n" + git("ls-files", "--others", "--exclude-standard")
    return sorted({l for l in out.splitlines() if l})


def triggered(paths):
    """{suite: [the first path that pulled it in]}, in TRIGGERS order."""
    out = {}
    for pats, suites in TRIGGERS:
        hit = next((p for p in paths if any(path_matches(p, q) for q in pats)), None)
        if hit:
            for s_ in suites:
                out.setdefault(s_, hit)
    return out


def cmd_triggers(args):
    base = args[args.index("--base") + 1] if "--base" in args else None
    paths = changed_paths(base)
    print("path triggers (tools/gate/gate.py TRIGGERS):")
    for pats, suites in TRIGGERS:
        print(f"  {', '.join(pats)}\n      -> {', '.join(suites)}")
    if paths is None:
        print("no branch main here: nothing to compare with")
        return 0
    t = triggered(paths)
    print(f"\nthis tree changes {len(paths)} paths against {base or 'its merge base with main'}; premerge adds "
          + (", ".join(f"{k} ({v})" for k, v in t.items()) if t else "nothing"))
    return 0


# --------------------------------------------------------------- timing lock

def live_locks():
    """Every live study/ports/.timing.lock in any worktree of this repository."""
    paths = [l[9:] for l in git("worktree", "list", "--porcelain", cwd=COMMON).splitlines()
             if l.startswith("worktree ")]
    out = []
    for p in paths:
        f = os.path.join(p, "study", "ports", ".timing.lock")
        if not os.path.exists(f):
            continue
        try:
            m = re.search(r"\d+", open(f).read())
            pid = int(m.group()) if m else None
        except OSError:
            pid = None
        alive = True
        if pid:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                alive = False
            except PermissionError:
                pass
        if alive:
            out.append(f)
    return out


# ------------------------------------------------------------------- running

def fmt_s(s):
    return f"{s:7.1f}s" if s < 600 else f"{s / 60:6.1f}m "


def rel(p):
    return os.path.relpath(p, ROOT)


def tree_state():
    sha = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    return sha, git("rev-parse", "--abbrev-ref", "HEAD"), dirty


# -------------------------------------------------------------------- cache

ENV_ALL = ["viv", "wasm", "cc", "node", "python"]


def coarse(c):
    """A whole check's cache identity: (ident, the inputs known before the run, why not)."""
    kkey, why = gcache.env_key(ENV_ALL)
    if kkey is None:
        return None, None, why
    paths = [] if c.cache == "tree" else c.cache.paths
    t = gcache.tree_hash(paths or None)
    return gcache.h(gcache.FORMAT, "check", c.name, c.cmd, kkey, t), [["t", ",".join(paths), t]], None


# the running checks' processes, by name: a fail-fast stop ends them (their process groups)
PROCS = {}
PROCS_LOCK = threading.Lock()


def run_cmd(c, log, env):
    with open(log, "w") as f:
        f.write(f"$ {c.cmd}\n# cwd {ROOT}\n# {dt.datetime.now().isoformat(timespec='seconds')}\n")
        f.flush()
        p = subprocess.Popen(["bash", "-c", c.cmd], cwd=ROOT, env=env, stdout=f,
                             stderr=subprocess.STDOUT, start_new_session=True)
        with PROCS_LOCK:
            PROCS[c.name] = p
        try:
            rc = p.wait(timeout=c.timeout)
        except subprocess.TimeoutExpired:
            kill_group(p)
            p.wait()
            rc = "timeout"
        finally:
            with PROCS_LOCK:
                PROCS.pop(c.name, None)
        f.write(f"\n# exit {rc}\n")
    return rc


def kill_group(p):
    try:
        os.killpg(p.pid, 15)
    except (ProcessLookupError, PermissionError):
        pass


def lane_budget(opts):
    """(cores, memory in GB) the running checks may use together."""
    if opts.get("serial"):
        return 1, 1e9
    cores = int(os.environ.get("GATE_CORES") or max(2, (os.cpu_count() or 4) - 2))
    return cores, float(os.environ.get("GATE_MEM_GB") or 10)


def select(tier, opts):
    """The checks of a run, and the post-merge suites the path triggers added: {name: path}."""
    checks = [c for c in CHECKS if c.runs_in(tier)]
    trig = {}
    if tier == "postmerge":
        # its checks need the compiler and viv.wasm, which belong to the quick level
        checks = [c for c in CHECKS if c.name in ("build", "wasm")] + checks
    if opts["only"]:
        want = opts["only"]
        unknown = want - {c.name for c in CHECKS}
        if unknown:
            sys.exit(f"gate: no check named {', '.join(sorted(unknown))}")
        return [c for c in CHECKS if c.name in want], trig
    if tier == "premerge" and not opts.get("no_triggers"):
        trig = triggered(changed_paths(opts.get("base")) or [])
        have = {c.name for c in checks}
        checks += [c for c in CHECKS if c.name in trig and c.name not in have and c.runs_in("postmerge")]
        trig = {k: v for k, v in trig.items() if k not in have}
    return checks, trig


_MACHINE_LOCK = None


def machine_lock(say):
    """One gate at a time on this machine, from any worktree.

    Each gate sizes its lanes to the whole machine, so two at once overcommit
    memory and swap. The lock is a file in ~/.cache/viv held with fcntl; the
    kernel releases it when the process ends, so a killed gate never leaves
    it stale. VIV_GATE_NO_LOCK=1 skips it (for the gate's own tests)."""
    global _MACHINE_LOCK
    if os.environ.get("VIV_GATE_NO_LOCK") == "1":
        return
    import fcntl
    path = os.path.join(os.path.expanduser("~"), ".cache", "viv", "gate.lock")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0)
        say(f"gate: another gate holds {path} ({f.read().strip() or 'unknown'}); waiting for it")
        fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0)
    f.truncate()
    f.write(f"pid {os.getpid()} {ROOT}\n")
    f.flush()
    _MACHINE_LOCK = f
    # checks this gate starts (the gate's own tests call run_tier) run under
    # its lock, not waiting for it
    os.environ["VIV_GATE_NO_LOCK"] = "1"


def run_tier(tier, opts):
    checks, trig = select(tier, opts)
    locks = live_locks()
    out = sys.stderr if opts["json"] else sys.stdout
    keep_going = opts["keep_going"] or tier not in FAIL_FAST
    io = threading.Lock()   # the report, the results and the statuses
    cache_lock = threading.Lock()   # tools/gate/cache.py's memo of the tree is not shared safely

    def say(s=""):
        print(s, file=out, flush=True)

    if not opts["list"]:
        machine_lock(say)

    if opts["list"]:
        say(f"{tier}: {len(checks)} checks (the ready ones start longest chain first), from {ROOT}")
        if trig:
            say("added by path triggers: " + ", ".join(f"{k} ({v})" for k, v in trig.items()))
        if locks:
            say(f"timing lock held: {', '.join(locks)}")
        rows = []
        for c in checks:
            flags = [f for f, on in (("heavy", c.heavy), ("timing", c.timing), ("san", c.san),
                                     ("provisional", c.provisional)) if on]
            blocked = locks and (c.timing or (c.heavy and not opts["ignore_lock"]))
            rows.append({"name": c.name, "command": c.cmd, "needs": c.needs, "flags": flags, "level": c.tier,
                         "weight": {"cores": c.cores, "mem_gb": c.mem, "est_s": c.est},
                         "triggered_by": trig.get(c.name), "blocked_now": bool(blocked), "what": c.what})
            say(f"  {c.name:20} {'BLOCKED now ' if blocked else ''}{'[' + ','.join(flags) + '] ' if flags else ''}{c.what}")
            say(f"  {'':20}   $ {c.cmd if not c.cmd.startswith('[ -e www') else '(link www/node_modules) ' + c.cmd.rsplit('; }; ', 1)[1]}")
        if opts["json"]:
            print(json.dumps({"tier": tier, "triggered": trig, "checks": rows}, indent=1))
        return 0

    logdir = os.path.join(ROOT, "target", "gate", tier)
    os.makedirs(logdir, exist_ok=True)
    sha, branch, dirty = tree_state()
    env = dict(os.environ)
    env.setdefault("CARGO_BUILD_JOBS", "4")
    env.setdefault("JOBS", "4")
    use_cache = not opts["no_cache"] and tier != "nightly"
    rate = opts["sample"] if opts["sample"] is not None else SAMPLE
    env.update(GATE_ROOT=ROOT, GATE_COMMON=COMMON, GATE_TIER=tier,
               VIV=VIV, VIV_WASM=os.path.join(ROOT, "www", "public", "viv", "viv.wasm"),
               GATE_CACHE="1" if use_cache else "0", GATE_CACHE_SAMPLE=str(rate if use_cache else 0),
               GATE_SHA=sha, GATE_BRANCH=branch)
    env.pop("VIV_TRACE_INPUTS", None)
    # this run works out its own key environment, after its build (GATE_ENV below)
    env.pop("GATE_ENV", None)
    os.environ.pop("GATE_ENV", None)
    os.environ.update(VIV=VIV, VIV_WASM=env["VIV_WASM"])
    cores, mem = lane_budget(opts)
    say(f"gate {tier}: {len(checks)} checks on {branch} {sha[:10]}{' (uncommitted changes)' if dirty else ''}, {ROOT}")
    say(f"lanes: at most {cores} cores and {mem:g} GB of checks at once, one sanitizer-heavy check at a time"
        + ("" if keep_going else "; the first failure stops the run (--keep-going runs the rest)"))
    if trig:
        say("path triggers added: " + ", ".join(f"{k} ({v})" for k, v in trig.items()))
    if use_cache:
        say(f"cache: on ({gcache.STORE}); a passing result whose inputs are unchanged is reused, "
            f"and {rate:.0%} of the reused ones rerun as a check on the cache (--no-cache: run everything)")
    else:
        say("cache: off (" + ("the nightly tier runs everything" if tier == "nightly" else "--no-cache") + ")")
    if locks:
        say(f"timing lock held: {', '.join(locks)}"
            + ("" if opts["ignore_lock"] else " -- heavy and timing checks are refused"))
    t_start = time.time()
    env_snapshot = {}
    results, status = [], {}
    stop = None        # the check whose failure stopped the run
    killed = set()     # running checks that stop ended
    peak = {"checks": 0, "cores": 0, "mem_gb": 0.0}

    def do_check(c):
        nonlocal stop
        log = os.path.join(logdir, c.name + ".log")
        rerun = f"{GATE} {tier} --only {c.name}"
        r = {"name": c.name, "log": rel(log), "command": c.cmd, "rerun": rerun, "seconds": 0.0}
        if c.name in trig:
            r["triggered_by"] = trig[c.name]
        with io:
            missing = [n for n in c.needs if n in status and status[n] != "PASS"]
            stopped = stop
        if stopped:
            r["status"], r["why"] = "NOT RUN", f"{stopped} failed (--keep-going runs the rest)"
        elif missing:
            r["status"], r["why"] = "SKIP", f"needs {', '.join(missing)}"
        elif locks and (c.timing or (c.heavy and not opts["ignore_lock"])):
            r["status"], r["why"] = "BLOCKED", "timing lock held: " + ", ".join(locks)
        else:
            t0 = time.time()
            cenv = dict(env)
            ident = entry = sampled = known = None
            whole = use_cache and c.cache is not None and c.cache != "items"
            with cache_lock:
                if use_cache and c.cache is not None and "GATE_ENV" not in env:
                    # the key's environment, once per run, after the build: every check
                    # of this run is keyed on the same compiler (tools/gate/cache.py)
                    gcache.forget()
                    for f in (gcache.compiler_id, gcache.env_part):
                        f.cache_clear()
                    parts = gcache.env_parts(ENV_ALL)
                    env["GATE_ENV"] = os.environ["GATE_ENV"] = json.dumps(parts)
                    with io:
                        say(f"cache key: compiler {parts['viv']}; {parts['wasm']}; {parts['cc'].split(' CC=')[0]}; "
                            f"{parts['node']}; {parts['python']}" + (f"; {parts['vivenv']}" if parts["vivenv"] else ""))
                        if (parts["viv"] or "").startswith("binary"):
                            say("  (the compiler is keyed by its binary, not its sources: its dep-info is older than "
                                "a source file, so entries from other worktrees will not match)")
                    env_snapshot.update(parts)
                if "GATE_ENV" in env:
                    cenv["GATE_ENV"] = env["GATE_ENV"]
                if whole:
                    gcache.forget()
                    ident, known, why = coarse(c)
                    entry = ident and gcache.lookup("checks", ident)
                    if entry and rate > 0 and random.random() < rate:
                        sampled, entry = entry, None
            if entry:
                with open(log, "w") as f:
                    f.write(f"$ {c.cmd}\n# not run: {gcache.badge(entry)}\n# the run it reuses: "
                            f"{entry['root']} on {entry['branch']} {entry['sha'][:10]}, {entry['time']}, "
                            f"{entry['seconds']}s\n")
                r["status"], r["cached"] = "PASS", gcache.badge(entry)
                r["seconds"] = round(time.time() - t0, 1)
            else:
                trace = None
                if whole and ident and isinstance(c.cache, Trace):
                    trace = os.path.join(logdir, c.name + ".trace")
                    if os.path.exists(trace):
                        os.remove(trace)
                    cenv["VIV_TRACE_INPUTS"] = trace
                report = os.path.join(logdir, c.name + ".cache.json")
                if os.path.exists(report):
                    os.remove(report)
                if c.cache == "items":
                    cenv["GATE_CACHE_REPORT"] = report
                if whole and ident:
                    with cache_lock:
                        gcache.forget()
                        before = gcache.tree_hash()
                else:
                    before = None
                rc = run_cmd(c, log, cenv)
                r["seconds"] = round(time.time() - t0, 1)
                r["exit"] = rc
                if c.name in killed:
                    r["status"], r["why"] = "NOT RUN", f"stopped when {stop} failed (--keep-going runs the rest)"
                elif rc == 0:
                    r["status"] = "PASS"
                elif c.provisional:
                    r["status"], r["why"] = "WARN", f"exit {rc} (provisional check)"
                elif rc == 75:
                    # EX_TEMPFAIL: the check's own runs were stopped by the machine's load
                    r["status"], r["why"] = "BLOCKED-LOAD", "the machine's load stopped some of its runs (exit 75)"
                elif rc == "timeout" and gload.busy():
                    r["status"], r["why"] = "BLOCKED-LOAD", (f"timed out after {c.timeout}s with the machine busy "
                                                             f"({gload.load1():.0f} load on {gload.ncpu()} cores)")
                else:
                    r["status"], r["why"] = "FAIL", f"exit {rc}" if rc != "timeout" else f"timed out after {c.timeout}s"
                if os.path.exists(report):
                    try:
                        r["cache"] = json.load(open(report))
                    except ValueError:
                        pass
                if c.name in killed:
                    pass
                elif sampled and rc != 0:
                    with cache_lock:
                        bid = gcache.record_mismatch("checks", c.name, sampled, open(log, errors="replace").read())
                        gcache.drop("checks", ident)
                    r["why"] = gcache.mismatch_text("check", c.name, sampled, bid, "failed")
                elif sampled:
                    r["sampled"] = "the rerun reproduced the cached PASS"
                if whole and ident and rc == 0 and c.name not in killed:
                    with cache_lock:
                        gcache.forget()
                        after = gcache.tree_hash()
                        inputs, why = known, None
                        if after != before:
                            why = "the working tree changed while it ran (it, or a check beside it)"
                        elif trace:
                            more, why = gcache.parse_trace(open(trace).read() if os.path.exists(trace) else "",
                                                           [], c.cache.io_ok)
                            inputs = None if more is None else known + more
                        if why:
                            r["uncached"] = why
                        else:
                            gcache.store("checks", ident, inputs, "", r["seconds"])
                elif whole and not ident:
                    r["uncached"] = why
        with io:
            status[c.name] = r["status"]
            prev = next((i for i, x in enumerate(results) if x["name"] == c.name), None)
            if prev is None:
                results.append(r)
            else:
                results[prev] = r
            extra = ""
            if r.get("cached"):
                extra = "  " + r["cached"]
            elif r.get("cache"):
                k = r["cache"]
                extra = (f"  [cache: {k.get('cached', 0)} of {k.get('items', 0)} reused, {k.get('run', 0)} run"
                         + (f", {k['sampled']} sampled" if k.get("sampled") else "")
                         + (f", {k['mismatch']} MISMATCH" if k.get("mismatch") else "") + "]")
            elif r.get("sampled"):
                extra = "  [sampled: the rerun reproduced the cached PASS]"
            elif r.get("uncached"):
                extra = f"  [not cached: {r['uncached']}]"
            if r["status"] == "NOT RUN" and stop and c.name not in killed and not r.get("exit"):
                return r["status"]   # never started: the summary counts it, the report stays short
            say(f"{r['status']:8}{' ' if len(r['status']) >= 8 else ''}{c.name:22}{fmt_s(r['seconds'])}  {r['log'] if 'exit' in r or r.get('cached') else r.get('why', '')}{extra}")
            if r["status"] in ("FAIL", "WARN", "BLOCKED-LOAD") and os.path.exists(log):
                tail = open(log, errors="replace").read().splitlines()[-12:]
                for line in tail:
                    say("    | " + line[:200])
                say(f"    rerun: {rerun}")
                say(f"       or: (cd {shlex.quote(ROOT)} && {c.cmd if len(c.cmd) < 300 else '...see the log header'})")
            if r["status"] in ("FAIL", "BLOCKED") and not keep_going and not stop:
                stop = c.name
                with PROCS_LOCK:
                    for n, p in PROCS.items():
                        if n != c.name:
                            killed.add(n)
                            kill_group(p)
        return r["status"]

    by = {c.name: c for c in checks}
    names = set(by)

    def job_of(c):
        needs = list(c.needs)
        if use_cache and c.cache is not None:
            # its cache key names the compiler and viv.wasm this run builds
            needs += [n for n in ("build", "wasm") if n in names and n != c.name and n not in needs]
        return glanes.Job(c.name, needs, cores=c.cores, mem_gb=c.mem, san=c.san, alone=c.timing, est=c.est)

    def on_start(j, running):
        with io:
            peak["checks"] = max(peak["checks"], len(running))
            peak["cores"] = max(peak["cores"], sum(x.cores for x in running.values()))
            peak["mem_gb"] = max(peak["mem_gb"], sum(x.mem_gb for x in running.values()))

    def run_set(cs):
        glanes.run_jobs([job_of(c) for c in cs], lambda j: do_check(by[j.name]), cores, mem,
                        stop=lambda: stop is not None, on_start=on_start)

    run_set(checks)
    # a check the machine's load stopped is not a verdict: when the load falls, run it
    # again, with every check that needed it (tools/gate/load.py)
    again = [c for c in checks if status.get(c.name) == "BLOCKED-LOAD"]
    if again:
        say(f"\n{len(again)} checks stopped by the machine's load ({', '.join(c.name for c in again)}); "
            f"waiting for load1 <= {gload.ncpu()} (at most 30 minutes; now {gload.load1():.0f}), then running them again")
        if gload.wait(30):
            redo = {c.name for c in again}
            for c in checks:
                if status.get(c.name) == "SKIP" and set(c.needs) & redo:
                    redo.add(c.name)
            for n in redo:
                status.pop(n, None)
            run_set([c for c in checks if c.name in redo])
        else:
            say("the load did not fall: those checks stay BLOCKED-LOAD (the gate has not passed; rerun it)")

    total = round(time.time() - t_start, 1)
    order = {c.name: i for i, c in enumerate(checks)}
    results.sort(key=lambda r: order.get(r["name"], len(order)))
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    bad = [r["name"] for r in results if r["status"] in ("FAIL", "BLOCKED", "BLOCKED-LOAD")]
    verdict = ("FAIL" if any(r["status"] in ("FAIL", "BLOCKED") for r in results)
               else "BLOCKED-LOAD" if bad else "PASS")
    say()
    reused = sum(1 for r in results if r.get("cached"))
    items = [r["cache"] for r in results if r.get("cache")]
    say(f"gate {tier}: {verdict} in {fmt_s(total).strip()}  "
        + "  ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        + (f"  (cache: {reused} checks reused whole; programs {sum(k.get('cached', 0) for k in items)} reused, "
           f"{sum(k.get('run', 0) for k in items)} run, {sum(k.get('sampled', 0) for k in items)} sampled)"
           if use_cache else "  (cache off)"))
    say(f"lanes: at most {peak['checks']} checks at once ({peak['cores']} cores, {peak['mem_gb']:g} GB by their weights); "
        f"the checks' own times add up to {fmt_s(sum(r['seconds'] for r in results)).strip()}")
    if bad:
        say("failed or blocked: " + ", ".join(bad))
        say(f"rerun them: {GATE} {tier} --only {','.join(bad)} --keep-going")
    summary = {"time": dt.datetime.now().isoformat(timespec="seconds"), "tier": tier,
               "verdict": verdict, "seconds": total, "root": ROOT, "branch": branch, "sha": sha,
               "dirty": dirty, "only": sorted(opts["only"]) if opts["only"] else None,
               "triggered": trig, "lanes": {"cores": cores, "mem_gb": mem, "peak": peak},
               "cache": {"on": use_cache, "sample": rate if use_cache else 0, "key": env_snapshot},
               "checks": results}
    with open(os.path.join(logdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    os.makedirs(SHARED, exist_ok=True)
    with open(HISTORY, "a") as f:
        row = {k: summary[k] for k in ("time", "tier", "verdict", "seconds", "branch", "sha", "dirty", "root", "only",
                                       "triggered")}
        row["checks"] = {r["name"]: r["status"] for r in results}
        row["secs"] = {r["name"]: r["seconds"] for r in results if "exit" in r}
        row["cache"] = summary["cache"]
        row["cached"] = [r["name"] for r in results if r.get("cached")]
        row["source"] = os.environ.get("GATE_SOURCE", "manual")
        f.write(json.dumps(row) + "\n")
    if opts["json"]:
        print(json.dumps(summary, indent=1))
    if use_cache and tier in ("premerge", "full") and not opts["only"]:
        with open(os.devnull, "w") as dn:
            subprocess.run([sys.executable, os.path.join(HERE, "cache.py"), "prune", "--days", str(PRUNE_DAYS)],
                           stdout=dn, stderr=dn)
    return 1 if bad else 0


# ------------------------------------------------------------------- status

def cmd_status(args):
    n = int(args[0]) if args else 10
    if not os.path.exists(HISTORY):
        print(f"no gate runs recorded yet ({HISTORY})")
        return 0
    rows = [json.loads(l) for l in open(HISTORY) if l.strip()]
    print(f"last {min(n, len(rows))} of {len(rows)} gate runs ({rel(HISTORY) if HISTORY.startswith(ROOT) else HISTORY})")
    for r in rows[-n:]:
        bad = [k for k, v in r["checks"].items() if v in ("FAIL", "BLOCKED")]
        scope = f" only {','.join(r['only'])}" if r.get("only") else ""
        print(f"  {r['time']}  {r['tier']:7} {r['verdict']}  {fmt_s(r['seconds']).strip():>7}  "
              f"{r['branch']}@{r['sha'][:9]}{'+' if r['dirty'] else ''}  [{r.get('source', 'manual')}]{scope}"
              + (f"  failed: {', '.join(bad)}" if bad else ""))
    for tier in TIERS:
        last = [r for r in rows if r["tier"] == tier and not r.get("only")]
        if last:
            r = last[-1]
            print(f"latest complete {tier}: {r['verdict']} {r['time']} {r['branch']}@{r['sha'][:9]}")
    return 0


# --------------------------------------------------------------------- bugs

FIELDS = ["id", "title", "found", "found_by", "engines", "disagreement", "status", "fixed",
          "commit", "test", "root_cause"]


def load_ledger():
    bugs, errs = [], []
    for i, line in enumerate(open(LEDGER), 1):
        if not line.strip():
            continue
        try:
            b = json.loads(line)
        except json.JSONDecodeError as e:
            errs.append(f"line {i}: not JSON ({e})")
            continue
        for k in FIELDS:
            if k not in b:
                errs.append(f"line {i} ({b.get('id')}): no field `{k}`")
        if b.get("status") not in ("open", "fixed"):
            errs.append(f"{b.get('id')}: status must be open or fixed")
        for k in ("found", "fixed"):
            if b.get(k):
                try:
                    dt.date.fromisoformat(b[k])
                except ValueError:
                    errs.append(f"{b.get('id')}: {k} is not YYYY-MM-DD")
        bugs.append(b)
    ids = [b.get("id") for b in bugs]
    for d in sorted({x for x in ids if ids.count(x) > 1}):
        errs.append(f"id {d} is used twice")
    return bugs, errs


def resolve_test(t):
    """(kind, details) for a test reference, or (None, reason)."""
    if t.startswith("viv:"):
        p = t[4:]
        return ("viv", p) if os.path.isfile(os.path.join(ROOT, p)) else (None, f"no file {p}")
    m = re.fullmatch(r"cargo:([\w-]+)/([\w-]+)::(\w+)", t)
    if not m:
        return None, f"not a test reference: {t}"
    pkg, target, fn = m.groups()
    meta = cargo_packages()
    if pkg not in meta:
        return None, f"no package {pkg}"
    pdir = meta[pkg]
    files = ([os.path.join(pdir, "tests", target + ".rs")] if target != "lib"
             else [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(pdir, "src")) for f in fs if f.endswith(".rs")])
    pat = re.compile(r"\bfn\s+" + re.escape(fn) + r"\s*\(")
    for f in files:
        if os.path.isfile(f) and pat.search(open(f, errors="replace").read()):
            return "cargo", (pkg, target, fn)
    return None, f"no fn {fn} in {pkg}'s {'src/' if target == 'lib' else 'tests/' + target + '.rs'}"


_PKGS = None


def cargo_packages():
    global _PKGS
    if _PKGS is None:
        out = subprocess.run(["cargo", "metadata", "--no-deps", "--format-version", "1", "--offline"],
                             cwd=ROOT, capture_output=True, text=True).stdout
        _PKGS = {p["name"]: os.path.dirname(p["manifest_path"]) for p in json.loads(out)["packages"]}
    return _PKGS


def bugs_check(run_cargo=False):
    """Every fixed bug names a test that exists. Its bugs/tests programs run in
    three engines (no other check runs them); its cargo tests run in
    cargo-test, so here they are only looked up, unless run_cargo (--run)."""
    bugs, errs = load_ledger()
    viv_by_dir, cargo_by_target = {}, {}
    for b in bugs:
        if b.get("status") != "fixed":
            continue
        if not b.get("test"):
            errs.append(f"{b['id']}: fixed, but no test")
            continue
        kind, det = resolve_test(b["test"])
        if kind is None:
            errs.append(f"{b['id']}: {det}")
        elif kind == "viv":
            d, f = os.path.split(det)
            viv_by_dir.setdefault(d, set()).add((f[:-4], b["id"]))
        else:
            pkg, target, fn = det
            cargo_by_target.setdefault((pkg, target), set()).add((fn, b["id"]))
    print(f"ledger: {len(bugs)} bugs, {sum(b.get('status') == 'fixed' for b in bugs)} fixed")
    # each bug's test is an item of its own, cached on its inputs (tools/gate/cargo_tests.py,
    # tools/gate/cache.py): a bug whose test and the code under it are unchanged is not rerun
    if cargo_by_target and not run_cargo:
        n = sum(len(v) for v in cargo_by_target.values())
        print(f"  {n} fixed bugs name a cargo test, each found in its test file (cargo-test runs them; --run reruns them here)")
    elif cargo_by_target:
        import cargo_tests
        bins = cargo_tests.binaries()
        by_item = {it: (exe, d) for it, exe, d in bins}
        wanted = []
        for (pkg, target), fns in sorted(cargo_by_target.items()):
            if target == "lib":
                base = next((it for it in by_item if it.startswith(pkg + ":")
                             and it.split(":")[1] in ("lib", "rlib", "cdylib")), None)
            else:
                base = f"{pkg}:test:{target}"
            for fn, bid in sorted(fns):
                if base not in by_item:
                    errs.append(f"{bid}: no test binary for {pkg}/{target}")
                    continue
                wanted.append((f"{base}?{fn}", bid, f"cargo {pkg}/{target}::{fn}"))
        results, counts, _ = cargo_tests.run(sorted({w[0] for w in wanted}), by_item)
        ok_of = {it: (ok, note) for it, ok, line, note, r in results}
        for it, bid, what in wanted:
            ok, note = ok_of[it]
            print(f"  {'pass' if ok else 'FAIL'}  {bid}  {what}" + (f"  {note}" if note else ""))
            if not ok:
                errs.append(f"{bid}: its test {what[6:]} did not pass")
        print("  " + gcache.summary_line(counts, None))
    for d, progs in sorted(viv_by_dir.items()):
        names = sorted({n for n, _ in progs})
        p = subprocess.run(["python3", "tools/gate/cache.py", "three-engines", d, *names], cwd=ROOT,
                           capture_output=True, text=True)
        lines = {l.split()[0]: l for l in p.stdout.splitlines() if l and not l.startswith(" ")}
        for n, bid in sorted(progs):
            line = lines.get(n, "")
            ok = p.returncode == 0 or (" 0 failed; identical output" in line)
            print(f"  {'pass' if ok else 'FAIL'}  {bid}  {d}/{n}.viv  {line[34:].strip()}")
            if not ok:
                errs.append(f"{bid}: {d}/{n}.viv does not pass in three engines")
    for e in errs:
        print("ERROR " + e)
    return 1 if errs else 0


def iso_week(d):
    y, w, _ = dt.date.fromisoformat(d).isocalendar()
    return f"{y}-W{w:02d}"


def cmd_bugs(args):
    if "--check" in args:
        return bugs_check("--run" in args)
    bugs, errs = load_ledger()
    for e in errs:
        print("ERROR " + e)
    opn = [b for b in bugs if b["status"] == "open"]
    fixed = [b for b in bugs if b["status"] == "fixed"]
    print(f"bug ledger ({rel(LEDGER)}): {len(bugs)} bugs, {len(opn)} open, {len(fixed)} fixed")
    weeks = sorted({iso_week(b["found"]) for b in bugs} | {iso_week(b["fixed"]) for b in fixed if b.get("fixed")})
    print(f"\n  {'week':9} {'found':>5} {'fixed':>5}  open at week's end")
    for w in weeks:
        nf = sum(iso_week(b["found"]) == w for b in bugs)
        nx = sum(1 for b in fixed if b.get("fixed") and iso_week(b["fixed"]) == w)
        still = sum(1 for b in bugs if iso_week(b["found"]) <= w
                    and not (b["status"] == "fixed" and b.get("fixed") and iso_week(b["fixed"]) <= w))
        print(f"  {w:9} {nf:5} {nx:5}  {still:3} {'#' * still}")
    by_engine = {}
    for b in opn:
        for e in b.get("engines") or ["?"]:
            by_engine[e] = by_engine.get(e, 0) + 1
    print("\n  open by engine: " + ", ".join(f"{k} {v}" for k, v in sorted(by_engine.items())))
    dis = sorted((b for b in bugs if b.get("disagreement")), key=lambda b: b["found"])
    if dis:
        last = dis[-1]
        days = (dt.date.today() - dt.date.fromisoformat(last["found"])).days
        print(f"  days since the last engine disagreement was found: {days} "
              f"({last['id']}, {last['found']}: {last['title'][:70]})")
        print(f"  engine disagreements: {len(dis)} found, {sum(b['status'] == 'open' for b in dis)} open")
    untested = [b["id"] for b in fixed if not b.get("test")]
    if untested:
        print(f"  fixed without a test: {', '.join(untested)}")
    return 1 if errs else 0


# ------------------------------------------------------------------ ratchet

def usage_counts():
    p = subprocess.run([VIV, "features", "--usage", "--json", "--max-lines", "0"], cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"viv features --usage failed:\n{p.stderr[-2000:]}")
    # the envelope: the report is its `result`
    return {x["id"]: x["count"] for x in json.loads(p.stdout)["result"]["patterns"]}


def cmd_ratchet(args):
    now = usage_counts()
    if "--update" in args or not os.path.exists(RATCHET):
        old = json.load(open(RATCHET)) if os.path.exists(RATCHET) else {}
        new = {k: min(v, old.get(k, v)) if "--force" not in args else v for k, v in sorted(now.items())}
        json.dump(new, open(RATCHET, "w"), indent=1, sort_keys=True)
        open(RATCHET, "a").write("\n")
        print(f"wrote {rel(RATCHET)}: {sum(new.values())} hand-written patterns over {len(new)} kinds")
        return 0
    limit = json.load(open(RATCHET))
    bad, lower = [], []
    for k, v in sorted(now.items()):
        lim = limit.get(k, 0)
        if v > lim:
            bad.append(f"  {k}: {v} (the ratchet allows {lim})")
        elif v < lim:
            lower.append(f"  {k}: {v} (was {lim})")
    print(f"viv features --usage: {sum(now.values())} hand-written patterns (ratchet {sum(limit.values())})")
    if lower:
        print("fewer than the ratchet (lower it: tools/gate.sh ratchet --update):\n" + "\n".join(lower))
    if bad:
        print("MORE than the ratchet: replace them with the feature (`viv features --usage <file>` "
              "names it), or raise the ratchet with tools/gate.sh ratchet --update --force and say why:\n"
              + "\n".join(bad))
        return 1
    return 0


# -------------------------------------------------------------------- ports

def ports():
    base = os.path.join(ROOT, "study", "ports")
    return sorted(d for d in os.listdir(base) if os.path.isfile(os.path.join(base, d, "port.toml")))


def parity_programs(port):
    """The parity programs: the cases marked `parity = true`, and every program
    of viv/ that is not a case or a case's generated variant (<case>_t, the
    timing program, and <case>_0); those are check modules such as pzcheck."""
    import tomllib
    conf = tomllib.load(open(os.path.join(ROOT, "study", "ports", port, "port.toml"), "rb"))
    cases = {c["name"]: c for c in conf.get("case", [])}
    vdir = os.path.join(ROOT, "study", "ports", port, "viv")
    stems = sorted(f[:-4] for f in os.listdir(vdir) if f.endswith(".viv"))
    variant = re.compile(r"(.+)_(t|\d+)$")
    out = []
    for s in stems:
        if s in cases:
            if cases[s].get("parity"):
                out.append(s)
            continue
        m = variant.fullmatch(s)
        if m and m.group(1) in cases:
            continue
        out.append(s)
    return out


def cmd_ports_parity(args):
    bad = 0
    for port in ports():
        names = parity_programs(port)
        print(f"== {port}: {' '.join(names)}", flush=True)
        # one program at a time, cached per program under the gate (tools/gate/cache.py)
        bad |= subprocess.run(["python3", "tools/gate/cache.py", "three-engines", f"study/ports/{port}/viv", *names],
                              cwd=ROOT).returncode
    return 1 if bad else 0


def cmd_ports_verify(args):
    bad = 0
    for port in ports():
        print(f"== port.py verify {port}", flush=True)
        bad |= subprocess.run(["python3", "study/ports/harness/port.py", "verify", port], cwd=ROOT).returncode
    return 1 if bad else 0


# ------------------------------------------------------------------- golden

RUN_FILES = {"interp", "native", "direct"}


def cmd_golden(args):
    """Record this tree's hub-golden snapshot and compare it with a base one.
    base: main's tip (the merge base with main); nightly: the last nightly's.
    A snapshot of a clean tree is kept under its commit, for later runs to
    compare against; at most KEEP are kept."""
    mode = args[0] if args else "base"
    sha, _, dirty = tree_state()
    os.makedirs(GOLDEN, exist_ok=True)
    if mode == "base":
        base = git("merge-base", "HEAD", "main")
        if base == sha:   # on main itself: compare with the previous commit's snapshot
            base = git("rev-parse", "HEAD~1")
    else:
        prev = [json.loads(l) for l in open(HISTORY)] if os.path.exists(HISTORY) else []
        prev = [r for r in prev if r["tier"] == "nightly" and r["sha"] != sha
                and os.path.isdir(os.path.join(GOLDEN, r["sha"]))]
        base = prev[-1]["sha"] if prev else None
    new = os.path.join(GOLDEN, sha) if not dirty else os.path.join(ROOT, "target", "gate", "golden-work")
    if not os.path.isdir(new) or dirty:
        shutil.rmtree(new, ignore_errors=True)
        print(f"recording {new}", flush=True)
        if subprocess.run(["tools/hub-golden.sh", "record", new], cwd=ROOT).returncode:
            print("hub-golden record failed")
            return 1
    snaps = sorted((d for d in os.listdir(GOLDEN) if re.fullmatch(r"[0-9a-f]{40}", d)),
                   key=lambda d: os.path.getmtime(os.path.join(GOLDEN, d)))
    for d in snaps[:-KEEP]:
        shutil.rmtree(os.path.join(GOLDEN, d), ignore_errors=True)
    if not base or not os.path.isdir(os.path.join(GOLDEN, base)):
        print(f"no snapshot of the base ({base or 'no previous nightly'}) in {GOLDEN}: "
              "nothing to compare with. This run's snapshot is kept for the next.")
        return 0
    p = subprocess.run(["tools/hub-golden.sh", "compare", os.path.join(GOLDEN, base), new],
                       cwd=ROOT, capture_output=True, text=True)
    print(p.stdout + p.stderr)
    changed = [l for l in p.stdout.splitlines()
               if l.startswith("DIFFERS: ") and os.path.basename(l.split()[-1]) in RUN_FILES]
    if changed and not os.environ.get("GATE_ACCEPT_GOLDEN"):
        print(f"{len(changed)} run outputs changed against {base[:10]}. If every change is intended, "
              "rerun with GATE_ACCEPT_GOLDEN=1 and say so in the merge message.")
        return 1
    print(f"no run output changed against {base[:10]}" if not changed else "run output changes accepted")
    return 0


KEEP = 6


# --------------------------------------------------------------------- main

def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__.split("\n\n")[1])
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    sub = {"status": cmd_status, "bugs": cmd_bugs, "ratchet": cmd_ratchet, "golden": cmd_golden,
           "cache": lambda a: gcache.main(a or ["stats"]),
           "ports-parity": cmd_ports_parity, "ports-verify": cmd_ports_verify, "triggers": cmd_triggers}
    if cmd in sub:
        return sub[cmd](rest)
    if cmd not in TIERS:
        sys.exit(f"gate: unknown tier or command `{cmd}` ({', '.join(TIERS)}, status, bugs, ratchet, triggers)")
    opts = {"list": False, "json": False, "keep_going": False, "ignore_lock": False, "only": None,
            "no_cache": False, "sample": None, "base": None, "no_triggers": False, "serial": False}
    i = 0
    while i < len(rest):
        a = rest[i]
        if a in ("--list", "--json", "--keep-going", "--ignore-lock", "--no-cache", "--no-triggers", "--serial"):
            opts[a[2:].replace("-", "_")] = True
        elif a == "--base":
            i += 1
            opts["base"] = rest[i]
        elif a == "--sample":
            i += 1
            opts["sample"] = float(rest[i].rstrip("%")) / (100 if rest[i].endswith("%") else 1)
        elif a == "--only":
            i += 1
            opts["only"] = set(rest[i].split(","))
        else:
            sys.exit(f"gate: unknown option {a}")
        i += 1
    return run_tier(cmd, opts)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
