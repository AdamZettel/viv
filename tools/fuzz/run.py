#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Differential testing of Viv's three engines on vivsmith's programs.

    python3 tools/fuzz/run.py [--seed S] [--count N | --seconds T] [--jobs J]
                              [--size X] [--san] [--engines native,wasm]
                              [--out DIR] [--known tools/fuzz/known.json] [--reduce]

Each seed's program (tools/fuzz/vivsmith.py) runs in the interpreter, as
native C (`viv build`, the exact build) and as wasm (the direct backend, in
node); with --san the C is also built with AddressSanitizer and
UndefinedBehaviorSanitizer. A seed fails when

  refused    the front end refuses the program (vivsmith only writes valid
             programs, so this is a front-end bug or a generator bug)
  interp     the interpreter stops with an error or a panic, or hangs
  build      `viv build` or the wasm compile fails on a program the interpreter ran
  crash      native or wasm stops with an error, a signal, or hangs
  differs    native or wasm prints other bytes than the interpreter
  san        a sanitizer report

Failing programs are kept in --out (default target/fuzz) as fail_<seed>.viv
with a .txt of the evidence; --reduce shrinks each one (tools/fuzz/reduce.py).
--known (tools/fuzz/known.json) names the seeds that fail on an open ledger
bug, and the kinds of failure an open bug makes on any seed ("open_kinds",
used with --open-kinds, for runs on new seeds): they are reported, and the run fails only on a
failure not listed (or a listed seed whose failure changed kind). Exits
nonzero on any unlisted failure.

The run is nice'd, and runs at most 4 programs at a time.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import vivsmith  # noqa: E402

VIV = os.environ.get("VIV") or os.path.join(ROOT, "target", "release", "viv")
NODE = os.environ.get("NODE", "node")
CC = os.environ.get("CC", "cc")
RUN_WASM = os.path.join(ROOT, "tools", "run-wasm.mjs")
TIMEOUT = 20
NOISE = re.compile(r"^(__graph__|__trace__)")


def run(cmd, timeout=TIMEOUT, env=None, cwd=None):
    """(returncode, stdout+stderr); returncode None on a timeout."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd,
                           errors="replace")
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return None, ""


def clean(out):
    return "\n".join(l for l in out.splitlines() if not NOISE.match(l))


def first_error(out):
    for l in out.splitlines():
        if l.startswith("error") or "panicked" in l:
            return l.strip()
    return ""


def check(src, engines=("native", "wasm"), san=False, workdir=None):
    """Run one program; (kind, detail) of its first failure, or (None, None).

    kind is one of refused, interp, build, crash, differs, san (module doc)."""
    own = workdir is None
    d = workdir or tempfile.mkdtemp(prefix="vivsmith-")
    try:
        p = os.path.join(d, "p.viv")
        with open(p, "w") as f:
            f.write(src)
        rc, out = run([VIV, "check", p], cwd=d)
        if rc != 0:
            return ("interp" if "panicked" in out else "refused"), (first_error(out) or f"exit {rc}") + " " + _where(out)
        rc, out = run([VIV, "run", p], cwd=d)
        if rc is None:
            return "interp", "the interpreter did not finish in %ds" % TIMEOUT
        ref = clean(out)
        if rc != 0 or first_error(ref):
            return "interp", (first_error(ref) or f"exit {rc}") + " " + _where(out)
        for eng in engines:
            if eng == "native":
                rc, out = run([VIV, "build", p, "-o", os.path.join(d, "p.bin")], timeout=60, cwd=d)
                if rc != 0:
                    return "build", "native: " + (first_error(out) or f"exit {rc}")
                rc, out = run([os.path.join(d, "p.bin")], cwd=d)
            else:
                rc, out = run([VIV, p, "-o", os.path.join(d, "p.wasm")], timeout=60, cwd=d)
                if rc != 0:
                    return "build", "wasm: " + (first_error(out) or f"exit {rc}")
                rc, out = run([NODE, RUN_WASM, os.path.join(d, "p.wasm")], cwd=d)
            got = clean(out)
            if rc is None:
                return "crash", f"{eng}: did not finish in {TIMEOUT}s"
            if rc != 0 or first_error(got):
                return "crash", f"{eng}: " + (first_error(got) or f"exit {rc}")
            if got != ref:
                return "differs", f"{eng}: " + _first_diff(ref, got)
        if san:
            rc, out = run([VIV, p, "-o", os.path.join(d, "p.c")], timeout=60, cwd=d)
            if rc != 0:
                return "build", "C: " + (first_error(out) or f"exit {rc}")
            rc, out = run([CC, "-w", "-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=undefined",
                           "-fno-omit-frame-pointer", "-o", os.path.join(d, "p.san"), os.path.join(d, "p.c"), "-lm"],
                          timeout=120, cwd=d)
            if rc != 0:
                return "build", "C (sanitizers): " + (first_error(out) or out.strip().splitlines()[-1:] or "")
            env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0:halt_on_error=1")
            rc, out = run([os.path.join(d, "p.san")], timeout=60, env=env, cwd=d)
            m = re.search(r"(ERROR: AddressSanitizer[^\n]*|runtime error:[^\n]*)", out)
            if m:
                return "san", m.group(1)[:200]
            if rc is None:
                return "crash", "sanitizer build: did not finish"
            if clean(out) != ref:
                return "differs", "sanitizer build: " + _first_diff(ref, clean(out))
        return None, None
    finally:
        if own:
            shutil.rmtree(d, ignore_errors=True)


def _where(out):
    m = re.search(r"-->\s*\S+:(\d+):(\d+)", out)
    return f"(line {m.group(1)})" if m else ""


def _first_diff(a, b):
    al, bl = a.splitlines(), b.splitlines()
    for i in range(max(len(al), len(bl))):
        x = al[i] if i < len(al) else "<end>"
        y = bl[i] if i < len(bl) else "<end>"
        if x != y:
            return f"line {i + 1}: interp `{x[:100]}` vs `{y[:100]}`"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--size", type=float, default=1.0)
    ap.add_argument("--san", action="store_true")
    ap.add_argument("--engines", default="native,wasm")
    ap.add_argument("--out", default=os.path.join(ROOT, "target", "fuzz"))
    ap.add_argument("--known", default=None)
    ap.add_argument("--reduce", action="store_true")
    ap.add_argument("--open-kinds", action="store_true",
                    help="also count as known every failure of a kind in known.json's open_kinds (new seeds)")
    ap.add_argument("--cover", action="store_true", help="print the constructs the run's programs used")
    a = ap.parse_args()
    jobs = max(1, min(a.jobs, 4))
    try:
        os.nice(10)
    except OSError:
        pass
    if not os.path.exists(VIV):
        sys.exit(f"run.py: no compiler at {VIV} (cargo build --release -p viv, or VIV=...)")
    engines = tuple(e for e in a.engines.split(",") if e)
    seed0 = a.seed if a.seed is not None else int(time.time())
    if a.count is None and a.seconds is None:
        a.count = 20
    known, open_kinds = {}, {}
    if a.known:
        kj = json.load(open(a.known))
        known = {int(k): v for k, v in kj.get("seeds", {}).items()}
        open_kinds = kj.get("open_kinds", {}) if a.open_kinds else {}

    def is_known(seed, kind):
        return (seed in known and known[seed]["kind"] == kind) or kind in open_kinds
    os.makedirs(a.out, exist_ok=True)
    start = time.time()
    cover = Counter()
    results = {}
    kinds = Counter()

    def one(seed):
        src, cv = vivsmith.generate(seed, a.size)
        kind, detail = check(src, engines, a.san)
        return seed, src, cv, kind, detail

    next_seed = seed0
    pending = set()
    with ThreadPoolExecutor(jobs) as ex:
        def more():
            nonlocal next_seed
            if a.count is not None and next_seed - seed0 >= a.count:
                return False
            if a.seconds is not None and time.time() - start >= a.seconds:
                return False
            pending.add(ex.submit(one, next_seed))
            next_seed += 1
            return True
        for _ in range(jobs):
            more()
        while pending:
            done = next(f for f in _wait(pending))
            pending.discard(done)
            seed, src, cv, kind, detail = done.result()
            cover.update(cv)
            results[seed] = (kind, detail)
            if kind:
                kinds[kind] += 1
                path = os.path.join(a.out, f"fail_{seed}.viv")
                with open(path, "w") as f:
                    f.write(src)
                with open(path[:-4] + ".txt", "w") as f:
                    f.write(f"{kind}: {detail}\n")
                tag = ""
                if is_known(seed, kind):
                    tag = f"  (known: {known[seed]['bug'] if seed in known else open_kinds[kind]})"
                print(f"  FAIL seed {seed}: {kind}: {detail}{tag}", flush=True)
            more()
    secs = time.time() - start
    n = len(results)
    unlisted = [s for s, (k, _) in results.items() if k and not is_known(s, k)]
    fixed = [s for s in known if s in results and results[s][0] is None]
    print(f"vivsmith: {n} programs (seeds {seed0}..{seed0 + n - 1}) in {secs:.1f}s, "
          f"{n / secs:.2f} programs/s, {jobs} jobs, engines interp+{'+'.join(engines)}{'+san' if a.san else ''}")
    print(f"  failures: {sum(kinds.values())} ({', '.join(f'{k} {v}' for k, v in sorted(kinds.items())) or 'none'}); "
          f"{len(unlisted)} not in {os.path.basename(a.known) if a.known else 'a known list'}")
    for s in fixed:
        print(f"  seed {s} passes now: {known[s]['bug']} may be fixed (update {a.known})")
    if a.cover:
        print(f"  constructs used ({len(cover)}):")
        for k, v in sorted(cover.items()):
            print(f"    {v:6d}  {k}")
    if a.reduce and unlisted:
        import reduce as red
        for s in sorted(unlisted):
            path = os.path.join(a.out, f"fail_{s}.viv")
            red.reduce_file(path, path[:-4] + ".min.viv", results[s][0], results[s][1], engines, a.san)
    return 1 if unlisted else 0


def _wait(fs):
    from concurrent.futures import wait, FIRST_COMPLETED
    done, _ = wait(fs, return_when=FIRST_COMPLETED)
    return done


if __name__ == "__main__":
    sys.exit(main())
