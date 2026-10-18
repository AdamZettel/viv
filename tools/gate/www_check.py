#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The site's `npm run stage && npm run check`, one script at a time, each cached.

    python3 tools/gate/www_check.py [--list]

The scripts are www/package.json's: every `node scripts/...` of `stage`, in
order, then every part of `check`, in order, each run as its own item
(cwd www/), with tools/gate/fstrace.c recording every file it and the
programs it starts open, look at or list, and every file it writes.

Under the gate (GATE_CACHE=1) a script whose inputs have not changed since
it passed is not run again (tools/gate/cache.py, kind www): what it wrote
(its outputs under the repository: the staged data, the site's public
files) is put back from the store instead, so the scripts after it see
what it would have written. Its inputs are the files it read, as they stood
when it finished, plus viv, viv.wasm, node, Python and cc; a file a later
script writes again is not compared (its state now is that script's last
product, which the later script's own key covers). A script that
starts git or a shell (claims-build reads `git log`) has inputs the trace
cannot see and always runs.

Two parts of `check` are left out because the gate runs the same commands
as checks of their own: `py2viv:rules` (python3 tools/py2viv/check_rules.py,
the gate's py2viv check) and `algo:check` (python3 tools/algo/check.py, the
gate's algo-spec; algo-run runs it with the programs too). Exits nonzero
when a script fails.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cache  # noqa: E402

ROOT = cache.ROOT
WWW = os.path.join(ROOT, "www")
# run by the gate as checks of their own (py2viv, algo-spec / algo-run)
ELSEWHERE = {"py2viv:rules": "py2viv", "algo:check": "algo-spec, algo-run"}


def scripts():
    """[(item, argv)] in the order `npm run stage && npm run check` runs them."""
    s = json.load(open(os.path.join(WWW, "package.json")))["scripts"]
    out = []

    def expand(name, seen=()):
        for part in (p.strip() for p in s[name].split("&&")):
            m = re.fullmatch(r"npm run (\S+)", part)
            if m:
                if m.group(1) in ELSEWHERE:
                    continue
                expand(m.group(1), seen + (name,))
            else:
                out.append((f"{name}: {part}", part.split()))
    expand("stage")
    expand("check")
    return out


def outputs(trace_text):
    """The files under the repository a run wrote (not cargo's target/, not the temporary directory)."""
    out = set()
    for line in trace_text.splitlines():
        p = line.split("\t")
        if p[0] == "w" and len(p) == 3:
            rp = os.path.realpath(p[1])
            if rp.startswith(ROOT + os.sep) and not rp.startswith(cache.BUILD) and os.path.isfile(rp) \
                    and "/node_modules/" not in rp and "/.git/" not in rp:
                out.add(os.path.relpath(rp, ROOT))
    return sorted(out)


def main(argv):
    items = scripts()
    if "--list" in argv:
        for it, a in items:
            print(f"{it}")
        for k, v in ELSEWHERE.items():
            print(f"(not here: {k}, which the gate runs as {v})")
        return 0
    # www/node_modules: this tree's, or the main repository's
    link = None
    nm = os.path.join(WWW, "node_modules")
    if not os.path.exists(nm):
        src = os.path.join(cache.COMMON, "www", "node_modules")
        if not os.path.isdir(src):
            sys.exit(f"no www/node_modules here or in {cache.COMMON}")
        os.symlink(src, nm)
        link = nm
    lib = cache.fstrace()
    argv_of = dict(items)
    produced = {}   # item -> a directory holding what it wrote, for the store
    work = tempfile.mkdtemp(prefix="gate-www-")

    def command(it, trace):
        env = dict(os.environ)
        if trace:
            env["VIV_TRACE_INPUTS"] = trace
            env["DYLD_INSERT_LIBRARIES"] = lib
        p = subprocess.run(argv_of[it], cwd=WWW, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if trace and os.path.exists(trace):
            d = os.path.join(work, str(len(produced)))
            for f in outputs(open(trace).read()):
                os.makedirs(os.path.dirname(os.path.join(d, f)), exist_ok=True)
                shutil.copy2(os.path.join(ROOT, f), os.path.join(d, f))
            os.makedirs(d, exist_ok=True)
            produced[it] = d
        return p.stdout.decode(errors="replace"), p.returncode

    def restore(it, cached):
        for dp, _, fs in os.walk(cached):
            for f in fs:
                src = os.path.join(dp, f)
                dst = os.path.join(ROOT, os.path.relpath(src, cached))
                if not (os.path.isfile(dst) and open(dst, "rb").read() == open(src, "rb").read()):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copyfile(src, dst)

    def outputs_of(it):
        """What the item wrote when it last passed (its stored files)."""
        e = cache.lookup_any("www", it)
        if not e or not e.get("artifacts"):
            return set()
        d = os.path.join(os.path.dirname(cache.entry_path("www", e["id"])), e["artifacts"])
        return {os.path.relpath(os.path.join(dp, f), d) for dp, _, fs in os.walk(d) for f in fs}

    names = [it for it, _ in items]
    later = {}
    acc = set()
    for it in reversed(names):
        later[it] = frozenset(acc)
        acc |= outputs_of(it)

    bad = 0
    total = {"items": 0, "cached": 0, "run": 0, "sampled": 0, "stored": 0, "uncacheable": 0, "mismatch": 0}
    try:
        for it, _ in items:   # in order: a script reads what the ones before it wrote
            res, counts, why_not = cache.run_items(
                "www", [it], command, lambda i: [os.path.join(WWW, "package.json")], lambda out, rc: rc == 0,
                artifacts=(lambda i: produced.get(i, work), restore), ignore=lambda i: later.get(i, frozenset()))
            for k in total:
                total[k] += counts.get(k, 0)
            r = res[0]
            tail = [l for l in r.output.splitlines() if l.strip()][-3:]
            print(f"== {it}" + (f"    {r.note}" if r.note else ""))
            for l in (tail if r.status != "fail" else r.output.splitlines()[-40:]):
                print("   " + l[:200])
            if r.status in ("fail", "mismatch"):
                bad = 1
                break
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if link:
            os.remove(link)
    print(cache.summary_line(total, None if cache.enabled() else "the cache is off"))
    for k, v in ELSEWHERE.items():
        print(f"(not run here: npm run {k}, which the gate runs as {v})")
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
