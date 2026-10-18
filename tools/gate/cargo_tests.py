#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The workspace's Rust tests, one test binary at a time, each cached on its inputs.

    python3 tools/gate/cargo_tests.py [--only pkg:kind:name,...] [--skip pkg:kind:name,...]

`cargo test --workspace --no-run` builds every test binary; then each binary
runs on its own (CARGO_TEST_JOBS at once, default 2, the slowest first), as `cargo test` would run
it: from its package's directory, with cargo's environment. The doc tests
run as one more item (`cargo test --workspace --doc`).

Under the gate (GATE_CACHE=1) a binary whose inputs have not changed since
it passed is not run again (tools/gate/cache.py, kind cargo-test). Its
inputs:

  - the sources it was built from: every file in its cargo dep-info;
  - every file it, or a program it starts, opens, looks at or lists, as
    tools/gate/fstrace.c records them (DYLD_INSERT_LIBRARIES); a program
    it starts counts by its own sources when cargo built it, by its bytes
    otherwise; viv also reports what it reads itself (VIV_TRACE_INPUTS);
  - rustc, cc, node, Python and viv.wasm, and the OS.

A binary that starts a shell (or another program macOS shields from
DYLD_INSERT_LIBRARIES that may read what its arguments do not name) has
inputs the trace cannot see, and is never cached; nor is a failure.

Exits nonzero when a binary fails. run() also takes `name?test,...` items,
which run only the tests those names match and pass only if each named test ran and passed:
gate.py's bugs check reuses each bug's test that way.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cache  # noqa: E402

ROOT = cache.ROOT


WS_DEPS = set()   # the dep-info files of the workspace's libraries


def binaries():
    """[(item, executable, package dir)] of every test binary, built."""
    p = subprocess.run(["cargo", "test", "--workspace", "--no-run", "--message-format=json-render-diagnostics"],
                       cwd=ROOT, stdout=subprocess.PIPE, text=True)
    if p.returncode:
        sys.exit("cargo test --no-run failed")
    out = []
    for line in p.stdout.splitlines():
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("reason") != "compiler-artifact":
            continue
        # a workspace library: every test binary may be built against it. Its own
        # unit-test binary's dep-info lists all of its sources.
        if (m["manifest_path"].startswith(ROOT + os.sep) and m.get("profile", {}).get("test")
                and m.get("executable") and set(m["target"]["kind"]) & {"lib", "rlib"}):
            WS_DEPS.add(m["executable"] + ".d")
        if not m.get("profile", {}).get("test") or not m.get("executable"):
            continue
        t = m["target"]
        item = f"{pkgname(m)}:{t['kind'][0]}:{t['name']}"
        out.append((item, m["executable"], os.path.dirname(m["manifest_path"])))
    return sorted(set(out))


def pkgname(m):
    # package_id: "path+file:///…/lang/viv#viv@0.1.0" (new cargo) or "viv 0.1.0 (path+file://…)" (old)
    pid = m["package_id"]
    if " " in pid:
        return pid.split()[0]
    tail = pid.rsplit("#", 1)[-1]
    return tail.split("@")[0] if "@" in tail else os.path.basename(os.path.dirname(m["manifest_path"]))


def cargo_env(pkgdir, name):
    env = dict(os.environ)
    meta = {"CARGO_MANIFEST_DIR": pkgdir, "CARGO_PKG_NAME": name, "CARGO_CRATE_NAME": name.replace("-", "_")}
    env.update(meta)
    deps = os.path.join(ROOT, "target", "debug", "deps")
    env["DYLD_FALLBACK_LIBRARY_PATH"] = os.pathsep.join(filter(None, [deps, env.get("DYLD_FALLBACK_LIBRARY_PATH")]))
    return env


def run(items, by_item):
    """Run the items (a binary's name, or `name?test1,test2` for those tests
    only, or "doc"), cached; [(item, ok, summary line, note)], and the counts."""
    lib = cache.fstrace()

    def filters_of(it):
        return it.split("?", 1)[1].split(",") if "?" in it else []

    # the slowest first, by what the store last recorded
    def last_seconds(it):
        e = cache.lookup_any("cargo-test", it)
        return -(e or {}).get("seconds", 0)
    order = sorted(items, key=last_seconds)

    def command(it, trace):
        base = it.split("?", 1)[0]
        if base == "doc":
            cmd, cwd, env = ["cargo", "test", "--workspace", "--doc"], ROOT, dict(os.environ)
        else:
            exe, d = by_item[base]
            f = filters_of(it)
            cmd, cwd, env = [exe, *f], d, cargo_env(d, base.split(":")[0])
        if trace:
            env["VIV_TRACE_INPUTS"] = trace
            env["DYLD_INSERT_LIBRARIES"] = lib
        p = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = p.stdout.decode(errors="replace")
        # a filter that matched nothing passes in cargo: here it must name tests that ran and passed
        missing = [t for t in filters_of(it) if not re.search(r"^test (\S+::)?" + re.escape(t.split("::")[-1]) + r" \.\.\. ok$", out, re.M)]
        return out + (f"\nnot run or not passed: {', '.join(missing)}\n" if missing else ""), (p.returncode or (1 if missing else 0))

    def inputs_of(it):
        base = it.split("?", 1)[0]
        if base == "doc":
            return []
        # the binary's own sources, and those of every workspace library it may link
        files = cache.built_from(by_item[base][0])
        for d in sorted(WS_DEPS):
            if os.path.isfile(d):
                files += cache.dep_files(d)
        return sorted(set(files))

    # two binaries at once (CARGO_TEST_JOBS): each runs its tests on every core already, and
    # some build C under ASan, a few GB a compile; cargo test itself runs one at a time
    workers = max(1, int(os.environ.get("CARGO_TEST_JOBS", "2")))
    res, counts, why_not = cache.run_items("cargo-test", order, command, inputs_of, lambda out, rc: rc == 0,
                                           workers=workers)
    by = {r.item: r for r in res}
    out = []
    for it in items:
        r = by[it]
        summary = [l for l in r.output.splitlines() if l.startswith("test result:")]
        line = summary[-1] if summary else (r.output.strip().splitlines() or ["(no output)"])[-1]
        out.append((it, r.status in ("cached", "pass"), line, r.note, r))
    return out, counts, why_not


def main(argv):
    only, skip = None, set()
    i = 0
    while i < len(argv):
        if argv[i] == "--only":
            only = set(argv[i + 1].split(","))
            i += 2
        elif argv[i] == "--skip":   # the gate's cargo-test leaves these to cargo-test-app
            skip = set(argv[i + 1].split(","))
            i += 2
        else:
            sys.exit(f"cargo_tests.py: unknown option {argv[i]}")
    bins = binaries()
    if only is not None:
        bins = [b for b in bins if b[0] in only]
    bins = [b for b in bins if b[0] not in skip]
    by_item = {it: (exe, d) for it, exe, d in bins}
    items = [it for it, _, _ in bins] + (["doc"] if only is None else [])
    results, counts, why_not = run(items, by_item)
    bad = 0
    for it, ok, line, note, r in results:
        print(f"{it:44} {line[:90]}" + (f"    {note}" if note else ""))
        if not ok:
            bad = 1
            print("\n".join("    | " + l for l in r.output.splitlines()[-40:]))
    print(cache.summary_line(counts, why_not))
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
