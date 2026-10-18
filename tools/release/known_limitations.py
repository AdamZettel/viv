#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Write KNOWN_LIMITATIONS.md from the records the repository keeps.

    python3 tools/release/known_limitations.py           write KNOWN_LIMITATIONS.md
    python3 tools/release/known_limitations.py --check   exit 1 if the file is stale

Sources, each read as it is (nothing is typed in here, except the short
Platforms section at the top):

- bugs/ledger.jsonl: every bug whose status is "open";
- study/rosetta/friction.jsonl: what the Rosetta Code study found missing or
  awkward, ranked by the tasks it affects (read from the `rosetta` branch
  when the directory is not in this tree);
- study/ports/pyscf/FRICTION.json: the Hartree-Fock port's open findings;
- study/benchmarks/limitations.json: the benchmark cells Viv could not run,
  with the reason recorded for each, and the numpy-benchmarks kernels py2viv
  refuses.

    python3 tools/release/known_limitations.py --benchmarks

rewrites limitations.json from the benchmark study's own records (its
results file, section "What could not run, and why", and its verification
run), which only the working repository has; run it after a benchmark pass.
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "KNOWN_LIMITATIONS.md")


def read(rel, branch=None, label=None):
    """The text of `rel` and how to cite it: its path, or `label` for a record
    the public repository does not carry."""
    path = os.path.join(ROOT, rel)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read(), label or f"`{rel}`"
    if branch:
        try:
            text = subprocess.run(["git", "show", f"{branch}:{rel}"], cwd=ROOT, check=True,
                                  capture_output=True, text=True).stdout
            return text, label or f"`{rel}` on branch `{branch}`"
        except subprocess.CalledProcessError:
            pass
    return None, label or f"`{rel}`"


def cell(s):
    return " ".join(str(s).split()).replace("|", "\\|")


def short(s, n=220):
    s = cell(s)
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def open_bugs():
    text, src = read("bugs/ledger.jsonl")
    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    bugs = [r for r in rows if r.get("status") == "open"]
    out = [f"## Open bugs\n",
           f"{len(bugs)} of the {len(rows)} bugs in {src} are open. Each fixed bug has a test "
           "the gate runs; an open one is listed here until it does.\n",
           "| id | found | engines | bug |", "|---|---|---|---|"]
    for b in bugs:
        out.append(f"| {b['id']} | {b.get('found', '')} | {', '.join(b.get('engines', []))} | {short(b['title'], 300)} |")
    return out


def rosetta():
    text, src = read("study/rosetta/friction.jsonl", branch="rosetta", label="the Rosetta Code study's records")
    if text is None:
        return ["## Language gaps found by the Rosetta Code study\n", "The study's records were not found.\n"]
    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    rank = lambda r: (-len(r.get("tasks", [])), r["feature"].lower())
    missing = sorted((r for r in rows if r["kind"] == "missing"), key=rank)
    friction = sorted((r for r in rows if r["kind"] != "missing"), key=rank)
    out = ["## Language gaps found by the Rosetta Code study\n",
           f"From {src}: 100 Rosetta Code tasks written in Python and in Viv. "
           "A *missing* feature is one a task needed and Viv lacks; *friction* is something "
           "Viv can do only in a longer or less direct way. Ranked by the number of tasks affected.\n",
           "### Missing\n", "| feature | tasks | note |", "|---|---|---|"]
    for r in missing:
        out.append(f"| {cell(r['feature'])} | {len(r['tasks'])}: {', '.join(r['tasks'])} | {short(r.get('note', ''))} |")
    out += ["", "### Friction\n", "| feature | tasks | note |", "|---|---|---|"]
    for r in friction:
        out.append(f"| {cell(r['feature'])} | {len(r['tasks'])}: {', '.join(r['tasks'])} | {short(r.get('note', ''))} |")
    return out


def pyscf():
    text, src = read("study/ports/pyscf/FRICTION.json")
    d = json.loads(text)
    rows = [f for f in d["findings"] if f.get("status") in ("missing", "partial")]
    out = ["## Gaps found by the Hartree-Fock port (lib/hf against PySCF)\n",
           f"From {src}: the places where the port's Viv is slower or longer than it should be "
           "because the language lacks something (status *missing*) or has it in part (*partial*).\n",
           "| id | status | where | gap | cost |", "|---|---|---|---|---|"]
    for f in rows:
        where = f"`{f.get('file', '')}` {f.get('def', '')}".strip()
        out.append(f"| {f['id']} | {f['status']} | {cell(where)} | {cell(f.get('gap', ''))} | {short(f.get('cost', ''), 200)} |")
    return out


BENCH_LIMITS = "study/benchmarks/limitations.json"
# the pointer to the per-kernel refusal files, which the public repository does
# not carry; the refusals table gives the same messages
REFUSED_FILES = " The full messages are in `viv/npb/<kernel>.refused.txt`."


def refresh_benchmarks():
    """limitations.json from the benchmark study's results and verification run."""
    text, _ = read("study/benchmarks/RESULTS.md")
    vtext, _ = read("study/benchmarks/results/npb_verify.json")
    if text is None or vtext is None:
        sys.exit("--benchmarks: the benchmark study's records are not in this checkout")
    m = re.search(r"^## What could not run, and why\n(.*?)(?=^## )", text, re.S | re.M)
    if not m:
        sys.exit("--benchmarks: the section \"What could not run, and why\" was not found")
    cells = []
    for line in m.group(1).strip().splitlines():
        parts = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
        if len(parts) != 2 or parts[0] in ("cell", "") or set(parts[0]) <= set("-:"):
            continue
        cells.append({"cell": parts[0], "why": parts[1].replace(REFUSED_FILES, "")})
    v = json.loads(vtext)
    refused = [{"kernel": k, "at": r.get("refusals_in_kernel", [])} for k, r in sorted(v.items()) if r.get("refused")]
    with open(os.path.join(ROOT, BENCH_LIMITS), "w", encoding="utf-8") as f:
        json.dump({"cells": cells, "kernels": len(v), "refused": refused}, f, indent=1)
        f.write("\n")
    print(f"wrote {BENCH_LIMITS}")


def benchmarks():
    text, src = read(BENCH_LIMITS)
    if text is None:
        return ["## Benchmark cells Viv could not run\n", "The study's records were not found.\n"]
    d = json.loads(text)
    out = ["## Benchmark cells Viv could not run\n",
           f"From {src}, written from the benchmark study's last pass. Some reasons may predate "
           "later work (for example, Viv now has `Int` and a `clock()` builtin); the next "
           "benchmark pass updates them.\n",
           "| cell | why |", "|---|---|"]
    for c in d["cells"]:
        out.append(f"| {c['cell']} | {c['why']} |")
    refused = d["refused"]
    out += ["", f"numpy-benchmarks kernels py2viv refuses ({len(refused)} of {d['kernels']}):\n",
            "| kernel | refused at |", "|---|---|"]
    for r in refused:
        out.append(f"| {r['kernel']} | {short('; '.join(r['at']), 260)} |")
    return out


def platforms():
    return ["## Platforms\n",
            "- **macOS and Linux** build the compiler and the C it writes, with Clang or GCC. "
            "Only macOS on arm64 has run the exported programs' checks so far.",
            "- **Windows**: MinGW-w64 GCC and Clang should build the generated C and the exported "
            "programs, but this is untested. MSVC is unsupported: it lacks C99 `_Complex`.",
            "- **Python**: a module written by `viv export` needs Python 3.9 or newer and NumPy; "
            "it refuses to import on an older Python."]


def build():
    parts = ["# Known limitations\n",
             "Viv 0.0.1 is a preview. This file lists what does not work yet, from the records the "
             "repository keeps. It is generated by `tools/release/known_limitations.py`; edit the "
             "sources, not this file.\n"]
    for section in (platforms, open_bugs, rosetta, pyscf, benchmarks):
        parts += section() + [""]
    return "\n".join(parts).rstrip() + "\n"


if __name__ == "__main__":
    if "--benchmarks" in sys.argv:
        refresh_benchmarks()
    text = build()
    if "--check" in sys.argv:
        cur = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
        if cur != text:
            sys.exit("KNOWN_LIMITATIONS.md is stale: run python3 tools/release/known_limitations.py")
        print("KNOWN_LIMITATIONS.md is current")
    else:
        with open(OUT, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {os.path.relpath(OUT, ROOT)}")
