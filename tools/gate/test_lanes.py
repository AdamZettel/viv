#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The gate's lanes (tools/gate/lanes.py), fail-fast, its tiers and path triggers:

    python3 tools/gate/test_lanes.py

  - order: among ready jobs the longest chain of work starts first;
  - needs: a job starts only after every job it needs has finished;
  - budget: the running jobs never hold more cores or memory than allowed,
    two sanitizer-heavy jobs never overlap, an `alone` job runs alone, and a
    job bigger than the budget still runs (by itself);
  - fail-fast: the first FAIL stops the running checks (NOT RUN) and starts
    nothing more; --keep-going runs everything;
  - tiers: full is premerge plus postmerge, every trigger names a post-merge
    check, and a branch's paths pull the right suites into premerge.

Runs stand-in jobs and checks (sleep, true, false); writes nothing but logs
under target/gate/quick/ and a temporary history. Exits nonzero on a failure.
"""
import os
import random
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate  # noqa: E402
import lanes  # noqa: E402

bad = []


def check(what, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        bad.append(what)


class Recorder:
    """A run function that sleeps each job's `est` (scaled) and records what overlapped."""
    def __init__(self, scale=0.002):
        self.lock = threading.Lock()
        self.running, self.starts, self.ends, self.scale = {}, [], {}, scale
        self.max_cores = self.max_mem = 0.0
        self.overlap_san = self.overlap_alone = False
        self.lone_oversized = []

    def __call__(self, j):
        with self.lock:
            others = list(self.running.values())
            if j.san and any(o.san for o in others):
                self.overlap_san = True
            if others and (j.alone or any(o.alone for o in others)):
                self.overlap_alone = True
            self.running[j.name] = j
            self.starts.append(j.name)
            if len(self.running) > 1:   # one job alone may exceed the budget
                self.max_cores = max(self.max_cores, sum(o.cores for o in self.running.values()))
                self.max_mem = max(self.max_mem, sum(o.mem_gb for o in self.running.values()))
            else:
                self.lone_oversized.append(j.name)
        time.sleep(j.est * self.scale)
        with self.lock:
            del self.running[j.name]
            self.ends[j.name] = len(self.starts)
        return "PASS"


print("order")
rec = Recorder()
jobs = [lanes.Job("a", est=10), lanes.Job("b", est=100), lanes.Job("c", est=50)]
lanes.run_jobs(jobs, rec, cores=1, mem_gb=100)
check("one lane: longest first (b, c, a)", rec.starts == ["b", "c", "a"], rec.starts)

rec = Recorder()
jobs = [lanes.Job("big", est=100), lanes.Job("pre", est=1), lanes.Job("post", needs=["pre"], est=1000)]
lanes.run_jobs(jobs, rec, cores=1, mem_gb=100)
check("one lane: a short job with a long chain behind it starts first", rec.starts[0] == "pre", rec.starts)

print("needs")
order = []
olock = threading.Lock()


def logged(j):
    with olock:
        order.append(("start", j.name))
    time.sleep(0.02)
    with olock:
        order.append(("end", j.name))
    return "PASS"


jobs = [lanes.Job("build", est=5), lanes.Job("t1", needs=["build"], est=50),
        lanes.Job("t2", needs=["build", "t1"], est=1), lanes.Job("free", est=1),
        lanes.Job("x", needs=["not-in-this-run"], est=1)]
res = lanes.run_jobs(jobs, logged, cores=8, mem_gb=100)
pos = {e: i for i, e in enumerate(order)}
check("every job ran once", sorted(res) == ["build", "free", "t1", "t2", "x"], res)
check("t1 starts after build ends", pos[("start", "t1")] > pos[("end", "build")])
check("t2 starts after build and t1 end", pos[("start", "t2")] > max(pos[("end", "build")], pos[("end", "t1")]))
check("a need outside the run is ignored", "x" in res)
check("independent jobs overlap (free starts before build ends)", pos[("start", "free")] < pos[("end", "build")])

print("budget")
rng = random.Random(7)
for trial in range(5):
    jobs = []
    for i in range(30):
        jobs.append(lanes.Job(f"j{i}", needs=[f"j{rng.randrange(i)}"] if i and rng.random() < 0.3 else [],
                              cores=rng.choice([1, 2, 4]), mem_gb=rng.choice([0.5, 1, 3, 6]),
                              san=rng.random() < 0.2, alone=rng.random() < 0.05, est=rng.randrange(1, 40)))
    jobs.append(lanes.Job("huge", cores=64, mem_gb=64, est=5))
    rec = Recorder(scale=0.001)
    res = lanes.run_jobs(jobs, rec, cores=8, mem_gb=10)
    ok = (len(res) == len(jobs) and not rec.overlap_san and not rec.overlap_alone
          and "huge" in rec.lone_oversized)
    check(f"random set {trial}: all ran, no two sanitizer jobs or an alone job overlapped, the oversized ran alone",
          ok, f"ran {len(res)}, san overlap {rec.overlap_san}, alone overlap {rec.overlap_alone}")
    check(f"random set {trial}: running together, at most 8 cores and 10 GB",
          rec.max_cores <= 8 and rec.max_mem <= 10, f"max mem {rec.max_mem}, max cores {rec.max_cores}")

# with the oversized job left out, the peaks must stay inside the budget exactly
rec = Recorder(scale=0.001)
jobs = [lanes.Job(f"m{i}", mem_gb=3, cores=1, est=10) for i in range(12)]
lanes.run_jobs(jobs, rec, cores=12, mem_gb=10)
check("12 jobs of 3 GB under 10 GB: at most 3 at once", rec.max_mem == 9, rec.max_mem)

print("fail-fast (gate.run_tier with stand-in checks)")
tmp = tempfile.mkdtemp()
gate.HISTORY = os.path.join(tmp, "history.jsonl")
OPTS = {"list": False, "json": False, "keep_going": False, "ignore_lock": True, "only": None,
        "no_cache": True, "sample": None}
mark = os.path.join(tmp, "late-ran")
gate.CHECKS = [gate.Check("bad", "sleep 0.3; exit 1", needs=[], est=1),
               gate.Check("slow", "sleep 30", needs=[], est=100),
               gate.Check("late", f"touch {mark}", needs=["bad"], est=1)]
t0 = time.time()
rc = gate.run_tier("quick", OPTS)
secs = time.time() - t0
import json  # noqa: E402
row = json.loads(open(gate.HISTORY).read().splitlines()[-1])
check("the first failure fails the gate", rc == 1)
check("the running check is stopped (NOT RUN) within seconds", row["checks"].get("slow") == "NOT RUN" and secs < 10,
      f"{row['checks']}, {secs:.1f}s")
check("nothing starts after the failure", not os.path.exists(mark) and row["checks"].get("late") == "NOT RUN",
      row["checks"])
gate.CHECKS = [gate.Check("bad", "exit 1", needs=[]), gate.Check("slow", "sleep 0.5", needs=[]),
               gate.Check("late", f"touch {mark}", needs=[])]
rc = gate.run_tier("quick", dict(OPTS, keep_going=True))
row = json.loads(open(gate.HISTORY).read().splitlines()[-1])
check("--keep-going runs the rest", rc == 1 and row["checks"] == {"bad": "FAIL", "slow": "PASS", "late": "PASS"},
      row["checks"])

print("tiers and path triggers")
import importlib  # noqa: E402
g = importlib.reload(gate)   # the real check list again
names = lambda t: {c.name for c in g.CHECKS if c.runs_in(t)}   # noqa: E731
check("full = premerge + postmerge", names("full") == names("premerge") | names("postmerge"),
      names("full") ^ (names("premerge") | names("postmerge")))
check("check names are unique", len({c.name for c in g.CHECKS}) == len(g.CHECKS))
post = set(g.POSTMERGE)
check("every triggered suite is a post-merge check",
      all(s in post for _, ss in g.TRIGGERS for s in ss), [s for _, ss in g.TRIGGERS for s in ss if s not in post])
check("premerge leaves out the application suites, py2viv, memcheck, www-check, hub-golden",
      not names("premerge") & {"3e-scf", "3e-lib-hf", "py2viv", "memcheck-quick", "www-check", "hub-golden", "dft"})
check("nightly-only suites stay out of full", not names("full") & {"polybench", "3e-study-benchmarks-viv-nas", "fuzz"})
check("needs name checks that exist", all(n in {c.name for c in g.CHECKS} for c in g.CHECKS for n in c.needs))
for paths, want, nowant in [
        (["lang/viv/src/parser.rs"], set(), {"3e-scf", "memcheck-quick"}),
        (["lib/hf/rhf.viv"], {"3e-lib-hf", "3e-lib-hf-tests", "3e-scf", "scf"}, {"dft"}),
        (["lang/viv/src/lower/c_emit.rs"], {"memcheck-quick"}, {"3e-scf"}),
        (["dft/pwdft.export.toml"], {"dft", "export"}, {"py2viv"}),
        (["posts/2026-09-22-compound-interest.viv"], {"3e-posts", "www-check"}, {"py2viv"}),
        (["tools/gate/gate.py"], post, set())]:
    t = set(g.triggered(paths))
    check(f"{paths[0]} -> {', '.join(sorted(want)) or 'nothing'}", want <= t and not (nowant & t), sorted(t))
g.changed_paths = lambda base=None: ["lib/hf/rhf.viv", "README.md"]
sel, trig = g.select("premerge", {"only": None})
check("premerge on a lib/hf branch runs the hf suites, after the premerge checks",
      {"3e-scf", "scf", "smoke-hf"} <= {c.name for c in sel} and set(trig) >= {"3e-scf"}, sorted(trig))
sel, trig = g.select("premerge", {"only": None, "no_triggers": True})
check("--no-triggers: premerge alone", not trig and "3e-scf" not in {c.name for c in sel})
sel, _ = g.select("postmerge", {"only": None})
check("postmerge builds the compiler and viv.wasm first", {"build", "wasm"} <= {c.name for c in sel})

print("\ntest_lanes: " + ("ok" if not bad else f"FAILED ({len(bad)}): " + "; ".join(bad)))
sys.exit(1 if bad else 0)
