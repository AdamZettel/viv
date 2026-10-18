#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The gate's scheduler: run independent checks at once, within a budget.

Each job has a weight: the cores it keeps busy, the memory it may hold at
its peak (GB), and whether it is sanitizer-heavy (ASan/UBSan builds, a few
GB a compile). The scheduler starts a job only when

  - every job it needs has finished (whatever its status: the run function
    decides what a failed dependency means),
  - the running jobs' cores plus its own fit in `cores`, and their memory
    plus its own fits in `mem_gb`,
  - it is not sanitizer-heavy while another sanitizer-heavy job runs,
  - it is not `alone` (a timing check) while anything else runs, and
    nothing starts while an `alone` job runs.

A job heavier than the whole budget still runs, by itself, so nothing waits
forever. Among the ready jobs the one with the longest critical path (its
own estimate plus the longest chain of jobs that need it) starts first;
smaller ready jobs fill what room is left.

`run(job)` is called on a worker thread and returns the job's status.
`stop()` is asked after every finish; once it says yes, the jobs not yet
started are still handed to run() (which reports them NOT RUN at once), so
every job gets exactly one result. tools/gate/test_lanes.py tests it.
"""
import threading


class Job:
    def __init__(self, name, needs=(), cores=1, mem_gb=0.5, san=False, alone=False, est=60.0):
        self.name, self.needs = name, list(needs)
        self.cores, self.mem_gb, self.san, self.alone, self.est = cores, mem_gb, san, alone, est


def critical_path(jobs):
    """name -> the job's estimate plus the longest chain of jobs that need it."""
    by = {j.name: j for j in jobs}
    users = {j.name: [] for j in jobs}
    for j in jobs:
        for n in j.needs:
            if n in users:
                users[n].append(j.name)
    memo = {}

    def cp(n, seen=()):
        if n not in memo:
            if n in seen:
                raise ValueError(f"dependency cycle through {n}")
            memo[n] = by[n].est + max((cp(u, seen + (n,)) for u in users[n]), default=0.0)
        return memo[n]
    return {j.name: cp(j.name) for j in jobs}


def run_jobs(jobs, run, cores, mem_gb, stop=lambda: False, on_start=None):
    """Run every job once; return {name: status}. Needs naming a job outside
    `jobs` are ignored (that check is not part of this run)."""
    names = {j.name for j in jobs}
    rank = critical_path(jobs)
    pending = sorted(jobs, key=lambda j: (-rank[j.name], j.name))
    running, done = {}, {}
    cv = threading.Condition()

    def fits(j):
        if not running:
            return True
        if j.alone or any(r.alone for r in running.values()):
            return False
        if j.san and any(r.san for r in running.values()):
            return False
        return (sum(r.cores for r in running.values()) + j.cores <= cores
                and sum(r.mem_gb for r in running.values()) + j.mem_gb <= mem_gb + 1e-9)

    def worker(j):
        try:
            st = run(j)
        except BaseException as e:   # a crash in run() must not hang the scheduler
            st = f"ERROR {e!r}"
        with cv:
            del running[j.name]
            done[j.name] = st
            cv.notify_all()

    threads = []
    with cv:
        while pending or running:
            stopping = stop()
            started = False
            for j in list(pending):
                if any(n in names and n not in done for n in j.needs):
                    continue
                # once stopping, the rest are handed over without weight: they do not run
                if not stopping and not fits(j):
                    continue
                pending.remove(j)
                if not stopping:
                    running[j.name] = j
                    if on_start:
                        on_start(j, dict(running))
                    t = threading.Thread(target=worker, args=(j,), daemon=True)
                    threads.append(t)
                    t.start()
                else:
                    done[j.name] = run(j)
                started = True
                if not stopping and j.alone:
                    break
            if not started:
                cv.wait()
    for t in threads:
        t.join()
    return done
