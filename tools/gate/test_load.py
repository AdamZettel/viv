#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The gate's handling of a check the machine's load stopped (tools/gate/load.py):

    python3 tools/gate/test_load.py

A check that exits 75 (a run of it stopped by the load) is BLOCKED-LOAD, not
FAIL; the checks after it still run; when the load falls the gate runs it
again, with the checks that needed it, and the gate passes. When the load
does not fall, the gate ends BLOCKED-LOAD (nonzero). A check that exits 1
is FAIL whatever the load. Runs with stand-in checks; writes nothing but
logs under target/gate/quick/ and a temporary history.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate  # noqa: E402
import load  # noqa: E402

tmp = tempfile.mkdtemp()
gate.HISTORY = os.path.join(tmp, "history.jsonl")
flag = os.path.join(tmp, "ran-once")
OPTS = {"list": False, "json": False, "keep_going": False, "ignore_lock": True, "only": None,
        "no_cache": True, "sample": None}


def checks(first_rc):
    # `flaky` exits first_rc on its first run and 0 after; `after` needs it
    return [gate.Check("flaky", f"if [ -e {flag} ]; then exit 0; fi; touch {flag}; exit {first_rc}", needs=[]),
            gate.Check("other", "true", needs=[]),
            gate.Check("after", "true", needs=["flaky"])]


def run(first_rc, load1, wait_minutes=0.05):
    if os.path.exists(flag):
        os.remove(flag)
    os.environ["GATE_LOAD1"] = str(load1)
    gate.CHECKS = checks(first_rc)
    real_wait = load.wait
    load.wait = lambda minutes=30: real_wait(wait_minutes)
    try:
        return gate.run_tier("quick", OPTS)
    finally:
        load.wait = real_wait


bad = 0
for what, rc_in, load1, want in [
        ("stopped by the load, the load falls: rerun, PASS", 75, 0, 0),
        ("stopped by the load, the load stays: BLOCKED-LOAD", 75, 1000, 1),
        ("a real failure: FAIL, never rerun", 1, 0, 1)]:
    print(f"\n=== {what}")
    rc = run(rc_in, load1)
    ok = rc == want
    bad |= not ok
    print(f"--> gate exit {rc}: {'ok' if ok else 'WRONG, wanted ' + str(want)}")
print("\ntest_load: " + ("ok" if not bad else "FAILED"))
sys.exit(1 if bad else 0)
