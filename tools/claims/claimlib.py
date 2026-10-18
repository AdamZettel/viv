# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Test blocks for a vipyr twin: the same PASS / FAIL lines `viv run` prints.

    from claimlib import show, test

    show("iterations", n)

    @test("Newton converges in at most 6 iterations")
    def _():
        assert n <= 6, f"{n} iterations"

A test runs where it is defined. The process exits 1 at the end when any
test failed (atexit), as a failing Viv program does.
"""
from __future__ import annotations

import atexit
import sys

_failed = []


def show(name: str, value) -> None:
    print(f"{name}: {value}", flush=True)


def test(name: str):
    def run(fn):
        try:
            fn()
        except Exception as e:  # an assertion or an error: the claim fails
            _failed.append(name)
            print(f"FAIL {name} -- {type(e).__name__}: {e}", flush=True)
        else:
            print(f"PASS {name}", flush=True)
        return fn
    return run


@atexit.register
def _exit() -> None:
    if _failed:
        print(f"{len(_failed)} test(s) FAILED", flush=True)
        sys.stdout.flush()
        import os
        os._exit(1)
