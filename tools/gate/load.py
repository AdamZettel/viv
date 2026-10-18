#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Machine load, for the gate's time limits.

    tools/gate/load.py factor          max(1, load1 / ncpu), at most CAP: what a wall-clock limit is multiplied by
    tools/gate/load.py scale <seconds> the limit times the factor, rounded up
    tools/gate/load.py busy            exit 0 when load1 > ncpu (a timeout now may be the load's, not the program's)
    tools/gate/load.py wait [minutes]  wait until load1 <= ncpu (default at most 30 minutes); exit 1 if it never is
    tools/gate/load.py show            load1, ncpu, factor, busy

A limit is either CPU time (a program's own work, which load does not
stretch) or wall time scaled by this factor. Either way an absolute cap of
CAP times the limit stays, so a real hang is still caught. A timeout taken
while the machine is busy is reported BLOCKED-LOAD and rerun when the load
falls, not FAIL.

GATE_LOAD1=<x> overrides the measured load (for tests).
"""
import math
import os
import sys
import time

CAP = 10   # the most a limit is stretched, and the absolute cap in units of the limit


def ncpu():
    return os.cpu_count() or 1


def load1():
    v = os.environ.get("GATE_LOAD1")
    if v:
        return float(v)
    try:
        return os.getloadavg()[0]
    except OSError:
        return 0.0


def factor():
    return min(CAP, max(1.0, load1() / ncpu()))


def busy():
    return load1() > ncpu()


def wait(minutes=30):
    end = time.time() + minutes * 60
    while busy():
        if time.time() > end:
            return False
        time.sleep(15)
    return True


def main(argv):
    cmd = argv[0] if argv else "show"
    if cmd == "factor":
        print(f"{factor():.2f}")
    elif cmd == "scale":
        print(math.ceil(float(argv[1]) * factor()))
    elif cmd == "busy":
        return 0 if busy() else 1
    elif cmd == "wait":
        return 0 if wait(float(argv[1]) if len(argv) > 1 else 30) else 1
    elif cmd == "show":
        print(f"load1 {load1():.1f}  ncpu {ncpu()}  factor {factor():.2f}  {'busy' if busy() else 'not busy'}")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
