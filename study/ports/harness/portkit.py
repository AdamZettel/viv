# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""What the port drivers share besides the lock and the gate (port.py): the
single-thread environment, the load average, process timing, the versions
of the tools, and writing results with no local path in them.

Standard library only."""

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RUN_WASM = os.path.join(REPO, "tools", "run-wasm.mjs")
JULIA = os.environ.get("JULIA", shutil.which("julia") or "julia")
NODE = shutil.which("node") or "node"
CC = os.environ.get("CC", "cc")

# One thread for every runtime and library, so a timing is one core's.
THREAD_ENV = {
    "JULIA_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}


def env():
    e = dict(os.environ)
    e.update(THREAD_ENV)
    return e


def loadavg():
    """The 1-minute load average (macOS sysctl, else os.getloadavg)."""
    try:
        out = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout
        return float(out.strip().strip("{}").split()[0])
    except Exception:
        return os.getloadavg()[0]


def run_proc(cmd, cwd, stdin_path=None, stdout_path=None, timeout=None, env_extra=None):
    """Run one process. Returns (wall seconds, returncode, stderr text, cpu
    seconds): cpu is the child's user + system time from wait4's rusage, so
    cpu / wall well above 1 means the process used more than one thread.
    stdout goes to stdout_path (or /dev/null); a timeout gives returncode -999."""
    e = env()
    if env_extra:
        e.update(env_extra)
    fin = open(stdin_path, "rb") if stdin_path else subprocess.DEVNULL
    fout = open(stdout_path or os.devnull, "wb")
    ferr = tempfile.TemporaryFile()
    try:
        t0 = time.perf_counter()
        p = subprocess.Popen(cmd, cwd=cwd, stdin=fin, stdout=fout, stderr=ferr, env=e)
        deadline = None if timeout is None else t0 + timeout
        while True:
            pid, status, ru = os.wait4(p.pid, os.WNOHANG)
            if pid:
                break
            if deadline and time.perf_counter() > deadline:
                p.kill()
                pid, status, ru = os.wait4(p.pid, 0)
                break
            time.sleep(0.0005)
        t1 = time.perf_counter()
        p.returncode = os.waitstatus_to_exitcode(status)
        if deadline and t1 > deadline:
            p.returncode = -999
        ferr.seek(0)
        err = ferr.read()
    finally:
        if stdin_path:
            fin.close()
        fout.close()
        ferr.close()
    return t1 - t0, p.returncode, err.decode("utf-8", "replace"), ru.ru_utime + ru.ru_stime


def versions():
    def out(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, env=env()).stdout.strip()
        except Exception as ex:
            return f"unavailable: {ex}"
    import numpy
    try:
        import numba
        nb = numba.__version__
    except Exception:
        nb = "unavailable"
    return {
        "machine": out(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "ncpu": out(["sysctl", "-n", "hw.ncpu"]),
        "memsize": out(["sysctl", "-n", "hw.memsize"]),
        "os": platform.platform(),
        "julia": out([JULIA, "--version"]),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "numba": nb,
        "cc": (out([CC, "--version"]).splitlines() or [""])[0],
        "node": out([NODE, "--version"]),
        "viv_commit": out(["git", "-C", REPO, "rev-parse", "HEAD"]),
    }


def save_json(path, obj):
    """Write JSON with the repository's absolute path cut to `<repo>` and the
    home directory to `~`, so no local path is committed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = json.dumps(obj, indent=1, sort_keys=True)
    text = text.replace(REPO, "<repo>").replace(os.path.expanduser("~"), "~")
    with open(path, "w") as f:
        f.write(text)
