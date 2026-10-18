# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The port harness.

    python3 study/ports/harness/port.py gen     <port>
    python3 study/ports/harness/port.py verify  <port>
    python3 study/ports/harness/port.py time    <port> [--final] [--max-wait S] [--cases a,b] [--engines native,release,fast,wasm,interp,original]
    python3 study/ports/harness/port.py profile <port> [--cases a,b]
    python3 study/ports/harness/port.py report  <port>

`gen` writes the port's generated programs (the port's driver says what).
`verify` runs study/ports/<port>/check.sh and, when it passes, records the
hash of the sources it checked. `time` refuses to run unless `verify` passed
for the current sources (the port's viv/, upstream/, oracle/ and the Viv
libraries its driver names). `time` takes study/ports/.timing.lock so that
two timers never run at once, and refuses while a benchmark bench.py or
another port.py time runs.

Load: a side thread reads the 1-minute load average every second during
each timed process and keeps the maximum. A run whose maximum reached the
port's load limit (2.0) is provisional: it is written to
results/provisional.json and marked `*` in the report. With --final the
harness first waits (polling every 5 minutes, at most --max-wait seconds)
for the load to fall below the limit, and writes final.json only from runs
that stayed below it throughout.

The port's own work is in study/ports/<port>/driver.py, which defines
gen(ctx), time(ctx), profile(ctx) and report(ctx). This file holds what
every port shares: the lock, the load monitor, the verification gate, and
building a Viv program with stage clocks (build_viv_timed).

The load average, process timing, the single-thread environment and
writing results are portkit.py's, beside this file.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PORTS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PORTS))
sys.path.insert(0, HERE)
import portkit  # noqa: E402

LOCK = os.path.join(PORTS, ".timing.lock")
VIV = os.environ.get("VIV", os.path.join(REPO, "target", "release", "viv"))


# ---- the lock ------------------------------------------------------------------------

class TimingLock:
    """study/ports/.timing.lock: one timer at a time. The file holds the pid
    and the port; a lock whose pid is gone is taken over."""
    def __init__(self, port):
        self.port = port

    def __enter__(self):
        others = other_timers()
        if others:
            raise SystemExit("another timer is running:\n  " + "\n  ".join(others))
        if os.path.exists(LOCK):
            try:
                pid = int(open(LOCK).read().split()[0])
                os.kill(pid, 0)
                raise SystemExit(f"{LOCK} is held by pid {pid}")
            except (ProcessLookupError, ValueError):
                pass
        with open(LOCK, "w") as f:
            f.write(f"{os.getpid()} {self.port} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        return self

    def __exit__(self, *exc):
        try:
            if open(LOCK).read().split()[0] == str(os.getpid()):
                os.remove(LOCK)
        except OSError:
            pass


def other_timers():
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    me = os.getpid()
    found = []
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if int(pid) == me:
            continue
        argv = cmd.split()
        # a python process whose script is bench.py, or port.py with `time`
        # (a shell that merely mentions them does not count)
        if not argv or not os.path.basename(argv[0]).startswith("python"):
            continue
        script = next((a for a in argv[1:] if not a.startswith("-")), "")
        if script.endswith("bench.py") and "benchmarks" in script:
            found.append(line.strip())
        elif script.endswith("port.py") and "time" in argv[argv.index(script) + 1:argv.index(script) + 2]:
            found.append(line.strip())
    return found


# ---- the load monitor ------------------------------------------------------------------

class LoadMonitor:
    """Reads the 1-minute load average every `poll` seconds while active."""
    def __init__(self, poll=1.0):
        self.poll = poll
        self.max = 0.0
        self.start = None
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        self.start = portkit.loadavg()
        self.max = self.start
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.wait(self.poll):
            self.max = max(self.max, portkit.loadavg())

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.max = max(self.max, portkit.loadavg())


def wait_quiet(limit, max_wait, poll=300, log=print):
    """Wait for the 1-minute load average to fall below `limit`, polling every
    `poll` seconds, at most `max_wait` seconds. Returns (quiet, load)."""
    t0 = time.time()
    while True:
        la = portkit.loadavg()
        if la < limit:
            return True, la
        if time.time() - t0 >= max_wait:
            return False, la
        log(f"[{time.strftime('%H:%M:%S')}] load {la:.2f} >= {limit}; next check in {poll} s")
        time.sleep(poll)


# ---- the verification gate ---------------------------------------------------------------

def sources_hash(port_dir, extra_dirs):
    h = hashlib.sha256()
    dirs = [os.path.join(port_dir, d) for d in ("viv", "upstream", "oracle")] + [os.path.join(REPO, d) for d in extra_dirs]
    for d in dirs:
        if os.path.isfile(d):
            h.update(os.path.relpath(d, REPO).encode())
            h.update(open(d, "rb").read())
            continue
        for root, _, files in sorted(os.walk(d)):
            for fn in sorted(files):
                p = os.path.join(root, fn)
                h.update(os.path.relpath(p, REPO).encode())
                h.update(open(p, "rb").read())
    h.update(open(VIV, "rb").read())
    return h.hexdigest()


# ---- Viv builds with stage clocks ------------------------------------------------------

# Inserted into the generated C ahead of viv_step: the time between a step's
# enter and exit events, less the time of the steps inside it, summed per
# `def/step`, written to stderr at exit as one line
#     __stagetime__ {"def/step": [exclusive seconds, count, inclusive seconds], ...}
# viv_step is called at every step boundary in both the default and the
# release build (its body is compiled out in release, the call is not), so
# the same clocks run in both. Nothing else in the program changes.
STAGE_CLOCK_C = r'''
#include <time.h>
#define PORT_ST_MAX 256
#define PORT_ST_DEPTH 64
static struct { const char* def; const char* id; double t; double ti; long n; } port_st[PORT_ST_MAX];
static int port_st_used = 0;
static double port_st_t0[PORT_ST_DEPTH], port_st_child[PORT_ST_DEPTH];
static int port_st_sp = 0;
static double port_st_now(void) { return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9; }
static double port_st_begin = 0.0;
__attribute__((constructor)) static void port_st_init(void) { port_st_begin = port_st_now(); }
static void port_st_add(const char* def, const char* id, double t, double ti) {
    for (int i = 0; i < port_st_used; i++)
        if ((port_st[i].def == def || strcmp(port_st[i].def, def) == 0) && (port_st[i].id == id || strcmp(port_st[i].id, id) == 0)) {
            port_st[i].t += t; port_st[i].ti += ti; port_st[i].n++; return;
        }
    if (port_st_used < PORT_ST_MAX) {
        port_st[port_st_used].def = def; port_st[port_st_used].id = id;
        port_st[port_st_used].t = t; port_st[port_st_used].ti = ti; port_st[port_st_used].n = 1; port_st_used++;
    }
}
static void port_step_clock(int enter, const char* def, const char* id) {
    double now = port_st_now();
    if (enter) {
        if (port_st_sp < PORT_ST_DEPTH) { port_st_t0[port_st_sp] = now; port_st_child[port_st_sp] = 0.0; }
        port_st_sp++;
        return;
    }
    port_st_sp--;
    if (port_st_sp < 0 || port_st_sp >= PORT_ST_DEPTH) { port_st_sp = port_st_sp < 0 ? 0 : port_st_sp; return; }
    double dt = now - port_st_t0[port_st_sp];
    port_st_add(def, id, dt - port_st_child[port_st_sp], dt);
    if (port_st_sp > 0) port_st_child[port_st_sp - 1] += dt;
}
__attribute__((destructor)) static void port_st_dump(void) {
    double p = port_st_now() - port_st_begin;
    fprintf(stderr, "__stagetime__ {\"__process__\": [%.9f, 1, %.9f]", p, p);
    for (int i = 0; i < port_st_used; i++)
        fprintf(stderr, ", \"%s/%s\": [%.9f, %ld, %.9f]", port_st[i].def, port_st[i].id, port_st[i].t, port_st[i].n, port_st[i].ti);
    fprintf(stderr, "}\n");
}
'''

# the step recorder's definition: one copy, in the program's unit of a split
# build (lang/viv/src/lower.rs SHARED_FUNCS); `static` in a one-file build
STEP_SIG = "void viv_step(int enter, const char* def, const char* id, int depth) {"


def patch_stage_clock(c_path):
    src = open(c_path).read()
    at = src.find("\n" + STEP_SIG)
    if at < 0:
        at = src.find("\nstatic " + STEP_SIG)
    if at < 0:
        raise RuntimeError(f"{c_path}: no viv_step definition to hook (the C backend changed)")
    head = src.index(STEP_SIG, at) + len(STEP_SIG)
    src = src[:at + 1] + STAGE_CLOCK_C + src[at + 1:head] + "\n    port_step_clock(enter, def, id);" + src[head:]
    open(c_path, "w").write(src)


def build_viv_timed(src, outdir, wasm=True):
    """viv -> C as units (+ stage clocks in the program's unit), then the
    exact, release and fast builds by `viv build --from-split`, each module's
    object from viv's object cache when it holds it; viv -> wasm. Returns paths and the
    compile times (the cold column adds them)."""
    os.makedirs(outdir, exist_ok=True)
    name = os.path.splitext(os.path.basename(src))[0]
    sdir = os.path.join(outdir, name + ".split")
    shutil.rmtree(sdir, ignore_errors=True)
    info = {"src": src, "name": name}
    t0 = time.perf_counter()
    r = subprocess.run([VIV, src, "--split", sdir], cwd=REPO, capture_output=True, text=True)
    info["viv_to_c_s"] = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"viv -> C failed for {src}:\n{r.stderr}")
    c = os.path.join(sdir, "main.c")
    patch_stage_clock(c)
    exe = os.path.join(outdir, name)
    builds = [("cc_s", exe, []), ("cc_release_s", exe + "_release", ["--release"]), ("cc_fast_s", exe + "_fast", ["--fast", "--release"])]
    for key, out, mode in builds:
        t0 = time.perf_counter()
        r = subprocess.run([VIV, "build", "--from-split", sdir] + mode + ["-o", out], cwd=REPO, capture_output=True, text=True)
        info[key] = time.perf_counter() - t0
        if r.returncode != 0:
            raise RuntimeError(f"viv build {' '.join(mode) or '(exact)'} failed for {src}:\n{(r.stdout + r.stderr)[:2000]}")
        info[key.replace("_s", "_units")] = r.stdout.strip().split("[", 1)[-1].rstrip("]")
    exe_rel = exe + "_release"
    info["exe_fast"] = exe + "_fast"
    info.update({"c": c, "exe": exe, "exe_release": exe_rel, "wasm": None})
    if wasm:
        w = os.path.join(outdir, name + ".wasm")
        t0 = time.perf_counter()
        r = subprocess.run([VIV, src, "-o", w], cwd=REPO, capture_output=True, text=True)
        info["viv_to_wasm_s"] = time.perf_counter() - t0
        if r.returncode == 0:
            info["wasm"] = w
        else:
            info["wasm_error"] = r.stderr.strip()[:500]
    return info


def parse_stagetime(stderr):
    for line in stderr.splitlines():
        if line.startswith("__stagetime__ "):
            return json.loads(line[len("__stagetime__ "):])
    return None


def run_timed(cmd, cwd, timeout=None, stdout_path=None):
    """One process under the load monitor: wall, cpu, rc, stderr, load max."""
    with LoadMonitor() as lm:
        wall, rc, err, cpu = portkit.run_proc(cmd, cwd=cwd, timeout=timeout, stdout_path=stdout_path)
    return {"wall": wall, "cpu": cpu, "rc": rc, "stderr": err, "load_start": lm.start, "load_max": lm.max}


def run_timed_capture(cmd, cwd, timeout=None):
    """run_timed, keeping stdout: the dict gains "stdout"."""
    import tempfile
    fd, path = tempfile.mkstemp(prefix="port_out")
    os.close(fd)
    try:
        r = run_timed(cmd, cwd, timeout=timeout, stdout_path=path)
        r["stdout"] = open(path, errors="replace").read()
    finally:
        os.remove(path)
    return r


# ---- the command line --------------------------------------------------------------------

class Ctx:
    def __init__(self, port, args):
        self.port = port
        self.dir = os.path.join(PORTS, port)
        self.args = args
        self.repo = REPO
        self.viv = VIV
        self.results = os.path.join(self.dir, "results")
        self.build = os.path.join(self.dir, "results", "build")


def load_driver(port_dir):
    spec = importlib.util.spec_from_file_location("driver", os.path.join(port_dir, "driver.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "verify", "time", "profile", "report"])
    ap.add_argument("port")
    ap.add_argument("--final", action="store_true", help="wait for a quiet machine and keep only quiet runs as final")
    ap.add_argument("--max-wait", type=float, default=3 * 3600)
    ap.add_argument("--cases", default="")
    ap.add_argument("--engines", default="")
    args = ap.parse_args()
    ctx = Ctx(args.port, args)
    drv = load_driver(ctx.dir)
    verified = os.path.join(ctx.results, ".verified")
    if args.cmd == "gen":
        drv.gen(ctx)
    elif args.cmd == "verify":
        r = subprocess.run([os.path.join(ctx.dir, "check.sh")], cwd=REPO)
        if r.returncode != 0:
            raise SystemExit("check.sh failed: no timing counts until it passes")
        os.makedirs(ctx.results, exist_ok=True)
        with open(verified, "w") as f:
            f.write(sources_hash(ctx.dir, drv.SOURCE_DIRS) + "\n")
        print("verified; sources hash recorded in results/.verified")
    elif args.cmd == "time":
        want = sources_hash(ctx.dir, drv.SOURCE_DIRS)
        have = open(verified).read().strip() if os.path.exists(verified) else ""
        if want != have:
            raise SystemExit("refused: `port.py verify` has not passed for the current sources")
        with TimingLock(args.port):
            drv.time(ctx)
    elif args.cmd == "profile":
        drv.profile(ctx)
    elif args.cmd == "report":
        drv.report(ctx)


if __name__ == "__main__":
    main()
