#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The gate's cache: a check, or one program of a check, whose inputs have
not changed since it passed is not run again.

    tools/gate/cache.py three-engines <dir> [names...]   tools/three-engines.sh, per program
    tools/gate/cache.py fast-identity <dir> [names...]   tools/fast-check.sh with IDENTITY=1 NO_INTERP=1
    tools/gate/cache.py items <kind> --outdir D [--jobs N] <dir:name...>
                                    run each item, as scf/check.sh's xargs did (kind both-engines)
    tools/gate/cache.py memcheck < programs               tools/memcheck/ci.sh's step 1, a row per program
    tools/gate/cache.py golden <snapshot-dir> < programs one hub-golden snapshot per program
    tools/gate/cache.py explain <kind> <program>        why a program's entry is or is not reused
    tools/gate/cache.py stats | prune [--days N] | clear

These run under the gate (tools/gate/gate.py), which sets GATE_CACHE=1 for
quick and full and GATE_CACHE=0 for nightly and --no-cache. Without
GATE_CACHE=1 every item runs (in parallel, JOBS at once), as before.

The key. A cached verdict is valid while everything that can change it is
unchanged:

  - the environment: the compiler (the contents of every source file cargo's
    dep-info lists for target/release/viv, so two worktrees with the same
    sources share entries; the binary's own hash when there is no fresh
    dep-info), viv.wasm, the C compiler's version, node's, Python's and
    NumPy's, the OS, every VIV_* variable and CC/CFLAGS/LDFLAGS;
  - the check: its kind, its scripts' contents, the item and the options;
  - the files the run read, as the compiler and the interpreter report them
    (VIV_TRACE_INPUTS, lang/viv/src/trace.rs): the program, every module its
    imports resolved to and every place an import looked and found nothing,
    the lens files (.memory, .debug), every file read_text read, and for a
    Python program all of tools/py2viv.

The entry stores the files and their hashes (git blob ids); a lookup rehashes
them. A program whose text names read_text, read_lines or program_graph is
cached only by a check whose runs include the interpreter, which traces what
it reads; files under the temporary directory are the run's own. A failed
or untraceable item is never stored. A cached PASS says so in the report:

    rhf    9 tests pass, 0 failed; identical output    PASS (cached 3f9a0c21d4, from e09b6b0 2026-10-06)

The sample. GATE_CACHE_SAMPLE (the gate: 0.05) is the share of cache hits
that run anyway. A sampled item that fails, or whose output differs from the
cached output, is a wrong key: the check fails with CACHE MISMATCH, the
entry is removed, and the mismatch is recorded in bugs/ledger.jsonl and in
target/gate/cache/mismatches.jsonl of the main repository.

The store is target/gate/cache/<kind>/ of the main repository, shared by
every worktree. An entry is written to a temporary name and renamed, so
concurrent readers see the old entry or the new one, never half of one.
"""
import datetime as dt
import functools
import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

FORMAT = 1
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("GATE_ROOT") or os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.realpath(ROOT)


def _git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd or ROOT, capture_output=True, text=True).stdout


COMMON = os.environ.get("GATE_COMMON") or os.path.dirname(os.path.abspath(
    os.path.join(ROOT, _git("rev-parse", "--git-common-dir").strip())))
STORE = os.environ.get("GATE_CACHE_DIR") or os.path.join(COMMON, "target", "gate", "cache")
VIV = os.environ.get("VIV") or os.path.join(ROOT, "target", "release", "viv")
VIV_WASM = os.environ.get("VIV_WASM") or os.path.join(ROOT, "www", "public", "viv", "viv.wasm")
TMPDIRS = tuple(sorted({os.path.realpath(d) + os.sep for d in
                        (tempfile.gettempdir(), "/tmp", "/private/tmp", "/var/folders", "/private/var/folders",
                         os.environ.get("TMPDIR") or "/tmp")}))


def enabled():
    return os.environ.get("GATE_CACHE") == "1"


def sample_rate():
    try:
        return float(os.environ.get("GATE_CACHE_SAMPLE", "0"))
    except ValueError:
        return 0.0


def jobs():
    try:
        return max(1, int(os.environ.get("JOBS", "4")))
    except ValueError:
        return 4


def h(*parts):
    m = hashlib.sha256()
    for p in parts:
        m.update(p if isinstance(p, bytes) else str(p).encode())
        m.update(b"\0")
    return m.hexdigest()


# ------------------------------------------------------------------ hashing

_lock = threading.Lock()
_shas = {}


def blob_sha(path):
    """The git blob id of a file's contents, or None when there is no file."""
    with _lock:
        if path in _shas:
            return _shas[path]
    try:
        with open(path, "rb") as f:
            data = f.read()
        s = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        s = None
    with _lock:
        _shas[path] = s
    return s


def rel_or_abs(p):
    """A repository path relative to ROOT, so that worktrees share entries."""
    rp = os.path.realpath(p)
    if rp == ROOT or rp.startswith(ROOT + os.sep):
        return os.path.relpath(rp, ROOT)
    return rp


def resolve(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def tree_files(paths=None):
    """{path: blob id} of every tracked or untracked (not ignored) file under
    `paths` (repository-relative; all of the tree when None), as the working
    tree has them."""
    spec = ["--", *paths] if paths else []
    files = _git("ls-files", "-co", "--exclude-standard", "-z", *spec).split("\0")
    staged = {}
    for line in _git("ls-files", "-s", "-z", *spec).split("\0"):
        if line:
            meta, path = line.split("\t", 1)
            staged[path] = meta.split()[1]
    changed = set(_git("diff", "--name-only", "-z", *spec).split("\0"))
    out = {}
    for f in files:
        if not f:
            continue
        if f in staged and f not in changed:
            out[f] = staged[f]
        else:
            out[f] = blob_sha(os.path.join(ROOT, f))
    return out


@functools.lru_cache(maxsize=None)
def dir_hash(d):
    """One hash of every file under the directory `d` (tracked or untracked, not ignored)."""
    r = rel_or_abs(d)
    if os.path.isabs(r):
        return None   # outside the repository: not determinable
    fs = tree_files([r])
    return h(*sorted(f"{k} {v}" for k, v in fs.items()))


def tree_hash(paths=None):
    fs = tree_files(paths)
    return h(*sorted(f"{k} {v}" for k, v in fs.items()))


def forget():
    """Drop the memoized hashes: files may have changed since."""
    with _lock:
        _shas.clear()
    dir_hash.cache_clear()


# -------------------------------------------------------------- environment

def _out(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (p.stdout + p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired):
        return "(none)"


@functools.lru_cache(maxsize=None)
def compiler_id():
    """The compiler: the sources cargo built target/release/viv from (cargo's
    dep-info, written with the binary), so that two worktrees with the same
    sources agree; the binary's bytes when there is no dep-info.

    No file times: a file whose time moved but whose bytes did not (a
    checkout, a test rewriting an expected output with the same text) does
    not change the compiler. The gate keys a run on this after its build
    step, so the dep-info describes the binary the run uses."""
    viv = os.path.realpath(VIV)
    dep = viv + ".d"
    if not os.path.isfile(viv):
        return None
    if os.path.isfile(dep) and os.path.dirname(viv) == os.path.realpath(os.path.join(ROOT, "target", "release")):
        text = open(dep).read().replace("\\\n", " ")
        _, _, deps = text.partition(": ")
        files = re.split(r"(?<!\\) ", deps.split("\n")[0].strip())
        files = [f.replace("\\ ", " ") for f in files if f]
        if files and all(os.path.exists(f) for f in files):
            manifests = sorted(f for f in tree_files().keys() if f == "Cargo.lock" or f.endswith("Cargo.toml"))
            parts = [f"{rel_or_abs(f)} {dir_hash(f) if os.path.isdir(f) else blob_sha(f)}" for f in sorted(files)]
            parts += [f"{m} {blob_sha(os.path.join(ROOT, m))}" for m in manifests]
            parts += [_out(["rustc", "-vV"]), os.environ.get("RUSTFLAGS", ""), os.environ.get("CARGO_PROFILE_RELEASE_OPT_LEVEL", "")]
            return "sources " + h(*parts)
    return "binary " + blob_sha(viv)


@functools.lru_cache(maxsize=None)
def env_part(what):
    # the gate works these out once, after its build, and hands them to every check
    # it runs (GATE_ENV), so one gate run keys everything on the same compiler
    fixed = os.environ.get("GATE_ENV")
    if fixed:
        try:
            v = json.loads(fixed)
            if what in v:
                return v[what]
        except ValueError:
            pass
    if what == "viv":
        return compiler_id()
    if what == "wasm":
        s = blob_sha(VIV_WASM)
        return s and "viv.wasm " + s
    if what == "cc":
        return " ".join([_out([os.environ.get("CC", "cc"), "--version"]).splitlines()[0]]
                        + [f"{k}={os.environ.get(k, '')}" for k in ("CC", "CFLAGS", "LDFLAGS")])
    if what == "node":
        return "node " + _out(["node", "--version"])
    if what == "python":
        try:
            import numpy
            npv = numpy.__version__
        except Exception:
            npv = "none"
        return f"python {sys.version.split()[0]} numpy {npv}"
    if what == "rustc":
        return _out(["rustc", "-vV"]) + " " + os.environ.get("RUSTFLAGS", "")
    if what == "os":
        return platform.platform() + " " + platform.machine()
    if what == "vivenv":
        skip = {"VIV", "VIV_WASM", "VIV_TRACE_INPUTS"}
        return " ".join(f"{k}={v}" for k, v in sorted(os.environ.items())
                        if (k.startswith("VIV_") or k.startswith("KNOT_")) and k not in skip)
    raise KeyError(what)


def env_parts(uses):
    return {u: env_part(u) for u in sorted(set(uses) | {"os", "vivenv"})}


def env_key(uses):
    parts = env_parts(uses)
    for u, v in parts.items():
        if v is None:
            return None, f"no {u}"
    return h(*parts.values()), None


# -------------------------------------------------------------------- kinds

TE_SCRIPTS = ["tools/three-engines.sh", "tools/run-wasm.mjs", "www/src/lib/vivDirect.js", "www/src/lib/uses.js"]


class Kind:
    """compare_output: a sampled rerun must print what the cached run printed
    (False for output that carries timings, such as cargo's test reports)."""
    def __init__(self, name, scripts, uses, io_ok, extras=(), compare_output=True):
        self.name, self.scripts, self.uses, self.io_ok, self.extras = name, scripts, uses, io_ok, extras
        self.compare_output = compare_output


KINDS = {
    # three engines: the interpreter runs every program, so what it reads is traced
    "three-engines": Kind("three-engines", TE_SCRIPTS, ["viv", "wasm", "cc", "node"], True),
    "both-engines": Kind("both-engines", ["tools/both-engines.sh"], ["viv", "cc"], True),
    # the fast build's C only: a program that reads files at run time is not cached
    "fast-identity": Kind("fast-identity", ["tools/fast-check.sh"], ["viv", "cc", "python"], False),
    # a Rust test binary, traced by fstrace.c (every file it and its children open)
    "cargo-test": Kind("cargo-test", ["tools/gate/cargo_tests.py", "tools/gate/fstrace.c"],
                       ["rustc", "cc", "node", "python", "wasm"], True, compare_output=False),
    # a part of the site's stage or check, traced by fstrace.c; what it wrote is restored on a hit
    "www": Kind("www", ["tools/gate/www_check.py", "tools/gate/fstrace.c"],
                ["viv", "wasm", "node", "python", "cc"], True, compare_output=False),
    "memcheck": Kind("memcheck", ["tools/memcheck/ci.sh", "tools/memcheck/ubsan_sites.py", "tools/memcheck/ci_slow.txt"],
                     ["viv", "cc", "python"], False, extras=("LIMIT", "LSAN")),
    "golden": Kind("golden", ["tools/hub-golden.sh", "tools/hub-golden.slow", "tools/run-wasm.mjs",
                              "www/src/lib/vivDirect.js", "www/src/lib/uses.js"],
                   ["viv", "wasm", "cc", "node"], True, extras=("RUN_TIMEOUT", "NO_RUN", "NO_C", "FULL")),
}


def kind_parts(kind):
    """Everything a kind's key is made of, by name (what `explain` compares)."""
    parts = {"env " + u: v for u, v in env_parts(kind.uses).items()}
    parts.update({"script " + s: blob_sha(resolve(s)) for s in kind.scripts})
    parts.update({"option " + x: os.environ.get(x, "") for x in kind.extras})
    return parts


def kind_key(kind):
    e, why = env_key(kind.uses)
    if e is None:
        return None, why
    scripts = [f"{s} {blob_sha(resolve(s))}" for s in kind.scripts]
    extras = [f"{x}={os.environ.get(x, '')}" for x in kind.extras]
    return h(FORMAT, kind.name, e, *scripts, *extras), None


# -------------------------------------------------------------------- store

def entry_path(kind, ident):
    return os.path.join(STORE, kind, ident[:2], ident + ".json")


def atomic_write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp-")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=0)
    os.replace(tmp, path)


def inputs_now(inputs):
    """The inputs' current hashes, in the stored form."""
    out = []
    for kind, p, _ in inputs:
        if kind == "f":
            out.append(["f", p, blob_sha(resolve(p))])
        elif kind == "l":   # a directory's entries, by name
            out.append(["l", p, listing(resolve(p))])
        elif kind == "t":   # the files under these paths (all of the tree when none), tracked or not
            out.append(["t", p, tree_hash(p.split(",") if p else None)])
        else:
            out.append(["d", p, dir_hash(resolve(p))])
    return out


def lookup(kind, ident, ignore=frozenset()):
    """The entry for `ident` when its inputs are unchanged, else None. Inputs
    named in `ignore` are not compared: files a later step of the same
    pipeline writes again (www_check.py), whose state now is that step's
    last product, not this step's input."""
    path = entry_path(kind, ident)
    try:
        e = json.load(open(path))
    except (OSError, ValueError):
        return None
    if e.get("format") != FORMAT:
        return None
    kept = [i for i in e["inputs"] if i[1] not in ignore]
    if inputs_now(kept) != kept:
        return None
    if e.get("artifacts") and not os.path.isdir(os.path.join(os.path.dirname(path), e["artifacts"])):
        return None
    try:
        os.utime(path)   # last used: `prune` goes by it
    except OSError:
        pass
    return e


# what the operating system and the toolchains supply: keyed by their versions (env_part), not file by file
SYSTEM = ("/System/", "/usr/", "/Library/", "/dev/", "/private/var/db/", "/private/etc/", "/etc/",
          "/Applications/", "/opt/homebrew/Cellar/", "/opt/homebrew/lib/", os.path.expanduser("~/.rustup/"),
          os.path.expanduser("~/.cargo/registry/"))
# programs macOS protects: dyld drops DYLD_* for them and for all they start (fstrace.c)
PROTECTED = ("/bin/", "/sbin/", "/usr/bin/", "/usr/sbin/", "/usr/libexec/", "/System/")
# protected programs that run other programs or read what their arguments do not name
OPAQUE = {"sh", "bash", "zsh", "dash", "ksh", "csh", "tcsh", "env", "xargs", "perl", "python", "python3",
          "ruby", "make", "find", "osascript", "nice", "nohup", "time", "xcrun", "open", "git"}


BUILD = os.path.join(ROOT, "target") + os.sep


def built_from(prog):
    """The files a program stands for: a binary cargo built, the sources its
    dep-info lists (so worktrees agree); any other program, itself."""
    rp = os.path.realpath(prog)
    d = rp + ".d"
    if rp.startswith(BUILD) and os.path.isfile(d):
        return dep_files(d)
    return [rp]


def dep_files(d):
    """The files a cargo dep-info file lists."""
    text = open(d).read().replace("\\\n", " ")
    first = text.split("\n")[0].partition(": ")[2]
    out = []
    for f in re.split(r"(?<!\\) ", first.strip()):
        f = f.replace("\\ ", " ")
        if not f:
            continue
        if not os.path.isabs(f):   # cargo writes a workspace member's paths from the workspace root
            f = os.path.join(ROOT, f)
        if os.path.isdir(f):
            out += [os.path.join(dp, x) for dp, _, xs in os.walk(f) for x in xs]
        else:
            out.append(f)
    return out


def fstrace():
    """tools/gate/fstrace.c built (target/gate/fstrace.dylib of this tree), for DYLD_INSERT_LIBRARIES."""
    src = os.path.join(HERE, "fstrace.c")
    lib = os.path.join(ROOT, "target", "gate", "fstrace.dylib")
    if not os.path.exists(lib) or os.path.getmtime(lib) < os.path.getmtime(src):
        os.makedirs(os.path.dirname(lib), exist_ok=True)
        # to a new file, then renamed: a running process keeps the library it mapped
        tmp = f"{lib}.{os.getpid()}"
        subprocess.run(["cc", "-dynamiclib", "-O2", "-o", tmp, src], check=True)
        os.replace(tmp, lib)
    return lib


def parse_trace(text, item_files, io_ok):
    """The inputs a trace lists, or (None, why) when they cannot be known."""
    files, dirs, lists = {}, set(), set()
    for f in item_files:
        files[rel_or_abs(f)] = True

    def add(path):
        p = os.path.realpath(path) if os.path.exists(path) else path
        if p.startswith(TMPDIRS) or p.startswith(SYSTEM) or p.startswith(BUILD):
            return   # the run's own temporary files; the system's; cargo's build state (its products
                     # count by their sources, built_from)
        files[rel_or_abs(p)] = True

    for line in text.splitlines():
        parts = line.split("\t")
        if parts[0] == "f" and len(parts) == 3:
            add(parts[1])
        elif parts[0] == "u" and len(parts) == 2:
            return None, f"it ran {parts[1]}, a sanitizer build, which the trace cannot follow"
        elif parts[0] == "w" and len(parts) == 3:
            pass   # an output; it is an input too (its f line), as it stands after the run
        elif parts[0] == "l" and len(parts) == 2:
            p = os.path.realpath(parts[1])
            if not p.startswith(TMPDIRS) and not p.startswith(SYSTEM):
                lists.add(rel_or_abs(p))
        elif parts[0] == "x" and len(parts) >= 2:
            prog = parts[1]
            if prog.startswith(PROTECTED):
                base = os.path.basename(prog)
                args = parts[2:]
                if base == "git" and len(args) >= 2 and args[0] == "-C" and \
                        os.path.realpath(args[1]).startswith(TMPDIRS):
                    # git in a run's own temporary directory (outside any repository): it reads
                    # that directory and the user's git configuration
                    add(os.path.expanduser("~/.gitconfig"))
                    continue
                if base in OPAQUE or base.startswith("python"):
                    return None, f"it ran {prog}, which the trace cannot follow ({' '.join(parts[2:])[:80]})"
                for a in parts[2:]:   # cc, cmp, cat...: what they read is what they are given
                    if a.startswith("/") and os.path.isfile(a):
                        add(a)
            elif os.path.isfile(prog):
                for f in built_from(prog):   # the program itself: a repository build by its sources
                    add(f)
        elif parts[0] == "d" and len(parts) == 2:
            r = rel_or_abs(parts[1])
            if os.path.isabs(r):
                return None, f"reads all of {parts[1]}, outside the repository"
            dirs.add(r)
        elif parts[0] == "io" and len(parts) == 2:
            if not io_ok and not parts[1].startswith(TMPDIRS):
                return None, f"{rel_or_abs(parts[1])} reads files at run time, and this check does not trace them"
        elif line.strip():
            return None, f"unreadable trace line {line[:80]!r}"
    inputs = ([["f", p, None] for p in sorted(files)] + [["d", d, None] for d in sorted(dirs)]
              + [["l", d, None] for d in sorted(lists)])
    try:
        return inputs_now(inputs), None
    except OSError as e:
        # a path the run touched that cannot be read back (macOS refuses
        # ~/.Trash): the result is not cached, and the check stands
        return None, f"reads {e.filename}, which the cache cannot hash ({e.strerror})"


def listing(d):
    try:
        return h(*sorted(os.listdir(d)))
    except OSError:
        return None


def store(kind, ident, inputs, output, seconds, artifacts_from=None, item=None):
    path = entry_path(kind, ident)
    vkey = h(ident, json.dumps(inputs))
    e = {"format": FORMAT, "kind": kind, "id": ident, "key": vkey, "inputs": inputs, "output": output,
         "seconds": round(seconds, 1), "sha": os.environ.get("GATE_SHA") or _git("rev-parse", "HEAD").strip(),
         "branch": os.environ.get("GATE_BRANCH", ""), "time": dt.datetime.now().isoformat(timespec="seconds"),
         "root": ROOT, "item": item or ident}
    old = None
    if artifacts_from:
        name = f"{ident}.{vkey[:12]}.d"
        dest = os.path.join(os.path.dirname(path), name)
        tmp = tempfile.mkdtemp(dir=os.path.dirname(path) if os.path.isdir(os.path.dirname(path)) else _mk(os.path.dirname(path)),
                               prefix=".tmp-")
        shutil.copytree(artifacts_from, os.path.join(tmp, "a"))
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(os.path.join(tmp, "a"), dest)
        os.rmdir(tmp)
        e["artifacts"] = name
        try:
            old = json.load(open(path)).get("artifacts")
        except (OSError, ValueError):
            pass
    if kind in KINDS:
        e["parts"] = kind_parts(KINDS[kind])
    atomic_write(path, e)
    # the latest entry of this item, whatever its key: `explain` starts from it
    atomic_write(os.path.join(STORE, kind, "latest", h(e.get("item", ident))[:24] + ".json"),
                 {"id": ident, "item": e.get("item"), "time": e["time"]})
    if old and old != e.get("artifacts"):
        shutil.rmtree(os.path.join(os.path.dirname(path), old), ignore_errors=True)
    return e


def _mk(d):
    os.makedirs(d, exist_ok=True)
    return d


def drop(kind, ident):
    try:
        os.remove(entry_path(kind, ident))
    except OSError:
        pass


def badge(e):
    return f"PASS (cached {e['key'][:10]}, from {e['sha'][:9]} {e['time'][:10]})"


# --------------------------------------------------------------- mismatches

def record_mismatch(kind, item, e, detail):
    """A cached PASS that a rerun did not reproduce: a key that missed an input."""
    when = dt.date.today().isoformat()
    row = {"time": dt.datetime.now().isoformat(timespec="seconds"), "kind": kind, "item": item,
           "entry": e["key"], "from": e["sha"], "root": ROOT, "detail": detail[-4000:]}
    _mk(STORE)
    with open(os.path.join(STORE, "mismatches.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    ledger = os.path.join(ROOT, "bugs", "ledger.jsonl")
    try:
        ids = [json.loads(l).get("id", "") for l in open(ledger) if l.strip()]
        n = max([int(i[1:]) for i in ids if re.fullmatch(r"B\d+", i)] or [0]) + 1
        bug = {"id": f"B{n:03d}",
               "title": f"gate cache: the cached PASS of {kind} {item} (entry {e['key'][:10]}, from {e['sha'][:9]}) "
                        f"did not reproduce when the sample reran it: its key misses an input",
               "found": when, "found_by": "gate cache sample (tools/gate/cache.py)", "engines": [],
               "disagreement": False, "status": "open", "fixed": None, "commit": None, "test": None,
               "root_cause": None}
        with open(ledger, "a") as f:
            f.write(json.dumps(bug) + "\n")
        return bug["id"]
    except OSError:
        return None


def mismatch_text(kind, item, e, bid, why):
    return (f"CACHE MISMATCH: {kind} {item}: cached {badge(e)}, but the sampled rerun {why}.\n"
            f"  The key missed an input; the entry is removed. Recorded as {bid or '(ledger not writable)'} "
            f"in bugs/ledger.jsonl and in {os.path.join(STORE, 'mismatches.jsonl')}.")


# ------------------------------------------------------------------ running

class Result:
    def __init__(self, item):
        self.item, self.output, self.rc, self.status, self.note = item, "", 0, "", ""


def run_items(kind, items, command, item_files, passed, artifacts=None, cacheable=lambda it: True,
              ignore=lambda it: frozenset(), workers=None):
    """Run `command(item, trace_file)` -> (output, rc) for each item not cached.
    `item_files(item)` lists files the check reads besides what viv traces
    (the program, a companion it looks for); `passed(output, rc)` says whether
    a result may be cached. `artifacts`, when given, is (dir_of(item),
    restore(item, cached_dir)): a cached item restores its files instead of
    running. An item is named by its repository path, which `cacheable`
    may refuse (a program written to a temporary file). Returns the results
    in the items' order, the counts, and why the cache is off (or None)."""
    k = KINDS[kind]
    use = enabled()
    kkey, why_not = kind_key(k) if use else (None, "the cache is off")
    rate = sample_rate()
    forget()   # files may have changed since an earlier call in this process
    results = {it: Result(it) for it in items}
    todo, counts = [], {"items": len(items), "cached": 0, "run": 0, "sampled": 0, "stored": 0, "uncacheable": 0,
                        "mismatch": 0}
    reasons = {}
    idents = {}
    for it in items:
        if kkey is None or not cacheable(it):
            todo.append((it, None))
            continue
        ident = h(kkey, it)   # `it`: the program's path from the repository root
        idents[it] = ident
        e = lookup(kind, ident, ignore(it))
        if e is None:
            todo.append((it, None))
        elif rate > 0 and random.random() < rate:
            todo.append((it, e))
            counts["sampled"] += 1
        else:
            r = results[it]
            r.status, r.note, r.cached = "cached", badge(e), e
            if artifacts:
                artifacts[1](it, os.path.join(os.path.dirname(entry_path(kind, ident)), e["artifacts"]))
            r.output = e["output"]
            counts["cached"] += 1

    tdir = tempfile.mkdtemp(prefix="gate-trace-")

    def bump(c):
        with _lock:
            counts[c] += 1

    def work(job):
        it, e = job
        r = results[it]
        tf = os.path.join(tdir, h(it)[:16] + ".trace")
        t0 = time.time()
        out, rc = command(it, tf if it in idents else None)
        r.output, r.rc = out, rc
        ok = passed(out, rc)
        r.status = "pass" if ok else "fail"
        bump("run")
        if kkey is None or it not in idents:
            return
        if e is not None:   # a sampled hit: it must reproduce
            if not ok or (k.compare_output and out != e["output"]) or (artifacts and not same_tree(artifacts[0](it),
                                                                             os.path.join(os.path.dirname(entry_path(kind, idents[it])), e["artifacts"]))):
                why = ("failed" if not ok else "printed something else" if out != e["output"]
                       else "made other files")
                detail = f"cached output:\n{e['output']}\nrerun output:\n{out}"
                bid = record_mismatch(kind, it, e, detail)
                drop(kind, idents[it])
                r.status, r.note = "mismatch", mismatch_text(kind, it, e, bid, why)
                bump("mismatch")
                return
            r.note = "(sampled: the rerun reproduced the cached PASS)"
        if not ok:
            drop(kind, idents[it])
            return
        trace = open(tf).read() if os.path.exists(tf) else ""
        forget()   # the run may have changed files hashed before it
        inputs, why = parse_trace(trace, item_files(it), k.io_ok)
        if inputs is None:
            bump("uncacheable")
            reasons[it] = why
            r.note = r.note or f"(not cached: {why})"
            return
        store(kind, idents[it], inputs, out, time.time() - t0, artifacts[0](it) if artifacts else None, item=it)
        bump("stored")

    with ThreadPoolExecutor(max_workers=workers or jobs()) as ex:
        list(ex.map(work, todo))
    shutil.rmtree(tdir, ignore_errors=True)
    rep = os.environ.get("GATE_CACHE_REPORT")
    if rep:
        with _lock:
            try:
                old = json.load(open(rep))
            except (OSError, ValueError):
                old = {}
            for c, v in counts.items():
                old[c] = old.get(c, 0) + v
            old["enabled"] = use
            atomic_write(rep, old)
    return [results[it] for it in items], counts, (why_not if kkey is None else None)


def same_tree(a, b):
    fa = sorted(os.path.relpath(os.path.join(d, f), a) for d, _, fs in os.walk(a) for f in fs)
    fb = sorted(os.path.relpath(os.path.join(d, f), b) for d, _, fs in os.walk(b) for f in fs)
    if fa != fb:
        return False
    return all(open(os.path.join(a, f), "rb").read() == open(os.path.join(b, f), "rb").read() for f in fa)


def sh(cmd, trace, env_extra=None):
    env = dict(os.environ)
    if trace:
        env["VIV_TRACE_INPUTS"] = trace
    else:
        env.pop("VIV_TRACE_INPUTS", None)
    if env_extra:
        env.update(env_extra)
    p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.stdout.decode(errors="replace"), p.returncode


def summary_line(counts, why_not):
    if why_not:
        return f"cache: off ({why_not}); {counts['run']} run"
    s = (f"cache: {counts['cached']} of {counts['items']} cached, {counts['run']} run "
         f"({counts['sampled']} sampled), {counts['stored']} stored")
    if counts["uncacheable"]:
        s += f", {counts['uncacheable']} not cacheable"
    if counts["mismatch"]:
        s += f", {counts['mismatch']} MISMATCH"
    return s


# ------------------------------------------------------------ the programs

def dir_programs(d, names):
    full = resolve(d)
    if names:
        return list(names)
    return sorted(f[:-4] for f in os.listdir(full) if f.endswith(".viv"))


def companions(d, name):
    return [os.path.join(resolve(d), name + ".viv"), os.path.join(resolve(d), "tests", name + ".viv")]


def cmd_dir_check(kind, argv, script_cmd, env_extra=None):
    """A check over the programs of one directory (three-engines,
    fast-identity), one program per run, cached per program."""
    d, names = argv[0], argv[1:]
    d = rel_or_abs(resolve(d))
    progs = [os.path.join(d, n) for n in dir_programs(d, names)]
    split = lambda p: os.path.split(p)
    res, counts, why_not = run_items(
        kind, progs, lambda p, tf: sh(script_cmd(*split(p)), tf, env_extra),
        lambda p: companions(*split(p)), lambda out, rc: rc == 0)
    bad = 0
    for r in res:
        out = r.output.rstrip("\n")
        lines = out.split("\n") if out else [f"{os.path.basename(r.item)} (no output)"]
        if r.status == "cached":
            lines[0] = f"{lines[0]}    {r.note}"
        print("\n".join(lines))
        if r.note and r.status != "cached":
            print("    " + r.note.replace("\n", "\n    "))
        if r.status in ("fail", "mismatch"):
            bad = 1
    print(summary_line(counts, why_not))
    return bad


def cmd_three_engines(argv):
    return cmd_dir_check("three-engines", argv, lambda d, n: ["sh", os.path.join(ROOT, "tools", "three-engines.sh"), d, n])


def cmd_fast_identity(argv):
    return cmd_dir_check("fast-identity", argv, lambda d, n: ["sh", os.path.join(ROOT, "tools", "fast-check.sh"), d, n],
                         {"IDENTITY": "1", "NO_INTERP": "1"})


def cmd_items(argv):
    """scf/check.sh's runner: each dir:name item through tools/both-engines.sh,
    its output and status in <outdir>/<dir_name>.out and .rc."""
    kind = argv[0]
    outdir, rest, i = None, [], 1
    while i < len(argv):
        if argv[i] == "--outdir":
            outdir = argv[i + 1]
            i += 2
        elif argv[i] == "--jobs":
            os.environ["JOBS"] = argv[i + 1]
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    assert kind == "both-engines" and outdir
    progs = [os.path.join(*it.split(":", 1)) for it in rest]
    split = lambda p: os.path.split(p)
    res, counts, why_not = run_items(
        kind, progs, lambda p, tf: sh(["sh", os.path.join(ROOT, "tools", "both-engines.sh"), *split(p)], tf),
        lambda p: companions(*split(p)), lambda out, rc: rc == 0)
    for it, r in zip(rest, res):
        key = it.replace("/", "_").replace(":", "_")
        out = r.output.rstrip("\n")
        if r.status == "cached":
            lines = out.split("\n")
            lines[0] = f"{lines[0]}    {r.note}"
            out = "\n".join(lines)
        elif r.note:
            out += "\n    " + r.note.replace("\n", "\n    ")
        open(os.path.join(outdir, key + ".out"), "w").write(out + "\n")
        open(os.path.join(outdir, key + ".rc"), "w").write(f"{0 if r.status in ('cached', 'pass') else (r.rc or 1)}\n")
    print(summary_line(counts, why_not), file=sys.stderr)
    return 0


MEMCHECK_BAD = {"ASAN", "UBSAN", "CC-FAILED", "TIMEOUT", "BLOCKED-LOAD"}


def cmd_memcheck(argv):
    """tools/memcheck/ci.sh's step 1, for the programs on stdin (one path a
    line): their rows on stdout, as `ci.sh one` prints them."""
    progs = [l.strip() for l in sys.stdin if l.strip()]
    ci = os.path.join(ROOT, "tools", "memcheck", "ci.sh")

    def ok(out, rc):
        f = out.split("\t")
        return rc == 0 and len(f) >= 2 and f[1] not in MEMCHECK_BAD and out.count("\n") == 1

    res, counts, why_not = run_items("memcheck", [rel_or_abs(p) for p in progs],
                                     lambda p, tf: sh(["sh", ci, "one", resolve(p)], tf),
                                     lambda p: [resolve(p)], ok, cacheable=lambda p: not os.path.isabs(p))
    bad = 0
    for r in res:
        row = r.output.rstrip("\n")
        if r.status == "cached":
            row += "\t" + r.note   # a fourth column: ci.sh reads the first three
        if row:
            print(row)
        if r.status == "mismatch":
            print(r.note, file=sys.stderr)
            bad = 1
    print("   " + summary_line(counts, why_not), file=sys.stderr)
    return bad


def cmd_golden(argv):
    """hub-golden's record step: one snapshot per program (stdin), into <dir>/<program>/."""
    snap = argv[0]
    progs = [l.strip() for l in sys.stdin if l.strip()]
    hg = os.path.join(ROOT, "tools", "hub-golden.sh")

    def restore(p, cached):
        dest = os.path.join(snap, p)
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(cached, dest)

    def passed(out, rc):
        # every recorded snapshot is an answer, except a run the time limit stopped
        return rc == 0

    def run(p, tf):
        out, rc = sh(["sh", hg, "one", p, snap], tf)
        d = os.path.join(snap, p)
        timed_out = any("exit timeout" in open(os.path.join(d, f), errors="replace").read()
                        for f in ("interp", "native", "direct") if os.path.isfile(os.path.join(d, f)))
        return out, (rc if not timed_out else 124)

    res, counts, why_not = run_items("golden", progs, run, lambda p: [os.path.join(ROOT, p)], passed,
                                     artifacts=(lambda p: os.path.join(snap, p), restore))
    bad = 0
    for r in res:
        if r.status == "mismatch":
            print(r.note)
            bad = 1
    print(summary_line(counts, why_not))
    return bad


# ------------------------------------------------------------ maintenance

def cmd_stats(argv):
    if not os.path.isdir(STORE):
        print(f"no cache at {STORE}")
        return 0
    for k in sorted(os.listdir(STORE)):
        d = os.path.join(STORE, k)
        if not os.path.isdir(d):
            continue
        n, size = 0, 0
        for dp, _, fs in os.walk(d):
            for f in fs:
                size += os.path.getsize(os.path.join(dp, f))
                n += f.endswith(".json")
        print(f"  {k:16} {n:6} entries  {size / 1e6:8.1f} MB")
    mm = os.path.join(STORE, "mismatches.jsonl")
    if os.path.exists(mm):
        print(f"  mismatches recorded: {sum(1 for _ in open(mm))} ({mm})")
    return 0


def cmd_prune(argv):
    days = float(argv[argv.index("--days") + 1]) if "--days" in argv else 14
    cut = time.time() - days * 86400
    gone = 0
    for dp, dirs, fs in os.walk(STORE):
        for f in fs:
            if f.endswith(".json") and os.path.getmtime(os.path.join(dp, f)) < cut:
                p = os.path.join(dp, f)
                try:
                    a = json.load(open(p)).get("artifacts")
                    if a:
                        shutil.rmtree(os.path.join(dp, a), ignore_errors=True)
                except (OSError, ValueError):
                    pass
                os.remove(p)
                gone += 1
    print(f"pruned {gone} entries unused for {days:g} days")
    return 0


def lookup_any(kind, item):
    """The latest stored entry of an item, whatever its key (for ordering by last time)."""
    try:
        ident = json.load(open(os.path.join(STORE, kind, "latest", h(item)[:24] + ".json")))["id"]
        return json.load(open(entry_path(kind, ident)))
    except (OSError, ValueError, KeyError):
        return None


def cmd_explain(argv):
    """explain <kind> <program>: why the latest stored entry of a program is or
    is not reused now: each part of its key, and each input, then and now."""
    kind, item = argv[0], rel_or_abs(resolve(argv[1]))
    e = None
    # a program check names its items without .viv (pde/wave_1d), the others with it
    for it in (item, item[:-4] if item.endswith(".viv") else item + ".viv"):
        try:
            ident = json.load(open(os.path.join(STORE, kind, "latest", h(it)[:24] + ".json")))["id"]
            e, item = json.load(open(entry_path(kind, ident))), it
            break
        except (OSError, ValueError):
            pass
    if e is None:
        print(f"no stored entry of {kind} {item}")
        return 1
    print(f"{kind} {item}: latest entry {e['key'][:10]} from {e['branch']} {e['sha'][:9]} {e['time']} ({e['root']})")
    k = KINDS[kind]
    kk, why = kind_key(k)
    same_id = kk is not None and h(kk, item) == ident
    print(f"  key parts: {'unchanged' if same_id else 'CHANGED' + (f' ({why})' if why else '')}")
    now = kind_parts(k)
    for name in sorted(set(now) | set(e.get("parts", {}))):
        a, b = e.get("parts", {}).get(name), now.get(name)
        print(f"    {'  ' if a == b else '!='} {name}: {a if a == b else f'{a}  ->  {b}'}")
    cur = inputs_now(e["inputs"])
    changed = [(o, n) for o, n in zip(e["inputs"], cur) if o != n]
    print(f"  inputs: {len(e['inputs'])}, {len(changed)} changed")
    for o, n in changed:
        print(f"    != {o[0]} {o[1]}: {o[2]} -> {n[2]}")
    print("  reused now" if same_id and not changed else "  not reused now")
    return 0


def cmd_clear(argv):
    shutil.rmtree(STORE, ignore_errors=True)
    print(f"removed {STORE}")
    return 0


def main(argv):
    cmds = {"three-engines": cmd_three_engines, "fast-identity": cmd_fast_identity, "items": cmd_items,
            "memcheck": cmd_memcheck, "golden": cmd_golden, "stats": cmd_stats, "prune": cmd_prune,
            "clear": cmd_clear, "explain": cmd_explain}
    if not argv or argv[0] not in cmds:
        print(__doc__.split("\n\n")[1])
        return 2
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
