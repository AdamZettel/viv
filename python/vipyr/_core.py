# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Compile a .viv file into a Python module: export, build, cache, load.

The steps of `load(path)`:

1. `viv export` writes the collection's C file, its ctypes wrapper and its
   type stub. A file with `<stem>.export.toml` beside it is exported by that
   manifest; any other file by `viv export --infer`, which takes each
   top-level def whose parameters are annotated.
2. The system C compiler builds the C file into a shared library, with the
   exact flags (the interpreter's bits) or, with `fast=True`, the fast ones.
3. The result is cached under `__pycache__/vipyr/` beside the source (or a
   user cache directory when that is not writable). An entry is keyed by the
   SHA-256 of the source, the manifest, the compiler binary and the flags,
   and is valid while every file the program imported still hashes the same.
4. The wrapper is executed as a module object, which loads the library.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import types
from pathlib import Path

EXACT_FLAGS = ["-O2", "-ffp-contract=off"]
FAST_FLAGS = ["-O3", "-DVIV_FAST", "-ffp-contract=fast"]
COMMON_FLAGS = ["-fvisibility=hidden", "-shared"] + (["-static-libgcc"] if sys.platform == "win32" else ["-fPIC"])
CACHE_FORMAT = "1"

_PKG = Path(__file__).resolve().parent
_lock = threading.RLock()
_ident_memo: dict = {}
stats = {"exports": 0, "compiles": 0, "cache_hits": 0}


class VivCompileError(SyntaxError):
    """`viv export` refused the file: a parse or type error, located in the
    .viv source (filename, lineno, offset, text), as Python's SyntaxError."""


class VivBuildError(ImportError):
    """The C compiler, or the compiler binary itself, could not be run or
    failed on the exported C."""


# ------------------------------------------------------------ the compiler

def compiler() -> str:
    """The `viv` binary: $VIV, the one inside this package (platform wheels),
    `viv` on PATH, or the repository's target/release/viv for a checkout."""
    env = os.environ.get("VIV")
    if env:
        return env
    exe = "viv.exe" if sys.platform == "win32" else "viv"
    bundled = _PKG / "bin" / exe
    if bundled.is_file():
        return str(bundled)
    found = shutil.which("viv")
    if found:
        return found
    repo = _repo_root()
    if repo is not None and (repo / "target" / "release" / exe).is_file():
        return str(repo / "target" / "release" / exe)
    raise VivBuildError("vipyr: no `viv` compiler: set $VIV, put `viv` on PATH, or install a platform wheel of vipyr")


def _repo_root():
    for d in _PKG.parents:
        if (d / "Cargo.toml").is_file() and (d / "lib").is_dir():
            return d
    return None


def viv_root():
    """The directory whose lib/ holds Viv's library modules: the copy inside
    this package (platform wheels), $VIV_ROOT, or the repository."""
    if (_PKG / "_root" / "lib").is_dir():
        return str(_PKG / "_root")
    if os.environ.get("VIV_ROOT"):
        return os.environ["VIV_ROOT"]
    repo = _repo_root()
    return str(repo) if repo else None


def _env():
    env = dict(os.environ)
    root = viv_root()
    if root:
        env["VIV_ROOT"] = root
    return env


def compiler_identity(path=None) -> str:
    """SHA-256 of the compiler binary and its `--version`, memoized per
    process by the file's size and modification time."""
    path = path or compiler()
    try:
        st = os.stat(path)
    except OSError as e:
        raise VivBuildError("vipyr: cannot read the compiler %s: %s" % (path, e))
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _ident_memo:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        try:
            ver = subprocess.run([path, "--version"], capture_output=True, text=True).stdout.strip()
        except OSError as e:
            raise VivBuildError("vipyr: cannot run the compiler %s: %s" % (path, e))
        _ident_memo[key] = "%s %s" % (ver or "viv", h.hexdigest())
    return _ident_memo[key]


def cc_command():
    """$CC, else `cc` (`gcc` on Windows, where vipyr builds with MinGW-w64)."""
    return os.environ.get("CC") or ("gcc" if sys.platform == "win32" else "cc")


def have_cc() -> bool:
    """Whether the C compiler `cc_command()` names can be found."""
    cmd = cc_command().split()
    return bool(cmd) and (shutil.which(cmd[0]) is not None or os.path.isfile(cmd[0]))


class NoCompiler(VivBuildError):
    """No C compiler is installed, and no precompiled library matches."""


def blas_flags():
    """The BLAS a fast build links, as `viv build --fast` chooses it
    (lang/viv/src/fast_cli.rs, blas_flags): Accelerate on macOS, elsewhere
    what $VIV_BLAS_LIBS names; $VIV_BLAS=0 turns it off."""
    if os.environ.get("VIV_BLAS") == "0":
        return []
    if sys.platform == "darwin":
        return ["-DVIV_BLAS", "-framework", "Accelerate"]
    libs = os.environ.get("VIV_BLAS_LIBS", "").split()
    return ["-DVIV_BLAS"] + libs if libs else []


def flags(fast=False):
    return (FAST_FLAGS + blas_flags() if fast else EXACT_FLAGS) + COMMON_FLAGS


def lib_ext():
    return ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"


# ----------------------------------------------------------------- errors

_LOC = re.compile(r"^\s*-->\s*(.+?):(\d+):(\d+)\s*$")


def compile_error(stderr: str, src: Path) -> VivCompileError:
    """The first located diagnostic of `viv export`'s stderr as a
    SyntaxError in the .viv file it names."""
    lines = stderr.splitlines()
    msg = next((l.strip() for l in lines if l.strip().startswith("error")), (lines or ["viv export failed"])[0])
    msg = re.sub(r"^error(\[[^\]]*\])?:\s*", "", msg)
    filename, lineno, offset, text = str(src), None, None, None
    for l in lines:
        m = _LOC.match(l)
        if m:
            name, lineno, offset = m.group(1), int(m.group(2)), int(m.group(3))
            p = Path(name)
            if not p.is_absolute():
                p = (src.parent / p).resolve()
            filename = str(p)
            try:
                text = p.read_text(encoding="utf-8").splitlines()[lineno - 1] + "\n"
            except (OSError, IndexError):
                text = None
            break
    err = VivCompileError(msg, (filename, lineno, offset, text))
    err.viv_output = stderr
    return err


# ------------------------------------------------------------------ cache

def _sha_file(p) -> str:
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def manifest_for(src: Path):
    """The manifest `viv export` reads for `src`: `src` itself when it is a
    manifest (`pwdft.export.toml`), else `<stem>.export.toml` beside it."""
    if src.name.endswith(".export.toml"):
        return src
    m = src.with_name(src.stem + ".export.toml")
    return m if m.is_file() else None


def cache_root(src: Path) -> Path:
    """`__pycache__/vipyr` beside the source when writable, else a user
    cache directory ($VIPYR_CACHE, $XDG_CACHE_HOME/vipyr or
    ~/.cache/vipyr), one subdirectory per source directory."""
    forced = os.environ.get("VIPYR_CACHE")
    if not forced:
        d = src.parent / "__pycache__" / "vipyr"
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / (".probe-%d" % os.getpid())
            probe.write_bytes(b"")
            probe.unlink()
            return d
        except OSError:
            pass
    base = Path(forced) if forced else Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "vipyr"
    d = base / hashlib.sha256(str(src.parent).encode()).hexdigest()[:16]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _head_key(src: Path, fast: bool) -> str:
    """The part of the key known before a build: source, manifest, compiler,
    C compiler and flags."""
    h = hashlib.sha256()
    h.update(("vipyr cache %s\n" % CACHE_FORMAT).encode())
    h.update(src.read_bytes())
    m = manifest_for(src)
    if m is not None and m != src:
        h.update(b"\0manifest\0" + m.read_bytes())
    h.update(("\0%s\0%s\0%s\0" % (compiler_identity(), cc_command(), " ".join(flags(fast)))).encode())
    return h.hexdigest()


class Entry:
    """A built module in the cache: its directory, its name and the
    collection's name (the generated files' stem)."""

    def __init__(self, path: Path, info: dict):
        self.path = path
        self.info = info
        self.name = info["name"]

    @property
    def py(self):
        return self.path / (self.name + ".py")

    @property
    def lib(self):
        return self.path / ("lib" + self.name + lib_ext())

    @property
    def pyi(self):
        return self.path / (self.name + ".pyi")

    @property
    def notes(self):
        return self.info.get("notes", [])


def _valid(slot: Path):
    """The entry `slot/current` names, if every file it was built from still
    hashes the same."""
    try:
        cur = (slot / "current").read_text().strip()
        d = slot / cur
        info = json.loads((d / "entry.json").read_text())
    except (OSError, ValueError):
        return None
    for p, sha in info.get("deps", {}).items():
        try:
            if _sha_file(p) != sha:
                return None
        except OSError:
            return None
    # a module looked for and absent then would be found first now
    if any(os.path.exists(p) for p in info.get("absent", [])):
        return None
    e = Entry(d, info)
    return e if e.py.is_file() and e.lib.is_file() else None


class _FileLock:
    """An advisory lock on a file (fcntl), so two processes do not build one
    entry at once; where there is no fcntl the atomic renames still keep the
    cache consistent."""

    def __init__(self, path: Path):
        self.path = path
        self.f = None

    def __enter__(self):
        try:
            import fcntl
        except ImportError:
            return self
        self.f = open(self.path, "a+")
        fcntl.flock(self.f.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if self.f is not None:
            import fcntl
            fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
            self.f.close()


def export(src: Path, out: Path):
    """Run `viv export` on `src` (its manifest when there is one) into
    `out`; a refusal is a VivCompileError located in the source. Returns the
    collection's name, the notes on defs left out, and the inputs the
    compiler reported (VIV_TRACE_INPUTS): {path: True if read, False if
    looked for and absent}."""
    m = manifest_for(src)
    cmd = [compiler(), "export"] + ([str(m)] if m else ["--infer", str(src)]) + ["-o", str(out)]
    stats["exports"] += 1
    trace = out / ".inputs"
    env = _env()
    env["VIV_TRACE_INPUTS"] = str(trace)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    except OSError as e:
        raise VivBuildError("vipyr: cannot run %s: %s" % (cmd[0], e))
    if r.returncode != 0:
        raise compile_error(r.stderr, src)
    notes = [l[len("note: "):] for l in r.stderr.splitlines() if l.startswith("note: ")]
    for l in r.stderr.splitlines():
        if l.startswith("warning: "):
            # a license obligation (an MPL or LGPL module in the build)
            import warnings
            warnings.warn("viv export: " + l[len("warning: "):], stacklevel=2)
    names = [p.stem for p in out.glob("*.h")]
    if len(names) != 1:
        raise VivBuildError("vipyr: `viv export` wrote no single header in %s" % out)
    inputs = {}
    if trace.is_file():
        for line in trace.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == 3 and parts[0] == "f" and parts[1].endswith((".viv", ".toml")):
                inputs[parts[1]] = inputs.get(parts[1], False) or parts[2] == "1"
        trace.unlink()
    return names[0], notes, inputs


def build_c(c_file: Path, lib: Path, fast=False):
    cmd = cc_command().split() + flags(fast) + [str(c_file), "-o", str(lib), "-lm"]
    stats["compiles"] += 1
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:
        raise VivBuildError("vipyr: no C compiler (%s); set $CC. The command was:\n    %s" % (e, " ".join(cmd)))
    if r.returncode != 0:
        raise VivBuildError("vipyr: building %s failed:\n    %s\n%s" % (c_file.name, " ".join(cmd), r.stderr))


def _say_building(src: Path):
    """A first call compiles a library and can take a while (tens of
    seconds for a large program): at a terminal, say so before it starts.
    VIPYR_QUIET=1 keeps it silent."""
    if os.environ.get("VIPYR_QUIET") or not getattr(sys.stderr, "isatty", lambda: False)():
        return
    sys.stderr.write("vipyr: compiling %s to a C library (once; later calls load it from the cache)\n" % src.name)
    sys.stderr.flush()


def build(src, fast=False) -> Entry:
    """The cache entry for `src`, built when missing or stale."""
    src = Path(src).resolve()
    if not src.is_file():
        raise FileNotFoundError(str(src))
    from . import _prebuilt
    pre = _prebuilt.lookup(src, fast)
    if pre is not None:
        return pre
    root = cache_root(src)
    slot = root / ("%s-%s%s" % (module_name(src), _head_key(src, fast)[:20], "-fast" if fast else ""))
    with _lock:
        e = _valid(slot)
        if e is not None:
            stats["cache_hits"] += 1
            return e
        if not have_cc():
            raise NoCompiler("vipyr: no C compiler (%r not found) and no precompiled library for %s; "
                             "install one (Xcode command line tools, gcc or clang; MinGW-w64 gcc on Windows) "
                             "or set $CC" % (cc_command(), src))
        slot.mkdir(parents=True, exist_ok=True)
        with _FileLock(slot / ".lock"):
            e = _valid(slot)              # another process built it meanwhile
            if e is not None:
                stats["cache_hits"] += 1
                return e
            tmp = Path(tempfile.mkdtemp(prefix=".build-", dir=slot))
            _say_building(src)
            try:
                name, notes, inputs = export(src, tmp)
                build_c(tmp / (name + ".c"), tmp / ("lib" + name + lib_ext()), fast)
                deps = {str(src): _sha_file(src)}
                m = manifest_for(src)
                if m is not None:
                    deps[str(m)] = _sha_file(m)
                for p, read in sorted(inputs.items()):
                    if read and os.path.isfile(p):
                        deps[str(Path(p).resolve())] = _sha_file(p)
                absent = sorted(p for p, read in inputs.items() if not read)
                full = hashlib.sha256((slot.name + json.dumps(sorted(deps.items())) + json.dumps(absent)).encode()).hexdigest()[:20]
                info = {"name": name, "source": str(src), "fast": fast, "flags": flags(fast), "key": full,
                        "deps": deps, "absent": absent, "notes": notes, "format": CACHE_FORMAT}
                (tmp / "entry.json").write_text(json.dumps(info, indent=1))
                final = slot / full
                if final.exists():
                    shutil.rmtree(tmp, ignore_errors=True)
                else:
                    os.replace(tmp, final)
                ptr = slot / (".current-%d" % os.getpid())
                ptr.write_text(full + "\n")
                os.replace(ptr, slot / "current")
            except BaseException:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
            e = _valid(slot)
            if e is None:
                raise VivBuildError("vipyr: the cache entry for %s did not validate after its build" % src)
            return e


# ---------------------------------------------------------------- modules

def exec_entry(e: Entry, module: types.ModuleType, source: Path):
    """Run the generated wrapper in `module`, loading the cached library.
    Afterwards `module.__file__` is the .viv source."""
    var = e.name.upper() + "_LIB"
    code = compile(e.py.read_text(encoding="utf-8"), str(e.py), "exec")
    with _lock:
        old = os.environ.get(var)
        os.environ[var] = str(e.lib)
        try:
            module.__dict__["__file__"] = str(e.py)
            exec(code, module.__dict__)
        finally:
            if old is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = old
    module.__file__ = str(source)
    module.__vipyr__ = e
    from .frames import wrap_module
    return wrap_module(module)


def module_name(src: Path) -> str:
    """`kernels.viv` -> kernels; `pwdft.export.toml` -> pwdft."""
    n = src.name
    return n[: -len(".export.toml")] if n.endswith(".export.toml") else src.stem


def load(path, fast=False, name=None) -> types.ModuleType:
    """Compile `path` (a .viv file, or a `<name>.export.toml` manifest of a
    program or kernel collection) if needed and return it as a module.

    The module has a function per exported def (`__all__`), the error classes
    (VivError and its subclasses), `version()` and `tests()`. `fast=True`
    builds with the fast flags, whose results may differ in the last bits.
    """
    src = Path(path).resolve()
    try:
        e = build(src, fast=fast)
    except NoCompiler:
        if manifest_for(src) is not None or os.environ.get("VIPYR_INTERP") == "0":
            raise
        from . import _interp
        return _interp.load(src, name or module_name(src))
    mod = types.ModuleType(name or module_name(src))
    return exec_entry(e, mod, src)


def write_stub(path, out=None) -> Path:
    """Write the type stub of `path` (`kernels.viv` -> `kernels.pyi`) beside
    the source, or at `out`."""
    src = Path(path).resolve()
    e = build(src)
    dst = Path(out) if out else src.with_suffix(".pyi")
    shutil.copyfile(e.pyi, dst)
    return dst


def spec_for_viv(fullname: str, viv_path: Path, loader):
    return importlib.util.spec_from_file_location(fullname, str(viv_path), loader=loader)
