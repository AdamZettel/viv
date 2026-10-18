# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""A PEP 517 build backend for packages with Viv modules.

In the library's pyproject.toml:

    [build-system]
    requires = ["vipyr"]
    build-backend = "vipyr.build"

    [project]
    name = "mypkg"
    version = "1.0"
    dependencies = ["numpy"]

    [tool.vipyr]
    packages = ["mypkg"]                 # Python packages copied into the wheel
    modules = ["mypkg/kernels.viv"]      # compiled into it

Each module is exported and built with the exact flags (`fast = true` for the
fast ones), and the wheel holds the generated wrapper (`mypkg/kernels.py`),
its shared library and its stub. Installing the wheel needs neither Viv nor
a C compiler: `from mypkg import kernels` loads the library directly. A wheel
with a library in it is tagged for this platform (`py3-none-<platform>`); one
without is `py3-none-any`.

Why a backend of its own rather than a setuptools plugin: the wheel is a zip
of files this module already has (the cache entry), so writing it directly
takes no build requirement beyond vipyr, works offline, and gives one
place that decides the platform tag. A setuptools plugin would add
setuptools to every build environment and still need the same tag logic.

`bundle-compiler = true` (vipyr's own pyproject) also puts the `viv`
binary at `vipyr/bin/viv` and Viv's library modules under
`vipyr/_root/lib`, so the platform wheel of vipyr carries its compiler.
The binary is $VIPYR_BUNDLE_VIV, or the checkout's target/release/viv.
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import sysconfig
import tarfile
import time
import zipfile
from pathlib import Path

from . import _core

# ------------------------------------------------------------ pyproject

def _toml_value(v: str):
    v = v.strip()
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1].encode().decode("unicode_escape")
    if v.startswith("'") and v.endswith("'"):
        return v[1:-1]
    if v in ("true", "false"):
        return v == "true"
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [_toml_value(x) for x in re.findall(r'"[^"]*"|\'[^\']*\'|[^,\s]+', inner)] if inner else []
    try:
        return int(v)
    except ValueError:
        return float(v)


def _mini_toml(text: str) -> dict:
    """The TOML pyproject files use, for Python 3.9 and 3.10 without tomli:
    tables, and keys whose values are strings, booleans, numbers or one-line
    or multi-line arrays of strings."""
    data: dict = {}
    table = data
    pending = None
    for raw in text.splitlines():
        line = raw.split(" #")[0].strip() if not raw.strip().startswith("#") else ""
        if not line:
            continue
        if pending is not None:
            pending[1] += " " + line
            if line.endswith("]"):
                table[pending[0]] = _toml_value(pending[1])
                pending = None
            continue
        if line.startswith("[") and line.endswith("]"):
            table = data
            for part in re.findall(r'"[^"]*"|[^.]+', line.strip("[]")):   # project.entry-points."vipyr.packages"
                table = table.setdefault(part.strip().strip('"'), {})
            continue
        key, _, val = line.partition("=")
        key, val = key.strip().strip('"'), val.strip()
        if val.startswith("[") and not val.endswith("]"):
            pending = [key, val]
            continue
        table[key] = _toml_value(val)
    return data


def read_pyproject(path="pyproject.toml") -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import tomllib
        return tomllib.loads(text)
    except ImportError:
        pass
    try:
        import tomli
        return tomli.loads(text)
    except ImportError:
        return _mini_toml(text)


# ----------------------------------------------------------------- wheel

def _dist_name(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def platform_tag() -> str:
    """This platform's wheel tag. On macOS 11 and later the minor version is
    0 in every tag installers accept (macosx_11_0_arm64), whatever
    sysconfig says the interpreter was built for (macosx-11.1-arm64)."""
    if os.environ.get("VIPYR_PLAT_TAG"):          # CI: the tag the binaries were built for
        return os.environ["VIPYR_PLAT_TAG"]
    plat = sysconfig.get_platform()
    m = re.fullmatch(r"macosx-(\d+)\.(\d+)-(\w+)", plat)
    if m and int(m.group(1)) >= 11:
        return "macosx_%s_0_%s" % (m.group(1), m.group(3))
    return re.sub(r"[-.]", "_", plat)


def _record_line(arc: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return "%s,sha256=%s,%d" % (arc, digest, len(data))


def _metadata(project: dict, licenses=()) -> str:
    lines = ["Metadata-Version: 2.1", "Name: %s" % project["name"], "Version: %s" % project["version"]]
    if project.get("description"):
        lines.append("Summary: %s" % project["description"])
    if isinstance(project.get("license"), str):
        lines.append("License: %s" % project["license"])
    for arc in licenses:
        lines.append("License-File: %s" % arc)
    if project.get("requires-python"):
        lines.append("Requires-Python: %s" % project["requires-python"])
    for d in project.get("dependencies", []):
        lines.append("Requires-Dist: %s" % d)
    for extra, deps in project.get("optional-dependencies", {}).items():
        lines.append("Provides-Extra: %s" % extra)
        for d in deps:
            lines.append('Requires-Dist: %s; extra == "%s"' % (d, extra))
    return "\n".join(lines) + "\n"


def _entry_points(project: dict) -> str:
    """entry_points.txt: [project.scripts] as console_scripts, then each
    [project.entry-points.<group>] (vipyr.packages for a distribution's families)."""
    groups = {}
    if project.get("scripts"):
        groups["console_scripts"] = project["scripts"]
    groups.update(project.get("entry-points", {}))
    return "".join("[%s]\n%s\n" % (g, "".join("%s = %s\n" % kv for kv in eps.items()))
                   for g, eps in groups.items() if eps)


def _package_files(packages):
    """(path on disk, path in the wheel) of each file of the packages,
    without caches and byte code."""
    out = []
    for pkg in packages:
        base = Path(pkg)
        for p in sorted(base.rglob("*")):
            if p.is_dir() or "__pycache__" in p.parts or p.suffix in (".pyc", ".so", ".dylib", ".dll"):
                continue
            if p.name.startswith(".") or p.parts[len(base.parts):][:1] in (("bin",), ("_root",), ("_prebuilt",), ("_py2viv",)):
                continue
            out.append((p, p.as_posix()))
    return out


DEFAULT_LICENSE_FILES = ["LICENSE*", "NOTICE*", "AUTHORS*", "COPYING*"]


def license_files(cfg: dict):
    """(path on disk, path under the wheel's .dist-info/licenses/) of each
    license and notice file: `[tool.vipyr] license-files`, globs relative to
    the pyproject (`../LICENSES/*.txt` reaches the repository), or by default
    LICENSE*, NOTICE*, AUTHORS* and COPYING* beside it (LICENSING.md)."""
    out, seen = [], set()
    for pat in cfg.get("license-files", DEFAULT_LICENSE_FILES):
        for p in sorted(Path(".").glob(pat)):
            if not p.is_file():
                continue
            arc = "/".join(x for x in p.as_posix().split("/") if x not in ("..", "."))
            if arc not in seen:
                seen.add(arc)
                out.append((p, arc))
    return out


def _bundle_files(dist_pkg: str):
    """The compiler and Viv's library modules, for vipyr's platform wheel."""
    viv = os.environ.get("VIPYR_BUNDLE_VIV")
    repo = _core._repo_root()
    for exe in ("viv", "viv.exe"):
        if not viv and repo is not None and (repo / "target" / "release" / exe).is_file():
            viv = str(repo / "target" / "release" / exe)
    if not viv:
        viv = _core.compiler()
    root = Path(os.environ.get("VIPYR_BUNDLE_ROOT") or (repo or ""))
    if not (root / "lib").is_dir():
        raise RuntimeError("vipyr.build: no Viv library (lib/) to bundle; set VIPYR_BUNDLE_ROOT")
    files = [(Path(viv), "%s/bin/%s" % (dist_pkg, "viv.exe" if str(viv).endswith(".exe") else "viv"))]
    for p in sorted((root / "lib").rglob("*.viv")):
        files.append((p, "%s/_root/lib/%s" % (dist_pkg, p.relative_to(root / "lib").as_posix())))
    return files


def _write_wheel(wheel_directory, project, files, native: bool, licenses=()) -> str:
    """`files`: (bytes or a path, path in the wheel, executable);
    `licenses`: (path, name) of the files for .dist-info/licenses/."""
    name, version = _dist_name(project["name"]), project["version"]
    tag = "py3-none-%s" % (platform_tag() if native else "any")
    whl = "%s-%s-%s.whl" % (name, version, tag)
    info = "%s-%s.dist-info" % (name, version)
    record = []
    fixed = (2020, 1, 1, 0, 0, 0)          # the same bytes for the same inputs
    Path(wheel_directory).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(Path(wheel_directory) / whl, "w", zipfile.ZIP_DEFLATED) as z:
        def add(arc, data, exe=False):
            zi = zipfile.ZipInfo(arc, fixed)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = ((0o755 if exe else 0o644) | 0o100000) << 16
            z.writestr(zi, data)
            record.append(_record_line(arc, data))

        for src, arc, exe in files:
            add(arc, src if isinstance(src, bytes) else Path(src).read_bytes(), exe)
        for src, arc in licenses:
            add(info + "/licenses/" + arc, Path(src).read_bytes())
        add(info + "/METADATA", _metadata(project, [a for _, a in licenses]).encode())
        add(info + "/WHEEL", ("Wheel-Version: 1.0\nGenerator: vipyr.build %s\nRoot-Is-Purelib: %s\nTag: %s\n"
                              % (_version(), "false" if native else "true", tag)).encode())
        eps = _entry_points(project)
        if eps:
            add(info + "/entry_points.txt", eps.encode())
        record.append(info + "/RECORD,,")
        zi = zipfile.ZipInfo(info + "/RECORD", fixed)
        zi.external_attr = 0o100644 << 16
        z.writestr(zi, "\n".join(record) + "\n")
    return whl


def _version():
    from . import __version__
    return __version__


def _config():
    data = read_pyproject()
    project = data.get("project", {})
    if not project.get("name") or not project.get("version"):
        raise RuntimeError("vipyr.build: [project] needs a name and a version")
    cfg = data.get("tool", {}).get("vipyr", {})
    packages = cfg.get("packages")
    if packages is None:
        guess = _dist_name(project["name"])
        packages = [guess] if Path(guess).is_dir() else []
    return project, cfg, packages


def compiled_modules(modules, fast=False):
    """(source, files in the wheel) per module: the wrapper, the library and
    the stub, beside the module's .viv path."""
    out = []
    for m in modules:
        src = Path(m)
        if src.suffix != ".viv" or not src.is_file():
            raise RuntimeError("vipyr.build: [tool.vipyr] modules names %s, which is not a .viv file" % m)
        e = _core.build(src, fast=fast)
        rel = src.parent.as_posix()
        pre = "" if rel in ("", ".") else rel + "/"
        out.append((src, [(e.py, pre + e.name + ".py", False), (e.lib, pre + e.lib.name, False),
                          (e.pyi, pre + e.name + ".pyi", False)]))
    return out


# ------------------------------------------------------------ prebuild

def _py2viv_files(dist_pkg: str):
    """The strict Python-to-Viv translator (tools/py2viv: its modules and
    rules), as `vipyr/_py2viv`, for vp.translate in an installed wheel."""
    repo = _core._repo_root()
    src = Path(os.environ.get("VIPYR_BUNDLE_ROOT") or (repo or "")) / "tools" / "py2viv"
    if not (src / "py2viv.py").is_file():
        return []
    out = []
    for p in sorted(src.glob("*.py")) + sorted((src / "rules").glob("*")):
        if p.is_file() and not p.name.startswith(("test_", ".")):
            out.append((p, "%s/_py2viv/%s" % (dist_pkg, p.relative_to(src).as_posix()), False))
    return out


def registry_paths(dist: str, root: Path):
    """The repository-relative files of a distribution's functions
    (python/vipyr/_registry.py): the manifest or source each one loads, the
    sources and cases files its spec names, and catalog/*.py for features."""
    import json
    from . import _registry
    paths, extra = [], []
    d = _registry.DISTS.get(dist)
    spec = {}
    if d is not None:
        f = root / d["path"] / d["module"] / "spec.json"
        if f.is_file():
            spec = json.loads(f.read_text(encoding="utf-8")).get("functions", {})
    for qual, reg in _registry.FUNCTIONS.items():
        if _registry.FAMILY_DIST.get(qual.split(".")[0]) != dist or reg["kind"] == "coming":
            continue
        if reg["kind"] == "feature":
            extra += [p.relative_to(root).as_posix() for p in sorted((root / "catalog").glob("*.py"))]
            if reg["source"] not in paths:
                paths.append(reg["source"])
            continue
        if reg["kind"] == "python":
            # a Python front end (vp.ode) that writes and compiles its driver
            # per problem on first use: nothing to precompile, ship the sources
            extra += list(reg.get("sources") or [])
            continue
        p = reg.get("manifest") or reg.get("source")
        if p not in paths:
            paths.append(p)
        s = spec.get(qual, {})
        extra += list(s.get("sources") or []) + ([s["cases_file"]] if s.get("cases_file") else [])
    return paths, sorted(set(extra))


def prebuilt_files(dist: str, cfg: dict, pkg: str):
    """Each program of the distribution exported and compiled now, as wheel
    files: `<pkg>/_prebuilt/<entry>/` (wrapper, library, stub),
    `<pkg>/_prebuilt/index.json` (vipyr/_prebuilt.py reads it), and the
    sources under `<pkg>/_root/` as laid out in the repository. A source
    under lib/ that is a dep only is not copied: vipyr's own wheel carries
    lib/. `prebuild-fast = true` (or $VIPYR_PREBUILD_FAST=1) adds the fast
    builds."""
    import json
    from . import _prebuilt
    root = Path(cfg.get("prebuild-root", ".")).resolve()
    if cfg["prebuild"] == "registry":
        paths, extra = registry_paths(dist, root)
    else:
        paths, extra = list(cfg["prebuild"]), list(cfg.get("prebuild-extra", []))
    kinds = [False] + ([True] if cfg.get("prebuild-fast") or os.environ.get("VIPYR_PREBUILD_FAST") == "1" else [])
    lib_root = _core.viv_root()
    entries, files, ship = [], [], set(extra)
    if not all((root / rel).is_file() for rel in paths):
        # an sdist, or a tree without the programs: nothing to precompile;
        # the installed package compiles on first use
        return []
    for rel in paths:
        src = (root / rel).resolve()
        for fast in kinds:
            e = _core.build(src, fast=fast)
            key = _prebuilt.content_key(src, fast)
            d = "%s-%s%s" % (e.name, key[:16], "-fast" if fast else "")
            deps = {_prebuilt.relative_dep(p, src, lib_root): sha for p, sha in e.info["deps"].items()}
            absent = [_prebuilt.relative_dep(p, src, lib_root) for p in e.info.get("absent", [])]
            info = {k: e.info[k] for k in ("name", "fast", "flags", "notes", "format")}
            entries.append({"key": key, "path": rel, "dir": d, "compiler": _core.compiler_identity(),
                            "cc": _core.cc_command(), "deps": deps, "absent": absent, "info": info})
            for f in (e.py, e.lib, e.pyi):
                if f.is_file():
                        files.append((f, "%s/_prebuilt/%s/%s" % (pkg, d, f.name), False))
            files.append((json.dumps(dict(info, key=key), indent=1).encode(),
                          "%s/_prebuilt/%s/entry.json" % (pkg, d), False))
            for p in e.info["deps"]:
                rp = Path(p).resolve()
                try:
                    r = rp.relative_to(root).as_posix()
                except ValueError:
                    continue
                if r.startswith("lib/") and r.endswith(".viv") and rp.parent != src.parent:
                    continue
                ship.add(r)
        ship.add(rel)
    for r in sorted(ship):
        p = root / r
        if p.is_file():
            files.append((p, "%s/_root/%s" % (pkg, r), False))
    index = {"format": _prebuilt.FORMAT, "platform": platform_tag(), "entries": entries}
    files.append((json.dumps(index, indent=1, sort_keys=True).encode(), "%s/_prebuilt/index.json" % pkg, False))
    return files


# ------------------------------------------------------------- PEP 517

def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_sdist(config_settings=None):
    return []


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    project, cfg, packages = _config()
    files = [(p, arc, False) for p, arc in _package_files(packages)]
    native = False
    built = compiled_modules(cfg.get("modules", []), fast=bool(cfg.get("fast", False)))
    for src, fs in built:
        files = [f for f in files if f[1] not in {a for _, a, _ in fs}]
        files += fs
        native = True
    if cfg.get("bundle-compiler"):
        files += [(p, arc, arc.endswith(("/bin/viv", "/bin/viv.exe"))) for p, arc in _bundle_files(_dist_name(project["name"]))]
        files += _py2viv_files(_dist_name(project["name"]))
        native = True
    if cfg.get("prebuild"):
        pkg = packages[0] if packages else _dist_name(project["name"])
        pre = prebuilt_files(project["name"], cfg, pkg)
        have = {a for _, a, _ in files}
        files += [f for f in pre if f[1] not in have]
        native = True
    return _write_wheel(wheel_directory, project, files, native, license_files(cfg))


def build_sdist(sdist_directory, config_settings=None):
    project, cfg, packages = _config()
    name, version = _dist_name(project["name"]), project["version"]
    base = "%s-%s" % (name, version)
    out = Path(sdist_directory) / (base + ".tar.gz")
    Path(sdist_directory).mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as t:
        def add(path: Path, arc: str):
            data = path.read_bytes()
            ti = tarfile.TarInfo("%s/%s" % (base, arc))
            ti.size, ti.mtime, ti.mode = len(data), int(time.time()), 0o644
            t.addfile(ti, io.BytesIO(data))

        add(Path("pyproject.toml"), "pyproject.toml")
        for p, arc in _package_files(packages):
            add(p, arc)
        lic = license_files(cfg)
        for p, arc in lic:
            add(p, arc)
        meta = _metadata(project, [a for _, a in lic]).encode()
        ti = tarfile.TarInfo("%s/PKG-INFO" % base)
        ti.size, ti.mtime, ti.mode = len(meta), int(time.time()), 0o644
        t.addfile(ti, io.BytesIO(meta))
    return out.name
