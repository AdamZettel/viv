# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vipyr end to end: load, the cache, inference, the import hook, stubs,
errors, NumPy arguments, the build backend and the platform wheel.

    VIV=target/release/viv python3 -m pytest -q python/tests

Each test writes its .viv files into its own temporary directory, so the
cache under __pycache__/vipyr starts empty. The wheel tests make a fresh
venv (`--system-site-packages`, for NumPy; nothing is downloaded) and run
it with PATH holding only the venv's bin: no viv, no C compiler.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import numpy as np
import pytest

PYDIR = Path(__file__).resolve().parents[1]
REPO = PYDIR.parent
sys.path.insert(0, str(PYDIR))

import vipyr  # noqa: E402
from vipyr import _core  # noqa: E402

KERN = textwrap.dedent('''\
    import minpack.enorm
    import helper

    #| The sum of squares of x.
    def sumsq(x: Arr) -> Num {
        s = 0.0
        for v in x {
            s += v * v
        }
        return s
    }

    #| MINPACK's overflow-safe norm.
    def safe_norm(x: Arr) -> Num {
        return minpack.enorm.enorm(len(x), x)
    }

    def scaled(x: Arr, a: Num) -> Arr {
        return helper.times(x, a)
    }

    def minmax(x: Arr) {
        return min(x), max(x)
    }

    def count_up(n: Int) -> [Int] {
        out: [Int] = []
        for i to n {
            out = append(out, i)
        }
        return out
    }

    def pick(x: Arr, i: Int) -> Num {
        return x[i]
    }

    def checked(x: Num) -> Num {
        if x < 0.0 {
            stop("checked: x must not be negative")
        }
        return sqrt(x)
    }

    def untyped(x) {
        return x
    }
    ''')

HELPER = textwrap.dedent('''\
    def times(x: Arr, a: Num) -> Arr {
        return a * x
    }
    ''')


def env_with_python():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PYDIR) + os.pathsep + env.get("PYTHONPATH", "")
    return env


@pytest.fixture
def kern(tmp_path):
    (tmp_path / "kern.viv").write_text(KERN)
    (tmp_path / "helper.viv").write_text(HELPER)
    return tmp_path


def run_py(code, cwd, env=None):
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=cwd, capture_output=True, text=True,
                       env=env or env_with_python())
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


# ------------------------------------------------------------------ load

def test_load_calls_each_annotated_def(kern):
    k = vipyr.load(kern / "kern.viv")
    assert k.__all__ == ["sumsq", "safe_norm", "scaled", "minmax", "count_up", "pick", "checked"]
    assert k.sumsq(np.array([1.0, 2.0, 3.0])) == 14.0
    assert k.safe_norm([3.0, 4.0]) == 5.0
    assert np.array_equal(k.scaled(np.arange(3.0), 2.0), [0.0, 2.0, 4.0])
    assert k.minmax(np.array([3.0, -1.0, 5.0])) == (-1.0, 5.0)
    c = k.count_up(4)
    assert c.dtype == np.int64 and c.tolist() == [0, 1, 2, 3]
    assert k.__file__ == str((kern / "kern.viv").resolve())
    assert "The sum of squares of x." in k.sumsq.__doc__


def test_inference_leaves_out_unannotated_defs_with_a_note(kern):
    k = vipyr.load(kern / "kern.viv")
    assert not hasattr(k, "untyped")
    notes = k.__vipyr__.notes
    assert any("`untyped`" in n and "has no type" in n for n in notes), notes


def test_a_file_with_nothing_to_export_is_a_compile_error(tmp_path):
    (tmp_path / "bare.viv").write_text("def f(x) {\n    return x\n}\n")
    with pytest.raises(vipyr.VivCompileError, match="no def can be exported"):
        vipyr.load(tmp_path / "bare.viv")


def test_run_time_errors_are_python_exceptions(kern):
    k = vipyr.load(kern / "kern.viv")
    with pytest.raises(IndexError) as ei:
        k.pick(np.zeros(3), 7)
    assert isinstance(ei.value, k.VivError)
    with pytest.raises(k.VivStop, match="must not be negative"):
        k.checked(-1.0)
    with pytest.raises(TypeError):
        k.pick(np.zeros(3), "a")
    assert k.checked(4.0) == 2.0           # the next call works


def test_contiguous_float64_arguments_are_not_copied(kern):
    k = vipyr.load(kern / "kern.viv")
    x = np.arange(6.0)
    assert k._check("x", "Arr", None, None, None, None, True, x) is x
    strided = np.arange(12.0)[::2]
    y = k._check("x", "Arr", None, None, None, None, True, strided)
    assert y is not strided and y.flags["C_CONTIGUOUS"]
    ints = np.arange(4)
    assert k._check("x", "Arr", None, None, None, None, True, ints).dtype == np.float64


def test_a_read_only_array_argument_is_lent_not_copied(tmp_path):
    """The C entry lends a def the caller's buffer when nothing in the def
    writes it (export_kernels::borrowable), and copies it when the def
    writes: the copy shows as a peak-RSS jump of the array's size."""
    (tmp_path / "lend.viv").write_text(textwrap.dedent('''\
        def total(a: Arr) -> Num {
            s = 0.0
            for i to len(a) {
                s += a[i]
            }
            return s
        }

        def total_w(a: Arr) -> Num {
            a[0] = a[0]
            s = 0.0
            for i to len(a) {
                s += a[i]
            }
            return s
        }
        '''))
    out = run_py('''
        import resource, sys, numpy as np, vipyr
        k = vipyr.load("lend.viv")
        x = np.ones(30_000_000)          # 240 MB
        k.total(np.ones(3)); k.total_w(np.ones(3))
        def peak():
            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return r if sys.platform == "darwin" else r * 1024
        p0 = peak(); s = k.total(x); p1 = peak(); w = k.total_w(x); p2 = peak()
        print(s == w == 3.0e7, p1 - p0, p2 - p1, x[0] == 1.0)
        ''', tmp_path)
    same, lent_growth, copied_growth, untouched = out.split()
    assert same == "True" and untouched == "True"
    assert int(lent_growth) < 20_000_000, out          # no 240 MB copy
    assert int(copied_growth) > 200_000_000, out       # the writer's copy


def test_fast_builds_its_own_entry(kern):
    exact = vipyr.load(kern / "kern.viv")
    fast = vipyr.load(kern / "kern.viv", fast=True)
    assert fast.__vipyr__.path != exact.__vipyr__.path
    assert fast.__vipyr__.info["flags"][:3] == _core.FAST_FLAGS
    assert fast.sumsq(np.array([1.0, 2.0])) == 5.0
    # a fast build links the BLAS `viv build --fast` links (Accelerate on macOS)
    if sys.platform == "darwin":
        assert "-DVIV_BLAS" in fast.__vipyr__.info["flags"]
    assert "-DVIV_BLAS" not in exact.__vipyr__.info["flags"]


def test_a_manifest_beside_the_file_is_used_instead_of_inference(tmp_path):
    (tmp_path / "ranged.viv").write_text("def half(x: Num) -> Num {\n    return 0.5 * x\n}\n\n"
                                         "def other(x: Num) -> Num {\n    return x\n}\n")
    (tmp_path / "ranged.export.toml").write_text(textwrap.dedent('''\
        name = "ranged"
        kind = "kernels"
        files = ["ranged.viv"]

        [kernels.half]
        viv = "ranged.viv:half"
        args = [{ name = "x", type = "Num", min = 0.0 }]
        results = [{ name = "y", type = "Num" }]
        '''))
    k = vipyr.load(tmp_path / "ranged.viv")
    assert k.__all__ == ["half"]
    assert k.half(3.0) == 1.5
    with pytest.raises(ValueError, match="at least 0"):
        k.half(-1.0)
    # the manifest is part of the key
    n = _core.stats["exports"]
    vipyr.load(tmp_path / "ranged.viv")
    assert _core.stats["exports"] == n
    (tmp_path / "ranged.export.toml").write_text((tmp_path / "ranged.export.toml").read_text().replace("min = 0.0", "min = 1.0"))
    with pytest.raises(ValueError, match="at least 1"):
        vipyr.load(tmp_path / "ranged.viv").half(0.5)
    assert _core.stats["exports"] == n + 1


# ----------------------------------------------------------------- cache

def test_the_cache_hits_and_misses(kern):
    src = kern / "kern.viv"
    before = dict(_core.stats)
    vipyr.load(src)
    assert _core.stats["exports"] == before["exports"] + 1
    vipyr.load(src)
    assert _core.stats["exports"] == before["exports"] + 1
    assert _core.stats["cache_hits"] == before["cache_hits"] + 1
    assert (kern / "__pycache__" / "vipyr").is_dir()
    # a fresh process finds the entry and does not export or compile
    out = run_py("""
        import vipyr, json
        k = vipyr.load("kern.viv")
        print(json.dumps(vipyr.cache_stats))
        """, kern)
    assert json.loads(out.splitlines()[-1]) == {"exports": 0, "compiles": 0, "cache_hits": 1}
    # an edit of the file is a miss
    src.write_text(KERN.replace("s += v * v", "s += 2.0 * v * v"))
    k = vipyr.load(src)
    assert _core.stats["exports"] == before["exports"] + 2
    assert k.sumsq(np.array([1.0])) == 2.0
    # so is an edit of a module it imports
    (kern / "helper.viv").write_text(HELPER.replace("a * x", "a * x + 1.0"))
    k = vipyr.load(src)
    assert _core.stats["exports"] == before["exports"] + 3
    assert np.array_equal(k.scaled(np.zeros(2), 3.0), [1.0, 1.0])


def test_a_module_created_where_an_import_looked_first_is_a_miss(kern):
    vipyr.load(kern / "kern.viv")
    n = _core.stats["exports"]
    # `import minpack.enorm` looked beside the file first: a module there now
    # makes the name ambiguous, which the compiler refuses
    (kern / "minpack").mkdir()
    (kern / "minpack" / "enorm.viv").write_text("def enorm(n: Num, x: Arr) -> Num {\n    return -1.0\n}\n")
    with pytest.raises(vipyr.VivCompileError, match="finds ./minpack/enorm.viv and lib/minpack/enorm.viv"):
        vipyr.load(kern / "kern.viv")
    assert _core.stats["exports"] == n + 1


def test_two_processes_building_at_once_agree(kern):
    procs = [subprocess.Popen([sys.executable, "-c", "import vipyr; print(vipyr.load('kern.viv').sumsq([2.0]))"],
                              cwd=kern, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env_with_python())
             for _ in range(3)]
    outs = [p.communicate() for p in procs]
    assert all(p.returncode == 0 for p in procs), outs
    assert all(o[0].strip() == "4.0" for o in outs), outs
    slots = [d for d in (kern / "__pycache__" / "vipyr").iterdir() if d.is_dir()]
    assert len(slots) == 1
    assert len([d for d in slots[0].iterdir() if d.is_dir()]) == 1     # one entry, no leftover build dirs


# ------------------------------------------------------------- the hook

def test_the_import_hook_finds_files_and_packages_and_does_not_recompile(kern):
    pkg = kern / "pkg" / "sub"
    pkg.mkdir(parents=True)
    (kern / "pkg" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "kernels.viv").write_text("#| Twice x.\ndef twice(x: Num) -> Num {\n    return 2.0 * x\n}\n")
    code = """
        import json, vipyr.auto, vipyr
        import kern
        from pkg.sub import kernels
        assert kern.sumsq([1.0, 2.0]) == 5.0
        assert kernels.twice(4.0) == 8.0
        assert kernels.__name__ == "pkg.sub.kernels"
        assert kernels.__file__.endswith("kernels.viv")
        print(json.dumps(vipyr.cache_stats))
        """
    first = json.loads(run_py(code, kern).splitlines()[-1])
    assert first["exports"] == 2
    second = json.loads(run_py(code, kern).splitlines()[-1])
    assert second == {"exports": 0, "compiles": 0, "cache_hits": 2}


def test_a_py_module_of_the_same_name_wins(kern):
    (kern / "kern.py").write_text("WHO = 'python'\n")
    out = run_py("import vipyr.auto, kern; print(kern.WHO)", kern)
    assert out.strip() == "python"


def test_a_compile_error_is_a_syntax_error_at_the_viv_line(tmp_path):
    (tmp_path / "broken.viv").write_text("def f(x: Num) -> Num {\n    y = x +* 2.0\n    return y\n}\n")
    out = run_py("""
        import vipyr.auto
        try:
            import broken
        except SyntaxError as e:
            print(repr((e.filename.rsplit("/", 1)[-1], e.lineno, e.offset, e.text)))
            print(type(e).__name__)
        """, tmp_path)
    loc, kind = out.strip().splitlines()[-2:]
    name, line, col, text = eval(loc)
    assert (name, line) == ("broken.viv", 2) and col is not None and "x +* 2.0" in text
    assert kind == "VivCompileError"


def test_a_type_error_is_a_syntax_error_in_the_imported_module(kern):
    (kern / "helper.viv").write_text("def times(x: Arr, a: Num) -> Arr {\n    return a * x\n}\n\nbad = \"s\" - 1.0\n")
    with pytest.raises(SyntaxError) as ei:
        vipyr.load(kern / "kern.viv")
    assert ei.value.filename.endswith("helper.viv") and ei.value.lineno == 5


# ----------------------------------------------------------------- stubs

def test_stubs_are_written_beside_the_source(kern):
    r = subprocess.run([sys.executable, "-m", "vipyr", "stubs", "kern.viv"], cwd=kern, capture_output=True, text=True,
                       env=env_with_python())
    assert r.returncode == 0, r.stderr
    pyi = (kern / "kern.pyi").read_text()
    assert "def sumsq(x: npt.ArrayLike) -> float:" in pyi
    assert "def minmax(x: npt.ArrayLike) -> Tuple[float, float]:" in pyi
    assert "def count_up(n: int) -> npt.NDArray[np.int64]:" in pyi
    assert "The sum of squares of x." in pyi
    compile(pyi, "kern.pyi", "exec")          # a stub is valid Python
    assert vipyr.load(kern / "kern.viv").__vipyr__.pyi.is_file()     # and one sits in the cache


# ---------------------------------------------------------- the wheels

def _venv(path: Path) -> Path:
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", "--without-pip", str(path)], check=True)
    return path / "bin" / "python"


def _install(py: Path, wheel: Path):
    """Unpack a wheel into the venv's site-packages: what pip does for these
    wheels (no compiled byte code, no scripts beyond the console entry)."""
    site = subprocess.run([str(py), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                          capture_output=True, text=True, check=True).stdout.strip()
    with zipfile.ZipFile(wheel) as z:
        for zi in z.infolist():
            z.extract(zi, site)
            mode = zi.external_attr >> 16
            if mode & 0o111:
                os.chmod(os.path.join(site, zi.filename), 0o755)
    return Path(site)


def _bare_env(venv_py: Path):
    """PATH with only the venv's bin: no viv, no cc."""
    return {"PATH": str(venv_py.parent), "HOME": os.environ.get("HOME", "/tmp")}


def _pip_install(py: Path, wheel: Path) -> bool:
    """pip, when the venv can use the system's (offline, --no-index)."""
    r = subprocess.run([str(py), "-m", "pip", "--version"], capture_output=True, text=True)
    if r.returncode != 0:
        return False
    r = subprocess.run([str(py), "-m", "pip", "install", "--no-index", "--no-deps", "-q", str(wheel)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return True


def test_a_library_wheel_runs_without_viv_or_a_c_compiler(kern, tmp_path_factory):
    proj = tmp_path_factory.mktemp("proj")
    (proj / "vivdemo").mkdir()
    (proj / "vivdemo" / "__init__.py").write_text("from . import kernels\n")
    (proj / "LICENSE").write_text("MIT License (a fixture)\n")
    (proj / "vivdemo" / "kernels.viv").write_text(KERN.replace("import helper\n", "").replace(
        "return helper.times(x, a)", "return a * x"))
    (proj / "pyproject.toml").write_text(textwrap.dedent('''\
        [build-system]
        requires = ["vipyr"]
        build-backend = "vipyr.build"

        [project]
        name = "vivdemo"
        version = "0.1"
        license = "MIT"
        dependencies = ["numpy"]

        [tool.vipyr]
        packages = ["vivdemo"]
        modules = ["vivdemo/kernels.viv"]
        '''))
    dist = tmp_path_factory.mktemp("dist")
    r = subprocess.run([sys.executable, "-c", "import vipyr.build as b, sys; print(b.build_wheel(sys.argv[1]))", str(dist)],
                       cwd=proj, capture_output=True, text=True, env=env_with_python())
    assert r.returncode == 0, r.stderr
    wheel = dist / r.stdout.strip().splitlines()[-1]
    assert wheel.name.startswith("vivdemo-0.1-py3-none-") and not wheel.name.endswith("-any.whl")
    names = zipfile.ZipFile(wheel).namelist()
    assert "vivdemo/kernels.py" in names and "vivdemo/kernels.pyi" in names
    assert any(n.startswith("vivdemo/libkernels.") for n in names)
    assert not any(n.endswith(".c") for n in names)
    assert "vivdemo-0.1.dist-info/licenses/LICENSE" in names
    meta = zipfile.ZipFile(wheel).read("vivdemo-0.1.dist-info/METADATA").decode()
    assert "License: MIT\n" in meta and "License-File: LICENSE\n" in meta
    py = _venv(tmp_path_factory.mktemp("venv") / "v")
    if not _pip_install(py, wheel):
        _install(py, wheel)
    out = subprocess.run([str(py), "-c", textwrap.dedent("""
        import importlib.util, shutil
        assert importlib.util.find_spec("vipyr") is None, "vipyr is importable"
        assert shutil.which("viv") is None and shutil.which("cc") is None
        from vivdemo import kernels
        print(kernels.sumsq([1.0, 2.0]), kernels.safe_norm([3.0, 4.0]), kernels.minmax([2.0, 7.0]))
        """)], cwd=tmp_path_factory.mktemp("elsewhere"), capture_output=True, text=True, env=_bare_env(py))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "5.0 5.0 (2.0, 7.0)"


@pytest.mark.skipif(os.environ.get("VIPYR_SKIP_PLATFORM_WHEEL") == "1", reason="VIPYR_SKIP_PLATFORM_WHEEL=1")
def test_the_platform_wheel_carries_the_compiler_and_the_library(tmp_path_factory):
    dist = tmp_path_factory.mktemp("vdist")
    r = subprocess.run([sys.executable, "-c", "import vipyr.build as b, sys; print(b.build_wheel(sys.argv[1]))", str(dist)],
                       cwd=PYDIR, capture_output=True, text=True, env=env_with_python())
    assert r.returncode == 0, r.stderr
    wheel = dist / r.stdout.strip().splitlines()[-1]
    names = zipfile.ZipFile(wheel).namelist()
    assert "vipyr/bin/viv" in names and "vipyr/_root/lib/minpack/enorm.viv" in names
    # the compiler links SQLite and Viv's libm: their notices travel with it
    lic = "vipyr-0.0.1.dist-info/licenses/"
    for f in ["LICENSE-MIT", "LICENSE-APACHE", "LICENSING.md", "THIRD_PARTY.md", "LICENSES/blessing.txt",
              "LICENSES/SunPro.txt", "LICENSES/MIT.txt", "LICENSES/OFL-1.1.txt", "third_party/sqlite/README.md"]:
        assert lic + f in names, f
    py = _venv(tmp_path_factory.mktemp("venv") / "v")
    if not _pip_install(py, wheel):
        _install(py, wheel)
    work = tmp_path_factory.mktemp("work")
    (work / "kern.viv").write_text(KERN)
    (work / "helper.viv").write_text(HELPER)
    env = _bare_env(py)
    env["CC"] = os.environ.get("CC", "/usr/bin/cc")       # building a module needs a C compiler; the wheel brings viv
    out = subprocess.run([str(py), "-c", textwrap.dedent("""
        import vipyr, shutil, subprocess, sys
        w = shutil.which("viv")                   # the wheel's `viv` command, nothing else
        assert w is not None and w.startswith(sys.prefix), w
        assert vipyr.compiler().startswith(vipyr.__path__[0])
        assert subprocess.run([w, "check", "kern.viv"], capture_output=True).returncode == 0
        k = vipyr.load("kern.viv")
        print(k.safe_norm([3.0, 4.0]), k.scaled([1.0], 2.0)[0])
        """)], cwd=work, capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "5.0 2.0"
