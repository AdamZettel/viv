# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vipyr: every listed function runs its documented example and returns
the documented fields; bad arguments are refused naming the parameter; the
specs and the site's copy are current; the docs have a page per function.

    python -m pytest python/tests/test_vipyr_programs.py

The first run compiles each program once (vivscf.c is 5 MB of C: about a
minute); later runs load the cached libraries.
"""
import json
import linecache
import os
import re
import subprocess
import textwrap
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import vipyr  # noqa: E402
import vipyr as vp  # noqa: E402
from vipyr import _spec  # noqa: E402

ROOT = _spec.repo_root()
SPECS = _spec.load_specs()["functions"]
BUILT = [q for q, s in SPECS.items() if s["kind"] != "coming"]
COMING = [q for q, s in SPECS.items() if s["kind"] == "coming"]


def fn(qual):
    family, name = qual.split(".")
    return getattr(getattr(vipyr, family), name)


@pytest.mark.parametrize("qual", BUILT)
def test_the_example_runs_and_returns_the_documented_fields(qual):
    s = SPECS[qual]
    ns = {}
    name = "<example %s>" % qual
    # as python -m vipyr.docs runs it: in linecache, so that py2viv can read a function the example defines
    linecache.cache[name] = (len(s["example"]), None, s["example"].splitlines(True), name)
    exec(compile(s["example"], name, "exec"), ns)
    r = ns["r"]
    assert type(r).__name__ == s["result"]
    assert [f for f in r._fields] == [x["name"] for x in s["results"]]
    for x in s["results"]:
        assert hasattr(r, x["name"])
    assert re.match(r"^rr-[0-9a-f]{16}$", r.run_id)
    assert r.record["id"] == r.run_id and r.record["program"]["module"] == "vipyr." + s["family"]
    rep = r.check()
    assert rep.passed, rep
    with pytest.raises(dataclass_frozen_error()):
        setattr(r, r._fields[0], None)


def dataclass_frozen_error():
    import dataclasses
    return dataclasses.FrozenInstanceError


@pytest.mark.parametrize("qual", BUILT)
def test_the_signature_and_docstring_come_from_the_spec(qual):
    import inspect
    f, s = fn(qual), SPECS[qual]
    assert [p for p in inspect.signature(f).parameters] == [p["name"] for p in s["params"]]
    for p in s["params"]:
        assert p["name"] in f.__doc__
    assert s["summary"] and s["summary"].split()[0] in f.__doc__


@pytest.mark.parametrize("qual", COMING)
def test_a_coming_function_says_so(qual):
    with pytest.raises(NotImplementedError, match="listed, not built"):
        fn(qual)()


@pytest.mark.parametrize("call, param", [
    (lambda: vp.scf.run(basis="sto-6g"), "basis"),
    (lambda: vp.scf.run(charge=40), "charge"),
    (lambda: vp.scf.run(atoms=[8, 1, 1]), "coords"),
    (lambda: vp.scf.run(atoms=[8, 1, 1], coords=np.zeros((2, 3))), "coords"),
    (lambda: vp.scf.run(atoms=[8, 1, 30], coords=np.zeros((3, 3))), "atoms"),
    (lambda: vp.dft.pw(material="xx"), "material"),
    (lambda: vp.dft.pw(kmesh=0), "kmesh"),
    (lambda: vp.dft.pw(ecut=0.1), "ecut"),
    (lambda: vp.nuclear.skyrme_hf(box=1.0), "box"),
    (lambda: vp.nuclear.skyrme_hf(spacing=0.0), "spacing"),
    (lambda: vp.nuclear.skyrme_hf(mixing="anderson"), "mixing"),
    (lambda: vp.plasma.landau(alpha=0.5), "alpha"),
    (lambda: vp.plasma.landau(k=float("nan")), "k"),
    (lambda: vp.special.boys(n_max=17, T=1.0), "n_max"),
    (lambda: vp.special.boys(n_max=2, T=-1.0), "T"),
    (lambda: vp.linalg.schur(A=np.ones((2, 3))), "A"),
    (lambda: vp.linalg.schur(A=np.ones(3)), "A"),
    (lambda: vp.features.sessions(n=0), "n"),
])
def test_a_bad_value_names_the_parameter(call, param):
    with pytest.raises(vipyr.ParameterError) as e:
        call()
    assert e.value.param == param
    assert re.search(r"\b%s\b" % re.escape(param), str(e.value))


@pytest.mark.parametrize("call, word", [
    (lambda: vp.scf.run(no_such=1), "no_such"),
    (lambda: vp.scf.run(charge=1.5), "charge"),
    (lambda: vp.dft.pw(kmesh="3"), "kmesh"),
    (lambda: vp.nuclear.skyrme_hf(converged=True), "converged"),
    (lambda: vp.special.boys(2), "T"),
    (lambda: vp.finance.bs_price(S=1.0, K=1.0, T=1.0, r=0.0, q=0.0, sigma=0.2, is_call=1), "is_call"),
])
def test_a_wrong_type_or_name_is_a_type_error_naming_it(call, word):
    with pytest.raises(TypeError) as e:
        call()
    assert word in str(e.value)


def test_kmesh_reaches_the_program_as_nkmesh():
    s = SPECS["dft.pw"]
    p = [p for p in s["params"] if p["name"] == "kmesh"][0]
    assert p["alias_of"] == "nkmesh"
    assert "nkmesh" not in [q["name"] for q in s["params"]]


def test_a_rerun_has_the_same_run_id_and_records_append(tmp_path):
    log = tmp_path / "runs.jsonl"
    vipyr.record_runs(log)
    try:
        a = vp.nuclear.skyrme_hf()
        b = vp.nuclear.skyrme_hf()
        c = vp.nuclear.skyrme_hf(spacing=0.2)
    finally:
        vipyr.record_runs(None)
    assert a.run_id == b.run_id != c.run_id
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    assert [r["id"] for r in recs] == [a.run_id, b.run_id, c.run_id]
    assert "mismatch_of" not in recs[1]


def test_check_at_other_settings_says_so():
    r = vp.nuclear.skyrme_hf(spacing=0.2)
    rep = r.check()
    assert not rep.at_deck_settings and rep.hash == {}
    d = vp.nuclear.skyrme_hf().check()
    assert d.at_deck_settings and d.hash["match"]


def test_skyrme_mixers_reach_the_same_nucleus():
    """Calcium-40 with SLy4 at e_tol 1e-13: linear, Pulay and Broyden
    mixing agree to 2e-14 MeV in E/A and 2e-13 fm in r_ch
    (gallery/science/plot_scf_mixers.py); Pulay takes fewer passes."""
    runs = {m: vp.nuclear.skyrme_hf(nucleus="40Ca", force="SLy4", e_tol=1e-13, max_iter=1000, mixing=m)
            for m in ("linear", "pulay", "broyden")}
    ref = runs["linear"]
    for r in runs.values():
        assert r.converged
        assert abs(r.e_per_a - ref.e_per_a) < 1e-12 and abs(r.r_ch - ref.r_ch) < 1e-11
    assert runs["pulay"].iterations < ref.iterations


def test_scf_matches_the_published_example():
    """Water in cc-pVDZ: PySCF 2.5.0 gives -76.02676567311998 hartree
    (study/ports/pyscf/viv/water_ccpvdz.viv, held there to 1e-10)."""
    r = vp.scf.run(basis="cc-pvdz")
    assert abs(r.e_tot - (-76.02676567311998)) < 1e-10
    assert r.mo_energy.shape == (24,) and r.mo_coeff.shape == (24, 24) and r.dm.shape == (24, 24)
    u = vp.scf.run(method="UHF", charge=1, spin=1)
    assert u.mo_energy.ndim == 2 and u.converged


def test_the_specs_are_current():
    """vipyr/spec.json and www/src/data/vipyr.json against a fresh derivation."""
    from vipyr import docs
    assert docs.main(["--check"]) == 0


def test_the_docs_have_a_page_per_function():
    site = json.loads((ROOT / "www" / "src" / "data" / "vipyr.json").read_text())
    assert set(site["functions"]) == set(SPECS)
    page = (ROOT / "www" / "src" / "pages" / "vipyr" / "[family]" / "[name].astro").read_text()
    assert "getStaticPaths" in page and "vipyr.json" in page
    for q, s in site["functions"].items():
        assert s["signature"].startswith("vp." + q)
        assert s["summary"]
        if s["kind"] != "coming":
            assert s["example"] and s["results"] and s["reference"]
            for p in s["params"]:
                assert "type" in p and "range" in p and "help" in p and "default" in p
    families = {s["family"] for s in site["functions"].values()}
    assert families == {f["id"] for f in site["fields"]}
    dist = ROOT / "www" / "dist" / "vipyr"
    if dist.is_dir():                           # after `npm run build`
        for q in site["functions"]:
            assert (dist / q.replace(".", "/") / "index.html").is_file(), q


# ------------------------------------------------------------ distributions

def test_the_checkout_finds_every_distribution_without_an_install():
    from vipyr import _packages, _registry
    fams = set(vp.families())
    assert set(_registry.FAMILY_DIST) <= fams
    for fam, dist in _registry.FAMILY_DIST.items():
        if dist != "vipyr":
            assert _packages.registered()[fam].startswith(_registry.DISTS[dist]["module"])


def test_a_missing_distribution_names_its_extra():
    code = textwrap.dedent("""
        import vipyr as vp
        assert "scf" not in vp.families()
        try:
            vp.scf
        except ImportError as e:
            print(e)
        """)
    env = dict(os.environ, VIPYR_DEV="0", PYTHONPATH=str(HERE.parent))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "vp.scf needs vipyr-chem: pip install vipyr[chem]"


def test_each_distribution_ships_only_its_own_families():
    from vipyr import _registry
    for dist in _registry.DISTS:
        p = _spec.spec_file(dist)
        d = json.loads(p.read_text())
        assert {f["dist"] for f in d["fields"]} == {dist}, p
        assert all(_registry.FAMILY_DIST[s["family"]] == dist for s in d["functions"].values()), p


def test_the_wheel_metadata_carries_extras_and_entry_points():
    from vipyr import build
    core = build.read_pyproject(HERE.parent / "pyproject.toml")["project"]
    meta = build._metadata(core)
    assert 'Requires-Dist: vipyr-chem==0.0.1; extra == "chem"' in meta and "Provides-Extra: all" in meta
    chem = build._mini_toml((ROOT / "packages" / "vipyr-chem" / "pyproject.toml").read_text())["project"]
    assert "[vipyr.packages]\nscf = vipyr_chem.scf\ndft = vipyr_chem.dft\n" in build._entry_points(chem)
