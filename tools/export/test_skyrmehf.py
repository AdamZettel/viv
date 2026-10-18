# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""Tests of the exported Skyrme Hartree-Fock program (lib/skyrme/skyrmehf.export.toml).

The default run is oxygen-16 with SLy5, the post's nucleus; the other
nuclei and forces are held to the numpy reference by the cases
(lib/skyrme/skyrmehf.cases.json, through the generated test and
tools/export/check.py) and by lib/skyrme/refs/compare.py.

    python3 tools/export/test_skyrmehf.py          (or python3 -m pytest tools/export/test_skyrmehf.py)

It runs `viv export` into target/export/skyrmehf ($VIV, default
target/release/viv), imports the generated wrapper (which builds the exact
library with cc) and checks:

  - the published numbers, through the library: the post's numpy twin
    (posts/refs/skyrme-hartree-fock-of-oxygen-16.out) on both of its meshes,
    and the post's extrapolated E/A, charge radius and orbital energies
    against skyrme_rpa's published run;
  - the bits: the default run's result hash is the one
    lib/skyrme/skyrmehf.exact_hashes.json records (tools/export/check.py
    --record wrote it after the library, the native build and the
    interpreter agreed), and at other settings the library, the native build
    and the interpreter (the parameter lines rewritten) give the same result bytes and the same
    printed lines;
  - the boundary: a bad parameter is refused before any work, with the
    parameter's name, and the next run gives the recorded bits; a run that
    does not settle is reported in the result, not raised;
  - repeated runs: the same bits each time, interleaved with other settings,
    and no growth of the process's peak memory;
  - the fast build: within 1e-10 MeV of the exact one.
"""
import json
import os
import re
import resource
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
VIV = os.environ.get("VIV", os.path.join(ROOT, "target", "release", "viv"))
MANIFEST = "lib/skyrme/skyrmehf.export.toml"
PROGRAM = os.path.join(ROOT, "lib", "skyrme", "skyrmehf.viv")
HASHES = os.path.join(ROOT, "lib", "skyrme", "skyrmehf.exact_hashes.json")
TWIN_OUT = os.path.join(ROOT, "posts", "refs", "skyrme-hartree-fock-of-oxygen-16.out")
OUT = os.path.join(ROOT, "target", "export", "skyrmehf")
REPORTS = OUT + "_reports"
TMP = os.path.join(ROOT, "target", "export", "check_tmp")

subprocess.run([VIV, "export", MANIFEST, "-o", OUT, "--reports", REPORTS], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
sys.path.insert(0, OUT)
sys.path.insert(0, HERE)
import check  # noqa: E402  (tools/export/check.py: result_hash, parse_report)
import skyrmehf  # noqa: E402

# ---------------------------------------------------------------- references


def twin():
    """The numpy twin's output: per mesh spacing, E (MeV), E/A, the radii and
    the levels [neutron 1s1/2 1p3/2 1p1/2, proton ...]; then its Richardson
    extrapolation and skyrme_rpa's published values beside it."""
    text = open(TWIN_OUT).read()
    num = r"(-?[0-9.]+)"
    mesh = {}
    for m in re.finditer(r"h = ([0-9.]+): (\d+) iterations  E = %s MeV  E/A = %s  r_p = %s  r_n = %s  r_ch = %s\n"
                         r"\s+neutron 1s1/2 1p3/2 1p1/2: %s  %s  %s\n\s+proton  1s1/2 1p3/2 1p1/2: %s  %s  %s"
                         % ((num,) * 11), text):
        g = [float(x) for x in m.groups()]
        mesh[g[0]] = dict(E=g[2], E_A=g[3], rp=g[4], rn=g[5], rc=g[6], levels=g[7:13])
    rich, pub = {}, {}
    for m in re.finditer(r"(E_A|rp|rn|rc): %s   skyrme_rpa sample: %s" % (num, num), text):
        rich[m.group(1)] = float(m.group(2))
        pub[m.group(1)] = float(m.group(3))
    for m in re.finditer(r"([np]) levels: %s  %s  %s   skyrme_rpa: %s  %s  %s" % ((num,) * 6), text):
        rich[m.group(1)] = [float(x) for x in m.groups()[1:4]]
        pub[m.group(1)] = [float(x) for x in m.groups()[4:7]]
    assert set(mesh) == {0.1, 0.05} and len(rich) == 6 == len(pub), "cannot read " + TWIN_OUT
    return mesh, rich, pub


TWIN, RICH, PUB = twin()
SLY5_BOX = 15.0
# the twin prints E/A and the radii to 1e-10, the levels to 1e-8 and E to
# 1e-8; the library test (lib/skyrme/skyrme.viv) holds the solver to the twin
# to 1e-7, the difference between dense eigh and inverse iteration
TWIN_TOL = 1.0e-7
TWIN_ITERATIONS = 25          # by Pulay mixing, the default (linear mixing at MIX 0.5 takes 38)
KEV = 1.0e-3


def flat_levels(r):
    return list(r["levels"].ravel())


DEFAULT = ""      # the default run's hash (the same as the case o16_sly5's)


def assert_twin(r, h):
    t = TWIN[h]
    got = [r["e_per_a"], r["r_p"], r["r_n"], r["r_ch"]] + flat_levels(r)
    want = [t["E_A"], t["rp"], t["rn"], t["rc"]] + t["levels"]
    worst = max(abs(a - b) for a, b in zip(got, want))
    assert worst < TWIN_TOL, (h, worst, got, want)
    # E is printed to 1e-8 and E/A to 1e-10: the total energy is E/A times 16
    assert abs(r["energy"] - t["E"]) < TWIN_TOL * 16, (r["energy"], t["E"])
    assert r["energy"] == r["e_per_a"] * 16


# ---------------------------------------------------------------- the published numbers


def test_the_default_run_is_the_numpy_twin_on_its_0p1_fm_mesh():
    r = skyrmehf.run()
    assert_twin(r, 0.1)
    assert r["iterations"] == TWIN_ITERATIONS and r["converged"] is True
    assert r["levels"].shape == (6,) and r["rho"].shape == r["tau"].shape == r["jso"].shape == (2, 150)
    assert r["mesh"][0] == 0.1 and len(r["mesh"]) == 150
    assert all(t[1] for t in r["tests"]) and len(r["tests"]) == 2, r["tests"]
    assert r["labels"] == "n 1s1/2,n 1p3/2,n 1p1/2,p 1s1/2,p 1p3/2,p 1p1/2" and (r["z"], r["n"]) == (8, 8)
    assert r["refills"] == 0


def test_spacing_0p05_is_the_numpy_twin_on_its_second_mesh():
    r = skyrmehf.run(spacing=0.05)
    assert_twin(r, 0.05)
    assert r["rho"].shape == (2, round(SLY5_BOX / 0.05))


def test_the_extrapolation_gives_the_posts_numbers_and_skyrme_rpas():
    a, b = skyrmehf.run(spacing=0.1), skyrmehf.run(spacing=0.05)

    def rich(x, y):
        return (4.0 * y - x) / 3.0
    e_a = rich(a["e_per_a"], b["e_per_a"])
    r_ch = rich(a["r_ch"], b["r_ch"])
    lev = [rich(x, y) for x, y in zip(flat_levels(a), flat_levels(b))]
    # the twin's extrapolation, printed to 1e-8 (E/A, radii) and 1e-4 (levels)
    assert abs(e_a - RICH["E_A"]) < TWIN_TOL, (e_a, RICH["E_A"])
    for key, x, y in (("rp", a["r_p"], b["r_p"]), ("rn", a["r_n"], b["r_n"]), ("rc", a["r_ch"], b["r_ch"])):
        assert abs(rich(x, y) - RICH[key]) < TWIN_TOL, key
    assert max(abs(x - y) for x, y in zip(lev, RICH["n"] + RICH["p"])) < 0.5e-4 + TWIN_TOL
    # the post's summary: 8.0250 MeV per nucleon and 2.7993 fm, against
    # skyrme_rpa's 8.0261 and 2.7991, and the six levels within 3 keV of it
    assert f"{-e_a:.4f}" == "8.0250" and f"{r_ch:.4f}" == "2.7993", (e_a, r_ch)
    assert f"{-PUB['E_A']:.4f}" == "8.0261" and f"{PUB['rc']:.4f}" == "2.7991"
    assert abs(e_a - PUB["E_A"]) < 1.2e-3
    assert max(abs(x - y) for x, y in zip(lev, PUB["n"] + PUB["p"])) < 3 * KEV


# ---------------------------------------------------------------- the bits


def test_the_default_run_has_the_recorded_bits():
    want = json.load(open(HASHES))[DEFAULT]
    assert check.result_hash(skyrmehf.run()) == want


def full_program(params, path):
    """The program with each parameter's top-level line set to the value
    given (as `--set` does, which takes only a number literal, not ALPHA's
    `1.0 / 6.0`) and the export's report appended: it prints every result in
    full, for `viv run` and a native build to be compared with the library."""
    import tomllib
    spec = tomllib.load(open(os.path.join(ROOT, MANIFEST), "rb"))["params"]
    text = open(PROGRAM).read()
    for k, v in params.items():
        lit = check.viv_literal(spec[k.upper()], v)
        text, n = re.subn(r"(?m)^(%s(?:: \w+)?) = .*$" % k.upper(), lambda m: "%s = %s" % (m.group(1), lit), text)
        assert n == 1, k
    with open(path, "w") as f:
        f.write(text + "\n" + open(os.path.join(REPORTS, "report.viv")).read())
    return path


OTHER_SETTINGS = [
    dict(spacing=0.05),
    dict(t0=-2490.0, w0=120.0, alpha=0.2, mix=0.4, e_tol=1.0e-10),
    dict(box=12.0, x3=1.3, max_iter=500),
    dict(max_iter=5),                 # stops before it settles
    dict(nucleus="48Ca", force="SIII", spacing=0.12),
    dict(nucleus="pb-208", force="skm*", j2=True, mix=0.4),
]


def test_library_native_and_interpreter_agree_at_other_settings():
    os.makedirs(TMP, exist_ok=True)
    for i, p in enumerate(OTHER_SETTINGS):
        src = full_program(p, os.path.join(TMP, "skyrmehf_set%d.viv" % i))
        r = skyrmehf.run(**p)
        h_lib = check.result_hash(r)
        c = os.path.join(TMP, "skyrmehf_set%d.c" % i)
        subprocess.run([VIV, src, "-o", c], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["cc", "-O2", "-ffp-contract=off", "-w", c, "-o", c[:-2], "-lm"], check=True)
        nat = subprocess.run([c[:-2]], capture_output=True, text=True)
        itp = subprocess.run([VIV, "run", src], cwd=ROOT, capture_output=True, text=True)
        for name, out in (("native", nat.stdout), ("interpreter", itp.stdout)):
            lines, res = check.parse_report(out, skyrmehf.RESULTS)
            # with the program as the main file, the engines also
            # print the debug lens's __story__ record, which the library's
            # build (the program as a --use file) does not
            lines = [x for x in lines if not x.startswith("__story__ ")]
            assert check.result_hash(dict(res, tests=None, log=None)) == h_lib, (p, name)
            assert lines == r["log"], (p, name, lines[:4], r["log"][:4])


# ---------------------------------------------------------------- the boundary

REFUSED = [
    (dict(spacing=0.0), ValueError, "spacing"),
    (dict(spacing=-0.1), ValueError, "spacing"),
    (dict(spacing=2.0), ValueError, "spacing"),
    (dict(box=1.0), ValueError, "box"),
    (dict(mix=0.0), ValueError, "mix"),
    (dict(mix=1.5), ValueError, "mix"),
    (dict(alpha=-0.1), ValueError, "alpha"),
    (dict(e_tol=-1.0e-9), ValueError, "e_tol"),
    (dict(max_iter=0), ValueError, "max_iter"),
    (dict(max_iter=2.5), ValueError, "max_iter"),
    (dict(max_iter="ten"), TypeError, "max_iter"),
    (dict(t0=float("nan")), ValueError, "t0"),
    (dict(w0=float("inf")), ValueError, "w0"),
    (dict(t3="13763"), TypeError, "t3"),
    (dict(z=8), TypeError, "z"),
    (dict(force="sly6"), ValueError, "force"),
    (dict(nucleus=208), TypeError, "nucleus"),
]


def test_bad_parameters_are_refused_before_any_work_and_the_next_run_is_unchanged():
    want = json.load(open(HASHES))[DEFAULT]
    for p, kind, name in REFUSED:
        log = []
        try:
            skyrmehf.run(log=log, **p)
        except kind as e:
            assert name in str(e), (p, e)
        else:
            raise AssertionError("not refused: %r" % p)
        assert log == [], p        # refused before the program ran: it prints four lines when it runs
        assert check.result_hash(skyrmehf.run()) == want, p


def test_a_run_that_does_not_settle_is_reported_not_raised():
    r = skyrmehf.run(max_iter=5)
    assert r["converged"] is False and r["iterations"] == 5
    assert [t[1] for t in r["tests"]] == [False, False]   # not settled; the start's normalization not yet mixed out



def test_open_shells_and_unknown_nuclei_stop_with_the_reason():
    for p, words in ((dict(nucleus="44Ca"), "neutrons: 24 is not a closed shell: the 1f7/2 orbital"),
                     (dict(nucleus="20Ne"), "10 is not a closed shell: the 1d5/2 orbital"),
                     (dict(nucleus="16Xx"), "'Xx' is not an element symbol"),
                     (dict(nucleus="O"), "write the mass number and the element"),
                     # no t3 repulsion: the nucleus collapses and its orbitals reorder
                     (dict(t3=0.0), "the iteration is running away")):
        try:
            skyrmehf.run(**p)
        except skyrmehf.VivError as e:
            assert words in str(e), (p, str(e).splitlines()[0])
        else:
            raise AssertionError("no error: %r" % p)
    assert check.result_hash(skyrmehf.run()) == json.load(open(HASHES))[DEFAULT]


def test_the_force_is_chosen_by_name_and_its_parameters_override_it():
    a = skyrmehf.run(nucleus="40Ca", force="SkM*")
    b = skyrmehf.run(nucleus="Ca40", force="skm*")
    assert check.result_hash(a) == check.result_hash(b)
    # SkM* by its numbers, starting from SLy5
    c = skyrmehf.run(nucleus="40Ca", force="sly5", t0=-2645.0, t1=410.0, t2=-135.0, t3=15595.0, x0=0.09, x1=0.0,
                     x2=0.0, x3=0.0, w0=130.0, alpha=1.0 / 6.0, j2=False)
    assert check.result_hash(a) == check.result_hash(c)
    assert a["labels"].split(",")[4] == "n 2s1/2" and a["converged"]


# ---------------------------------------------------------------- repeated runs


def peak_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def test_repeated_runs_give_the_same_bits_and_do_not_grow():
    want = json.load(open(HASHES))[DEFAULT]
    for _ in range(10):
        skyrmehf.run(spacing=0.05)
    before = peak_bytes()
    for i in range(60):
        if i % 3 == 0:
            skyrmehf.run(**OTHER_SETTINGS[i % len(OTHER_SETTINGS)])
        assert check.result_hash(skyrmehf.run()) == want, i
    grew = peak_bytes() - before
    assert grew < 4 << 20, grew


# ---------------------------------------------------------------- the fast build


def test_the_fast_build_is_within_1e_10_mev_of_the_exact_one():
    code = ("import json, sys; sys.path.insert(0, %r); import skyrmehf as s; r = s.run(); "
            "print(json.dumps([r['e_per_a'], r['energy'], r['r_ch']] + list(r['levels'].ravel())))" % OUT)
    env = dict(os.environ, SKYRMEHF_FAST="1")
    got = json.loads(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout)
    r = skyrmehf.run()
    want = [r["e_per_a"], r["energy"], r["r_ch"]] + flat_levels(r)
    assert max(abs(a - b) for a, b in zip(got, want)) < 1.0e-10, (got, want)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            t = time.time()
            f()
            print("ok %-90s %6.2f s" % (name, time.time() - t), flush=True)
