# SPDX-License-Identifier: MIT OR Apache-2.0
"""viv-xc against libxc 6.1.0 (the copy PySCF 2.5 ships).

    python3 xc/tests/oracle.py [--smoke] [--only a,b] [--json out.json]

For each translated functional (xc/catalog.json), evaluates the Viv module
(compiled through vipyr) and libxc's xc_lda_exc_vxc / xc_gga_exc_vxc on the
same points, unpolarized and spin-polarized, and compares zk, vrho, vsigma.

Error measure: |viv - libxc| / max(|libxc|, |libxc| of the same output's
other spin components at that point, FLOOR * scale), where scale is the
largest |libxc| of that output over the point's density decade: an output
that is analytically zero (vrho_b of a fully polarized density) is judged
against the size of its siblings, not against zero.

A point passes when every output is within RTOL; or within twice libxc's own
spread: the most libxc's output moves when its inputs move by up to 4 ulps
(NPERT runs). That spread is large only where libxc's result is decided by
rounding, e.g. at zeta = +-1 after libxc clamps the empty spin to
dens_threshold, where (1 - zeta)^(-1/3) terms meet 1 - zeta of a few ulps.

An edge point (one spin density below EDGE times the total, or a zero
sigma_ss, which libxc clamps to sigma_threshold^2 so that d/dsigma carries
1/sqrt(sigma_threshold^2) = 1e20 times a rounding-sized numerator) that still
disagrees is checked against the translated formula at 60 digits
(tools/vivxc/exact.py, stored in xc/tests/exact.json by --refresh-exact): it
passes only if Viv is at least as close to that value as libxc is. Points
away from the edge get no fallback, so a translation error shows there.

Status per functional: PASS (every point passes), EDGE (every failing point
is an edge point; the count where libxc itself is off the 60-digit value by
more than RTOL is reported beside it), FAIL (a point away from the edge fails).
xc/tests/expected.json records the statuses; --check fails on any change."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XC = os.path.dirname(HERE)
ROOT = os.path.dirname(XC)
sys.path.insert(0, os.path.join(ROOT, "tools", "vivxc"))
sys.path.insert(0, os.path.join(ROOT, "python"))

import libxc as X  # noqa: E402

RTOL = 1e-10
FLOOR = 1e-10
NPERT = 6          # libxc runs on inputs moved by a few ulps, for its own spread
ULP_MOVE = 4.0 * 2.0 ** -52
EDGE = 1e-6        # a point is an edge point when min(rho_a, rho_b) < EDGE * (rho_a + rho_b)
SMOKE = ["lda_x", "lda_c_pw", "lda_c_vwn", "gga_x_pbe", "gga_c_pbe", "gga_x_b88", "gga_c_lyp", "gga_x_pw91"]

RHO = np.logspace(-10, 4, 15)                      # total density
S = np.array([0.0, 1e-3, 0.1, 0.5, 1.0, 2.5, 8.0])   # reduced gradient s = |grad n| / (2 (3 pi^2)^(1/3) n^(4/3))
ZETA = np.array([0.0, 0.3, -0.6, 0.9, -0.99, 1.0 - 1e-6, -1.0 + 1e-9, 1.0, -1.0])
CAB = np.array([1.0, -0.4])                        # sigma_ab / sqrt(sigma_aa sigma_bb)
KF = 2.0 * (3.0 * math.pi ** 2) ** (1.0 / 3.0)


def points(fam, pol):
    """-> rho (np,) or (np, 2), sigma (np,) or (np, 3), decade label per point"""
    rows_r, rows_s, dec = [], [], []
    for k, n in enumerate(RHO):
        svals = S if fam == "gga" else [0.0]
        if not pol:
            for s in svals:
                rows_r.append(n)
                rows_s.append((s * KF * n ** (4.0 / 3.0)) ** 2)
                dec.append(k)
            continue
        for z in ZETA:
            ra, rb = n * (1 + z) / 2, n * (1 - z) / 2
            for s in svals:
                for c in (CAB if fam == "gga" else [1.0]):
                    saa = (s * KF * ra ** (4.0 / 3.0)) ** 2
                    sbb = (s * 0.7 * KF * rb ** (4.0 / 3.0)) ** 2
                    rows_r.append([ra, rb])
                    rows_s.append([saa, c * math.sqrt(saa * sbb), sbb])
                    dec.append(k)
    return np.array(rows_r, dtype=float), np.array(rows_s, dtype=float), np.array(dec)


def scaled_error(ref, got, dec):
    """|got - ref| / den for one output (np, k), den as the module docstring says."""
    scale = np.zeros(ref.shape[0])
    fin_abs = np.where(np.isfinite(ref), np.abs(ref), 0.0)
    for k in np.unique(dec):
        m = dec == k
        scale[m] = np.max(fin_abs[m])
    den = np.maximum(np.abs(ref), FLOOR * scale[:, None])
    den = np.maximum(den, np.max(fin_abs, axis=1, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.abs(got - ref) / np.where(den > 0, den, 1.0)
    same = (ref == got) | (np.isnan(ref) & np.isnan(got))
    e = np.where(same, 0.0, e)
    return np.where(np.isnan(e), np.inf, e), den


def screen(fam, pol, r, s, dt):
    """libxc's input screening (src/work_lda_inc.c, work_gga_inc.c), for the exact reference."""
    st2 = (dt ** (4.0 / 3.0)) ** 2
    if not pol:
        return [max(dt, r)] + ([max(st2, s)] if fam == "gga" else [])
    x = [max(dt, r[0]), max(dt, r[1])]
    if fam == "gga":
        a, b = max(st2, s[0]), max(st2, s[2])
        ave = 0.5 * (a + b)
        x += [a, min(max(s[1], -ave), ave), b]
    return x


def run_one(entry, vipyr, exact, refresh=None):
    fam = entry.get("mix_family") or entry["path"].split("/")[0]
    mod = vipyr.load(os.path.join(XC, entry["path"]))
    out = {}
    for pol in (False, True):
        tag = "pol" if pol else "unpol"
        rho, sigma, dec = points(fam, pol)
        with X.Func(entry["id"], X.XC_POLARIZED if pol else X.XC_UNPOLARIZED) as f:
            def call(r, s):
                if fam == "lda":
                    zk, vrho = f.lda(r)
                    return {"zk": zk[:, None], "vrho": vrho}
                zk, vrho, vsigma = f.gga(r, s)
                return {"zk": zk[:, None], "vrho": vrho, "vsigma": vsigma}
            ref = call(rho, sigma)
            spread = {k: np.zeros_like(v) for k, v in ref.items()}
            rng = np.random.default_rng(7)
            for _ in range(NPERT):
                alt = call(rho * (1.0 + ULP_MOVE * rng.uniform(-1, 1, rho.shape)),
                           sigma * (1.0 + ULP_MOVE * rng.uniform(-1, 1, sigma.shape)))
                for k in ref:
                    with np.errstate(invalid="ignore"):
                        d = np.abs(alt[k] - ref[k])
                    spread[k] = np.fmax(spread[k], np.where(np.isfinite(d), d, 0.0))
        if fam == "lda":
            v = (mod.eval_pol(rho.ravel()) if pol else mod.eval_unpol(rho)).reshape(len(rho), -1)
        else:
            v = (mod.eval_pol(rho.ravel(), sigma.ravel()) if pol else mod.eval_unpol(rho, sigma)).reshape(len(rho), -1)
        ns = 2 if pol else 1
        cols = {"zk": slice(0, 1), "vrho": slice(1, 1 + ns), "vsigma": slice(1 + ns, None)}
        hard_pts = set()
        stats = {}
        for key in ref:
            got = v[:, cols[key]]
            e, den = scaled_error(ref[key], got, dec)
            with np.errstate(invalid="ignore"):
                by_spread = (e > RTOL) & (np.abs(got - ref[key]) <= 2.0 * spread[key])
            hard = (e > RTOL) & ~by_spread
            stats[key] = [e, den, got, by_spread, hard]
            hard_pts |= set(np.where(hard.any(axis=1))[0].tolist())
        # the 60-digit fallback is for edge points only: a spin density below
        # EDGE * total, where libxc clamps it or 1 - |zeta| is a few ulps
        edge = np.zeros(len(rho), bool) if not pol else (np.min(rho, axis=1) < EDGE * np.sum(rho, axis=1))
        if fam == "gga":   # a zero sigma, which libxc clamps to sigma_threshold^2: d/dsigma meets 1/sqrt(sigma)
            edge |= (sigma == 0) if not pol else ((sigma[:, 0] == 0) | (sigma[:, 2] == 0))
        hard_pts = {i for i in hard_pts if edge[i]}
        ex = exact.get(entry["name"], {}).get(tag, {})
        if refresh is not None and hard_pts and not entry.get("mix_family"):
            row, src = refresh
            ex = {}
            for i in sorted(hard_pts):
                x = screen(fam, pol, rho[i], sigma[i] if fam == "gga" else None, entry["dens_threshold"])
                ex[str(i)] = exact_mod.reference(row, src, fam, pol, x)
            exact.setdefault(entry["name"], {})[tag] = ex
        n_exact = 0
        n_edge = 0
        n_inner = 0
        inner_worst = 0.0
        for key, (e, den, got, by_spread, hard) in stats.items():
            for i, j in zip(*np.where(hard)):
                xi = ex.get(str(i)) if edge[i] else None
                if xi is None:
                    continue
                xv = np.array(xi)[cols[key]][j]
                if abs(got[i, j] - xv) <= abs(ref[key][i, j] - xv) + RTOL * den[i, j]:
                    hard[i, j] = False
                    e[i, j] = 0.0
                    n_exact += 1
                elif abs(ref[key][i, j] - xv) > RTOL * den[i, j]:
                    n_edge += 1      # libxc itself is off the 60-digit value here
            n_inner += int(np.sum(hard & ~edge[:, None]))
            e_kept = np.where(by_spread, 0.0, e)
            inner_worst = max(inner_worst, float(np.max(e_kept[~edge])) if (~edge).any() else 0.0)
            out[f"{key}_{tag}"] = (float(np.max(e_kept)), int(hard.sum()), int(by_spread.sum()))
        out[f"exact_{tag}"] = (0.0, 0, n_exact)
        out[f"edge_{tag}"] = (0.0, 0, n_edge)
        out[f"inner_{tag}"] = (inner_worst, 0, n_inner)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help=f"only {', '.join(SMOKE)}")
    ap.add_argument("--only", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--check", action="store_true", help="exit 1 only when a status differs from expected.json")
    ap.add_argument("--write-expected", action="store_true")
    ap.add_argument("--refresh-exact", metavar="LIBXC_SRC", default="",
                    help="recompute xc/tests/exact.json from the 6.1.0 source tree (needs tools/vivxc)")
    a = ap.parse_args()
    if X.version() != "6.1.0":
        sys.exit(f"oracle: libxc {X.version()} found, 6.1.0 needed")
    os.environ.setdefault("VIV_ROOT", ROOT)
    os.environ.setdefault("VIV", os.path.join(ROOT, "target", "release", "viv"))
    os.environ.setdefault("VIPYR_CACHE", os.path.join(ROOT, "target", "vivxc-cache"))
    import vipyr
    cat = json.load(open(os.path.join(XC, "catalog.json")))
    todo = [e for e in cat if e["status"] == "translated"]
    if a.smoke:
        todo = [e for e in todo if e["name"] in SMOKE]
    if a.only:
        keep = set(a.only.split(","))
        todo = [e for e in todo if e["name"] in keep]
    exact_path = os.path.join(HERE, "exact.json")
    exact = json.load(open(exact_path)) if os.path.exists(exact_path) else {}
    rows = {}
    if a.refresh_exact:
        global exact_mod
        import catalog
        import exact as exact_mod
        rows = {r["name"]: r for r in catalog.catalog(a.refresh_exact)}
    results = {}
    fails = []
    t0 = time.time()
    for e in todo:
        try:
            refresh = (rows[e["name"]], a.refresh_exact) if a.refresh_exact else None
            r = run_one(e, vipyr, exact, refresh)
            worst = max(w for k, (w, b, n) in r.items() if not k.startswith("inner"))
            ok = all(b == 0 for w, b, n in r.values())
            results[e["name"]] = dict(worst=worst, ok=ok,
                                      by_spread=sum(n for k, (w, b, n) in r.items() if k[:5] not in ("exact", "edge_", "inner")),
                                      by_exact=sum(n for k, (w, b, n) in r.items() if k.startswith("exact")),
                                      edge=sum(n for k, (w, b, n) in r.items() if k.startswith("edge_")),
                                      inner=sum(n for k, (w, b, n) in r.items() if k.startswith("inner")),
                                      failing_points=sum(b for w, b, n in r.values()),
                                      detail={k: w for k, (w, b, n) in r.items() if k[:5] not in ("exact", "edge_", "inner")})
            res = results[e["name"]]
            res["worst_inner"] = max(w for k, (w, b, n) in r.items() if k.startswith("inner"))
            res["status"] = "pass" if ok else ("edge" if res["inner"] == 0 else "fail")
        except Exception as ex:  # a module that does not compile or run is a failure, reported
            results[e["name"]] = dict(worst=math.inf, ok=False, status="fail",
                                      error=f"{type(ex).__name__}: {str(ex)[:300]}")
        r = results[e["name"]]
        tag = r["status"].upper()
        msg = r.get("error") or (f"spread {r['by_spread']:3d} exact {r['by_exact']:3d} bad {r['failing_points']:3d} "
                                 f"(edge, libxc off too: {r['edge']:3d})  " +
                                 " ".join(f"{k}={w:.1e}" for k, w in r["detail"].items()))
        print(f"{tag} {e['name']:28s} worst {r['worst']:.2e}  {msg}", flush=True)
        if not r["ok"]:
            fails.append(e["name"])
    from collections import Counter
    counts = Counter(r["status"] for r in results.values())
    print(f"oracle: {counts.get('pass', 0)} pass, {counts.get('edge', 0)} edge, {counts.get('fail', 0)} fail "
          f"of {len(todo)} functionals (RTOL {RTOL:g} against libxc 6.1.0, {time.time() - t0:.0f} s)")
    if a.refresh_exact:
        with open(exact_path, "w") as fh:
            json.dump(exact, fh, sort_keys=True)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(results, fh, indent=1, sort_keys=True)
    exp_path = os.path.join(HERE, "expected.json")
    if a.write_expected:
        old = json.load(open(exp_path)) if os.path.exists(exp_path) else {}
        old.update({k: r["status"] for k, r in results.items()})
        with open(exp_path, "w") as fh:
            json.dump(old, fh, indent=1, sort_keys=True)
    if a.check:
        want = json.load(open(exp_path))
        changed = [(k, want.get(k), r["status"]) for k, r in results.items() if want.get(k) != r["status"]]
        for k, w, g in changed:
            print(f"oracle: {k} was {w}, now {g}")
        sys.exit(1 if changed else 0)
    sys.exit(1 if fails else 0)   # without --check: any non-pass status fails


if __name__ == "__main__":
    main()
