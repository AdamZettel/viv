# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""The interconnection-study tools of lib/power against ANDES, pandapower
and PYPOWER, run on the same files. Needs the reference packages
(pandapower 3.5.6, PYPOWER 5.1.21, ANDES 2.0.0) in the Python that runs it:

    ~/.venvs/power/bin/python lib/power/compare_study.py raw n1 study cpf sc

Each part prints its table and stores it in lib/power/compare_study.json;
lib/power/COMPARISON.md quotes them. ANDES's sample RAW cases are read from
the installed package (they are GPL-3.0 and not copied here).
"""
import json
import logging
import sys
import time
from types import SimpleNamespace
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))
import vipyr as vp  # noqa: E402
from vipyr import _power  # noqa: E402

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
OUT = HERE / "compare_study.json"
ANDES_RAW = [
    "ieee14/ieee14.raw",
    "wscc9/wscc9.raw",
    "kundur/kundur.raw",
    "ieee39/ieee39.raw",
    "npcc/npcc.raw",
    "wecc/wecc.raw",
    "nordic44/N44_BC.raw",
]


def _andes():
    import andes

    andes.config_logger(stream_level=50)
    return andes


def andes_pf(path):
    """ANDES's import and power flow: bus numbers, V, Ybus, and its
    pandapower conversion's power flow and Ybus."""
    andes = _andes()
    import pandapower as pp
    from andes.interop.pandapower import to_pandapower

    ss = andes.load(path, setup=True, no_output=True, default_config=True)
    ss.PFlow.config.tol = 1e-10
    ss.PFlow.run()
    ids = list(ss.Bus.idx.v)
    v_a = ss.Bus.v.v * np.exp(1j * ss.Bus.a.v)
    n = len(ids)
    Y = np.zeros((n, n), complex)
    yl = ss.Line.build_y()
    for i, j, val in zip(yl.I, yl.J, yl.V):
        Y[i, j] += val
    pos = {b: k for k, b in enumerate(ids)}
    for mdl in (ss.Shunt,):
        for b, g, bb, u in zip(mdl.bus.v, mdl.g.v, mdl.b.v, mdl.u.v):
            Y[pos[b], pos[b]] += u * (g + 1j * bb)
    net = to_pandapower(ss, verify=False)
    try:
        pp.runpp(net, tolerance_mva=1e-8, max_iteration=30)
        v_p = net.res_bus.vm_pu.values * np.exp(
            1j * np.radians(net.res_bus.va_degree.values)
        )
    except pp.LoadflowNotConverged:
        v_p = np.full(len(ids), np.nan)
    Yp = net._ppc["internal"]["Ybus"].toarray()
    return ids, v_a, Y, v_p, Yp, bool(ss.PFlow.converged)


def part_raw():
    andes = _andes()
    rows = []
    for name in ANDES_RAW:
        path = andes.get_case(name)
        t0 = time.time()
        c = _power.read_raw(path)
        t_read = time.time() - t0
        r = vp.power.runpf(c, tol=1e-11)
        G, B = _power._api().ybus_dense(
            c["baseMVA"], c["bus"], c["gen"], c["branch"]
        )
        Yv = G + 1j * B
        ids, v_a, Ya, v_p, Yp, ok = andes_pf(path)
        order = [list(c["bus"][:, 0].astype(int)).index(b) for b in ids]
        vv_v = (r.vm * np.exp(1j * np.radians(r.va)))[order]
        Yv = Yv[np.ix_(order, order)]
        # ANDES's pandapower conversion puts the slack at angle 0; the
        # voltages are compared with every angle measured from the slack's
        ref = int(np.where(c["bus"][order, 1] == 3)[0][0])
        vv_v, v_a, v_p = (
            x * np.exp(-1j * np.angle(x[ref])) for x in (vv_v, v_a, v_p)
        )
        rows.append(
            dict(
                case=Path(name).name,
                rev=c["rev"],
                buses=len(ids),
                branches=int(c["branch"].shape[0]),
                read_s=round(t_read, 3),
                dY_andes=float(np.abs(Yv - Ya).max()),
                dY_pandapower=float(np.abs(Yv - Yp).max()),
                dV_andes=float(np.abs(vv_v - v_a).max()),
                dV_pandapower=float(np.abs(vv_v - v_p).max()),
                converged=bool(r.success and ok),
            )
        )
        print(_fmt(rows[-1]))
    return rows


def _fmt(d):
    return "  ".join(
        "%s=%s"
        % (
            k,
            ("%.1e" % v) if isinstance(v, float) and k.startswith("d") else v,
        )
        for k, v in d.items()
    )


def ppc_of(c):
    """A PYPOWER ppc of a case dict (gen widened to 21 columns)."""
    gen = np.zeros((c["gen"].shape[0], 21))
    gen[:, : c["gen"].shape[1]] = c["gen"][:, :21]
    branch = np.zeros((c["branch"].shape[0], 13))
    branch[:, :13] = c["branch"][:, :13]
    return {
        "version": "2",
        "baseMVA": c["baseMVA"],
        "bus": c["bus"].copy(),
        "gen": gen,
        "branch": branch,
    }


def pypower_n1(c, kinds, elems):
    """Brute force: PYPOWER's runpf on the case with each element out.
    Rows as vp.power's: base first. NaN rows where PYPOWER cannot solve
    (an islanded network)."""
    from pypower.api import ppoption, runpf

    opt = ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-10)
    nl, nb = c["branch"].shape[0], c["bus"].shape[0]
    S = np.full((len(kinds) + 1, nl), np.nan)
    V = np.full((len(kinds) + 1, nb), np.nan)
    for row in range(len(kinds) + 1):
        ppc = ppc_of(c)
        if row > 0:
            k, e = kinds[row - 1], elems[row - 1]
            if k == 1:
                ppc["branch"][e, 10] = 0
            else:
                ppc["gen"][e, 7] = 0
        try:
            r, ok = runpf(ppc, opt)
        except Exception:
            ok = False
        if ok:
            b = r["branch"]
            S[row] = np.maximum(
                np.hypot(b[:, 13], b[:, 14]), np.hypot(b[:, 15], b[:, 16])
            )
            V[row] = r["bus"][:, 7]
    return S, V


def pandapower_n1(c, kinds, elems):
    """pandapower's run_contingency on from_ppc's network, every branch and
    non-slack generator out in turn: per bus, the lowest and highest Vm
    over the base case and the contingencies."""
    import pandapower as pp
    from pandapower.contingency import run_contingency
    from pandapower.converter.pypower.from_ppc import from_ppc

    net = from_ppc(ppc_of(c), f_hz=60, validate_conversion=False)
    at = {idx: k for k, idx in enumerate(net.bus.index)}
    pairs = {}
    for el, a, b in (
        ("line", "from_bus", "to_bus"),
        ("trafo", "hv_bus", "lv_bus"),
        ("impedance", "from_bus", "to_bus"),
    ):
        for i in net[el].index:
            key = frozenset((at[net[el].at[i, a]], at[net[el].at[i, b]]))
            pairs.setdefault(key, []).append((el, i))
    pos = {int(b): k for k, b in enumerate(c["bus"][:, 0])}
    cases = {"line": [], "trafo": [], "impedance": [], "gen": []}
    seen = {}
    for k in range(c["branch"].shape[0]):
        key = frozenset(
            (pos[int(c["branch"][k, 0])], pos[int(c["branch"][k, 1])])
        )
        n = seen.get(key, 0)
        seen[key] = n + 1
        if k in set(elems[kinds == 1]):
            el, i = pairs[key][n]
            cases[el].append(i)
    gk = {}
    for i in net.gen.index:
        gk.setdefault(at[net.gen.at[i, "bus"]], []).append(i)
    for g in elems[kinds == 2]:
        lst = gk.get(pos[int(c["gen"][g, 0])], [])
        if lst:
            cases["gen"].append(lst.pop(0))
    opts = dict(tolerance_mva=1e-8, max_iteration=20)
    run_contingency(
        net,
        {el: {"index": ix} for el, ix in cases.items() if ix},
        pf_options=opts,
        pf_options_nminus1=opts,
    )
    return (
        net.res_bus.min_vm_pu.values,
        net.res_bus.max_vm_pu.values,
        {el: len(ix) for el, ix in cases.items()},
    )


def part_n1():
    rows = []
    for name in ("case30", "case118", "case300"):
        c, _ = _power.load_case(name)
        t0 = time.time()
        r = SimpleNamespace(**_power._contingency(name, tol=1e-10))
        t_vv = time.time() - t0
        kinds, elems = r.kinds, r.elems
        t0 = time.time()
        S, V = pypower_n1(c, kinds, elems)
        t_pp = time.time() - t0
        ok = ~np.isnan(S[:, 0]) & r.converged
        vmin_pp, vmax_pp, used = pandapower_n1(c, kinds, elems)
        vmv = np.where(r.vm > 0, r.vm, np.nan)
        # pandapower's contingencies leave out the slack's generator
        skip = [
            k + 1
            for k in range(len(kinds))
            if kinds[k] == 2
            and c["bus"][
                [list(c["bus"][:, 0]).index(c["gen"][elems[k], 0])], 1
            ][0]
            == 3
        ]
        keep = r.converged.copy()
        keep[skip] = False
        py_fail = np.isnan(S[:, 0]) & (r.islanded == 0)
        rows.append(
            dict(
                case=name,
                contingencies=len(kinds),
                islanding=int((r.islanded[1:] > 0).sum()),
                vv_s=round(t_vv, 2),
                pypower_s=round(t_pp, 2),
                compared=int(ok.sum()),
                dS_pypower_MVA=float(np.nanmax(np.abs(r.flows[ok] - S[ok]))),
                dVm_pypower=float(np.nanmax(np.abs(r.vm[ok] - V[ok]))),
                pandapower_cases=sum(used.values()),
                dVmin_pandapower=float(
                    np.nanmax(np.abs(np.nanmin(vmv[keep], 0) - vmin_pp))
                ),
                dVmax_pandapower=float(
                    np.nanmax(np.abs(np.nanmax(vmv[keep], 0) - vmax_pp))
                ),
                vv_diverged=int((~r.converged).sum()),
                pypower_diverged_same=int((py_fail & ~r.converged).sum()),
                pypower_diverged_only=int((py_fail & r.converged).sum()),
                violations=int(r.violations.shape[0]),
            )
        )
        print(_fmt(rows[-1]))
    return rows


def _viol_keys(S, V, c, kinds, elems, lim):
    """(contingency, kind, element) of every violation in PYPOWER's rows,
    with the limits vp.power uses."""
    keys = set()
    for row in range(S.shape[0]):
        cont = row - 1
        if np.isnan(S[row, 0]):
            keys.add((cont, 4, -1))
            continue
        pre = row == 0
        rate = c["branch"][:, 5 if pre else 6]
        vlo, vhi = (
            (lim["vmin"], lim["vmax"])
            if pre
            else (lim["vmin_post"], lim["vmax_post"])
        )
        on = c["branch"][:, 10] > 0
        if row > 0 and kinds[row - 1] == 1:
            on = on.copy()
            on[elems[row - 1]] = False
        for k in np.where(on & (rate > 0) & (S[row] > rate))[0]:
            keys.add((cont, 1, int(k)))
        for b in np.where(V[row] < vlo)[0]:
            keys.add((cont, 2, int(b)))
        for b in np.where(V[row] > vhi)[0]:
            keys.add((cont, 3, int(b)))
    return keys


def part_study():
    """The transfer limit, checked by PYPOWER: N-1 brute force at the limit
    (no new violation) and tol_mw above it (a new one)."""
    rows = []
    for name, poi, mw in (
        ("case30", 30, 10.0),
        ("case118", 38, 200.0),
        ("case118", 87, 100.0),
    ):
        c, _ = _power.load_case(name)
        t0 = time.time()
        r = vp.power.interconnect(
            name,
            poi,
            mw,
            mw_max=1500.0 if name == "case118" else 200.0,
            tol_mw=2.0,
        )
        t_vv = time.time() - t0
        lim = _power._limits(None)
        row = int(np.where(c["bus"][:, 0] == poi)[0][0])
        vset = vp.power.runpf(name).vm[row]
        q = mw * np.tan(np.arccos(0.95))
        pre = _viol_keys(
            *pypower_n1(c, r.kinds, r.elems), c, r.kinds, r.elems, lim
        )
        new = []
        for at in (
            r.transfer_limit_mw,
            min(
                r.steps[r.steps[:, 1] > 0, 0][
                    r.steps[r.steps[:, 1] > 0, 0] > r.transfer_limit_mw
                ]
            ),
        ):
            bus, gen = _power._api().inject_arrays(
                c["baseMVA"],
                c["bus"],
                c["gen"],
                c["branch"],
                row,
                at,
                vset,
                -q,
                q,
                1,
            )
            ci = dict(c, bus=bus, gen=gen)
            post = _viol_keys(
                *pypower_n1(ci, r.kinds, r.elems), ci, r.kinds, r.elems, lim
            )
            new.append(len(post - pre))
        rows.append(
            dict(
                case=name,
                poi=poi,
                mw=mw,
                limit_mw=round(r.transfer_limit_mw, 2),
                vv_s=round(t_vv, 2),
                pypower_new_at_limit=new[0],
                pypower_new_above=new[1],
                violations_pre=len(pre),
            )
        )
        print(_fmt(rows[-1]))
    return rows


def part_cpf():
    """The PV curve's nose against PYPOWER's runcpf (pseudo-arclength, full
    trace) and the largest transfer at which PYPOWER's Newton power flow
    still converges; and every vp.power curve point re-solved by PYPOWER."""
    import pypower.cpf_corrector as pcc
    import pypower.cpf_predictor as pcp
    from pypower.api import ppoption, runpf
    from pypower.cpf_p_jac import cpf_p_jac
    from pypower.runcpf import runcpf

    # PYPOWER 5.1.21's predictor stacks a Python scalar into a sparse matrix,
    # which SciPy 1.13+ refuses; the same value as a 1 x 1 array
    def p_jac(*a):
        dv, dl = cpf_p_jac(*a)
        return dv, np.array([[dl]]) if np.ndim(dl) == 0 else dl

    pcp.cpf_p_jac = p_jac
    if hasattr(pcc, "cpf_p_jac"):
        pcc.cpf_p_jac = p_jac
    rows = []
    for name, poi in (("case9", 5), ("case30", 30), ("case118", 38)):
        c, _ = _power.load_case(name)
        t0 = time.time()
        r = vp.power.cpf(
            name, poi, step=0.2, max_step=1.0, singular_values=True
        )
        t_vv = time.time() - t0
        row = int(np.where(c["bus"][:, 0] == poi)[0][0])
        base = ppc_of(c)
        target = ppc_of(c)
        target["bus"][row, 2] -= 100.0
        opt = ppoption(
            VERBOSE=0,
            OUT_ALL=0,
            CPF_STOP_AT="FULL",
            CPF_STEP=0.02,
            PF_TOL=1e-10,
        )
        t0 = time.time()
        res, ok = runcpf(base, target, opt)
        t_py = time.time() - t0
        cp = res["cpf"]
        lam_py = float(np.nanmax(cp["lam_c"])) * 100.0
        # the largest transfer where Newton still converges, by bisection with a warm start
        pf_opt = ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-10, PF_MAX_IT=30)
        lo, hi, warm = 0.0, r.nose_mw * 1.2, ppc_of(c)
        while hi - lo > 1e-3:
            mid = 0.5 * (lo + hi)
            t = dict(warm, bus=warm["bus"].copy())
            t["bus"][row, 2] = c["bus"][row, 2] - mid
            out, okk = runpf(t, pf_opt)
            if okk:
                lo, warm = mid, dict(out, bus=out["bus"].copy())
                warm["bus"][row, 2] = c["bus"][row, 2]
            else:
                hi = mid
        # each curve point solved by PYPOWER at its transfer from vp.power's voltages
        dv = 0.0
        for k in range(len(r.mw)):
            t = ppc_of(c)
            t["bus"][:, 7], t["bus"][:, 8] = r.vm[k], np.degrees(r.va[k])
            t["bus"][row, 2] -= r.mw[k]
            gb = [list(c["bus"][:, 0]).index(g) for g in c["gen"][:, 0]]
            t["gen"][:, 5] = r.vm[k][gb]
            out, okk = runpf(t, pf_opt)
            if okk:
                dv = max(dv, np.abs(out["bus"][:, 7] - r.vm[k]).max())
        rows.append(
            dict(
                case=name,
                poi=poi,
                points=len(r.mw),
                vv_nose_MW=round(r.nose_mw, 3),
                pypower_cpf_nose_MW=round(lam_py, 3),
                newton_limit_MW=round(lo, 3),
                smin_ratio=float(r.smin[r.nose] / r.smin[0]),
                dVm_points=float(dv),
                vv_s=round(t_vv, 2),
                pypower_s=round(t_py, 2),
            )
        )
        print(_fmt(rows[-1]))
    return rows


def part_sc():
    """IEC 60909 Ik'' at every bus against pandapower.shortcircuit on the
    same network (from_ppc, the slack made a generator, transformers on
    the system base so that K_T sees their impedance in per unit)."""
    import pandapower as pp
    import pandapower.shortcircuit as sc
    from pandapower.converter.pypower.from_ppc import from_ppc

    andes = _andes()
    rows = []
    for label, path in (
        ("case118_v33.raw", str(HERE / "cases" / "raw" / "case118_v33.raw")),
        ("ieee14.raw (ANDES)", andes.get_case("ieee14/ieee14.raw")),
        ("ieee39.raw (ANDES)", andes.get_case("ieee39/ieee39.raw")),
    ):
        c = _power.read_raw(path)
        t0 = time.time()
        r = vp.power.short_circuit(path)
        t_vv = time.time() - t0
        net = from_ppc(ppc_of(c), f_hz=60)
        base = c["baseMVA"]
        f = base / net.trafo.sn_mva
        net.trafo["vk_percent"] *= f
        net.trafo["vkr_percent"] *= f
        net.trafo["sn_mva"] = base
        # every generator, the slack's too, as a machine with its RAW source impedance
        net.gen.drop(net.gen.index, inplace=True)
        net.ext_grid.drop(net.ext_grid.index, inplace=True)
        at = {k: idx for k, idx in enumerate(net.bus.index)}
        pos = {int(b): k for k, b in enumerate(c["bus"][:, 0])}
        for g in range(c["gen"].shape[0]):
            if c["gen"][g, 7] <= 0:
                continue
            b = at[pos[int(c["gen"][g, 0])]]
            mb, zr, zx = c["machine"][g]
            vn = net.bus.at[b, "vn_kv"]
            pp.create_gen(
                net,
                b,
                p_mw=c["gen"][g, 1],
                vm_pu=c["gen"][g, 5],
                sn_mva=mb,
                vn_kv=vn,
                xdss_pu=zx,
                rdss_ohm=zr * vn**2 / mb,
                cos_phi=0.85,
                slack=not len(net.gen),
            )
        sc.calc_sc(net, fault="3ph", case="max", ip=False)
        ik = net.res_bus_sc.ikss_ka.values
        # from_ppc reads a transformer of ratio exactly 1 as a plain impedance,
        # without K_T; vp.power applies K_T to every RAW transformer. The same
        # network with those branches given ratio 0 (MATPOWER's "not a
        # transformer") isolates that one difference.
        unit = (c["branch"][:, 8] == 1.0) & (c["branch"][:, 9] == 0)
        c0 = dict(c, branch=c["branch"].copy())
        c0["branch"][unit, 8] = 0.0
        r0 = vp.power.short_circuit(c0)
        rows.append(
            dict(
                case=label,
                buses=len(ik),
                ikss_min_kA=round(float(ik.min()), 3),
                ikss_max_kA=round(float(ik.max()), 3),
                unit_ratio_xfmrs=int(unit.sum()),
                rel_diff=float(np.max(np.abs(r.ikss_ka - ik) / ik)),
                rel_diff_same_kt=float(np.max(np.abs(r0.ikss_ka - ik) / ik)),
                vv_s=round(t_vv, 2),
            )
        )
        print(_fmt(rows[-1]))
    return rows


PARTS = {
    "raw": part_raw,
    "n1": part_n1,
    "study": part_study,
    "cpf": part_cpf,
    "sc": part_sc,
}

if __name__ == "__main__":
    store = json.loads(OUT.read_text()) if OUT.exists() else {}
    for part in sys.argv[1:] or list(PARTS):
        print("==", part)
        store[part] = PARTS[part]()
    OUT.write_text(json.dumps(store, indent=1) + "\n")
