# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vp.power: AC and DC power flow on MATPOWER cases (lib/power).

`runpf(case, method='nr')` takes a case as

    "case118"                a case shipped in lib/power/cases (MATPOWER's)
    "path/to/case.m"         a MATPOWER case file, format version 1 or 2
    {"baseMVA", "bus", "gen", "branch"}   a dict of MATPOWER's arrays, as
                             PYPOWER's and pandapower's `ppc` hold them

and solves it in lib/power/api.viv (Newton-Raphson, fast-decoupled XB or DC).
The .m reader here is the Python twin of lib/power/case.viv's read_case: the
same four items, the rest ignored.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np

from . import _core, _spec

METHODS = {"nr": 1, "fdxb": 2, "dc": 3}
_lock = threading.Lock()
_mod = None


def _api():
    global _mod
    with _lock:
        if _mod is None:
            _mod = _core.load(_spec.source("lib/power/api.viv"))
        return _mod


def cases_dir() -> Path:
    return Path(_spec.source("lib/power/cases"))


def read_case(path) -> dict:
    """baseMVA, bus, gen and branch of a MATPOWER .m case file."""
    text = Path(path).read_text(encoding="utf-8")
    lines = [ln.split("%", 1)[0] for ln in text.splitlines()]
    out = {}
    for ln in lines:
        m = re.match(r"\s*(?:mpc\.)?baseMVA\s*=\s*([^;]+);", ln)
        if m:
            out["baseMVA"] = float(m.group(1))
    for name in ("bus", "gen", "branch"):
        rows, inside = [], False
        for ln in lines:
            s = ln.strip()
            if not inside:
                m = re.match(r"(?:mpc\.)?%s\s*=\s*\[(.*)$" % name, s)
                if not m:
                    continue
                inside, s = True, m.group(1)
            done = "]" in s
            s = s.split("]", 1)[0]
            for part in s.split(";"):
                f = part.split()
                if f:
                    rows.append([float(x) for x in f])
            if done:
                break
        if not rows:
            raise ValueError("vp.power: %s has no %s matrix" % (path, name))
        out[name] = np.array(rows, dtype=float)
    if "baseMVA" not in out:
        raise ValueError("vp.power: %s sets no baseMVA" % path)
    return out


def read_raw(path) -> dict:
    """A PSS/E RAW file (revision 32 to 34), read by lib/power/raw.viv, as
    MATPOWER's arrays plus `machine` (MBASE, ZR, ZX per generator), `areas`
    and `rev`."""
    data = Path(path).read_bytes()
    codes = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    base, bus, gen, branch, machine, areas, rev = _api().raw_arrays(codes)
    return {"baseMVA": float(base), "bus": bus, "gen": gen, "branch": branch,
            "machine": machine, "areas": areas, "rev": int(rev)}


def load_case(case) -> tuple[dict, str]:
    """(the case's arrays, a label) from a name, a path or a dict."""
    if isinstance(case, dict):
        need = {"baseMVA", "bus", "gen", "branch"} - set(case)
        if need:
            raise ValueError("vp.power: the case dict lacks %s" % ", ".join(sorted(need)))
        return {k: (np.asarray(case[k], dtype=float) if k != "baseMVA" else float(case[k]))
                for k in ("baseMVA", "bus", "gen", "branch", "machine") if k in case}, "dict"
    p = Path(str(case))
    if p.suffix.lower() == ".raw":
        return read_raw(p), p.stem
    if p.suffix != ".m":
        shipped = cases_dir() / (str(case) + ".m")
        if not shipped.is_file():
            names = sorted(f.stem for f in cases_dir().glob("*.m"))
            raise ValueError("vp.power: no case %r; shipped: %s, or give a .m path" % (case, ", ".join(names)))
        p = shipped
    return read_case(p), p.stem


def _runpf(case, method="nr", enforce_q_lims=False, tol=1e-8, max_it=None):
    if method not in METHODS:
        raise ValueError("vp.power.runpf: method must be one of %s; got %r" % (", ".join(METHODS), method))
    c, label = load_case(case)
    if max_it is None:
        max_it = 30 if method == "fdxb" else 10
    api = _api()
    bus, gen, branch, its, ok, mis = api.runpf_arrays(c["baseMVA"], c["bus"], c["gen"], c["branch"],
                                                 METHODS[method], bool(enforce_q_lims), float(tol), int(max_it))
    nbr = c["branch"].shape[0]
    return {
        "success": bool(ok), "iterations": int(its), "method": method, "mismatch": float(mis),
        "vm": bus[:, 7].copy(), "va": bus[:, 8].copy(), "pg": gen[:, 1].copy(), "qg": gen[:, 2].copy(),
        "pf": branch[:nbr, 13].copy(), "qf": branch[:nbr, 14].copy(), "pt": branch[:nbr, 15].copy(),
        "qt": branch[:nbr, 16].copy(), "bus": bus, "gen": gen, "branch": branch, "baseMVA": c["baseMVA"],
        "_problem": label,
    }


# ------------------------------------------------------------ N-1

KINDS = {"branch": 1, "gen": 2}
VIOLATION_KINDS = {1: "thermal", 2: "low voltage", 3: "high voltage", 4: "no convergence"}
RATES = {"A": 5, "B": 6, "C": 7}
LIMITS = dict(vmin=0.95, vmax=1.05, vmin_post=0.90, vmax_post=1.10,
              rate_pre="A", rate_post="B")


def contingency_list(c, contingencies):
    """(kinds, elems) from 'n-1' (every branch and generator in service),
    'branches', 'gens', or a list of ('branch', k) / ('gen', g)."""
    if isinstance(contingencies, str):
        if contingencies not in ("n-1", "branches", "gens"):
            raise ValueError("vp.power: contingencies must be 'n-1', 'branches', "
                             "'gens' or a list of ('branch', k) / ('gen', g); "
                             "got %r" % contingencies)
        out = []
        if contingencies != "gens":
            out += [("branch", k) for k in np.where(c["branch"][:, 10] > 0)[0]]
        if contingencies != "branches":
            out += [("gen", g) for g in np.where(c["gen"][:, 7] > 0)[0]]
        contingencies = out
    kinds, elems = [], []
    for kind, k in contingencies:
        if kind not in KINDS:
            raise ValueError("vp.power: a contingency is ('branch', k) or "
                             "('gen', g); got %r" % (kind,))
        n = c["branch" if kind == "branch" else "gen"].shape[0]
        if not 0 <= int(k) < n:
            raise ValueError("vp.power: %s %d is not in the case (%d rows)"
                             % (kind, k, n))
        kinds.append(KINDS[kind])
        elems.append(int(k))
    return (np.array(kinds, dtype=np.int64), np.array(elems, dtype=np.int64))


def _limits(limits):
    lim = dict(LIMITS)
    for k, v in (limits or {}).items():
        if k not in lim:
            raise ValueError("vp.power: unknown limit %r; the limits are %s"
                             % (k, ", ".join(lim)))
        lim[k] = v
    for k in ("rate_pre", "rate_post"):
        if lim[k] not in RATES:
            raise ValueError("vp.power: %s must be 'A', 'B' or 'C'" % k)
    return lim


def _n1(c, kinds, elems, lim, tol=1e-8, max_it=10):
    return _api().n1_arrays(c["baseMVA"], c["bus"], c["gen"], c["branch"],
                            kinds, elems, float(lim["vmin"]),
                            float(lim["vmax"]), float(lim["vmin_post"]),
                            float(lim["vmax_post"]), RATES[lim["rate_pre"]],
                            RATES[lim["rate_post"]], float(tol), int(max_it))


def _contingency(case, contingencies="n-1", limits=None, screen=0.9,
                 tol=1e-8, max_it=10):
    c, label = load_case(case)
    kinds, elems = contingency_list(c, contingencies)
    lim = _limits(limits)
    conv, its, isl, lost, flows, vm, viol = _n1(c, kinds, elems, lim, tol,
                                                max_it)
    ptdf, lodf, bridge, flow, post = _api().dc_arrays(
        c["baseMVA"], c["bus"], c["gen"], c["branch"], kinds, elems)
    # a flow that did not converge has no state to report
    flows[conv == 0] = np.nan
    vm[conv == 0] = np.nan
    rate = c["branch"][:, RATES[lim["rate_post"]]]
    flagged = (np.abs(post) > screen * rate[None, :]) & (rate[None, :] > 0)
    flagged &= ~((kinds == 1) & (bridge[elems] == 1))[:, None]
    return {
        "kinds": kinds, "elems": elems, "converged": conv.astype(bool),
        "iterations": its, "islanded": isl, "lost_mw": lost, "flows": flows,
        "vm": vm, "violations": viol, "dc_flows": post, "dc_flagged": flagged,
        "ptdf": ptdf, "lodf": lodf, "bridges": np.where(bridge == 1)[0],
        "success": bool(conv[0]), "_problem": label,
    }


# ------------------------------------------------------------ the study

SINKS = {"slack": 1, "gens": 2}


def _bus_row(c, bus):
    rows = np.where(c["bus"][:, 0] == bus)[0]
    if len(rows) == 0:
        raise ValueError("vp.power: bus %r is not in the case" % (bus,))
    return int(rows[0])


def _sink(sink):
    if sink not in SINKS:
        raise ValueError("vp.power: sink must be 'slack' or 'gens'; got %r"
                         % (sink,))
    return SINKS[sink]


def _read_raw(path):
    c = read_raw(path)
    return dict(c, _problem=Path(path).stem, success=True)


def _interconnect(case, poi_bus, mw, mvar_range=None, contingencies="n-1",
                  limits=None, sink="slack", mw_max=None, tol_mw=1.0,
                  worse_pct=1.0):
    c, label = load_case(case)
    poi = _bus_row(c, poi_bus)
    if mvar_range is None:
        # a power factor of 0.95 either way at full output
        q = float(mw) * np.tan(np.arccos(0.95))
        mvar_range = (-q, q)
    qmin, qmax = (float(x) for x in mvar_range)
    if qmin > qmax:
        raise ValueError("vp.power.interconnect: mvar_range is (qmin, qmax)")
    if mw_max is None:
        mw_max = 4.0 * float(mw)
    kinds, elems = contingency_list(c, contingencies)
    lim = _limits(limits)
    (pre, post, status, df, limit, found, steps, conv, flows,
     vm) = _api().study_arrays(
        c["baseMVA"], c["bus"], c["gen"], c["branch"], poi, float(mw), qmin,
        qmax, _sink(sink), kinds, elems, float(lim["vmin"]),
        float(lim["vmax"]), float(lim["vmin_post"]),
        float(lim["vmax_post"]), RATES[lim["rate_pre"]],
        RATES[lim["rate_post"]], float(mw_max), float(tol_mw),
        float(worse_pct))
    return {
        "violations_before": pre, "violations_after": post,
        "change": status, "new": post[status == 1],
        "worsened": post[status == 2], "df": df,
        "transfer_limit_mw": float(limit), "limit_found": bool(found),
        "steps": steps, "kinds": kinds, "elems": elems,
        "converged": conv.astype(bool), "flows": flows, "vm": vm,
        "success": bool(conv[0]), "_problem": label,
    }


def _cpf(case, poi_bus, sink="slack", step=0.5, max_step=2.0,
         stop_frac=0.7, max_steps=300, tol=1e-10, nose_tol=1e-4,
         singular_values=False):
    c, label = load_case(case)
    poi = _bus_row(c, poi_bus)
    lam, vp, vmin, vm, va, lmax, nose, smin = _api().cpf_arrays(
        c["baseMVA"], c["bus"], c["gen"], c["branch"], poi, _sink(sink),
        float(step), float(max_step), float(stop_frac), int(max_steps),
        float(tol), float(nose_tol), bool(singular_values))
    base = c["baseMVA"]
    return {
        "mw": lam * base, "v_poi": vp, "v_min": vmin, "vm": vm, "va": va,
        "nose_mw": float(lmax * base), "nose": int(nose), "smin": smin,
        "success": int(nose) > 0, "_problem": label,
    }


def _short_circuit(case, buses=None, xdss=0.2, cos_phi=0.85):
    c, label = load_case(case)
    if "machine" in c:
        machine = c["machine"]
    else:
        machine = np.zeros((c["gen"].shape[0], 3))
        machine[:, 0] = np.where(c["gen"][:, 6] > 0, c["gen"][:, 6],
                                 c["baseMVA"])
        machine[:, 2] = float(xdss)
    if buses is None:
        rows = np.arange(c["bus"].shape[0])
    else:
        rows = np.array([_bus_row(c, b) for b in np.atleast_1d(buses)])
    if np.any(c["bus"][rows, 9] <= 0):
        raise ValueError("vp.power.short_circuit: a faulted bus has no base kV")
    ik, sk, rk, xk = _api().sc_arrays(c["baseMVA"], c["bus"], c["gen"],
                                      c["branch"], machine, float(cos_phi),
                                      rows.astype(np.int64))
    return {"buses": c["bus"][rows, 0].copy(), "ikss_ka": ik,
            "skss_mva": sk, "rk": rk, "xk": xk, "success": True,
            "_problem": label}


def format_violations(res, which="new"):
    """Text rows of a study's or contingency run's violations."""
    v = res[which] if isinstance(res, dict) else getattr(res, which)
    kinds, elems = (res["kinds"], res["elems"]) if isinstance(res, dict) \
        else (res.kinds, res.elems)
    out = []
    for row in v:
        k = int(row[0])
        cont = "base case" if k < 0 else "%s %d out" % (
            "branch" if kinds[k] == 1 else "gen", elems[k])
        what = VIOLATION_KINDS[int(row[1])]
        out.append("%-16s %-14s %5d  %9.3f  %8.3f  %6.1f%%"
                   % (cont, what, int(row[2]), row[3], row[4], row[5]))
    return out
