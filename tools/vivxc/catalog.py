# SPDX-License-Identifier: MIT OR Apache-2.0
"""Which libxc 6.1.0 functional comes from which Maple script, and with what data.

The source of each mapping is libxc's own C front file (src/<name>.c): the
`#include "maple2c/<type>/<script>.c"` line names the Maple script, and the
`xc_func_info_<name>` block names the external-parameter setter. Parameter
values, thresholds and mixing come from the running libxc (libxc.py)."""
import glob
import os
import re

import libxc as X

INC = re.compile(r'#include\s+"maple2c/(\w+)/(\w+)\.c"')
INFO = re.compile(r'xc_func_info_type\s+xc_func_info_(\w+)\s*=\s*\{(.*?)\n\};', re.S)
CPY = {"set_ext_params_cpy", "set_ext_params_cpy_omega", "set_ext_params_cpy_exx", "set_ext_params_cpy_cam",
       "set_ext_params_cpy_camy", "set_ext_params_cpy_cam_sr", "set_ext_params_cpy_lc", "NULL"}


STRUCT = re.compile(r'typedef\s+struct\s*\{(.*?)\}\s*(\w+)\s*;', re.S)


def struct_fields(text):
    """The params struct of a C front file, flattened: [(field, index or None)]."""
    for body, name in STRUCT.findall(text):
        if not name.endswith("params"):
            continue
        body = re.sub(r'/\*.*?\*/', '', body, flags=re.S)
        body = re.sub(r'//[^\n]*', '', body)
        fields = []
        for decl in body.split(";"):
            decl = decl.strip()
            if not decl:
                continue
            m = re.match(r'(double|int)\s+(.*)$', decl, re.S)
            if not m or m.group(1) != "double":
                return None
            for item in m.group(2).split(","):
                item = item.strip()
                mm = re.match(r'(\w+)\s*((?:\[\s*\d+\s*\])*)$', item)
                if not mm:
                    return None
                dims = [int(d) for d in re.findall(r'\d+', mm.group(2))]
                if not dims:
                    fields.append((mm.group(1), None))
                elif len(dims) == 1:
                    fields += [(mm.group(1), k) for k in range(dims[0])]
                elif len(dims) == 2:
                    fields += [(mm.group(1), (i, k)) for i in range(dims[0]) for k in range(dims[1])]
                else:
                    return None
        return fields
    return None


def case_literals(text):
    """XC_NAME -> {field: value} from `case XC_NAME: params->x = <number>; ... break;` blocks."""
    out = {}
    for m in re.finditer(r'((?:case\s+XC_\w+\s*:\s*)+)(.*?)break\s*;', text, re.S):
        vals = dict((f, float(v)) for f, v in re.findall(
            r'params->(\w+)\s*=\s*([-+]?[0-9.]+(?:[eE][-+]?\d+)?)\s*;', m.group(2)))
        for name in re.findall(r'case\s+XC_(\w+)', m.group(1)):
            out[name.lower()] = vals
    return out


def init_literals(text):
    """params->x = <number>; assignments that occur once (init-time defaults)."""
    seen = {}
    for f, v in re.findall(r'params->(\w+)\s*=\s*([-+]?[0-9.]+(?:[eE][-+]?\d+)?)\s*;', text):
        seen.setdefault(f, set()).add(float(v))
    return {f: vs.pop() for f, vs in seen.items() if len(vs) == 1}


def scan_sources(libxc_src):
    """name -> dict(file, scripts=[(type, script)], setter)"""
    out = {}
    for path in sorted(glob.glob(os.path.join(libxc_src, "src", "*.c"))):
        text = open(path, encoding="utf-8", errors="replace").read()
        scripts = INC.findall(text)
        cases = case_literals(text)
        for name, body in INFO.findall(text):
            setter = None
            m = re.search(r'\{\s*[\w]+\s*,\s*[\w]+\s*,\s*[\w]+\s*,\s*[\w]+\s*,\s*(\w+)\s*\}', body)
            if m:
                setter = m.group(1)
            else:
                m = re.search(r'\{\s*0\s*,\s*NULL\s*,\s*NULL\s*,\s*NULL\s*,\s*NULL\s*\}', body)
                setter = "NULL" if m else "?"
            out[name] = dict(file=os.path.basename(path), scripts=scripts, setter=setter,
                             fields=struct_fields(text), literals={**init_literals(text), **cases.get(name, {})})
    return out


def catalog(libxc_src, families=("lda", "gga", "hyb_gga", "hyb_lda")):
    src = scan_sources(libxc_src)
    rows = []
    for fid in X.all_ids():
        nm = X.name(fid)
        try:
            with X.Func(fid) as f:
                info = f.info()
        except ValueError:
            continue
        if info["family"] not in families:
            continue
        s = src.get(nm, dict(file=None, scripts=[], setter="?", fields=None, literals={}))
        info.update(name=nm, cfile=s["file"], scripts=s["scripts"], setter=s["setter"], fields=s["fields"],
                    literals=s["literals"])
        rows.append(info)
    return rows


if __name__ == "__main__":
    import sys
    rows = catalog(sys.argv[1])
    from collections import Counter
    print(len(rows), Counter(r["family"] for r in rows))
    print("setters", Counter(r["setter"] for r in rows))
    print("nscripts", Counter(len(r["scripts"]) for r in rows))
    for r in rows:
        if len(r["scripts"]) != 1:
            print(r["name"], r["cfile"], r["scripts"], r["aux"][:3])
