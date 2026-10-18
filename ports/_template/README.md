# ports/_template: a port of an upstream code as a vipyr distribution

A port is an upstream program rewritten in Viv and checked against the
upstream code. Each one lives in `ports/<name>/` and is the distribution
`vipyr-<name>`. It keeps the upstream's license and credits the upstream
authors first. Nothing here is a port yet: copy this directory to start one.

    cp -R ports/_template ports/<name>
    mv ports/<name>/vipyr_NAME ports/<name>/vipyr_<name>

Then replace every `NAME` and fill in each file:

| file | what goes in it |
|---|---|
| `LICENSE` | the upstream license, verbatim. The port is distributed under it. |
| `AUTHORS` | the upstream authors first, as the upstream lists them, then the port's authors. |
| `CITATION.cff` | how to cite the upstream code (and its paper), then the port. |
| `UPSTREAM.md` | the upstream's URL, the commit or release ported, and every difference from it. |
| `pyproject.toml` | the distribution `vipyr-<name>` and its `vipyr.packages` entry point. |
| `vipyr_<name>/` | the family module; its functions come from the program's `<name>.export.toml` manifest through vipyr/_registry.py and `python -m vipyr.docs`. |

A directory whose name starts with `_` (this one) is not a distribution:
vipyr's checkout discovery (python/vipyr/_packages.py) skips it.
