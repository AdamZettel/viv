# SPDX-License-Identifier: CC0-1.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""vp.liquids: generated from vipyr_liquids/spec.json (python -m vipyr.docs); see vipyr/_registry.py.

The programs are Allen and Tildesley's examples for Computer Simulation of
Liquids (https://github.com/Allen-Tildesley/examples, CC0), ported to Viv.
If you use them, cite M. P. Allen and D. J. Tildesley, Computer Simulation
of Liquids, 2nd ed. (Oxford University Press, 2017).
"""
from pathlib import Path as _Path

from vipyr._runtime import populate as _populate

_populate(globals(), "liquids", _Path(__file__).resolve().with_name("spec.json"))
