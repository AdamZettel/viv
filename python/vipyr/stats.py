"""vipyr.stats: generated from vipyr/spec.json (python -m vipyr.docs); see
vipyr/_registry.py and vipyr/_stats.py."""
from ._runtime import populate as _populate
from ._stats import (
    FAMILY_OBJECTS as _FAMILY_OBJECTS,
    GLM_FITTERS,
    LSTSQ_SOLVERS,
    SPECIAL_FUNCTIONS as _SPECIAL_FUNCTIONS,
    VARIANCE_METHODS,
    Distribution,
    GLMWarning,
)

_populate(globals(), "stats")
globals().update(_FAMILY_OBJECTS)
globals().update(_SPECIAL_FUNCTIONS)
__all__ += (
    list(_FAMILY_OBJECTS)
    + list(_SPECIAL_FUNCTIONS)
    + ["Distribution", "GLMWarning", "LSTSQ_SOLVERS", "GLM_FITTERS"]
    + ["VARIANCE_METHODS"]
)
