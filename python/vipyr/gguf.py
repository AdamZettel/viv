"""vipyr.gguf: generated from vipyr/spec.json (python -m vipyr.docs); see vipyr/_registry.py and vipyr/_gguf.py."""
from ._gguf import TYPE_NAMES, TensorInfo
from ._runtime import populate as _populate

_populate(globals(), "gguf")
__all__ += ["TensorInfo", "TYPE_NAMES"]
