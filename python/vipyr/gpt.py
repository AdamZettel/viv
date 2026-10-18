"""vipyr.gpt: generated from vipyr/spec.json (python -m vipyr.docs); see vipyr/_registry.py and vipyr/_gpt.py."""
from ._gpt import GptConfig, Model
from ._runtime import populate as _populate

_populate(globals(), "gpt")
__all__ += ["GptConfig", "Model"]
