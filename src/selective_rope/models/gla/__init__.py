from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from selective_rope.models.gla.configuration import GLAConfig
from selective_rope.models.gla.modeling import GLAForCausalLM, GLAModel

AutoConfig.register(GLAConfig.model_type, GLAConfig, exist_ok=True)
AutoModel.register(GLAConfig, GLAModel, exist_ok=True)
AutoModelForCausalLM.register(GLAConfig, GLAForCausalLM, exist_ok=True)

__all__ = ["GLAConfig", "GLAForCausalLM", "GLAModel"]
