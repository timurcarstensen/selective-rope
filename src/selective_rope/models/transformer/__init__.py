from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from selective_rope.models.transformer.configuration import TransformerConfig
from selective_rope.models.transformer.modeling import (
    TransformerForCausalLM,
    TransformerModel,
)

AutoConfig.register(TransformerConfig.model_type, TransformerConfig, exist_ok=True)
AutoModel.register(TransformerConfig, TransformerModel, exist_ok=True)
AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM, exist_ok=True)

__all__ = ["TransformerConfig", "TransformerForCausalLM", "TransformerModel"]
