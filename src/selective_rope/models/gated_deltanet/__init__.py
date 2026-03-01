from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from selective_rope.models.gated_deltanet.configuration import GatedDeltaNetConfig
from selective_rope.models.gated_deltanet.modeling import (
    GatedDeltaNetForCausalLM,
    GatedDeltaNetModel,
)

AutoConfig.register(GatedDeltaNetConfig.model_type, GatedDeltaNetConfig, exist_ok=True)
AutoModel.register(GatedDeltaNetConfig, GatedDeltaNetModel, exist_ok=True)
AutoModelForCausalLM.register(
    GatedDeltaNetConfig, GatedDeltaNetForCausalLM, exist_ok=True
)

__all__ = ["GatedDeltaNetConfig", "GatedDeltaNetForCausalLM", "GatedDeltaNetModel"]
