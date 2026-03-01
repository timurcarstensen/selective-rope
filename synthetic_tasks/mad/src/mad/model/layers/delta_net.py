from typing import TYPE_CHECKING

import torch

from selective_rope.layers.gated_deltanet import GatedDeltaNet as SRGatedDeltaNet

if TYPE_CHECKING:
    from fla.models.utils import Cache
    from transformers.processing_utils import Unpack


_PE_KEY_MAP = {
    "selective_rope_use_bias": "use_bias",
    "use_gate": "use_gate",
    "chunked_linear_weight_norm": "chunked_linear_weight_norm",
    "skip_conv": "skip_conv",
    "phi_parametrization": "phi_parametrization",
    "phi_proj_rank": "phi_proj_rank",
}


def _extract_position_embedding(kwargs: dict) -> dict:
    pe_type = kwargs.pop("position_embedding_type", "nope")
    pe_config = {"type": pe_type}
    for yaml_key, pe_key in _PE_KEY_MAP.items():
        if yaml_key in kwargs:
            pe_config[pe_key] = kwargs.pop(yaml_key)
    return pe_config


class GatedDeltaNet(SRGatedDeltaNet):
    def __init__(
        self,
        dim: int = 1024,
        layernorm_eps: float = 1e-5,
        **kwargs,
    ) -> None:
        kwargs.pop("max_length", None)
        position_embedding = kwargs.pop("position_embedding", None)
        if position_embedding is None:
            position_embedding = _extract_position_embedding(kwargs)
        expand_k = kwargs.pop("expand_k", 1.0)
        num_heads = kwargs.get("num_heads", 8)
        head_dim = int(dim * expand_k / num_heads)
        super().__init__(
            hidden_size=dim,
            head_dim=head_dim,
            norm_eps=layernorm_eps,
            position_embedding=position_embedding,
            **kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        o, _, _ = super().forward(
            hidden_states,
            attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
            **kwargs,
        )

        return o
