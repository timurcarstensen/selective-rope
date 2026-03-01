from typing import TYPE_CHECKING

import torch
from fla.layers import GatedDeltaProduct as FLAGatedDeltaProduct  # type: ignore

if TYPE_CHECKING:
    from fla.models.utils import Cache  # type: ignore
    from transformers.processing_utils import Unpack


class GatedDeltaProduct(FLAGatedDeltaProduct):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

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
