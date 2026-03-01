from __future__ import annotations

from typing import Any

import torch
from fla.models.utils import Cache as FLACache


class Cache(FLACache):
    """FLACache extended with rotary_state support for SelectiveRoPE."""

    def update(
        self,
        recurrent_state: tuple[torch.Tensor] | None = None,
        attn_state: tuple[torch.Tensor] | None = None,
        conv_state: tuple[torch.Tensor] | None = None,
        ffn_state: tuple[torch.Tensor] | None = None,
        rotary_state: Any | None = None,
        layer_idx: int = 0,
        offset: int | None = 1,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = super().update(
            recurrent_state=recurrent_state,
            attn_state=attn_state,
            conv_state=conv_state,
            ffn_state=ffn_state,
            layer_idx=layer_idx,
            offset=offset,
            cache_kwargs=cache_kwargs,
        )
        if rotary_state is not None:
            state["rotary_state"] = rotary_state
        return state
