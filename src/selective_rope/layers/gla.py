# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import (
    FusedRMSNormGated,
    RMSNorm,
    RotaryEmbedding,
    ShortConvolution,
)
from fla.modules.activations import ACT2FN
from fla.modules.l2norm import l2_norm
from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla

from selective_rope.modules.rotary import SelectiveRoPE

if TYPE_CHECKING:
    from fla.models.utils import Cache
    from transformers.processing_utils import Unpack


def _build_rotary(pe_config: dict, num_heads: int, head_k_dim: int, hidden_size: int):
    pe_type = pe_config.get("type", "nope")
    pe_kwargs = {k: v for k, v in pe_config.items() if k != "type"}
    match pe_type:
        case "selective_rope":
            return SelectiveRoPE(
                num_heads=num_heads,
                head_k_dim=head_k_dim,
                model_dim=hidden_size,
                **pe_kwargs,
            )
        case "rope":
            return RotaryEmbedding(
                dim=head_k_dim,
                base=pe_kwargs.get("theta", 500000),
            )
        case "nope" | None:
            return None
        case _:
            raise ValueError(f"Position embedding type {pe_type} not supported.")


class GatedLinearAttention(nn.Module):
    r"""
    The layer implementaion for [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635).  # noqa
    """

    def __init__(
        self,
        mode: str = "chunk",
        hidden_size: int = 1024,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 4,
        num_kv_heads: int | None = None,
        feature_map: str | None = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_output_gate: bool = True,
        gate_fn: str = "swish",
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        clamp_min: float | None = None,
        fuse_norm: bool = True,
        norm_q: bool = False,
        norm_k: bool = False,
        layer_idx: int | None = None,
        max_position_embeddings: int = 2048,
        position_embedding: dict | None = None,
    ):
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.feature_map_fn = ACT2FN[feature_map] if feature_map is not None else None

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.use_output_gate = use_output_gate

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.clamp_min = clamp_min
        self.norm_q = norm_q
        self.norm_k = norm_k
        self.layer_idx = layer_idx
        self.max_position_embeddings = max_position_embeddings

        self.position_embedding = position_embedding or {"type": "nope"}
        self.position_embedding_type = self.position_embedding.get("type", "nope")
        self.rope_skip_conv = self.position_embedding.get("skip_conv", False)

        assert mode in ["chunk", "fused_recurrent", "fused_chunk"], (
            f"Not supported mode `{mode}`."
        )
        assert self.key_dim % num_heads == 0, (
            f"key dim must be divisible by num_heads of {num_heads}"
        )
        assert self.value_dim % num_heads == 0, (
            f"value dim must be divisible by num_heads of {num_heads}"
        )

        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)
        if self.use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim_per_group,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim_per_group,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )

        self.gk_proj = nn.Sequential(
            nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
            nn.Linear(gate_low_rank_dim, self.key_dim_per_group, bias=True),
        )
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        if gate_fn == "swish" and fuse_norm and use_output_gate:
            self.g_norm_swish_gate = FusedRMSNormGated(
                hidden_size=self.head_v_dim,
                elementwise_affine=elementwise_affine,
                eps=norm_eps,
            )
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = RMSNorm(
                hidden_size=self.head_v_dim,
                elementwise_affine=elementwise_affine,
                eps=norm_eps,
            )
            self.gate_fn = ACT2FN[gate_fn]

        self.gate_logit_normalizer = gate_logit_normalizer

        self.rotary = _build_rotary(
            self.position_embedding, self.num_heads, self.head_k_dim, hidden_size
        )

        self.gk_raw_id = torch.nn.Identity()
        self.gk_post_id = torch.nn.Identity()

        self.q_id = torch.nn.Identity()
        self.k_id = torch.nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        mode = "fused_recurrent" if hidden_states.shape[1] <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices
            ).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        gk = self.gk_proj(hidden_states)

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)

        if self.rotary is not None:
            if self.position_embedding_type == "selective_rope":
                conv_state_rotary = None
                if last_state is not None and not self.rope_skip_conv:
                    conv_state_rotary = last_state["rotary_state"]
                q, k, conv_state_rotary = self.rotary(
                    q=q,
                    k=k,
                    output_final_state=use_cache,
                    cache=conv_state_rotary,
                    hidden_states=hidden_states,
                    cu_seqlens=cu_seqlens,
                )
            else:
                seqlen_offset, max_seqlen = 0, q_len
                if past_key_values is not None:
                    seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
                    max_seqlen = q.shape[1] + seqlen_offset
                if self.max_position_embeddings is not None:
                    max_seqlen = max(max_seqlen, self.max_position_embeddings)
                q, k = self.rotary(
                    q,
                    k,
                    seqlen_offset=seqlen_offset,
                    max_seqlen=max_seqlen,
                    cu_seqlens=cu_seqlens,
                )

        if self.norm_q:
            q = l2_norm(q)
        if self.norm_k:
            k = l2_norm(k)

        q = rearrange(q, "... h d -> ... (h d)")
        k = rearrange(k, "... h d -> ... (h d)")

        if self.feature_map_fn is not None:
            q, k = map(self.feature_map_fn, (q, k))
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        if self.num_kv_groups > 1:
            k, gk = (
                repeat(
                    x,
                    "... (h d) -> ... (h g) d",
                    g=self.num_kv_groups,
                    d=self.head_k_dim,
                )
                for x in (k, gk)
            )
            v = repeat(
                v, "... (h d) -> ... (h g) d", g=self.num_kv_groups, d=self.head_v_dim
            )
        else:
            k, gk = (
                rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (k, gk)
            )
            v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        # logging activations
        gk_raw = self.gk_raw_id(gk)
        gk = F.logsigmoid(gk_raw) / self.gate_logit_normalizer
        gk = self.gk_post_id(gk)

        q = self.q_id(q)
        k = self.k_id(k)

        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        if mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_gla(
                q=q,
                k=k,
                v=v,
                gk=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        elif mode == "fused_chunk":
            o, recurrent_state = fused_chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
            )
        elif mode == "chunk":
            o, recurrent_state = chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v)
                if self.use_short_conv
                else None,
                rotary_state=conv_state_rotary
                if self.position_embedding_type == "selective_rope"
                and not self.rope_skip_conv
                else None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        if self.use_output_gate:
            g = self.g_proj(hidden_states)
            if self.fuse_norm_and_gate:
                g = rearrange(g, "... (h d) -> ... h d", d=self.head_v_dim)
                o = self.g_norm_swish_gate(o, g)
                o = rearrange(o, "... h d -> ... (h d)")
            else:
                o = rearrange(self.g_norm(o), "... h d -> ... (h d)")
                o = o * self.gate_fn(g)
        else:
            o = rearrange(self.g_norm(o), "... h d -> ... (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
