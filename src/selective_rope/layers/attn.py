# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange
from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.utils.index import prepare_lens_from_mask
from transformers.utils import logging

from selective_rope.modules.rotary import SelectiveRoPE

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None

logger = logging.get_logger(__name__)


def _build_rotary(pe_config: dict, num_heads: int, head_dim: int, hidden_size: int):
    pe_type = pe_config.get("type", "nope")
    pe_kwargs = {k: v for k, v in pe_config.items() if k != "type"}
    match pe_type:
        case "selective_rope":
            return SelectiveRoPE(
                num_heads=num_heads,
                head_k_dim=head_dim,
                model_dim=hidden_size,
                **pe_kwargs,
            )
        case "rope":
            return RotaryEmbedding(
                dim=head_dim,
                base=pe_kwargs.get("theta", 500000),
            )
        case "nope" | None:
            return None
        case _:
            raise ValueError(f"Position embedding type {pe_type} not supported.")


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        max_position_embeddings: int | None = None,
        layer_idx: int | None = None,
        position_embedding: dict | None = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.window_size = window_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        self.position_embedding = position_embedding or {"type": "nope"}
        self.position_embedding_type = self.position_embedding.get("type", "nope")
        self.rope_skip_conv = self.position_embedding.get("skip_conv", False)

        if flash_attn_func is None:
            raise ImportError(
                "Please install Flash Attention via `pip install flash-attn --no-build-isolation` first"
            )

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.rotary = _build_rotary(
            self.position_embedding, self.num_heads, self.head_dim, self.hidden_size
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q = rearrange(
            self.q_proj(hidden_states), "... (h d) -> ... h d", d=self.head_dim
        )
        k = rearrange(
            self.k_proj(hidden_states), "... (h d) -> ... h d", d=self.head_dim
        )
        v = rearrange(
            self.v_proj(hidden_states), "... (h d) -> ... h d", d=self.head_dim
        )

        apply_qk_norm_pre_rotary = (
            self.qk_norm and self.position_embedding_type != "selective_rope"
        )
        if apply_qk_norm_pre_rotary:
            q, k = self.q_norm(q), self.k_norm(k)

        # equivalent to cu_seqlens in `flash_attn`
        cu_seqlens = kwargs.get("cu_seqlens", None)

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = (
                    seqlen_offset
                    + prepare_lens_from_mask(attention_mask)
                    - attention_mask.shape[-1]
                )
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        rotary_state = None
        if self.position_embedding_type == "selective_rope":
            if (
                past_key_values is not None
                and not self.rope_skip_conv
                and last_state is not None
            ):
                rotary_state = last_state.get("rotary_state", None)
            q, k, rotary_state = self.rotary(
                q=q,
                k=k,
                hidden_states=hidden_states,
                output_final_state=use_cache,
                cache=rotary_state,
                cu_seqlens=cu_seqlens,
            )
        elif self.position_embedding_type == "rope":
            q, k = self.rotary(
                q,
                k,
                seqlen_offset=seqlen_offset,
                max_seqlen=max_seqlen,
                cu_seqlens=cu_seqlens,
            )
        else:
            # nope
            pass

        if self.qk_norm and self.position_embedding_type == "selective_rope":
            q, k = self.q_norm(q), self.k_norm(k)

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )["attn_state"]
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, "... (h d) -> ... h d", d=self.head_dim)
                v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            if q.shape[1] == 1 and self.window_size is not None:
                attention_mask = attention_mask[:, -self.window_size :]
            q, (k, v), indices_q, cu_seqlens, max_seq_lens = unpad_input(
                q, (k, v), attention_mask, q_len
            )
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=(-1, -1)
                if self.window_size is None
                else (self.window_size - 1, 0),
            )
            o = pad_input(o, indices_q, batch_size, q_len)
        elif cu_seqlens is not None:
            o = flash_attn_varlen_func(
                q.squeeze(0),
                k.squeeze(0),
                v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(-1, -1)
                if self.window_size is None
                else (self.window_size - 1, 0),
            ).unsqueeze(0)
        else:
            o = flash_attn_func(
                q,
                k,
                v,
                causal=True,
                window_size=(-1, -1)
                if self.window_size is None
                else (self.window_size - 1, 0),
            )
        o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o)

        if (
            past_key_values is not None
            and self.position_embedding_type == "selective_rope"
            and not self.rope_skip_conv
            and rotary_state is not None
        ):
            past_key_values.update(
                rotary_state=rotary_state,
                layer_idx=self.layer_idx,
                offset=q_len,
            )
        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values
