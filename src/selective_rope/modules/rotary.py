from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules.convolution import ShortConvolution
from fla.modules.l2norm import l2_norm


def rotary_temperature(
    type: Literal["rfa", "rope"],
    theta: float,
    pe_dim: int,
    temp_max: float,
) -> torch.Tensor:
    match type:
        case "rfa":
            assert theta < 1.0 and theta > 0.0
            theta_tensor = torch.linspace(
                0.0,
                float(theta * torch.pi),
                steps=pe_dim // 2,
                dtype=torch.float32,
            )
            temperature = torch.sqrt(
                (1 - torch.cos(theta_tensor)) / (1 + torch.cos(theta_tensor))
            )
            return (temp_max * temperature / torch.max(temperature)).reshape(
                1, 1, 1, -1
            )
        case "rope":
            assert theta >= 1.0
            return (
                temp_max
                / (theta ** (torch.arange(0, pe_dim, 2, dtype=torch.float32) / pe_dim))
            ).reshape(1, 1, 1, -1)  # type: ignore
        case _:
            raise ValueError(f"Invalid temperature type: {type}")


class ChunkedLinear(nn.Module):
    """Per-head linear projections with weight normalization."""

    def __init__(self, in_channels: int, out_channels: int, num_heads: int):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(num_heads, in_channels, out_channels), requires_grad=True
        )
        for idx in range(num_heads):
            nn.init.kaiming_uniform_(self.weight[idx], a=0.0)

        self.scalar = nn.Parameter(torch.ones(num_heads), requires_grad=True)
        self.scalar._no_weight_decay = True

    def forward(self, x):
        weight = self.scalar.reshape(-1, 1, 1) * F.normalize(self.weight, dim=-2)
        return torch.einsum("b n i, n i o -> b n o", x, weight)


class SelectiveRoPE(nn.Module):
    def __init__(
        self,
        head_k_dim: int,
        model_dim: int,
        num_heads: int = 1,
        skip_conv: bool = False,
        dtype: torch.dtype | None = None,
        d_conv: int = 4,
        temp_type: Literal["rfa", "rope"] = "rope",
        temp_theta: float = 500000,
        temp_max: float = 1.0,
        temp_grad: bool = False,
        phi_proj_rank: int = 16,
        phi_parametrization: Literal["low-rank", "full-rank"] = "low-rank",
        conv_act: str | None = None,
        use_gate: bool = False,
        gate_parametrization: Literal["standard", "low-rank", "gla"] = "standard",
        gate_low_rank_dim: int = 16,
        use_bias: bool = False,
        use_triton: bool = False,
        fuse_gate: bool = False,
        # **kwargs,
    ):
        super().__init__()

        self.head_k_dim = head_k_dim
        self.num_heads = num_heads
        self.skip_conv = skip_conv
        self.use_triton = use_triton
        self.phi_proj_rank = phi_proj_rank
        self.phi_parametrization = phi_parametrization
        self.conv_act = conv_act
        self.use_gate = use_gate
        self.gate_parametrization = gate_parametrization
        self.gate_low_rank_dim = gate_low_rank_dim
        self.use_bias = use_bias
        self.model_dim = model_dim
        self.pe_dim = head_k_dim // 2
        self.fuse_gate = fuse_gate

        assert model_dim % num_heads == 0

        if phi_parametrization == "low-rank":
            self.phi_proj = nn.Sequential(
                nn.Linear(model_dim, phi_proj_rank, bias=False),
                nn.Linear(phi_proj_rank, num_heads * self.pe_dim, bias=False),
            )
        else:
            self.phi_proj = ChunkedLinear(
                model_dim // num_heads, self.pe_dim, num_heads
            )

        if not skip_conv:
            self.phi_conv1d = ShortConvolution(
                hidden_size=num_heads * self.pe_dim,
                kernel_size=d_conv,
                bias=False,
                activation=self.conv_act,
                dtype=dtype,
            )

        if use_bias:
            self.phi_bias = nn.Parameter(
                torch.zeros(1, 1, num_heads, self.pe_dim).float(),
                requires_grad=True,
            )

        self.temperature = nn.Parameter(
            rotary_temperature(temp_type, temp_theta, head_k_dim, temp_max),
            requires_grad=temp_grad,
        )

        if self.use_gate:
            if gate_parametrization == "standard":
                self.phase_gate_proj = nn.Linear(model_dim, num_heads, bias=True)
            elif gate_parametrization == "low-rank":
                self.phase_gate_proj = nn.Sequential(
                    nn.Linear(model_dim, gate_low_rank_dim, bias=False),
                    nn.Linear(gate_low_rank_dim, num_heads, bias=True),
                )
            elif gate_parametrization == "gla":
                self.phase_gate_proj = nn.Sequential(
                    nn.Linear(model_dim, gate_low_rank_dim, bias=False),
                    nn.Linear(gate_low_rank_dim, num_heads * self.pe_dim, bias=True),
                )
            else:
                raise ValueError(
                    f"Incorrect gate_parametrization: {gate_parametrization}"
                )

    def apply_conv(
        self,
        phi: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: None = None,
        cache: torch.Tensor | None = None,
    ):
        conv_cache = None
        if not self.skip_conv:
            phi, conv_cache = self.phi_conv1d(
                rearrange(phi, "b d t -> b t d"),
                cache=cache,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            )

            phi = rearrange(
                phi,
                "b t (h d) -> b t h d",
                h=self.num_heads,
            )
        else:
            phi = phi - torch.cat(
                [torch.zeros_like(phi[..., :1]), phi[..., :-1]], dim=-1
            )
            phi = rearrange(phi, "b (h d) t -> b t h d", h=self.num_heads)

        return phi, conv_cache

    @torch._dynamo.config.patch(cache_size_limit=32)
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        hidden_states: torch.Tensor,
        output_final_state: bool = False,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        conv_cache_in = None
        phi_tilde_offset = None
        if cache is not None:
            conv_cache_in, phi_tilde_offset = cache

        conv_cache = None

        if len(hidden_states.shape) < 4:
            hidden_states = rearrange(
                hidden_states, "b t (h d) -> b t h d", h=self.num_heads
            )

        hidden_states = l2_norm(hidden_states)

        if self.phi_parametrization == "low-rank":
            phi = rearrange(
                self.phi_proj(rearrange(hidden_states, "b t h d -> b t (h d)")),
                "b t (h d) -> b (h d) t",
                h=self.num_heads,
            )
        else:
            phi = rearrange(
                self.phi_proj(rearrange(hidden_states, "b t h d -> (b t) h d")),
                "(b t) h d -> b (h d) t",
                b=hidden_states.shape[0],
            )

        fuse_conv_generate = (
            self.use_triton
            and q.shape[1] == 1
            and not self.skip_conv
            and not self.use_bias
        )

        if fuse_conv_generate:
            phi = rearrange(phi, "b (h d) t -> b t h d", h=self.num_heads)
            conv_cache = conv_cache_in
            if conv_cache is None:
                conv_cache = phi.new_zeros(
                    phi.shape[0],
                    self.num_heads * self.pe_dim,
                    self.phi_conv1d.kernel_size[0],
                )
        else:
            phi, conv_cache = self.apply_conv(
                phi, output_final_state, cu_seqlens, conv_cache_in
            )

        if self.use_bias:
            phi = phi + torch.exp(self.phi_bias)
        if self.use_gate:
            phase_gate = self.phase_gate_proj(
                rearrange(hidden_states, "b t h d -> b t (h d)")
            ).sigmoid()

            if self.gate_parametrization == "gla":
                phase_gate = rearrange(
                    phase_gate, "b t (h d) -> b t h d", h=self.num_heads
                )
                if not self.fuse_gate:
                    phi = phi * phase_gate
            else:
                if not self.fuse_gate:
                    phi = phi * phase_gate.unsqueeze(-1)
                else:
                    phase_gate = phase_gate.unsqueeze(-1)

        if self.use_triton:
            if q.shape[1] == 1:
                from selective_rope.ops.selective_rope import selective_rope_generate

                conv_kwargs = {}
                if fuse_conv_generate:
                    conv_kwargs = dict(
                        conv_cache=conv_cache,
                        conv_weight=rearrange(self.phi_conv1d.weight, "d 1 w -> d w"),
                        conv_has_act=self.conv_act is not None,
                    )

                q_out, k_out, triton_offset = selective_rope_generate(
                    q=q,
                    k=k,
                    phi=phi,
                    temperature=self.temperature,
                    gate=phase_gate if (self.use_gate and self.fuse_gate) else None,
                    offset=phi_tilde_offset,
                    output_final_state=output_final_state,
                    **conv_kwargs,
                )
            else:
                from selective_rope.ops.selective_rope import selective_rope_triton

                q_out, k_out, triton_offset = selective_rope_triton(
                    q=q,
                    k=k,
                    phi=phi,
                    temperature=self.temperature,
                    offset=phi_tilde_offset,
                    output_final_state=output_final_state,
                )

            new_cache = None
            if output_final_state:
                new_cache = (conv_cache, triton_offset)
            return q_out, k_out, new_cache

        phi_tilde = torch.cumsum(phi, dim=1)
        if phi_tilde_offset is not None:
            phi_tilde = phi_tilde + phi_tilde_offset

        new_cache = None
        if output_final_state:
            new_phi_tilde_offset = phi_tilde[:, -1:, :, :]
            new_cache = (conv_cache, new_phi_tilde_offset)

        qk = torch.cat([q, k], dim=2).float()
        qk_0 = qk[..., : self.pe_dim]
        qk_1 = qk[..., self.pe_dim :]

        qk_phi_tilde = torch.cat([phi_tilde, phi_tilde], dim=2)
        qk_angle = self.temperature * qk_phi_tilde
        cos = torch.cos(qk_angle)
        sin = torch.sin(qk_angle)

        rotated_qk = torch.cat(
            [qk_0 * cos - qk_1 * sin, qk_0 * sin + qk_1 * cos], dim=-1
        )

        q, k = torch.split(rotated_qk.type_as(q), q.shape[2], dim=2)

        return q, k, new_cache
