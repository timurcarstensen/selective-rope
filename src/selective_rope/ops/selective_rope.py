import os

import torch
import triton
import triton.language as tl
from fla.utils import IS_AMD, autotune_cache_kwargs, input_guard
from torch.library import triton_op, wrap_triton

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [2, 4, 8, 16, 32]

_TESTING = os.environ.get("SELECTIVE_ROPE_TESTING", "0") == "1"


def _autotune_configs(configs):
    """Use a single config during testing to skip autotuning."""
    return configs[:1] if _TESTING else configs


@triton.jit
def sincosf_sfu(x_fp32):
    s, c = tl.inline_asm_elementwise(
        asm="{ sin.approx.f32 $0, $2;  cos.approx.f32 $1, $2; }",
        constraints="=f,=f,f",
        args=[x_fp32],
        dtype=(tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )
    return s, c


@triton.jit
def selective_rope_generate_kernel(
    q,
    k,
    phi,
    temp,
    gate,
    offset,
    y1,
    y2,
    new_offset,
    conv_cache,
    conv_weight,
    stride_qk_b,
    stride_qk_h,
    stride_qk_d,
    stride_phi_b,
    stride_phi_h,
    stride_phi_d,
    stride_off_b,
    stride_off_h,
    stride_off_d,
    stride_gate_b,
    stride_gate_h,
    stride_gate_d,
    stride_cc_b,
    stride_cc_d,
    stride_cc_w,
    stride_cw_d,
    stride_cw_w,
    FUSE_GATE: tl.constexpr,
    R: tl.constexpr,
    BD: tl.constexpr,
    HAS_OFFSET: tl.constexpr,
    OUTPUT_OFFSET: tl.constexpr,
    FUSE_CONV: tl.constexpr,
    BW: tl.constexpr,
    CONV_W: tl.constexpr,
    CONV_HAS_ACT: tl.constexpr,
):
    i_b = tl.program_id(0)
    i_h = tl.program_id(1)

    o_r = tl.arange(0, BD // 2)
    r_mask = o_r < R

    # Load phi[b, 0, h, :R]
    phi_ptr = phi + i_b * stride_phi_b + i_h * stride_phi_h + o_r * stride_phi_d
    b_phi = tl.load(phi_ptr, mask=r_mask, other=0.0).to(tl.float32)

    if FUSE_CONV:
        conv_ch = i_h * R + o_r
        w_idx = tl.arange(0, BW)

        # Read shifted cache: cache[ch, w+1] for w=0..W-2, then phi at W-1
        cache_base = conv_cache + i_b * stride_cc_b
        shifted_ptrs = (
            cache_base
            + conv_ch[:, None] * stride_cc_d
            + (w_idx[None, :] + 1) * stride_cc_w
        )
        shift_mask = r_mask[:, None] & (w_idx[None, :] < (CONV_W - 1))
        b_cache = tl.load(shifted_ptrs, mask=shift_mask, other=0.0).to(tl.float32)
        insert_mask = w_idx[None, :] == (CONV_W - 1)
        b_cache = tl.where(insert_mask, b_phi[:, None], b_cache)
        cache_ptrs = (
            cache_base + conv_ch[:, None] * stride_cc_d + w_idx[None, :] * stride_cc_w
        )
        w_mask = r_mask[:, None] & (w_idx[None, :] < CONV_W)
        tl.store(cache_ptrs, b_cache, mask=w_mask)

        weight_ptrs = (
            conv_weight + conv_ch[:, None] * stride_cw_d + w_idx[None, :] * stride_cw_w
        )
        b_weight = tl.load(weight_ptrs, mask=w_mask, other=0.0).to(tl.float32)
        b_phi = tl.sum(b_cache * b_weight, axis=1)

        if CONV_HAS_ACT:
            b_phi = b_phi * tl.sigmoid(b_phi)

    if FUSE_GATE:
        gate_ptr = (
            gate + i_b * stride_gate_b + i_h * stride_gate_h + o_r * stride_gate_d
        )
        b_gate = tl.load(gate_ptr, mask=r_mask, other=0.0).to(tl.float32)
        b_phi = b_phi * b_gate

    if HAS_OFFSET:
        off_ptr = offset + i_b * stride_off_b + i_h * stride_off_h + o_r * stride_off_d
        b_off = tl.load(off_ptr, mask=r_mask, other=0.0).to(tl.float32)
        b_phi = b_phi + b_off

    temp_ptr = temp + o_r
    b_temp = tl.load(temp_ptr, mask=r_mask, other=0.0).to(tl.float32)
    b_angle = b_phi * b_temp

    b_sin, b_cos = tl.sin(b_angle), tl.cos(b_angle)

    q_base = q + i_b * stride_qk_b + i_h * stride_qk_h
    b_q0 = tl.load(q_base + o_r * stride_qk_d, mask=r_mask, other=0.0).to(tl.float32)
    b_q1 = tl.load(q_base + (o_r + R) * stride_qk_d, mask=r_mask, other=0.0).to(
        tl.float32
    )

    k_base = k + i_b * stride_qk_b + i_h * stride_qk_h
    b_k0 = tl.load(k_base + o_r * stride_qk_d, mask=r_mask, other=0.0).to(tl.float32)
    b_k1 = tl.load(k_base + (o_r + R) * stride_qk_d, mask=r_mask, other=0.0).to(
        tl.float32
    )

    b_oq0 = b_q0 * b_cos - b_q1 * b_sin
    b_oq1 = b_q0 * b_sin + b_q1 * b_cos
    b_ok0 = b_k0 * b_cos - b_k1 * b_sin
    b_ok1 = b_k0 * b_sin + b_k1 * b_cos

    y1_base = y1 + i_b * stride_qk_b + i_h * stride_qk_h
    tl.store(y1_base + o_r * stride_qk_d, b_oq0, mask=r_mask)
    tl.store(y1_base + (o_r + R) * stride_qk_d, b_oq1, mask=r_mask)

    y2_base = y2 + i_b * stride_qk_b + i_h * stride_qk_h
    tl.store(y2_base + o_r * stride_qk_d, b_ok0, mask=r_mask)
    tl.store(y2_base + (o_r + R) * stride_qk_d, b_ok1, mask=r_mask)

    if OUTPUT_OFFSET:
        no_ptr = (
            new_offset + i_b * stride_off_b + i_h * stride_off_h + o_r * stride_off_d
        )
        tl.store(no_ptr, b_phi, mask=r_mask)


@triton_op("selective_rope::generate", mutates_args=("conv_cache",))
@input_guard  # type: ignore
def selective_rope_generate(
    q: torch.Tensor,
    k: torch.Tensor,
    phi: torch.Tensor,
    temperature: torch.Tensor,
    gate: torch.Tensor | None = None,
    offset: torch.Tensor | None = None,
    output_final_state: bool = False,
    conv_cache: torch.Tensor | None = None,
    conv_weight: torch.Tensor | None = None,
    conv_has_act: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, _, H, D = q.shape
    R = phi.shape[-1]
    BD = triton.next_power_of_2(D)

    y1 = torch.empty_like(q)
    y2 = torch.empty_like(k)
    new_offset = torch.empty(B, 1, H, R, dtype=torch.float32, device=q.device)

    qk_strides = q.stride()
    phi_strides = phi.stride()

    if offset is not None:
        off_strides = offset.stride()
    else:
        off_strides = new_offset.stride()

    if gate is not None:
        gate = gate.expand_as(phi)
        gate_strides = gate.stride()
    else:
        gate_strides = phi_strides

    fuse_conv = conv_cache is not None and conv_weight is not None
    if fuse_conv:
        cc_strides = conv_cache.stride()
        cw_strides = conv_weight.stride()
        W = conv_cache.shape[-1]
        BW = triton.next_power_of_2(W)
    else:
        cc_strides = (0, 0, 0)
        cw_strides = (0, 0)
        W = 0
        BW = 1

    grid = (B, H)
    wrap_triton(selective_rope_generate_kernel)[grid](
        q=q,
        k=k,
        phi=phi,
        temp=temperature,
        gate=gate,
        offset=offset,
        y1=y1,
        y2=y2,
        new_offset=new_offset,
        conv_cache=conv_cache,
        conv_weight=conv_weight,
        stride_qk_b=qk_strides[0],
        stride_qk_h=qk_strides[2],
        stride_qk_d=qk_strides[3],
        stride_phi_b=phi_strides[0],
        stride_phi_h=phi_strides[2],
        stride_phi_d=phi_strides[3],
        stride_off_b=off_strides[0],
        stride_off_h=off_strides[2],
        stride_off_d=off_strides[3],
        stride_gate_b=gate_strides[0],
        stride_gate_h=gate_strides[2],
        stride_gate_d=gate_strides[3],
        stride_cc_b=cc_strides[0],
        stride_cc_d=cc_strides[1],
        stride_cc_w=cc_strides[2],
        stride_cw_d=cw_strides[0],
        stride_cw_w=cw_strides[1],
        FUSE_CONV=fuse_conv,
        BW=BW,
        CONV_W=W,
        CONV_HAS_ACT=conv_has_act,
        FUSE_GATE=gate is not None,
        R=R,
        BD=BD,
        HAS_OFFSET=offset is not None,
        OUTPUT_OFFSET=output_final_state,
    )

    return y1, y2, new_offset


@triton.autotune(
    configs=_autotune_configs(
        [
            triton.Config(
                {"BLOCK_T": bt, "BLOCK_R": br, "PIPELINE_STAGES": ps},
                num_warps=nw,
                num_stages=ns,
            )
            for bt in [32, 64, 128]
            for br in [16, 32, 64]
            for nw in NUM_WARPS_AUTOTUNE
            for ns in [2, 4]
            for ps in [2, 3, 4]
        ]
    ),
    key=["H", "R"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def selective_rope_fwd_kernel(
    phi_ptr,
    q_ptr,
    k_ptr,
    temp_ptr,
    offset_ptr,
    q_out_ptr,
    k_out_ptr,
    final_state_ptr,
    angle_ptr,
    T: tl.constexpr,
    R: tl.constexpr,
    H: tl.constexpr,
    stride_phi_b,
    stride_phi_t,
    stride_phi_h,
    stride_phi_d,
    stride_qk_b,
    stride_qk_t,
    stride_qk_h,
    stride_qk_d,
    stride_off_b,
    stride_off_h,
    stride_off_d,
    stride_angle_b,
    stride_angle_t,
    stride_angle_h,
    stride_angle_d,
    BLOCK_T: tl.constexpr,
    BLOCK_R: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    HAS_TEMP: tl.constexpr,
    HAS_OFFSET: tl.constexpr,
    OUTPUT_FINAL_STATE: tl.constexpr,
    STORE_ANGLE: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b_idx = pid_bh // H
    h_idx = pid_bh - b_idx * H

    r_idx = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = r_idx < R

    phi_base = phi_ptr + b_idx * stride_phi_b + h_idx * stride_phi_h
    q_base = q_ptr + b_idx * stride_qk_b + h_idx * stride_qk_h
    k_base = k_ptr + b_idx * stride_qk_b + h_idx * stride_qk_h
    qo_base = q_out_ptr + b_idx * stride_qk_b + h_idx * stride_qk_h
    ko_base = k_out_ptr + b_idx * stride_qk_b + h_idx * stride_qk_h

    if STORE_ANGLE:
        angle_base = angle_ptr + b_idx * stride_angle_b + h_idx * stride_angle_h

    t_vec = tl.arange(0, BLOCK_T)
    t_tiles = (T + BLOCK_T - 1) // BLOCK_T

    carry = tl.zeros([BLOCK_R], dtype=tl.float32)

    if HAS_OFFSET:
        off_ptrs = (
            offset_ptr
            + b_idx * stride_off_b
            + h_idx * stride_off_h
            + r_idx * stride_off_d
        )
        carry = tl.load(off_ptrs, mask=r_mask, other=0).to(tl.float32)

    if HAS_TEMP:
        temp = tl.load(temp_ptr + r_idx, mask=r_mask, other=0).to(tl.float32)

    for tile in tl.range(0, t_tiles, num_stages=PIPELINE_STAGES):
        t = tile * BLOCK_T
        t_idx = t + t_vec
        mask = r_mask[None, :] & (t_idx[:, None] < T)

        phi_ptrs = (
            phi_base + t_idx[:, None] * stride_phi_t + r_idx[None, :] * stride_phi_d
        )
        b_phi = tl.load(phi_ptrs, mask=mask, other=0).to(tl.float32)

        s = tl.cumsum(b_phi, axis=0) + carry[None, :]

        if HAS_TEMP:
            s = s * temp

        if STORE_ANGLE:
            a_ptrs = (
                angle_base
                + t_idx[:, None] * stride_angle_t
                + r_idx[None, :] * stride_angle_d
            )
            tl.store(a_ptrs, s, mask=mask)

        b_sin, b_cos = sincosf_sfu(s)

        q0_ptrs = q_base + t_idx[:, None] * stride_qk_t + r_idx[None, :] * stride_qk_d
        b_q0 = tl.load(q0_ptrs, mask=mask, other=0).to(tl.float32)
        b_q1 = tl.load(q0_ptrs + R * stride_qk_d, mask=mask, other=0).to(tl.float32)

        b_oq0 = b_q0 * b_cos - b_q1 * b_sin
        b_oq1 = b_q0 * b_sin + b_q1 * b_cos

        qo_ptrs = qo_base + t_idx[:, None] * stride_qk_t + r_idx[None, :] * stride_qk_d
        tl.store(qo_ptrs, b_oq0, mask=mask)
        tl.store(qo_ptrs + R * stride_qk_d, b_oq1, mask=mask)

        k0_ptrs = k_base + t_idx[:, None] * stride_qk_t + r_idx[None, :] * stride_qk_d
        b_k0 = tl.load(k0_ptrs, mask=mask, other=0).to(tl.float32)
        b_k1 = tl.load(k0_ptrs + R * stride_qk_d, mask=mask, other=0).to(tl.float32)

        b_ok0 = b_k0 * b_cos - b_k1 * b_sin
        b_ok1 = b_k0 * b_sin + b_k1 * b_cos

        ko_ptrs = ko_base + t_idx[:, None] * stride_qk_t + r_idx[None, :] * stride_qk_d
        tl.store(ko_ptrs, b_ok0, mask=mask)
        tl.store(ko_ptrs + R * stride_qk_d, b_ok1, mask=mask)

        # Update carry (raw cumsum, without temperature)
        carry = tl.where(r_mask, carry + tl.sum(b_phi, axis=0), carry)

    if OUTPUT_FINAL_STATE:
        fs_ptrs = (
            final_state_ptr
            + b_idx * stride_off_b
            + h_idx * stride_off_h
            + r_idx * stride_off_d
        )
        tl.store(fs_ptrs, carry, mask=r_mask)


@triton.autotune(
    configs=_autotune_configs(
        [
            triton.Config(
                {"BLOCK_T": bt, "BLOCK_R": br, "PIPELINE_STAGES": ps},
                num_warps=nw,
                num_stages=ns,
            )
            for bt in [64, 128]
            for br in [16, 32]
            for nw in NUM_WARPS_AUTOTUNE
            for ns in [2, 4]
            for ps in [2, 3, 4]
        ]
    ),
    key=["H", "R"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def selective_rope_bwd_fused_kernel(
    grad_q_out_ptr,
    grad_k_out_ptr,
    q_ptr,
    k_ptr,
    angle_ptr,
    temp_ptr,
    phi_ptr,
    grad_q_ptr,
    grad_k_ptr,
    grad_phi_ptr,
    grad_temp_ptr,
    T: tl.constexpr,
    R: tl.constexpr,
    H: tl.constexpr,
    stride_qk_b,
    stride_qk_t,
    stride_qk_h,
    stride_qk_d,
    stride_gqk_b,
    stride_gqk_t,
    stride_gqk_h,
    stride_gqk_d,
    stride_angle_b,
    stride_angle_t,
    stride_angle_h,
    stride_angle_d,
    stride_phi_b,
    stride_phi_t,
    stride_phi_h,
    stride_phi_d,
    BLOCK_T: tl.constexpr,
    BLOCK_R: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    HAS_TEMP: tl.constexpr,
    ACCUMULATE_GRAD_TEMP: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b_idx = pid_bh // H
    h_idx = pid_bh - b_idx * H

    r_idx = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = r_idx < R

    gqo_base = grad_q_out_ptr + b_idx * stride_gqk_b + h_idx * stride_gqk_h
    gko_base = grad_k_out_ptr + b_idx * stride_gqk_b + h_idx * stride_gqk_h
    q_base = q_ptr + b_idx * stride_qk_b + h_idx * stride_qk_h
    k_base = k_ptr + b_idx * stride_qk_b + h_idx * stride_qk_h
    angle_base = angle_ptr + b_idx * stride_angle_b + h_idx * stride_angle_h
    gq_base = grad_q_ptr + b_idx * stride_gqk_b + h_idx * stride_gqk_h
    gk_base = grad_k_ptr + b_idx * stride_gqk_b + h_idx * stride_gqk_h
    gphi_base = grad_phi_ptr + b_idx * stride_angle_b + h_idx * stride_angle_h

    if ACCUMULATE_GRAD_TEMP:
        phi_base = phi_ptr + b_idx * stride_phi_b + h_idx * stride_phi_h
        acc_grad_temp = tl.zeros([BLOCK_R], dtype=tl.float32)

    t_vec = tl.arange(0, BLOCK_T)
    t_tiles = (T + BLOCK_T - 1) // BLOCK_T
    carry = tl.zeros([BLOCK_R], dtype=tl.float32)

    if HAS_TEMP:
        temp = tl.load(temp_ptr + r_idx, mask=r_mask, other=0).to(tl.float32)

    for i in tl.range(0, t_tiles, num_stages=PIPELINE_STAGES):
        tile = t_tiles - 1 - i
        t = tile * BLOCK_T
        t_rev = t + (BLOCK_T - 1 - t_vec)
        mask = r_mask[None, :] & (t_rev[:, None] < T)

        a_ptrs = (
            angle_base
            + t_rev[:, None] * stride_angle_t
            + r_idx[None, :] * stride_angle_d
        )
        b_angle = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)
        b_sin, b_cos = sincosf_sfu(b_angle)

        gqo_ptrs = (
            gqo_base + t_rev[:, None] * stride_gqk_t + r_idx[None, :] * stride_gqk_d
        )
        b_gq0 = tl.load(gqo_ptrs, mask=mask, other=0.0).to(tl.float32)
        b_gq1 = tl.load(gqo_ptrs + R * stride_gqk_d, mask=mask, other=0.0).to(
            tl.float32
        )
        gq_ptrs = (
            gq_base + t_rev[:, None] * stride_gqk_t + r_idx[None, :] * stride_gqk_d
        )
        tl.store(gq_ptrs, b_gq0 * b_cos + b_gq1 * b_sin, mask=mask)
        tl.store(gq_ptrs + R * stride_gqk_d, -b_gq0 * b_sin + b_gq1 * b_cos, mask=mask)

        q_ptrs = q_base + t_rev[:, None] * stride_qk_t + r_idx[None, :] * stride_qk_d
        b_q0 = tl.load(q_ptrs, mask=mask, other=0.0).to(tl.float32)
        b_q1 = tl.load(q_ptrs + R * stride_qk_d, mask=mask, other=0.0).to(tl.float32)
        b_grad_cos = b_gq0 * b_q0 + b_gq1 * b_q1
        b_grad_sin = -b_gq0 * b_q1 + b_gq1 * b_q0

        gko_ptrs = (
            gko_base + t_rev[:, None] * stride_gqk_t + r_idx[None, :] * stride_gqk_d
        )
        b_gk0 = tl.load(gko_ptrs, mask=mask, other=0.0).to(tl.float32)
        b_gk1 = tl.load(gko_ptrs + R * stride_gqk_d, mask=mask, other=0.0).to(
            tl.float32
        )
        gk_ptrs = (
            gk_base + t_rev[:, None] * stride_gqk_t + r_idx[None, :] * stride_gqk_d
        )
        tl.store(gk_ptrs, b_gk0 * b_cos + b_gk1 * b_sin, mask=mask)
        tl.store(gk_ptrs + R * stride_gqk_d, -b_gk0 * b_sin + b_gk1 * b_cos, mask=mask)

        k_ptrs = k_base + t_rev[:, None] * stride_qk_t + r_idx[None, :] * stride_qk_d
        b_k0 = tl.load(k_ptrs, mask=mask, other=0.0).to(tl.float32)
        b_k1 = tl.load(k_ptrs + R * stride_qk_d, mask=mask, other=0.0).to(tl.float32)
        b_grad_cos += b_gk0 * b_k0 + b_gk1 * b_k1
        b_grad_sin += -b_gk0 * b_k1 + b_gk1 * b_k0

        b_grad_angle = b_grad_sin * b_cos - b_grad_cos * b_sin

        if HAS_TEMP:
            b_grad_angle = b_grad_angle * temp[None, :]

        cs = tl.cumsum(b_grad_angle, axis=0) + carry[None, :]
        gphi_ptrs = (
            gphi_base
            + t_rev[:, None] * stride_angle_t
            + r_idx[None, :] * stride_angle_d
        )
        tl.store(gphi_ptrs, cs.to(gphi_base.dtype.element_ty), mask=mask)
        carry = carry + tl.sum(b_grad_angle, axis=0)

        if ACCUMULATE_GRAD_TEMP:
            phi_ptrs = (
                phi_base + t_rev[:, None] * stride_phi_t + r_idx[None, :] * stride_phi_d
            )
            b_phi = tl.load(phi_ptrs, mask=mask, other=0.0).to(tl.float32)
            acc_grad_temp += tl.sum(b_phi * cs, axis=0)

    if ACCUMULATE_GRAD_TEMP:
        acc_grad_temp = acc_grad_temp / temp
        gt_ptrs = grad_temp_ptr + r_idx
        tl.atomic_add(gt_ptrs, acc_grad_temp, mask=r_mask)


class SelectiveRoPEFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx, q, k, phi, temperature, offset, output_final_state, grad_enabled: bool
    ):
        assert phi.ndim == 4, "expected phi of shape [B, T, H, R]"
        B, T, H, R = phi.shape

        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)

        final_state = None
        if output_final_state:
            final_state = torch.empty(B, H, R, dtype=torch.float32, device=q.device)

        angle_out = None
        if grad_enabled:
            angle_out = torch.empty(B, T, H, R, dtype=torch.float32, device=q.device)

        # offset_squeezed = None
        if offset is not None:
            offset = offset.squeeze(1)
            off_strides = offset.stride()

        elif final_state is not None:
            off_strides = final_state.stride()
        else:
            off_strides = (H * R, R, 1)

        if angle_out is not None:
            angle_strides = angle_out.stride()
        else:
            angle_strides = (T * H * R, H * R, R, 1)

        phi_strides = phi.stride()
        qk_strides = q.stride()

        def grid(meta):
            return (triton.cdiv(R, meta["BLOCK_R"]), B * H)

        selective_rope_fwd_kernel[grid](
            phi_ptr=phi,
            q_ptr=q,
            k_ptr=k,
            temp_ptr=temperature,
            offset_ptr=offset,
            q_out_ptr=q_out,
            k_out_ptr=k_out,
            final_state_ptr=final_state,
            angle_ptr=angle_out,
            T=T,
            R=R,
            H=H,
            stride_phi_b=phi_strides[0],
            stride_phi_t=phi_strides[1],
            stride_phi_h=phi_strides[2],
            stride_phi_d=phi_strides[3],
            stride_qk_b=qk_strides[0],
            stride_qk_t=qk_strides[1],
            stride_qk_h=qk_strides[2],
            stride_qk_d=qk_strides[3],
            stride_off_b=off_strides[0],
            stride_off_h=off_strides[1],
            stride_off_d=off_strides[2],
            stride_angle_b=angle_strides[0],
            stride_angle_t=angle_strides[1],
            stride_angle_h=angle_strides[2],
            stride_angle_d=angle_strides[3],
            HAS_TEMP=temperature is not None,
            HAS_OFFSET=offset is not None,
            OUTPUT_FINAL_STATE=output_final_state,
            STORE_ANGLE=grad_enabled,
        )

        if final_state is not None:
            final_state = final_state.unsqueeze(1)

        if grad_enabled:
            ctx.save_for_backward(q, k, phi, temperature, angle_out)

        return q_out, k_out, final_state

    @staticmethod
    @input_guard
    def backward(ctx, grad_q_out, grad_k_out, grad_final_state):
        q, k, phi, temperature, angle = ctx.saved_tensors

        B, T, H, R = phi.shape

        grad_q = torch.empty_like(grad_q_out)
        grad_k = torch.empty_like(grad_k_out)
        grad_phi = torch.empty_like(phi)

        has_temp = temperature is not None
        grad_temperature = None
        if has_temp:
            grad_temperature = torch.zeros(R, dtype=torch.float32, device=phi.device)

        qk_strides = q.stride()
        gqk_strides = grad_q_out.stride()
        angle_strides = angle.stride()
        phi_strides = phi.stride()

        def grid(meta):
            return (triton.cdiv(R, meta["BLOCK_R"]), B * H)

        selective_rope_bwd_fused_kernel[grid](
            grad_q_out_ptr=grad_q_out,
            grad_k_out_ptr=grad_k_out,
            q_ptr=q,
            k_ptr=k,
            angle_ptr=angle,
            temp_ptr=temperature,
            phi_ptr=phi,
            grad_q_ptr=grad_q,
            grad_k_ptr=grad_k,
            grad_phi_ptr=grad_phi,
            grad_temp_ptr=grad_temperature,
            T=T,
            R=R,
            H=H,
            stride_qk_b=qk_strides[0],
            stride_qk_t=qk_strides[1],
            stride_qk_h=qk_strides[2],
            stride_qk_d=qk_strides[3],
            stride_gqk_b=gqk_strides[0],
            stride_gqk_t=gqk_strides[1],
            stride_gqk_h=gqk_strides[2],
            stride_gqk_d=gqk_strides[3],
            stride_angle_b=angle_strides[0],
            stride_angle_t=angle_strides[1],
            stride_angle_h=angle_strides[2],
            stride_angle_d=angle_strides[3],
            stride_phi_b=phi_strides[0],
            stride_phi_t=phi_strides[1],
            stride_phi_h=phi_strides[2],
            stride_phi_d=phi_strides[3],
            HAS_TEMP=has_temp,
            ACCUMULATE_GRAD_TEMP=has_temp,
        )

        if grad_temperature is not None:
            grad_temperature = grad_temperature.reshape_as(temperature)

        return (
            grad_q,
            grad_k,
            grad_phi,
            grad_temperature,
            None,
            None,
            None,
        )


@torch.compiler.disable
def selective_rope_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    phi: torch.Tensor,
    temperature: torch.Tensor,
    offset: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, D = phi.shape

    q_out, k_out, final_state = SelectiveRoPEFunction.apply(
        q, k, phi, temperature, offset, output_final_state, torch.is_grad_enabled()
    )
    return q_out, k_out, final_state


def precompute_selective_rope_weights(
    dim: int,
    theta: float = 500000.0,
) -> torch.Tensor:
    return 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    ).reshape(1, 1, 1, -1)
