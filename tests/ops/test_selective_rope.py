from contextlib import nullcontext

import pytest
import torch
from fla.utils import assert_close

from selective_rope.modules.rotary import SelectiveRoPE

DEVICE = "cuda"
BATCH, SEQLEN = 2, 64
NUM_HEADS = 4
HEAD_K_DIM = 64
MODEL_DIM = 256


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


CONFIGS = {
    "default": {},
    "full_rank": {"phi_parametrization": "full-rank"},
    "use_phase_gate": {"use_gate": True},
}

SEQLENS = [64, 2048]

# Subset for expensive long-sequence cross-path tests.
TRITON_LONG_CONFIGS = {
    "default": CONFIGS["default"],
    "use_phase_gate": CONFIGS["use_phase_gate"],
}

KERNEL_EXTRAS = {
    "triton": {"use_triton": True},
}


def _make_module(overrides: dict, dtype: torch.dtype = torch.float32) -> SelectiveRoPE:
    return (
        SelectiveRoPE(
            head_k_dim=HEAD_K_DIM,
            model_dim=MODEL_DIM,
            num_heads=NUM_HEADS,
            **overrides,
        )
        .to(DEVICE)
        .to(dtype)
    )


def _make_inputs(seqlen: int = SEQLEN, dtype: torch.dtype = torch.float32):
    q = torch.randn(BATCH, seqlen, NUM_HEADS, HEAD_K_DIM, device=DEVICE, dtype=dtype)
    k = torch.randn(BATCH, seqlen, NUM_HEADS, HEAD_K_DIM, device=DEVICE, dtype=dtype)
    hidden = torch.randn(BATCH, seqlen, MODEL_DIM, device=DEVICE, dtype=dtype)
    return q, k, hidden


def _make_pair(
    overrides: dict, extra: dict, dtype: torch.dtype = torch.float32
) -> tuple[SelectiveRoPE, SelectiveRoPE]:
    ref = _make_module(overrides, dtype)
    variant = _make_module({**overrides, **extra}, dtype)
    variant.load_state_dict(ref.state_dict())
    return ref, variant


def _amp_ctx(dtype: torch.dtype):
    # Autocast so cumsum (on the fp32 promotion list) matches Triton's fp32 chain.
    if dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


@pytest.mark.parametrize("cfg_name", CONFIGS.keys())
def test_forward_shape(cfg_name):
    module = _make_module(CONFIGS[cfg_name])
    q, k, hidden = _make_inputs()
    with torch.no_grad():
        q_out, k_out, cache = module(q, k, hidden)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert cache is None


@pytest.mark.parametrize("cfg_name", CONFIGS.keys())
def test_backward(cfg_name):
    module = _make_module(CONFIGS[cfg_name])
    q, k, hidden = _make_inputs()
    q.requires_grad_(True)
    k.requires_grad_(True)
    hidden.requires_grad_(True)
    q_out, k_out, _ = module(q, k, hidden)
    loss = q_out.sum() + k_out.sum()
    loss.backward()
    assert q.grad is not None
    assert k.grad is not None
    assert hidden.grad is not None


@pytest.mark.parametrize("cfg_name", CONFIGS.keys())
def test_output_final_state(cfg_name):
    module = _make_module(CONFIGS[cfg_name])
    q, k, hidden = _make_inputs()
    with torch.no_grad():
        q_out, k_out, cache = module(q, k, hidden, output_final_state=True)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert cache is not None
    _, phi_tilde_offset = cache
    assert phi_tilde_offset.shape == (BATCH, 1, NUM_HEADS, HEAD_K_DIM // 2)


@pytest.mark.parametrize("cfg_name", CONFIGS.keys())
def test_triton_forward_shape(cfg_name):
    module = _make_module({**CONFIGS[cfg_name], "use_triton": True})
    q, k, hidden = _make_inputs()
    with torch.no_grad():
        q_out, k_out, cache = module(q, k, hidden)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert cache is None


@pytest.mark.parametrize("kernel", KERNEL_EXTRAS.keys())
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("seqlen", SEQLENS)
@pytest.mark.parametrize("cfg_name", TRITON_LONG_CONFIGS.keys())
def test_kernel_matches_pytorch(cfg_name, seqlen, dtype, kernel):
    ref, variant = _make_pair(
        TRITON_LONG_CONFIGS[cfg_name], KERNEL_EXTRAS[kernel], dtype
    )
    q, k, hidden = _make_inputs(seqlen, dtype)
    with torch.no_grad(), _amp_ctx(dtype):
        q_ref, k_ref, _ = ref(q, k, hidden)
        q_var, k_var, _ = variant(q, k, hidden)
    assert_close(" q", q_ref, q_var, ratio=0.002)
    assert_close(" k", k_ref, k_var, ratio=0.002)


@pytest.mark.parametrize("kernel", KERNEL_EXTRAS.keys())
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("seqlen", SEQLENS)
@pytest.mark.parametrize("cfg_name", TRITON_LONG_CONFIGS.keys())
def test_kernel_output_final_state(cfg_name, seqlen, dtype, kernel):
    ref, variant = _make_pair(
        TRITON_LONG_CONFIGS[cfg_name], KERNEL_EXTRAS[kernel], dtype
    )
    q, k, hidden = _make_inputs(seqlen, dtype)
    with torch.no_grad(), _amp_ctx(dtype):
        q_ref, k_ref, cache_ref = ref(q, k, hidden, output_final_state=True)
        q_var, k_var, cache_var = variant(q, k, hidden, output_final_state=True)
    assert cache_var is not None
    _, offset_var = cache_var
    _, offset_ref = cache_ref
    assert offset_var.shape == (BATCH, 1, NUM_HEADS, HEAD_K_DIM // 2)
    assert offset_var.dtype == offset_ref.dtype
    assert_close(" q", q_ref, q_var, ratio=0.002)
    assert_close(" k", k_ref, k_var, ratio=0.002)
    assert_close("offset", offset_ref, offset_var, ratio=0.002)


@pytest.mark.parametrize(
    (
        "use_gate",
        "phi_parametrization",
        "gate_parametrization",
        "fuse_gate",
        "conv_act",
    ),
    [
        pytest.param(False, "low-rank", None, False, None, id="default"),
        pytest.param(False, "full-rank", None, False, None, id="full_rank"),
        pytest.param(True, "low-rank", None, False, None, id="gate"),
        pytest.param(
            True, "low-rank", "standard", True, None, id="fused_gate_standard"
        ),
        pytest.param(True, "low-rank", "gla", True, None, id="fused_gate_gla"),
        pytest.param(False, "low-rank", None, False, "silu", id="conv_act_silu"),
    ],
)
def test_generate(
    use_gate: bool,
    phi_parametrization: str,
    gate_parametrization: str | None,
    fuse_gate: bool,
    conv_act: str | None,
):
    "Triton sequential generate matches PyTorch prefill."
    base = {}
    if use_gate:
        base["use_gate"] = True
    if phi_parametrization != "low-rank":
        base["phi_parametrization"] = phi_parametrization
    if gate_parametrization is not None:
        base["gate_parametrization"] = gate_parametrization
    if conv_act is not None:
        base["conv_act"] = conv_act

    extra = {"use_triton": True}
    if fuse_gate:
        extra["fuse_gate"] = True

    ref, variant = _make_pair(base, extra)
    q, k, hidden = _make_inputs(SEQLEN)

    with torch.no_grad():
        q_full, k_full, _ = ref(q, k, hidden)

    q_parts, k_parts = [], []
    cache = None
    with torch.no_grad():
        for t in range(SEQLEN):
            q_t = q[:, t : t + 1]
            k_t = k[:, t : t + 1]
            h_t = hidden[:, t : t + 1]
            q_out, k_out, cache = variant(
                q_t, k_t, h_t, output_final_state=True, cache=cache
            )
            q_parts.append(q_out)
            k_parts.append(k_out)

    q_seq = torch.cat(q_parts, dim=1)
    k_seq = torch.cat(k_parts, dim=1)
    assert_close(" q", q_full, q_seq, ratio=0.002)
    assert_close(" k", k_full, k_seq, ratio=0.002)


@pytest.mark.parametrize("cfg_name", TRITON_LONG_CONFIGS.keys())
def test_triton_chunked_prefill(cfg_name):
    "Chunked triton prefill (T>1 with offset) matches single-pass prefill."
    ref, tri = _make_pair(TRITON_LONG_CONFIGS[cfg_name], {"use_triton": True})
    q, k, hidden = _make_inputs(SEQLEN)

    with torch.no_grad():
        q_full, k_full, _ = ref(q, k, hidden)

    mid = SEQLEN // 2
    with torch.no_grad():
        q1, k1, cache = tri(
            q[:, :mid], k[:, :mid], hidden[:, :mid], output_final_state=True
        )
        q2, k2, _ = tri(q[:, mid:], k[:, mid:], hidden[:, mid:], cache=cache)

    q_chunked = torch.cat([q1, q2], dim=1)
    k_chunked = torch.cat([k1, k2], dim=1)
    assert_close(" q", q_full, q_chunked, ratio=0.002)
    assert_close(" k", k_full, k_chunked, ratio=0.002)


@pytest.mark.parametrize("seqlen", SEQLENS)
@pytest.mark.parametrize("cfg_name", TRITON_LONG_CONFIGS.keys())
def test_triton_backward_matches_pytorch(cfg_name, seqlen):
    ref, tri = _make_pair(TRITON_LONG_CONFIGS[cfg_name], {"use_triton": True})
    q, k, hidden = _make_inputs(seqlen)
    q_r = q.clone().requires_grad_(True)
    k_r = k.clone().requires_grad_(True)
    h_r = hidden.clone().requires_grad_(True)
    q_t = q.clone().requires_grad_(True)
    k_t = k.clone().requires_grad_(True)
    h_t = hidden.clone().requires_grad_(True)

    q_ref, k_ref, _ = ref(q_r, k_r, h_r)
    (q_ref.sum() + k_ref.sum()).backward()

    q_tri, k_tri, _ = tri(q_t, k_t, h_t)
    (q_tri.sum() + k_tri.sum()).backward()

    assert_close("dq", q_r.grad, q_t.grad, ratio=0.002)
    assert_close("dk", k_r.grad, k_t.grad, ratio=0.002)
    assert_close("dh", h_r.grad, h_t.grad, ratio=0.002)


@pytest.mark.parametrize("cfg_name", TRITON_LONG_CONFIGS.keys())
def test_triton_param_grads(cfg_name):
    ref, tri = _make_pair(TRITON_LONG_CONFIGS[cfg_name], {"use_triton": True})
    q, k, hidden = _make_inputs()

    q_ref, k_ref, _ = ref(q, k, hidden)
    (q_ref.sum() + k_ref.sum()).backward()

    q_tri, k_tri, _ = tri(q, k, hidden)
    (q_tri.sum() + k_tri.sum()).backward()

    for (name_r, p_r), (name_t, p_t) in zip(
        ref.named_parameters(), tri.named_parameters()
    ):
        if p_r.grad is not None:
            assert p_t.grad is not None, f"{name_t} grad is None"
            assert_close(f"grad_{name_r}", p_r.grad, p_t.grad, ratio=0.005)


@pytest.mark.parametrize("cfg_name", TRITON_LONG_CONFIGS.keys())
def test_triton_grad_temperature(cfg_name):
    "Triton grad_temperature matches PyTorch when temp_grad=True."
    base = {**TRITON_LONG_CONFIGS[cfg_name], "temp_grad": True}
    ref, tri = _make_pair(base, {"use_triton": True})
    q, k, hidden = _make_inputs()

    q_ref, k_ref, _ = ref(q, k, hidden)
    (q_ref.sum() + k_ref.sum()).backward()

    q_tri, k_tri, _ = tri(q, k, hidden)
    (q_tri.sum() + k_tri.sum()).backward()

    assert ref.temperature.grad is not None
    assert tri.temperature.grad is not None
    assert_close(
        "grad_temperature", ref.temperature.grad, tri.temperature.grad, ratio=0.005
    )
