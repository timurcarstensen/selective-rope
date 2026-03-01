import pytest
import torch

from selective_rope.models.gla.configuration import GLAConfig
from selective_rope.models.gla.modeling import GLAForCausalLM

DEVICE = "cuda"
BATCH, SEQLEN = 2, 64

SMALL_MODEL = dict(
    hidden_size=256,
    num_hidden_layers=2,
    num_heads=4,
    vocab_size=256,
    intermediate_size=512,
    max_position_embeddings=512,
    fuse_cross_entropy=False,
    fuse_linear_cross_entropy=False,
    fuse_norm=False,
    fuse_swiglu=False,
)

PE_CONFIGS = {
    "nope": {"type": "nope"},
    "rope": {"type": "rope"},
    "selective_rope": {"type": "selective_rope"},
    "selective_rope_skip_conv": {"type": "selective_rope", "skip_conv": True},
    "selective_rope_full_rank": {
        "type": "selective_rope",
        "phi_parametrization": "full-rank",
    },
    "selective_rope_no_phase_gate": {
        "type": "selective_rope",
        "use_gate": False,
    },
    "selective_rope_phase_gate_low_rank": {
        "type": "selective_rope",
        "gate_parametrization": "low-rank",
    },
    "selective_rope_phase_gate_gla": {
        "type": "selective_rope",
        "gate_parametrization": "gla",
    },
    "selective_rope_with_bias": {"type": "selective_rope", "use_bias": True},
    "selective_rope_rfa_temp": {
        "type": "selective_rope",
        "temp_type": "rfa",
        "temp_theta": 0.5,
        "temp_max": 2.0,
    },
}


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


@pytest.mark.parametrize("pe_key", PE_CONFIGS.keys())
def test_gla_forward(pe_key):
    config = GLAConfig(position_embedding=PE_CONFIGS[pe_key], **SMALL_MODEL)
    model = GLAForCausalLM(config).to(DEVICE)
    input_ids = torch.randint(0, 256, (BATCH, SEQLEN), device=DEVICE)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    assert out.logits.shape == (BATCH, SEQLEN, 256)


@pytest.mark.parametrize(
    "pe_key", ["nope", "selective_rope", "selective_rope_skip_conv"]
)
def test_gla_backward(pe_key):
    config = GLAConfig(position_embedding=PE_CONFIGS[pe_key], **SMALL_MODEL)
    model = GLAForCausalLM(config).to(DEVICE)
    input_ids = torch.randint(0, 256, (BATCH, SEQLEN), device=DEVICE)
    labels = torch.randint(0, 256, (BATCH, SEQLEN), device=DEVICE)
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    assert any(p.grad is not None for p in model.parameters())
