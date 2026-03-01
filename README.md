# Selective Rotary Position Embeddings

This repository contains the official implementation of the paper [**Selective Rotary Position Embedding**](https://openreview.net/forum?id=AQo1SEElNb) (ICLR 2026).

> We introduce Selective RoPE, an input-dependent rotary embedding that enhances gated linear transformers. We demonstrate that softmax attention performs implicit rotations on query-key pairs and show how real and imaginary components in state-space models manage forgetting and positional encoding respectively. Our method improves performance on language modeling and sequence tasks including copying, state tracking, and retrieval.

## Installation

Make sure you have [`uv`](https://docs.astral.sh/uv/) installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and sync the environment:
```bash
git clone https://github.com/<org>/selective-rope.git
cd selective-rope
uv sync -p 3.12
```

## Usage

All commands should be run through `uv run`:

```bash
uv run --frozen --no-sync python <script.py>
```

### Instantiating a model

```python
import torch
from selective_rope.models.gla import GLAConfig, GLAForCausalLM

config = GLAConfig(
    hidden_size=512,
    num_hidden_layers=4,
    num_heads=4,
    vocab_size=32000,
    position_embedding={"type": "selective_rope", "phi_proj_rank": 32},
)
model = GLAForCausalLM(config).cuda()

input_ids = torch.randint(0, config.vocab_size, (1, 128)).cuda()
with torch.no_grad():
    output = model(input_ids)
print(output.logits.shape)  # (1, 128, 32000)

# Autoregressive generation
output_ids = model.generate(input_ids, max_new_tokens=20)
```

Set `position_embedding={"type": "nope"}` or `{"type": "rope"}` to use the baseline variants instead.

### Before scheduling experiments

Before running any scheduling script (`schedule_lm.py`, `schedule_mad.py`, etc.), fill in the following placeholders that were left `null` for the public release:

| File | Field | Description |
|------|-------|-------------|
| `configs/language_modeling/cluster/capella.yaml` | `data_home` | Path to tokenized training data |
| `configs/language_modeling/language_modeling.yaml` | `logger.wandb_project` | WandB project name |
| `configs/*/logger/logger.yaml` | `wandb_entity` | WandB entity (user or team) |

For the evaluation script (`configs/language_modeling/scripts/capella_eval.sh`), set `HF_HOME` and `HF_DATASETS_CACHE` if you want HuggingFace data cached outside the default `~/.cache/huggingface`.

## Reproducing Experiments

### Language Modeling

Training is launched via SLURM using the scheduling script, which creates isolated git worktree snapshots of the current code state:

```bash
uv run --frozen --no-sync python schedule_lm.py
```

Configuration is Hydra-based. Position embedding variants are swapped via config groups:
- `model/position_embedding=nope` — no positional embedding (baseline)
- `model/position_embedding=rope` — standard RoPE
- `model/position_embedding=selective_rope` — Selective RoPE (ours)

See `configs/language_modeling/` for the full configuration structure.

### MAD Benchmark

First, pre-generate the data:
```bash
uv run --frozen --no-sync python synthetic_tasks/mad/generate_data.py \
    --data-path ./data/mad --num-workers 8
```

Then schedule training:
```bash
uv run --frozen --no-sync python synthetic_tasks/mad/schedule_mad.py
```

### Copying & State Tracking

These follow the same Hydra + SLURM pattern. See `configs/copying/` and `configs/state_tracking/` for configurations.

## Project Structure

```
src/selective_rope/
├── modules/rotary.py          # SelectiveRoPE: input-dependent frequency selection
├── ops/selective_rope.py      # Triton kernels (forward pass only)
├── layers/
│   ├── gla.py                 # Gated Linear Attention with SelectiveRoPE
│   ├── gated_deltanet.py      # Gated DeltaNet (comparison architecture)
│   └── attn.py                # Standard multi-head attention with RoPE
└── models/                    # Full causal LM implementations (HuggingFace-compatible)
    ├── gla/
    ├── gated_deltanet/
    └── transformer/

configs/                       # Hydra configs organized by task
evals/                         # Evaluation harness and benchmarks
tests/                         # Unit tests
synthetic_tasks/               # MAD, Zoology, copying, state tracking
train_language_modeling.py      # Training entry point (launched by schedule_lm.py)
```

## Citation

```bibtex
@inproceedings{movahedi2026selective,
  title={Selective Rotary Position Embedding},
  author={Movahedi, Sajad and Carstensen, Timur and Afzal, Arshia and Hutter, Frank and Orvieto, Antonio and Cevher, Volkan},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=AQo1SEElNb}
}
```

## Acknowledgments

The synthetic evaluation tasks build on [MAD-Lab](https://github.com/athms/mad-lab), [Zoology](https://github.com/HazyResearch/zoology), and [DeltaProduct](https://github.com/automl/DeltaProduct).

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.
