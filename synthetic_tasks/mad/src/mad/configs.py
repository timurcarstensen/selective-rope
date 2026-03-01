import os
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf
from torch import nn

from mad.data.instances import generate_kv_map
from mad.paths import get_base_path, make_dataset_path
from mad.registry import layer_registry, model_registry, task_registry

# Root of the MAD package (synthetic_tasks/mad/) for resolving registry paths
_MAD_ROOT = Path(__file__).resolve().parents[2]


def load_yml(path):
    """Helper function to load a yaml file."""
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class BaseConfig:
    def update_from_kwargs(self, kwargs):
        """Update fields of the config with kwargs."""
        valid_keys = {field.name for field in fields(self)}
        for key, value in kwargs.items():
            if key in valid_keys:
                setattr(self, key, value)


@dataclass
class MADConfig(BaseConfig):
    """MAD configuration."""

    # task settings:
    task: str = "in-context-recall"
    vocab_size: int = 16
    seq_len: int = 128
    frac_noise: float = 0.0
    noise_vocab_size: int = 0
    num_tokens_to_copy: int = 0
    k_motif_size: int = 1
    v_motif_size: int = 1
    multi_query: bool = True
    num_train_examples: int = 12_800
    num_test_examples: int = 1_280

    # data settings:
    data_path: str = "./data/mad"
    num_data_workers: int = 0
    persistent_data_workers: bool = True

    # training settings:
    batch_size: int = 128
    epochs: int = 200
    lr: float = 5e-4
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    min_lr: float = 1e-6
    early_stop: bool = True
    stop_patience: int = 20
    plateau_patience: int = 5
    plateau_factor: float = 0.9
    accelerator: str = "cuda"
    devices: int = 1
    save_checkpoints: bool = True
    precision: str = "bf16"

    # misc:
    seed: int = 12345
    target_ignore_index: int = -100

    @property
    def instance_fn(self) -> Callable:
        """returns function from registry used to generate an instance of the task"""
        if self.task in task_registry:
            return task_registry[self.task]["instance_fn"]
        else:
            return None

    @property
    def instance_fn_kwargs(self) -> dict:
        """returns dict of all kwargs required to create an instance with self.instance_fn"""
        if self.task == "memorization":
            # We need to generate a kv_map for the memorization task.
            # As this mapping is fixed, we can generate it here,
            # avoiding that it is recreated every time a new task instance is created.
            if self.k_motif_size > 1 or self.v_motif_size > 1:
                print(
                    "/!\ setting {k,v}_motif_size to 1, as motifs>1 are not supported for the memorization task."
                )
            kv_map = generate_kv_map(
                vocab_size=self.vocab_size - 1,  # also account for insert tokens
                k_motif_size=1,
                v_motif_size=1,
                seed=self.seed,
            )
        else:
            kv_map = None
        return dict(
            vocab_size=self.vocab_size,
            seq_len=self.seq_len,
            k_motif_size=self.k_motif_size,
            v_motif_size=self.v_motif_size,
            frac_noise=self.frac_noise,
            noise_vocab_size=self.noise_vocab_size,
            num_tokens_to_copy=self.num_tokens_to_copy,
            rng=np.random.default_rng(self.seed),
            multi_query=self.multi_query,
            kv_map=kv_map,
        )

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "MADConfig":
        """Build MADConfig from a composed Hydra DictConfig."""
        mad_cfg = cls()
        mad_cfg.update_from_kwargs(OmegaConf.to_container(cfg.task, resolve=True))
        mad_cfg.update_from_kwargs(OmegaConf.to_container(cfg.trainer, resolve=True))
        # top-level overrides
        for key in ("data_path", "save_checkpoints"):
            if key in cfg:
                setattr(mad_cfg, key, cfg[key])
        return mad_cfg

    @property
    def dataset_path(self):
        return make_dataset_path(self)

    @property
    def train_dataset_path(self) -> str:
        return os.path.join(self.dataset_path, "train")

    @property
    def test_dataset_path(self) -> str:
        return os.path.join(self.dataset_path, "test")


def build_model_from_hydra_config(cfg: DictConfig):
    """Build a model from a composed Hydra DictConfig.

    The model config (``cfg.model``) should specify:
      - ``layers``: list of layer names (registry keys)
      - ``backbone``: model backbone name (registry key)
      - ``dim``, ``max_length``: shared across all layers
      - ``position_embedding``: dict with at least ``type`` key
      - Per-layer-type kwargs as sub-keys matching the layer name

    Example model config::

        layers: [gla, swiglu, gla, swiglu]
        backbone: language-model
        dim: 128
        max_length: 1280
        position_embedding:
          type: selective_rope
        gla:
          num_heads: 8
          ...
        swiglu:
          dim_inner: 512
    """
    model_cfg = cfg.model
    pe = OmegaConf.to_container(model_cfg.position_embedding, resolve=True)

    layer_modules = []
    layer_cfgs = []
    for layer_name in model_cfg.layers:
        module_cls = layer_registry[layer_name]["module"]
        # Get layer-type-specific kwargs
        layer_kwargs = {}
        if layer_name in model_cfg:
            layer_kwargs = OmegaConf.to_container(model_cfg[layer_name], resolve=True)
        # Add common params
        layer_kwargs["dim"] = model_cfg.dim
        layer_kwargs["max_length"] = model_cfg.max_length
        layer_kwargs["position_embedding"] = pe
        layer_modules.append(module_cls)
        layer_cfgs.append(layer_kwargs)

    return model_registry[model_cfg.backbone](
        dim=model_cfg.dim,
        vocab_size=cfg.task.vocab_size,
        layers=layer_modules,
        layer_cfgs=layer_cfgs,
        max_length=model_cfg.max_length,
    )


@dataclass
class MADModelConfig(BaseConfig):
    """Model configuration for models built from architecture
    components provided in this repository."""

    layers: list[str] = None
    backbone: str = "language-model"
    dim: int = 128
    max_length: int = 1_280
    vocab_size: int = 16
    norm: nn.Module = nn.LayerNorm
    position_embeds: Callable = None
    embed_drop_rate: float = 0.0

    def build_model_from_registry(self):
        """build a model from components registered in MAD"""
        layer_configs = []
        for layer in self.layers:
            _cfg = load_yml(os.path.join(get_base_path(), layer_registry[layer]["cfg"]))
            _cfg["dim"] = self.dim
            _cfg["max_length"] = self.max_length
            layer_configs.append(_cfg)
        model = model_registry[self.backbone](
            dim=self.dim,
            vocab_size=self.vocab_size,
            layers=[layer_registry[l]["module"] for l in self.layers],
            layer_cfgs=layer_configs,
            max_length=self.max_length,
            norm=self.norm,
            position_embeds=self.position_embeds,
            embed_drop_rate=self.embed_drop_rate,
        )
        return model


def make_benchmark_mad_configs(**kwargs):
    """Returns a list containing all MADConfigs of the MAD benchmark."""

    lrs = [1e-4, 5e-4, 1e-3]
    wds = [0.0, 0.1]
    mad_configs = []
    for task in task_registry.keys():
        task_cfg = load_yml(_MAD_ROOT / task_registry[task]["cfg"])
        baseline = task_cfg["baseline"]
        baseline["task"] = task
        for k, v in kwargs.items():
            baseline[k] = v
        changes = task_cfg["changes"]

        for lr in lrs:
            for wd in wds:
                # baseline task setting:
                mad_config = MADConfig(lr=lr, weight_decay=wd)
                mad_config.update_from_kwargs(baseline)
                mad_configs.append(mad_config)
                # changes to baseline setting, varying task difficulty:
                for change_key in changes:
                    change_cfg = dict(baseline)
                    for change_value in changes[change_key]:
                        change_cfg[change_key] = change_value
                        mad_config = MADConfig(lr=lr, weight_decay=wd)
                        mad_config.update_from_kwargs(change_cfg)
                        mad_configs.append(mad_config)

    return mad_configs
