import os
from collections.abc import Callable

import hydra
import numpy as np
import ray
import ray.util.multiprocessing as mp
import torch
from omegaconf import DictConfig, OmegaConf

import mad
from mad.analysis import compute_model_mad_scores
from mad.configs import make_benchmark_mad_configs
from mad.paths import make_log_path
from mad.registry import layer_registry, model_registry
from mad.train import train


def check_benchmark_data_present(mad_configs):
    """Make sure benchmark data are present."""
    for mad_config in mad_configs:
        assert os.path.isdir(mad_config.train_dataset_path)
        assert os.path.isdir(mad_config.test_dataset_path)


def benchmark(
    make_model_fn: Callable,
    model_id: str,
    gpus: int = 1,
    cpus: int = 12,
    num_trials_gpu: int = 1,
    num_cpus_trial: int = 2,
    data_path: str = "./benchmark/data",
    logs_path: str = "./benchmark/logs",
    log_to_csv: bool = True,
    log_to_wandb: bool = False,
    wandb_project: str = "MAD",
    save_checkpoints: bool = True,
    precision: str = "bf16",
    persistent_workers: bool = True,
    ray_tmp_path: str = "/tmp/ray",
):
    """
    Benchmark a model on MAD.

    Args:
        make_model_fn (callable): function that returns a PyTorch model
        model_id (str): unique identifier for the model
        gpus (int): number of gpus to use for training
        cpus (int): number of cpus to use for training
        num_trials_gpu (int): number of trials to run per gpu
        num_cpus_trial (int): number of cpus to allocate to each trial
        logs_path (str): path where logs are stored
        log_to_csv (bool): if True, training metrics are locally saved to csv
        log_to_wandb (bool): if True, training metrics are logged to Weights & Biases
        wandb_project (str): project name to use when logging to Weights & Biases
        save_checkpoints (bool): if True, last and best model checkpoints of each training run are saved in the log directory
        ray_tmp_path (str): tmp path to be used by ray

    Returns:
        MAD scores for the model
    """

    # create all MAD configs for benchmark:
    mad_configs = make_benchmark_mad_configs(
        data_path=data_path, precision=precision, persistent_workers=persistent_workers
    )

    def setup_model_and_train(mad_config):
        """Helper to setup model and train it according to MAD config."""
        # Ensure checkpoint loading works under torch 2.6+ (weights_only=True) inside Ray workers
        torch.serialization.add_safe_globals([mad.configs.MADConfig])
        log_path = make_log_path(
            base_path=logs_path,
            mad_config=mad_config,
            model_id=model_id,
        )
        model = make_model_fn(
            task=mad_config.task,
            vocab_size=mad_config.vocab_size,
            max_length=mad_config.seq_len,
        )

        mad_config.num_data_workers = 8
        results = train(
            model=model,
            mad_config=mad_config,
            log_path=log_path,
            log_to_csv=log_to_csv,
            log_to_wandb=log_to_wandb,
            save_checkpoints=save_checkpoints,
            wandb_project=wandb_project,
        )
        return results

    if gpus > 1:

        @ray.remote(num_gpus=1.0 / num_trials_gpu, num_cpus=num_cpus_trial)
        def select_gpu_and_train(args):
            """Helper to select a gpu and train a model; used in multiprocessing pool."""
            job_id, mad_config = args
            gpu_id = job_id % gpus
            torch.cuda.device(gpu_id)
            return setup_model_and_train(mad_config)

        if not ray.is_initialized():
            ray.init(
                num_gpus=gpus,
                num_cpus=cpus,
                _temp_dir=ray_tmp_path,
                object_store_memory=10000000000,
            )
        pool = mp.Pool(gpus * num_trials_gpu)
        instances = pool.map(select_gpu_and_train.remote, enumerate(mad_configs))
        ray.get(instances)

    else:
        for mad_config in mad_configs:
            setup_model_and_train(mad_config)

    mad_scores = compute_model_mad_scores(model_id=model_id, logs_path=logs_path)
    print("\n----")
    print("MAD scores for each synthetic task:")
    for task, score in zip(mad_scores.index, mad_scores.values):
        print(f"  {task}: {score}")
    print(f"Mean across Tasks: {np.mean(mad_scores.values)}")

    return mad_scores


@hydra.main(config_path="../../../../configs/mad", config_name="mad", version_base=None)
def main(cfg: DictConfig):
    torch.serialization.add_safe_globals([mad.configs.MADConfig])

    model_cfg = cfg.model
    pe = OmegaConf.to_container(model_cfg.position_embedding, resolve=True)

    # Extract layer modules and per-layer-type kwargs from Hydra config.

    layer_names = list(model_cfg.layers)
    layers = [layer_registry[l]["module"] for l in layer_names]
    layer_configs = []
    for layer_name in layer_names:
        layer_kwargs = {}
        if layer_name in model_cfg:
            layer_kwargs = OmegaConf.to_container(model_cfg[layer_name], resolve=True)
        layer_kwargs["position_embedding"] = pe
        layer_configs.append(layer_kwargs)

    model_id = "-".join(layer_registry[l]["shorthand"] for l in layer_names)

    # Factory function to create the model for each benchmark task.
    # (backbone, vocab_size, and max_length change across tasks)

    def make_model_fn(
        task: str,
        vocab_size: int,
        max_length: int,
        dim: int = model_cfg.dim,
        layers: tuple[Callable] = tuple(layers),
        layer_configs: tuple[dict] = tuple(layer_configs),
    ) -> torch.nn.Module:
        for lc in layer_configs:
            lc["max_length"] = max_length
            lc["dim"] = dim
        backbone = "language-model" if task not in {"compression"} else "autoencoder"
        return model_registry[backbone](
            dim=dim,
            vocab_size=vocab_size,
            layers=layers,
            layer_cfgs=layer_configs,
            max_length=max_length,
        )

    benchmark(
        make_model_fn=make_model_fn,
        model_id=model_id,
        gpus=cfg.benchmark.gpus,
        cpus=cfg.benchmark.cpus,
        num_trials_gpu=cfg.benchmark.num_trials_gpu,
        num_cpus_trial=cfg.benchmark.num_cpus_trial,
        data_path=cfg.data_path,
        logs_path=cfg.log_base_path,
        log_to_csv=cfg.log_to_csv,
        log_to_wandb=cfg.log_to_wandb,
        wandb_project=cfg.logger.wandb_project,
        save_checkpoints=cfg.save_checkpoints,
        precision=cfg.trainer.precision,
        persistent_workers=cfg.trainer.persistent_data_workers,
        ray_tmp_path=cfg.benchmark.ray_tmp_path,
    )


if __name__ == "__main__":
    main()
