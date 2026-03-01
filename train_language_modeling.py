import math
import os
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
import wandb
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM

import selective_rope  # noqa


def print_master(msg: str):
    if int(os.environ.get("RANK", 0)) == 0:
        print(msg)


@dataclass(frozen=True)
class ResumeState:
    resume_dir: str
    micro_step: int
    step: int
    accumulated_samples: int
    world_size: int


class OffsetSampler(torch.utils.data.Sampler):
    """Wrap a base sampler but start yielding from an offset (in number of samples)."""

    def __init__(self, base_sampler, offset: int):
        self.base_sampler = base_sampler
        self.offset = int(offset)

    def __iter__(self):
        if self.offset <= 0:
            yield from iter(self.base_sampler)
            return
        yield from islice(iter(self.base_sampler), self.offset, None)

    def __len__(self):
        base_len = len(self.base_sampler)
        return max(0, base_len - max(0, self.offset))


def _rng_state_dict(device: str) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if "cuda" in str(device) and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()
    return state


def _load_rng_state(state: dict, device: str) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if "cuda" in str(device) and torch.cuda.is_available():
        if "torch_cuda" in state:
            torch.cuda.set_rng_state(state["torch_cuda"])
        elif "torch_cuda_all" in state:
            # Legacy checkpoint compatibility
            torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def _gather_rng_by_rank(device: str) -> list[dict] | None:
    """Gather RNG states from all ranks onto rank0. Returns list on rank0, else None."""
    local = _rng_state_dict(device)
    if not torch.distributed.is_initialized():
        return [local]

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    gathered = [None] * world_size if rank == 0 else None
    torch.distributed.gather_object(local, gathered, dst=0)
    return gathered


def save_checkpoint(
    ckpt_dir: str,
    model: torch.nn.Module,
    engine: "TorchEngine",
    micro_step: int,
    step: int,
    world_size: int,
    *,
    cfg: DictConfig,
    master_process: bool,
):
    # Collect per-rank RNG on rank0 so resume can restore rank-specific RNG streams.
    rng_by_rank = _gather_rng_by_rank(engine.device)
    if torch.distributed.is_initialized() and not master_process:
        # Non-master ranks participate in the gather but do not write files.
        return

    os.makedirs(ckpt_dir, exist_ok=True)

    # Save model weights/config in HF format for easy restore.
    model.save_pretrained(ckpt_dir)

    trainer_state = {
        "micro_step": int(micro_step),
        "step": int(step),
        "accumulated_samples": int(engine.accumulated_samples),
        "engine_micro_steps": int(engine.micro_steps),
        "world_size": int(world_size),
        "optimizer": engine.optimizer.state_dict(),
        "scheduler": engine.scheduler.state_dict() if engine.scheduler else None,
        "scaler": engine.scaler.state_dict() if engine.scaler else None,
        # New format: per-rank RNG for better reproducibility across resume.
        "rng_by_rank": rng_by_rank,
        # Legacy single RNG (rank0) for backward compatibility / convenience.
        "rng": rng_by_rank[0]
        if rng_by_rank is not None
        else _rng_state_dict(engine.device),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(trainer_state, os.path.join(ckpt_dir, "trainer_state.pt"))


def load_resume_state(resume_dir: str) -> ResumeState:
    state_path = os.path.join(resume_dir, "trainer_state.pt")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Missing trainer state: {state_path}")
    st = torch.load(state_path, map_location="cpu", weights_only=False)
    micro_step = int(st.get("micro_step", 0))
    step = int(st.get("step", micro_step))
    accumulated_samples = int(st.get("accumulated_samples", 0))
    world_size = int(st.get("world_size", 1))
    return ResumeState(
        resume_dir=resume_dir,
        micro_step=micro_step,
        step=step,
        accumulated_samples=accumulated_samples,
        world_size=world_size,
    )


def restore_engine_from_checkpoint(
    resume_dir: str,
    engine: "TorchEngine",
    *,
    device: str,
    world_size: int,
):
    state_path = os.path.join(resume_dir, "trainer_state.pt")
    st = torch.load(state_path, map_location="cpu", weights_only=False)

    ckpt_world_size = int(st.get("world_size", 1))
    if ckpt_world_size != int(world_size):
        raise ValueError(
            f"World size mismatch: checkpoint has {ckpt_world_size}, current run has {world_size}"
        )

    if st.get("optimizer") is not None:
        engine.optimizer.load_state_dict(st["optimizer"])
    if st.get("scheduler") is not None and engine.scheduler is not None:
        engine.scheduler.load_state_dict(st["scheduler"])
    if st.get("scaler") is not None and engine.scaler is not None:
        engine.scaler.load_state_dict(st["scaler"])

    engine.micro_steps = int(st.get("engine_micro_steps", st.get("micro_step", 0)))
    engine.accumulated_samples = int(st.get("accumulated_samples", 0))

    # We only support exact resume cleanly when checkpointed at step boundaries.
    if engine.accumulated_samples != 0:
        print_master(
            f"WARNING: checkpoint accumulated_samples={engine.accumulated_samples}. "
            "This code does not restore partial gradients; set to 0 and resume at next step boundary."
        )
        engine.accumulated_samples = 0

    if "rng_by_rank" in st and st["rng_by_rank"] is not None:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        _load_rng_state(st["rng_by_rank"][rank], device=device)
    else:
        _load_rng_state(st.get("rng", {}), device=device)


def save_checkpoint_with_barrier(
    cfg: DictConfig,
    model: torch.nn.Module,
    engine: "TorchEngine",
    micro_step: int,
    step: int,
    world_size: int,
    master_process: bool,
    label: str = "",
):
    """Save checkpoint and synchronize across ranks."""
    ckpt_path = f"{cfg.trainer.output_dir}/checkpoints/step_{step}"
    if master_process:
        print_master(f"Saving {label}checkpoint to: {ckpt_path}")
    save_checkpoint(
        ckpt_path,
        model=model,
        engine=engine,
        micro_step=micro_step,
        step=step,
        world_size=world_size,
        cfg=cfg,
        master_process=master_process,
    )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if master_process:
        print_master(f"Finished saving {label}checkpoint to: {ckpt_path}")


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Return latest checkpoint directory inside output_dir/checkpoints (e.g. step_123), or None."""
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    if not output_dir or not os.path.isdir(checkpoints_dir):
        return None

    best_step = None
    best_path = None

    for name in os.listdir(checkpoints_dir):
        if not name.startswith("step_"):
            continue
        step_str = name[len("step_") :]
        try:
            step = int(step_str)
        except ValueError:
            continue

        ckpt_dir = os.path.join(checkpoints_dir, name)
        if not os.path.isdir(ckpt_dir):
            continue
        if not os.path.exists(os.path.join(ckpt_dir, "trainer_state.pt")):
            continue

        if best_step is None or step > best_step:
            best_step = step
            best_path = ckpt_dir

    return best_path


def get_dataloaders(cfg: DictConfig, *, start_micro_step: int = 0):
    """Load trainset and perhaps validset. Returns correspondent DataLoaders."""

    train_set = load_from_disk(cfg.cluster.data_home)
    train_set = train_set.with_format("torch")

    train_sampler = _get_sampler(train_set, cfg, start_micro_step=start_micro_step)

    # only used with intra-document masking
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
            "docs_lengths": [x["docs_lengths"].tolist() for x in batch],
        }

    return DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=cfg.trainer.micro_batch_size,
        num_workers=cfg.trainer.num_workers,
        pin_memory=True,
        prefetch_factor=4 if cfg.trainer.num_workers > 0 else None,
        persistent_workers=True if cfg.trainer.num_workers > 0 else False,
        collate_fn=collate_fn if "docs_lengths" in train_set.column_names else None,
    )


def _get_sampler(train_set, cfg: DictConfig, start_micro_step: int = 0):
    """Initlaizes a sampler for a torch.Dataloader.
    Options:
      - random sampler
      - sequential sampler
      - stateful random sampler
      - stateful sequential sampler
    We implement "stateful" sequential samplers for resuming training from a specified step.
    """
    ddp = dist.is_initialized()

    if ddp:
        sampler = DistributedSampler(train_set, shuffle=False, drop_last=True)
    else:
        sampler = SequentialSampler(train_set)

    if start_micro_step and int(start_micro_step) > 0:
        # DataLoader batches consecutive samples. Skipping K micro-steps means skipping
        # K * micro_batch_size samples on each rank (sampler is rank-local in DDP).
        sample_offset = int(start_micro_step) * int(cfg.trainer.micro_batch_size)
        sampler = OffsetSampler(sampler, offset=sample_offset)

    return sampler


def log(
    cfg: DictConfig,
    metrics: dict,
    micro_step: int,
    train_loss: torch.Tensor,
    train_loss_array: list,
    optimizer: torch.optim.Optimizer,
    world_size: int,
    last_grnorm: float | torch.Tensor,
    per_module_grad_norms: dict | None = None,
    steps_per_second: float | None = None,
):
    """Update metrics, print to console, log on wandb."""

    if isinstance(train_loss_array, list):
        train_loss_avg = torch.stack(train_loss_array).mean().item()
    elif isinstance(train_loss_array, torch.Tensor):
        train_loss_avg = train_loss_array.item()

    new_metrics = {
        "micro_step": micro_step,
        "step": micro_step // cfg.trainer.grad_accumulation_steps,
        "tokens": micro_step
        * cfg.trainer.micro_batch_size
        * cfg.trainer.seq_len
        * world_size,
        "lr": optimizer.param_groups[0].get("lr", float("NaN")),
        "train/loss": train_loss.item(),
        "train/loss_avg": train_loss_avg,
        "train/ppl": math.exp(train_loss),
        "train/ppl_avg": math.exp(train_loss_avg),
        "train/grad_norm": last_grnorm.to("cpu").item()
        if not isinstance(last_grnorm, float)
        else last_grnorm,
    }
    if steps_per_second is not None:
        new_metrics["steps_per_second"] = steps_per_second
    if cfg.logger.print_progress:
        msg = " | ".join(
            f"{key}: {value:.3e}" if isinstance(value, float) else f"{key}: {value}"
            for key, value in new_metrics.items()
        )
        print(msg)

    # Add per-module gradient norms if provided
    if per_module_grad_norms:
        for module_name, grad_norm in per_module_grad_norms.items():
            # Sanitize module name for wandb (replace dots and slashes with underscores)
            new_metrics[f"train/grad_norms_per_module/{module_name}"] = grad_norm

    for k, v in new_metrics.items():
        metrics[k].append(v)

    if cfg.logger.use_wandb:
        wandb.log(new_metrics)


def pytorch_setup(cfg):
    """Returns device, rank, seed, etc and initialize DDP"""
    ddp = int(os.environ.get("RANK", -1)) != -1  # check if DDP is enabled

    if ddp:
        init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        # torch.cuda.device(device)
        torch.cuda.set_device(device)
        master_process = rank == 0
        seed_offset = rank
    else:
        master_process = True
        seed_offset = 0
        local_rank = None
        world_size = 1
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"  # NOTE: macOS metal support to be tested

    random.seed(cfg.trainer.seed + seed_offset)
    np.random.seed(cfg.trainer.seed + seed_offset)
    torch.manual_seed(cfg.trainer.seed + seed_offset)

    # allow TF32, if not specified, we follow PyTorch 2.0 default
    # https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
    torch.backends.cuda.matmul.allow_tf32 = getattr(cfg, "cuda_matmul_allow_tf32", True)
    torch.backends.cudnn.allow_tf32 = getattr(cfg, "cudnn_allow_tf32", True)
    torch.backends.cudnn.benchmark = True

    return local_rank, world_size, device, master_process


def destroy_ddp():
    if torch.distributed.is_initialized():
        torch.cuda.synchronize()  # finish GPU work
        torch.distributed.barrier()  # wait for all ranks
        destroy_process_group()  # cleanly tear down comms


class CustomLRSchedule(ABC):
    """An abstract parent class for custom LR Schedules."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def set_optim_lr(self, lr):
        """Set a learning rate for all parameter groups."""
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def state_dict(self):
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    @abstractmethod
    def step(self):
        pass


class WarmupCosine(CustomLRSchedule):
    """Linear warmup followed by Cosine Decay."""

    def __init__(self, optimizer, lr_start, lr_max, lr_end, warmup_steps, T):
        super().__init__(optimizer)
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_end = lr_end
        self.warmup_steps = warmup_steps
        self.T = T
        self.iter = 0
        self.set_optim_lr(lr_start)

    def get_lr(self, t):
        """Computes and returns lr(t), where t is the current step."""
        if t <= self.warmup_steps:
            return self.lr_start + (self.lr_max - self.lr_start) / self.warmup_steps * t
        elif t <= self.T:
            progress = (t - self.warmup_steps) / (self.T - self.warmup_steps)
            return self.lr_end + 0.5 * (self.lr_max - self.lr_end) * (
                1 + math.cos(math.pi * progress)
            )
        return self.lr_end

    def step(self):
        self.iter += 1
        lr = self.get_lr(self.iter)
        self.set_optim_lr(lr)


def _move_to_device(batch, seq_len, device):
    """Slice batch to get inputs and targets, and move them to device."""

    inputs = batch["input_ids"][:, : seq_len - 1]
    targets = batch["input_ids"][:, 1:]

    attn_mask = None

    if "cuda" in device:
        # pin arrays allows to move them to GPU asynchronously (non_blocking=True)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
    else:
        inputs, targets = inputs.to(device), targets.to(device)

    return inputs, targets, attn_mask


class TorchEngine(torch.nn.Module):
    """
    A module containing model, optimizer, scheduler, grad scaler.
    Wraps together a training step. Takes care of grad accumulation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: DictConfig,
        device: torch.device,
        local_rank: int,
    ):
        super().__init__()

        self.micro_steps = 0
        self.accumulated_samples = 0
        self.last_per_module_grad_norms = None
        self.last_grad_norm: float = 0.0

        self.seq_len = cfg.trainer.seq_len
        self.accumulation_steps = cfg.trainer.grad_accumulation_steps
        self.grad_clip = cfg.trainer.grad_clip
        self.dtype = cfg.trainer.dtype
        self.device = device

        # Move model to device and to DDP
        self.model: nn.Module = model.to(self.device)
        if torch.distributed.is_initialized():
            self.model = DDP(self.model, device_ids=[local_rank])

        # Compile
        if cfg.trainer.torch_compile:
            print_master("Compiling the model...")
            self.model = torch.compile(self.model)

        # AMP
        self.ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

        # Grad scaler if training in fp16, if enabled=False, scaler is a no-op
        self.scaler = torch.amp.GradScaler(enabled=(self.dtype == "float16"))

        # Loss
        self.criterion = CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.trainer.lr_start,
            betas=(cfg.trainer.beta1, cfg.trainer.beta2),
            weight_decay=cfg.trainer.weight_decay,
            eps=cfg.trainer.eps,
            fused=cfg.trainer.fused_optim,
        )

        # Scheduler
        self.scheduler = WarmupCosine(
            self.optimizer,
            lr_start=cfg.trainer.lr_start,
            lr_max=cfg.trainer.learning_rate,
            lr_end=cfg.trainer.lr_end,
            warmup_steps=cfg.trainer.scheduler.warmup_steps * cfg.trainer.steps,
            T=cfg.trainer.steps,
        )

    def get_per_module_grad_norms(self):
        """Compute gradient norm for each named module in the network."""
        per_module_norms = {}

        # Get the underlying model if wrapped in DDP
        model = self.model
        if isinstance(model, DDP):
            model = model.module

        for name, module in model.named_modules():
            total_norm = 0.0
            param_count = 0
            for param in module.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            if param_count > 0:
                total_norm = total_norm ** (1.0 / 2)
                per_module_norms[name] = total_norm

        return per_module_norms

    def step(self, batch, compute_per_module_grad_norms=False):
        """Wraps a fwd pass, bwd pass, and optimization step.

        Args:
            batch: Input batch
            compute_per_module_grad_norms: If True, compute and store per-module gradient norms
        """

        self.model.train()

        self.micro_steps += 1
        self.accumulated_samples += 1

        inputs, targets, attn_mask = _move_to_device(batch, self.seq_len, self.device)

        # sync (reduce) gradients at the last accumulation step
        if torch.distributed.is_initialized():
            self.model.require_backward_grad_sync = (
                self.accumulated_samples == self.accumulation_steps
            )

        # forward pass with autocasting
        with self.ctx:
            output = self.model(inputs, attn_mask)
            logits = getattr(output, "logits", output)
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = loss / self.accumulation_steps

        # detach for logging (scale up to undo the division above)
        loss_val = loss.detach() * self.accumulation_steps
        if torch.isnan(loss_val):
            raise ValueError("Train loss is nan")

        # backward pass, with gradient scaling if training in fp16
        self.scaler.scale(loss).backward()

        # step after accumulation
        if self.accumulated_samples == self.accumulation_steps:
            self.accumulated_samples = 0

            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                self.last_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
            # Compute per-module grad norms before zeroing (if requested)
            if compute_per_module_grad_norms:
                self.last_per_module_grad_norms = self.get_per_module_grad_norms()
            else:
                self.last_per_module_grad_norms = None

            # step the optimizer, step the scaler if training in fp16
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # flush the gradients
            self.optimizer.zero_grad(set_to_none=True)

            # step the scheduler
            if self.scheduler:
                self.scheduler.step()

        return loss_val


def main(cfg: DictConfig):
    # Load resume metadata early (so we can start the sampler at the right point).
    resume_dir = getattr(cfg.trainer, "resume_from_checkpoint", None)
    if not resume_dir and getattr(cfg.trainer, "always_resume", False):
        resume_dir = find_latest_checkpoint(cfg.trainer.output_dir)
        if resume_dir:
            print_master(f"always_resume: found latest checkpoint: {resume_dir}")
    resume = None
    if resume_dir:
        resume = load_resume_state(resume_dir)
        print_master(
            f"Resuming from checkpoint: {resume_dir} (step={resume.step}, micro_step={resume.micro_step})"
        )

    # load the model (from checkpoint if resuming)
    if resume_dir:
        model = AutoModelForCausalLM.from_pretrained(resume_dir)
    else:
        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(cfg.trainer.model_name_or_path)
        )

    local_rank, world_size, device, master_process = pytorch_setup(cfg)

    if cfg.logger.use_wandb and master_process:
        wandb.init(
            project=cfg.logger.wandb_project,
            name=cfg.logger.wandb_run_name,
            dir=cfg.trainer.output_dir,
            config=OmegaConf.to_container(cfg),
        )

    start_micro_step = resume.micro_step if resume is not None else 0
    trainloader = get_dataloaders(cfg, start_micro_step=start_micro_step)

    engine = TorchEngine(model, cfg, device, local_rank)

    # Restore optimizer/scheduler/scaler + RNG after engine is created.
    if resume_dir:
        restore_engine_from_checkpoint(
            resume_dir, engine, device=device, world_size=world_size
        )

    print_master(f"Model num params: {model.num_parameters()}")

    # If we are just cooling down, we set budget = resume + cooldown
    steps_budget = cfg.trainer.steps
    micro_step_budget = steps_budget * cfg.trainer.grad_accumulation_steps
    if micro_step_budget > len(trainloader):
        raise ValueError("trainloader too short!")

    # Start the dataloader from the correct micro-batch
    if resume is not None:
        micro_step_start = resume.micro_step
        step_start = micro_step_start // cfg.trainer.grad_accumulation_steps
    else:
        step_start = 0
        micro_step_start = step_start * cfg.trainer.grad_accumulation_steps
    print_master(
        f"=== Start Training from step: {step_start}/{steps_budget}, micro_step: {micro_step_start}/{micro_step_budget} ==="
    )

    # Bookkeeping
    metrics = defaultdict(list)
    train_loss_array = []
    last_log_time = time.time()
    last_log_step = step_start

    # Training
    for micro_step, micro_batch in enumerate(trainloader, micro_step_start + 1):
        step = micro_step // cfg.trainer.grad_accumulation_steps
        is_step = micro_step % cfg.trainer.grad_accumulation_steps == 0
        if step > steps_budget and is_step:
            break

        # Check if we need to compute per-module grad norms for this step
        should_log_per_module = (
            master_process
            and step % cfg.trainer.logging_steps == 0
            and is_step
            and getattr(cfg.logger, "log_per_module_grad_norms", False)
        )

        train_loss = engine.step(
            micro_batch, compute_per_module_grad_norms=should_log_per_module
        )
        train_loss_array.append(train_loss)

        if master_process and step % cfg.trainer.logging_steps == 0 and is_step:
            current_time = time.time()
            elapsed = current_time - last_log_time
            steps_done = step - last_log_step
            steps_per_second = steps_done / elapsed if elapsed > 0 else 0.0
            log(
                cfg,
                metrics,
                micro_step,
                train_loss,
                train_loss_array,
                engine.optimizer,
                world_size,
                last_grnorm=engine.last_grad_norm,
                per_module_grad_norms=engine.last_per_module_grad_norms,
                steps_per_second=steps_per_second,
            )
            train_loss_array = []
            last_log_time = current_time
            last_log_step = step

        # Checkpoint (DDP-safe): rank0 writes, all ranks sync.
        if (
            cfg.trainer.save_intermediate_checkpoints
            and step % cfg.trainer.save_every_steps == 0
            and is_step
        ):
            save_checkpoint_with_barrier(
                cfg, model, engine, micro_step, step, world_size, master_process
            )

    # End of training: log and save checkpoint
    print_master("=== Training Completed! ===")
    if cfg.trainer.save_last_checkpoint:
        save_checkpoint_with_barrier(
            cfg,
            model,
            engine,
            micro_step,
            step,
            world_size,
            master_process,
            label="final ",
        )

    # DDP slaughtering
    destroy_ddp()


def cleanup_on_error(master_process: bool = False):
    """Gracefully cleanup resources on error."""
    try:
        if master_process:
            # Attempt to finish wandb gracefully
            try:
                import wandb

                if wandb.run is not None:
                    wandb.finish(exit_code=1)
            except Exception as e:
                print(f"[cleanup] Failed to finish wandb: {e}")

        # Cleanup DDP
        destroy_ddp()
    except Exception as e:
        print(f"[cleanup] Error during cleanup: {e}")


if __name__ == "__main__":
    import argparse
    import traceback

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)

    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg)

    try:
        main(cfg)
    except torch.cuda.OutOfMemoryError as e:
        rank = int(os.environ.get("RANK", 0))
        master_process = rank == 0
        print(f"[rank{rank}] CUDA OOM Error: {e}")
        traceback.print_exc()
        cleanup_on_error(master_process)
        raise SystemExit(1)
    except Exception as e:
        rank = int(os.environ.get("RANK", 0))
        master_process = rank == 0
        print(f"[rank{rank}] Unhandled exception: {e}")
        traceback.print_exc()
        cleanup_on_error(master_process)
        raise SystemExit(1)
