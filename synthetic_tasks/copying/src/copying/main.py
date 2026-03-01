"""Accelerate-based training loop for copying tasks using FLA models."""

import argparse
import logging
from collections import defaultdict
from typing import Any, cast

import torch
import torch.nn.functional as F  # noqa: N812
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, get_scheduler

import selective_rope  # noqa
from copying.data_utils import CopyDataset, EvalCopyDataset, get_tokenizer


def _get_sampler(train_set: Dataset, cfg: DictConfig):
    """Initlaizes a sampler for a torch.Dataloader.
    Options:
      - sequential sampler
      - random sampler
    We implement "stateful" sequential samplers for resuming training from a specified step.
    """
    ddp = dist.is_initialized()

    if ddp:
        sampler = DistributedSampler(train_set, drop_last=True)
    else:
        sampler = SequentialSampler(
            train_set  # type: ignore
        )  # equivalent to StatefulSequentialSampler(..., start_idx=0)

    if cfg.sampler_type == "random" and cfg.sampler_seed is not None and not ddp:
        generator = torch.Generator().manual_seed(cfg.sampler_seed)
        sampler = RandomSampler(train_set, generator=generator)  # type: ignore

    return sampler


def _masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    pad_token_id: int | None,
) -> torch.Tensor:
    if mask is not None:
        if not isinstance(logits, torch.Tensor):
            logits = logits["logits"]
        loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=(pad_token_id if pad_token_id is not None else -100),
            reduction="none",
        )
        loss = loss_fct(logits.view(-1, logits.size(-1)), targets.flatten())
        mask = mask.contiguous().view(-1)
        return torch.sum(loss * mask) / torch.clamp(mask.sum(), min=1)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def _move_to_device(batch: dict, seq_len: int, device: torch.device):
    seq_len = min(seq_len, batch["input_ids"].shape[1] - 1)
    inputs = batch["input_ids"][:, :seq_len]
    targets = batch["input_ids"][:, 1 : (seq_len + 1)]
    masks = batch.get("mask", None)
    if masks is not None:
        masks = masks[:, 1 : (seq_len + 1)]
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    if masks is not None:
        masks = masks.to(device, non_blocking=True)
    return inputs, targets, masks


def _init_scheduler(optimizer, cfg, total_training_steps: int):
    name = getattr(cfg.trainer, "scheduler", None)
    if name is None:
        return None
    if name == "linear":
        warmup = getattr(cfg.trainer, "warmup_steps", 0)
        if isinstance(warmup, float):
            warmup = int(warmup * cfg.trainer.steps_budget)
        return get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_training_steps,
        )
    return None


def _build_eval_lengths(cfg):
    start_len = getattr(cfg.data, "min_eval_len", getattr(cfg.data, "min_length", 8))
    end_len = getattr(cfg.data, "max_eval_len", getattr(cfg.data, "max_length", 30))
    return list(range(start_len, end_len))


def _score_batch(tokenizer, x, pred):
    str_acc_sum = 0
    char_acc_sum = 0.0
    for i in range(len(x)):
        x_out = tokenizer.decode(x[i])
        x_out = x_out.split(".")[0] + "."
        pred_out = tokenizer.decode(pred[i])
        index = x_out.index("|")
        gt = x_out[index + 1 :][:-1]
        start_idx = index
        end_idx = start_idx + len(gt)
        pred_model = pred_out[start_idx:end_idx]
        str_acc_sum += int(gt == pred_model)
        char_acc_sum += sum(map(str.__eq__, gt, pred_model)) / max(
            len(gt), len(pred_model)
        )
    return str_acc_sum / len(x), char_acc_sum / len(x)


def main(cfg: DictConfig):
    tokenizer, TO_TOKEN, _ = get_tokenizer(cfg.data)

    train_dataset = CopyDataset(
        tokenizer=tokenizer,
        vocab_size=cfg.data.vocab_size,
        n_gram=cfg.data.n_gram,
        length_answer=cfg.data.length_answer,
        train_task=cfg.data.train_task,
        sequence_length=cfg.data.sequence_length,
        min_length=cfg.data.min_length,
        max_length=cfg.data.max_length,
        batch_size=1,
        num_examples=cfg.trainer.steps_budget * cfg.trainer.micro_batch_size,
    )
    train_sampler = _get_sampler(
        train_set=train_dataset,  # type: ignore
        cfg=cfg.data,
    )
    train_dataloader = DataLoader(
        cast(Dataset[Any], train_dataset),
        sampler=train_sampler,
        batch_size=cfg.trainer.micro_batch_size,
        num_workers=cfg.trainer.num_workers,
        pin_memory=True,
        prefetch_factor=2 if cfg.trainer.num_workers > 0 else None,
        persistent_workers=True if cfg.trainer.num_workers > 0 else False,
    )

    # Note: we construct eval datasets per target length later for the sweep

    set_seed(cfg.trainer.seed)
    accelerator = Accelerator(log_with=("wandb" if cfg.trainer.logging else None))
    log.setLevel(cfg.trainer.log_level)

    project_hps = {
        "batch_size": cfg.trainer.batch_size,
        "betas": (cfg.trainer.beta1, cfg.trainer.beta2),
        "compile": cfg.trainer.compile,
        "mixed_precision": cfg.trainer.mix_precision,
        "scheduler": getattr(cfg.trainer, "scheduler", None),
        "epochs": cfg.trainer.epochs,
        "eps": cfg.trainer.op_eps,
        "gradient_clip": cfg.trainer.gradient_clip,
        "lr": cfg.trainer.lr,
        "seed": cfg.trainer.seed,
        "weight_decay": cfg.trainer.weight_decay,
        "position_embedding": cfg.model.position_embedding,
    }
    if cfg.trainer.logging:
        accelerator.init_trackers(
            cfg.logger.wandb_project,
            config=project_hps,
            init_kwargs={"wandb": {"entity": cfg.logger.wandb_entity}},
        )

    model = AutoModelForCausalLM.from_config(  # type: ignore[attr-defined]
        AutoConfig.from_pretrained(cfg.trainer.model_name_or_path)
    )
    if cfg.trainer.compile:
        torch.set_float32_matmul_precision("high")
        log.info("Compiling model...")
        model = torch.compile(model, dynamic=True)
        log.info("Model compiled!")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.trainer.lr,
        betas=(cfg.trainer.beta1, cfg.trainer.beta2),
        eps=cfg.trainer.op_eps,
        weight_decay=cfg.trainer.weight_decay,
    )
    total_training_steps = cfg.trainer.steps_budget
    scheduler = _init_scheduler(optimizer, cfg, total_training_steps)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    device = accelerator.device

    metrics = defaultdict(list)

    try:
        model.train()
        pbar = tqdm(train_dataloader, disable=not accelerator.is_local_main_process)
        for micro_step, batch in enumerate(pbar, start=1):
            step = micro_step // cfg.trainer.grad_accumulation_steps
            if step > cfg.trainer.steps_budget:
                break

            with accelerator.accumulate(model):
                inputs, targets, masks = _move_to_device(
                    batch, cfg.data.sequence_length, device
                )
                autocast_dtype = torch.bfloat16 if cfg.trainer.mix_precision else None
                with torch.autocast(
                    device_type=("cuda" if device.type == "cuda" else "cpu"),
                    dtype=autocast_dtype,
                    enabled=cfg.trainer.mix_precision,
                ):
                    output = model(inputs)
                    logits = getattr(output, "logits", output)
                    loss = _masked_cross_entropy(
                        logits, targets, masks, pad_token_id=TO_TOKEN.get("*", None)
                    )

                accelerator.backward(loss)
                if cfg.trainer.gradient_clip is not None:
                    accelerator.clip_grad_norm_(
                        model.parameters(), cfg.trainer.gradient_clip
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

            metrics["train/loss"].append(loss.detach().item())
            metrics["lr"].append(optimizer.param_groups[0]["lr"])
            metrics["micro_step"].append(micro_step)
            metrics["step"].append(step)
            metrics["tokens"].append(
                micro_step * cfg.trainer.micro_batch_size * cfg.data.sequence_length
            )

            if step % cfg.trainer.log_every_steps == 0:
                log_msg = {
                    "train/loss": sum(metrics["train/loss"])
                    / max(len(metrics["train/loss"]), 1),
                    "lr": optimizer.param_groups[0]["lr"],
                    "micro_step": micro_step,
                    "step": step,
                }
                accelerator.log(log_msg, step=step)
                metrics = defaultdict(list)

        model.eval()
        lengths = _build_eval_lengths(cfg)
        for tgt_len in lengths:
            fixed_eval = EvalCopyDataset(
                tokenizer,
                TO_TOKEN,
                vocab_size=cfg.data.vocab_size,
                n_gram=cfg.data.n_gram,
                length_answer=cfg.data.length_answer,
                eval_task=cfg.data.eval_task,
                sequence_length=max(2 * tgt_len + 1, cfg.data.sequence_length),
                min_length=tgt_len,
                max_length=tgt_len,
                batch_size=cfg.trainer.eval_batch_size,
            )
            fixed_loader = DataLoader(
                cast(Dataset[Any], fixed_eval),
                batch_size=cfg.trainer.eval_batch_size,
                num_workers=cfg.trainer.num_workers,
                sampler=SequentialSampler(fixed_eval),
            )
            fixed_loader = accelerator.prepare(fixed_loader)

            str_acc_total = 0.0
            char_acc_total = 0.0
            nbatches = 0
            with torch.no_grad():
                for batch in fixed_loader:
                    x = batch["input_ids"].to(device)
                    with torch.autocast(
                        device_type=("cuda" if device.type == "cuda" else "cpu"),
                        dtype=(torch.bfloat16 if cfg.trainer.mix_precision else None),
                        enabled=cfg.trainer.mix_precision,
                    ):
                        out = model(x)
                        logits = getattr(out, "logits", out)
                        pred = torch.argmax(logits, dim=-1)
                    s_acc, c_acc = _score_batch(tokenizer, x, pred)
                    str_acc_total += s_acc
                    char_acc_total += c_acc
                    nbatches += 1

            mean_str = str_acc_total / max(nbatches, 1)
            mean_char = char_acc_total / max(nbatches, 1)
            accelerator.log(
                {
                    "test/mean_str_acc": mean_str,
                    "test/mean_char_acc": mean_char,
                    "test/seq_len": tgt_len,
                },
                step=cfg.trainer.steps_budget + tgt_len - cfg.data.min_eval_len + 1,
            )

        accelerator.end_training()
    except Exception as e:
        if cfg.trainer.logging:
            import traceback

            wandb.log({"error_traceback": traceback.format_exc(), "error": str(e)})
        raise
    finally:
        if cfg.trainer.logging:
            wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%d-%m %H:%M:%S",
        level=logging.INFO,
    )

    log = get_logger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)

    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg)

    main(cfg)
