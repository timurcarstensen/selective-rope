import argparse
import json
import math
import os
import re
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import wandb
from datasets import load_from_disk
from omegaconf import OmegaConf
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM

import selective_rope  # noqa: F401

# Use __file__ instead of git repo root to support running from worktrees
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _move_to_device(batch, seq_len, device: torch.device):
    """Slice batch to get inputs and targets, and move them to device."""

    inputs = batch["input_ids"][:, : seq_len - 1]
    targets = batch["input_ids"][:, 1:]

    attn_mask = None

    if "cuda" in device.type:
        # pin arrays allows to move them to GPU asynchronously (non_blocking=True)
        inputs = inputs.pin_memory().to(device, non_blocking=True)
        targets = targets.pin_memory().to(device, non_blocking=True)
    else:
        inputs, targets = inputs.to(device), targets.to(device)

    return inputs, targets, attn_mask


MICRO_STEP_PATTERN = re.compile(r"micro_step:\s*(\d+)")
SUMMARY_FILENAME = "eval_summary.jsonl"


def is_run_directory(path: Path) -> bool:
    return (path / ".hydra" / "config.yaml").is_file()


def discover_run_directories(root: Path) -> list[Path]:
    if is_run_directory(root):
        return [root.resolve()]

    run_dirs = {
        cfg_path.parent.parent.resolve()
        for cfg_path in root.rglob(str(Path(".hydra") / "config.yaml"))
    }
    return sorted(run_dirs)


def load_existing_summary(summary_path: Path) -> set[tuple[str, str]]:
    existing: set[tuple[str, str]] = set()
    if not summary_path.exists():
        return existing

    buffer: list[str] = []
    depth = 0

    with summary_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not buffer and "{" not in line:
                continue

            buffer.append(line)
            depth += line.count("{") - line.count("}")

            if depth <= 0:
                text = "".join(buffer).strip()
                buffer = []
                depth = 0

                if not text:
                    continue

                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    continue

                run_dir = record.get("run_dir")
                checkpoint = record.get("checkpoint")
                if run_dir and checkpoint:
                    existing.add((run_dir, checkpoint))

    return existing


def append_summary_record(summary_path: Path, record: dict):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def build_lm_eval_command(
    checkpoint_dir: Path, output_dir: Path, gen_kwargs: str | None = None
) -> list[str]:
    """Build lm-eval-harness command (mirrors run_lm_eval_batch.py)."""
    model_args = (
        f"pretrained={checkpoint_dir}",
        "dtype=bfloat16",
        "tokenizer=mistralai/Mistral-7B-v0.1",
    )
    cmd: list[str] = [
        sys.executable,
        "evals/harness.py",
        "--model",
        "fla",
        "--model_args",
        ",".join(model_args),
        "--batch_size",
        "16",
        "--tasks",
        "wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,swde,squad_completion,fda",
        "--num_fewshot",
        "0",
        "--device",
        "cuda",
        "--show_config",
        "--output_path",
        str(output_dir),
        "--trust_remote_code",
        "--seed",
        "42",
    ]
    if gen_kwargs:
        cmd.extend(["--gen_kwargs", gen_kwargs])
    return cmd


def run_lm_eval_harness(
    checkpoint_dir: Path,
    results_dir: Path,
    gen_kwargs: str | None = None,
) -> Path | None:
    """Run lm-eval-harness for a checkpoint.

    Returns the path to the results JSON file, or None if failed.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    stdout_file = results_dir / "stdout.log"
    stderr_file = results_dir / "stderr.log"

    cmd = build_lm_eval_command(checkpoint_dir, results_dir, gen_kwargs=gen_kwargs)

    print(f"Running lm-eval-harness for checkpoint: {checkpoint_dir}")
    print(f"Output directory: {results_dir}")

    with stdout_file.open("wb") as out_f, stderr_file.open("wb") as err_f:
        process = subprocess.Popen(
            cmd,
            stdout=out_f,
            stderr=err_f,
            cwd=str(PROJECT_ROOT),
        )
        returncode = process.wait()

    if returncode != 0:
        print(f"WARNING: lm-eval-harness failed with code {returncode}")
        print(f"Check stderr log: {stderr_file}")
        return None

    # Find the results JSON (lm-eval creates a nested directory structure)
    results_files = list(results_dir.rglob("results_*.json"))
    if not results_files:
        print(f"WARNING: No results JSON found in {results_dir}")
        return None

    # Return the most recent one
    return max(results_files, key=lambda p: p.stat().st_mtime)


def parse_lm_eval_results(results_path: Path) -> dict[str, float]:
    """Parse lm-eval-harness results JSON into flat metrics dict."""
    with results_path.open("r") as f:
        data = json.load(f)

    metrics = {}
    results = data.get("results", {})

    for task_name, task_results in results.items():
        for metric_key, value in task_results.items():
            if metric_key == "alias":
                continue
            # Skip stderr values
            if "_stderr" in metric_key:
                continue
            # Handle "N/A" values
            if value == "N/A" or value is None:
                continue
            # Create flat metric name: lm_eval/task/metric
            # metric_key format is like "acc,none" or "perplexity,none"
            metric_name = metric_key.split(",")[0]
            metrics[f"lm_eval/{task_name}/{metric_name}"] = float(value)

    return metrics


@torch.inference_mode()
def evaluate_length_extrapolation(
    model,
    data_module,
    device: torch.device,
    max_length: int,
    num_batches: int = 20,
) -> dict:
    model.eval()
    total_loss_sum = torch.zeros(max_length - 1, device=device)
    total_accuracy_sum = torch.zeros(max_length - 1, device=device)
    total_count = 0
    per_token_losses = []

    if len(data_module.val_dataloader()) == 0:
        data_loader = data_module.train_dataloader()
    else:
        data_loader = data_module.val_dataloader()

    for idx, batch in tqdm(
        enumerate(data_loader),
        desc="Evaluating length extrapolation",
        total=min(num_batches, len(data_loader)),
    ):
        if idx >= num_batches:
            break
        src_seq = batch["src_seq"].to(device)
        trg_seq = batch["trg_seq"].to(device)

        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            outputs = model(src_seq)
        logits = outputs.logits

        loss = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2).float(), trg_seq, reduction="none"
        )

        if torch.isnan(loss).any():
            print(f"Warning: Loss is nan for batch {idx}, skipping")
            continue

        predictions = torch.argmax(logits, dim=-1)
        correct_predictions = predictions == trg_seq

        cum_loss = torch.cumsum(loss, dim=1)
        cum_correct = torch.cumsum(correct_predictions.float(), dim=1)

        token_positions = torch.arange(1, cum_loss.size(1) + 1, device=device)
        avg_cum_loss = cum_loss / token_positions
        avg_cum_accuracy = cum_correct / token_positions

        total_loss_sum += avg_cum_loss.sum(dim=0)
        total_accuracy_sum += avg_cum_accuracy.sum(dim=0)
        total_count += src_seq.size(0)
        per_token_losses.append(loss.cpu().float().numpy())

    mean_losses = (total_loss_sum / total_count).cpu().numpy()
    mean_accuracies = (total_accuracy_sum / total_count).cpu().numpy()
    per_token_losses_avg = np.concatenate(per_token_losses).mean(axis=0)
    perplexities = np.exp(mean_losses)

    return {
        "perplexities": perplexities.tolist(),
        "accuracies": mean_accuracies.tolist(),
        "token_losses": per_token_losses_avg.tolist(),
    }


def run_length_extrapolation(
    checkpoint_dir: Path,
    results_dir: Path,
    max_length: int = 4096,
    dataset_name: str = "codeparrot",
    batch_size: int = 16,
    num_cpu_workers: int = 8,
    num_batches: int = 20,
) -> Path | None:
    try:
        from evals.ssmworkbench.text_datamodule import TextArrowFileModule
    except ImportError:
        print(
            "Warning: Could not import TextArrowFileModule, skipping length extrapolation"
        )
        return None

    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"{dataset_name}_{max_length}.json"

    if results_file.exists():
        print(f"Length extrapolation results already exist at {results_file}, skipping")
        return results_file

    print(f"Running length extrapolation for checkpoint: {checkpoint_dir}")
    print(f"Max length: {max_length}, dataset: {dataset_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        device_map={"": device},
        torch_dtype=torch.float,
    )
    model.eval()

    data_dir = os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    cache_dir = os.getenv("HF_DATASETS_CACHE") or os.path.expanduser(
        "~/.cache/huggingface/datasets"
    )

    data_module = TextArrowFileModule(
        tokenizer="mistralai/Mistral-7B-v0.1",
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_cpu_worker=num_cpu_workers,
        max_sample_len=max_length,
        data_dir=data_dir,
        cache_dir=cache_dir,
        val_ratio=0.0005,
        val_split_seed=1337,
        seed=1337,
    )

    results = evaluate_length_extrapolation(
        model, data_module, device, max_length, num_batches=num_batches
    )

    with results_file.open("w") as f:
        json.dump({"results": results}, f, indent=2)

    print(f"Length extrapolation results saved to {results_file}")
    return results_file


def parse_length_extrap_results(
    results_path: Path, dataset_name: str = ""
) -> dict[str, float]:
    with results_path.open("r") as f:
        data = json.load(f)

    results = data.get("results", {})
    perplexities = results.get("perplexities", [])
    token_losses = results.get("token_losses", [])

    if not perplexities and not token_losses:
        return {}

    prefix = f"length_extrap/{dataset_name}" if dataset_name else "length_extrap"

    metrics = {}
    if perplexities:
        metrics[f"{prefix}/final_perplexity"] = float(perplexities[-1])
        metrics[f"{prefix}/mean_perplexity"] = float(np.mean(perplexities))
    if token_losses:
        metrics[f"{prefix}/final_loss"] = float(token_losses[-1])
        metrics[f"{prefix}/mean_loss"] = float(np.mean(token_losses))

    positions = [512, 1024, 2048, 4096, 8192]
    for pos in positions:
        if perplexities and pos <= len(perplexities):
            metrics[f"{prefix}/ppl_at_{pos}"] = float(perplexities[pos - 1])
        if token_losses and pos <= len(token_losses):
            metrics[f"{prefix}/loss_at_{pos}"] = float(token_losses[pos - 1])

    return metrics


def get_wandb_run_id(run_dir: Path) -> str | None:
    """Extract wandb run ID from run directory.

    Tries multiple sources:
    1. wandb-resume.json (preferred)
    2. latest-run symlink name (fallback)
    3. Any run-* directory name (last resort)
    """
    wandb_dir = run_dir / "wandb"
    if not wandb_dir.exists():
        return None

    # Try wandb-resume.json first
    resume_file = wandb_dir / "wandb-resume.json"
    if resume_file.exists():
        try:
            return json.loads(resume_file.read_text())["run_id"]
        except (json.JSONDecodeError, KeyError):
            pass

    # Try latest-run symlink (format: run-YYYYMMDD_HHMMSS-RUNID)
    latest_run = wandb_dir / "latest-run"
    if latest_run.is_symlink():
        target = latest_run.resolve().name  # e.g., "run-20260113_110333-an69y427"
        parts = target.split("-")
        if len(parts) >= 3:
            return parts[-1]  # Last part is the run ID

    # Try any run-* directory as last resort
    for item in wandb_dir.iterdir():
        if item.is_dir() and item.name.startswith("run-"):
            parts = item.name.split("-")
            if len(parts) >= 3:
                return parts[-1]

    return None


def log_to_wandb(
    run_dir: Path,
    result: dict,
    lm_eval_results_path: Path | None = None,
    length_extrap_results: dict[str, Path] | None = None,
) -> bool:
    """Log eval results to the training wandb run.

    Args:
        run_dir: Path to the training run directory.
        result: Validation perplexity results dict.
        lm_eval_results_path: Optional path to lm-eval-harness results JSON.
        length_extrap_results: Optional dict mapping dataset names to their
            length extrapolation results JSON paths.

    Returns True if logging succeeded, False otherwise.
    """
    run_id = get_wandb_run_id(run_dir)
    if run_id is None:
        print(
            f"Warning: Could not find wandb run ID in {run_dir / 'wandb'}, skipping wandb logging"
        )
        return False
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

    entity = getattr(cfg.logger, "wandb_entity", None)
    project = cfg.logger.wandb_project

    lm_eval_metrics = {}
    if lm_eval_results_path is not None and lm_eval_results_path.exists():
        lm_eval_metrics = parse_lm_eval_results(lm_eval_results_path)

    length_extrap_metrics = {}
    if length_extrap_results:
        for ds_name, ds_path in length_extrap_results.items():
            if ds_path is not None and ds_path.exists():
                length_extrap_metrics.update(
                    parse_length_extrap_results(ds_path, dataset_name=ds_name)
                )

    # Use wandb.init with resume to properly connect to the existing run
    # This is required to log new artifacts (the public API only accepts existing artifacts)
    run = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="allow",
        reinit=True,
    )

    # Log lm-eval-harness results as artifact if available
    if lm_eval_results_path is not None and lm_eval_results_path.exists():
        checkpoint_name = result.get("checkpoint", "latest")
        artifact_metadata = {
            "run_id": run_id,
            "checkpoint": checkpoint_name,
            **lm_eval_metrics,
        }
        if "loss" in result:
            artifact_metadata["eval/loss"] = result["loss"]
            artifact_metadata["eval/perplexity"] = result["perplexity"]
        artifact = wandb.Artifact(
            name=f"lm_eval_results_{run_id}_{checkpoint_name}",
            type="evaluation",
            description="Full lm-eval-harness results JSON",
            metadata=artifact_metadata,
        )
        artifact.add_file(str(lm_eval_results_path))
        run.log_artifact(artifact)

    if length_extrap_results:
        for ds_name, ds_path in length_extrap_results.items():
            if ds_path is None or not ds_path.exists():
                continue
            checkpoint_name = result.get("checkpoint", "latest")
            ds_metrics = {
                k: v for k, v in length_extrap_metrics.items() if ds_name in k
            }
            artifact = wandb.Artifact(
                name=f"length_extrap_{ds_name}_{run_id}_{checkpoint_name}",
                type="evaluation",
                metadata={
                    "run_id": run_id,
                    "checkpoint": checkpoint_name,
                    "dataset": ds_name,
                    **ds_metrics,
                },
            )
            artifact.add_file(str(ds_path))
            run.log_artifact(artifact)

            with ds_path.open("r") as f:
                length_extrap_data = json.load(f)
            results_data = length_extrap_data.get("results", {})
            perplexities = results_data.get("perplexities", [])
            token_losses = results_data.get("token_losses", [])

            log_dict = {}

            if perplexities:
                ppl_table = wandb.Table(columns=["position", "perplexity"])
                for i, ppl in enumerate(perplexities):
                    ppl_table.add_data(i + 1, ppl)
                log_dict[f"length_extrap/{ds_name}/perplexity_curve"] = wandb.plot.line(
                    ppl_table,
                    "position",
                    "perplexity",
                    title=f"Length Extrapolation Perplexity ({ds_name})",
                )

            if token_losses:
                loss_table = wandb.Table(columns=["position", "loss"])
                for i, loss in enumerate(token_losses):
                    loss_table.add_data(i + 1, loss)
                log_dict[f"length_extrap/{ds_name}/loss_curve"] = wandb.plot.line(
                    loss_table,
                    "position",
                    "loss",
                    title=f"Length Extrapolation Loss ({ds_name})",
                )

            if log_dict:
                run.log(log_dict)

    # Update run summary with eval metrics (doesn't add new history steps)
    if "loss" in result:
        run.summary["eval/loss"] = result["loss"]
        run.summary["eval/perplexity"] = result["perplexity"]
        run.summary["eval/checkpoint_step"] = result.get("micro_steps_completed", 0)
        run.summary["eval/samples"] = result["eval_samples"]
        run.summary["eval/tokens_evaluated"] = result["tokens_evaluated"]

    for key, value in lm_eval_metrics.items():
        run.summary[key] = value

    for key, value in length_extrap_metrics.items():
        run.summary[key] = value

    run.finish()

    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained checkpoint on held-out training samples beyond the "
            "range seen during optimization."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=(
            "Path to a completed training run directory or a parent directory that "
            "contains multiple runs (e.g. logs/language_modeling/2025-11-12/10-55-06/0)."
        ),
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Checkpoint step number to evaluate. Defaults to the highest available step.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=20,
        help="Number of unseen samples to evaluate on (default: 1000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output path. For single runs this writes a JSON file "
            "(default: <run-dir>/eval_<checkpoint>.json). For multi-run sweeps this "
            "path is treated as the summary JSONL (default: <run-dir>/eval_summary.jsonl)."
        ),
    )
    parser.add_argument(
        "--micro-steps",
        type=int,
        default=None,
        help="Override the detected number of completed micro steps.",
    )
    parser.add_argument(
        "--log-to-wandb",
        action="store_true",
        help="Log eval results to the training wandb run.",
    )
    parser.add_argument(
        "--run-lm-eval",
        action="store_true",
        help="Run lm-eval-harness benchmarks in addition to validation perplexity.",
    )
    parser.add_argument(
        "--gen-kwargs",
        type=str,
        default='{"do_sample": false, "temperature": 0, "until": ["\\n\\n"], "use_cache": false}',
        help="Generation kwargs for lm-eval-harness (JSON string).",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation perplexity evaluation (useful when only running length extrapolation).",
    )
    parser.add_argument(
        "--run-length-extrap",
        action="store_true",
        help="Run length extrapolation evaluation in addition to validation perplexity.",
    )
    parser.add_argument(
        "--length-extrap-max-len",
        type=int,
        default=4096,
        help="Maximum sequence length for length extrapolation (default: 4096).",
    )
    parser.add_argument(
        "--length-extrap-dataset",
        type=str,
        default="codeparrot",
        help="Comma-separated list of datasets for length extrapolation (default: codeparrot).",
    )
    parser.add_argument(
        "--length-extrap-num-batches",
        type=int,
        default=20,
        help="Number of batches to evaluate for length extrapolation (default: 20).",
    )
    return parser.parse_args()


def locate_checkpoint(run_dir: Path, checkpoint_step: int | None) -> tuple[Path, int]:
    # Look for checkpoints in checkpoints/ subdirectory first (new layout),
    # then fall back to step_* directly in run_dir (old layout)
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.is_dir():
        search_dir = checkpoints_dir
    else:
        search_dir = run_dir

    step_dirs = sorted(
        (p for p in search_dir.glob("step_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not step_dirs:
        raise FileNotFoundError(f"No checkpoint directories found in {run_dir}")

    if checkpoint_step is None:
        checkpoint_path = step_dirs[-1]
        step_value = int(checkpoint_path.name.split("_")[1])
        return checkpoint_path, step_value

    candidate = search_dir / f"step_{checkpoint_step}"
    if not candidate.is_dir():
        raise FileNotFoundError(
            f"Checkpoint step_{checkpoint_step} not found under {run_dir}"
        )
    return candidate, checkpoint_step


def detect_micro_steps(run_dir: Path) -> int | None:
    micro_steps = []

    # Look for log files in logs/ subdirectory (new layout) or run_dir (old layout)
    logs_dir = run_dir / "logs"
    if logs_dir.is_dir():
        search_dirs = [logs_dir, run_dir]
    else:
        search_dirs = [run_dir]

    for search_dir in search_dirs:
        # Try both old pattern (gla_*.out) and new pattern (gla_*_out.log)
        for pattern in ["gla_*.out", "gla_*_out.log"]:
            for logfile in sorted(search_dir.glob(pattern), key=os.path.getmtime):
                last = None
                with logfile.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        match = MICRO_STEP_PATTERN.search(line)
                        if match:
                            last = int(match.group(1))
                if last is not None:
                    micro_steps.append(last)

    if micro_steps:
        return max(micro_steps)

    log_file = run_dir / "language_modeling.log"
    if log_file.exists():
        last = None
        with log_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                match = MICRO_STEP_PATTERN.search(line)
                if match:
                    last = int(match.group(1))
        if last is not None:
            return last

    return None


def build_eval_loader(cfg, start_index: int, eval_samples: int) -> DataLoader:
    dataset = load_from_disk(cfg.cluster.data_home)
    dataset = dataset.with_format("torch")

    dataset_length = len(dataset)
    end_index = start_index + eval_samples
    if end_index > dataset_length:
        raise ValueError(
            f"Requested evaluation range [{start_index}, {end_index}) exceeds dataset "
            f"length {dataset_length}."
        )

    subset = dataset.select(range(start_index, end_index))

    def collate_fn(batch):
        collated = {
            "input_ids": torch.stack([sample["input_ids"] for sample in batch], dim=0)
        }
        if "docs_lengths" in subset.column_names:
            collated["docs_lengths"] = [
                sample["docs_lengths"].tolist() for sample in batch
            ]
        return collated

    num_workers = 4
    loader_kwargs = {
        "batch_size": 1,
        "sampler": SequentialSampler(subset),
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True

    return DataLoader(subset, **loader_kwargs)


def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    seq_len: int,
    autocast_dtype: torch.dtype,
):
    criterion = CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs, targets, attn_mask = _move_to_device(batch, seq_len, device)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if device.type == "cuda"
                else nullcontext()
            )

            with autocast_ctx:
                outputs = model(inputs, attn_mask)
                logits = getattr(outputs, "logits", outputs)
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                )

            tokens = targets.numel()
            total_loss += loss.item() * tokens
            total_tokens += tokens

    mean_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(mean_loss)
    return {"loss": mean_loss, "perplexity": ppl, "tokens": total_tokens}


def evaluate_run_directory(
    run_dir: Path,
    checkpoint_step: int | None,
    eval_samples: int,
    micro_steps_override: int | None,
    skip_keys: set[tuple[str, str]] | None = None,
):
    run_dir = run_dir.resolve()
    checkpoint_path, step_value = locate_checkpoint(run_dir, checkpoint_step)
    checkpoint_name = checkpoint_path.name
    result_key = (str(run_dir), checkpoint_name)

    if skip_keys is not None and result_key in skip_keys:
        return None, checkpoint_name

    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing Hydra config at {cfg_path}")

    cfg = OmegaConf.load(cfg_path)

    micro_steps = micro_steps_override
    if micro_steps is None:
        detected = detect_micro_steps(run_dir)
        if detected is not None:
            micro_steps = detected
        else:
            micro_steps = int(getattr(cfg.trainer, "steps", 0))

    micro_steps = int(micro_steps)
    micro_batch_size = getattr(cfg.trainer, "micro_batch_size", 1)
    seen_samples = micro_steps * micro_batch_size

    dataloader = build_eval_loader(cfg, seen_samples, eval_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)

    metrics = evaluate(
        model,
        dataloader,
        device,
        cfg.trainer.seq_len,
        autocast_dtype=torch_dtype,
    )

    result = {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
        "learning_rate": float(cfg.trainer.learning_rate),
        "micro_steps_completed": int(micro_steps),
        "samples_seen": int(seen_samples),
        "eval_samples": int(eval_samples),
        "eval_start_index": int(seen_samples),
        "loss": float(metrics["loss"]),
        "perplexity": float(metrics["perplexity"]),
        "tokens_evaluated": int(metrics["tokens"]),
        "device": str(device),
    }

    return result, checkpoint_name


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist.")

    if is_run_directory(run_dir):
        if args.skip_validation:
            checkpoint_path, checkpoint_step = locate_checkpoint(
                run_dir, args.checkpoint_step
            )
            checkpoint_name = checkpoint_path.name
            # Extract micro_steps from checkpoint name (e.g., step_19075 -> 19075)
            micro_steps = (
                int(checkpoint_name.split("_")[1]) if "_" in checkpoint_name else 0
            )
            result = {
                "run_dir": str(run_dir),
                "checkpoint": checkpoint_name,
                "checkpoint_path": str(checkpoint_path),
                "micro_steps_completed": micro_steps,
            }
            print(f"Skipping validation, using checkpoint: {checkpoint_name}")
        else:
            result, checkpoint_name = evaluate_run_directory(
                run_dir,
                args.checkpoint_step,
                args.eval_samples,
                args.micro_steps,
            )

            output_path = args.output
            if output_path is None:
                output_path = run_dir / f"eval_{checkpoint_name}.json"
            else:
                output_path = output_path.resolve()

            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)

            print(json.dumps(result, indent=2))

        lm_eval_results_path = None
        if args.run_lm_eval:
            checkpoint_path = Path(result["checkpoint_path"])
            lm_eval_output_dir = run_dir / "lm_eval_results" / checkpoint_name
            lm_eval_results_path = run_lm_eval_harness(
                checkpoint_path,
                lm_eval_output_dir,
                gen_kwargs=args.gen_kwargs,
            )
            if lm_eval_results_path:
                print(f"lm-eval-harness results saved to: {lm_eval_results_path}")

        length_extrap_results = {}
        if args.run_length_extrap:
            checkpoint_path = Path(result["checkpoint_path"])
            length_extrap_output_dir = run_dir / "length_extrapolation"
            datasets = [d.strip() for d in args.length_extrap_dataset.split(",")]
            for dataset_name in datasets:
                ds_path = run_length_extrapolation(
                    checkpoint_path,
                    length_extrap_output_dir,
                    max_length=args.length_extrap_max_len,
                    dataset_name=dataset_name,
                    num_batches=args.length_extrap_num_batches,
                )
                if ds_path:
                    length_extrap_results[dataset_name] = ds_path
                    print(
                        f"Length extrapolation results ({dataset_name}) saved to: {ds_path}"
                    )

        if args.log_to_wandb:
            if log_to_wandb(
                run_dir,
                result,
                lm_eval_results_path,
                length_extrap_results or None,
            ):
                print("Logged eval results to wandb run")

        return

    run_dirs = discover_run_directories(run_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No Hydra run directories found under {run_dir}")

    summary_path = args.output.resolve() if args.output else run_dir / SUMMARY_FILENAME
    existing = load_existing_summary(summary_path)

    processed_any = False

    for leaf_dir in run_dirs:
        try:
            eval_result, checkpoint_name = evaluate_run_directory(
                leaf_dir,
                args.checkpoint_step,
                args.eval_samples,
                args.micro_steps,
                skip_keys=existing,
            )
        except FileNotFoundError as exc:
            print(f"Skipping {leaf_dir} - {exc}")
            continue

        if eval_result is None:
            print(
                f"Skipping {leaf_dir} - checkpoint {checkpoint_name} already recorded in {summary_path}"
            )
            continue

        append_summary_record(summary_path, eval_result)
        existing.add((eval_result["run_dir"], eval_result["checkpoint"]))
        print(json.dumps(eval_result, indent=2))

        lm_eval_results_path = None
        if args.run_lm_eval:
            checkpoint_path = Path(eval_result["checkpoint_path"])
            lm_eval_output_dir = leaf_dir / "lm_eval_results" / checkpoint_name
            lm_eval_results_path = run_lm_eval_harness(
                checkpoint_path,
                lm_eval_output_dir,
                gen_kwargs=args.gen_kwargs,
            )
            if lm_eval_results_path:
                print(f"lm-eval-harness results saved to: {lm_eval_results_path}")

        length_extrap_results = {}
        if args.run_length_extrap:
            checkpoint_path = Path(eval_result["checkpoint_path"])
            length_extrap_output_dir = leaf_dir / "length_extrapolation"
            datasets = [d.strip() for d in args.length_extrap_dataset.split(",")]
            for dataset_name in datasets:
                ds_path = run_length_extrapolation(
                    checkpoint_path,
                    length_extrap_output_dir,
                    max_length=args.length_extrap_max_len,
                    dataset_name=dataset_name,
                    num_batches=args.length_extrap_num_batches,
                )
                if ds_path:
                    length_extrap_results[dataset_name] = ds_path
                    print(
                        f"Length extrapolation results ({dataset_name}) saved to: {ds_path}"
                    )

        if args.log_to_wandb:
            if log_to_wandb(
                leaf_dir,
                eval_result,
                lm_eval_results_path,
                length_extrap_results or None,
            ):
                print(f"Logged eval results to wandb run for {leaf_dir}")

        processed_any = True

    if not processed_any:
        print("No new checkpoints to evaluate.")


if __name__ == "__main__":
    main()
