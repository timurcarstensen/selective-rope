import json
import re
import subprocess
from pathlib import Path

import git
import hydra
from omegaconf import DictConfig, OmegaConf

import selective_rope  # noqa

git_root = git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")


def snapshot_code_worktree(output_dir: Path, repo_root: str) -> Path:
    """Create isolated code snapshot using git worktree.

    Creates a detached worktree at {output_dir}/snapshot/code with the current HEAD,
    then applies any uncommitted changes. This ensures the job runs with
    exactly the code state at submission time, isolated from later changes.

    Returns the path to the worktree (code directory).
    """
    repo = git.Repo(repo_root)
    snapshot_dir = output_dir / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    code_dir = snapshot_dir / "code"

    # Save commit hash for reproducibility
    commit_hash = repo.head.commit.hexsha
    (snapshot_dir / "git_commit.txt").write_text(commit_hash + "\n")

    # Save uncommitted changes (staged + unstaged) as a patch
    # Exclude paths that won't be in the sparse checkout
    diff_result = subprocess.run(
        [
            "git",
            "diff",
            "HEAD",
            "--",
            ".",
            ":(exclude)plotting",
            ":(exclude)logs",
            ":(exclude)tests",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    uncommitted_diff = diff_result.stdout
    patch_file = snapshot_dir / "git_diff.patch"
    if uncommitted_diff:
        patch_file.write_text(uncommitted_diff)

    # Create detached worktree at current HEAD (without checking out files yet)
    subprocess.run(
        ["git", "worktree", "add", "--detach", "--no-checkout", str(code_dir), "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    # Configure sparse checkout to exclude heavy directories (non-cone mode for negations)
    subprocess.run(
        ["git", "sparse-checkout", "init", "--no-cone"],
        cwd=code_dir,
        check=True,
        capture_output=True,
    )
    # Set sparse-checkout patterns via stdin: include everything, exclude heavy directories
    sparse_excludes = [
        "/*",
        "!/plotting/",
        "!/logs/",
        "!/tests/",
    ]
    subprocess.run(
        ["git", "sparse-checkout", "set", "--no-cone", "--stdin"],
        input="\n".join(sparse_excludes) + "\n",
        cwd=code_dir,
        check=True,
        capture_output=True,
        text=True,
    )

    # Now checkout the files (respecting sparse checkout)
    subprocess.run(
        ["git", "checkout", "HEAD"],
        cwd=code_dir,
        check=True,
        capture_output=True,
    )

    # Apply uncommitted changes to the worktree
    if uncommitted_diff:
        result = subprocess.run(
            ["git", "apply", str(patch_file)],
            cwd=code_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to apply patch:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

    return code_dir


def extract_job_id(sbatch_output: str) -> str | None:
    """Extract job ID from sbatch output like 'Submitted batch job 12345'."""
    match = re.search(r"Submitted batch job (\d+)", sbatch_output)
    return match.group(1) if match else None


def validate_config(cfg: DictConfig, output_dir: Path) -> None:
    """Validate config before scheduling. Raises on failure."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        import selective_rope  # noqa

        # Model config must be loadable and model must be instantiable
        config = AutoConfig.from_pretrained(str(output_dir / "model_config.json"))
        with torch.device("meta"):
            AutoModelForCausalLM.from_config(config)

        # Data path must exist
        assert Path(cfg.cluster.data_home).exists(), (
            f"Data path does not exist: {cfg.cluster.data_home}"
        )

        # Sanity check step budget
        assert cfg.trainer.steps * cfg.trainer.grad_accumulation_steps > 0, (
            "Invalid step budget"
        )


@hydra.main(
    version_base=None,
    config_path=str(Path(git_root) / "configs/language_modeling"),
    config_name="language_modeling",
)
def main(cfg: DictConfig) -> None:
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # type: ignore

    # Create subdirectories for organized output
    configs_dir = Path(output_dir) / "configs"
    scripts_dir = Path(output_dir) / "scripts"
    logs_dir = Path(output_dir) / "logs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # set trainer output dir
    cfg.trainer.output_dir = output_dir

    # save model config to json in configs_dir and point launch dir there
    with open(configs_dir / "model_config.json", "w") as fp:
        model_config = OmegaConf.to_container(cfg.model, resolve=True)
        json.dump(model_config, fp, indent=0)

    cfg.trainer.model_name_or_path = str(configs_dir / "model_config.json")

    with open(configs_dir / "trainer_config.json", "w") as fp:
        trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
        json.dump(trainer_config, fp, indent=0)

    # Validate config before scheduling
    validate_config(cfg, configs_dir)

    # Prune stale worktrees (cleanup from deleted log directories)
    subprocess.run(["git", "worktree", "prune"], cwd=git_root, capture_output=True)

    # Create isolated code snapshot via git worktree
    code_dir = snapshot_code_worktree(Path(output_dir), git_root)

    # Generate training job script
    with open(
        Path(git_root) / f"configs/language_modeling/{cfg.cluster.filename}"
    ) as f:
        sweep_template = f.read()

        sweep_file = sweep_template.format(
            project_home=git_root,
            code_dir=str(code_dir),
            wandb_resume=cfg.logger.wandb_resume,
            wandb_project=cfg.logger.wandb_project,
            wandb_entity=cfg.logger.wandb_entity,
            wandb_group=cfg.logger.wandb_group,
            wandb_dir=str(Path(output_dir)),
            wandb_run_name=cfg.logger.wandb_run_name,
            output_dir=cfg.trainer.output_dir,
            logs_dir=str(logs_dir),
            trainer_config_path=str(Path(output_dir) / ".hydra/config.yaml"),
            nodes=cfg.cluster.nodes,
            gpus_per_node=cfg.cluster.gpus_per_node,
            partition=cfg.cluster.partition,
            job_name=cfg.cluster.job_name,
            mem_per_gpu=cfg.cluster.mem_per_gpu,
            cpus_per_gpu=cfg.cluster.cpus_per_gpu,
            time_limit=cfg.cluster.time_limit,
            exclude=cfg.cluster.exclude,
            mail_user=cfg.cluster.mail_user,
            mail_type=cfg.cluster.mail_type,
            licenses=cfg.cluster.licenses,
        )

        sweep_path = scripts_dir / "job_script.sh"
        with open(sweep_path, "w") as f:
            f.write(sweep_file)

        sweep_path.chmod(0o755)

        with open(Path(output_dir) / ".hydra/config.yaml", "w") as f:
            OmegaConf.save(cfg, f)

    # Generate evaluation job script
    eval_template_path = (
        Path(git_root)
        / f"configs/language_modeling/{cfg.cluster.filename.replace('.sh', '_eval.sh')}"
    )
    if eval_template_path.exists():
        with open(eval_template_path) as f:
            eval_template = f.read()

        eval_file = eval_template.format(
            project_home=git_root,
            code_dir=str(code_dir),
            output_dir=cfg.trainer.output_dir,
            logs_dir=str(logs_dir),
        )

        eval_script_path = scripts_dir / "eval_job_script.sh"
        with open(eval_script_path, "w") as f:
            f.write(eval_file)
        eval_script_path.chmod(0o755)

    # Submit training job and capture job ID
    result = subprocess.run(
        ["sbatch", str(sweep_path)],
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip())

    # Submit eval job with dependency on training job
    train_job_id = extract_job_id(result.stdout)
    if train_job_id and eval_template_path.exists():
        eval_result = subprocess.run(
            [
                "sbatch",
                f"--dependency=afterok:{train_job_id}",
                str(eval_script_path),
            ],
            capture_output=True,
            text=True,
        )
        print(eval_result.stdout.strip())


if __name__ == "__main__":
    main()
