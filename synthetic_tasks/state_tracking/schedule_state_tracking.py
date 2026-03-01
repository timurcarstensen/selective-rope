import re
import subprocess
from pathlib import Path

import git
import hydra
from omegaconf import DictConfig, OmegaConf

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

    # Configure sparse checkout to exclude heavy directories
    subprocess.run(
        ["git", "sparse-checkout", "init", "--no-cone"],
        cwd=code_dir,
        check=True,
        capture_output=True,
    )
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

    # Checkout the files (respecting sparse checkout)
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


@hydra.main(
    version_base=None,
    config_path=str(Path(git_root) / "configs/state_tracking"),
    config_name="state_tracking",
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

    # Save resolved config
    with open(Path(output_dir) / ".hydra/config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # Prune stale worktrees (cleanup from deleted log directories)
    subprocess.run(["git", "worktree", "prune"], cwd=git_root, capture_output=True)

    # Create isolated code snapshot via git worktree
    code_dir = snapshot_code_worktree(Path(output_dir), git_root)

    # Generate job script from template
    with open(Path(git_root) / f"configs/state_tracking/{cfg.cluster.filename}") as f:
        job_template = f.read()

    job_file = job_template.format(
        project_home=git_root,
        code_dir=str(code_dir),
        wandb_resume=cfg.logger.wandb_resume,
        wandb_project=cfg.logger.wandb_project_name,
        wandb_entity=cfg.logger.wandb_entity,
        wandb_group=cfg.logger.wandb_group,
        wandb_dir=str(Path(output_dir)),
        wandb_run_name=cfg.logger.run_name,
        config_path=str(Path(output_dir) / ".hydra"),
        logs_dir=str(logs_dir),
    )

    job_path = scripts_dir / "job_script.sh"
    with open(job_path, "w") as f:
        f.write(job_file)
    job_path.chmod(0o755)

    # Submit job
    result = subprocess.run(
        ["sbatch", str(job_path)],
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
