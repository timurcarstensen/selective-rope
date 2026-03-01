"""Pre-generate all MAD benchmark datasets.

Usage (from repo root, in an interactive session):
    uv run --frozen --no-sync python synthetic_tasks/mad/generate_data.py [--data-path ./data/mad] [--num-workers 8]
"""

import argparse
import os

from mad.configs import make_benchmark_mad_configs
from mad.data import generate_data


def main():
    parser = argparse.ArgumentParser(description="Pre-generate MAD benchmark datasets")
    parser.add_argument(
        "--data-path", default="./data/mad", help="Base path for datasets"
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Workers for parallel generation"
    )
    args = parser.parse_args()

    mad_configs = make_benchmark_mad_configs(data_path=args.data_path)

    # Deduplicate by dataset path (many configs share data, differing only in lr/wd).
    seen = set()
    unique_configs = []
    for cfg in mad_configs:
        if cfg.dataset_path not in seen:
            seen.add(cfg.dataset_path)
            unique_configs.append(cfg)

    print(
        f"Found {len(unique_configs)} unique datasets ({len(mad_configs)} total configs)"
    )

    for i, cfg in enumerate(unique_configs):
        train_path = cfg.train_dataset_path
        test_path = cfg.test_dataset_path
        if os.path.exists(train_path) and os.path.exists(test_path):
            print(
                f"[{i + 1}/{len(unique_configs)}] {cfg.dataset_path} -- already exists, skipping"
            )
            continue

        print(f"[{i + 1}/{len(unique_configs)}] Generating {cfg.dataset_path} ...")
        generate_data(
            instance_fn=cfg.instance_fn,
            instance_fn_kwargs=cfg.instance_fn_kwargs,
            train_data_path=train_path,
            test_data_path=test_path,
            num_train_examples=cfg.num_train_examples,
            num_test_examples=cfg.num_test_examples,
            num_workers=args.num_workers,
        )

    print("Done!")


if __name__ == "__main__":
    main()
