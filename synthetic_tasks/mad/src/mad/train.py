import os
import random
import shutil

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from mad.configs import MADConfig, build_model_from_hydra_config
from mad.data import generate_data
from mad.model import PLModelWrap
from mad.paths import make_log_path
from mad.registry import layer_registry

# train model according to mad_config:


def train(
    model: torch.nn.Module,
    mad_config: MADConfig,
    log_path: str,
    log_to_csv: bool = True,
    log_to_wandb: bool = False,
    wandb_project: str = "MAD",
    save_checkpoints: bool = True,
) -> pd.DataFrame:
    """
    Train a model with given configuration and log results.

    Args:
        model (nn.Module): model to train
        mad_config (MADConfig): MAD configuration
        log_path (str): path to logs
        log_to_csv (bool): if True, log results to csv in log_path
        log_to_wandb (bool): if True, log results to Weights & Biases
        wandb_project (str): name of Weights & Biases project to log to
        save_checkpoints (bool): if True, save model checkpoints

    Returns:
        results_df (pd.DataFrame): results of training
    """

    # Set random seed.

    random.seed(mad_config.seed)
    np.random.seed(mad_config.seed)
    torch.manual_seed(mad_config.seed)

    # Check if results exist already.

    if os.path.exists(log_path):
        path_results_df = os.path.join(log_path, "results.csv")
        if os.path.exists(path_results_df):
            results_df = pd.read_csv(path_results_df)
            print(f'Log path "{log_path}" exists, retrieved results from there...')
            return results_df
        else:
            shutil.rmtree(log_path)

    # PyTorch Lightning Model Wrap.

    model_wrapped = PLModelWrap(model=model, mad_config=mad_config)

    # Make Data.

    data = generate_data(
        instance_fn=mad_config.instance_fn,
        instance_fn_kwargs=mad_config.instance_fn_kwargs,
        train_data_path=mad_config.train_dataset_path,
        test_data_path=mad_config.test_dataset_path,
        num_train_examples=mad_config.num_train_examples,
        num_test_examples=mad_config.num_test_examples,
        num_workers=mad_config.num_data_workers,
    )

    # Make Dataloaders.

    train_dl = DataLoader(
        dataset=data["train"],
        batch_size=mad_config.batch_size,
        shuffle=True,
        num_workers=mad_config.num_data_workers,
        persistent_workers=mad_config.persistent_data_workers
        and mad_config.num_data_workers > 0,
    )

    test_dl = DataLoader(
        dataset=data["test"],
        batch_size=mad_config.batch_size,
        shuffle=False,
        num_workers=mad_config.num_data_workers,
        persistent_workers=mad_config.persistent_data_workers
        and mad_config.num_data_workers > 0,
    )

    # Make Loggers & Callbacks.

    early_stop = pl.callbacks.EarlyStopping(
        monitor="test/Accuracy_epoch",
        min_delta=0.00,
        stopping_threshold=0.999,
        patience=mad_config.stop_patience
        if mad_config.early_stop
        else mad_config.epochs,
        verbose=True,
        mode="max",
    )
    callbacks = [early_stop]

    if save_checkpoints and log_path is not None:
        checkpoint_best = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            monitor="test/Perplexity_epoch",
            mode="min",
            dirpath=os.path.join(log_path, "checkpoints"),
            filename="best",
        )
        checkpoint_last = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            monitor="epoch",
            mode="max",
            dirpath=os.path.join(log_path, "checkpoints"),
            filename="last",
        )
        callbacks += [checkpoint_best, checkpoint_last]

    loggers = []
    if log_to_csv and log_path is not None:
        loggers.append(pl.loggers.CSVLogger(save_dir=log_path, name="logs", version=""))

    if log_to_wandb:
        # We import wandb here so it doesn't create any random directories in /tmp
        # when not used
        import wandb

        wandb.init(
            project=wandb_project,
            name=os.path.basename(log_path) if log_path is not None else None,
        )
        loggers.append(pl.loggers.WandbLogger())

    # set default precision of float32 matrix multiplications:
    torch.set_float32_matmul_precision("high")

    # Make Trainer.

    trainer = pl.Trainer(
        max_epochs=mad_config.epochs,
        accelerator=mad_config.accelerator if torch.cuda.is_available() else "cpu",
        devices=mad_config.devices,
        logger=loggers,
        enable_checkpointing=mad_config.save_checkpoints,
        callbacks=callbacks,
        precision=mad_config.precision,
    )

    # Train.

    trainer.fit(model_wrapped, train_dl, test_dl)

    # Evaluate Final Performance.

    results_train = trainer.validate(dataloaders=train_dl)[0]
    results_test = trainer.validate(dataloaders=test_dl)[0]
    results_df = pd.DataFrame(
        {
            # training data:
            "train_acc": results_train[
                "test/Accuracy_epoch"
            ],  # its called "test/..." because we compute results with trainer.validate
            "train_ppl": results_train["test/Perplexity_epoch"],
            "train_loss": results_train["test/Loss_epoch"],
            # test data:
            "test_acc": results_test["test/Accuracy_epoch"],
            "test_ppl": results_test["test/Perplexity_epoch"],
            "test_loss": results_test["test/Loss_epoch"],
        },
        index=[0],
    )
    results_df.to_csv(os.path.join(log_path, "results.csv"), index=False)

    # Done!

    return results_df


@hydra.main(config_path="../../../../configs/mad", config_name="mad", version_base=None)
def main(cfg: DictConfig):

    # Build MADConfig from Hydra config.

    mad_config = MADConfig.from_hydra(cfg)

    # Build model from Hydra config.

    model = build_model_from_hydra_config(cfg)
    model_id = "-".join(layer_registry[l]["shorthand"] for l in cfg.model.layers)

    # Create log path.

    log_path = make_log_path(
        base_path=cfg.log_base_path, mad_config=mad_config, model_id=model_id
    )

    # Train.

    train(
        model=model,
        mad_config=mad_config,
        log_path=log_path,
        log_to_csv=cfg.log_to_csv,
        log_to_wandb=cfg.log_to_wandb,
        wandb_project=cfg.logger.wandb_project,
        save_checkpoints=cfg.save_checkpoints,
    )


if __name__ == "__main__":
    main()
