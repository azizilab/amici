import os
import warnings
from contextlib import nullcontext, redirect_stdout

import numpy as np
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


def _log_attention_penalty_coef_update(epoch, epoch_start, epoch_end, start_val, end_val):
    if epoch < epoch_start:
        return 0
    elif epoch_start <= epoch < epoch_end:
        log_space = np.logspace(
            np.log10(start_val),
            np.log10(end_val),
            num=(epoch_end - epoch_start),
            endpoint=True,
            base=10.0,
        )
        return log_space[epoch - epoch_start]
    else:
        return end_val


def _linear_attention_penalty_coef_update(epoch, epoch_start, epoch_end, start_val, end_val):
    if epoch < epoch_start:
        return 0
    elif epoch_start <= epoch < epoch_end:
        return start_val + (end_val - start_val) * (epoch - epoch_start) / (epoch_end - epoch_start)
    else:
        return end_val


class AttentionPenaltyMonitor(Callback):
    def __init__(
        self,
        epoch_start=10,
        epoch_end=30,
        start_val=1e-4,
        end_val=1e-3,
        flavor="log",
    ):
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end
        self.start_val = start_val
        self.end_val = end_val
        self.flavor = flavor

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.flavor == "log":
            attention_penalty_coef = _log_attention_penalty_coef_update(
                pl_module.current_epoch,
                self.epoch_start,
                self.epoch_end,
                self.start_val,
                self.end_val,
            )
        elif self.flavor == "linear":
            attention_penalty_coef = _linear_attention_penalty_coef_update(
                pl_module.current_epoch,
                self.epoch_start,
                self.epoch_end,
                self.start_val,
                self.end_val,
            )
        pl_module.module.attention_penalty_coef = attention_penalty_coef


class ModelInterpretationLogging(Callback):
    def __init__(self, n_epochs_plot: int = 1, verbose: bool = False):
        self.epoch = 0
        self.n_epochs_plot = n_epochs_plot
        self.verbose = verbose

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.epoch += 1
        if self.epoch % self.n_epochs_plot == 0:
            model = trainer._model
            with (
                open(os.devnull, "w") as f,
                redirect_stdout(f) if self.verbose else nullcontext(),
                warnings.catch_warnings() if self.verbose else nullcontext(),
            ):
                warnings.simplefilter("ignore")
                attention_patterns = model.get_attention_patterns(model.adata, epoch=self.epoch, wandb_log=True)
                attention_patterns.plot_attention_summary(wandb_log=True)
                explained_variance_scores = model.get_expl_variance_scores(
                    model.adata, epoch=self.epoch, wandb_log=True
                )
                explained_variance_scores.plot_explained_variance_barplot(wandb_log=True)
