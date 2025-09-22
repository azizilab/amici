import logging
import warnings

import numpy as np
import pandas as pd
import wandb
from pytorch_lightning.loggers import WandbLogger
from scvi import settings
from scvi.model.base import BaseModelClass
from scvi.train import TrainRunner

logger = logging.getLogger(__name__)


class WandbTrainRunner(TrainRunner):
    def __init__(self, model: BaseModelClass, wandb_logger: WandbLogger, wandb_run, **trainer_kwargs):
        super().__init__(model, **trainer_kwargs)

        assert wandb_run is not None and wandb_logger is not None

        self.wandb_log = {}

        self.trainer.loggers = [wandb_logger]
        self.trainer._model = model

    def _update_history(self):
        if self.model.is_trained_ is True:
            if not isinstance(self.model.history_, dict):
                warnings.warn(
                    "Training history cannot be updated. Logger can be accessed from " "`model.trainer.logger`",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
                return
            else:
                new_history = self.trainer.logger.history
                for key, val in self.model.history_.items():
                    if key not in new_history:
                        continue
                    prev_len = len(val)
                    new_len = len(new_history[key])
                    index = np.arange(prev_len, prev_len + new_len)
                    new_history[key].index = index

                    if key == "train_loss_epoch" or key == "validation_loss":
                        self.wandb_log[key] = new_history[key]

                    self.model.history_[key] = pd.concat(
                        [
                            val,
                            new_history[key],
                        ]
                    )
                    self.model.history_[key].index.name = val.index.name
                wandb.log(self.wandb_log)
                self.wandb_log = {}
        else:
            try:
                self.model.history_ = self.trainer.logger.history
            except AttributeError:
                self.history_ = None
