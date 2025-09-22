from typing import Optional, Union

import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger
from scvi.dataloaders import DataSplitter
from scvi.model._utils import get_max_epochs_heuristic
from scvi.train import TrainingPlan, TrainRunner
from scvi.utils._docstrings import devices_dsp

from ._wandb_trainrunner import WandbTrainRunner


class WandbUnsupervisedTrainingMixin:
    _data_splitter_cls = DataSplitter
    _training_plan_cls = TrainingPlan
    _train_runner_cls = WandbTrainRunner

    @devices_dsp.dedent
    def train(
        self,
        max_epochs: Optional[int] = None,
        accelerator: str = "auto",
        devices: Union[int, list[int], str] = "auto",
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        external_indexing: Optional[list[np.array, np.array, np.array]] = None,
        shuffle_set_split: bool = True,
        batch_size: int = 128,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        **trainer_kwargs,
    ):
        """Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        %(param_accelerator)s
        %(param_devices)s
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        shuffle_set_split
            Whether to shuffle indices before splitting. If `False`, the val, train, and test set are split in the
            sequential order of the data according to `validation_size` and `train_size` percentages.
        external_indexing
            A list of data split indices in the order of training, validation, and test sets.
            Validation and test set are not required and can be left empty.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        wandb_kwargs = {}
        if use_wandb:
            self._train_runner_cls = WandbTrainRunner
            wandb_logger = WandbLogger(
                project=wandb_project,
            )
            run_name = wandb_run_name if wandb_run_name else "test_run"

            wandb_run = wandb.init(project=wandb_project, entity=wandb_entity, name=run_name)
            wandb_kwargs["wandb_logger"] = wandb_logger
            wandb_kwargs["wandb_run"] = wandb_run

        else:
            self._train_runner_cls = TrainRunner

        if max_epochs is None:
            max_epochs = get_max_epochs_heuristic(self.adata.n_obs)

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}

        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            shuffle_set_split=shuffle_set_split,
            external_indexing=external_indexing,
        )
        training_plan = self._training_plan_cls(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]

        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            **wandb_kwargs,
            **trainer_kwargs,
        )

        return runner()
