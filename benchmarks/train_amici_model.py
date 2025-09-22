import itertools
import json
import os
import shutil
from multiprocessing import Manager, Process
from typing import Any

import numpy as np
import pytorch_lightning as pl
import scanpy as sc
from anndata import AnnData

from amici import AMICI
from amici.callbacks import AttentionPenaltyMonitor


def train_model(
    adata: AnnData,
    dataset_config: dict,
    penalty_params: dict,
    exp_params: dict,
    run_id: str,
) -> int:
    """
    Train an AMICI model with the given parameters.

    Args:
        seed: The seed for the random number generator.
        penalty_params: The parameters for the penalty.
        model_params: The parameters for the model.
        exp_params: The parameters for the experiment.
        run_id: The run ID.

    Returns
    -------
        model: The trained model.
        test_elbo: The test elbo score of the model.
    """
    model_path = f"results/{snakemake.wildcards.dataset}_{snakemake.wildcards.seed}/saved_models/amici_model_{run_id}"  # noqa: F821

    if not os.path.exists(os.path.join(model_path, "model.pt")):
        adata_train = adata[adata.obs["train_test_split"] == "train"]

        pl.seed_everything(exp_params.get("seed", 42))

        AMICI.setup_anndata(
            adata_train,
            labels_key=dataset_config["labels_key"],
            coord_obsm_key="spatial",
            n_neighbors=int(exp_params.get("n_neighbors", 50)),
        )
        model = AMICI(
            adata_train,
            n_heads=8,
            value_l1_penalty_coef=penalty_params.get("value_l1_penalty_coef", 1e-6),
        )

        plan_kwargs = {}
        if "lr" in exp_params:
            plan_kwargs["lr"] = float(exp_params.get("lr", 1e-3))

        model.train(
            max_epochs=int(exp_params.get("epochs", 400)),
            batch_size=int(exp_params.get("batch_size", 128)),
            plan_kwargs=plan_kwargs,
            early_stopping=exp_params.get("early_stopping", True),
            early_stopping_monitor=exp_params.get("early_stopping_monitor", "elbo_validation"),
            early_stopping_patience=int(exp_params.get("early_stopping_patience", 20)),
            check_val_every_n_epoch=1,
            use_wandb=False,
            callbacks=[
                AttentionPenaltyMonitor(
                    start_val=float(penalty_params.get("start_val", 1e-6)),
                    end_val=float(penalty_params.get("end_val", 1e-2)),
                    epoch_start=int(penalty_params.get("epoch_start", 10)),
                    epoch_end=int(penalty_params.get("epoch_end", 40)),
                    flavor=penalty_params.get("flavor", "linear"),
                ),
            ],
        )

        AMICI.setup_anndata(
            adata,
            labels_key=dataset_config["labels_key"],
            coord_obsm_key="spatial",
            n_neighbors=int(exp_params.get("n_neighbors", 50)),
        )

        model.save(
            model_path,
            overwrite=True,
        )

    model = AMICI.load(
        model_path,
        adata=adata,
    )

    test_recons = (
        model.get_reconstruction_error(
            adata, indices=np.where(adata.obs["train_test_split"] == "test")[0], batch_size=128
        )["reconstruction_loss"]
        .detach()
        .cpu()
        .numpy()
    )

    return model_path, test_recons


def train_and_evaluate(
    adata: AnnData,
    dataset_config: dict,
    penalty_params: dict,
    exp_params: dict,
    results_dict: dict[str, Any],
    run_id: str,
) -> None:
    """
    Run a single training job and store results in shared dictionary

    Args:
        adata: The AnnData object.
        dataset_config: The dataset configuration.
        penalty_params: The penalty parameters.
        model_params: The model parameters.
        exp_params: The experiment parameters.
        results_dict: The shared dictionary to store results.
        run_id: The run ID.
    """
    model_path, test_recons = train_model(adata, dataset_config, penalty_params, exp_params, run_id)
    model_config = {
        "penalty_params": penalty_params,
        "exp_params": exp_params,
    }
    results_dict[run_id] = {
        "model_config": model_config,
        "test_recons": test_recons,
        "model_path": model_path,
        "seed": exp_params["seed"],
    }


def main():
    """Train the AMICI model."""
    adata = sc.read_h5ad(snakemake.input.adata_path)  # noqa: F821
    adata.obs_names_make_unique()

    dataset_config = snakemake.config["datasets"][snakemake.wildcards.dataset]  # noqa: F821

    # Define parameter grid
    end_penalty_values = [1e-2, 1e-3, 1e-4]
    value_l1_penalty_values = [1e-6, 1e-5, 1e-4]
    schedule_flavors = ["linear"]
    seeds = [22, 38, 17, 11, 42, 33, 18]

    # Create all parameter combinations
    all_runs = []
    for item in itertools.product(end_penalty_values, schedule_flavors, value_l1_penalty_values, seeds):
        run_params = {
            "penalty_params": {"end_val": item[0], "flavor": item[1], "value_l1_penalty_coef": item[2]},
            "exp_params": {"seed": item[3]},
        }
        all_runs.append(run_params)

    # Set up parallel processing
    num_agents = 4
    manager = Manager()
    results_dict = manager.dict()

    # Process runs in batches
    for i in range(0, len(all_runs), num_agents):
        batch = all_runs[i : i + num_agents]
        processes = []

        # Start processes for current batch
        for j, run_params in enumerate(batch):
            run_id = f"run_{i+j}"
            p = Process(
                target=train_and_evaluate,
                args=(
                    adata,
                    dataset_config,
                    run_params["penalty_params"],
                    run_params["exp_params"],
                    results_dict,
                    run_id,
                ),
            )
            p.start()
            processes.append(p)

        # Wait for all processes in batch to complete
        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            for p in processes:
                p.terminate()
        finally:
            for p in processes:
                p.join()

    # Find best model
    best_run_id = min(list(results_dict.keys()), key=lambda k: results_dict[k]["test_recons"])
    best_model_path = results_dict[best_run_id]["model_path"]
    best_recons = results_dict[best_run_id]["test_recons"]

    # Save best model
    shutil.copy(os.path.join(best_model_path, "model.pt"), snakemake.output[0])  # noqa: F821

    # Save best model seed
    with open(snakemake.output[1], "w") as f:  # noqa: F821
        f.write(str(results_dict[best_run_id]["seed"]))

    # Save the model parameters
    with open(
        f"results/{snakemake.wildcards.dataset}_{snakemake.wildcards.seed}/model_params.json",  # noqa: F821
        "w",
    ) as f:
        model = AMICI.load(
            best_model_path,
            adata=adata,
        )
        penalty_params = results_dict[best_run_id]["model_config"]["penalty_params"]
        exp_params = results_dict[best_run_id]["model_config"]["exp_params"]
        model_config = {
            "n_heads": model.module.n_heads,
            "value_l1_penalty_coef": penalty_params["value_l1_penalty_coef"],
            "end_penalty_coef": penalty_params["end_val"],
            "seed": exp_params["seed"],
            "n_neighbors": model.n_neighbors,
            "penalty_flavor": penalty_params["flavor"],
        }
        json.dump(model_config, f)

    # Save the reconstruction loss
    with open(snakemake.output[2], "w") as f:  # noqa: F821
        f.write(str(best_recons))


if __name__ == "__main__":
    main()
