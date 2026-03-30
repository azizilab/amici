import itertools
import json
import os
import shutil
import tempfile
from multiprocessing import Manager, Process
from typing import Any

import numpy as np
import pytorch_lightning as pl
import scanpy as sc
from anndata import AnnData
from gpu_utils import select_gpu

from amici import AMICI
from amici.callbacks import AttentionPenaltyMonitor


def train_model(
    adata: AnnData,
    dataset_config: dict,
    penalty_params: dict,
    exp_params: dict,
    run_id: str,
    eval_indices=None,
    save_model: bool = True,
) -> tuple:
    """
    Train an AMICI model with the given parameters.

    Args:
        adata: The full AnnData object. Its train_test_split column determines
            which cells are used for training (== "train").
        dataset_config: Dataset configuration dict.
        penalty_params: The parameters for the penalty.
        exp_params: The parameters for the experiment.
        run_id: The run ID (used as cache/save path key when save_model=True).
        eval_indices: Indices into adata to evaluate reconstruction loss on.
            Defaults to test cells when None (original behaviour).
        save_model: If False, train without persisting the model to disk
            (used during CV sweeps).

    Returns
    -------
        (model_path_or_None, recons_loss)
    """
    if save_model:
        model_path = (
            f"results/{snakemake.wildcards.dataset}_{snakemake.wildcards.seed}/saved_models/amici_model_{run_id}"  # noqa: F821
        )
    else:
        model_path = None

    adata_train = adata[adata.obs["train_test_split"] == "train"]

    # Cache hit: skip re-training when the model file already exists on disk
    if save_model and model_path and os.path.exists(os.path.join(model_path, "model.pt")):
        model = AMICI.load(model_path, adata=adata)
        if eval_indices is None:
            eval_indices = np.where(adata.obs["train_test_split"] == "test")[0]
        recons = (
            model.get_reconstruction_error(adata, indices=eval_indices, batch_size=128)["reconstruction_loss"]
            .detach()
            .cpu()
            .numpy()
        )
        return model_path, recons

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model_path = os.path.join(tmpdir, "model")

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

        # Re-register the full adata before evaluating
        AMICI.setup_anndata(
            adata,
            labels_key=dataset_config["labels_key"],
            coord_obsm_key="spatial",
            n_neighbors=int(exp_params.get("n_neighbors", 50)),
        )

        save_path = model_path if save_model else tmp_model_path
        model.save(save_path, overwrite=True)

        if eval_indices is None:
            eval_indices = np.where(adata.obs["train_test_split"] == "test")[0]

        recons = (
            model.get_reconstruction_error(adata, indices=eval_indices, batch_size=128)["reconstruction_loss"]
            .detach()
            .cpu()
            .numpy()
        )

    if save_model:
        return model_path, recons
    return None, recons


def train_and_evaluate(
    adata: AnnData,
    dataset_config: dict,
    penalty_params: dict,
    exp_params: dict,
    results_dict: dict[str, Any],
    run_id: str,
    eval_indices=None,
    save_model: bool = True,
) -> None:
    """Run a single training job and store results in shared dictionary."""
    model_path, test_recons = train_model(
        adata,
        dataset_config,
        penalty_params,
        exp_params,
        run_id,
        eval_indices=eval_indices,
        save_model=save_model,
    )
    results_dict[run_id] = {
        "model_config": {"penalty_params": penalty_params, "exp_params": exp_params},
        "test_recons": test_recons,
        "model_path": model_path,
        "seed": exp_params["seed"],
    }


def _run_sweep_parallel(adata, dataset_config, all_runs, eval_indices, save_model, base_run_id=0):
    """Run hyperparameter configurations in batches of 4 parallel processes.

    Returns a dict mapping run_id -> result dict.
    """
    num_agents = 4
    manager = Manager()
    results_dict = manager.dict()

    for i in range(0, len(all_runs), num_agents):
        batch = all_runs[i : i + num_agents]
        processes = []

        for j, run_params in enumerate(batch):
            run_id = f"run_{base_run_id + i + j}"
            p = Process(
                target=train_and_evaluate,
                args=(
                    adata,
                    dataset_config,
                    run_params["penalty_params"],
                    run_params["exp_params"],
                    results_dict,
                    run_id,
                    eval_indices,
                    save_model,
                ),
            )
            p.start()
            processes.append(p)

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            for p in processes:
                p.terminate()
        finally:
            for p in processes:
                p.join()

    return dict(results_dict)


def _build_run_configs():
    end_penalty_values = [1e-2, 1e-3, 1e-4]
    value_l1_penalty_values = [1e-6, 1e-5, 1e-4]
    schedule_flavors = ["linear"]
    seeds = [22, 38, 17, 11, 42, 33, 18]

    all_runs = []
    for item in itertools.product(end_penalty_values, schedule_flavors, value_l1_penalty_values, seeds):
        all_runs.append(
            {
                "penalty_params": {"end_val": item[0], "flavor": item[1], "value_l1_penalty_coef": item[2]},
                "exp_params": {"seed": item[3]},
            }
        )
    return all_runs


def _save_best_model(adata, best_run_params, best_model_path, best_recons):
    """Copy best model to snakemake outputs and write param/loss files."""
    shutil.copy(os.path.join(best_model_path, "model.pt"), snakemake.output[0])  # noqa: F821

    with open(snakemake.output[1], "w") as f:  # noqa: F821
        f.write(str(best_run_params["exp_params"]["seed"]))

    with open(
        f"results/{snakemake.wildcards.dataset}_{snakemake.wildcards.seed}/model_params.json",  # noqa: F821
        "w",
    ) as f:
        model = AMICI.load(best_model_path, adata=adata)
        penalty_params = best_run_params["penalty_params"]
        exp_params = best_run_params["exp_params"]
        model_config = {
            "n_heads": model.module.n_heads,
            "value_l1_penalty_coef": penalty_params["value_l1_penalty_coef"],
            "end_penalty_coef": penalty_params["end_val"],
            "seed": exp_params["seed"],
            "n_neighbors": model.n_neighbors,
            "penalty_flavor": penalty_params["flavor"],
        }
        json.dump(model_config, f)

    with open(snakemake.output[2], "w") as f:  # noqa: F821
        f.write(str(best_recons))


def main():
    """Train the AMICI model."""
    select_gpu()
    adata = sc.read_h5ad(snakemake.input.adata_path)  # noqa: F821
    adata.obs_names_make_unique()

    dataset_config = snakemake.config["datasets"][snakemake.wildcards.dataset]  # noqa: F821
    use_cv = dataset_config.get("use_cross_validation", False)

    all_runs = _build_run_configs()

    if not use_cv:
        # ── Original behaviour ────────────────────────────────────────────────
        results_dict = _run_sweep_parallel(adata, dataset_config, all_runs, eval_indices=None, save_model=True)

        best_run_id = min(results_dict.keys(), key=lambda k: results_dict[k]["test_recons"])
        best_info = results_dict[best_run_id]
        _save_best_model(adata, best_info["model_config"], best_info["model_path"], best_info["test_recons"])

        for run_info in results_dict.values():
            run_path = run_info["model_path"]
            if run_path and os.path.exists(run_path):
                shutil.rmtree(run_path)

    else:
        # ── Cross-validation mode ─────────────────────────────────────────────
        # Step 1: 3-fold CV sweep — no models saved to disk
        n_folds = 3
        # cv_scores[run_idx] = list of val metrics across folds
        cv_scores = {i: [] for i in range(len(all_runs))}
        train_mask = adata.obs["train_test_split"] == "train"

        for fold_id in range(n_folds):
            print(f"\n=== CV fold {fold_id + 1}/{n_folds} ===")

            cv_val_indices = np.where(train_mask & (adata.obs["cv_fold"] == fold_id))[0]

            # Make a view of adata where fold cells are excluded from training
            adata_cv = adata.copy()
            adata_cv.obs.loc[adata_cv.obs["cv_fold"] == fold_id, "train_test_split"] = "test"

            fold_results = _run_sweep_parallel(
                adata_cv,
                dataset_config,
                all_runs,
                eval_indices=cv_val_indices,
                save_model=False,
                base_run_id=fold_id * len(all_runs),
            )

            for local_idx in range(len(all_runs)):
                run_id = f"run_{fold_id * len(all_runs) + local_idx}"
                cv_scores[local_idx].append(fold_results[run_id]["test_recons"])

        # Step 2: Average across folds, select best config
        avg_cv_scores = {i: float(np.mean(scores)) for i, scores in cv_scores.items()}
        best_run_idx = min(avg_cv_scores, key=avg_cv_scores.get)
        best_run_params = all_runs[best_run_idx]
        print(
            f"\nBest config (idx={best_run_idx}): {best_run_params} " f"avg_val_loss={avg_cv_scores[best_run_idx]:.6f}"
        )

        # Step 3: Train final model on ALL train cells with best config
        print("\n=== Training final model on all training data ===")
        final_results = _run_sweep_parallel(
            adata,
            dataset_config,
            [best_run_params],
            eval_indices=None,
            save_model=True,
            base_run_id=0,
        )

        final_run_id = "run_0"
        final_info = final_results[final_run_id]
        _save_best_model(adata, best_run_params, final_info["model_path"], final_info["test_recons"])

        # Clean up the final sweep directory
        if final_info["model_path"] and os.path.exists(final_info["model_path"]):
            shutil.rmtree(final_info["model_path"])


if __name__ == "__main__":
    main()
