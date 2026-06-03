import itertools
import json
import os
import random
import shutil
import sys

import numpy as np
import scanpy as sc
import tensorflow as tf
from gpu_utils import select_gpu
from ncem_benchmark_utils import get_model_parameters, plot_ncem_loss_curves, train_ncem

RESULTS_PATH = "results"


def _results_path():
    return RESULTS_PATH.rstrip("/")


def train_single_run(
    adata,
    labels_key,
    exp_params,
    model_params,
    train_params,
    run_model_path,
    run_model_args_path,
    run_results_path,
    train_indices,
    eval_indices,
):
    """
    Train a single NCEM run, evaluate on eval_indices, and return the reconstruction loss.

    Loads from cache when run_results_path already exists.

    Args:
        adata: AnnData object
        labels_key: str, key in adata.obs for cell type labels
        exp_params: dict, experiment parameters
        model_params: dict, model architecture parameters
        train_params: dict, training parameters
        run_model_path: str, path to save/load model weights
        run_model_args_path: str, path to save/load model arguments
        run_results_path: str, path to cache the result JSON
        train_indices: array-like, indices of training cells
        eval_indices: array-like, indices of cells to evaluate on

    Returns
    -------
        tuple of (eval_recons, model_history) where eval_recons is the reconstruction loss
        and model_history is a dict of training metrics
    """
    if run_results_path is not None and os.path.exists(run_results_path):
        with open(run_results_path) as f:
            cached = json.load(f)
        print(f"  Loaded cached result: eval_recons={cached['test_recons']:.4f}")
        return cached["test_recons"], cached["model_history"]

    model_history, trainer = train_ncem(
        adata,
        labels_key,
        exp_params,
        model_params,
        train_params,
        run_model_path,
        run_model_args_path,
        train_indices=train_indices,
        test_indices=eval_indices,
    )

    eval_result = trainer.estimator.evaluate_any(
        img_keys=trainer.estimator.img_keys_test,
        node_idx=trainer.estimator.nodes_idx_test,
    )
    eval_recons = float(eval_result["gaussian_reconstruction_loss"])
    print(f"  Eval reconstruction loss: {eval_recons:.4f}")

    if run_results_path is not None:
        with open(run_results_path, "w") as f:
            json.dump(
                {
                    "test_recons": eval_recons,
                    "model_history": {k: [float(v) for v in vs] for k, vs in model_history.items()},
                },
                f,
            )

    return eval_recons, model_history


def _run_sweep(
    adata,
    labels_key,
    all_runs,
    niche_size,
    model_dir,
    train_indices,
    eval_indices,
    run_prefix,
):
    """
    Run the hyperparameter sweep over learning rate and l1 coefficient combinations.

    Args:
        adata: AnnData object
        labels_key: str, key in adata.obs for cell type labels
        all_runs: list of (lr, l1_coef) tuples
        niche_size: int, number of neighbors to use
        model_dir: str, directory to save run results
        train_indices: array-like, indices of training cells
        eval_indices: array-like, indices of cells to evaluate on
        run_prefix: str, prefix for run file names

    Returns
    -------
        tuple of (best_run_id, best_eval_recons, scores, best_model_history)
    """
    best_eval_recons = float("inf")
    best_run_id = None
    best_model_history = None
    scores = {}

    for run_id, (lr, l1_coef) in enumerate(all_runs):
        run_model_path = os.path.join(model_dir, f"{run_prefix}_run_{run_id}_checkpoint")
        run_model_args_path = os.path.join(model_dir, f"{run_prefix}_run_{run_id}.pickle")
        run_results_path = os.path.join(model_dir, f"{run_prefix}_run_{run_id}_results.json")

        exp_params, model_params, train_params = get_model_parameters(niche_size)
        model_params["learning_rate"] = lr
        model_params["l1_coef"] = l1_coef

        print(f"  Run {run_id}/{len(all_runs)}: lr={lr}, l1_coef={l1_coef}")

        eval_recons, model_history = train_single_run(
            adata,
            labels_key,
            exp_params,
            model_params,
            train_params,
            run_model_path,
            run_model_args_path,
            run_results_path,
            train_indices,
            eval_indices,
        )

        scores[run_id] = eval_recons

        if eval_recons < best_eval_recons:
            best_eval_recons = eval_recons
            best_run_id = run_id
            best_model_history = model_history

    return best_run_id, best_eval_recons, scores, best_model_history


def _save_best_run(
    dataset,
    seed,
    niche_size,
    model_dir,
    model_path,
    model_args_path,
    all_runs,
    best_run_id,
    best_recons,
    best_model_history,
):
    best_lr, best_l1_coef = all_runs[best_run_id]
    best_run_model_path = os.path.join(model_dir, f"ncem_{niche_size}_final_run_{best_run_id}_checkpoint")
    best_run_model_args_path = os.path.join(model_dir, f"ncem_{niche_size}_final_run_{best_run_id}.pickle")

    print(f"Best final run {best_run_id}: lr={best_lr}, l1_coef={best_l1_coef}, test_recons={best_recons:.4f}")

    for suffix in [".data-00000-of-00001", ".index"]:
        src = best_run_model_path + suffix
        if os.path.exists(src):
            shutil.copy(src, model_path + suffix)
    shutil.copy(best_run_model_args_path, model_args_path)

    with open(f"{_results_path()}/{dataset}_{seed}/ncem_{niche_size}_model_params.json", "w") as f:
        json.dump({"learning_rate": best_lr, "l1_coef": best_l1_coef, "test_recons": best_recons}, f)

    plot_ncem_loss_curves(best_model_history, niche_size, f"{_results_path()}/{dataset}_{seed}/figures")


def main(
    input_path,
    labels_key,
    dataset,
    seed,
    niche_size,
    sweep_baselines=False,
    results_path="results",
    use_cross_validation=False,
):
    """
    Train the NCEM model with a hyperparameter sweep, keeping the best run.

    Args:
        input_path: str, path to the input AnnData h5ad file
        labels_key: str, key in adata.obs for cell type labels
        dataset: str, dataset name
        seed: str, random seed identifier
        niche_size: int, number of neighbors to use
        sweep_baselines: bool, whether to sweep over multiple hyperparameter configurations
    """
    global RESULTS_PATH
    RESULTS_PATH = results_path

    adata = sc.read_h5ad(input_path)
    if "spatial" not in adata.uns:
        adata.uns["spatial"] = adata.obsm["spatial"].copy()

    model_dir = f"{_results_path()}/{dataset}_{seed}/saved_models"
    model_path = os.path.join(model_dir, f"ncem_{niche_size}_checkpoint_{dataset}_{seed}")
    model_args_path = os.path.join(model_dir, f"ncem_{niche_size}_{dataset}_{seed}.pickle")

    if os.path.exists(model_path + ".index") and os.path.exists(model_args_path):
        print(f"Model path {model_path} and model args path {model_args_path} already exist, skipping")
        return

    train_indices = np.where(adata.obs["train_test_split"] == "train")[0]
    test_indices = np.where(adata.obs["train_test_split"] == "test")[0]

    if sweep_baselines:
        learning_rates = [0.05, 0.005]
        l1_coefs = [1e-4, 1e-3]
        all_runs = list(itertools.product(learning_rates, l1_coefs))
    else:
        all_runs = [(0.05, 0.0)]

    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    use_cv = use_cross_validation

    if use_cv:
        if "cv_fold" not in adata.obs:
            raise ValueError("use_cross_validation is true, but adata.obs['cv_fold'] is missing")

        fold_values = sorted(int(fold) for fold in adata.obs.loc[adata.obs["cv_fold"] >= 0, "cv_fold"].unique())
        cv_scores = {run_id: [] for run_id in range(len(all_runs))}

        for fold_id in fold_values:
            print(f"\n=== NCEM CV fold {fold_id + 1}/{len(fold_values)} ===")
            fold_val_indices = np.where((adata.obs["train_test_split"] == "train") & (adata.obs["cv_fold"] == fold_id))[
                0
            ]
            fold_train_indices = np.where(
                (adata.obs["train_test_split"] == "train") & (adata.obs["cv_fold"] != fold_id)
            )[0]

            _, _, fold_scores, _ = _run_sweep(
                adata,
                labels_key,
                all_runs,
                niche_size,
                model_dir,
                train_indices=fold_train_indices,
                eval_indices=fold_val_indices,
                run_prefix=f"ncem_{niche_size}_cv_fold_{fold_id}",
            )

            for run_id, score in fold_scores.items():
                cv_scores[run_id].append(score)

        avg_cv_scores = {run_id: float(np.mean(scores)) for run_id, scores in cv_scores.items()}
        best_cv_run_id = min(avg_cv_scores, key=avg_cv_scores.get)
        print(
            f"\nBest NCEM CV config {best_cv_run_id}: lr={all_runs[best_cv_run_id][0]}, "
            f"l1_coef={all_runs[best_cv_run_id][1]}, avg_val_recons={avg_cv_scores[best_cv_run_id]:.4f}"
        )

        best_run_id, best_test_recons, _, best_model_history = _run_sweep(
            adata,
            labels_key,
            [all_runs[best_cv_run_id]],
            niche_size,
            model_dir,
            train_indices=train_indices,
            eval_indices=test_indices,
            run_prefix=f"ncem_{niche_size}_final",
        )
        all_runs_to_save = [all_runs[best_cv_run_id]]
    else:
        best_run_id, best_test_recons, _, best_model_history = _run_sweep(
            adata,
            labels_key,
            all_runs,
            niche_size,
            model_dir,
            train_indices=train_indices,
            eval_indices=test_indices,
            run_prefix=f"ncem_{niche_size}_final",
        )
        all_runs_to_save = all_runs

    if best_run_id is None:
        raise RuntimeError("No runs completed successfully")

    _save_best_run(
        dataset,
        seed,
        niche_size,
        model_dir,
        model_path,
        model_args_path,
        all_runs_to_save,
        best_run_id,
        best_test_recons,
        best_model_history,
    )

    # Clean up sweep run files
    for run_id_cleanup in range(len(all_runs)):
        for suffix in [".data-00000-of-00001", ".index"]:
            for prefix in [f"ncem_{niche_size}_sweep", f"ncem_{niche_size}_final"]:
                path = os.path.join(model_dir, f"{prefix}_run_{run_id_cleanup}_checkpoint") + suffix
                if os.path.exists(path):
                    os.remove(path)
        for prefix in [f"ncem_{niche_size}_sweep", f"ncem_{niche_size}_final"]:
            pickle_path = os.path.join(model_dir, f"{prefix}_run_{run_id_cleanup}.pickle")
            if os.path.exists(pickle_path):
                os.remove(pickle_path)


if __name__ == "__main__":
    print("Started training NCEM")
    select_gpu()

    args = sys.argv[1:]
    input_path = args[0]
    output_path = args[1]
    labels_key = args[2]
    dataset = args[3]
    seed = args[4]
    niche_size = args[5]
    sweep_baselines = len(args) > 6 and args[6].lower() == "true"
    results_path = args[7] if len(args) > 7 else "results"
    use_cross_validation = len(args) > 8 and args[8].lower() == "true"

    try:
        niche_size = int(niche_size)
        main(
            input_path,
            labels_key,
            dataset,
            seed,
            niche_size,
            sweep_baselines=sweep_baselines,
            results_path=results_path,
            use_cross_validation=use_cross_validation,
        )
        with open(output_path, "w") as f:
            f.write("Training done")
            f.flush()
            os.fsync(f.fileno())
        print("Finished training NCEM")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
