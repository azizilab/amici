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
    """Train a single NCEM run, evaluate on eval_indices, and return the reconstruction loss.

    Loads from cache when run_results_path already exists.
    Pass run_results_path=None to disable caching (used during CV sweeps).
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
    cache_results=True,
):
    """Run the hyperparameter sweep and return (best_run_id, best_eval_recons, scores_per_run, best_history).

    When cache_results=False the model weights and result JSON are not written to disk.
    """
    best_eval_recons = float("inf")
    best_run_id = None
    best_model_history = None
    scores = {}

    for run_id, (lr, l1_coef) in enumerate(all_runs):
        run_model_path = os.path.join(model_dir, f"{run_prefix}_run_{run_id}_checkpoint")
        run_model_args_path = os.path.join(model_dir, f"{run_prefix}_run_{run_id}.pickle")
        run_results_path = os.path.join(model_dir, f"{run_prefix}_run_{run_id}_results.json") if cache_results else None

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

        # Always remove transient model files when not caching (CV mode)
        if not cache_results:
            for suffix in [".data-00000-of-00001", ".index"]:
                path = run_model_path + suffix
                if os.path.exists(path):
                    os.remove(path)
            if os.path.exists(run_model_args_path):
                os.remove(run_model_args_path)

    return best_run_id, best_eval_recons, scores, best_model_history


def main(input_path, labels_key, dataset, seed, niche_size, use_cv=False):
    """Train the NCEM model, optionally using 3-fold spatial CV for hyperparameter selection."""
    adata = sc.read_h5ad(input_path)
    if "spatial" not in adata.uns:
        adata.uns["spatial"] = adata.obsm["spatial"].copy()

    model_dir = f"results/{dataset}_{seed}/saved_models"
    model_path = os.path.join(model_dir, f"ncem_{niche_size}_checkpoint_{dataset}_{seed}")
    model_args_path = os.path.join(model_dir, f"ncem_{niche_size}_{dataset}_{seed}.pickle")

    if os.path.exists(model_path + ".index") and os.path.exists(model_args_path):
        print(f"Model path {model_path} and model args path {model_args_path} already exist, skipping")
        return

    train_indices = np.where(adata.obs["train_test_split"] == "train")[0]
    test_indices = np.where(adata.obs["train_test_split"] == "test")[0]

    learning_rates = [0.05, 0.005]
    l1_coefs = [1e-4, 1e-3]
    all_runs = list(itertools.product(learning_rates, l1_coefs))

    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    if not use_cv:
        # ── Original behaviour ────────────────────────────────────────────────
        best_run_id, best_test_recons, _, best_model_history = _run_sweep(
            adata,
            labels_key,
            all_runs,
            niche_size,
            model_dir,
            train_indices=train_indices,
            eval_indices=test_indices,
            run_prefix=f"ncem_{niche_size}_sweep",
            cache_results=True,
        )

        if best_run_id is None:
            raise RuntimeError("No runs completed successfully")

        best_lr, best_l1_coef = all_runs[best_run_id]
        best_run_model_path = os.path.join(model_dir, f"ncem_{niche_size}_sweep_run_{best_run_id}_checkpoint")
        best_run_model_args_path = os.path.join(model_dir, f"ncem_{niche_size}_sweep_run_{best_run_id}.pickle")

        print(f"Best run {best_run_id}: lr={best_lr}, l1_coef={best_l1_coef}, test_recons={best_test_recons:.4f}")

        for suffix in [".data-00000-of-00001", ".index"]:
            src = best_run_model_path + suffix
            if os.path.exists(src):
                shutil.copy(src, model_path + suffix)
        shutil.copy(best_run_model_args_path, model_args_path)

        with open(f"results/{dataset}_{seed}/ncem_{niche_size}_model_params.json", "w") as f:
            json.dump({"learning_rate": best_lr, "l1_coef": best_l1_coef, "test_recons": best_test_recons}, f)

        # Clean up sweep run files
        for run_id_cleanup in range(len(all_runs)):
            for suffix in [".data-00000-of-00001", ".index"]:
                path = os.path.join(model_dir, f"ncem_{niche_size}_sweep_run_{run_id_cleanup}_checkpoint") + suffix
                if os.path.exists(path):
                    os.remove(path)
            pickle_path = os.path.join(model_dir, f"ncem_{niche_size}_sweep_run_{run_id_cleanup}.pickle")
            if os.path.exists(pickle_path):
                os.remove(pickle_path)

        plot_ncem_loss_curves(best_model_history, niche_size, f"results/{dataset}_{seed}/figures")

    else:
        # ── Cross-validation mode ─────────────────────────────────────────────
        # Step 1: 3-fold CV sweep — model files deleted after each fold
        n_folds = 3
        cv_scores = {i: [] for i in range(len(all_runs))}
        train_mask = adata.obs["train_test_split"] == "train"

        for fold_id in range(n_folds):
            print(f"\n=== CV fold {fold_id + 1}/{n_folds} ===")

            cv_val_indices = np.where(train_mask & (adata.obs["cv_fold"] == fold_id))[0]
            cv_train_indices = np.where(train_mask & (adata.obs["cv_fold"] != fold_id))[0]

            _, _, fold_scores, _ = _run_sweep(
                adata,
                labels_key,
                all_runs,
                niche_size,
                model_dir,
                train_indices=cv_train_indices,
                eval_indices=cv_val_indices,
                run_prefix=f"ncem_{niche_size}_cv_fold{fold_id}",
                cache_results=False,
            )

            for run_id, score in fold_scores.items():
                cv_scores[run_id].append(score)

        # Step 2: Select best config by average CV score
        avg_cv_scores = {i: float(np.mean(scores)) for i, scores in cv_scores.items()}
        best_run_idx = min(avg_cv_scores, key=avg_cv_scores.get)
        best_lr, best_l1_coef = all_runs[best_run_idx]
        print(
            f"\nBest config (idx={best_run_idx}): lr={best_lr}, l1_coef={best_l1_coef} "
            f"avg_val_loss={avg_cv_scores[best_run_idx]:.6f}"
        )

        # Step 3: Train final model on all train cells with best config
        print("\n=== Training final model on all training data ===")
        exp_params, model_params, train_params = get_model_parameters(niche_size)
        model_params["learning_rate"] = best_lr
        model_params["l1_coef"] = best_l1_coef

        final_recons, best_model_history = train_single_run(
            adata,
            labels_key,
            exp_params,
            model_params,
            train_params,
            model_path,
            model_args_path,
            run_results_path=None,
            train_indices=train_indices,
            eval_indices=test_indices,
        )
        print(f"Final model test_recons={final_recons:.4f}")

        with open(f"results/{dataset}_{seed}/ncem_{niche_size}_model_params.json", "w") as f:
            json.dump({"learning_rate": best_lr, "l1_coef": best_l1_coef, "test_recons": final_recons}, f)

        plot_ncem_loss_curves(best_model_history, niche_size, f"results/{dataset}_{seed}/figures")


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
    use_cv = len(args) > 6 and args[6].lower() == "true"

    try:
        niche_size = int(niche_size)
        main(input_path, labels_key, dataset, seed, niche_size, use_cv=use_cv)
        with open(output_path, "w") as f:
            f.write("Training done")
            f.flush()
            os.fsync(f.fileno())
        print("Finished training NCEM")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
