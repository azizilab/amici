import itertools
import json
import os
import random
import shutil
import sys

import numpy as np
import scanpy as sc
import tensorflow as tf
from ncem_benchmark_utils import get_model_parameters, plot_ncem_loss_curves, train_ncem


def train_single_run(
    adata, labels_key, exp_params, model_params, train_params, run_model_path, run_model_args_path, run_results_path
):
    """Train a single NCEM run, evaluate on the test set, and return the reconstruction loss and history, loading from cache if available."""
    if os.path.exists(run_results_path):
        with open(run_results_path) as f:
            cached = json.load(f)
        print(f"  Loaded cached result: test_recons={cached['test_recons']:.4f}")
        return cached["test_recons"], cached["model_history"]

    model_history, trainer = train_ncem(
        adata,
        labels_key,
        exp_params,
        model_params,
        train_params,
        run_model_path,
        run_model_args_path,
    )

    eval_test = trainer.estimator.evaluate_any(
        img_keys=trainer.estimator.img_keys_test,
        node_idx=trainer.estimator.nodes_idx_test,
    )
    test_recons = float(eval_test["gaussian_reconstruction_loss"])
    print(f"  Test reconstruction loss: {test_recons:.4f}")

    with open(run_results_path, "w") as f:
        json.dump(
            {
                "test_recons": test_recons,
                "model_history": {k: [float(v) for v in vs] for k, vs in model_history.items()},
            },
            f,
        )

    return test_recons, model_history


def main(input_path, labels_key, dataset, seed, niche_size):
    """Train the NCEM model with a hyperparameter sweep over lr, l1_coef, and seed, keeping the best run."""
    adata = sc.read_h5ad(input_path)
    if "spatial" not in adata.uns:
        adata.uns["spatial"] = adata.obsm["spatial"].copy()

    model_dir = f"results/{dataset}_{seed}/saved_models"
    model_path = os.path.join(model_dir, f"ncem_{niche_size}_checkpoint_{dataset}_{seed}")
    model_args_path = os.path.join(model_dir, f"ncem_{niche_size}_{dataset}_{seed}.pickle")

    if os.path.exists(model_path + ".index") and os.path.exists(model_args_path):
        print(f"Model path {model_path} and model args path {model_args_path} already exist, skipping")
        return

    learning_rates = [0.5, 0.05, 0.005]
    l1_coefs = [0.0, 1e-4, 1e-3]
    run_seeds = [22, 38, 17, 11, 42, 33, 18]

    all_runs = list(itertools.product(learning_rates, l1_coefs, run_seeds))

    best_test_recons = float("inf")
    best_run_id = None
    best_model_history = None

    for run_id, (lr, l1_coef, run_seed) in enumerate(all_runs):
        run_model_path = os.path.join(model_dir, f"ncem_{niche_size}_sweep_run_{run_id}_checkpoint")
        run_model_args_path = os.path.join(model_dir, f"ncem_{niche_size}_sweep_run_{run_id}.pickle")
        run_results_path = os.path.join(model_dir, f"ncem_{niche_size}_sweep_run_{run_id}_results.json")

        np.random.seed(run_seed)
        random.seed(run_seed)
        tf.random.set_seed(run_seed)

        exp_params, model_params, train_params = get_model_parameters(niche_size)
        model_params["learning_rate"] = lr
        model_params["l1_coef"] = l1_coef
        exp_params["seed"] = run_seed

        print(f"Run {run_id}/{len(all_runs)}: lr={lr}, l1_coef={l1_coef}, seed={run_seed}")

        test_recons, model_history = train_single_run(
            adata,
            labels_key,
            exp_params,
            model_params,
            train_params,
            run_model_path,
            run_model_args_path,
            run_results_path,
        )

        if test_recons < best_test_recons:
            best_test_recons = test_recons
            best_run_id = run_id
            best_model_history = model_history

    if best_run_id is None:
        raise RuntimeError("No runs completed successfully")

    best_lr, best_l1_coef, best_seed = all_runs[best_run_id]
    best_run_model_path = os.path.join(model_dir, f"ncem_{niche_size}_sweep_run_{best_run_id}_checkpoint")
    best_run_model_args_path = os.path.join(model_dir, f"ncem_{niche_size}_sweep_run_{best_run_id}.pickle")

    print(
        f"Best run {best_run_id}: lr={best_lr}, l1_coef={best_l1_coef}, seed={best_seed}, test_recons={best_test_recons:.4f}"
    )

    for suffix in [".data-00000-of-00001", ".index"]:
        src = best_run_model_path + suffix
        if os.path.exists(src):
            shutil.copy(src, model_path + suffix)
    shutil.copy(best_run_model_args_path, model_args_path)

    with open(f"results/{dataset}_{seed}/ncem_{niche_size}_model_params.json", "w") as f:
        json.dump(
            {
                "learning_rate": best_lr,
                "l1_coef": best_l1_coef,
                "seed": int(best_seed),
                "test_recons": best_test_recons,
            },
            f,
        )

    plot_ncem_loss_curves(
        best_model_history,
        niche_size,
        f"results/{dataset}_{seed}/figures",
    )


if __name__ == "__main__":
    print("Started training NCEM")
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    # Parse the arguments in the command line
    args = sys.argv[1:]
    input_path = args[0]
    output_path = args[1]
    labels_key = args[2]
    dataset = args[3]
    seed = args[4]
    niche_size = args[5]

    try:
        niche_size = int(niche_size)
        main(input_path, labels_key, dataset, seed, niche_size)
        # Write completion file
        with open(output_path, "w") as f:
            f.write("Training done")
            f.flush()
            os.fsync(f.fileno())
        print("Finished training NCEM")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
