import json
import os
import random
import shutil

import gitiii
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from gitiii_benchmark_utils import convert_adata_to_csv


def train_single_run(converted_df_path, gene_names, lr, distance_threshold, run_seed, run_dir, run_results_path):
    """Train a single GITIII run and return the best validation MSE, loading from cache if available."""
    if os.path.exists(run_results_path):
        with open(run_results_path) as f:
            cached = json.load(f)
        print(f"  Loaded cached result: val_mse={cached['val_mse']:.6f}")
        return cached["val_mse"]

    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)

    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    random.seed(run_seed)

    estimator = gitiii.estimator.GITIII_estimator(
        df_path=converted_df_path,
        genes=gene_names,
        use_log_normalize=False,
        species="human",
        use_nichenetv2=True,
        visualize_when_preprocessing=False,
        distance_threshold=distance_threshold,
        process_num_neighbors=50,
        num_neighbors=50,
        batch_size_train=128,
        lr=lr,
        epochs=50,
        node_dim=256,
        edge_dim=48,
        att_dim=8,
        batch_size_val=128,
    )
    estimator.preprocess_dataset()
    estimator.train()

    records_path = os.path.join(run_dir, "record_GRIT.csv")
    val_mse = pd.read_csv(records_path)["val_loss_interaction"].min()
    print(f"  best val_mse={val_mse:.6f}")

    return float(val_mse)


def main():
    """Train the GITIII model with a hyperparameter sweep, keeping the best run."""
    project_root = os.path.abspath(os.getcwd())

    dataset_config = snakemake.config["datasets"][snakemake.wildcards.dataset]  # noqa: F821
    labels_key = dataset_config["labels_key"]
    dataset_path = snakemake.input.adata_path  # noqa: F821
    dataset = snakemake.wildcards.dataset  # noqa: F821
    seed = snakemake.wildcards.seed  # noqa: F821

    adata = sc.read_h5ad(dataset_path)

    models_dir = os.path.abspath(f"results/{dataset}_{seed}/saved_models")
    converted_df_path = f"../../../data/{dataset_path.split('/')[-1].split('.')[0]}_converted.csv"

    convert_adata_to_csv(adata, labels_key, models_dir, converted_df_path)
    abs_converted_df_path = os.path.abspath(converted_df_path)
    os.chdir(project_root)

    gene_names = adata.var_names.tolist()

    learning_rates = [1e-3, 1e-4, 1e-5]
    distance_thresholds = [50, 80, 120]
    run_seeds = [42, 22, 38]
    all_runs = [(lr, dt, s) for lr in learning_rates for dt in distance_thresholds for s in run_seeds]

    best_val_mse = float("inf")
    best_run_id = None

    for run_id, (lr, distance_threshold, run_seed) in enumerate(all_runs):
        run_dir = os.path.join(models_dir, f"gitiii_sweep_run_{run_id}")
        run_results_path = os.path.join(models_dir, f"gitiii_sweep_run_{run_id}_results.json")

        print(f"Run {run_id + 1}/{len(all_runs)}: lr={lr}, distance_threshold={distance_threshold}, seed={run_seed}")

        val_mse = train_single_run(
            abs_converted_df_path,
            gene_names,
            lr,
            distance_threshold,
            run_seed,
            run_dir,
            run_results_path,
        )

        if not os.path.exists(run_results_path):
            with open(run_results_path, "w") as f:
                json.dump({"val_mse": val_mse}, f)

        os.chdir(project_root)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_run_id = run_id

    if best_run_id is None:
        raise RuntimeError("No runs completed successfully")

    best_lr, best_distance_threshold, best_seed_val = all_runs[best_run_id]
    best_run_dir = os.path.join(models_dir, f"gitiii_sweep_run_{best_run_id}")

    print(
        f"Best run {best_run_id}: lr={best_lr}, distance_threshold={best_distance_threshold}, "
        f"seed={best_seed_val}, val_mse={best_val_mse:.6f}"
    )

    for fname in os.listdir(best_run_dir):
        src = os.path.join(best_run_dir, fname)
        dst = os.path.join(models_dir, fname)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        elif os.path.isfile(src) and not fname.endswith("_results.json"):
            shutil.copy2(src, dst)

    with open(os.path.join(models_dir, "gitiii_best_params.json"), "w") as f:
        json.dump(
            {
                "learning_rate": best_lr,
                "distance_threshold": best_distance_threshold,
                "seed": int(best_seed_val),
                "val_mse": best_val_mse,
            },
            f,
        )


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    main()
