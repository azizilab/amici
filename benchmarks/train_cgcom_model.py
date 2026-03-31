import json
import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from benchmark_utils import plot_interaction_graph, plot_interaction_matrix
from cgcom.scripts import train_model
from cgcom.utils import get_exp_params, get_model_params
from gpu_utils import select_gpu


def build_cgcom_interaction_matrix(comm_csv_path, adata, labels_key):
    """Aggregate CGCom attention scores by cell type pair into a sender x receiver matrix."""
    comm_df = pd.read_csv(comm_csv_path)
    cell_type_map = adata.obs[labels_key].to_dict()

    comm_df["sender_type"] = comm_df["neighbor_cell"].map(cell_type_map)
    comm_df["receiver_type"] = comm_df["center_cell"].map(cell_type_map)
    comm_df = comm_df.dropna(subset=["sender_type", "receiver_type"])

    matrix = comm_df.groupby(["sender_type", "receiver_type"])["attention_score"].mean().unstack(fill_value=0)
    cell_types = sorted(set(matrix.index) | set(matrix.columns))
    return matrix.reindex(index=cell_types, columns=cell_types, fill_value=0)


def train_single_run(
    dataset_path, labels_key, lr, neighbor_threshold_ratio, run_model_path, run_results_path, train_obs_names
):
    """Train a single CGCom run and return the best validation loss.

    Loads from cache when run_results_path already exists.
    Pass run_results_path=None to disable caching (used during CV sweeps).
    """
    if run_results_path is not None and os.path.exists(run_results_path):
        with open(run_results_path) as f:
            cached = json.load(f)
        print(f"  Loaded cached result: val_loss={cached['val_loss']:.6f}")
        return cached["val_loss"]

    os.makedirs(os.path.dirname(run_model_path), exist_ok=True)

    exp_params = get_exp_params(lr=lr, num_epochs=50, neighbor_threshold_ratio=neighbor_threshold_ratio)
    model_params = get_model_params(
        fc_hidden_channels_2=500,
        fc_hidden_channels_3=512,
        fc_hidden_channels_4=64,
        num_classes=3,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ligand_channel=500,
        receptor_channel=500,
        TF_channel=500,
        mask_indexes=None,
        disable_lr_masking=True,
    )

    _model, _model_path, train_losses, val_losses = train_model(
        exp_params,
        model_params,
        dataset_path=dataset_path,
        model_path=run_model_path,
        labels_key=labels_key,
        disable_lr_masking=True,
        train_obs_names=train_obs_names,
    )

    best_val_loss = float(min(val_losses))
    print(f"  best val_loss={best_val_loss:.6f}")

    if run_results_path is not None:
        with open(run_results_path, "w") as f:
            json.dump({"val_loss": best_val_loss, "train_losses": train_losses, "val_losses": val_losses}, f)

    return best_val_loss


def _run_sweep(
    dataset_path,
    labels_key,
    all_runs,
    models_dir,
    train_obs_names,
    run_prefix,
    cache_results=True,
):
    """Run the hyperparameter sweep.

    Returns (best_run_id, best_val_loss, all_scores) where all_scores maps run_id -> val_loss.
    When cache_results=False, result JSON files are not written and run dirs are deleted
    after each run (used during CV sweeps).
    """
    best_val_loss = float("inf")
    best_run_id = None
    all_scores = {}

    for run_id, (lr, neighbor_threshold_ratio) in enumerate(all_runs):
        run_dir = os.path.join(models_dir, f"{run_prefix}_run_{run_id}")
        run_model_path = os.path.join(run_dir, "cgcom_model.pt")
        run_results_path = (
            os.path.join(models_dir, f"{run_prefix}_run_{run_id}_results.json") if cache_results else None
        )

        print(f"Run {run_id + 1}/{len(all_runs)}: lr={lr}, neighbor_threshold_ratio={neighbor_threshold_ratio}")

        val_loss = train_single_run(
            dataset_path,
            labels_key,
            lr,
            neighbor_threshold_ratio,
            run_model_path,
            run_results_path,
            train_obs_names,
        )

        all_scores[run_id] = val_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_run_id = run_id

        # In CV mode, delete run directory immediately to free disk space
        if not cache_results and os.path.exists(run_dir):
            shutil.rmtree(run_dir)

    return best_run_id, best_val_loss, all_scores


def main():
    """Train the CGCom model with a hyperparameter sweep, optionally using 3-fold spatial CV."""
    select_gpu()
    dataset_config = snakemake.config["datasets"][snakemake.wildcards.dataset]  # noqa: F821
    labels_key = dataset_config["labels_key"]
    dataset_path = snakemake.input.adata_path  # noqa: F821
    dataset = snakemake.wildcards.dataset  # noqa: F821
    seed = snakemake.wildcards.seed  # noqa: F821
    final_model_path = os.path.abspath(snakemake.output[0])  # noqa: F821
    use_cv = dataset_config.get("use_cross_validation", False)

    models_dir = os.path.dirname(final_model_path)
    os.makedirs(models_dir, exist_ok=True)

    adata = sc.read_h5ad(dataset_path)
    train_obs_names = adata.obs_names[adata.obs["train_test_split"] == "train"].tolist()

    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    learning_rates = [1e-3, 1e-4]
    neighbor_threshold_ratios = [0.005, 0.01, 0.02]
    all_runs = [(lr, ntr) for lr in learning_rates for ntr in neighbor_threshold_ratios]

    # CGCom requires integer-encoded cell labels. Build a temporary h5ad with a
    # numeric labels column so that cgcom.utils.get_cell_label_dict can parse them.
    cgcom_labels_key = "cgcom_int_labels"
    adata.obs[cgcom_labels_key] = adata.obs[labels_key].astype("category").cat.codes.astype(str)
    cgcom_dataset_path = dataset_path.replace(".h5ad", "_cgcom.h5ad")
    adata.write_h5ad(cgcom_dataset_path)

    if not use_cv:
        # ── Original behaviour ────────────────────────────────────────────────
        best_run_id, best_val_loss, _ = _run_sweep(
            cgcom_dataset_path,
            cgcom_labels_key,
            all_runs,
            models_dir,
            train_obs_names=train_obs_names,
            run_prefix="cgcom_sweep",
            cache_results=True,
        )

        if best_run_id is None:
            raise RuntimeError("No runs completed successfully")

        best_lr, best_ntr = all_runs[best_run_id]
        best_run_dir = os.path.join(models_dir, f"cgcom_sweep_run_{best_run_id}")

        print(
            f"Best run {best_run_id}: lr={best_lr}, neighbor_threshold_ratio={best_ntr}, val_loss={best_val_loss:.6f}"
        )

        for fname in os.listdir(best_run_dir):
            src = os.path.join(best_run_dir, fname)
            dst = os.path.join(models_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

        best_run_model = os.path.join(models_dir, "cgcom_model.pt")
        if best_run_model != final_model_path:
            shutil.copy2(best_run_model, final_model_path)

        best_params_path = os.path.join(models_dir, "cgcom_best_params.json")
        with open(best_params_path, "w") as f:
            json.dump(
                {"learning_rate": best_lr, "neighbor_threshold_ratio": best_ntr, "seed": 42, "val_loss": best_val_loss},
                f,
            )

        with open(os.path.join(models_dir, f"cgcom_sweep_run_{best_run_id}_results.json")) as f:
            best_results = json.load(f)

        # Clean up all sweep run directories
        for run_id_cleanup in range(len(all_runs)):
            run_dir = os.path.join(models_dir, f"cgcom_sweep_run_{run_id_cleanup}")
            if os.path.exists(run_dir):
                shutil.rmtree(run_dir)

        _write_figures(best_results, adata, labels_key, models_dir, dataset, seed)
        if os.path.exists(cgcom_dataset_path):
            os.remove(cgcom_dataset_path)

    else:
        # ── Cross-validation mode ─────────────────────────────────────────────
        # Step 1: 3-fold CV sweep — run dirs deleted after each run
        n_folds = 3
        train_mask = adata.obs["train_test_split"] == "train"
        # cv_scores[run_idx] = list of val metrics across folds
        cv_scores = {i: [] for i in range(len(all_runs))}

        for fold_id in range(n_folds):
            print(f"\n=== CV fold {fold_id + 1}/{n_folds} ===")

            cv_train_obs_names = adata.obs_names[train_mask & (adata.obs["cv_fold"] != fold_id)].tolist()

            _, _, fold_scores = _run_sweep(
                cgcom_dataset_path,
                cgcom_labels_key,
                all_runs,
                models_dir,
                train_obs_names=cv_train_obs_names,
                run_prefix=f"cgcom_cv_fold{fold_id}",
                cache_results=False,
            )

            for run_id, val_loss in fold_scores.items():
                cv_scores[run_id].append(val_loss)

        # Step 2: Select best config by average CV score
        avg_cv_scores = {i: float(np.mean(scores)) for i, scores in cv_scores.items()}
        best_run_idx = min(avg_cv_scores, key=avg_cv_scores.get)
        best_lr, best_ntr = all_runs[best_run_idx]
        print(
            f"\nBest config (idx={best_run_idx}): lr={best_lr}, neighbor_threshold_ratio={best_ntr} "
            f"avg_val_loss={avg_cv_scores[best_run_idx]:.6f}"
        )

        # Step 3: Train final model on all train cells with best config
        print("\n=== Training final model on all training data ===")
        final_run_dir = os.path.join(models_dir, "cgcom_final_run")
        final_model_tmp = os.path.join(final_run_dir, "cgcom_model.pt")
        final_results_path = os.path.join(models_dir, "cgcom_final_run_results.json")
        final_val_loss = train_single_run(
            cgcom_dataset_path,
            cgcom_labels_key,
            best_lr,
            best_ntr,
            final_model_tmp,
            run_results_path=final_results_path,
            train_obs_names=train_obs_names,
        )
        print(f"Final model val_loss={final_val_loss:.6f}")

        # Copy final model to expected output path
        shutil.copy2(final_model_tmp, final_model_path)

        with open(os.path.join(models_dir, "cgcom_best_params.json"), "w") as f:
            json.dump(
                {
                    "learning_rate": best_lr,
                    "neighbor_threshold_ratio": best_ntr,
                    "seed": 42,
                    "val_loss": final_val_loss,
                },
                f,
            )

        with open(final_results_path) as f:
            best_results = json.load(f)

        _write_figures(best_results, adata, labels_key, models_dir, dataset, seed)

        if os.path.exists(final_run_dir):
            shutil.rmtree(final_run_dir)
        if os.path.exists(final_results_path):
            os.remove(final_results_path)
        if os.path.exists(cgcom_dataset_path):
            os.remove(cgcom_dataset_path)


def _write_figures(best_results, adata, labels_key, models_dir, dataset, seed):
    """Write loss curve and interaction matrix figures."""
    figures_dir = os.path.join(f"results/{dataset}_{seed}/figures")
    os.makedirs(figures_dir, exist_ok=True)

    if best_results.get("train_losses") and best_results.get("val_losses"):
        plt.figure(figsize=(10, 5))
        plt.plot(best_results["train_losses"], label="Train Loss")
        plt.plot(best_results["val_losses"], label="Validation Loss", linestyle="--")
        plt.legend()
        plt.savefig(os.path.join(figures_dir, "cgcom_loss_curve.png"))
        plt.savefig(os.path.join(figures_dir, "cgcom_loss_curve.svg"))
        plt.close()

    comm_csv_path = os.path.join(models_dir, "communication_scores_default.csv")
    if os.path.exists(comm_csv_path):
        matrix = build_cgcom_interaction_matrix(comm_csv_path, adata, labels_key)
        plot_interaction_matrix(
            matrix,
            figures_dir,
            title="CGCom cell-type interaction matrix",
            filename="cgcom_interaction_matrix",
            colorbar_label="Mean attention score",
        )
        plot_interaction_graph(
            matrix,
            figures_dir,
            title="CGCom cell-type interaction graph",
            filename="cgcom_interaction_graph",
        )
    else:
        print(f"Warning: communication scores not found at {comm_csv_path}, skipping interaction figures")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    main()
