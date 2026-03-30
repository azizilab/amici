import json
import os
import random
import shutil

import gitiii
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from gitiii_benchmark_utils import convert_adata_to_csv
from gpu_utils import select_gpu


<<<<<<< HEAD
def _mark_test_cells_in_processed_csv(run_dir, test_cell_coords):
    """Set flag=False for test cells in GITIII's processed CSV, matched by (centerx, centery)."""
=======
def _build_gitiii_interaction_matrix(best_run_dir, num_neighbors, node_dim, edge_dim, att_dim):
    """Load the best trained GITIII model and compute a cell-type interaction matrix.

    Uses the mean absolute output of the graph transformer across genes as a proxy
    for how strongly each neighbor cell type influences each center cell type.
    y_pred[1][0] has shape [batch, num_genes, num_neighbors-1, 1]; neighbor j in the
    attention corresponds to cell_types[:, j+1] (index 0 is the center cell).
    """
    from gitiii.dataloader import GITIII_dataset
    from gitiii.model import GITIII
    from torch.utils.data import DataLoader as TorchDataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(best_run_dir, "data", "processed") + "/"
    ligands_info = torch.load(os.path.join(best_run_dir, "data", "ligands.pth"), weights_only=False)
    genes = torch.load(os.path.join(best_run_dir, "data", "genes.pth"), weights_only=False)

    dataset = GITIII_dataset(processed_dir=data_dir, num_neighbors=num_neighbors)
    cell_types_list = list(dataset.cell_types_dict.keys())

    model = GITIII(
        genes,
        ligands_info,
        node_dim=node_dim,
        edge_dim=edge_dim,
        num_heads=2,
        n_layers=1,
        node_dim_small=16,
        att_dim=att_dim,
        use_cell_type_embedding=True,
    ).to(device)
    model.load_state_dict(
        torch.load(os.path.join(best_run_dir, "GRIT_best.pth"), map_location=device, weights_only=False)
    )
    model.eval()

    n_types = len(cell_types_list)
    scores = np.zeros((n_types, n_types))
    counts = np.zeros_like(scores)

    loader = TorchDataLoader(dataset, batch_size=256, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            y_pred = model(batch)

            # [batch, num_genes, num_neighbors-1, att_dim] -> mean over genes and att_dim -> [batch, num_neighbors-1]
            strength = y_pred[1][0].abs().mean(dim=(1, -1)).cpu().numpy()
            cell_types = batch["cell_types"].cpu()
            center_types = cell_types[:, 0]  # [batch]
            neighbor_types = cell_types[:, 1:]  # [batch, num_neighbors-1]

            for i in range(strength.shape[0]):
                c = center_types[i].item()
                for j in range(strength.shape[1]):
                    n = neighbor_types[i, j].item()
                    if 0 <= c < n_types and 0 <= n < n_types:
                        scores[n, c] += strength[i, j]  # sender=neighbor, receiver=center
                        counts[n, c] += 1

    mean_scores = np.divide(scores, counts, where=counts > 0)
    display_names = [name.replace("ct_", "") for name in cell_types_list]
    return pd.DataFrame(mean_scores, index=display_names, columns=display_names)


def _mark_excluded_cells_in_processed_csv(run_dir, excluded_cell_coords):
    """Set flag=False for excluded cells in GITIII's processed CSV, matched by (centerx, centery).

    excluded_cell_coords is a set of (x, y) tuples for cells that should NOT be used in training
    (covers both held-out test cells and, during CV, the current validation fold).
    """
>>>>>>> 256431f (add cross-validation option for realistic semisynthetic dataset)
    processed_csv_path = os.path.join(run_dir, "data", "processed", "slide1.csv")
    df = pd.read_csv(processed_csv_path)
    if "flag" not in df.columns:
        df["flag"] = True
    is_excluded = df.apply(lambda r: (r["centerx"], r["centery"]) in excluded_cell_coords, axis=1)
    df.loc[is_excluded, "flag"] = False
    df.to_csv(processed_csv_path, index=False)


def train_single_run(
    converted_df_path,
    gene_names,
    lr,
    distance_threshold,
    run_dir,
    run_results_path,
    excluded_cell_coords,
):
    """Train a single GITIII run and return the best validation MSE.

    excluded_cell_coords: set of (x, y) tuples for cells excluded from training
    (test cells, and optionally a CV validation fold).
    Pass run_results_path=None to disable caching.
    """
    if run_results_path is not None and os.path.exists(run_results_path):
        with open(run_results_path) as f:
            cached = json.load(f)
        print(f"  Loaded cached result: val_mse={cached['val_mse']:.6f}")
        return cached["val_mse"]

    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)

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
    _mark_excluded_cells_in_processed_csv(run_dir, excluded_cell_coords)
    estimator.train()

    records_path = os.path.join(run_dir, "record_GRIT.csv")
    val_mse = pd.read_csv(records_path)["val_loss_interaction"].min()
    print(f"  best val_mse={val_mse:.6f}")

    return float(val_mse)


def _run_sweep(
    project_root,
    abs_converted_df_path,
    gene_names,
    all_runs,
    models_dir,
    excluded_cell_coords,
    run_prefix,
    cache_results=True,
):
    """Run the hyperparameter sweep.

    Returns (best_run_id, best_val_mse, best_run_dir, all_scores) where
    all_scores is a dict mapping run_id -> val_mse.

    When cache_results=False, result JSON files are not written and run dirs are deleted
    after each run (used during CV sweeps).
    """
    best_val_mse = float("inf")
    best_run_id = None
    best_run_dir = None
    all_scores = {}

    for run_id, (lr, distance_threshold) in enumerate(all_runs):
        run_dir = os.path.join(models_dir, f"{run_prefix}_run_{run_id}")
        run_results_path = (
            os.path.join(models_dir, f"{run_prefix}_run_{run_id}_results.json") if cache_results else None
        )

        print(f"Run {run_id + 1}/{len(all_runs)}: lr={lr}, distance_threshold={distance_threshold}")

        val_mse = train_single_run(
            abs_converted_df_path,
            gene_names,
            lr,
            distance_threshold,
            run_dir,
            run_results_path,
            excluded_cell_coords,
        )

        if cache_results and run_results_path is not None and not os.path.exists(run_results_path):
            with open(run_results_path, "w") as f:
                json.dump({"val_mse": val_mse}, f)

        os.chdir(project_root)
        all_scores[run_id] = val_mse

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_run_id = run_id
            best_run_dir = run_dir

        # In CV mode, delete run directory immediately to free disk space
        if not cache_results and os.path.exists(run_dir):
            shutil.rmtree(run_dir)

    return best_run_id, best_val_mse, best_run_dir, all_scores


def _write_figures(source_run_dir, figures_dir):
    """Write loss curve and interaction figures from a completed run directory."""
    records_path = os.path.join(source_run_dir, "record_GRIT.csv")
    if os.path.exists(records_path):
        records_df = pd.read_csv(records_path)
        plt.figure(figsize=(10, 5))
        plt.plot(records_df["train_loss_interaction"].values, label="Train Loss")
        plt.plot(records_df["val_loss_interaction"].values, label="Validation Loss", linestyle="--")
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(os.path.join(figures_dir, "gitiii_loss_curve.png"))
        plt.savefig(os.path.join(figures_dir, "gitiii_loss_curve.svg"))
        plt.close()

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
                "seed": 42,
                "val_mse": best_val_mse,
            },
            f,
        )

    # Clean up all sweep run directories (best model already copied to models_dir)
    for run_id_cleanup in range(len(all_runs)):
        run_dir_cleanup = os.path.join(models_dir, f"gitiii_sweep_run_{run_id_cleanup}")
        if os.path.exists(run_dir_cleanup):
            shutil.rmtree(run_dir_cleanup)
    figures_dir = f"results/{dataset}_{seed}/figures"
    os.makedirs(figures_dir, exist_ok=True)

    records_df = pd.read_csv(os.path.join(models_dir, "record_GRIT.csv"))
    plt.figure(figsize=(10, 5))
    plt.plot(records_df["train_loss_interaction"].values, label="Train Loss")
    plt.plot(records_df["val_loss_interaction"].values, label="Validation Loss", linestyle="--")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, "gitiii_loss_curve.png"))
    plt.savefig(os.path.join(figures_dir, "gitiii_loss_curve.svg"))
    plt.close()


def main():
    """Train the GITIII model with a hyperparameter sweep, keeping the best run."""
    select_gpu()
    project_root = os.path.abspath(os.getcwd())

    dataset_config = snakemake.config["datasets"][snakemake.wildcards.dataset]  # noqa: F821
    labels_key = dataset_config["labels_key"]
    dataset_path = snakemake.input.adata_path  # noqa: F821
    dataset = snakemake.wildcards.dataset  # noqa: F821
    seed = snakemake.wildcards.seed  # noqa: F821
    use_cv = dataset_config.get("use_cross_validation", False)

    adata = sc.read_h5ad(dataset_path)

    models_dir = os.path.abspath(f"results/{dataset}_{seed}/saved_models")
    converted_df_path = f"../../../data/{dataset_path.split('/')[-1].split('.')[0]}_converted.csv"

    convert_adata_to_csv(adata, labels_key, models_dir, converted_df_path)
    abs_converted_df_path = os.path.abspath(converted_df_path)
    os.chdir(project_root)

    test_mask = adata.obs["train_test_split"] == "test"
    # Support both DataFrame-style and numpy-array obsm
    spatial = adata.obsm["spatial"]
    if hasattr(spatial, "iloc"):
        x_all = spatial["X"].values
        y_all = spatial["Y"].values
    else:
        x_all = spatial[:, 0]
        y_all = spatial[:, 1]

    test_cell_coords = set(zip(x_all[test_mask], y_all[test_mask], strict=False))

    gene_names = adata.var_names.tolist()

    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    learning_rates = [1e-3, 1e-4]
    distance_thresholds = [50, 80, 120]
    all_runs = [(lr, dt) for lr in learning_rates for dt in distance_thresholds]

    figures_dir = f"results/{dataset}_{seed}/figures"
    os.makedirs(figures_dir, exist_ok=True)

    if not use_cv:
        # ── Original behaviour ────────────────────────────────────────────────
        best_run_id, best_val_mse, best_run_dir, _ = _run_sweep(
            project_root,
            abs_converted_df_path,
            gene_names,
            all_runs,
            models_dir,
            excluded_cell_coords=test_cell_coords,
            run_prefix="gitiii_sweep",
            cache_results=True,
        )

        if best_run_id is None:
            raise RuntimeError("No runs completed successfully")

        best_lr, best_distance_threshold = all_runs[best_run_id]
        print(
            f"Best run {best_run_id}: lr={best_lr}, distance_threshold={best_distance_threshold}, "
            f"val_mse={best_val_mse:.6f}"
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
                    "seed": 42,
                    "val_mse": best_val_mse,
                },
                f,
            )

        # Clean up all sweep run directories
        for run_id_cleanup in range(len(all_runs)):
            run_dir_cleanup = os.path.join(models_dir, f"gitiii_sweep_run_{run_id_cleanup}")
            if os.path.exists(run_dir_cleanup):
                shutil.rmtree(run_dir_cleanup)

        _write_figures(best_run_dir, figures_dir)

    else:
        # ── Cross-validation mode ─────────────────────────────────────────────
        # Step 1: 3-fold CV sweep — run dirs deleted after each run
        n_folds = 3
        train_mask = adata.obs["train_test_split"] == "train"
        # cv_scores[run_idx] = list of val metrics across folds
        cv_scores = {i: [] for i in range(len(all_runs))}

        for fold_id in range(n_folds):
            print(f"\n=== CV fold {fold_id + 1}/{n_folds} ===")

            val_mask = train_mask & (adata.obs["cv_fold"] == fold_id)
            # Exclude both test cells AND current validation fold from training
            excluded_coords = test_cell_coords | set(zip(x_all[val_mask], y_all[val_mask], strict=False))

            _, _, _, fold_scores = _run_sweep(
                project_root,
                abs_converted_df_path,
                gene_names,
                all_runs,
                models_dir,
                excluded_cell_coords=excluded_coords,
                run_prefix=f"gitiii_cv_fold{fold_id}",
                cache_results=False,
            )

            for run_id, val_mse in fold_scores.items():
                cv_scores[run_id].append(val_mse)

        # Step 2: Select best config by average CV score
        avg_cv_scores = {i: float(np.mean(scores)) for i, scores in cv_scores.items()}
        best_run_idx = min(avg_cv_scores, key=avg_cv_scores.get)
        best_lr, best_distance_threshold = all_runs[best_run_idx]
        print(
            f"\nBest config (idx={best_run_idx}): lr={best_lr}, distance_threshold={best_distance_threshold} "
            f"avg_val_loss={avg_cv_scores[best_run_idx]:.6f}"
        )

        # Step 3: Train final model on all train cells (only test cells excluded)
        print("\n=== Training final model on all training data ===")
        final_run_dir = os.path.join(models_dir, "gitiii_final_run")
        final_val_mse = train_single_run(
            abs_converted_df_path,
            gene_names,
            best_lr,
            best_distance_threshold,
            final_run_dir,
            run_results_path=None,
            excluded_cell_coords=test_cell_coords,
        )
        os.chdir(project_root)
        print(f"Final model val_mse={final_val_mse:.6f}")

        for fname in os.listdir(final_run_dir):
            src = os.path.join(final_run_dir, fname)
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
                    "seed": 42,
                    "val_mse": final_val_mse,
                },
                f,
            )

        _write_figures(final_run_dir, figures_dir)

        if os.path.exists(final_run_dir):
            shutil.rmtree(final_run_dir)


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    main()
