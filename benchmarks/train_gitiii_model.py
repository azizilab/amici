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
    """
    Load the best trained GITIII model and compute a cell-type interaction matrix.

    Uses the mean absolute output of the graph transformer across genes as a proxy
    for how strongly each neighbor cell type influences each center cell type.
    y_pred[1][0] has shape [batch, num_genes, num_neighbors-1, 1]; neighbor j in the
    attention corresponds to cell_types[:, j+1] (index 0 is the center cell).

    Args:
        best_run_dir: str, path to the best run directory containing model weights and data
        num_neighbors: int, number of neighbors used during training
        node_dim: int, node embedding dimension
        edge_dim: int, edge embedding dimension
        att_dim: int, attention output dimension

    Returns
    -------
        pd.DataFrame, cell-type interaction matrix with sender types as rows and receiver types as columns
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
    """
    Set flag=False for excluded cells in GITIII's processed CSV, matched by (centerx, centery).

    Args:
        run_dir: str, path to the run directory containing the processed CSV
        excluded_cell_coords: set of (x, y) tuples for cells that should NOT be used in training
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
    """
    Train a single GITIII run and return the best validation MSE.

    Args:
        converted_df_path: str, path to the converted CSV file
        gene_names: list, list of gene names
        lr: float, learning rate
        distance_threshold: float, distance threshold for neighbor selection
        run_dir: str, path to the run directory
        run_results_path: str, path to cache the result JSON
        excluded_cell_coords: set of (x, y) tuples for cells excluded from training (test cells)

    Returns
    -------
        float, best validation MSE
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
):
    """
    Run the hyperparameter sweep over learning rate and distance threshold combinations.

    Args:
        project_root: str, absolute path to the project root directory
        abs_converted_df_path: str, absolute path to the converted CSV file
        gene_names: list, list of gene names
        all_runs: list of (lr, distance_threshold) tuples
        models_dir: str, directory to save run results
        excluded_cell_coords: set of (x, y) tuples for cells excluded from training
        run_prefix: str, prefix for run directory and result file names

    Returns
    -------
        tuple of (best_run_id, best_val_mse, best_run_dir, all_scores) where
        all_scores maps run_id -> val_mse
    """
    best_val_mse = float("inf")
    best_run_id = None
    best_run_dir = None
    all_scores = {}

    for run_id, (lr, distance_threshold) in enumerate(all_runs):
        run_dir = os.path.join(models_dir, f"{run_prefix}_run_{run_id}")
        run_results_path = os.path.join(models_dir, f"{run_prefix}_run_{run_id}_results.json")

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

        if not os.path.exists(run_results_path):
            with open(run_results_path, "w") as f:
                json.dump({"val_mse": val_mse}, f)

        os.chdir(project_root)
        all_scores[run_id] = val_mse

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_run_id = run_id
            best_run_dir = run_dir

    return best_run_id, best_val_mse, best_run_dir, all_scores


def _write_figures(source_run_dir, figures_dir):
    """
    Write loss curve and interaction figures from a completed run directory.

    Args:
        source_run_dir: str, path to the completed run directory
        figures_dir: str, directory to save figure files
    """
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

    sweep_baselines = dataset_config.get("sweep_baselines", False)
    if sweep_baselines:
        learning_rates = [1e-3, 1e-4]
        distance_thresholds = [50, 80, 120]
        all_runs = [(lr, dt) for lr in learning_rates for dt in distance_thresholds]
    else:
        all_runs = [(1e-4, 80)]

    figures_dir = f"results/{dataset}_{seed}/figures"
    os.makedirs(figures_dir, exist_ok=True)

    best_run_id, best_val_mse, best_run_dir, _ = _run_sweep(
        project_root,
        abs_converted_df_path,
        gene_names,
        all_runs,
        models_dir,
        excluded_cell_coords=test_cell_coords,
        run_prefix="gitiii_sweep",
    )

    if best_run_id is None:
        raise RuntimeError("No runs completed successfully")

    best_lr, best_distance_threshold = all_runs[best_run_id]
    print(
        f"Best run {best_run_id}: lr={best_lr}, distance_threshold={best_distance_threshold}, "
        f"val_mse={best_val_mse:.6f}"
    )

    if os.path.exists(best_run_dir):
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


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    main()
