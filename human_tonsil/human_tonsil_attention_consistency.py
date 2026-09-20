# %% Import libraries
import json
import os

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import spearmanr

from amici import AMICI
from amici.interpretation._attention_module import AMICIAttentionModule

# %% Load data and run dirs
DATA_DATE = "2026-07-25"
LABELS_KEY = "celltype_manual_fine"
COORD_KEY = "spatial"
DATA_DIR = "data"
WANDB_DIR = "wandb"
SAVED_MODELS_DIR = "saved_models"
FIGURE_DIR = "figures/attention_consistency"
RESULTS_DIR = "results/attention_consistency"
CACHE_DIR = os.path.join(RESULTS_DIR, "signature_cache")

# Attention is accumulated over the full dataset in chunks so that the per-head
# attention frame for 492k cells x n_heads x 50 neighbors never has to fit in memory.
CHUNK_SIZE = 20000
BATCH_SIZE = 128

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# %% Collect sweep runs
def collect_sweep_runs():
    """Join local wandb summaries to saved models, skipping runs that crashed before logging.

    Returns
    -------
        pd.DataFrame: One row per usable run with columns:
            - run_id: wandb run id
            - test_reconstruction_loss: held-out reconstruction loss logged by the run
            - model_path: path to the run's saved model directory
            - n_neighbors / n_heads / seed: run config values
    """
    records = []
    for run_dir in sorted(os.listdir(WANDB_DIR)):
        if not run_dir.startswith("run-"):
            continue
        summary_path = os.path.join(WANDB_DIR, run_dir, "files", "wandb-summary.json")
        config_path = os.path.join(WANDB_DIR, run_dir, "files", "config.yaml")
        if not os.path.exists(summary_path) or not os.path.exists(config_path):
            continue

        try:
            summary = json.load(open(summary_path))
            config = yaml.safe_load(open(config_path))
        except (json.JSONDecodeError, yaml.YAMLError):
            continue

        test_loss = summary.get("test_reconstruction_loss")
        model_path = summary.get("model_path")
        if test_loss is None or not model_path or not os.path.isdir(model_path):
            continue

        records.append(
            {
                "run_id": run_dir.split("-")[-1],
                "test_reconstruction_loss": float(test_loss),
                "model_path": model_path,
                "n_neighbors": int(config.get("n_neighbors", {}).get("value", 50)),
                "n_heads": int(config.get("n_heads", {}).get("value", 16)),
                "seed": config.get("seed", {}).get("value"),
                "end_attention_penalty": config.get("end_attention_penalty", {}).get("value"),
                "value_l1_penalty_coef": config.get("value_l1_penalty_coef", {}).get("value"),
            }
        )
    return pd.DataFrame(records).sort_values("test_reconstruction_loss").reset_index(drop=True)


# %% Attention signature
def compute_attention_signature(model, adata, label_codes, n_cell_types):
    """Mean attention from each receiver cell type to each sender cell type.

    Attention is averaged over heads first, which makes the signature invariant to the
    arbitrary ordering of heads across independently trained models, then averaged over
    every (receiver cell, neighbor slot) pair grouped by the two cell type labels.

    Args:
        model: Trained AMICI model.
        adata: AnnData the model has been set up on.
        label_codes: Integer cell type code per cell, aligned to adata.obs_names.
        n_cell_types: Number of cell type categories.

    Returns
    -------
        tuple[np.ndarray, np.ndarray]: The n_cell_types x n_cell_types matrix of mean
        attention (receiver x sender) and the matching count matrix.
    """
    nn_sender_codes = label_codes[np.asarray(adata.obsm["_nn_idx"])]  # n_cells x n_neighbors
    n_neighbors = nn_sender_codes.shape[1]

    attention_sums = np.zeros(n_cell_types * n_cell_types, dtype=np.float64)
    attention_counts = np.zeros(n_cell_types * n_cell_types, dtype=np.float64)

    for chunk_start in range(0, adata.n_obs, CHUNK_SIZE):
        chunk_indices = np.arange(chunk_start, min(chunk_start + CHUNK_SIZE, adata.n_obs))
        attention_patterns = AMICIAttentionModule.compute(
            model,
            adata=adata,
            indices=chunk_indices,
            batch_size=BATCH_SIZE,
            prog_bar=False,
        )
        attention_df = attention_patterns._attention_patterns_df
        neighbor_cols = [f"neighbor_{i}" for i in range(n_neighbors)]

        # Rows are stacked head-major (all cells for head 0, then head 1, ...) by
        # AMICIAttentionModule.compute, so reshaping recovers the head dimension.
        attention = attention_df[neighbor_cols].to_numpy(dtype=np.float32)
        attention = attention.reshape(-1, len(chunk_indices), n_neighbors).mean(axis=0)

        sender_codes = nn_sender_codes[chunk_indices]
        receiver_codes = np.repeat(label_codes[chunk_indices][:, None], n_neighbors, axis=1)

        flat_pair_idx = (receiver_codes * n_cell_types + sender_codes).ravel()
        attention_sums += np.bincount(flat_pair_idx, weights=attention.ravel(), minlength=n_cell_types * n_cell_types)
        attention_counts += np.bincount(flat_pair_idx, minlength=n_cell_types * n_cell_types)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_attention = attention_sums / attention_counts
    return (
        mean_attention.reshape(n_cell_types, n_cell_types),
        attention_counts.reshape(n_cell_types, n_cell_types),
    )


def get_run_signature(run, adata, label_codes, n_cell_types):
    """Compute a run's attention signature, caching it so long sweeps can be resumed."""
    cache_path = os.path.join(CACHE_DIR, f"{run['run_id']}.npz")
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return cached["mean_attention"], cached["counts"]

    AMICI.setup_anndata(
        adata,
        labels_key=LABELS_KEY,
        coord_obsm_key=COORD_KEY,
        n_neighbors=int(run["n_neighbors"]),
    )
    model = AMICI.load(run["model_path"], adata=adata)
    mean_attention, counts = compute_attention_signature(model, adata, label_codes, n_cell_types)
    np.savez(cache_path, mean_attention=mean_attention, counts=counts)
    return mean_attention, counts


# %% Cross-run consistency
def compute_pairwise_spearman(signatures):
    """Spearman correlation between every pair of run signatures.

    Args:
        signatures: n_runs x n_cell_types^2 array of flattened attention matrices.

    Returns
    -------
        np.ndarray: n_runs x n_runs matrix of Spearman correlations.
    """
    n_runs = signatures.shape[0]
    correlations = np.eye(n_runs)
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            rho = spearmanr(signatures[i], signatures[j]).statistic
            correlations[i, j] = correlations[j, i] = rho
    return correlations


def plot_consistency(runs_df, correlations):
    """Plot loss vs. cross-run attention consistency and the pairwise correlation matrix."""
    n_runs = len(runs_df)
    off_diagonal = ~np.eye(n_runs, dtype=bool)
    mean_consistency = np.array([correlations[i][off_diagonal[i]].mean() for i in range(n_runs)])
    runs_df = runs_df.assign(mean_spearman=mean_consistency)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    scatter = axes[0].scatter(
        runs_df["test_reconstruction_loss"],
        runs_df["mean_spearman"],
        c=runs_df["mean_spearman"],
        cmap="viridis",
        s=70,
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0].set_xlabel("Test reconstruction loss")
    axes[0].set_ylabel("Mean Spearman $\\rho$ of attention vs. other runs")
    axes[0].set_title(f"Attention consistency vs. held-out loss ({n_runs} runs)")
    axes[0].grid(alpha=0.3)
    fig.colorbar(scatter, ax=axes[0], label="Mean Spearman $\\rho$")

    sns.heatmap(
        pd.DataFrame(correlations, index=runs_df["run_id"], columns=runs_df["run_id"]),
        cmap="viridis",
        vmin=np.nanmin(correlations[off_diagonal]),
        vmax=1.0,
        square=True,
        cbar_kws={"label": "Spearman $\\rho$"},
        ax=axes[1],
    )
    axes[1].set_title("Pairwise attention agreement (runs ordered by loss)")
    axes[1].set_xlabel("Run")
    axes[1].set_ylabel("Run")

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIGURE_DIR, f"human_tonsil_attention_consistency.{ext}"), dpi=300)
    plt.close(fig)
    return runs_df


# %% Main
def main():
    """Compute and plot cross-run attention consistency for the human tonsil sweep."""
    adata = ad.read_h5ad(f"{DATA_DIR}/human_tonsil_filtered_{DATA_DATE}.h5ad")
    cell_types = pd.Categorical(adata.obs[LABELS_KEY])
    label_codes = cell_types.codes.astype(np.int64)
    n_cell_types = len(cell_types.categories)

    runs_df = collect_sweep_runs()
    print(f"Found {len(runs_df)} usable runs (crashed runs without summaries are skipped)")
    print(runs_df[["run_id", "test_reconstruction_loss", "seed", "n_heads"]].to_string(index=False))

    signatures = []
    kept_runs = []
    shared_counts = None
    for _, run in runs_df.iterrows():
        print(f"Computing attention signature for run {run['run_id']}")
        mean_attention, counts = get_run_signature(run, adata, label_codes, n_cell_types)
        shared_counts = counts if shared_counts is None else np.minimum(shared_counts, counts)
        signatures.append(mean_attention)
        kept_runs.append(run)

    runs_df = pd.DataFrame(kept_runs).reset_index(drop=True)
    # Only compare cell type pairs that are actually observed in every run's neighbor graph.
    valid_pairs = (shared_counts > 0).ravel()
    signatures = np.vstack([signature.ravel()[valid_pairs] for signature in signatures])
    print(f"Comparing {signatures.shape[1]} of {n_cell_types**2} receiver-sender cell type pairs")

    correlations = compute_pairwise_spearman(signatures)
    runs_df = plot_consistency(runs_df, correlations)

    runs_df.to_csv(os.path.join(RESULTS_DIR, "human_tonsil_attention_consistency_runs.csv"), index=False)
    pd.DataFrame(correlations, index=runs_df["run_id"], columns=runs_df["run_id"]).to_csv(
        os.path.join(RESULTS_DIR, "human_tonsil_attention_consistency_correlations.csv")
    )

    off_diagonal = correlations[~np.eye(len(runs_df), dtype=bool)]
    print(f"Median pairwise Spearman rho: {np.median(off_diagonal):.4f}")
    print(f"Min pairwise Spearman rho:    {np.min(off_diagonal):.4f}")
    print(f"Figures written to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
