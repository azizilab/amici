# %% Imports
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.size"] = 11

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import seaborn as sns
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance

from amici import AMICI


# %% Configuration
SEED = 18
LABELS_KEY = "celltype_train_grouped"
STROMAL_LABEL = "Stromal"

N_NEIGHBORS = 50
BATCH_SIZE = 2048
ATTENTION_SAMPLE_SIZE = 60000
TOP_ATTENTION_FRACTION = 0.05
DISTANCE_MATCH_BIN_WIDTH = 10.0

DATA_PATH = Path("data/xenium_proseg_filtered_2025-04-08.h5ad")
MODEL_PATH = Path("saved_models/xenium_18_sweep_pp3avuwy_rttj3z4d_params_2025-04-08")
NOSTROMA_DATA_PATH = Path("data/xenium_sample1/xenium_sample1_filtered_2025-05-01.h5ad")
NOSTROMA_MODEL_PATH = Path(
    "saved_models/xenium_sample1_proseg_sweep_2025-05-01_model_2025-05-02/"
    "xenium_18_sweep_g3mucw4s_te7pkv3z_params_2025-05-02"
)

OUT_DIR = Path("figures/stromal_bleeding_analysis")
CACHE_DIR = OUT_DIR / "cache"


# %% Reused helpers
def save_figure(fig, stem):
    """Save a matplotlib figure as PNG and SVG."""
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_model_and_data(data_path, model_path):
    """Load Xenium data and attach a saved AMICI model."""
    adata = sc.read_h5ad(data_path)
    adata.obs_names_make_unique()
    AMICI.setup_anndata(
        adata,
        labels_key=LABELS_KEY,
        coord_obsm_key="spatial",
        n_neighbors=N_NEIGHBORS,
    )
    model = AMICI.load(str(model_path), adata=adata)
    return adata, model


def get_interaction_matrix(model, adata, cache_stem):
    """Load or compute the AMICI sender-receiver interaction matrix."""
    cache_path = CACHE_DIR / f"{cache_stem}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    scores = model.get_neighbor_ablation_scores(adata=adata, compute_z_value=True)
    matrix = scores._get_interaction_weight_matrix()
    matrix.to_parquet(cache_path)
    matrix.to_csv(CACHE_DIR / f"{cache_stem}.csv")
    return matrix


def sample_attention_cells(adata):
    """Sample cells for attention extraction while keeping rare cell types represented."""
    rng = np.random.default_rng(SEED)
    if ATTENTION_SAMPLE_SIZE >= adata.n_obs:
        return np.arange(adata.n_obs)

    labels = adata.obs[LABELS_KEY].astype(str).to_numpy()
    keep = []
    per_type = max(250, ATTENTION_SAMPLE_SIZE // max(1, len(np.unique(labels))))
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        keep.extend(rng.choice(idx, size=min(len(idx), per_type), replace=False))

    keep = np.unique(np.asarray(keep, dtype=int))
    if len(keep) < ATTENTION_SAMPLE_SIZE:
        remaining = np.setdiff1d(np.arange(adata.n_obs), keep, assume_unique=False)
        extra = rng.choice(
            remaining,
            size=min(ATTENTION_SAMPLE_SIZE - len(keep), len(remaining)),
            replace=False,
        )
        keep = np.concatenate([keep, extra])
    return np.sort(keep[:ATTENTION_SAMPLE_SIZE])


def get_attention_edges(model, adata):
    """Load or compute long-form receiver-neighbor attention edges."""
    cache_path = CACHE_DIR / f"attention_edges_n{ATTENTION_SAMPLE_SIZE}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    attention = model.get_attention_patterns(
        adata=adata,
        indices=sample_attention_cells(adata),
        batch_size=BATCH_SIZE,
        prog_bar=True,
    )
    attention_df = attention._attention_patterns_df.copy()
    nn_idx_df = attention._nn_idxs_df.copy()
    nn_dist_df = attention._nn_dists_df.copy()
    neighbor_cols = [c for c in attention_df.columns if c.startswith("neighbor_")]

    # Collapse heads by max so each receiver-neighbor edge has one high-attention score.
    max_attention = attention_df.groupby("cell_idx")[neighbor_cols].max()
    labels = adata.obs[LABELS_KEY].astype(str)

    rows = []
    for cell_id, scores in max_attention.iterrows():
        receiver = labels.loc[cell_id]
        neighbor_ids = nn_idx_df.loc[cell_id]
        distances = nn_dist_df.loc[cell_id]
        for col in neighbor_cols:
            sender_cell = neighbor_ids[col]
            rows.append(
                {
                    "receiver_cell": cell_id,
                    "sender_cell": sender_cell,
                    "receiver": receiver,
                    "sender": labels.loc[sender_cell],
                    "attention": float(scores[col]),
                    "distance": float(distances[col]),
                }
            )

    edge_df = pd.DataFrame(rows)
    edge_df.to_parquet(cache_path)
    edge_df.to_csv(CACHE_DIR / f"attention_edges_n{ATTENTION_SAMPLE_SIZE}.csv", index=False)
    return edge_df


def make_distance_matched_edges(edge_df):
    """Sample non-stromal edges with the same distance-bin counts as stromal edges."""
    stromal = edge_df[edge_df["sender"] == STROMAL_LABEL].copy()
    other = edge_df[edge_df["sender"] != STROMAL_LABEL].copy()
    stromal["dist_bin"] = (stromal["distance"] / DISTANCE_MATCH_BIN_WIDTH).round().astype(int)
    other["dist_bin"] = (other["distance"] / DISTANCE_MATCH_BIN_WIDTH).round().astype(int)

    rng = np.random.default_rng(SEED)
    matched = []
    for dist_bin, group in stromal.groupby("dist_bin"):
        candidates = other[other["dist_bin"] == dist_bin]
        if candidates.empty:
            continue
        n_edges = min(len(group), len(candidates))
        choice = rng.choice(len(candidates), size=n_edges, replace=False)
        matched.append(candidates.iloc[choice])

    if not matched:
        return other.iloc[:0].copy()
    return pd.concat(matched, ignore_index=True)


def top_fraction_mean(values):
    """Mean attention among the highest-attention edges."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan
    n_top = max(1, int(np.ceil(values.size * TOP_ATTENTION_FRACTION)))
    return float(np.mean(np.sort(values)[-n_top:]))


# %% Set up output folders
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
pl.seed_everything(SEED)


# %% Load stromal and stromal-excluded AMICI models
adata, model = load_model_and_data(DATA_PATH, MODEL_PATH)
nostroma_adata, nostroma_model = load_model_and_data(NOSTROMA_DATA_PATH, NOSTROMA_MODEL_PATH)


# %% Load or compute sender-receiver interaction matrices
matrix = get_interaction_matrix(model, adata, "interaction_weight_matrix")
nostroma_matrix = get_interaction_matrix(
    nostroma_model,
    nostroma_adata,
    "interaction_weight_matrix_no_stroma_final_model",
)


# %% Align non-stromal sender-receiver scores across the two models
shared_cell_types = [
    ct
    for ct in matrix.index
    if ct in nostroma_matrix.index and ct in matrix.columns and ct in nostroma_matrix.columns
]
shared_cell_types = sorted([ct for ct in shared_cell_types if ct != STROMAL_LABEL])

with_stroma = matrix.loc[shared_cell_types, shared_cell_types]
without_stroma = nostroma_matrix.loc[shared_cell_types, shared_cell_types]
delta = without_stroma - with_stroma

with_stroma.to_csv(CACHE_DIR / "with_stroma_nonstromal_interaction_matrix.csv")
without_stroma.to_csv(CACHE_DIR / "without_stroma_interaction_matrix.csv")
delta.to_csv(CACHE_DIR / "without_minus_with_stroma_interaction_matrix.csv")

edge_scores = pd.DataFrame(
    {
        "sender": np.repeat(shared_cell_types, len(shared_cell_types)),
        "receiver": np.tile(shared_cell_types, len(shared_cell_types)),
        "with_stroma": with_stroma.to_numpy().ravel(),
        "without_stroma": without_stroma.to_numpy().ravel(),
        "delta_without_minus_with": delta.to_numpy().ravel(),
    }
)
edge_scores = edge_scores.query("sender != receiver").copy()
edge_scores["abs_delta"] = edge_scores["delta_without_minus_with"].abs()
edge_scores.to_csv(CACHE_DIR / "with_without_stroma_edge_scores.csv", index=False)


# %% Plot interaction heatmaps before and after stromal exclusion
vmax = np.nanquantile(
    np.concatenate([with_stroma.to_numpy().ravel(), without_stroma.to_numpy().ravel()]),
    0.98,
)
diff_abs = np.nanmax(np.abs(delta.to_numpy()))

fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
sns.heatmap(
    with_stroma,
    ax=axes[0],
    cmap="Reds",
    vmin=0,
    vmax=vmax,
    square=True,
    cbar_kws={"label": "Interaction score"},
)
axes[0].set_title("Model Trained With Stromal Cells")

sns.heatmap(
    without_stroma,
    ax=axes[1],
    cmap="Reds",
    vmin=0,
    vmax=vmax,
    square=True,
    cbar_kws={"label": "Interaction score"},
)
axes[1].set_title("Final Model After Stromal Exclusion")

sns.heatmap(
    delta,
    ax=axes[2],
    cmap="vlag",
    center=0,
    vmin=-diff_abs,
    vmax=diff_abs,
    square=True,
    cbar_kws={"label": "Without - with stroma"},
)
axes[2].set_title("Change After Stromal Exclusion")

for ax in axes:
    ax.set_xlabel("Receiver")
    ax.set_ylabel("Sender")
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)
fig.suptitle("Effect of Stromal Exclusion on Non-Stromal Interaction Scores", y=1.03)
save_figure(fig, "with_without_stroma_interaction_heatmaps")


# %% Plot matched non-stromal interaction scores across the two models
rho, pval = spearmanr(edge_scores["with_stroma"], edge_scores["without_stroma"])
top_k = min(20, len(edge_scores))
top_with = set(edge_scores.nlargest(top_k, "with_stroma").index)
top_without = set(edge_scores.nlargest(top_k, "without_stroma").index)

pd.DataFrame(
    [
        {
            "spearman_rho": rho,
            "spearman_pval": pval,
            "top20_edge_jaccard": len(top_with & top_without) / max(1, len(top_with | top_without)),
            "mean_absolute_delta": float(edge_scores["abs_delta"].mean()),
            "median_absolute_delta": float(edge_scores["abs_delta"].median()),
        }
    ]
).to_csv(CACHE_DIR / "with_without_stroma_interaction_metrics.csv", index=False)

max_score = max(edge_scores["with_stroma"].max(), edge_scores["without_stroma"].max())
fig, ax = plt.subplots(figsize=(6.5, 6))
scatter = ax.scatter(
    edge_scores["with_stroma"],
    edge_scores["without_stroma"],
    c=edge_scores["abs_delta"],
    cmap="viridis",
    s=35,
    alpha=0.75,
    linewidths=0,
)
ax.plot([0, max_score], [0, max_score], color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Interaction score with stromal cells")
ax.set_ylabel("Interaction score after stromal exclusion")
ax.set_title(f"Non-Stromal Interaction Score Stability (Spearman rho={rho:.2f})")
fig.colorbar(scatter, ax=ax, label="Absolute change")
save_figure(fig, "with_without_stroma_interaction_scatter")


# %% Load or compute stromal-model attention edges
edge_df = get_attention_edges(model, adata)


# %% Compare high-attention stromal edges to a distance-matched non-stromal null
stromal_edges = edge_df[edge_df["sender"] == STROMAL_LABEL].copy()
matched_edges = make_distance_matched_edges(edge_df)

metrics = []
if not stromal_edges.empty and not matched_edges.empty:
    stromal_values = stromal_edges["attention"].to_numpy()
    matched_values = matched_edges["attention"].to_numpy()
    metrics.append(
        {
            "receiver": "all",
            "n_stromal_edges": len(stromal_values),
            "n_matched_edges": len(matched_values),
            "top_fraction": TOP_ATTENTION_FRACTION,
            "stromal_top_fraction_mean_attention": top_fraction_mean(stromal_values),
            "matched_top_fraction_mean_attention": top_fraction_mean(matched_values),
            "top_fraction_mean_delta_stromal_minus_matched": (
                top_fraction_mean(stromal_values) - top_fraction_mean(matched_values)
            ),
            "wasserstein": wasserstein_distance(stromal_values, matched_values),
            "ks_statistic": ks_2samp(stromal_values, matched_values).statistic,
        }
    )

    receiver_rows = []
    for receiver, receiver_stromal in stromal_edges.groupby("receiver"):
        receiver_matched = matched_edges[matched_edges["receiver"] == receiver]
        if len(receiver_stromal) < 50 or len(receiver_matched) < 50:
            continue
        receiver_rows.append(
            {
                "receiver": receiver,
                "n_stromal_edges": len(receiver_stromal),
                "n_matched_edges": len(receiver_matched),
                "top_fraction": TOP_ATTENTION_FRACTION,
                "stromal_top_fraction_mean_attention": top_fraction_mean(receiver_stromal["attention"]),
                "matched_top_fraction_mean_attention": top_fraction_mean(receiver_matched["attention"]),
                "top_fraction_mean_delta_stromal_minus_matched": (
                    top_fraction_mean(receiver_stromal["attention"])
                    - top_fraction_mean(receiver_matched["attention"])
                ),
                "wasserstein": wasserstein_distance(receiver_stromal["attention"], receiver_matched["attention"]),
                "ks_statistic": ks_2samp(receiver_stromal["attention"], receiver_matched["attention"]).statistic,
            }
        )

    metrics_df = pd.DataFrame(metrics + receiver_rows)
    metrics_df.to_csv(CACHE_DIR / "distance_matched_stromal_attention_null_metrics.csv", index=False)

    receiver_df = pd.DataFrame(receiver_rows).sort_values("top_fraction_mean_delta_stromal_minus_matched")
    colors = np.where(
        receiver_df["top_fraction_mean_delta_stromal_minus_matched"] > 0,
        "#B22222",
        "#2E7D32",
    )

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(
        receiver_df["receiver"],
        receiver_df["top_fraction_mean_delta_stromal_minus_matched"],
        color=colors,
        alpha=0.85,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Top 5% mean edge attention: stromal - distance-matched non-stromal")
    ax.set_ylabel("Receiver cell type")
    ax.set_title("Distance-Matched Stromal Attention Enrichment in High-Attention Edges")
    save_figure(fig, "distance_matched_stromal_attention_null")


# %% Done
print(f"Saved stromal bleeding analysis outputs to {OUT_DIR}")
