# %% Imports
import os

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# %% Settings
SEED = 18
K_NEIGHBORS = 8
MAX_CELLS_PER_TYPE = 5000
MIN_CELLS_PER_TYPE = 300
N_PCS = 5
NORMALIZE_TARGET_SUM = 1e4

DATA_PATH = "data/xenium_sample1_filtered_lowres_2026-03-27.h5ad"
LOWRES_KEY = "celltype_lowres"
HIGHRES_KEY = "celltype_train_grouped"
FIGURE_DIR = "figures/within_type_autocorrelation"
CACHE_DIR = "data/xenium_sample1/within_type_autocorrelation_cache"
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# %% Helpers
def spatial_coords(adata):
    """Return spatial coordinates as an array."""
    coords = adata.obsm["spatial"]
    if isinstance(coords, pd.DataFrame):
        return coords[["X", "Y"]].to_numpy()
    return np.asarray(coords)[:, :2]


def dense_matrix(x):
    """Convert sparse or dense matrix slices to dense arrays."""
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def normalized_expression(adata):
    """Return log-normalized expression from count-like values."""
    if "counts_wo_bg" in adata.layers:
        x = dense_matrix(adata.layers["counts_wo_bg"])
        source = "counts_wo_bg"
    elif "counts" in adata.layers:
        x = dense_matrix(adata.layers["counts"])
        source = "counts"
    else:
        x = dense_matrix(adata.X)
        source = "X"

    x = x.astype(float, copy=False)
    totals = x.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    x = x / totals * NORMALIZE_TARGET_SUM
    return np.log1p(x), source


def zscore_columns(x):
    """Column-wise z-score with stable handling of constant features."""
    x = x.astype(float, copy=False)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    scale = np.nanstd(x, axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    return x / scale


def knn_weight_matrix(coords):
    """Build an undirected within-label spatial kNN graph."""
    k_eff = min(K_NEIGHBORS + 1, coords.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff).fit(coords)
    graph = nn.kneighbors_graph(coords, mode="connectivity").tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph = graph.maximum(graph.T)
    graph.eliminate_zeros()
    return graph


def morans_i(values, graph):
    """Compute Moran's I for one or more feature vectors."""
    x = zscore_columns(np.asarray(values))
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    w_sum = graph.sum()
    denom = np.sum(x * x, axis=0)
    numer = np.asarray(graph @ x * x).sum(axis=0)
    return (n / w_sum) * numer / np.maximum(denom, 1e-12)


def sample_label_indices(adata, label, rng):
    """Sample cells from one coarse annotation label."""
    all_indices = np.where(adata.obs[LOWRES_KEY].astype(str).to_numpy() == label)[0]
    if len(all_indices) > MAX_CELLS_PER_TYPE:
        all_indices = rng.choice(all_indices, size=MAX_CELLS_PER_TYPE, replace=False)
    return np.sort(all_indices)


def compute_validation_scores(adata):
    """Compare unsupervised low-res autocorrelation scores with high-res-label oracle scores."""
    rng = np.random.default_rng(SEED)
    rows = []

    lowres_labels = adata.obs[LOWRES_KEY].astype(str)
    for label in lowres_labels.value_counts().index:
        if (lowres_labels == label).sum() < MIN_CELLS_PER_TYPE:
            continue

        idx = sample_label_indices(adata, label, rng)
        sub = adata[idx].copy()
        highres = sub.obs[HIGHRES_KEY].astype(str)
        if highres.nunique() < 2:
            continue

        coords = spatial_coords(sub)
        graph = knn_weight_matrix(coords)
        x, expression_source = normalized_expression(sub)
        pcs = PCA(n_components=min(N_PCS, x.shape[1]), random_state=SEED).fit_transform(zscore_columns(x))
        pc_morans = morans_i(pcs, graph)

        highres_onehot = pd.get_dummies(highres).to_numpy(dtype=float)
        subtype_morans = morans_i(highres_onehot, graph)

        rows.append(
            {
                "lowres_label": label,
                "n_cells_sampled": sub.n_obs,
                "n_cells_total": int((lowres_labels == label).sum()),
                "n_highres_subtypes": highres.nunique(),
                "expression_source": expression_source,
                "deployable_max_pc_morans_i": float(np.max(pc_morans)),
                "deployable_mean_top2_pc_morans_i": float(np.mean(np.sort(pc_morans)[-min(2, len(pc_morans)) :])),
                "oracle_max_subtype_morans_i": float(np.max(subtype_morans)),
                "oracle_mean_subtype_morans_i": float(np.mean(subtype_morans)),
                "top_pc": int(np.argmax(pc_morans) + 1),
                "top_pc_morans_i": float(np.max(pc_morans)),
                "top_subtype": pd.get_dummies(highres).columns[int(np.argmax(subtype_morans))],
                "top_subtype_morans_i": float(np.max(subtype_morans)),
            }
        )

    scores = pd.DataFrame(rows)
    scores.to_csv(os.path.join(CACHE_DIR, "within_type_autocorrelation_validation_scores.csv"), index=False)
    return scores


def plot_lowres_composition(adata):
    """Show how low-resolution labels merge high-resolution annotated states."""
    comp = pd.crosstab(
        adata.obs[LOWRES_KEY].astype(str),
        adata.obs[HIGHRES_KEY].astype(str),
        normalize="index",
    )
    comp.to_csv(os.path.join(CACHE_DIR, "lowres_highres_label_composition.csv"))

    fig, ax = plt.subplots(figsize=(11, 5))
    comp.plot(kind="bar", stacked=True, ax=ax, width=0.85, colormap="tab20")
    ax.set_ylabel("Fraction of low-resolution label")
    ax.set_xlabel("Low-resolution label")
    ax.set_title("High-resolution state composition within low-resolution labels")
    ax.legend(title="High-resolution label", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIGURE_DIR, f"lowres_highres_label_composition.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_deployable_vs_oracle(scores):
    """Plot whether deployable expression autocorrelation matches high-res oracle structure."""
    x_col = "deployable_mean_top2_pc_morans_i"
    y_col = "oracle_max_subtype_morans_i"
    rho, p_value = spearmanr(scores[x_col], scores[y_col])

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.regplot(
        data=scores,
        x=x_col,
        y=y_col,
        ax=ax,
        scatter_kws={"s": 70, "color": "#4C78A8", "edgecolor": "black", "linewidths": 0.5},
        line_kws={"color": "0.25", "linewidth": 1.5},
        ci=None,
    )
    for _, row in scores.iterrows():
        ax.text(row[x_col], row[y_col], row["lowres_label"], fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("Deployable score: mean top-2 PC Moran's I")
    ax.set_ylabel("Oracle score: max subtype-label Moran's I")
    ax.set_title(f"Unsupervised autocorrelation agrees with subtype-label oracle\nSpearman rho={rho:.2f}, p={p_value:.3g}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIGURE_DIR, f"autocorrelation_deployable_vs_oracle.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_score_comparison(scores):
    """Plot deployable and oracle autocorrelation scores by coarse label."""
    plot_df = scores[
        ["lowres_label", "deployable_mean_top2_pc_morans_i", "oracle_max_subtype_morans_i", "n_highres_subtypes"]
    ].copy()
    plot_df = plot_df.sort_values("oracle_max_subtype_morans_i", ascending=False)
    melted = plot_df.melt(
        id_vars=["lowres_label", "n_highres_subtypes"],
        value_vars=["deployable_mean_top2_pc_morans_i", "oracle_max_subtype_morans_i"],
        var_name="score_type",
        value_name="morans_i",
    )
    melted["score_type"] = melted["score_type"].map(
        {
            "deployable_mean_top2_pc_morans_i": "Deployable expression PCs",
            "oracle_max_subtype_morans_i": "High-res subtype oracle",
        }
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(
        data=melted,
        x="lowres_label",
        y="morans_i",
        hue="score_type",
        order=plot_df["lowres_label"],
        palette=["#4C78A8", "#F58518"],
        ax=ax,
    )
    subtype_counts = plot_df.set_index("lowres_label")["n_highres_subtypes"]
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels([f"{label}\n({subtype_counts[label]} subtypes)" for label in plot_df["lowres_label"]], rotation=35, ha="right")
    ax.set_xlabel("Low-resolution cell type")
    ax.set_ylabel("Moran's I")
    ax.set_title("Spatial autocorrelation diagnostic by coarse annotation")
    ax.legend(frameon=False, title="")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIGURE_DIR, f"autocorrelation_score_comparison_by_label.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# %% Run
print(f"Loading {DATA_PATH}", flush=True)
adata = sc.read_h5ad(DATA_PATH)
scores_df = compute_validation_scores(adata)
plot_lowres_composition(adata)
plot_deployable_vs_oracle(scores_df)
plot_score_comparison(scores_df)
print(f"Saved outputs to {FIGURE_DIR} and {CACHE_DIR}", flush=True)
