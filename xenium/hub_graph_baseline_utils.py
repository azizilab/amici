# %% Shared utilities for Xenium communication hub graph baselines
import os

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import seaborn as sns
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from amici import AMICI


CELL_TYPE_PALETTE = {
    "CD8+_T_Cells": "#56B4E9",
    "CD4+_T_Cells": "#009E4E",
    "DCIS_1": "#E69F00",
    "DCIS_2": "#1a476e",
    "IRF7+_DCs": "#7f7f7f",
    "LAMP3+_DCs": "#305738",
    "Macrophages_1": "#e0a4dc",
    "Macrophages_2": "#de692a",
    "Myoepi_ACTA2+": "#823960",
    "Myoepi_KRT15+": "#575396",
    "Invasive_Tumor": "#cf4242",
    "Stromal": "#968253",
    "B_Cells": "#c5a9e8",
    "Mast_Cells": "#947b79",
    "Perivascular-Like": "#872727",
    "Endothelial": "#277987",
}

CLUSTER_PALETTE = {
    0: "#2E5BBA",
    1: "#FF8C42",
    2: "#228B22",
    3: "#8E44AD",
    4: "#B22222",
    5: "#FFD700",
    6: "#5D6D7E",
    7: "#16A085",
    8: "#FF6347",
    9: "#8B4513",
    10: "#FF1493",
    11: "#32CD32",
}

SEED = 18
LABELS_KEY = "celltype_train_grouped"
DATA_DATE = "2025-05-01"
MODEL_DATE = "2025-05-02"
N_NEIGHBORS = 50
ATTENTION_QUANTILE = 0.9
WANDB_RUN_ID = "te7pkv3z"
WANDB_SWEEP_ID = "g3mucw4s"

DATA_PATH = f"data/xenium_sample1/xenium_sample1_filtered_{DATA_DATE}.h5ad"
MODEL_PATH = os.path.join(
    f"saved_models/xenium_sample1_proseg_sweep_{DATA_DATE}_model_{MODEL_DATE}",
    f"xenium_{SEED}_sweep_{WANDB_SWEEP_ID}_{WANDB_RUN_ID}_params_{MODEL_DATE}",
)
CACHE_DIR = "data/xenium_sample1/hub_graph_baseline_cache"
FIGURE_DIR = "figures/hub_graph_baseline_comparison"


def ensure_dirs():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)


def load_xenium_model():
    pl.seed_everything(SEED)
    adata = sc.read_h5ad(DATA_PATH)
    AMICI.setup_anndata(
        adata,
        labels_key=LABELS_KEY,
        coord_obsm_key="spatial",
        n_neighbors=N_NEIGHBORS,
    )
    model = AMICI.load(MODEL_PATH, adata=adata)
    attention_patterns = model.get_attention_patterns()
    return adata, model, attention_patterns


def find_optimal_clusters(features_df, random_state=SEED, min_k=2, max_k=12):
    best_k = min_k
    best_score = -1
    max_k = min(max_k, len(features_df) - 1)
    for k in range(min_k, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(
            features_df.values
        )
        score = silhouette_score(features_df.values, labels)
        if score > best_score:
            best_k = k
            best_score = score
    return best_k, best_score


def compute_attention_hubs(attention_patterns, force=False):
    ensure_dirs()
    path = os.path.join(CACHE_DIR, "attention_hub_features.csv")
    if os.path.exists(path) and not force:
        hub_df = pd.read_csv(path, index_col=0)
        n_clusters = int(hub_df["hub_cluster"].nunique())
        return hub_df, n_clusters

    hub_features = attention_patterns.compute_communication_hubs(
        attention_quantile_threshold=ATTENTION_QUANTILE,
        n_clusters=2,
        random_state=SEED,
    ).drop(columns=["hub_cluster"])
    n_clusters, score = find_optimal_clusters(hub_features)
    print(f"Attention hub clusters: k={n_clusters}, silhouette={score:.4f}")

    hub_df = attention_patterns.compute_communication_hubs(
        attention_quantile_threshold=ATTENTION_QUANTILE,
        n_clusters=n_clusters,
        random_state=SEED,
    )
    hub_df.to_csv(path)
    return hub_df, n_clusters


def normalize_composition(features_df):
    row_sums = features_df.sum(axis=1)
    return features_df.div(row_sums.replace(0, np.nan), axis=0).fillna(0)


def cluster_features(features_df, cluster_col, n_clusters):
    clustered = features_df.copy()
    labels = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10).fit_predict(
        clustered.values
    )
    clustered[cluster_col] = labels
    return clustered


def compute_knn_composition_features(adata, nn_idxs_df):
    cell_types = list(adata.obs[LABELS_KEY].unique())
    nn_labels = nn_idxs_df.applymap(lambda obs_name: adata.obs.at[obs_name, LABELS_KEY])
    long_df = nn_labels.reset_index().melt(
        id_vars="index",
        var_name="neighbor_position",
        value_name="neighbor_label",
    )
    counts = long_df.groupby(["index", "neighbor_label"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=adata.obs_names, columns=cell_types, fill_value=0)
    return normalize_composition(counts)


def spatial_coords(adata):
    coords = adata.obsm["spatial"]
    if isinstance(coords, pd.DataFrame):
        return coords[["X", "Y"]].to_numpy()
    return np.asarray(coords)[:, :2]


def auto_radius_from_knn(attention_patterns):
    dists = getattr(attention_patterns, "_nn_dists_df", None)
    if dists is not None:
        return float(dists.iloc[:, -1].median())
    coords = spatial_coords(attention_patterns._adata)
    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    return float(np.median(distances[:, -1]))


def radius_neighbor_indices(adata, radius):
    coords = spatial_coords(adata)
    nbrs = NearestNeighbors(radius=radius).fit(coords)
    indices = nbrs.radius_neighbors(coords, return_distance=False)
    obs_names = np.asarray(adata.obs_names)
    result = []
    for i, neigh in enumerate(indices):
        neigh = neigh[neigh != i]
        result.append(obs_names[neigh])
    return result


def compute_radius_composition_features(adata, radius):
    cell_types = list(adata.obs[LABELS_KEY].unique())
    labels = adata.obs[LABELS_KEY]
    rows = []
    for obs_name, neigh_names in zip(adata.obs_names, radius_neighbor_indices(adata, radius)):
        counts = labels.loc[neigh_names].value_counts()
        rows.append(counts.reindex(cell_types, fill_value=0))
    counts_df = pd.DataFrame(rows, index=adata.obs_names, columns=cell_types)
    return normalize_composition(counts_df)


def load_panel_lr_pairs(adata):
    ensure_dirs()
    cache_path = os.path.join(CACHE_DIR, "omnipath_panel_lr_pairs.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    from omnipath.interactions import import_intercell_network

    lr_df = import_intercell_network(
        interactions_params={"datasets": ["ligrecextra", "omnipath"]},
        transmitter_params={"categories": ["ligand"]},
        receiver_params={"categories": ["receptor"]},
    )
    lr_df = lr_df.rename(
        columns={
            "genesymbol_intercell_source": "ligand",
            "genesymbol_intercell_target": "receptor",
        }
    )[["ligand", "receptor"]].dropna().drop_duplicates()
    panel_genes = set(adata.var_names)
    lr_df = lr_df[lr_df["ligand"].isin(panel_genes) & lr_df["receptor"].isin(panel_genes)]
    lr_df.to_csv(cache_path, index=False)
    print(f"Cached {len(lr_df)} panel ligand-receptor pairs to {cache_path}")
    return lr_df


def expression_matrix(adata):
    x = adata.X
    if sparse.issparse(x):
        return x.tocsr()
    return sparse.csr_matrix(np.asarray(x))


def compute_lr_weighted_radius_features(adata, radius):
    lr_df = load_panel_lr_pairs(adata)
    if len(lr_df) == 0:
        raise ValueError("No ligand-receptor pairs overlap the Xenium panel.")

    cell_types = list(adata.obs[LABELS_KEY].unique())
    labels = adata.obs[LABELS_KEY].to_numpy()
    label_to_col = {label: i for i, label in enumerate(cell_types)}
    gene_to_idx = {gene: i for i, gene in enumerate(adata.var_names)}
    ligand_idxs = np.array([gene_to_idx[g] for g in lr_df["ligand"]], dtype=int)
    receptor_idxs = np.array([gene_to_idx[g] for g in lr_df["receptor"]], dtype=int)
    x = expression_matrix(adata)

    features = np.zeros((adata.n_obs, len(cell_types)), dtype=float)
    for i, neigh_names in enumerate(radius_neighbor_indices(adata, radius)):
        if len(neigh_names) == 0:
            continue
        neigh_pos = adata.obs_names.get_indexer(neigh_names)
        receptor_expr = np.asarray(x[i, receptor_idxs].todense()).ravel()
        if not np.any(receptor_expr):
            continue
        ligand_expr = x[neigh_pos][:, ligand_idxs].multiply(receptor_expr)
        weights = np.asarray(ligand_expr.sum(axis=1)).ravel()
        for pos, weight in zip(neigh_pos, weights):
            if weight > 0:
                features[i, label_to_col[labels[pos]]] += weight

    features_df = pd.DataFrame(features, index=adata.obs_names, columns=cell_types)
    return normalize_composition(features_df)


def comparison_metrics(adata, attention_df, baseline_df, cluster_col, feature_cols):
    attention_labels = attention_df.loc[adata.obs_names, "hub_cluster"].astype(str).values
    baseline_labels = baseline_df.loc[adata.obs_names, cluster_col].astype(str).values
    cell_type_labels = adata.obs[LABELS_KEY].astype(str).values
    values = baseline_df.loc[adata.obs_names, feature_cols].values
    sil = silhouette_score(values, baseline_labels) if len(set(baseline_labels)) > 1 else np.nan
    overlap_df, overlap_score = matched_cluster_overlap(attention_labels, baseline_labels)
    return {
        "n_cells": adata.n_obs,
        "n_clusters": int(baseline_df[cluster_col].nunique()),
        "ari_vs_attention_hubs": adjusted_rand_score(attention_labels, baseline_labels),
        "ami_vs_attention_hubs": adjusted_mutual_info_score(attention_labels, baseline_labels),
        "matched_overlap_vs_attention_hubs": overlap_score,
        "ari_vs_cell_type": adjusted_rand_score(cell_type_labels, baseline_labels),
        "ami_vs_cell_type": adjusted_mutual_info_score(cell_type_labels, baseline_labels),
        "silhouette": sil,
    }


def matched_cluster_overlap(reference_labels, query_labels):
    overlap_counts = pd.crosstab(
        pd.Series(reference_labels, name="attention_hub_cluster"),
        pd.Series(query_labels, name="graph_cluster"),
    )
    row_sums = overlap_counts.sum(axis=1)
    overlap_fraction = overlap_counts.div(row_sums.replace(0, np.nan), axis=0).fillna(0)
    best_overlap = overlap_fraction.max(axis=1)
    weighted_score = float((best_overlap * row_sums / row_sums.sum()).sum())
    overlap_df = overlap_fraction.reset_index()
    overlap_df["best_graph_cluster"] = overlap_fraction.idxmax(axis=1).values
    overlap_df["best_overlap_fraction"] = best_overlap.values
    overlap_df["attention_hub_size"] = row_sums.values
    return overlap_df, weighted_score


def plot_spatial_clusters(adata, cluster_col, out_prefix):
    plot_df = pd.DataFrame(spatial_coords(adata), columns=["X", "Y"], index=adata.obs_names)
    plot_df[cluster_col] = adata.obs[cluster_col].astype(int).values
    plt.figure(figsize=(20, 6))
    sns.scatterplot(
        data=plot_df,
        x="X",
        y="Y",
        hue=cluster_col,
        palette=CLUSTER_PALETTE,
        alpha=0.7,
        s=8,
        linewidth=0,
    )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(cluster_col)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_spatial.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_prefix}_spatial.svg", dpi=300, bbox_inches="tight")
    plt.close()


def plot_cluster_composition(clustered_df, cluster_col, out_prefix):
    feature_cols = [c for c in clustered_df.columns if c != cluster_col]
    composition = clustered_df.groupby(cluster_col)[feature_cols].mean()
    composition = normalize_composition(composition)
    composition.plot(
        kind="bar",
        stacked=True,
        color=[CELL_TYPE_PALETTE.get(c, "#888888") for c in feature_cols],
        figsize=(12, 8),
    )
    plt.xlabel(cluster_col)
    plt.ylabel("Mean sender cell type proportion")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_composition.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_prefix}_composition.svg", dpi=300, bbox_inches="tight")
    plt.close()


def get_attention_vs_cell_type_metrics():
    path = os.path.join(CACHE_DIR, "attention_hub_features.csv")
    if not os.path.exists(path):
        return None

    attention_df = pd.read_csv(path, index_col=0)
    adata = sc.read_h5ad(DATA_PATH, backed="r")
    shared = adata.obs_names.intersection(attention_df.index)
    attention_labels = attention_df.loc[shared, "hub_cluster"].astype(str).values
    cell_type_labels = adata.obs.loc[shared, LABELS_KEY].astype(str).values
    return {
        "Method": "Cell type",
        "ari": adjusted_rand_score(attention_labels, cell_type_labels),
        "ami": adjusted_mutual_info_score(attention_labels, cell_type_labels),
    }


def build_attention_comparison_metrics():
    metric_records = []
    cell_type_metrics = get_attention_vs_cell_type_metrics()
    if cell_type_metrics:
        metric_records.extend(
            [
                {"Method": cell_type_metrics["Method"], "Metric": "ARI", "Score": cell_type_metrics["ari"]},
                {"Method": cell_type_metrics["Method"], "Metric": "AMI", "Score": cell_type_metrics["ami"]},
            ]
        )

    for filename in os.listdir(CACHE_DIR):
        if not filename.endswith("_metrics.csv"):
            continue
        if filename.startswith("combined_"):
            continue
        method_name = filename.replace("_metrics.csv", "")
        metrics_df = pd.read_csv(os.path.join(CACHE_DIR, filename))
        if metrics_df.empty:
            continue
        row = metrics_df.iloc[0]
        if "ari_vs_attention_hubs" not in row or "ami_vs_attention_hubs" not in row:
            continue
        label = {"lr_graph": "LR graph", "radius_graph": "Radius graph"}.get(
            method_name,
            method_name.replace("_", " ").title(),
        )
        metric_records.extend(
            [
                {
                    "Method": label,
                    "Metric": "ARI",
                    "Score": row["ari_vs_attention_hubs"],
                },
                {
                    "Method": label,
                    "Metric": "AMI",
                    "Score": row["ami_vs_attention_hubs"],
                },
            ]
        )

    return pd.DataFrame(metric_records)


def plot_metric_barplots(metrics, method_name, out_prefix):
    label = {"lr_graph": "LR graph", "radius_graph": "Radius graph"}.get(
        method_name,
        method_name.replace("_", " ").title(),
    )
    plot_df = pd.DataFrame(
        [
            {"Metric": "ARI", "Method": label, "Score": metrics["ari_vs_attention_hubs"]},
            {"Metric": "AMI", "Method": label, "Score": metrics["ami_vs_attention_hubs"]},
        ]
    )

    plt.figure(figsize=(5, 4.2))
    ax = sns.barplot(data=plot_df, x="Metric", y="Score", hue="Method", palette=["#4C78A8"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(min(-0.05, float(plot_df["Score"].min()) - 0.05), max(0.15, float(plot_df["Score"].max()) + 0.08))
    ax.set_xlabel("")
    ax.set_ylabel("Agreement with AMICI hubs")
    ax.set_title(f"AMICI Hubs vs {label}")
    ax.legend_.remove()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_ari_ami_barplot.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_prefix}_ari_ami_barplot.svg", dpi=300, bbox_inches="tight")
    plt.close()


def plot_combined_metric_barplots():
    plot_df = build_attention_comparison_metrics()
    if plot_df.empty:
        return

    plot_df.to_csv(os.path.join(CACHE_DIR, "combined_ari_ami_metrics.csv"), index=False)

    method_order = [m for m in ["Cell type", "LR graph", "Radius graph"] if m in set(plot_df["Method"])]
    palette = {
        "Cell type": "#4C78A8",
        "LR graph": "#F58518",
        "Radius graph": "#54A24B",
    }
    plt.figure(figsize=(7, 4.5))
    ax = sns.barplot(
        data=plot_df,
        x="Metric",
        y="Score",
        hue="Method",
        hue_order=method_order,
        palette=palette,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(min(-0.05, float(plot_df["Score"].min()) - 0.05), max(0.15, float(plot_df["Score"].max()) + 0.08))
    ax.set_xlabel("")
    ax.set_ylabel("Agreement with AMICI hubs")
    ax.set_title("AMICI Hub Agreement with Alternative Label Sets")
    ax.legend(frameon=False, title="")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "xenium_graph_baseline_ari_ami_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(FIGURE_DIR, "xenium_graph_baseline_ari_ami_summary.svg"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_baseline_outputs(adata, attention_df, baseline_df, method_name, extra_metrics=None):
    ensure_dirs()
    cluster_col = f"{method_name}_cluster"
    feature_cols = [c for c in baseline_df.columns if c != cluster_col]
    metrics = comparison_metrics(adata, attention_df, baseline_df, cluster_col, feature_cols)
    if extra_metrics:
        metrics.update(extra_metrics)

    adata.obs[cluster_col] = baseline_df.loc[adata.obs_names, cluster_col].astype(int).values
    out_prefix = os.path.join(FIGURE_DIR, f"xenium_{method_name}")
    baseline_df.to_csv(os.path.join(CACHE_DIR, f"{method_name}_features.csv"))
    pd.DataFrame([metrics]).to_csv(
        os.path.join(CACHE_DIR, f"{method_name}_metrics.csv"), index=False
    )
    overlap_df, _ = matched_cluster_overlap(
        attention_df.loc[adata.obs_names, "hub_cluster"].astype(str).values,
        baseline_df.loc[adata.obs_names, cluster_col].astype(str).values,
    )
    overlap_df.to_csv(os.path.join(CACHE_DIR, f"{method_name}_overlap_matrix.csv"), index=False)
    plot_spatial_clusters(adata, cluster_col, out_prefix)
    plot_cluster_composition(baseline_df, cluster_col, out_prefix)
    plot_metric_barplots(metrics, method_name, out_prefix)
    plot_combined_metric_barplots()
    print(pd.Series(metrics).to_string())
    return metrics
