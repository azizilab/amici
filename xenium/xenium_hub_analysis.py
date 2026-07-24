# %% Imports
import os
import warnings

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

import gseapy as gp

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# %% Settings
SEED = 18
LABELS_KEY = "celltype_train_grouped"
DATA_DATE = "2025-05-01"
Q = 0.9
K = 10
RADIUS = 40.0
KNN_NEIGHBORS = 30
KNN_LEIDEN_RESOLUTION = 0.05
MIN_CELLS_PER_CLUSTER = 30
FDR_THRESHOLD = 0.25
POSITIVE_ONLY = True
EXPRESSION_LAYER = "counts_wo_bg"
NORMALIZE_TARGET_SUM = 1e4

DATA_PATH = f"data/xenium_sample1/xenium_sample1_filtered_{DATA_DATE}.h5ad"
ATTENTION_HUB_PATH = f"data/xenium_sample1/grid_search_cache/hub_results/hub_q{Q}_k{K}.csv"
ATTENTION_HUB_FALLBACK = "data/xenium_sample1/hub_graph_baseline_cache/attention_hub_features.csv"
CACHE_DIR = "data/xenium_sample1/hub_analysis_cache_graph_expression_v2"
FIGURE_DIR = "figures/hub_analysis"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

TUMOR_CELL_TYPES = ["Invasive_Tumor", "DCIS_1", "DCIS_2"]
IMMUNE_CELL_TYPES = [
    "CD8+_T_Cells",
    "CD4+_T_Cells",
    "IRF7+_DCs",
    "LAMP3+_DCs",
    "Macrophages_1",
    "Macrophages_2",
    "B_Cells",
    "Mast_Cells",
]
STROMAL_CELL_TYPES = ["Stromal", "Myoepi_ACTA2+", "Myoepi_KRT15+", "Perivascular-Like", "Endothelial"]
ALL_CELL_TYPES = TUMOR_CELL_TYPES + IMMUNE_CELL_TYPES + STROMAL_CELL_TYPES
BAR_CHART_PANELS = [
    ("Tumor (Hallmark)", TUMOR_CELL_TYPES),
    ("Immune (Reactome)", IMMUNE_CELL_TYPES),
    ("Stromal (Reactome)", STROMAL_CELL_TYPES),
]
CT_GENE_SETS = {ct: "MSigDB_Hallmark_2020" for ct in TUMOR_CELL_TYPES}
CT_GENE_SETS.update({ct: "Reactome_Pathways_2024" for ct in IMMUNE_CELL_TYPES + STROMAL_CELL_TYPES})

METHODS = {
    "amici_hub": {"label": "AMICI hubs", "cluster_col": "amici_hub_cluster", "color": "#1f77b4"},
    "radius40": {"label": "Radius graph", "cluster_col": "radius40_cluster", "color": "#2ca02c"},
    "knn30": {"label": "kNN graph", "cluster_col": "knn30_cluster", "color": "#9467bd"},
}

AGREEMENT_METHODS = {
    "Cell type": {"color": "#7f7f7f"},
    "Radius graph": {"color": METHODS["radius40"]["color"]},
    "kNN graph": {"color": METHODS["knn30"]["color"]},
}


# %% Helpers
def spatial_coords(adata):
    """Return spatial coordinates as an array."""
    coords = adata.obsm["spatial"]
    if isinstance(coords, pd.DataFrame):
        return coords[["X", "Y"]].to_numpy()
    return np.asarray(coords)[:, :2]


def normalize_rows(df):
    """Normalize rows to proportions."""
    row_sums = df.sum(axis=1)
    return df.div(row_sums.replace(0, np.nan), axis=0).fillna(0)


def expression_matrix(adata):
    """Return log-normalized expression used for graph-expression baselines."""
    if EXPRESSION_LAYER in adata.layers:
        x = adata.layers[EXPRESSION_LAYER].copy()
    elif "counts" in adata.layers:
        x = adata.layers["counts"].copy()
    else:
        x = adata.X.copy()

    if not sparse.issparse(x):
        x = sparse.csr_matrix(x)
    totals = np.asarray(x.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    x = sparse.diags(NORMALIZE_TARGET_SUM / totals) @ x
    x.data = np.log1p(x.data)
    return x.tocsr()


def build_radius_graph(adata, radius):
    """Build a fixed-radius spatial adjacency graph."""
    coords = spatial_coords(adata)
    graph = NearestNeighbors(radius=radius).fit(coords).radius_neighbors_graph(coords, mode="connectivity").tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph = graph.maximum(graph.T).tocsr()
    return graph


def compute_radius_expression_features(adata, radius):
    """Compute radius-neighborhood averaged expression vectors."""
    feature_path = os.path.join(CACHE_DIR, f"radius{int(radius)}_mean_expression_features.csv")
    if os.path.exists(feature_path):
        print(f"Loading cached radius expression features from {feature_path}", flush=True)
        return pd.read_csv(feature_path, index_col=0).loc[adata.obs_names]

    print(f"Computing radius-{radius:g} neighborhood mean expression features", flush=True)
    graph = build_radius_graph(adata, radius)
    x = expression_matrix(adata)
    neighbor_counts = np.asarray(graph.sum(axis=1)).ravel()
    neighbor_counts[neighbor_counts == 0] = 1.0
    features = graph @ x
    features = sparse.diags(1.0 / neighbor_counts) @ features
    features_df = pd.DataFrame(
        features.toarray(),
        index=adata.obs_names,
        columns=adata.var_names,
    )
    features_df.to_csv(feature_path)
    return features_df


def build_spatial_knn_graph(adata):
    """Build a spatial kNN adjacency graph."""
    coords = spatial_coords(adata)
    nn = NearestNeighbors(n_neighbors=KNN_NEIGHBORS + 1, algorithm="kd_tree").fit(coords)
    graph = nn.kneighbors_graph(coords, mode="connectivity").tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph.maximum(graph.T).tocsr()


def build_graph_adata(adata, graph, key):
    """Build an AnnData object containing a precomputed graph for Leiden."""
    knn_adata = sc.AnnData(X=sparse.csr_matrix((adata.n_obs, 1)), obs=adata.obs[[LABELS_KEY]].copy())
    knn_adata.obsp[f"{key}_connectivities"] = graph
    knn_adata.obsp[f"{key}_distances"] = graph.copy()
    knn_adata.uns[key] = {
        "connectivities_key": f"{key}_connectivities",
        "distances_key": f"{key}_distances",
        "params": {"method": "precomputed"},
    }
    return knn_adata


def load_or_build_clusters(adata, attention_df):
    """Load or construct AMICI, radius, and kNN graph clusters."""
    n_clusters = int(attention_df["hub_cluster"].nunique())
    cluster_path = os.path.join(CACHE_DIR, "hub_cluster_assignments.csv")
    if os.path.exists(cluster_path):
        clusters = pd.read_csv(cluster_path, index_col=0)
        for config in METHODS.values():
            adata.obs[config["cluster_col"]] = clusters.loc[adata.obs_names, config["cluster_col"]].astype(str)
        return clusters

    clusters = pd.DataFrame(index=adata.obs_names)
    clusters["amici_hub_cluster"] = attention_df.loc[adata.obs_names, "hub_cluster"].astype(str)

    print("Building radius graph clusters", flush=True)
    radius_features = compute_radius_expression_features(adata, RADIUS)
    radius_labels = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10).fit_predict(radius_features.values)
    clusters["radius40_cluster"] = radius_labels.astype(str)

    print("Building kNN graph clusters", flush=True)
    knn_cache_path = os.path.join(CACHE_DIR, "knn30_clusters.csv")
    if os.path.exists(knn_cache_path):
        print(f"Loading cached kNN clusters from {knn_cache_path}", flush=True)
        clusters["knn30_cluster"] = pd.read_csv(knn_cache_path, index_col=0).loc[adata.obs_names, "knn30_cluster"].astype(str)
    else:
        print("  computing spatial kNN adjacency", flush=True)
        knn_adata = build_graph_adata(adata, build_spatial_knn_graph(adata), "spatial_knn30")
        print(f"  running Leiden at resolution={KNN_LEIDEN_RESOLUTION:.3f}", flush=True)
        sc.tl.leiden(
            knn_adata,
            resolution=KNN_LEIDEN_RESOLUTION,
            neighbors_key="spatial_knn30",
            key_added="knn30_cluster",
            random_state=SEED,
        )
        clusters["knn30_cluster"] = knn_adata.obs["knn30_cluster"].astype(str).values
        pd.DataFrame({"knn30_cluster": clusters["knn30_cluster"]}, index=adata.obs_names).to_csv(knn_cache_path)
        print(
            f"kNN30 Leiden resolution={KNN_LEIDEN_RESOLUTION:.3f}; target k={n_clusters}; "
            f"observed k={clusters['knn30_cluster'].nunique()}",
            flush=True,
        )
    clusters.to_csv(cluster_path)

    radius_features.assign(radius40_cluster=clusters["radius40_cluster"]).to_csv(
        os.path.join(CACHE_DIR, "radius40_mean_expression_features.csv")
    )

    for config in METHODS.values():
        adata.obs[config["cluster_col"]] = clusters[config["cluster_col"]].astype(str)
    return clusters


def compute_deg_results(adata):
    """Compute DEGs for each cell type and cluster method."""
    print("Computing cluster DE genes", flush=True)
    deg_results = {}
    for ct in ALL_CELL_TYPES:
        ct_adata = adata[adata.obs[LABELS_KEY].astype(str) == ct].copy()
        if ct_adata.n_obs < MIN_CELLS_PER_CLUSTER * 2:
            continue
        for method, config in METHODS.items():
            cluster_col = config["cluster_col"]
            counts = ct_adata.obs[cluster_col].value_counts()
            valid_clusters = counts[counts >= MIN_CELLS_PER_CLUSTER].index.tolist()
            if len(valid_clusters) < 2:
                continue
            ct_sub = ct_adata[ct_adata.obs[cluster_col].isin(valid_clusters)].copy()
            sc.tl.rank_genes_groups(ct_sub, groupby=cluster_col, method="wilcoxon")
            for cluster in valid_clusters:
                deg_results[(ct, method, str(cluster))] = sc.get.rank_genes_groups_df(ct_sub, group=str(cluster))
    return deg_results


def parse_matched_size(tag_pct):
    """Parse the matched pathway size from gseapy Tag % output."""
    try:
        return int(str(tag_pct).split("/")[1])
    except (IndexError, ValueError):
        return np.nan


def run_gsea(deg_results):
    """Run preranked GSEA for each cell type, method, and cluster."""
    print("Running GSEA summaries", flush=True)
    gsea_path = os.path.join(CACHE_DIR, "hub_analysis_gsea_results.csv")
    if os.path.exists(gsea_path):
        return pd.read_csv(gsea_path)

    rows = []
    for (ct, method, cluster), deg_df in tqdm(deg_results.items(), desc="Running GSEA"):
        ranked = deg_df.set_index("names")["scores"].dropna()
        ranked = ranked[~ranked.index.duplicated(keep="first")].sort_values(ascending=False)
        if len(ranked) < 5:
            continue
        try:
            pre_res = gp.prerank(
                rnk=ranked,
                gene_sets=CT_GENE_SETS[ct],
                min_size=5,
                max_size=500,
                permutation_num=1000,
                seed=SEED,
                no_plot=True,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"GSEA failed for {ct}/{method}/{cluster}: {type(exc).__name__}: {exc}")
            continue
        if pre_res.res2d is None or pre_res.res2d.empty:
            continue
        res_df = pre_res.res2d.copy()
        for col in ["NES", "FDR q-val"]:
            res_df[col] = pd.to_numeric(res_df[col], errors="coerce")
        for _, row in res_df.iterrows():
            rows.append(
                {
                    "cell_type": ct,
                    "method": method,
                    "method_label": METHODS[method]["label"],
                    "cluster": cluster,
                    "gene_set_library": CT_GENE_SETS[ct],
                    "pathway": row.get("Term", row.name),
                    "nes": row.get("NES", np.nan),
                    "fdr": row.get("FDR q-val", np.nan),
                    "matched_size": parse_matched_size(row.get("Tag %", "")),
                }
            )

    gsea_df = pd.DataFrame(rows)
    if not gsea_df.empty:
        gsea_df["significant"] = gsea_df["fdr"].lt(FDR_THRESHOLD)
        if POSITIVE_ONLY:
            gsea_df["significant"] &= gsea_df["nes"].gt(0)
    gsea_df.to_csv(gsea_path, index=False)
    return gsea_df


def summarize_pathway_overlap(gsea_df):
    """Summarize method-unique and shared significant pathways."""
    rows = []
    method_keys = list(METHODS)
    for ct in ALL_CELL_TYPES:
        ct_df = gsea_df[gsea_df["cell_type"] == ct]
        sig_sets = {
            method: set(ct_df.loc[(ct_df["method"] == method) & ct_df["significant"], "pathway"].dropna())
            for method in method_keys
        }
        method_counts = {}
        for pathway in set.union(*sig_sets.values()) if any(sig_sets.values()) else set():
            present = tuple(method for method in method_keys if pathway in sig_sets[method])
            method_counts[present] = method_counts.get(present, 0) + 1

        categories = {
            "AMICI unique": sum(count for present, count in method_counts.items() if present == ("amici_hub",)),
            "Radius graph unique": sum(count for present, count in method_counts.items() if present == ("radius40",)),
            "kNN graph unique": sum(count for present, count in method_counts.items() if present == ("knn30",)),
            "Shared by 2": sum(count for present, count in method_counts.items() if len(present) == 2),
            "Shared by all 3": sum(count for present, count in method_counts.items() if len(present) == 3),
        }
        for category, n_pathways in categories.items():
            rows.append({"cell_type": ct, "category": category, "n_pathways": n_pathways})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(CACHE_DIR, "hub_analysis_pathway_overlap_summary.csv"), index=False)
    return summary


def plot_pathway_overlap(summary):
    """Plot stacked GSEA pathway overlap summaries."""
    category_colors = {
        "AMICI unique": METHODS["amici_hub"]["color"],
        "Radius graph unique": METHODS["radius40"]["color"],
        "kNN graph unique": METHODS["knn30"]["color"],
        "Shared by 2": "#9ecae1",
        "Shared by all 3": "#ffbf7f",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), squeeze=False)
    for ax, (title, cell_types) in zip(axes[0], BAR_CHART_PANELS):
        panel = summary[summary["cell_type"].isin(cell_types)]
        x_labels = []
        x_positions = []
        for cursor, ct in enumerate(cell_types):
            sub = panel[panel["cell_type"] == ct]
            bottom = 0
            for category, color in category_colors.items():
                value = int(sub.loc[sub["category"] == category, "n_pathways"].sum())
                ax.bar(cursor, value, bottom=bottom, color=color, edgecolor="black", linewidth=0.4)
                if value > 0:
                    ax.text(cursor, bottom + value / 2, str(value), ha="center", va="center", fontsize=8)
                bottom += value
            x_labels.append(ct.replace("_", " "))
            x_positions.append(cursor)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Significant pathways\n(FDR < 0.25)")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in category_colors.values()]
    fig.legend(handles, category_colors.keys(), loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("GSEA Pathway Overlap Across Hub Definitions", y=0.98, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(os.path.join(FIGURE_DIR, "hub_analysis_gsea_pathway_overlap.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURE_DIR, "hub_analysis_gsea_pathway_overlap.svg"), bbox_inches="tight")
    plt.close(fig)


def plot_ari_ami(adata):
    """Plot ARI/AMI agreement with AMICI hub labels."""
    reference = adata.obs["amici_hub_cluster"].astype(str)
    records = []
    comparisons = {
        "Cell type": adata.obs[LABELS_KEY].astype(str),
        "Radius graph": adata.obs["radius40_cluster"].astype(str),
        "kNN graph": adata.obs["knn30_cluster"].astype(str),
    }
    for label, values in comparisons.items():
        records.append({"Method": label, "Metric": "ARI", "Score": adjusted_rand_score(reference, values)})
        records.append({"Method": label, "Metric": "AMI", "Score": adjusted_mutual_info_score(reference, values)})
    metrics = pd.DataFrame(records)
    metrics.to_csv(os.path.join(CACHE_DIR, "hub_analysis_ari_ami.csv"), index=False)

    plt.figure(figsize=(6.5, 4.2))
    ax = sns.barplot(
        data=metrics,
        x="Metric",
        y="Score",
        hue="Method",
        hue_order=list(AGREEMENT_METHODS),
        palette={label: config["color"] for label, config in AGREEMENT_METHODS.items()},
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Agreement with AMICI hubs")
    ax.set_xlabel("")
    ax.set_title("Cluster Agreement with AMICI Communication Hubs")
    ax.legend(frameon=False, title="")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "hub_analysis_ari_ami.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURE_DIR, "hub_analysis_ari_ami.svg"), bbox_inches="tight")
    plt.close(fig)


# %% Run analysis
adata = sc.read_h5ad(DATA_PATH)
adata = adata[adata.obs["sample"].astype(str) == "0"].copy()
print(f"Loaded adata: {adata.n_obs} cells, {adata.n_vars} genes")

hub_path = ATTENTION_HUB_PATH if os.path.exists(ATTENTION_HUB_PATH) else ATTENTION_HUB_FALLBACK
attention_df = pd.read_csv(hub_path, index_col=0)
attention_df = attention_df.loc[adata.obs_names]
print(f"Loaded AMICI hubs from {hub_path}: {attention_df['hub_cluster'].nunique()} clusters")

clusters_df = load_or_build_clusters(adata, attention_df)
plot_ari_ami(adata)

gsea_path = os.path.join(CACHE_DIR, "hub_analysis_gsea_results.csv")
if os.path.exists(gsea_path):
    gsea_df = pd.read_csv(gsea_path)
else:
    gsea_df = run_gsea(compute_deg_results(adata))
if gsea_df.empty:
    raise RuntimeError("No GSEA results were generated.")
summary_df = summarize_pathway_overlap(gsea_df)
plot_pathway_overlap(summary_df)
print(f"Saved outputs to {FIGURE_DIR} and {CACHE_DIR}")
