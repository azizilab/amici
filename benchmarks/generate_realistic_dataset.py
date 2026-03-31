import os

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp_sparse
import scvi
import torch
from gpu_utils import select_gpu

from amici.tools import is_count_data

SEED = 42
SPATIAL_BOUNDS = ((0, 3200), (0, 2800))
N_HVGS = 500
N_SUBCLUSTERS_PER_CLUSTER = 2
N_GENES_FOR_SUBCLUSTERING = 200

# QC thresholds
MIN_GENES = 200
MIN_CELLS = 3
MIN_COUNTS = 500
MT_PCT_THRESHOLD = 20

# Spatial test region bounds
TEST_X_MIN, TEST_X_MAX = 1250, 2400
TEST_Y_MIN, TEST_Y_MAX = 0, 800

# Cell types to exclude from both Flex and Xenium data
EXCLUDE_CELL_TYPES = [
    "T_Cell_&_Tumor_Hybrid",
    "Stromal_&_T_Cell_Hybrid",
    "IRF7+_DCs",
    "LAMP3+_DCs",
    "Mast_Cells",
    "Perivascular-Like",
    "Prolif_Invasive_Tumor",
]

# Rename Flex subtypes to canonical cell types matching Xenium spatial labels
FLEX_RENAME_MAP = {
    "Macrophages_1": "Macrophages",
    "Macrophages_2": "Macrophages",
    "DCIS 1": "DCIS",
    "DCIS 2": "DCIS",
    "Myoepi_ACTA2+": "Myoepi",
    "Myoepi_KRT15+": "Myoepi",
    "CD4+_T_Cells": "T_Cells",
    "CD8+_T_Cells": "T_Cells",
}

# Canonical cell types after filtering and renaming (must match Xenium)
CELL_TYPES = [
    "T_Cells",
    "B_Cells",
    "Invasive_Tumor",
    "DCIS",
    "Macrophages",
    "Endothelial",
    "Stromal",
    "Myoepi",
]

# Ground truth interactions: receiver <- sender, with spatial length scale
# Chosen based on UMAP centroid distances — far-apart pairs ensure distinct
# expression signatures between interacting (sub1) and non-interacting (sub0) subtypes.
GT_INTERACTIONS = [
    {"receiver": "DCIS", "sender": "Macrophages", "length_scale": 40.0},
    {"receiver": "Endothelial", "sender": "T_Cells", "length_scale": 25.0},
    {"receiver": "Myoepi", "sender": "Invasive_Tumor", "length_scale": 15.0},
]


def generate_realistic_dataset(flex_h5_path, annot_path, xenium_path, output_path, scvi_model_dir=None):
    """
    Generate a semi-synthetic spatial dataset from breast cancer Flex and Xenium data.

    Args:
        flex_h5_path: Path to the raw 10x H5 file for the Flex scFFPE-seq dataset.
        annot_path: Path to the cell type annotation Excel file.
        xenium_path: Path to the Xenium spatial h5ad file.
        output_path: Path to write the output semi-synthetic h5ad.
        scvi_model_dir: Directory to save/load the scVI model.
    """
    if os.path.exists(output_path):
        print(f"Output already exists at {output_path}, skipping.")
        return

    flex_ad = _load_flex_data(flex_h5_path, annot_path)
    flex_pp, scvi_model = _preprocess_flex(flex_ad, n_hvgs=N_HVGS, model_save_dir=scvi_model_dir)
    flex_pp = _subcluster_by_celltype(flex_pp, N_SUBCLUSTERS_PER_CLUSTER, N_GENES_FOR_SUBCLUSTERING)

    spatial_subset = _load_xenium_spatial(xenium_path, SPATIAL_BOUNDS)

    # Filter GT_INTERACTIONS to only those where both cell types are present
    present_cell_types = set(spatial_subset.obs["Cluster"].unique())
    active_interactions = [
        i for i in GT_INTERACTIONS if i["receiver"] in present_cell_types and i["sender"] in present_cell_types
    ]

    receptor_clusters = sorted({i["receiver"] for i in active_interactions})
    interaction_df = _create_interaction_df(active_interactions, receptor_clusters)

    spatial_data = _determine_spatial_subtypes(spatial_subset, interaction_df)

    rng = np.random.default_rng(SEED)
    sampled_data = _sample_flex_by_subtype(flex_pp, spatial_data, scvi_model, rng)

    sampled_data = _create_train_test_split(sampled_data)
    sampled_data.obs_names_make_unique()
    sampled_data.write_h5ad(output_path)


def _load_flex_data(flex_h5_path, annot_path, annot_sheet="Fig. 2a scFFPE-seq UMAP"):
    """
    Load and QC-filter the Flex scFFPE-seq dataset with cell type annotations.

    Args:
        flex_h5_path: Path to the raw 10x H5 count matrix.
        annot_path: Path to the supplementary Excel file with cell type annotations.
        annot_sheet: Sheet name in the Excel file containing the annotations.

    Returns
    -------
        The QC-filtered AnnData with cell_type obs column.
    """
    print("Loading Flex 10x h5 file...")
    flex_ad = sc.read_10x_h5(flex_h5_path)
    flex_ad.var_names_make_unique()

    print("Loading cell type annotations...")
    annot = pd.read_excel(annot_path, sheet_name=annot_sheet)
    annotated_barcodes = set(annot["Barcode"])
    flex_ad = flex_ad[flex_ad.obs_names.isin(annotated_barcodes)].copy()

    annot_map = annot.set_index("Barcode")
    flex_ad.obs["cell_type"] = annot_map.loc[flex_ad.obs_names, "Annotation"].values
    flex_ad.obs["cell_type"] = flex_ad.obs["cell_type"].astype("category")

    # QC filtering
    flex_ad.var["mt"] = flex_ad.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(flex_ad, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    sc.pp.filter_cells(flex_ad, min_genes=MIN_GENES)
    sc.pp.filter_cells(flex_ad, min_counts=MIN_COUNTS)
    flex_ad = flex_ad[flex_ad.obs["pct_counts_mt"] < MT_PCT_THRESHOLD, :].copy()
    sc.pp.filter_genes(flex_ad, min_cells=MIN_CELLS)

    # Remove excluded cell types and rename subtypes to canonical labels
    flex_ad = flex_ad[~flex_ad.obs["cell_type"].isin(EXCLUDE_CELL_TYPES)].copy()
    flex_ad.obs["cell_type"] = flex_ad.obs["cell_type"].cat.remove_unused_categories()
    flex_ad.obs["cell_type"] = flex_ad.obs["cell_type"].astype(str).replace(FLEX_RENAME_MAP).astype("category")

    print(f"Flex data shape after QC and filtering: {flex_ad.shape}")
    print(flex_ad.obs["cell_type"].value_counts())
    return flex_ad


def _preprocess_flex(adata, n_hvgs=500, model_save_dir=None):
    """
    Preprocess Flex data and train (or load) an scVI model for posterior sampling.

    Args:
        adata: Input AnnData with raw counts in X.
        n_hvgs: Number of highly variable genes to select.
        model_save_dir: Directory to save/load the scVI model.

    Returns
    -------
        adata_pp: Preprocessed AnnData with scVI embeddings in obsm["X_scvi"].
        scvi_model: Trained scVI model used later for posterior sampling.
    """
    adata_pp = adata.copy()
    if is_count_data(adata_pp.X):
        if "counts" not in adata_pp.layers:
            adata_pp.layers["counts"] = adata_pp.X.copy()
        sc.pp.normalize_total(adata_pp)
        sc.pp.log1p(adata_pp)

    sc.pp.highly_variable_genes(adata_pp, flavor="seurat_v3", layer="counts", n_top_genes=n_hvgs, subset=True)
    scvi.model.SCVI.setup_anndata(adata_pp, layer="counts", batch_key=None)

    if model_save_dir is not None and os.path.exists(model_save_dir):
        print(f"Loading scVI model from {model_save_dir}...")
        scvi_model = scvi.model.SCVI.load(model_save_dir, adata=adata_pp)
    else:
        print("Training scVI model...")
        scvi_model = scvi.model.SCVI(adata_pp)
        scvi_model.train()
        if model_save_dir is not None:
            scvi_model.save(model_save_dir, overwrite=True)
            print(f"scVI model saved to {model_save_dir}.")

    adata_pp.obsm["X_scvi"] = scvi_model.get_latent_representation()
    return adata_pp, scvi_model


def _subcluster_by_celltype(adata, n_subcluster_per_cluster=2, n_genes_for_subclustering=200):
    """
    Subcluster each cell type into subtypes using per-cell-type HVGs + PCA + Leiden.

    Args:
        adata: AnnData with cell_type obs column.
        n_subcluster_per_cluster: Number of subclusters per cell type.
        n_genes_for_subclustering: Number of genes to use for per-type PCA.

    Returns
    -------
        AnnData with subtype obs column added.
    """
    print(f"Subclustering with {n_subcluster_per_cluster} subclusters per cell type...")
    N_PCA_COMPS = 20
    adata.obs["subtype"] = "unassigned"

    for ct in adata.obs["cell_type"].unique():
        ct_mask = adata.obs["cell_type"] == ct
        ct_adata = adata[ct_mask].copy()

        n_hvg = min(n_genes_for_subclustering, ct_adata.shape[1] - 1)
        if "counts" in ct_adata.layers:
            sc.pp.highly_variable_genes(ct_adata, flavor="seurat_v3", layer="counts", n_top_genes=n_hvg)
        else:
            sc.pp.highly_variable_genes(ct_adata, n_top_genes=n_hvg)

        ct_hvg = ct_adata[:, ct_adata.var["highly_variable"]].copy()
        n_comps = min(N_PCA_COMPS, ct_hvg.shape[1] - 1, ct_hvg.shape[0] - 1)
        sc.pp.pca(ct_hvg, n_comps=n_comps)

        rep_key = f"X_rep_subclustering_{ct}"
        if rep_key not in adata.obsm:
            adata.obsm[rep_key] = np.zeros((adata.n_obs, n_comps))
        adata.obsm[rep_key][ct_mask.values] = ct_hvg.obsm["X_pca"]

        _leiden_subclustering_binary_search(adata, ct, n_subcluster_per_cluster)

    adata.obs["subtype"] = adata.obs["subtype"].astype("category")
    return adata


def _leiden_subclustering_binary_search(adata, cell_type, target_subclusters):
    """
    Binary search for the Leiden resolution that yields target_subclusters within a cell type.

    Args:
        adata: Full AnnData object (modified in-place via obs["subtype"]).
        cell_type: The cell type to subcluster (used to select rows and name keys).
        target_subclusters: Desired number of subclusters.
    """
    cluster_data = adata[adata.obs["cell_type"] == cell_type].copy()
    sc.pp.neighbors(cluster_data, use_rep=f"X_rep_subclustering_{cell_type}")

    leiden_col = f"leiden_sub_{cell_type}"
    min_resolution = 0.01
    max_resolution = 5.0

    while min_resolution <= max_resolution:
        resolution = (min_resolution + max_resolution) / 2
        sc.tl.leiden(
            cluster_data, resolution=resolution, key_added=leiden_col, flavor="igraph", n_iterations=2, directed=False
        )
        n_found = cluster_data.obs[leiden_col].nunique()

        if n_found < target_subclusters:
            min_resolution = resolution + 0.01
        elif n_found > target_subclusters:
            max_resolution = resolution - 0.01
        else:
            break

    cluster_sizes = cluster_data.obs[leiden_col].value_counts()
    n_found = len(cluster_sizes)
    if n_found > target_subclusters:
        top_clusters = cluster_sizes.nlargest(target_subclusters).index.tolist()
        label_map = {cl: str(i) for i, cl in enumerate(top_clusters)}
        for cl in cluster_sizes.index:
            if cl not in label_map:
                label_map[cl] = "0"
        cluster_data.obs[leiden_col] = cluster_data.obs[leiden_col].astype(str).map(label_map)
        print(f"  {cell_type}: {n_found} clusters found, consolidated to {target_subclusters}")
    else:
        label_map = {cl: str(i) for i, cl in enumerate(cluster_sizes.index)}
        cluster_data.obs[leiden_col] = cluster_data.obs[leiden_col].astype(str).map(label_map)
        print(f"  {cell_type}: {n_found} subclusters at resolution {resolution:.3f}")

    subcluster_labels = cluster_data.obs[leiden_col].astype(str)
    subcluster_labels = [f"{cell_type}_sub{label}" for label in subcluster_labels]
    adata.obs.loc[cluster_data.obs.index, "subtype"] = subcluster_labels


def _load_xenium_spatial(xenium_path, spatial_bounds):
    """
    Load the Xenium spatial dataset and subset to the configured spatial region.

    Args:
        xenium_path: Path to the Xenium h5ad file.
        spatial_bounds: Tuple of ((x_min, x_max), (y_min, y_max)).

    Returns
    -------
        The spatially subset AnnData with only the expected canonical cell types.
    """
    print(f"Loading Xenium spatial data from {xenium_path}...")
    adata = ad.read_h5ad(xenium_path)

    x_lim, y_lim = spatial_bounds
    coords = adata.obsm["spatial"]
    mask = (
        (coords[:, 0] >= x_lim[0])
        & (coords[:, 0] <= x_lim[1])
        & (coords[:, 1] >= y_lim[0])
        & (coords[:, 1] <= y_lim[1])
    )
    adata = adata[mask].copy()

    # Rename and filter clusters to match canonical cell types
    rename_map = {
        "Macrophages_1": "Macrophages",
        "Macrophages_2": "Macrophages",
        "DCIS 1": "DCIS",
        "DCIS 2": "DCIS",
        "Myoepi_ACTA2+": "Myoepi",
        "Myoepi_KRT15+": "Myoepi",
        "CD4+_T_Cells": "T_Cells",
        "CD8+_T_Cells": "T_Cells",
    }
    drop_clusters = [
        "Mast_Cells",
        "Perivascular-Like",
        "Unlabeled",
        "LAMP3+_DCs",
        "IRF7+_DCs",
        "Stromal_&_T_Cell_Hybrid",
        "Prolif_Invasive_Tumor",
        "T_Cell_&_Tumor_Hybrid",
    ]
    adata.obs["Cluster"] = adata.obs["Cluster"].astype(str).replace(rename_map).astype("category")
    adata = adata[~adata.obs["Cluster"].isin(drop_clusters)].copy()
    adata = adata[adata.obs["Cluster"].isin(CELL_TYPES)].copy()
    adata.obs["Cluster"] = adata.obs["Cluster"].cat.remove_unused_categories()

    print(f"Xenium spatial subset shape: {adata.shape}")
    print(adata.obs["Cluster"].value_counts())
    return adata


def _create_interaction_df(gt_interactions, receptor_clusters):
    """
    Build the interaction DataFrame from ground truth interaction configs.

    Args:
        gt_interactions: List of dicts with keys: receiver, sender, length_scale.
        receptor_clusters: List of receiver cell type names (for neutral rows).

    Returns
    -------
        The interaction DataFrame.
    """
    interaction_dicts = [
        {
            "receptor_cell": receptor,
            "receptor_subtype": f"{receptor}_sub0",
            "interaction_type": "neutral",
        }
        for receptor in receptor_clusters
    ]
    for interaction in gt_interactions:
        interaction_dicts.append(
            {
                "receptor_cell": interaction["receiver"],
                "sender_cell": interaction["sender"],
                "receptor_subtype": f"{interaction['receiver']}_sub1",
                "radius_of_effect": interaction["length_scale"],
                "interaction_type": "interaction",
            }
        )
    return pd.DataFrame(interaction_dicts)


def _determine_spatial_subtypes(spatial_subset, interaction_df, spatial_key="spatial", cluster_key="Cluster"):
    """
    Assign a subtype to each spatial position based on proximity to sender cells.

    Args:
        spatial_subset: AnnData with spatial coordinates in obsm[spatial_key].
        interaction_df: DataFrame produced by _create_interaction_df.
        spatial_key: Key in obsm for spatial coordinates.
        cluster_key: obs column containing cell type labels.

    Returns
    -------
        DataFrame with columns: X, Y, Cell_Type, Subtype.
    """
    coords = spatial_subset.obsm[spatial_key]
    spatial_data = pd.DataFrame(coords, columns=["X", "Y"], index=spatial_subset.obs_names)
    spatial_data["Cell_Type"] = spatial_subset.obs[cluster_key].astype(str).values
    spatial_data["Subtype"] = spatial_data["Cell_Type"].astype(str) + "_sub0"

    neutral_types = dict(
        zip(
            interaction_df[interaction_df["interaction_type"] == "neutral"]["receptor_cell"],
            interaction_df[interaction_df["interaction_type"] == "neutral"]["receptor_subtype"],
            strict=False,
        )
    )
    mapped = spatial_data["Cell_Type"].map(neutral_types)
    spatial_data["Subtype"] = mapped.fillna(spatial_data["Subtype"])

    for rule in interaction_df[interaction_df["interaction_type"] == "interaction"].to_dict(orient="records"):
        if rule["receptor_cell"] not in neutral_types:
            continue
        receptor_cells = spatial_data[
            (spatial_data["Cell_Type"] == rule["receptor_cell"])
            & (spatial_data["Subtype"] == neutral_types[rule["receptor_cell"]])
        ]
        sender_cells = spatial_data[spatial_data["Cell_Type"] == rule["sender_cell"]]
        if receptor_cells.empty or sender_cells.empty:
            continue
        for idx, receptor_cell in receptor_cells.iterrows():
            distances = np.sqrt(
                (sender_cells["X"] - receptor_cell["X"]) ** 2 + (sender_cells["Y"] - receptor_cell["Y"]) ** 2
            )
            if (distances <= rule["radius_of_effect"]).any():
                spatial_data.at[idx, "Subtype"] = rule["receptor_subtype"]

    return spatial_data


def _sample_flex_by_subtype(flex_adata, spatial_data, scvi_model, rng=None):
    """
    Generate novel expression for each spatial position via scVI posterior predictive sampling.

    Args:
        flex_adata: Preprocessed Flex AnnData (the scVI training data).
        spatial_data: DataFrame with columns X, Y, Cell_Type, Subtype.
        scvi_model: Trained scVI model for posterior sampling.
        rng: NumPy random Generator for reproducibility.

    Returns
    -------
        Concatenated AnnData with generated counts, spatial coordinates,
        and obs columns: cell_type, subtype, flex_subtype.
    """
    rng = rng or np.random.default_rng()

    # Build pools: flex subtype -> array of cell indices in flex_adata
    pools = {
        flex_subtype: np.where(flex_adata.obs["subtype"] == flex_subtype)[0]
        for flex_subtype in flex_adata.obs["subtype"].unique()
    }

    sampled_adatas = []
    for target_subtype in spatial_data["Subtype"].unique():
        flex_subtype = target_subtype  # 1:1 mapping for breast cancer
        if flex_subtype not in pools:
            cell_type = target_subtype.rsplit("_sub", 1)[0]
            flex_subtype = f"{cell_type}_sub0"
            if flex_subtype not in pools:
                print(f"Warning: No Flex pool for {target_subtype}, skipping.")
                continue

        subtype_mask = spatial_data["Subtype"] == target_subtype
        num_samples = subtype_mask.sum()
        if num_samples == 0:
            continue

        pool = pools[flex_subtype]
        sampled_idx = rng.choice(pool, num_samples, replace=True)

        source_adata = flex_adata[sampled_idx].copy()
        generated_counts = scvi_model.posterior_predictive_sample(source_adata, n_samples=1)

        # Convert pydata sparse GCXS (n_cells, 1, n_genes) -> scipy CSR
        if hasattr(generated_counts, "todense"):
            generated_counts = np.array(generated_counts.todense())
        if generated_counts.ndim == 3:
            generated_counts = generated_counts[:, 0, :]
        generated_counts = sp_sparse.csr_matrix(generated_counts)

        sampled_adata = ad.AnnData(X=generated_counts, var=flex_adata.var.copy())
        if is_count_data(sampled_adata.X):
            sampled_adata.layers["counts"] = sampled_adata.X.copy()
            sc.pp.normalize_total(sampled_adata)
            sc.pp.log1p(sampled_adata)

        sampled_adata.obsm["spatial"] = spatial_data.loc[subtype_mask, ["X", "Y"]].values
        sampled_adata.obs["spatial_id"] = spatial_data.loc[subtype_mask].index.values
        sampled_adata.obs["cell_type"] = spatial_data.loc[subtype_mask, "Cell_Type"].values
        sampled_adata.obs["subtype"] = target_subtype
        sampled_adata.obs["flex_subtype"] = flex_subtype
        sampled_adatas.append(sampled_adata)

    sampled = sc.concat(sampled_adatas, axis=0)
    sampled.obs_names_make_unique()
    return sampled


def _create_train_test_split(adata):
    """
    Assign train/test labels based on spatial coordinates.

    Args:
        adata: AnnData with spatial coordinates in obsm["spatial"].

    Returns
    -------
        AnnData with train_test_split obs column ("train" or "test") and
        cv_fold obs column (0/1/2 for spatially contiguous train folds, -1 for test cells).
    """
    coords = adata.obsm["spatial"]
    test_mask = (
        (coords[:, 0] >= TEST_X_MIN)
        & (coords[:, 0] <= TEST_X_MAX)
        & (coords[:, 1] >= TEST_Y_MIN)
        & (coords[:, 1] <= TEST_Y_MAX)
    )
    adata.obs["train_test_split"] = "train"
    adata.obs.loc[test_mask, "train_test_split"] = "test"

    test_pct = test_mask.sum() / len(adata) * 100
    print(f"Train: {(~test_mask).sum()} cells ({100 - test_pct:.1f}%)")
    print(f"Test:  {test_mask.sum()} cells ({test_pct:.1f}%)")

    # Assign spatially contiguous CV folds to training cells.
    # Folds are equal-width bands along the X axis (by percentile of training X coords).
    train_mask = ~test_mask
    x_coords = coords[:, 0]
    train_x = x_coords[train_mask]
    fold_edges = np.percentile(train_x, [0, 33.33, 66.67, 100])

    adata.obs["cv_fold"] = -1
    for fold_id, (lo, hi) in enumerate(zip(fold_edges[:-1], fold_edges[1:], strict=False)):
        if fold_id == 2:
            fold_mask = train_mask & (x_coords >= lo)
        else:
            fold_mask = train_mask & (x_coords >= lo) & (x_coords < hi)
        adata.obs.loc[fold_mask, "cv_fold"] = fold_id

    fold_counts = adata.obs["cv_fold"].value_counts().sort_index()
    print("CV fold sizes:", dict(fold_counts.items()))

    return adata


CELL_TYPE_PALETTE = {
    "T_Cells": "#56B4E9",
    "B_Cells": "#009E4E",
    "Invasive_Tumor": "#cf4242",
    "DCIS": "#E69F00",
    "Macrophages": "#de692a",
    "Endothelial": "#277987",
    "Stromal": "#968253",
    "Myoepi": "#823960",
}


def plot_spatial_distribution(adata, figure_path):
    """
    Save a two-panel spatial overview of the semi-synthetic dataset.

    Panel 1 – cell types: training cells as filled circles, test cells as ×.
    Panel 2 – interaction subtypes: sub1 (interacting) bright/outlined, sub0 dim.

    Args:
        adata: The generated AnnData with obsm["spatial"], obs["cell_type"],
            obs["subtype"], and obs["train_test_split"].
        figure_path: File path to write the PNG figure.
    """
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)

    coords = adata.obsm["spatial"]
    cell_types = adata.obs["cell_type"].astype(str).values
    subtypes = adata.obs["subtype"].astype(str).values
    splits = adata.obs["train_test_split"].astype(str).values
    test_mask = splits == "test"

    fig, axes = plt.subplots(1, 2, figsize=(26, 9))

    # ── Panel 1: cell types ──────────────────────────────────────────────────
    ax = axes[0]
    for ct, color in CELL_TYPE_PALETTE.items():
        ct_mask = cell_types == ct
        if ct_mask.sum() == 0:
            continue
        train_ct = ct_mask & ~test_mask
        test_ct = ct_mask & test_mask
        ax.scatter(
            coords[train_ct, 0],
            coords[train_ct, 1],
            c=color,
            s=4,
            alpha=0.6,
            linewidths=0,
            label=ct,
        )
        if test_ct.sum() > 0:
            ax.scatter(
                coords[test_ct, 0],
                coords[test_ct, 1],
                c=color,
                s=5,
                alpha=0.7,
                marker="x",
                linewidths=0.5,
            )

    if test_mask.sum() > 0:
        tx, ty = coords[test_mask, 0], coords[test_mask, 1]
        pad = 20
        rect = mpatches.Rectangle(
            (tx.min() - pad, ty.min() - pad),
            tx.max() - tx.min() + 2 * pad,
            ty.max() - ty.min() + 2 * pad,
            linewidth=1.5,
            edgecolor="black",
            facecolor="none",
            linestyle="--",
            label="Test region",
        )
        ax.add_patch(rect)

    ax.set_title("Cell types  (train = filled, test = ×)", fontsize=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", markerscale=2.5, fontsize=8)

    # ── Panel 2: interaction subtypes ────────────────────────────────────────
    ax = axes[1]
    for ct, color in CELL_TYPE_PALETTE.items():
        sub0 = subtypes == f"{ct}_sub0"
        sub1 = subtypes == f"{ct}_sub1"
        if sub0.sum() > 0:
            ax.scatter(coords[sub0, 0], coords[sub0, 1], c=color, s=3, alpha=0.25, linewidths=0)
        if sub1.sum() > 0:
            ax.scatter(
                coords[sub1, 0],
                coords[sub1, 1],
                c=color,
                s=6,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.3,
                label=f"{ct} sub1",
            )

    ax.set_title(
        "Subtypes  (bright+outlined = sub1 / interacting,  dim = sub0 / neutral)",
        fontsize=11,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", markerscale=2.5, fontsize=8)

    plt.suptitle(f"Realistic semi-synthetic dataset — {adata.n_obs} cells", fontsize=13)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved spatial distribution figure to {figure_path}")


def main():
    """Generate the breast cancer semi-synthetic spatial dataset."""
    select_gpu()
    output_path = snakemake.output[0]  # noqa: F821

    flex_h5_path = snakemake.input.flex_h5_path  # noqa: F821
    annot_path = snakemake.input.annot_path  # noqa: F821
    xenium_path = snakemake.input.xenium_path  # noqa: F821
    scvi_model_dir = snakemake.params.get("scvi_model_dir", None)  # noqa: F821

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    scvi.settings.seed = SEED

    generate_realistic_dataset(flex_h5_path, annot_path, xenium_path, output_path, scvi_model_dir=scvi_model_dir)
    print(f"Successfully wrote output to {output_path}")

    figure_path = (
        f"results/{snakemake.wildcards.dataset}_{snakemake.wildcards.seed}"  # noqa: F821
        f"/figures/spatial_distribution.png"
    )
    adata = ad.read_h5ad(output_path)
    plot_spatial_distribution(adata, figure_path)


if __name__ == "__main__":
    main()
