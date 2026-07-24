# %% Import libraries
from datetime import date
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from amici.tools import is_count_data

# %% Paths and parameters
DATA_DIR = Path("atera_breast/data")
FIGURE_DIR = Path("atera_breast/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MATRIX_PATH = DATA_DIR / "cell_feature_matrix.h5"
CELLS_PATH = DATA_DIR / "cells.parquet"
CELL_GROUPS_PATH = DATA_DIR / "WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv"

labels_key = "cell_group"
labels_key_grouped = "celltype_train_grouped"
n_top_genes = 5000

# %% Read Atera data and merge vendor cell-group annotations
if not MATRIX_PATH.exists() or not CELLS_PATH.exists():
    raise FileNotFoundError(
        "Missing Atera matrix/cell metadata. Download and extract "
        "cell_feature_matrix.h5 and cells.parquet into atera_breast/data."
    )

adata = sc.read_10x_h5(MATRIX_PATH)
cells = pd.read_parquet(CELLS_PATH)
cell_groups = pd.read_csv(CELL_GROUPS_PATH)

if "cell_id" not in cells.columns:
    cells = cells.reset_index().rename(columns={"index": "cell_id"})
cells = cells.set_index("cell_id")
cell_groups = cell_groups.set_index("cell_id")

adata = adata[cells.index.intersection(adata.obs_names)].copy()
adata.obs = adata.obs.join(cells, how="left")
adata.obs[labels_key] = adata.obs_names.map(cell_groups["group"])
adata.obs["cell_group_color"] = adata.obs_names.map(cell_groups["color"])

print(f"Number of cells before QC: {adata.n_obs}")
print(f"Number of genes before QC: {adata.n_vars}")
print(f"Missing cell-group annotations: {adata.obs[labels_key].isna().sum()}")
print("Cell-group annotations:")
print(adata.obs[labels_key].value_counts(dropna=False))

# %% Add counts and spatial coordinates
if "counts" not in adata.layers:
    if not is_count_data(adata.X):
        raise ValueError("AnnData object does not contain count data")
    adata.layers["counts"] = adata.X.copy()

coord_candidates = [("x_centroid", "y_centroid"), ("cell_centroid_x", "cell_centroid_y"), ("x", "y")]
coord_cols = next(((x, y) for x, y in coord_candidates if x in adata.obs and y in adata.obs), None)
if coord_cols is None:
    raise KeyError(f"Could not find spatial centroid columns in cells.parquet: {adata.obs.columns.tolist()}")

adata.obsm["spatial"] = pd.DataFrame(
    {
        "X": adata.obs[coord_cols[0]].astype(float),
        "Y": adata.obs[coord_cols[1]].astype(float),
    },
    index=adata.obs_names,
)

# %% Group cell types for training
cell_label_map = {
    "11q13 High Grade DCIS Cells": "High_Grade_DCIS",
    "11q13 High Grade DCIS Cells (Mitotic)": "High_Grade_DCIS",
    "11q13 High Grade DCIS Tumor Cells (G1/S)": "High_Grade_DCIS",
    "Basal-like Structured DCIS Cells": "Basal_DCIS",
    "Luminal-like Amorphous DCIS Cells": "Luminal_DCIS",
    "CAFs, Low Grade DCIS Associated": "CAFs",
    "CAFs, High Grade DCIS Associated": "CAFs",
    "CXCL14+ Fibroblasts": "Fibroblasts",
    "T Lymphocytes": "T_Cells",
    "Endothelial Cells": "Endothelial",
    "Myoepithelial Cells": "Myoepithelial",
    "Myeloid Cells": "Myeloid",
    "Mast Cells": "Mast_Cells",
    "Plasma Cells": "Plasma_Cells",
    "Plasma & Mast Cell Mixture": "Plasma_Mast_Mixture",
}

adata.obs[labels_key_grouped] = adata.obs[labels_key].replace(cell_label_map)
print("Grouped cell types:")
print(adata.obs[labels_key_grouped].value_counts(dropna=False))

# %% Visualize histogram of total counts per cell
total_counts = np.asarray(adata.layers["counts"].sum(axis=1)).ravel()
plt.figure(figsize=(8, 6))
plt.hist(total_counts, bins=80)
plt.xlabel("Total Counts per Cell")
plt.ylabel("Frequency")
plt.title("Atera Total Counts per Cell")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "atera_total_counts_histogram.png", dpi=300)
plt.show()

print(f"Median total counts per cell: {np.median(total_counts):.1f}")

# %% Visualize total counts detected per gene
gene_counts = np.asarray(adata.layers["counts"].sum(axis=0)).ravel()
plt.figure(figsize=(8, 6))
plt.hist(gene_counts, bins=100)
plt.xlabel("Total Counts per Gene")
plt.ylabel("Frequency")
plt.title("Atera Total Counts per Gene")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "atera_total_counts_gene_histogram.png", dpi=300)
plt.show()

# %% Filter cells and genes
sc.pp.filter_cells(adata, min_counts=50)
sc.pp.filter_genes(adata, min_cells=10)
adata = adata[adata.obs[labels_key].notna()].copy()

print(f"Number of cells after QC and annotation filter: {adata.n_obs}")
print(f"Number of genes after QC: {adata.n_vars}")

# %% Filter highly variable genes
sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=min(n_top_genes, adata.n_vars),
    layer="counts",
    subset=True,
)
print(f"Number of HVGs retained: {adata.n_vars}")

# %% Normalize data
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# %% Select subset of cell data for training
cell_subset = [
    "High_Grade_DCIS",
    "Basal_DCIS",
    "Luminal_DCIS",
    "CAFs",
    "Fibroblasts",
    "Endothelial",
    "Myoepithelial",
    "Pericytes",
    "T_Cells",
    "Dendritic Cells",
    "Macrophages",
    "Myeloid",
    "Mast_Cells",
    "Plasma_Cells",
    "Plasma_Mast_Mixture",
    "Apocrine Cells",
]

adata_sub = adata[adata.obs[labels_key_grouped].isin(cell_subset)].copy()
print("Subset of cells used for training:")
print(len(adata_sub))
print(adata_sub.obs[labels_key_grouped].value_counts())

# %% Visualize spatial distribution of data
palette_df = adata_sub.obs[[labels_key_grouped, "cell_group_color"]].dropna().drop_duplicates(labels_key_grouped)
cell_type_palette = dict(zip(palette_df[labels_key_grouped], palette_df["cell_group_color"]))


def visualize_spatial_distribution(adata, labels_key, dataset, x_lim=None, y_lim=None):
    plt.figure(figsize=(12, 10))
    plot_df = adata.obsm["spatial"].copy()
    plot_df[labels_key] = adata.obs[labels_key].values
    sns.scatterplot(
        plot_df,
        x="X",
        y="Y",
        hue=labels_key,
        alpha=0.7,
        s=2,
        linewidth=0,
        palette=cell_type_palette,
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"{dataset} Spatial Plot")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, markerscale=5)
    if x_lim is not None:
        plt.xlim(0, x_lim)
    if y_lim is not None:
        plt.ylim(0, y_lim)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"atera_spatial_distribution_{dataset.replace(" ", "_").lower()}.png", dpi=300)
    plt.show()


visualize_spatial_distribution(adata_sub, labels_key_grouped, "Atera Breast Filtered")

# %% Add cell radiuses to the adata object
if "cell_area" in adata_sub.obs:
    adata_sub.obs["cell_radius"] = np.sqrt(adata_sub.obs["cell_area"].astype(float) / np.pi)
else:
    adata_sub.obs["cell_radius"] = 0.0

# %% Select a contiguous spatial slice for the held out test set
adata_sub.obs["train_test_split"] = "train"
spatial_coords = adata_sub.obsm["spatial"]
y_cutoff = np.quantile(spatial_coords["Y"], 0.10)
is_test = spatial_coords["Y"] <= y_cutoff
adata_sub.obs.loc[is_test.to_numpy(), "train_test_split"] = "test"

adata_train = adata_sub[adata_sub.obs["train_test_split"] == "train"].copy()
adata_test = adata_sub[adata_sub.obs["train_test_split"] == "test"].copy()

print(f"Test slice y <= {y_cutoff:.2f}")
print(f"Number of total cells: {len(adata_sub)}")
print(f"Number of training cells: {len(adata_train)}")
print(f"Number of test cells: {len(adata_test)}")
print(f"Test fraction: {len(adata_test) / len(adata_sub):.3f}")

visualize_spatial_distribution(adata_train, labels_key_grouped, "Atera Breast Training Set")
visualize_spatial_distribution(
    adata_test,
    labels_key_grouped,
    "Atera Breast Test Set",
    x_lim=adata_sub.obsm["spatial"]["X"].max(),
    y_lim=adata_sub.obsm["spatial"]["Y"].max(),
)

# %% Write out the data with the current date and train-test splits
today = date.today()
adata_sub.write_h5ad(DATA_DIR / f"atera_breast_filtered_{today}.h5ad")
adata_train.write_h5ad(DATA_DIR / f"atera_breast_filtered_train_{today}.h5ad")
adata_test.write_h5ad(DATA_DIR / f"atera_breast_filtered_test_{today}.h5ad")

# %%
