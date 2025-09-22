# %% Import libraries
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import seaborn as sns
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from scipy.stats import mode
from datetime import date
from sklearn.neighbors import NearestNeighbors

from amici.tools import is_count_data

# %% Read data and add annotations if needed and combine cell type labels for training
proseg_data = True
data_path = f"data/xenium_sample1/xenium_sample1.h5ad"
labels_key = "predictions_resolvi_proseg"
adata = ad.read_h5ad(data_path)

# %% 
# Add counts and spatial coordinates
if "counts" not in adata.layers:
    if not is_count_data(adata.X):
        raise ValueError("AnnData object does not contain count data")
    adata.layers["counts"] = adata.X.copy()
if proseg_data:
    adata.obsm["spatial"] = pd.DataFrame({"X": adata.obs["x"], "Y": adata.obs["y"]})
else:
    adata.obsm["spatial"] = pd.DataFrame({"X": adata.obsm["spatial"][:, 0], "Y": adata.obsm["spatial"][:, 1]}, index=adata.obs_names)

# %% 
# Group cell types
group_cells = True
labels_key_grouped = "celltype_train_grouped"
if group_cells:
    cell_label_map = {
        "Invasive_Tumor": "Invasive_Tumor",
        "Prolif_Invasive_Tumor": "Invasive_Tumor",
    }

    adata.obs[labels_key_grouped] = adata.obs[labels_key].copy()
    for key, val in cell_label_map.items():
        adata.obs[labels_key_grouped].replace(key, val, inplace=True)
    print("New cell types after grouping:")
    print(list(adata.obs[labels_key_grouped].unique()))
else:
    adata.obs[labels_key_grouped] = adata.obs[labels_key].copy()

# %% 
# Correct for cell type labels for DCIS 1 and DCIS 2 for each replicate
def correct_dcis_labels(adata, labels_key, left_cutoff, right_cutoff, rep="0"):
    rep_mask = adata.obs["sample"] == rep
    dcis_1_mask = (adata.obs[labels_key] == "DCIS_1") & (adata.obsm["spatial"]["X"] < left_cutoff) & rep_mask
    adata.obs.loc[dcis_1_mask, labels_key] = "DCIS_2"

    dcis_2_mask = (adata.obs[labels_key] == "DCIS_2") & (adata.obsm["spatial"]["X"] > right_cutoff) & rep_mask
    adata.obs.loc[dcis_2_mask, labels_key] = "DCIS_1"

    center_mask = (adata.obs[labels_key].isin(["DCIS_1", "DCIS_2"])) & \
                    (adata.obsm["spatial"]["X"] >= left_cutoff) & \
                    (adata.obsm["spatial"]["X"] <= right_cutoff) & \
                    rep_mask
    adata_center = adata[center_mask]

    nn = NearestNeighbors(n_neighbors=10, metric="euclidean").fit(adata_center.obsm["spatial"])
    _, nn_idx = nn.kneighbors(adata_center.obsm["spatial"])
    nn_labels = adata_center.obs[labels_key].values[nn_idx].to_numpy()
    nn_num_labels = np.ones_like(nn_labels, dtype=int)
    nn_num_labels[nn_labels == "DCIS_2"] = 2

    max_vote_num_labels = mode(nn_num_labels, axis=1).mode.flatten()
    max_vote_labels = np.repeat("DCIS_1", max_vote_num_labels.shape[0])
    max_vote_labels[max_vote_num_labels == 2] = "DCIS_2"
    adata.obs.loc[center_mask, labels_key] = max_vote_labels
    return adata

adata = correct_dcis_labels(adata, labels_key_grouped, 4000, 6500, rep="0")
adata = correct_dcis_labels(adata, labels_key_grouped, 4000, 5000, rep="1")

# %% 
# Visualize histogram of total counts per cell
plt.figure(figsize=(8, 6))
total_counts = np.asarray(adata.layers["counts"].sum(axis=1))

plt.hist(total_counts, bins=50)
plt.xlabel('Total Counts per Cell')
plt.ylabel('Frequency')
plt.title('Histogram of Total Counts per Cell')

plt.savefig(f"figures/xenium_sample1_proseg_total_counts_histogram.png")
plt.show()

# Visualize total counts detected per gene
plt.figure(figsize=(8, 6))
total_counts = np.asarray(adata.layers["counts"].sum(axis=0)).reshape(-1, 1)

plt.hist(total_counts, bins=100)
plt.xlim(0, 150000)
plt.xlabel('Total Counts per Gene')
plt.ylabel('Frequency')
plt.title('Histogram of Total Counts per Gene')

plt.savefig("figures/xenium_sample1_proseg_total_counts_gene_histogram.png")
plt.show()

# Visualize histogram of number of cells with nonzero counts for each gene
plt.figure(figsize=(8, 6))
nonzero_cells = np.asarray(adata.layers["counts"].toarray() > 0).sum(axis=0)
plt.hist(nonzero_cells, bins=100)
plt.xlabel('Number of Cells with Nonzero Counts')
plt.ylabel('Frequency')
plt.title('Histogram of Number of Cells with Nonzero Counts for Each Gene')
plt.savefig("figures/xenium_sample1_proseg_nonzero_counts_per_gene_histogram.png")
plt.show()

# %% 
# Filter highly variable genes
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=250, subset=True)

# %% 
# Filter cells with low counts
sc.pp.filter_cells(adata, min_counts=50)

# %% 
# Filter the cells based on probabilities of cell type annotations
adata = adata[adata.obs["predicted_celltype_prob_resolvi"] >= 0.5]

# %% Normalize data
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# %% Select subset of cell data for training
cell_subset = [
    "CD8+_T_Cells", 
    "DCIS_1",
    "DCIS_2",
    "CD4+_T_Cells", 
    "Macrophages_1",
    "Macrophages_2",
    "Myoepi_ACTA2+", 
    "Myoepi_KRT15+", 
    "Invasive_Tumor", 
    "IRF7+_DCs",
    "LAMP3+_DCs",
    "B_Cells",
    "Mast_Cells",
    "Perivascular-Like",
    "Endothelial",
]
adata_sub = adata[adata.obs[labels_key] != "Unlabeled"].copy()
adata_sub = adata_sub[adata_sub.obs[labels_key_grouped].isin(cell_subset)].copy()
print("Subset of cells used for training:")
print(len(adata_sub))

# %% 
# Shift the spatial data for replicate 2 by large number
rep_2_mask = adata_sub.obs["sample"] == "1"
adata_sub.obsm["spatial"].loc[rep_2_mask, "X"] = 10000 + adata_sub.obsm["spatial"].loc[rep_2_mask, "X"].copy()

# %% Visualize spatial distribution of data
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
    # "Stromal": "#968253",
    "B_Cells": "#c5a9e8",
    "Mast_Cells": "#947b79",
    "Perivascular-Like": "#872727",
    "Endothelial": "#277987",
}
def visualize_spatial_distribution(adata, labels_key, dataset, x_lim=None, y_lim=None):
    plt.figure(figsize=(20, 6))
    plot_df = adata.obsm["spatial"].copy()
    plot_df[labels_key] = adata.obs[labels_key]
    sns.scatterplot(
        plot_df, x="X", y="Y", hue=labels_key, alpha=0.7, s=8, palette=CELL_TYPE_PALETTE
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"{dataset} Spatial plot after filtering low counts")
    plt.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, markerscale=2
    )
    if x_lim is not None:
        plt.xlim(0, x_lim)
    if y_lim is not None:
        plt.ylim(0, y_lim)
    plt.tight_layout()
    plt.savefig(f"figures/xenium_sample1_proseg_spatial_distribution_{dataset}.png")
    plt.show()

visualize_spatial_distribution(adata_sub, labels_key_grouped, "Proseg Xenium")

# %% 
# Add the cell radiuses to the adata object for both replicates
adata_sub.obs["cell_radius"] = 0.0

rep1_cells_path = "data/xenium_sample1/cells_rep1.csv"
cells_rep1 = pd.read_csv(rep1_cells_path)
cells_rep1 = cells_rep1[["cell_area"]]
cells_rep1["adata_idx"] = [f"query_{idx + 1}_0" for idx in cells_rep1.index]

rep2_cells_path = "data/xenium_sample1/cells_rep2.csv"
cells_rep2 = pd.read_csv(rep2_cells_path)
cells_rep2 = cells_rep2[["cell_area"]]
cells_rep2["adata_idx"] = [f"query_{idx + 1}_1" for idx in cells_rep2.index]

cell_radius_df = pd.concat([cells_rep1, cells_rep2])
cell_radius_df["cell_radius"] = np.sqrt(cell_radius_df["cell_area"] / np.pi)
cell_radius_df.set_index("adata_idx", inplace=True)
adata_sub.obs["cell_radius"] = adata_sub.obs_names.map(cell_radius_df["cell_radius"])

# %%
# Select a slide of the data to use for the held out test set
adata_sub.obs["train_test_split"] = "train"

# Select slide from replicate 1
rep1_mask = adata_sub.obs["sample"] == "0"
spatial_coords = adata_sub.obsm["spatial"].copy()
is_test_rep1 = (spatial_coords["X"] > 1600) & (spatial_coords["X"] < 8000) & \
            (spatial_coords["Y"] > 0) & (spatial_coords["Y"] < 800) & rep1_mask
adata_sub.obs.loc[is_test_rep1.to_numpy(), 'train_test_split'] = 'test'

rep2_mask = adata_sub.obs["sample"] == "1"
is_test_rep2 = (spatial_coords["X"] > 11600) & (spatial_coords["X"] < 18000) & \
            (spatial_coords["Y"] > 0) & (spatial_coords["Y"] < 800) & rep2_mask
adata_sub.obs.loc[is_test_rep2.to_numpy(), 'train_test_split'] = 'test'

print(f"Number of total cells: {len(adata_sub)}")
adata_train = adata_sub[adata_sub.obs['train_test_split'] == 'train']
print(f"Number of training cells: {len(adata_train)}")
adata_test = adata_sub[adata_sub.obs['train_test_split'] == 'test']
print(f"Number of test cells: {len(adata_test)}")

visualize_spatial_distribution(adata_train, labels_key_grouped, "Proseg Xenium Training Set")
visualize_spatial_distribution(adata_test, labels_key_grouped, "Proseg Xenium Test Set", x_lim=adata_sub.obsm["spatial"]["X"].max(), y_lim=adata_sub.obsm["spatial"]["Y"].max())

# %%
# Write out the data with the current date and train-test splits
today = date.today()
adata_sub.write_h5ad(f"data/xenium_sample1/xenium_sample1_filtered_{today}.h5ad")

adata_train.write_h5ad(f"data/xenium_sample1/xenium_sample1_filtered_train_{today}.h5ad")
adata_test.write_h5ad(f"data/xenium_sample1/xenium_sample1_filtered_test_{today}.h5ad")

# %%
