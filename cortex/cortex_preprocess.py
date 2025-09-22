"""
This script preprocesses the cortex data.

The data was downloaded from the following link:
https://download.brainimagelibrary.org/cf/1c/cf1c1a431ef8d021/processed_data/
"""

# %%
import os
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# %%
data_dir = "./data"

# %%
# Load the data
adata = sc.read_h5ad(os.path.join(data_dir, "counts.h5ad"))
cell_labels = pd.read_csv(os.path.join(data_dir, "cell_labels.csv"), index_col=0)
segmented_cells = pd.concat(
    [
        pd.read_csv(
            os.path.join(data_dir, f"segmented_cells_mouse1sample{i}.csv"),
            index_col=0,
        )
        for i in range(1, 7)
    ]
)
# %%
obs_df = cell_labels.join(segmented_cells.drop(columns=["slice_id"]), how="inner")

# %%
sample_id = "mouse1_sample4"
sample_obs_df = obs_df[obs_df["sample_id"] == sample_id]


# %%
def compute_centroid(x_coords, y_coords):
    """
    Compute the centroid of a polygon given its boundary coordinates.

    Args:
        x_coords: List or array of x coordinates of the boundary
        y_coords: List or array of y coordinates of the boundary

    Returns:
        tuple: (centroid_x, centroid_y)
    """
    if isinstance(x_coords, float):
        x_coords = np.array([x_coords])
    if isinstance(y_coords, float):
        y_coords = np.array([y_coords])
    n = len(x_coords)
    if n == 0:
        return np.nan, np.nan

    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)

    return centroid_x, centroid_y


# Extract boundary coordinates
boundary_x = sample_obs_df["boundaryX_z3"].apply(
    lambda x: np.array(eval(x)) if isinstance(x, str) else np.array([])
)
boundary_y = sample_obs_df["boundaryY_z3"].apply(
    lambda y: np.array(eval(y)) if isinstance(y, str) else np.array([])
)

# Calculate centroids for each cell
centroids = [
    compute_centroid(boundary_x[i], boundary_y[i]) for i in range(len(boundary_x))
]
sample_obs_df["centroid_x"] = [c[0] for c in centroids]
sample_obs_df["centroid_y"] = [c[1] for c in centroids]

# %%
import seaborn as sns

sns.scatterplot(
    data=sample_obs_df,
    x="centroid_x",
    y="centroid_y",
    hue="subclass",
    s=3,
)
# %%
sample_adata = adata[sample_obs_df.index].copy()
sample_adata.obs = sample_obs_df
sample_adata
# %%
sample_adata.layers["vol_normalized"] = sample_adata.X.copy()
sample_adata.X = sc.pp.normalize_total(
    sample_adata, layer="vol_normalized", target_sum=1e3, inplace=False
)["X"]
sc.pp.log1p(sample_adata)
# %%
plt.figure(figsize=(8, 6))
total_counts = np.asarray(sample_adata.X.sum(axis=0)).reshape(-1, 1)

os.makedirs("figures", exist_ok=True)
plt.hist(total_counts, bins=50)
plt.xlabel("Total Counts per Gene")
plt.ylabel("Frequency")
plt.title("Histogram of Total Counts per Gene")

plt.savefig("figures/total_counts_per_gene_histogram.png")
plt.show()

# Visualize histogram of number of cells with nonzero counts for each gene
plt.figure(figsize=(8, 6))
nonzero_cells = np.asarray(sample_adata.X > 0).sum(axis=0)
plt.hist(nonzero_cells, bins=50)
plt.xlabel("Number of Cells with Nonzero Counts")
plt.ylabel("Frequency")
plt.title("Histogram of Number of Cells with Nonzero Counts for Each Gene")
plt.savefig("figures/nonzero_counts_per_gene_histogram.png")
plt.show()

# %%
sample_adata.X = csr_matrix(sample_adata.X)
# %%
# set one slice as test
sample_adata.obs["in_test"] = sample_adata.obs["slice_id"] == "mouse1_slice180"
# %%
sns.scatterplot(
    x="centroid_x", y="centroid_y", hue="in_test", data=sample_adata.obs, s=3
)
plt.savefig("figures/train_test_split.pdf")
plt.show()
# %%
sample_adata.obs["in_test"] = sample_adata.obs["slice_id"] == "mouse1_slice180"
# %%
sns.scatterplot(
    x="centroid_x",
    y="centroid_y",
    hue="subclass",
    data=sample_adata.obs[sample_adata.obs["slice_id"] == "mouse1_slice162"],
    s=5,
)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
plt.savefig("figures/spatial_subclass_labeled_slice162.pdf")
plt.show()
# %%
from datetime import date

today = date.today()
sample_adata.write_h5ad(f"./data/cortex_processed_{today}.h5ad")
# %%
