# %%
import os

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import seaborn as sns

# %%
# R code to load and convert data
# remotes::install_github("satijalab/seurat", "seurat5", quiet = TRUE)

# if (!require("BiocManager", quietly = TRUE))
#      install.packages("BiocManager")
# BiocManager::install("zellkonverter")

# cosmx <- readRDS("/home/justinhong/data/cosmx/LiverDataReleaseSeurat_newUMAP.RDS")

# sce_obj <- as.SingleCellExperiment(cosmx, assay = c("RNA"))
# library(zellkonverter)
# writeH5AD(sce_obj, "/home/justinhong/data/cosmx/cosmx_liver.h5ad", X_name = 'counts')

# %%
adata = sc.read_h5ad("/home/justinhong/data/cosmx/cosmx_liver.h5ad")

# important obs attributes
# - cellType
# - x_slide_mm
# - y_slide_mm
# - slide_ID_numeric
# - Run_Tissue_name (label of numeric)
adata

# %%
# check annotation resolution
plot_df = adata.obsm["APPROXIMATEUMAP"].copy()
plot_df["cellType"] = adata.obs["cellType"]
plot_df["slide_ID_numeric"] = adata.obs["slide_ID_numeric"]
plt.figure(figsize=(12, 10))
scatter = plt.scatter(
    plot_df["APPROXIMATEUMAP_1"],
    plot_df["APPROXIMATEUMAP_2"],
    c=plot_df["slide_ID_numeric"],
    s=1,
    alpha=0.7,
)

# Create a legend
handles, labels = scatter.legend_elements(prop="colors")
plt.legend(
    handles,
    labels,
    title="Slide ID",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)

plt.title("UMAP of Cell Types")
plt.xlabel("UMAP_1")
plt.ylabel("UMAP_2")
plt.tight_layout()

plt.show()
# %%
# color by cell type
plt.figure(figsize=(12, 10))
scatter = plt.scatter(
    plot_df["APPROXIMATEUMAP_1"],
    plot_df["APPROXIMATEUMAP_2"],
    c=plot_df["cellType"].cat.codes,
    cmap="tab20",
    s=1,
    alpha=0.7,
)

# Create a legend
handles, labels = scatter.legend_elements(prop="colors")
plt.legend(
    handles,
    adata.obs["cellType"].cat.categories,
    title="Cell Type",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
plt.title("UMAP of Cell Types")
plt.xlabel("UMAP_1")
plt.ylabel("UMAP_2")
plt.tight_layout()
plt.show()
# %%
# check log counts distributions (if its really just log1p)
gene_idx = 8
fig, ax = plt.subplots(figsize=(12, 10))

# Create a DataFrame combining both log counts
count_transformations = {
    "Raw": adata.X[:, gene_idx].todense().A1,
    "StoredLog": adata.layers["logcounts"][:, gene_idx].todense().A1,
    "CalculatedLog": np.log1p(
        adata.X[:, gene_idx].todense() / adata.obs["nCount_RNA"].values[:, None] * 1e4
    ).A1.flatten(),
}
count_transformations["CenteredLog"] = (
    count_transformations["CalculatedLog"]
    - np.mean(count_transformations["CalculatedLog"])
) / np.std(count_transformations["CalculatedLog"])
# del count_transformations["StoredLog"]
# del count_transformations["CenteredLog"]
# print(count_transformations["CalculatedLog"].min())
# del count_transformations["CalculatedLog"]
# print(count_transformations["CenteredLog"].min())
df = pd.DataFrame(count_transformations)


# Melt the DataFrame to long format
df_melted = df.melt(var_name="Log Type", value_name="Log Counts")

# Plot with one histplot call
sns.histplot(
    data=df_melted,
    x="Log Counts",
    hue="Log Type",
    multiple="layer",
    bins=500,
    alpha=0.5,
    ax=ax,
)

ax.set_xlabel("Log Counts")
ax.set_ylabel("Density")
ax.set_title(f"Gene {adata.var.index[gene_idx]} Log Counts")
plt.show()

# %%
# save newly normalized adata
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
sc.pp.log1p(adata)
adata

# %%
# save spatial to obsm
adata.obsm["spatial"] = adata.obs[["x_slide_mm", "y_slide_mm"]].values
# %%
# save separate adata objects for each slide
for tissue_name in adata.obs["Run_Tissue_name"].cat.categories:
    tissue_adata = adata[adata.obs["Run_Tissue_name"] == tissue_name]
    if not os.path.exists(
        f"/home/justinhong/data/cosmx/cosmx_liver_{tissue_name}.h5ad"
    ):
        tissue_adata.write_h5ad(
            f"/home/justinhong/data/cosmx/cosmx_liver_{tissue_name}.h5ad"
        )

# %%
# convert sparse dataset to csr
cancer_train_adata = sc.read(
    "/home/justin/data/cosmx/liver/cosmx_liver_CancerousLiver.h5ad"
)
cancer_train_adata.X = cancer_train_adata.X.tocsr()
cancer_train_adata.write_h5ad(
    "/home/justin/data/cosmx/liver/cosmx_liver_CancerousLiver_csr.h5ad"
)

# %%
#########
# subset cancer adata for faster hyperparameter search
#########
cancer_train_adata = sc.read(
    "/home/justin/data/cosmx/liver/cosmx_liver_CancerousLiver_csr.h5ad"
)
# %%
sc.pl.embedding(cancer_train_adata, basis="spatial", color="cellType")
# %%
cancer_adata_sub = cancer_train_adata[
    (cancer_train_adata.obs["y_slide_mm"] >= 6)
    & (cancer_train_adata.obs["y_slide_mm"] <= 9)
]
sc.pl.embedding(
    cancer_adata_sub,
    basis="spatial",
    color="cellType",
)
plt.savefig("./figures/cancer_adata_sub_spatial.png")

# %%
print(cancer_adata_sub.obs["cellType"].value_counts())
print(cancer_train_adata.obs["cellType"].value_counts())

# %%
# Remove cell types with less than 1000 cells
cell_type_counts = cancer_adata_sub.obs["cellType"].value_counts()
cell_types_to_keep = cell_type_counts[cell_type_counts >= 1000].index

cancer_adata_sub = cancer_adata_sub[
    cancer_adata_sub.obs["cellType"].isin(cell_types_to_keep)
].copy()

print("Cell types after removal:")
print(cancer_adata_sub.obs["cellType"].value_counts())

# Update the plot with the filtered data
sc.pl.embedding(
    cancer_adata_sub,
    basis="spatial",
    color="cellType",
    title="Spatial plot after removing rare cell types",
)
plt.savefig("./figures/cancer_adata_sub_spatial_filtered.png")
plt.show()

# %%
# train test split by x/y coordinate
cancer_adata_sub_test_idxs = np.where(
    (cancer_adata_sub.obs["y_slide_mm"] >= 8)
    & (cancer_adata_sub.obs["x_slide_mm"] >= 7)
)[0]
cancer_adata_sub_test = cancer_adata_sub[cancer_adata_sub_test_idxs].copy()
cancer_adata_sub_train = cancer_adata_sub[
    np.setdiff1d(np.arange(cancer_adata_sub.n_obs), cancer_adata_sub_test_idxs)
].copy()

sc.pl.embedding(
    cancer_adata_sub_train,
    basis="spatial",
    color="cellType",
)
plt.savefig("./figures/cancer_adata_sub_train_spatial.png")
sc.pl.embedding(
    cancer_adata_sub_test,
    basis="spatial",
    color="cellType",
)
plt.savefig("./figures/cancer_adata_sub_test_spatial.png")
print(cancer_adata_sub_train.obs["cellType"].value_counts())
print(cancer_adata_sub_test.obs["cellType"].value_counts())
print(f"Num Train: {cancer_adata_sub_train.shape[0]}")
print(f"Num Test: {cancer_adata_sub_test.shape[0]}")
# %%
# save cancer adata sub train and val
cancer_adata_sub_train.write_h5ad(
    "/home/justin/data/cosmx/liver/cosmx_liver_cancer_sub_train.h5ad"
)
cancer_adata_sub_test.write_h5ad(
    "/home/justin/data/cosmx/liver/cosmx_liver_cancer_sub_test.h5ad"
)

# %%
##############
# Check cell type annotations with SCVI
##############
cancer_adata = adata[adata.obs["Run_Tissue_name"] == "CancerousLiver"].copy()
cancer_adata
# %%
# train scvi on cancer adata
scvi.model.SCVI.setup_anndata(cancer_adata, batch_key=None)
model = scvi.model.SCVI(cancer_adata)
model.train()

# %%
# save model
model.save(
    "/home/justinhong/data/cosmx/cosmx_liver_cancer_scvi_model", save_anndata=False
)

# %%
# run scvi on cancer adata
latent = model.get_latent_representation(cancer_adata)
# %%
cancer_adata.obsm["X_scVI"] = latent
# %%
# save cancer adata
cancer_adata.write_h5ad("/home/justinhong/data/cosmx/cosmx_liver_cancer_w_scvi.h5ad")
# %%
# load cancer adata
cancer_adata = sc.read_h5ad(
    "/home/justinhong/data/cosmx/cosmx_liver_cancer_w_scvi.h5ad"
)

# %%
# check scvi latent space by cell type
# Compute UMAP on scVI latent space
cancer_adata_subsample = cancer_adata[
    np.random.choice(cancer_adata.n_obs, 10000)
].copy()
sc.pp.neighbors(cancer_adata_subsample, use_rep="X_scVI")
sc.tl.umap(cancer_adata_subsample)

# Plot UMAP colored by cell_type
sc.pl.umap(cancer_adata_subsample, color="cellType", title="UMAP of scVI latent space")
plt.show()

# %%
# check cell type by spatial domain
cancer_adata.obsm["spatial"] = cancer_adata.obs[["x_slide_mm", "y_slide_mm"]].values
sc.pl.embedding(cancer_adata, basis="spatial", color="cellType")
plt.show()

# %%
sc.pl.embedding(cancer_adata, basis="spatial", color="niche")
plt.show()
