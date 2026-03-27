# %% Import libraries
import anndata as ad
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import date

# %% Load preprocessed data saved from xenium_preprocess.py
data_date = "2025-05-01"
adata = ad.read_h5ad(f"data/xenium_sample1_filtered_{data_date}.h5ad")

# %% Combine cell type labels into coarser (low-resolution) groups
labels_key_grouped = "celltype_train_grouped"
labels_key_lowres = "celltype_lowres"

cell_label_map_lowres = {
    "CD8+_T_Cells": "T_Cells",
    "CD4+_T_Cells": "T_Cells",
    "DCIS_1": "DCIS",
    "DCIS_2": "DCIS",
    "IRF7+_DCs": "DCs",
    "LAMP3+_DCs": "DCs",
    "Macrophages_1": "Macrophages",
    "Macrophages_2": "Macrophages",
    "Myoepi_ACTA2+": "Myoepithelial",
    "Myoepi_KRT15+": "Myoepithelial",
    "Perivascular-Like": "Vascular",
    "Endothelial": "Vascular",
}

adata.obs[labels_key_lowres] = adata.obs[labels_key_grouped].copy()
for key, val in cell_label_map_lowres.items():
    adata.obs[labels_key_lowres] = adata.obs[labels_key_lowres].replace(key, val)

print("Low-resolution cell types after grouping:")
print(sorted(list(adata.obs[labels_key_lowres].unique())))

# %% Visualize spatial distribution of lowres cell types
CELL_TYPE_PALETTE_LOWRES = {
    "T_Cells": "#56B4E9",
    "DCIS": "#E69F00",
    "DCs": "#7f7f7f",
    "Macrophages": "#e0a4dc",
    "Myoepithelial": "#823960",
    "Invasive_Tumor": "#cf4242",
    "B_Cells": "#c5a9e8",
    "Mast_Cells": "#947b79",
    "Vascular": "#277987",
}

def visualize_spatial_distribution(adata, labels_key, dataset, x_lim=None, y_lim=None):
    plt.figure(figsize=(20, 6))
    plot_df = adata.obsm["spatial"].copy()
    plot_df[labels_key] = adata.obs[labels_key].values
    sns.scatterplot(
        plot_df, x="X", y="Y", hue=labels_key, alpha=0.7, s=8, palette=CELL_TYPE_PALETTE_LOWRES
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"{dataset} Spatial plot (low-resolution labels)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, markerscale=2)
    if x_lim is not None:
        plt.xlim(0, x_lim)
    if y_lim is not None:
        plt.ylim(0, y_lim)
    plt.tight_layout()
    plt.savefig(f"figures/xenium_sample1_lowres_spatial_distribution_{dataset}.png")
    plt.show()

visualize_spatial_distribution(adata, labels_key_lowres, "Lowres_Full")

# %% Split into train and test using existing train_test_split column
adata_train = adata[adata.obs["train_test_split"] == "train"].copy()
adata_test = adata[adata.obs["train_test_split"] == "test"].copy()

print(f"Number of total cells: {len(adata)}")
print(f"Number of training cells: {len(adata_train)}")
print(f"Number of test cells: {len(adata_test)}")

# %% Save lowres anndata objects
today = date.today()
adata.write_h5ad(f"data/xenium_sample1_filtered_lowres_{today}.h5ad")
adata_train.write_h5ad(f"data/xenium_sample1_filtered_lowres_train_{today}.h5ad")
adata_test.write_h5ad(f"data/xenium_sample1_filtered_lowres_test_{today}.h5ad")

print(f"Saved lowres anndata objects with date: {today}")

# %%
