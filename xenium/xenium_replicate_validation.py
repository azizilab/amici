# %% Import necessary libraries
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from amici import AMICI
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

# %% Load the two replicates separately as different anndata objects
data_date = "2025-05-01"
adata_rep1 = sc.read_h5ad(f"./data/xenium_sample1/xenium_sample1_rep1_filtered_{data_date}.h5ad")
adata_rep2 = sc.read_h5ad(f"./data/xenium_sample1/xenium_sample1_rep2_filtered_{data_date}.h5ad")

# %% Load the two models for each replicate separately
# Replicate 1
model_rep1_date = "2025-05-13"
wandb_run_id_rep1 = "6xyu2ted"
wandb_sweep_id_rep1 = "4jrcb6jd"
seed_rep1 = 42

saved_models_dir_rep1 = f"saved_models/xenium_sample1_rep1_proseg_sweep_{data_date}_model_{model_rep1_date}"
model_path_rep1 = os.path.join(
    saved_models_dir_rep1,
    f"xenium_{seed_rep1}_sweep_{wandb_sweep_id_rep1}_{wandb_run_id_rep1}_params_{model_rep1_date}",
)

# Replicate 2
model_rep2_date = "2025-05-14"
wandb_run_id_rep2 = "8h73cxui"
wandb_sweep_id_rep2 = "pwyd8qid"
seed_rep2 = 22

saved_models_dir_rep2 = f"saved_models/xenium_sample1_rep2_proseg_sweep_{data_date}_model_{model_rep2_date}"
model_path_rep2 = os.path.join(
    saved_models_dir_rep2,
    f"xenium_{seed_rep2}_sweep_{wandb_sweep_id_rep2}_{wandb_run_id_rep2}_params_{model_rep2_date}",
)

# %%
# Create color palette for each cell type of interest
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

# %% Load the models for both replicates
model_rep1 = AMICI.load(
    model_path_rep1,
    adata=adata_rep1,
)
AMICI.setup_anndata(
    adata_rep1,
    labels_key="celltype_train_grouped",
    coord_obsm_key="spatial",
    n_neighbors=50,
)

model_rep2 = AMICI.load(
    model_path_rep2,
    adata=adata_rep2,
)
AMICI.setup_anndata(
    adata_rep2,
    labels_key="celltype_train_grouped",
    coord_obsm_key="spatial",
    n_neighbors=50,
)

# %% Compute the ablation scores for the two models
ablation_residuals_rep1 = model_rep1.get_neighbor_ablation_scores(
    adata=adata_rep1,
    compute_z_value=True,
)
ablation_residuals_rep2 = model_rep2.get_neighbor_ablation_scores(
    adata=adata_rep2,
    compute_z_value=True,
)

# %% Visualize the heatmap of the interaction scores
ablation_residuals_rep1.plot_interaction_weight_heatmap(save_png=True, save_svg=True, save_dir="./figures/rep1")
ablation_residuals_rep2.plot_interaction_weight_heatmap(save_png=True, save_svg=True, save_dir="./figures/rep2")

# %% Compute the mean absolute difference in interaction weight matrices
interaction_weight_matrix_df_rep1 = ablation_residuals_rep1._get_interaction_weight_matrix()
interaction_weight_matrix_df_rep2 = ablation_residuals_rep2._get_interaction_weight_matrix()
mean_abs_diff = np.mean(np.abs(interaction_weight_matrix_df_rep1 - interaction_weight_matrix_df_rep2))

print(f"Mean absolute difference in interaction weights: {mean_abs_diff}")

# %% Compute the Spearman correlation on the *flattened* interaction-weight matrices
# so that corresponding sender–receiver pairs are compared one-to-one.
flat_rep1 = interaction_weight_matrix_df_rep1.values.flatten()
flat_rep2 = interaction_weight_matrix_df_rep2.values.flatten()

# Create labels for the sender and receiver for each point
pair_labels = [f"{sender}->{receiver}" for sender in interaction_weight_matrix_df_rep1.index for receiver in interaction_weight_matrix_df_rep1.columns]
sender_labels = [f"{interaction.split('->')[0]}" for interaction in pair_labels]
receiver_labels = [f"{interaction.split('->')[1]}" for interaction in pair_labels]

# Spearman correlation coefficient (rho) and associated p-value
spearman_rho, spearman_p = spearmanr(flat_rep1, flat_rep2)
plt.figure(figsize=(6, 6))
sns.scatterplot(x=flat_rep1, y=flat_rep2, hue=sender_labels, palette=CELL_TYPE_PALETTE)

# Identity line for reference
lims = [min(flat_rep1.min(), flat_rep2.min()), max(flat_rep1.max(), flat_rep2.max())]
plt.plot(lims, lims, "--", color="grey", linewidth=1)

plt.xlabel("Interaction weight (Replicate 1)")
plt.ylabel("Interaction weight (Replicate 2)")
plt.title(f"Spearman ρ = {spearman_rho:.2f} (p = {spearman_p:.1e}) - Colored by sender")
plt.xlim(lims)
plt.ylim(lims)
plt.tight_layout()
plt.savefig("figures/xenium_replicate_validation_interaction_correlation.svg", dpi=300, bbox_inches='tight')
plt.savefig("figures/xenium_replicate_validation_interaction_correlation.png", dpi=300, bbox_inches='tight')
plt.show()

# %% Compute the top genes scored by the two models for a given cell-type pair
sender_ct = "CD8+_T_Cells"
receiver_ct = "Macrophages_1"

ablation_ct_residuals_rep1 = model_rep1.get_neighbor_ablation_scores(
    adata=adata_rep1,
    head_idx=None,
    cell_type=receiver_ct,
    ablated_neighbor_ct_sub=[sender_ct],
    compute_z_value=True,
)

ablation_ct_residuals_rep2 = model_rep2.get_neighbor_ablation_scores(
    adata=adata_rep2,
    head_idx=None,
    cell_type=receiver_ct,
    ablated_neighbor_ct_sub=[sender_ct],
    compute_z_value=True,
)

# %% Compare the top genes using ablation scores for the two models
ablation_ct_residuals_rep1.plot_featurewise_contributions_dotplot(
    cell_type=receiver_ct,
    color_by="diff",
    size_by="nl10_pval_adj",
    n_top_genes=5,
    min_size_by=1.3,
    save_svg=True,
    save_png=True,
    save_dir="./figures/rep1"
)

ablation_ct_residuals_rep2.plot_featurewise_contributions_dotplot(
    cell_type=receiver_ct,
    color_by="diff",
    size_by="nl10_pval_adj",
    n_top_genes=5,
    min_size_by=1.3,
    save_svg=True,
    save_png=True,
    save_dir="./figures/rep2"
)
