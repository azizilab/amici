# %% Import libraries
import os

import scanpy as sc
import pytorch_lightning as pl
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from amici import AMICI

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
    "B_Cells": "#c5a9e8",
    "Mast_Cells": "#947b79",
    "Perivascular-Like": "#872727",
    "Endothelial": "#277987",
}

# %% Seed everything
seed = 18
pl.seed_everything(seed)

# %% Load old model data
old_labels_key = "celltype_train_grouped"
old_data_date = "2025-05-01"
old_model_date = "2025-05-02"

adata_old = sc.read_h5ad(f"data/xenium_sample1_filtered_{old_data_date}.h5ad")
adata_old_train = sc.read_h5ad(
    f"data/xenium_sample1_filtered_train_{old_data_date}.h5ad"
)

old_saved_models_dir = f"saved_models/"
old_wandb_run_id = "te7pkv3z"
old_wandb_sweep_id = "g3mucw4s"
old_model_path = os.path.join(
    old_saved_models_dir,
    f"xenium_{seed}_sweep_{old_wandb_sweep_id}_{old_wandb_run_id}_params_{old_model_date}",
)

AMICI.setup_anndata(
    adata_old,
    labels_key=old_labels_key,
    coord_obsm_key="spatial",
    n_neighbors=50,
)
model_old = AMICI.load(old_model_path, adata=adata_old)

# %% Load new lowres model data
lowres_seed = 42
lowres_labels_key = "celltype_lowres"
lowres_data_date = "2026-03-27"  # update to match when xenium_preprocess_lowres.py was run
lowres_model_date = "2026-03-29"
lowres_wandb_sweep_id = "xsjrwnof"
lowres_wandb_run_id = "5tfi78lp"

adata_new = sc.read_h5ad(f"data/xenium_sample1_filtered_lowres_{lowres_data_date}.h5ad")
adata_new_train = sc.read_h5ad(
    f"data/xenium_sample1_filtered_lowres_train_{lowres_data_date}.h5ad"
)

new_saved_models_dir = f"saved_models/xenium_sample1_lowres_sweep_{lowres_data_date}_model_{lowres_model_date}"
new_model_path = os.path.join(
    new_saved_models_dir,
    f"xenium_{lowres_seed}_sweep_{lowres_wandb_sweep_id}_{lowres_wandb_run_id}_params_{lowres_model_date}",
)

AMICI.setup_anndata(
    adata_new,
    labels_key=lowres_labels_key,
    coord_obsm_key="spatial",
    n_neighbors=50,
)
model_new = AMICI.load(new_model_path, adata=adata_new)

# %% Compute interaction matrices for both models
ablation_old = model_old.get_neighbor_ablation_scores(
    adata=adata_old,
    compute_z_value=True,
)
interaction_matrix_old = ablation_old._get_interaction_weight_matrix()

ablation_new = model_new.get_neighbor_ablation_scores(
    adata=adata_new,
    compute_z_value=True,
)
interaction_matrix_new = ablation_new._get_interaction_weight_matrix()

# %% Aggregate the old model's high-res interaction matrix to lowres cell types
# Uses the same grouping map as xenium_preprocess_lowres.py
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

interaction_matrix_old_lowres = interaction_matrix_old.copy()
interaction_matrix_old_lowres.index = [
    cell_label_map_lowres.get(ct, ct) for ct in interaction_matrix_old.index
]
interaction_matrix_old_lowres.columns = [
    cell_label_map_lowres.get(ct, ct) for ct in interaction_matrix_old.columns
]
# Average over merged groups for both senders (rows) and receivers (columns)
interaction_matrix_old_lowres = interaction_matrix_old_lowres.groupby(level=0).mean()
interaction_matrix_old_lowres = interaction_matrix_old_lowres.T.groupby(level=0).mean().T

# %% Plot side-by-side interaction weight heatmaps at lowres resolution
# Order lowres cell types explicitly, then expand to highres with subtypes grouped together
lowres_order = ["T_Cells", "DCIS", "DCs", "Macrophages", "Myoepithelial", "Invasive_Tumor", "B_Cells", "Mast_Cells", "Vascular"]
highres_groups = {
    "T_Cells": ["CD8+_T_Cells", "CD4+_T_Cells"],
    "DCIS": ["DCIS_1", "DCIS_2"],
    "DCs": ["IRF7+_DCs", "LAMP3+_DCs"],
    "Macrophages": ["Macrophages_1", "Macrophages_2"],
    "Myoepithelial": ["Myoepi_ACTA2+", "Myoepi_KRT15+"],
    "Invasive_Tumor": ["Invasive_Tumor"],
    "B_Cells": ["B_Cells"],
    "Mast_Cells": ["Mast_Cells"],
    "Vascular": ["Perivascular-Like", "Endothelial"],
}
highres_order = [ct for group in lowres_order for ct in highres_groups.get(group, [])]

shared_lowres = [ct for ct in lowres_order if ct in interaction_matrix_old_lowres.index and ct in interaction_matrix_new.index]
shared_highres = [ct for ct in highres_order if ct in interaction_matrix_old.index]

mat_old_hires = interaction_matrix_old.loc[shared_highres, shared_highres]
mat_old_shared = interaction_matrix_old_lowres.loc[shared_lowres, shared_lowres]
mat_new_shared = interaction_matrix_new.loc[shared_lowres, shared_lowres]

fig, axes = plt.subplots(1, 3, figsize=(34, 9))

sns.heatmap(
    mat_old_hires,
    ax=axes[0],
    cmap="Reds",
    cbar_kws={"label": "Interaction Weight"},
    linewidths=0.5,
)
axes[0].set_title("Old Model (high-res labels)")
axes[0].set_xlabel("Receiver Cell Type")
axes[0].set_ylabel("Sender Cell Type")

sns.heatmap(
    mat_old_shared,
    ax=axes[1],
    cmap="Reds",
    cbar_kws={"label": "Interaction Weight"},
    linewidths=0.5,
)
axes[1].set_title("Old Model (aggregated to lowres labels)")
axes[1].set_xlabel("Receiver Cell Type")
axes[1].set_ylabel("Sender Cell Type")

sns.heatmap(
    mat_new_shared,
    ax=axes[2],
    cmap="Reds",
    cbar_kws={"label": "Interaction Weight"},
    linewidths=0.5,
)
axes[2].set_title("New Model (lowres-trained)")
axes[2].set_xlabel("Receiver Cell Type")
axes[2].set_ylabel("Sender Cell Type")

plt.tight_layout()
plt.savefig("figures/xenium_sample1_interaction_matrix_comparison.png", dpi=200)
plt.savefig("figures/xenium_sample1_interaction_matrix_comparison.svg")
plt.show()

# %% Compute interaction matrix similarity score with permutation null
# Exclude diagonal (same cell type as sender and receiver)
off_diag_mask = ~np.eye(len(shared_lowres), dtype=bool)
A = mat_old_shared.values[off_diag_mask]
B = mat_new_shared.values[off_diag_mask]

cosine_sim = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
r_matrix, p_matrix = pearsonr(A, B)

# Permute cell-type labels (rows and columns together) to build a null distribution
n_permutations = 1000
rng = np.random.default_rng(seed)
perm_cosine = np.zeros(n_permutations)
perm_pearson = np.zeros(n_permutations)
n_ct = len(shared_lowres)
mat_old_vals = mat_old_shared.values
mat_new_vals = mat_new_shared.values

for i in range(n_permutations):
    perm = rng.permutation(n_ct)
    B_perm = mat_new_vals[np.ix_(perm, perm)][off_diag_mask]
    perm_cosine[i] = np.dot(A, B_perm) / (np.linalg.norm(A) * np.linalg.norm(B_perm))
    perm_pearson[i] = pearsonr(A, B_perm)[0]

cosine_z = (cosine_sim - perm_cosine.mean()) / perm_cosine.std()
pearson_z = (r_matrix - perm_pearson.mean()) / perm_pearson.std()

# Empirical p-value: fraction of permutations >= observed; if none, report < 1/N
def empirical_pval(observed, null_vals):
    p = (null_vals >= observed).sum() / len(null_vals)
    return p if p > 0 else None

p_cosine_emp = empirical_pval(cosine_sim, perm_cosine)
p_pearson_emp = empirical_pval(r_matrix, perm_pearson)

def fmt_pval(p, n):
    return f"< {1/n:.4f}" if p is None else f"{p:.4f}"

print("Interaction Matrix Similarity (cell-type x cell-type)")
print("=" * 56)
print(f"Cosine similarity:  {cosine_sim:.4f}  |  null: {perm_cosine.mean():.4f} ± {perm_cosine.std():.4f}  |  z = {cosine_z:.2f}  |  p = {fmt_pval(p_cosine_emp, n_permutations)}")
print(f"Pearson r:          {r_matrix:.4f}  |  null: {perm_pearson.mean():.4f} ± {perm_pearson.std():.4f}  |  z = {pearson_z:.2f}  |  p = {fmt_pval(p_pearson_emp, n_permutations)}")

# %% Plot null distributions vs observed similarity scores
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, null_vals, observed, metric, p_emp in zip(
    axes,
    [perm_cosine, perm_pearson],
    [cosine_sim, r_matrix],
    ["Cosine Similarity", "Pearson r"],
    [p_cosine_emp, p_pearson_emp],
):
    z = (observed - null_vals.mean()) / null_vals.std()
    p_str = fmt_pval(p_emp, n_permutations)
    ax.hist(null_vals, bins=40, color="steelblue", alpha=0.7, label="Shuffled null")
    ax.axvline(observed, color="crimson", linewidth=2, label=f"Observed ({observed:.3f})")
    ax.axvline(null_vals.mean(), color="steelblue", linewidth=1.5, linestyle="--", label=f"Null mean ({null_vals.mean():.3f})")
    ax.set_xlabel(metric)
    ax.set_ylabel("Count")
    ax.set_title(f"{metric}:  z = {z:.2f},  p = {p_str}")
    ax.legend()

plt.suptitle("Interaction Matrix Similarity vs Shuffled Null", fontsize=13)
plt.tight_layout()
plt.savefig("figures/xenium_sample1_interaction_matrix_similarity_null.png", dpi=200)
plt.savefig("figures/xenium_sample1_interaction_matrix_similarity_null.svg")
plt.show()

# %%
