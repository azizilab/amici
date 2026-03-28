# %% Import libraries
import os

import scanpy as sc
import pytorch_lightning as pl
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import pearsonr, spearmanr
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
lowres_labels_key = "celltype_lowres"
lowres_data_date = "2026-03-27"  # update to match when xenium_preprocess_lowres.py was run
lowres_model_date = "2026-03-27"
lowres_wandb_sweep_id = "u9ymyr7k"
lowres_wandb_run_id = "n7nwnmf9"

adata_new = sc.read_h5ad(f"data/xenium_sample1_filtered_lowres_{lowres_data_date}.h5ad")
adata_new_train = sc.read_h5ad(
    f"data/xenium_sample1_filtered_lowres_train_{lowres_data_date}.h5ad"
)

new_saved_models_dir = f"saved_models/xenium_sample1_lowres_sweep_{lowres_data_date}_model_{lowres_model_date}"
new_model_path = os.path.join(
    new_saved_models_dir,
    f"xenium_{seed}_sweep_{lowres_wandb_sweep_id}_{lowres_wandb_run_id}_params_{lowres_model_date}",
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
shared_cell_types = sorted(
    set(interaction_matrix_old_lowres.index) & set(interaction_matrix_new.index)
)
mat_old_shared = interaction_matrix_old_lowres.loc[shared_cell_types, shared_cell_types]
mat_new_shared = interaction_matrix_new.loc[shared_cell_types, shared_cell_types]

fig, axes = plt.subplots(1, 2, figsize=(22, 9))

sns.heatmap(
    mat_old_shared,
    ax=axes[0],
    cmap="Reds",
    cbar_kws={"label": "Interaction Weight"},
    linewidths=0.5,
)
axes[0].set_title("Old Model (aggregated to lowres labels)")
axes[0].set_xlabel("Receiver Cell Type")
axes[0].set_ylabel("Sender Cell Type")

sns.heatmap(
    mat_new_shared,
    ax=axes[1],
    cmap="Reds",
    cbar_kws={"label": "Interaction Weight"},
    linewidths=0.5,
)
axes[1].set_title("New Model (lowres-trained)")
axes[1].set_xlabel("Receiver Cell Type")
axes[1].set_ylabel("Sender Cell Type")

plt.tight_layout()
plt.savefig("figures/xenium_sample1_interaction_matrix_comparison.png", dpi=200)
plt.savefig("figures/xenium_sample1_interaction_matrix_comparison.svg")
plt.show()

# %% Compute interaction matrix similarity score with permutation null
A = mat_old_shared.values.flatten()
B = mat_new_shared.values.flatten()

cosine_sim = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
r_matrix, p_matrix = pearsonr(A, B)

# Permute cell-type labels (rows and columns together) to build a null distribution
n_permutations = 1000
rng = np.random.default_rng(seed)
perm_cosine = np.zeros(n_permutations)
perm_pearson = np.zeros(n_permutations)
n_ct = len(shared_cell_types)
mat_old_vals = mat_old_shared.values
mat_new_vals = mat_new_shared.values

for i in range(n_permutations):
    perm = rng.permutation(n_ct)
    B_perm = mat_new_vals[np.ix_(perm, perm)].flatten()
    perm_cosine[i] = np.dot(A, B_perm) / (np.linalg.norm(A) * np.linalg.norm(B_perm))
    perm_pearson[i] = pearsonr(A, B_perm)[0]

cosine_z = (cosine_sim - perm_cosine.mean()) / perm_cosine.std()
pearson_z = (r_matrix - perm_pearson.mean()) / perm_pearson.std()

print("Interaction Matrix Similarity (cell-type x cell-type)")
print("=" * 56)
print(f"Cosine similarity:  {cosine_sim:.4f}  |  null: {perm_cosine.mean():.4f} ± {perm_cosine.std():.4f}  |  z = {cosine_z:.2f}")
print(f"Pearson r:          {r_matrix:.4f}  |  null: {perm_pearson.mean():.4f} ± {perm_pearson.std():.4f}  |  z = {pearson_z:.2f}")
print(f"  (p = {p_matrix:.2e})")

# %% Plot null distributions vs observed similarity scores
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, null_vals, observed, metric in zip(
    axes,
    [perm_cosine, perm_pearson],
    [cosine_sim, r_matrix],
    ["Cosine Similarity", "Pearson r"],
):
    ax.hist(null_vals, bins=40, color="steelblue", alpha=0.7, label="Shuffled null")
    ax.axvline(observed, color="crimson", linewidth=2, label=f"Observed ({observed:.3f})")
    ax.axvline(null_vals.mean(), color="steelblue", linewidth=1.5, linestyle="--", label=f"Null mean ({null_vals.mean():.3f})")
    ax.set_xlabel(metric)
    ax.set_ylabel("Count")
    ax.set_title(f"{metric}: z = {(observed - null_vals.mean()) / null_vals.std():.2f}")
    ax.legend()

plt.suptitle("Interaction Matrix Similarity vs Shuffled Null", fontsize=13)
plt.tight_layout()
plt.savefig("figures/xenium_sample1_interaction_matrix_similarity_null.png", dpi=200)
plt.savefig("figures/xenium_sample1_interaction_matrix_similarity_null.svg")
plt.show()

# %% Extract cell-cell attention scores averaged over heads
def get_cell_pair_attention(model, adata):
    attention_patterns = model.get_attention_patterns(adata, batch_size=32)
    nn_idxs_df = attention_patterns._nn_idxs_df  # (n_cells, n_neighbors)

    head_attn_arrays = []
    for head_idx in range(model.module.n_heads):
        head_df = (
            attention_patterns._attention_patterns_df[
                attention_patterns._attention_patterns_df["head"] == head_idx
            ]
            .drop(columns=["label", "head"])
            .set_index("cell_idx")
            .reindex(nn_idxs_df.index)
        )
        head_attn_arrays.append(head_df.values)

    mean_attn = np.mean(head_attn_arrays, axis=0)  # (n_cells, n_neighbors)
    query_idxs = np.repeat(nn_idxs_df.index.values, nn_idxs_df.shape[1])
    neighbor_idxs = nn_idxs_df.values.flatten()

    return pd.DataFrame({
        "query_idx": query_idxs,
        "neighbor_idx": neighbor_idxs,
        "attention": mean_attn.flatten(),
    })

cell_attn_old = get_cell_pair_attention(model_old, adata_old).rename(columns={"attention": "attention_old"})
cell_attn_new = get_cell_pair_attention(model_new, adata_new).rename(columns={"attention": "attention_new"})

# %% Correlation plot of cell-cell attention scores
# Color by high-res cell type labels — both adatas share the same cells and obs_names
cell_attn_merged = cell_attn_old.merge(cell_attn_new, on=["query_idx", "neighbor_idx"])
cell_attn_merged["query_ct"] = cell_attn_merged["query_idx"].map(adata_new.obs[old_labels_key])

r_cell, p_cell = pearsonr(cell_attn_merged["attention_new"], cell_attn_merged["attention_old"])
r_cell_sp, p_cell_sp = spearmanr(cell_attn_merged["attention_new"], cell_attn_merged["attention_old"])

fig, ax = plt.subplots(figsize=(8, 7))

for ct, group in cell_attn_merged.groupby("query_ct"):
    ax.scatter(
        group["attention_new"],
        group["attention_old"],
        color=CELL_TYPE_PALETTE.get(ct, "#808080"),
        label=ct,
        alpha=0.3,
        s=4,
        rasterized=True,
    )

lims = [
    min(cell_attn_merged["attention_new"].min(), cell_attn_merged["attention_old"].min()),
    max(cell_attn_merged["attention_new"].max(), cell_attn_merged["attention_old"].max()),
]
ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1, label="y = x")

ax.set_xlabel("New Model (lowres-trained) — Attention Score", fontsize=12)
ax.set_ylabel("Old Model — Attention Score", fontsize=12)
ax.set_title(
    f"Cell-Cell Attention Score Correlation\n"
    f"Pearson r = {r_cell:.3f} (p={p_cell:.2e}),  "
    f"Spearman ρ = {r_cell_sp:.3f} (p={p_cell_sp:.2e})",
    fontsize=11,
)
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, markerscale=3, fontsize=8)
plt.tight_layout()
plt.savefig("figures/xenium_sample1_cell_attention_correlation.png", dpi=200)
plt.savefig("figures/xenium_sample1_cell_attention_correlation.svg")
plt.show()

# %%
