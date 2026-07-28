# %% Import libraries
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import seaborn as sns

from amici import AMICI

# %%
# Create color palette for each cell type of interest
CELL_TYPE_PALETTE = {
    "High_Grade_DCIS": "#D55E00",
    "Luminal_DCIS": "#E69F00",
    "Basal_DCIS": "#CC79A7",
    "CAFs": "#6A3D9A",
    "Fibroblasts": "#A6761D",
    "Myoepithelial": "#0072B2",
    "Endothelial": "#009E73",
    "Pericytes": "#66A61E",
    "T_Cells": "#56B4E9",
    "Dendritic Cells": "#1B9E77",
    "Macrophages": "#7570B3",
    "Myeloid": "#E7298A",
    "Mast_Cells": "#8DD3C7",
    "Plasma_Cells": "#FDB462",
    "Plasma_Mast_Mixture": "#B3B3B3",
    "Apocrine Cells": "#BC80BD",
}

# %% Seed everything
seed = 38
pl.seed_everything(seed)

# %% Load data
labels_key = "celltype_train_grouped"
data_date = "2026-07-22"
model_date = "2026-07-25"
adata = sc.read_h5ad(f"./data/atera_breast_filtered_{data_date}.h5ad")
adata_train = sc.read_h5ad(f"./data/atera_breast_filtered_train_{data_date}.h5ad")
adata_test = sc.read_h5ad(f"./data/atera_breast_filtered_test_{data_date}.h5ad")

figures_dir = "./figures/atera_breast_analysis"
os.makedirs(figures_dir, exist_ok=True)

saved_models_dir = f"saved_models/atera_breast_sweep_{data_date}_model_{model_date}"
wandb_run_id = "8qyy7mxt"
wandb_sweep_id = "25xpxkuk"
model_seed = seed
model_path = os.path.join(
    saved_models_dir,
    f"atera_breast_{model_seed}_sweep_{wandb_sweep_id}_{wandb_run_id}_params_{model_date}",
)
model_path = os.environ.get("ATERA_AMICI_MODEL_DIR", model_path)

print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
print(f"Train cells: {adata_train.n_obs}; test cells: {adata_test.n_obs}")
print(f"Loading model from {model_path}")


# %% Select subset of cell types for interpretation and visualize
def visualize_spatial_distribution(
    adata, labels_key="celltype_train_grouped", x_lim=None, y_lim=None
):
    plt.figure(figsize=(20, 7))
    plot_df = pd.DataFrame(adata.obsm["spatial"], index=adata.obs_names, columns=["X", "Y"])
    plot_df[labels_key] = adata.obs[labels_key].astype(str).values
    plot_df["train_test"] = adata.obs["train_test_split"].astype(str).values

    sns.scatterplot(
        plot_df, x="X", y="Y", hue=labels_key, alpha=0.7, s=3, palette=CELL_TYPE_PALETTE
    )

    test_df = plot_df[plot_df["train_test"] == "test"]
    if len(test_df) > 0:
        min_x, max_x = test_df["X"].min(), test_df["X"].max()
        min_y, max_y = test_df["Y"].min(), test_df["Y"].max()
        width = max_x - min_x
        height = max_y - min_y

        padding = 20
        rect = plt.Rectangle(
            (min_x - padding, min_y - padding),
            width + 2 * padding,
            height + 2 * padding,
            fill=False,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Test Region",
        )
        plt.gca().add_patch(rect)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Atera breast spatial plot for analysis")

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles=handles,
        labels=labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        markerscale=3,
    )

    if x_lim is not None:
        plt.xlim(0, x_lim)
    if y_lim is not None:
        plt.ylim(0, y_lim)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/atera_breast_spatial_subset_analysis.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{figures_dir}/atera_breast_spatial_subset_analysis.svg", dpi=300, bbox_inches="tight")
    plt.show()


cell_type_sub = None

if cell_type_sub is not None:
    adata_viz = adata[adata.obs[labels_key].isin(cell_type_sub)].copy()
    visualize_spatial_distribution(adata_viz)
else:
    visualize_spatial_distribution(adata)

# %% Load model
model = AMICI.load(
    model_path,
    adata=adata,
)
AMICI.setup_anndata(
    adata,
    labels_key=labels_key,
    coord_obsm_key="spatial",
    n_neighbors=50,
)

# %% Visualize directed graph of interactions between cell types
ablation_residuals = model.get_neighbor_ablation_scores(
    adata=adata,
    compute_z_value=True,
)

# %% Plot the interaction weight matrix as a heatmap
ablation_residuals.plot_interaction_weight_heatmap(
    save_png=True,
    save_svg=True,
    save_dir=figures_dir,
)

# %% Grab 80 quantile to get threshold for weight matrix
interaction_weight_matrix_df = ablation_residuals._get_interaction_weight_matrix()
interaction_weight_matrix = interaction_weight_matrix_df.values.flatten()
quantile = 0.80
weight_threshold = np.quantile(interaction_weight_matrix, quantile)
print(f"{quantile} quantile threshold: {weight_threshold:.2f}")

sns.kdeplot(
    x=interaction_weight_matrix
)
plt.title("Distribution of interaction weights")
plt.xlabel("Interaction weight")
plt.ylabel("Density")
plt.axvline(weight_threshold, color="r", linestyle="--", label=f"{quantile} quantile threshold: {weight_threshold:.2f}")
plt.legend()
plt.savefig(f"{figures_dir}/atera_breast_interaction_weight_distribution.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{figures_dir}/atera_breast_interaction_weight_distribution.svg", dpi=300, bbox_inches="tight")
plt.show()

# %% Create a subset of interesting tumor, stromal, vascular, and immune cell types
cell_type_sub = [
    "High_Grade_DCIS",
    "Luminal_DCIS",
    "Basal_DCIS",
    "CAFs",
    "Fibroblasts",
    "Myoepithelial",
    "Endothelial",
    "Pericytes",
    "T_Cells",
    "Dendritic Cells",
    "Macrophages",
    "Myeloid",
    "Plasma_Cells",
    "Mast_Cells",
]
cell_type_sub = [ct for ct in cell_type_sub if ct in interaction_weight_matrix_df.index]

# %%
g = sns.clustermap(
    interaction_weight_matrix_df,
    method="ward",
    cmap="Reds",
    figsize=(12, 10),
    cbar_kws={"label": "Interaction Weight"},
    linewidths=0.5,
    fmt=".1f",
)
g.ax_heatmap.set_xlabel("Receiver Cell Type")
g.ax_heatmap.set_ylabel("Sender Cell Type")
plt.title("Hierarchical Clustering of Interaction Weight Matrix")
plt.savefig(f"{figures_dir}/atera_breast_interaction_weight_clustermap.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{figures_dir}/atera_breast_interaction_weight_clustermap.svg", dpi=300, bbox_inches="tight")
plt.show()

# %% Hierarchical clustering of the interaction weight matrix for subset of cell types
interaction_weight_matrix_sub_df = interaction_weight_matrix_df.loc[cell_type_sub, cell_type_sub]
g = sns.clustermap(
    interaction_weight_matrix_sub_df,
    method="ward",
    cmap="Reds",
    figsize=(12, 10),
    cbar_kws={"label": "Interaction Weight"},
    linewidths=0.5,
    fmt=".1f",
)
g.ax_heatmap.set_xlabel("Receiver Cell Type")
g.ax_heatmap.set_ylabel("Sender Cell Type")
plt.title("Hierarchical Clustering of Interaction Weight Matrix for Subset")
plt.savefig(f"{figures_dir}/atera_breast_interaction_weight_clustermap_subset.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{figures_dir}/atera_breast_interaction_weight_clustermap_subset.svg", dpi=300, bbox_inches="tight")
plt.show()

# %% Plot the directed graph of interaction between all cell types
ablation_residuals.plot_interaction_directed_graph(
    significance_threshold=0.05,
    weight_threshold=weight_threshold,
    palette=CELL_TYPE_PALETTE,
    save_svg=True,
    save_dir=figures_dir,
)

# %% Plot the directed graph of interactions between cell types of interest
ablation_residuals.plot_interaction_directed_graph(
    cell_type_sub=cell_type_sub,
    significance_threshold=0.05,
    weight_threshold=weight_threshold,
    palette=CELL_TYPE_PALETTE,
    save_svg=True,
    save_dir=figures_dir,
)

# %% Compute variance scores for the heads
if cell_type_sub is not None:
    expl_variance_scores = model.get_expl_variance_scores(
        adata,
        cell_type_sub=cell_type_sub,
        run_permutation_test=False,
    )
else:
    expl_variance_scores = model.get_expl_variance_scores(
        adata,
        run_permutation_test=False,
    )

# %% Plot barplot of variance scores per head per cell type
if cell_type_sub is not None:
    expl_variance_scores.plot_explained_variance_barplot(
        palette=CELL_TYPE_PALETTE,
        cell_type_sub=cell_type_sub,
        wandb_log=False,
        save_png=True,
        show=True,
    )
else:
    expl_variance_scores.plot_explained_variance_barplot(
        palette=CELL_TYPE_PALETTE,
        wandb_log=False,
        save_png=True,
        show=True,
    )

# %% Get attention patterns
attention_patterns = model.get_attention_patterns(
    adata,
    batch_size=32,
)

# %% Plot attention summary for subset of cell types
if cell_type_sub is not None:
    attention_patterns.plot_attention_summary(
        cell_type_sub=cell_type_sub,
        palette=CELL_TYPE_PALETTE,
        wandb_log=False,
        save_png=True,
        show=True,
    )
else:
    attention_patterns.plot_attention_summary(
        palette=CELL_TYPE_PALETTE,
        wandb_log=False,
        save_png=True,
        show=True,
    )

# %% Define the target cell type of interest and max expl variance head
receiver_ct = "T_Cells"
sender_cts = ["CAFs", "Dendritic Cells", "High_Grade_DCIS"]
sender_cts = [ct for ct in sender_cts if ct in adata.obs[labels_key].unique()]
head_idx = min(4, model.module.n_heads - 1)

# %%
# Plot important attention patterns for cell types of interest
attention_patterns.plot_attention_summary(
    cell_type_sub=[receiver_ct],
    plot_histogram=False,
    palette=CELL_TYPE_PALETTE,
    wandb_log=False,
    show=True,
    save_png=True,
)

# %% Plot neighbor cell type neighbor ablation scores
ablation_ct_residuals = model.get_neighbor_ablation_scores(
    adata=adata,
    cell_type=receiver_ct,
    ablated_neighbor_ct_sub=sender_cts,
    compute_z_value=True,
)

# %% Plot summary and featurewise ablation heatmap of neighbor cell type influence
ablation_ct_residuals.plot_neighbor_ablation_scores(
    score_col="ablation",
    palette=CELL_TYPE_PALETTE,
    wandb_log=False,
    show=True,
    save_png=True,
    save_dir=figures_dir,
)
ablation_ct_residuals.plot_featurewise_ablation_heatmap(
    score_col="z_value",
    wandb_log=False,
    show=True,
    save_png=True,
    save_dir=figures_dir,
)
ablation_ct_residuals.plot_featurewise_contributions_heatmap(
    sort_by="z_value",
    n_top_genes=10,
    wandb_log=False,
    save_png=True,
    show=True,
    save_dir=figures_dir,
)

# %% Plot the dotplot of p-values by neighbor contribution scores for the target cell type
ablation_ct_residuals.plot_featurewise_contributions_dotplot(
    cell_type=receiver_ct,
    color_by="diff",
    size_by="z_value",
    n_top_genes=10,
    min_size_by=-10,
    step=5,
    save_svg=True,
    save_png=True,
    save_dir=figures_dir,
)

# %% Compute counterfactual attention scores for a query cell type and attention head
counterfactual_attention_patterns = model.get_counterfactual_attention_patterns(
    cell_type=receiver_ct,
    adata=adata,
)

# %% Plot length scales for pairs of cells based on attention scores
length_scale_df = counterfactual_attention_patterns.plot_length_scale_distribution(
    head_idxs=[head_idx],
    sender_types=sender_cts,
    attention_threshold=0.1,
    sample_threshold=0.001,
    max_length_scale=150,
    palette=CELL_TYPE_PALETTE,
    save_png=True,
    save_svg=True,
    save_dir=figures_dir,
    show=True,
)

# %% Plot counterfactual attention summary for relevant neighbors for all heads
neighbor_ct_sub = [ct for ct in sender_cts if ct != receiver_ct]
for head_idx in range(model.module.n_heads):
    counterfactual_attention_patterns.plot_counterfactual_attention_summary(
        head_idx=head_idx,
        distances=np.linspace(0, 150, num=25),
        neighbor_ct_sub=neighbor_ct_sub,
        palette=CELL_TYPE_PALETTE,
        save_png=True,
        wandb_log=False,
        show=True,
        save_dir=figures_dir,
    )

# %% Plot counterfactual attention summary for a specific head
head_of_interest = min(4, model.module.n_heads - 1)
counterfactual_attention_patterns.plot_counterfactual_attention_summary(
    head_idx=head_of_interest,
    distances=np.linspace(0, 150, num=25),
    neighbor_ct_sub=sender_cts,
    palette=CELL_TYPE_PALETTE,
    save_dir=figures_dir,
)

# %% Plot the length scales as boxplots for each head and sender type
length_scale_df = counterfactual_attention_patterns.plot_length_scale_distribution(
    head_idxs=range(model.module.n_heads),
    sender_types=sender_cts,
    attention_threshold=0.1,
    sample_threshold=0.01,
    max_length_scale=300,
    palette=CELL_TYPE_PALETTE,
    save_png=True,
    save_svg=True,
    save_dir=figures_dir,
    show=True,
)
