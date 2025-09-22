# %% Import libraries
import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
from amici import AMICI
from amici.interpretation import (
    AMICICounterfactualAttentionModule,
    AMICIAttentionModule,
)

# %% Create the color palette for the cell types
CELL_TYPE_PALETTE = {
    # Excitatory Neurons
    "L2/3 IT": "#e41a1c",  # 4532 cells
    "L4/5 IT": "#ff7f00",  # 4617 cells
    "L5 IT": "#fdbf6f",  # 2319 cells
    "L5 ET": "#e31a1c",  # 846 cells
    "L6 IT": "#6a3d9a",  # 1941 cells
    "L6 IT Car3": "#cab2d6",  # 391 cells
    "L6 CT": "#fb9a99",  # 3109 cells
    "L5/6 NP": "#a6cee3",  # 345 cells
    "L6b": "#1f78b4",  # 499 cells
    # Inhibitory Neurons
    "Pvalb": "#8dd3c7",  # 880 cells
    "Sst": "#80b1d3",  # 479 cells
    "Lamp5": "#33a02c",  # 334 cells
    "Vip": "#b2df8a",  # 257 cells
    "Sncg": "#bc80bd",  # 35 cells
    # Glial Cells
    "Astro": "#bebada",  # 2560 cells
    "Oligo": "#fb8072",  # 2786 cells
    "OPC": "#b3de69",  # 660 cells
    "Micro": "#fccde5",  # 981 cells
    "VLMC": "#d9d9d9",  # 764 cells
    # Vascular Cells
    "Endo": "#ffff33",  # 2478 cells
    "Peri": "#ffffb3",  # 904 cells
    "PVM": "#fdb462",  # 570 cells
    "SMC": "#8dd3c7",  # 507 cells
    # Other
    "other": "#999999",  # 1037 cells
}

# %% Seed everything
seed = 18
pl.seed_everything(seed)

# %% Load the dataset and the model
data_date = "2025-04-28"
model_date = "2025-05-05"
wandb_sweep_id = "plm73bmg"
wandb_run_id = "xrtcnlt0"

labels_key = "subclass"
adata = sc.read_h5ad(f"./data/cortex_processed_{data_date}.h5ad")
adata.obsm["spatial"] = adata.obs[["centroid_x", "centroid_y"]].values
model_path = f"./saved_models/cortex_{seed}_sweep_{wandb_sweep_id}_{wandb_run_id}_params_{model_date}"

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

# %% Visualize entire dataset
def visualize_spatial_distribution(adata, labels_key="subclass", x_lim=None, y_lim=None):
    plot_df = pd.DataFrame(adata.obsm["spatial"].copy(), columns=["X", "Y"])
    plot_df[labels_key] = adata.obs[labels_key].values
    plot_df["in_test"] = adata.obs["in_test"].values
    plot_df["slice_id"] = adata.obs["slice_id"].values

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        plot_df, x="X", y="Y", hue=labels_key, alpha=0.7, s=8, palette=CELL_TYPE_PALETTE
    )

    # Add a box around test observations for the current slice
    test_df = plot_df[plot_df["in_test"] == True]
    if len(test_df) > 0:
        min_x, max_x = test_df["X"].min(), test_df["X"].max()
        min_y, max_y = test_df["Y"].min(), test_df["Y"].max()
        width = max_x - min_x
        height = max_y - min_y

        # Add rectangle patch with slightly larger bounds for visibility
        padding = 20  # Adjust padding as needed
        rect = plt.Rectangle(
            (min_x - padding, min_y - padding),
            width + 2*padding,
            height + 2*padding,
        fill=False,
        color='black',
        linestyle='--',
        linewidth=2,
        label=f'Test Region'
    )
    plt.gca().add_patch(rect)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Spatial plot for entire dataset")

    # Add legend for both cell types and test region box
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles=handles,
        labels=labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        markerscale=2
    )

    if x_lim is not None:
        plt.xlim(0, x_lim)
    if y_lim is not None:
        plt.ylim(0, y_lim)
    plt.tight_layout()
    plt.savefig(f"figures/cortex_spatial_distribution.png")
    plt.show()

visualize_spatial_distribution(adata)

# %% Visualize dataset
def visualize_spatial_distribution_per_slice(adata, labels_key="subclass", x_lim=None, y_lim=None):
    # Iterate over each unique slice_id
    for slice_id in adata.obs["slice_id"].unique():
        # Filter the data for the current slice
        slice_data = adata[adata.obs["slice_id"] == slice_id]
        plot_df = pd.DataFrame(slice_data.obsm["spatial"].copy(), columns=["X", "Y"])
        plot_df[labels_key] = slice_data.obs[labels_key].values
        plot_df["in_test"] = slice_data.obs["in_test"].values
        plot_df["slice_id"] = slice_data.obs["slice_id"].values

        # Create scatter plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            plot_df, x="X", y="Y", hue=labels_key, alpha=0.7, s=8, palette=CELL_TYPE_PALETTE
        )

        # Add a box around test observations for the current slice
        test_df = plot_df[plot_df["in_test"] == True]
        if len(test_df) > 0:
            min_x, max_x = test_df["X"].min(), test_df["X"].max()
            min_y, max_y = test_df["Y"].min(), test_df["Y"].max()
            width = max_x - min_x
            height = max_y - min_y

            # Add rectangle patch with slightly larger bounds for visibility
            padding = 20  # Adjust padding as needed
            rect = plt.Rectangle(
                (min_x - padding, min_y - padding),
                width + 2*padding,
                height + 2*padding,
                fill=False,
                color='black',
                linestyle='--',
                linewidth=2,
                label=f'Test Region {slice_id}'
            )
            plt.gca().add_patch(rect)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Spatial plot for Slice {slice_id}")

        # Add legend for both cell types and test region box
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(
            handles=handles,
            labels=labels,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            markerscale=2
        )

        if x_lim is not None:
            plt.xlim(0, x_lim)
        if y_lim is not None:
            plt.ylim(0, y_lim)
        plt.tight_layout()
        plt.savefig(f"figures/cortex_slice_{slice_id}_spatial_distribution.png")
        plt.show()

visualize_spatial_distribution_per_slice(adata)

# %% Visualize directed graph of interactions between cell types
ablation_residuals = model.get_neighbor_ablation_scores(
    adata=adata,
    compute_z_value=True,
)

# %% Plot the interaction weight matrix as a heatmap
ablation_residuals.plot_interaction_weight_heatmap(save_png=True, save_svg=True, save_dir="./figures")

# %% Grab quantile to get threshold for weight matrix
interaction_weight_matrix_df = ablation_residuals._get_interaction_weight_matrix()
interaction_weight_matrix = interaction_weight_matrix_df.values.flatten()
quantile = 0.86
weight_threshold = np.quantile(interaction_weight_matrix, quantile)
print(f"{quantile} quantile threshold: {weight_threshold:.2f}")

sns.kdeplot(
    x=interaction_weight_matrix
)
plt.title("Distribution of interaction weights")
plt.xlabel("Interaction weight")
plt.ylabel("Density")
plt.axvline(weight_threshold, color='r', linestyle='--', label=f'{quantile} quantile threshold: {weight_threshold:.2f}')
plt.legend()
plt.show()

# %% Create a subset of cell types of interest to analyze
cell_type_sub = [
    "L2/3 IT",
    "L4/5 IT",
    "L6b",
    "Oligo",
    "Astro",
]

# %% Hierarchical clustering of the interaction weight matrix for all cell types
g = sns.clustermap(
    interaction_weight_matrix_df,
    method='ward',
    cmap='Reds',
    figsize=(12, 10),
    cbar_kws={'label': 'Interaction Weight'},
    linewidths=0.5,
    fmt='.1f'
)
g.ax_heatmap.set_xlabel('Receiver Cell Type')  # Set x-axis title
g.ax_heatmap.set_ylabel('Sender Cell Type')  # Set y-axis title
plt.title('Hierarchical Clustering of Interaction Weight Matrix')
plt.show()

# %% Hierarchical clustering of the interaction weight matrix for the subset of cell types
interaction_weight_matrix_sub_df = interaction_weight_matrix_df.loc[cell_type_sub, cell_type_sub]
g = sns.clustermap(
    interaction_weight_matrix_sub_df,
    method='ward',
    cmap='Reds',
    figsize=(12, 10),
    cbar_kws={'label': 'Interaction Weight'},
    linewidths=0.5,
    fmt='.1f',
)
g.ax_heatmap.set_xlabel('Receiver Cell Type')  # Set x-axis title
g.ax_heatmap.set_ylabel('Sender Cell Type')  # Set y-axis title
plt.title('Hierarchical Clustering of Interaction Weight Matrix for Subset')
plt.show()

# %% Plot the directed graph of interaction between all cell types
ablation_residuals.plot_interaction_directed_graph(
    significance_threshold=0.05,
    weight_threshold=weight_threshold,
    palette=CELL_TYPE_PALETTE,
    save_svg=True,
    save_dir="./figures"
)

# %% Plot the directed graph of interactions between cell types of interest
ablation_residuals.plot_interaction_directed_graph(
    cell_type_sub=cell_type_sub,
    significance_threshold=0.05,
    weight_threshold=weight_threshold,
    palette=CELL_TYPE_PALETTE,
    save_svg=True,
    save_dir="./figures"
)

# %% Compute explained variance scores
expl_variance_scores = model.get_expl_variance_scores(
    adata,
    run_permutation_test=False,
)

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

# %% Define a target cell type of interest
target_ct = "Astro"

# %% Plot neighbor cell type neighbor ablation scores
ablation_residuals = model.get_neighbor_ablation_scores(
    adata=adata,
    cell_type=target_ct,
    ablated_neighbor_ct_sub=["L2/3 IT", "Oligo"],
    compute_z_value=True,
)

# %% Plot summary and featurewise ablation heatmap of neighbor cell type influence
ablation_residuals.plot_neighbor_ablation_scores(
    score_col="ablation",
    palette=CELL_TYPE_PALETTE,
    wandb_log=False,
    show=True,
    save_png=True,
)
ablation_residuals.plot_featurewise_ablation_heatmap(
    score_col="z_value",
    wandb_log=False,
    show=True,
    save_png=True,
)
ablation_residuals.plot_featurewise_contributions_heatmap(
    sort_by="z_value",
    n_top_genes=10,
    wandb_log=False,
    save_png=True,
    show=True,
)

# %% Plot the dotplot of z-values by neighbor contribution scores for the target cell type
ablation_residuals.plot_featurewise_contributions_dotplot(
    cell_type=target_ct,
    color_by="diff",
    size_by="z_value",
    min_size_by=15,
    step=10,
    n_top_genes=8,
    save_png=True,
    save_svg=True,
)

# %% Compute counterfactual attention scores for a query cell type and attention head
counterfactual_attention_patterns = model.get_counterfactual_attention_patterns(
    cell_type=target_ct,
    adata=adata,
)

# %% Select sender types of interest
sender_types = ["L4/5 IT", "L6 CT", "L2/3 IT", "L6b"]

# %% Plot counterfactual attention summary for relevant neighbors for all heads
neighbor_ct_sub = [ct for ct in sender_types if ct != target_ct]
for head_idx in range(model.module.n_heads):
    counterfactual_attention_patterns.plot_counterfactual_attention_summary(
        head_idx=head_idx,
        distances=np.linspace(0, 50, num=25),
        neighbor_ct_sub=neighbor_ct_sub,
        palette=CELL_TYPE_PALETTE,
        save_png=True,
        wandb_log=False,
        show=True,
    )

# %% Plot counterfactual attention summary for a specific head
head_of_interest = 1
counterfactual_attention_patterns.plot_counterfactual_attention_summary(
    head_idx=head_of_interest,
    distances=np.linspace(0, 50, num=25),
    neighbor_ct_sub=sender_types,
    palette=CELL_TYPE_PALETTE,
)

# %% Plot length scales for pairs of cells based on attention scores
length_scale_df = counterfactual_attention_patterns.plot_length_scale_distribution(
    head_idxs=range(model.module.n_heads),
    sender_types=sender_types,
    attention_threshold=0.1,
    sample_threshold=0.01,
    max_length_scale=300,
    plot_kde=True,
    palette=CELL_TYPE_PALETTE,
    save_png=True,
    save_svg=True,
    show=True
)

# %% Plot the length scales as boxplots for each head and sender type
length_scale_df = counterfactual_attention_patterns.plot_length_scale_distribution(
    head_idxs=range(model.module.n_heads),
    sender_types=sender_types,
    attention_threshold=0.1,
    sample_threshold=0.01,
    max_length_scale=50,
    palette=CELL_TYPE_PALETTE,
    save_png=True,
    save_svg=True,
    show=True
)