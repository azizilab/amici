# %% Import libraries
import os
from functools import reduce

import scanpy as sc
import pytorch_lightning as pl
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import distinctipy

from scvi import REGISTRY_KEYS
from amici import AMICI
from amici.interpretation import (
    AMICICounterfactualAttentionModule,
    AMICIAttentionModule,
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

# %% Seed everything
seed = 18
pl.seed_everything(seed)

# %% Load data
labels_key = "celltype_train_grouped"
data_date = "2025-05-01"
model_date = "2025-05-02"
adata = sc.read_h5ad(f"./data/xenium_sample1/xenium_sample1_filtered_{data_date}.h5ad")
adata_train = sc.read_h5ad(
    f"./data/xenium_sample1/xenium_sample1_filtered_train_{data_date}.h5ad"
)
adata_test = sc.read_h5ad(
    f"./data/xenium_sample1/xenium_sample1_filtered_test_{data_date}.h5ad"
)

saved_models_dir = f"saved_models/xenium_sample1_proseg_sweep_{data_date}_model_{model_date}"
wandb_run_id = "te7pkv3z"
wandb_sweep_id = "g3mucw4s"
model_path = os.path.join(
    saved_models_dir,
    f"xenium_{seed}_sweep_{wandb_sweep_id}_{wandb_run_id}_params_{model_date}",
)


# %% Select subset of cell types for interpretation and visualize
def visualize_spatial_distribution(
    adata, labels_key="celltype_train_grouped", x_lim=None, y_lim=None
):
    plt.figure(figsize=(20, 6))
    plot_df = adata.obsm["spatial"].copy()
    plot_df["sample"] = adata.obs["sample"]
    plot_df[labels_key] = adata.obs[labels_key]
    plot_df["train_test"] = adata.obs["train_test_split"]
    
    # Create scatter plot
    sns.scatterplot(
        plot_df, x="X", y="Y", hue=labels_key, alpha=0.7, s=8, palette=CELL_TYPE_PALETTE
    )

    # Add two boxes around test observations for each sample
    for sample in adata.obs["sample"].unique():
        plot_sample_df = plot_df[plot_df["sample"] == sample]
        test_df = plot_sample_df[plot_sample_df["train_test"] == "test"]
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
                label=f'Test Region {sample}'
            )
            plt.gca().add_patch(rect)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Spatial plot of subset for analysis")
    
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
    plt.savefig(f"figures/xenium_sample1_proseg_spatial_subset_analysis.png")
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
ablation_residuals.plot_interaction_weight_heatmap(save_png=True, save_svg=True, save_dir="./figures")

# %% Grab 90 quantile to get threshold for weight matrix
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
plt.axvline(weight_threshold, color='r', linestyle='--', label=f'{quantile} quantile threshold: {weight_threshold:.2f}')
plt.legend()
plt.show()

# %% Create a subset of interesting immune-tumor cell types
cell_type_sub = [
    "Invasive_Tumor",
    "DCIS_1", 
    "DCIS_2", 
    "Macrophages_2",
    "Macrophages_1",
    "CD8+_T_Cells", 
    "CD4+_T_Cells", 
    "B_Cells",  
    "IRF7+_DCs",
    "LAMP3+_DCs", 
]

# %%
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


# %% Hierarchical clustering of the interaction weight matrix for subset of cell types
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
receiver_ct = "Macrophages_1"
sender_cts = ["CD8+_T_Cells"]

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
    head_idx=4,
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
)
ablation_ct_residuals.plot_featurewise_ablation_heatmap(
    score_col="z_value",
    wandb_log=False,
    show=True,
    save_png=True,
)
ablation_ct_residuals.plot_featurewise_contributions_heatmap(
    sort_by="z_value",
    n_top_genes=10,
    wandb_log=False,
    save_png=True,
    show=True,
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
)

# %% Compute counterfactual attention scores for a query cell type and attention head
counterfactual_attention_patterns = model.get_counterfactual_attention_patterns(
    cell_type=receiver_ct,
    adata=adata,
)

# %% Plot length scales for pairs of cells based on attention scores
length_scale_df = counterfactual_attention_patterns.plot_length_scale_distribution(
    head_idxs=[5],
    sender_types=sender_cts,
    attention_threshold=0.1,
    sample_threshold=0.001,
    max_length_scale=50,
    plot_kde=True,
    palette=CELL_TYPE_PALETTE,
    save_png=True,
    save_svg=True,
    show=True
)

# %% Plot counterfactual attention summary for relevant neighbors for all heads
neighbor_ct_sub = [ct for ct in sender_cts if ct != receiver_ct]
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
head_of_interest = 5
counterfactual_attention_patterns.plot_counterfactual_attention_summary(
    head_idx=head_of_interest,
    distances=np.linspace(0, 50, num=25),
    neighbor_ct_sub=sender_cts,
    palette=CELL_TYPE_PALETTE,
)

# %% Plot the length scales as boxplots for each head and sender type
length_scale_df = counterfactual_attention_patterns.plot_length_scale_distribution(
    head_idxs=range(model.module.n_heads),
    sender_types=sender_cts,
    attention_threshold=0.1,
    sample_threshold=0.01,
    max_length_scale=200,
    palette=CELL_TYPE_PALETTE,
    save_png=True,
    save_svg=True,
    show=True
)

# %% Plot the length scales as KDE plots for different sender receiver pairs
interactions = {
    "CD8+_T_Cells": {
        "senders": ["Macrophages_2", "CD4+_T_Cells"],
        "head_idxs": [4, 5],
        "sample_threshold": 0.02,
    },
    "Macrophages_1": {
        "senders": ["Invasive_Tumor", "CD8+_T_Cells", "DCIS_1", "Macrophages_2"],
        "head_idxs": [4, 5],
        "sample_threshold": 0.02,
    },
    "B_Cells": {
        "senders": ["CD4+_T_Cells"],
        "head_idxs": [4, 5],
        "sample_threshold": 0.02,
    },
    "Invasive_Tumor": {
        "senders": ["CD8+_T_Cells"],
        "head_idxs": [5],
        "sample_threshold": 0.001,
    },
    "CD4+_T_Cells": {
        "senders": ["B_Cells", "Macrophages_2"],
        "head_idxs": [5],
        "sample_threshold": 0.02,
    },
}

select_best_head = False

length_scale_dfs = []
for receiver_ct, interaction_config in interactions.items():
    counterfactual_attention_patterns = model.get_counterfactual_attention_patterns(
        cell_type=receiver_ct,
        adata=adata,
    )
    length_scale_df = counterfactual_attention_patterns._calculate_length_scales(
        head_idxs=interaction_config["head_idxs"],
        sender_types=interaction_config["senders"],
        attention_threshold=0.1,
        sample_threshold=interaction_config["sample_threshold"],
    )
    filtered_length_scale_dfs = []
    for sender in interaction_config["senders"]:
        # Only keep the heads with positive length scales and less than 50
        sender_df = length_scale_df[length_scale_df["sender_type"] == sender].copy()
        median_length_scales = sender_df.groupby("head_idx")["length_scale"].median()
        heads_positive_length_scale = median_length_scales[
            (median_length_scales > 0) & 
            (median_length_scales <= 50)
        ].index.tolist()

        sender_df = sender_df[sender_df["head_idx"].isin(heads_positive_length_scale)]

        if select_best_head:
            if len(heads_positive_length_scale) > 1:
                for head_idx in heads_positive_length_scale:
                    ablation_scores = model.get_neighbor_ablation_scores(
                        adata=adata,
                        head_idx=head_idx,
                        cell_type=receiver_ct,
                        ablated_neighbor_ct_sub=[sender],
                        compute_z_value=True,
                    )
                    ablation_df = ablation_scores._ablation_scores_df

                    head_gene_contributions = {}
                    significant_genes = len(ablation_df[
                        (ablation_df[f"{sender}_diff"] > 0) & 
                        (ablation_df[f"{sender}_nl10_pval_adj"] > 2)
                    ])
                    head_gene_contributions[head_idx] = significant_genes
                
                best_head_idx = max(head_gene_contributions, key=head_gene_contributions.get)
                sender_df = sender_df[sender_df["head_idx"] == best_head_idx].copy()

        filtered_length_scale_dfs.append(sender_df)
    
    filtered_length_scale_df = pd.concat(filtered_length_scale_dfs)

    filtered_length_scale_df["target_ct"] = receiver_ct
    filtered_length_scale_df["interaction"] = filtered_length_scale_df["sender_type"] + " -> " + filtered_length_scale_df["target_ct"] + " (head " + filtered_length_scale_df["head_idx"].astype(str) + ")"
    length_scale_dfs.append(filtered_length_scale_df)

length_scale_df = pd.concat(length_scale_dfs)


# %% Plot the length scales as KDE plots for different sender receiver pairs
colors = distinctipy.get_colors(len(length_scale_df["interaction"].unique()))
color_map = dict(zip(length_scale_df["interaction"].unique(), colors))

# Immune vs tumor interactions palette
interaction_colors = {
    # Immune-Tumor interactions (Red palette)
    "CD8+_T_Cells -> Invasive_Tumor (head 5)": "#8B0000",        # Dark red
    "Invasive_Tumor -> Macrophages_1 (head 5)": "#CC2936",       # Deep red
    "Macrophages_2 -> Macrophages_1 (head 5)": "#E63946",        # Bright red
    "DCIS_1 -> Macrophages_1 (head 5)": "#FF6B6B",               # Medium red
    "Macrophages_2 -> CD8+_T_Cells (head 5)": "#FF8E53",         # Red-orange
    "Macrophages_2 -> CD4+_T_Cells (head 5)": "#FFA07A",         # Light salmon
    
    # Immune-Immune interactions (Green palette)
    "CD4+_T_Cells -> CD8+_T_Cells (head 4)": "#2D5016",         # Dark forest green
    "CD4+_T_Cells -> CD8+_T_Cells (head 5)": "#1A3009",         # Very dark forest green
    "CD8+_T_Cells -> Macrophages_1 (head 4)": "#2D5016",         # Dark forest green
    "CD8+_T_Cells -> Macrophages_1 (head 5)": "#4F772D",         # Olive green
    "CD4+_T_Cells -> B_Cells (head 4)": "#52734D",               # Medium forest green
    "CD4+_T_Cells -> B_Cells (head 5)": "#90A955",               # Light olive green
    "B_Cells -> CD4+_T_Cells (head 5)": "#B5E48C",               # Soft green
}

# Calculate median length_scale for each interaction to determine order
median_order = length_scale_df.groupby('interaction')['length_scale'].median().sort_values().index.tolist()

# Create boxplot with color_map palette
sns.violinplot(
    data=length_scale_df,
    x="length_scale",
    y="interaction",
    palette=color_map,
    order=median_order,
)
plt.xlim(0, 50)
plt.title("Length Scales by Interaction (Color Map Palette)")
plt.tight_layout()
plt.savefig(f"figures/xenium_sample1_proseg_length_scales_comparison_boxplot_colormap.svg")
plt.show()

# Create boxplot with interaction_colors palette
sns.violinplot(
    data=length_scale_df,
    x="length_scale",
    y="interaction",
    palette=interaction_colors,
    order=median_order,
)
plt.xlim(0, 50)
plt.title("Length Scales by Interaction (Interaction Colors Palette)")
plt.tight_layout()
plt.savefig(f"figures/xenium_sample1_proseg_length_scales_comparison_boxplot_interaction_colors.svg")
plt.show()

# %% Plot the expression of a gene of interest and for a cell type of interest on the spatial plot
gene = "GNLY"
cell_type = "Endothelial"

adata_sub = adata[adata.obs[labels_key] == cell_type].copy()
plot_df = adata_sub.obsm["spatial"].copy()
plot_df["expression"] = adata_sub.X[:, adata_sub.var_names.get_loc(gene)].toarray()

plt.figure(figsize=(20, 6))
scatter = sns.scatterplot(
    plot_df, x="X", y="Y", hue="expression", alpha=0.7, s=8, palette="viridis"
)
ax = plt.gca()
ax.set_title(f"Expression of {gene} in {cell_type}")
norm = plt.Normalize(plot_df["expression"].min(), plot_df["expression"].max())
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label=f"{gene} expression")
ax.set_xlabel("X")
ax.set_ylabel("Y")
scatter.get_legend().remove()
plt.show()

# %% For each receiver cell type, plot volcano plot with the diff on x and z values on y
for receiver_ct in adata.obs[labels_key].unique():
    print(f"Computing ablation scores for {receiver_ct}")
    ablation_scores = model.get_neighbor_ablation_scores(
        adata=adata,
        cell_type=receiver_ct,
        compute_z_value=True,
    )

    z_score_cols = [col for col in ablation_scores._ablation_scores_df.columns if col.endswith("_z_value")]
    diff_cols = [col for col in ablation_scores._ablation_scores_df.columns if col.endswith("_diff")]
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get sender cell types (remove receiver from the list)
    sender_cell_types = [col.replace("_diff", "") for col in diff_cols]
    
    # Plot each sender cell type
    print(f"Plotting {len(sender_cell_types)} sender cell types")
    for i, sender_ct in enumerate(sender_cell_types):
        diff_col = f"{sender_ct}_diff"
        z_col = f"{sender_ct}_z_value"
        
        x_data = ablation_scores._ablation_scores_df[diff_col]
        y_data = np.abs(ablation_scores._ablation_scores_df[z_col])
        
        ax.scatter(
            x_data, 
            y_data, 
            c=CELL_TYPE_PALETTE.get(sender_ct, '#808080'),  # Use palette color or gray as fallback
            label=sender_ct, 
            alpha=0.6, 
            s=20
        )
    
    # Set labels and title
    ax.set_xlabel('Neighbor Contribution (diff)')
    ax.set_ylabel('Z-value')
    ax.set_title(f'Diff vs Z-value for {receiver_ct}')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Sender Cell Type')
    
    # Add reference lines
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"figures/diff_vs_zvalue_scatter_{receiver_ct}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"figures/diff_vs_zvalue_scatter_{receiver_ct}.svg", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Plotted volcano plot for {receiver_ct}")
    
    

# %%
# Abundance vs attention correlation check
# ########################################
# for a cell type pair, at a given distance, compute abundance of the neighbor cell type
# then compute the counterfactual values and plot the average against the abundance
def compute_abundance_vs_attention(
    model,
    adata,
    index_cell_type,
    neighbor_cell_type,
    distance,
):
    labels_key = model.adata_manager.get_state_registry(
        REGISTRY_KEYS.LABELS_KEY
    ).original_key
    ct_idxs = np.where(adata.obs[labels_key] == index_cell_type)[0]
    empirical_attention_module: AMICIAttentionModule = model.get_attention_patterns(
        adata=adata,
        indices=ct_idxs,
    )
    nn_idxs = empirical_attention_module._nn_idxs_df
    nn_dists = empirical_attention_module._nn_dists_df
    nn_labels = np.reshape(
        adata[nn_idxs.values.flatten()].obs[labels_key].values, nn_idxs.shape
    )
    pos_mask = (nn_labels == neighbor_cell_type) & (nn_dists.values <= distance)
    avg_abundance = pos_mask.sum(axis=1).mean()
    # fetch attention scores for each head
    head_neighbor_attention_cols = []
    for head_idx in range(model.module.n_heads):
        head_attention_scores = (
            empirical_attention_module._attention_patterns_df.loc[
                empirical_attention_module._attention_patterns_df["head"] == head_idx
            ]
            .drop(columns=["label", "head"])
            .set_index("cell_idx")
        )
        head_attention_scores = head_attention_scores.loc[nn_idxs.index]
        head_neighbor_attention_scores = pd.Series(
            np.where(pos_mask, head_attention_scores.values, 0).sum(axis=1),
            index=nn_idxs.index,
            name=f"head_{head_idx}",
        )
        head_neighbor_attention_cols.append(head_neighbor_attention_scores)
    head_neighbor_attention_df = pd.concat(head_neighbor_attention_cols, axis=1)
    sum_empirical_neighbor_attention = head_neighbor_attention_df.sum(axis=1).mean()

    neighbor_ct_idxs = np.where(adata.obs[labels_key] == neighbor_cell_type)[0]
    counterfactual_attention_patterns: AMICICounterfactualAttentionModule = (
        model.get_counterfactual_attention_patterns(
            cell_type=index_cell_type,
            adata=adata,
            head_idxs=None,
            indices=neighbor_ct_idxs,
        )
    )
    cf_attention_dfs = []
    for head_idx in range(model.module.n_heads):
        cf_attention_df = counterfactual_attention_patterns.calculate_counterfactual_attention_at_distances(
            head_idx=head_idx,
            distances=[distance],
        )[
            ["neighbor_idx", f"head_{head_idx}"]
        ]
        cf_attention_dfs.append(cf_attention_df)
    cf_attention_df = reduce(
        lambda left, right: pd.merge(left, right, on="neighbor_idx", how="inner"),
        cf_attention_dfs,
    )
    cf_attention_df["max_attention"] = cf_attention_df[
        [f"head_{i}" for i in range(model.module.n_heads)]
    ].max(axis=1)
    cf_attention_df["mean_attention"] = cf_attention_df[
        [f"head_{i}" for i in range(model.module.n_heads)]
    ].mean(axis=1)
    average_max_attention = cf_attention_df["max_attention"].mean()
    average_mean_attention = cf_attention_df["mean_attention"].mean()

    return {
        "avg_abundance": avg_abundance,
        "average_max_attention": average_max_attention,
        "average_mean_attention": average_mean_attention,
        "sum_empirical_neighbor_attention": sum_empirical_neighbor_attention,
    }


# %%
distances = [10, 20, 30]
for distance in distances:
    labels_key = model.adata_manager.get_state_registry(
        REGISTRY_KEYS.LABELS_KEY
    ).original_key
    cell_types = adata.obs[labels_key].unique()
    out_rows = []
    for index_ct in cell_types:
        for neighbor_ct in cell_types:
            if index_ct == neighbor_ct:
                continue
            outs = compute_abundance_vs_attention(
                model,
                adata,
                index_ct,
                neighbor_ct,
                distance,
            )
            out_rows.append(
                {
                    "index_ct": index_ct,
                    "neighbor_ct": neighbor_ct,
                    "avg_abundance": outs["avg_abundance"],
                    "average_max_attention": outs["average_max_attention"],
                    "average_mean_attention": outs["average_mean_attention"],
                    "sum_empirical_neighbor_attention": outs[
                        "sum_empirical_neighbor_attention"
                    ],
                }
            )
    out_df = pd.DataFrame.from_records(out_rows)
    out_df.head()
    fig, axs = plt.subplots(1, 3, figsize=(25, 5))
    sns.scatterplot(
        out_df,
        x="avg_abundance",
        y="average_max_attention",
        hue="index_ct",
        ax=axs[0],
        legend=False,
    )
    sns.scatterplot(
        out_df,
        x="avg_abundance",
        y="average_mean_attention",
        hue="index_ct",
        ax=axs[1],
    )
    sns.scatterplot(
        out_df,
        x="avg_abundance",
        y="sum_empirical_neighbor_attention",
        hue="index_ct",
        ax=axs[2],
    )
    plt.tight_layout()
    plt.suptitle(f"Evaluated at distance: {distance}")
    plt.savefig(f"figures/xenium_sample1_proseg_abundance_vs_attention_{distance}.png")
    plt.show()
    plt.clf()
# %%
