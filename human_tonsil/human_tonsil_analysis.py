# %% Import libraries
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import seaborn as sns

from amici import AMICI
from amici.interpretation._ablation_module import AMICIAblationModule

# %% Seed everything
seed = 40
pl.seed_everything(seed)

# %% Load data
labels_key = "celltype_manual_fine"
data_date = "2026-07-25"
model_date = "2026-07-26"
adata = sc.read_h5ad(f"./data/human_tonsil_filtered_{data_date}.h5ad")
adata_train = sc.read_h5ad(f"./data/human_tonsil_filtered_train_{data_date}.h5ad")
adata_test = sc.read_h5ad(f"./data/human_tonsil_filtered_test_{data_date}.h5ad")

figures_dir = "./figures/human_tonsil_analysis"
os.makedirs(figures_dir, exist_ok=True)
ablation_cache_dir = os.path.join(figures_dir, "cached_ablation_scores")
os.makedirs(ablation_cache_dir, exist_ok=True)

saved_models_dir = f"saved_models/human_tonsil_sweep_{data_date}_model_{model_date}"
wandb_run_id = "4303v0zu"
wandb_sweep_id = "0f9rk0na"
model_seed = seed
model_path = os.path.join(
    saved_models_dir,
    f"human_tonsil_{model_seed}_sweep_{wandb_sweep_id}_{wandb_run_id}_params_{model_date}",
)
model_path = os.environ.get("HUMAN_TONSIL_AMICI_MODEL_DIR", model_path)

CELL_TYPE_PALETTE = dict(
    zip(
        sorted(adata.obs[labels_key].astype(str).unique()),
        sns.color_palette("tab20", n_colors=adata.obs[labels_key].nunique()).as_hex(),
    )
)

print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
print(f"Train cells: {adata_train.n_obs}; test cells: {adata_test.n_obs}")
print(f"Loading model from {model_path}")
print(adata.obs[labels_key].value_counts().to_string())


# %% Visualize spatial distribution
def visualize_spatial_distribution(adata, labels_key="celltype_manual_fine"):
    plt.figure(figsize=(16, 10))
    plot_df = pd.DataFrame(adata.obsm["spatial"], index=adata.obs_names, columns=["X", "Y"])
    plot_df[labels_key] = adata.obs[labels_key].astype(str).values
    plot_df["train_test"] = adata.obs["train_test_split"].astype(str).values

    sns.scatterplot(
        plot_df,
        x="X",
        y="Y",
        hue=labels_key,
        alpha=0.65,
        s=2,
        palette=CELL_TYPE_PALETTE,
        linewidth=0,
    )

    test_df = plot_df[plot_df["train_test"] == "test"]
    if len(test_df) > 0:
        min_x, max_x = test_df["X"].min(), test_df["X"].max()
        min_y, max_y = test_df["Y"].min(), test_df["Y"].max()
        padding = 20
        rect = plt.Rectangle(
            (min_x - padding, min_y - padding),
            (max_x - min_x) + 2 * padding,
            (max_y - min_y) + 2 * padding,
            fill=False,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Test Region",
        )
        plt.gca().add_patch(rect)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Human tonsil spatial plot for analysis")
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        markerscale=4,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/human_tonsil_spatial_subset_analysis.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{figures_dir}/human_tonsil_spatial_subset_analysis.svg", dpi=300, bbox_inches="tight")
    plt.show()


cell_type_sub = [
    "Naive B cell (mantle zone)",
    "GC B cell (resting)",
    "GC B cell (cycling)",
    "CD4 T cell",
    "CD8 T cell",
    "Tfh cell",
    "Plasma cell (mature)",
    "Plasmablast (proliferating)",
    "Macrophage (C1Q+)",
    "Macrophage (GPNMB+)",
    "cDC2",
    "pDC",
    "Follicular dendritic cell",
    "Fibroblast",
    "Vascular (endothelial + pericyte)",
    "Lymphatic endothelial cell",
    "Mast cell",
    "Squamous epithelium",
    "Crypt epithelium",
]
cell_type_sub = [ct for ct in cell_type_sub if ct in adata.obs[labels_key].astype(str).unique()]
visualize_spatial_distribution(adata[adata.obs[labels_key].isin(cell_type_sub)].copy())

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

# %% Compute neighbor ablation scores
ablation_cache_path = os.path.join(ablation_cache_dir, "all_cell_types_ablation_scores.pkl")
if os.path.exists(ablation_cache_path):
    ablation_residuals = AMICIAblationModule.load_object(ablation_cache_path)
else:
    ablation_residuals = model.get_neighbor_ablation_scores(
        adata=adata,
        compute_z_value=True,
    )
    ablation_residuals.save_object(ablation_cache_path)

# %% Plot the interaction weight matrix as a heatmap
ablation_residuals.plot_interaction_weight_heatmap(
    save_png=True,
    save_svg=True,
    save_dir=figures_dir,
)
plt.close("all")

# %% Grab 80 quantile to get threshold for weight matrix
interaction_weight_matrix_df = ablation_residuals._get_interaction_weight_matrix()
interaction_weight_matrix = interaction_weight_matrix_df.values.flatten()
quantile = 0.80
weight_threshold = np.quantile(interaction_weight_matrix, quantile)
print(f"{quantile} quantile threshold: {weight_threshold:.2f}")

plt.figure(figsize=(8, 5))
sns.kdeplot(x=interaction_weight_matrix)
plt.title("Distribution of interaction weights")
plt.xlabel("Interaction weight")
plt.ylabel("Density")
plt.axvline(weight_threshold, color="r", linestyle="--", label=f"{quantile} quantile threshold: {weight_threshold:.2f}")
plt.legend()
plt.savefig(f"{figures_dir}/human_tonsil_interaction_weight_distribution.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{figures_dir}/human_tonsil_interaction_weight_distribution.svg", dpi=300, bbox_inches="tight")
plt.show()

# %% Plot interaction matrix for selected cell types
cell_type_sub = [ct for ct in cell_type_sub if ct in interaction_weight_matrix_df.index]
interaction_weight_matrix_sub_df = interaction_weight_matrix_df.loc[cell_type_sub, cell_type_sub]

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
plt.savefig(f"{figures_dir}/human_tonsil_interaction_weight_clustermap.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{figures_dir}/human_tonsil_interaction_weight_clustermap.svg", dpi=300, bbox_inches="tight")
plt.show()

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
plt.savefig(f"{figures_dir}/human_tonsil_interaction_weight_clustermap_subset.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{figures_dir}/human_tonsil_interaction_weight_clustermap_subset.svg", dpi=300, bbox_inches="tight")
plt.show()

# %% Plot directed interaction graphs
ablation_residuals.plot_interaction_directed_graph(
    significance_threshold=0.05,
    weight_threshold=weight_threshold,
    palette=CELL_TYPE_PALETTE,
    save_svg=True,
    save_dir=figures_dir,
)
ablation_residuals.plot_interaction_directed_graph(
    cell_type_sub=cell_type_sub,
    significance_threshold=0.05,
    weight_threshold=weight_threshold,
    palette=CELL_TYPE_PALETTE,
    save_svg=True,
    save_dir=figures_dir,
)

# %% Compute and plot variance scores for heads
expl_variance_scores = model.get_expl_variance_scores(
    adata,
    cell_type_sub=cell_type_sub,
    run_permutation_test=False,
)
expl_variance_scores.plot_explained_variance_barplot(
    palette=CELL_TYPE_PALETTE,
    cell_type_sub=cell_type_sub,
    wandb_log=False,
    save_png=True,
    show=True,
)

# %% Get and plot attention patterns
attention_patterns = model.get_attention_patterns(
    adata,
    batch_size=32,
)
attention_patterns.plot_attention_summary(
    cell_type_sub=cell_type_sub,
    palette=CELL_TYPE_PALETTE,
    wandb_log=False,
    save_png=True,
    show=True,
)

# %% Define a target cell type for focused interpretation
receiver_ct = "GC B cell (resting)"
sender_cts = [
    "Tfh cell",
    "Follicular dendritic cell",
    "Mast cell",
    "Squamous epithelium (activated)",
    "Macrophage (GPNMB+)"
]
sender_cts = [ct for ct in sender_cts if ct in adata.obs[labels_key].astype(str).unique()]
head_idx = min(4, model.module.n_heads - 1)

# %% Plot attention summary for the receiver cell type
attention_patterns.plot_attention_summary(
    cell_type_sub=[receiver_ct],
    plot_histogram=False,
    palette=CELL_TYPE_PALETTE,
    wandb_log=False,
    show=True,
    save_png=True,
)

# %% Plot neighbor cell type ablation scores
ablation_ct_residuals = model.get_neighbor_ablation_scores(
    adata=adata,
    cell_type=receiver_ct,
    ablated_neighbor_ct_sub=sender_cts,
    compute_z_value=True,
)
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

# %% Compute counterfactual attention patterns for a query cell type
counterfactual_attention_patterns = model.get_counterfactual_attention_patterns(
    cell_type=receiver_ct,
    adata=adata,
)

# %% Plot length scales and counterfactual attention summaries
counterfactual_attention_patterns.plot_length_scale_distribution(
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

neighbor_ct_sub = [ct for ct in sender_cts if ct != receiver_ct]
for idx in range(model.module.n_heads):
    counterfactual_attention_patterns.plot_counterfactual_attention_summary(
        head_idx=idx,
        distances=np.linspace(0, 150, num=25),
        neighbor_ct_sub=neighbor_ct_sub,
        palette=CELL_TYPE_PALETTE,
        save_png=True,
        wandb_log=False,
        show=True,
        save_dir=figures_dir,
    )

counterfactual_attention_patterns.plot_length_scale_distribution(
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
