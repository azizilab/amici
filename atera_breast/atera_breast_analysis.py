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
model_date = "2026-07-28"
adata = sc.read_h5ad(f"./data/atera_breast_filtered_{data_date}.h5ad")
adata_train = sc.read_h5ad(f"./data/atera_breast_filtered_train_{data_date}.h5ad")
adata_test = sc.read_h5ad(f"./data/atera_breast_filtered_test_{data_date}.h5ad")

figures_dir = "./figures/atera_breast_analysis"
os.makedirs(figures_dir, exist_ok=True)
ablation_cache_dir = os.path.join(figures_dir, "cached_ablation_scores")
os.makedirs(ablation_cache_dir, exist_ok=True)

saved_models_dir = f"saved_models/atera_breast_sweep_{data_date}_model_{model_date}"
wandb_run_id = "zhtraubt"
wandb_sweep_id = "25xpxkuk"
model_seed = 33
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


def _get_gene_expression(adata, gene):
    """Return a dense expression vector for a gene."""
    if gene not in adata.var_names:
        raise KeyError(f"{gene} not found in adata.var_names.")
    gene_expr = adata[:, gene].X
    if hasattr(gene_expr, "toarray"):
        gene_expr = gene_expr.toarray()
    return np.asarray(gene_expr).reshape(-1)


def plot_spatial_feature(
    adata,
    values,
    title,
    colorbar_label,
    save_prefix,
    point_size=2,
    cmap="viridis",
):
    """Plot a continuous feature on spatial coordinates."""
    plot_df = pd.DataFrame(adata.obsm["spatial"], index=adata.obs_names, columns=["X", "Y"])
    plot_df["value"] = values

    plt.figure(figsize=(20, 7))
    plt.scatter(
        plot_df["X"],
        plot_df["Y"],
        c=plot_df["value"],
        s=point_size,
        cmap=cmap,
        linewidths=0,
        alpha=0.85,
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label(colorbar_label)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/{save_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{figures_dir}/{save_prefix}.svg", dpi=300, bbox_inches="tight")
    plt.show()


def get_total_sender_attention_to_receiver(attention_patterns, receiver_type, sender_type):
    """Sum attention from one sender cell type into each receiver cell."""
    attention_df = attention_patterns._attention_patterns_df.copy()
    nn_df = attention_patterns._nn_idxs_df
    neighbor_cols = [col for col in attention_df.columns if col.startswith("neighbor_")]

    receiver_cells = adata.obs_names[adata.obs[labels_key].astype(str) == receiver_type]
    receiver_attention_df = attention_df[attention_df["cell_idx"].isin(receiver_cells)]
    receiver_nn_df = nn_df.loc[receiver_cells]
    neighbor_labels_df = receiver_nn_df.apply(lambda col: adata.obs.loc[col, labels_key].astype(str).values)

    attention_by_cell = pd.Series(0.0, index=adata.obs_names)
    for _, row in receiver_attention_df.iterrows():
        receiver_idx = row["cell_idx"]
        sender_mask = neighbor_labels_df.loc[receiver_idx, neighbor_cols].values == sender_type
        attention_by_cell.loc[receiver_idx] += row[neighbor_cols].astype(float).values[sender_mask].sum()
    return attention_by_cell


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

# %% Plot spatial maps for T cell attention to CAFs and marker expression
t_cell_to_caf_attention = get_total_sender_attention_to_receiver(
    attention_patterns,
    receiver_type="CAFs",
    sender_type="T_Cells",
)
plot_spatial_feature(
    adata,
    t_cell_to_caf_attention,
    title="Total T cell attention to CAF receiver cells",
    colorbar_label="Total T cell attention",
    save_prefix="atera_breast_t_cell_to_caf_total_attention_spatial",
    point_size=3,
    cmap="coolwarm",
)

for gene in ["EMILIN1", "C3"]:
    plot_spatial_feature(
        adata,
        _get_gene_expression(adata, gene),
        title=f"{gene} spatial expression",
        colorbar_label=f"{gene} expression",
        save_prefix=f"atera_breast_{gene}_spatial_expression",
        point_size=3,
        cmap="viridis",
    )

# %% Define the target cell type of interest and max expl variance head
receiver_ct = "Luminal_DCIS"
sender_cts = ["Myoepithelial", "CAFs"]
sender_cts = [ct for ct in sender_cts if ct in adata.obs[labels_key].unique()]
head_idx = min(4, model.module.n_heads - 1)

# %% Plot neighbor cell type neighbor ablation scores
ablation_ct_residuals = model.get_neighbor_ablation_scores(
    adata=adata,
    # head_idx=head_idx,
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
    # flag_segmentation_artifacts=True,
    # segmentation_adata=adata,
    color_by="diff",
    size_by="z_value",
    n_top_genes=8,
    min_size_by=-10,
    step=15,
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

# %% Subcluster CAFs using CAF-specific HVGs to check EMILIN1/C3 programs
caf_adata_full = adata[adata.obs[labels_key].astype(str) == "CAFs"].copy()
caf_adata = caf_adata_full.copy()
caf_n_hvgs = min(500, caf_adata.n_vars)
print(f"CAF cells for subclustering: {caf_adata.n_obs}")

if caf_adata.n_obs > 0:
    sc.pp.highly_variable_genes(caf_adata, n_top_genes=caf_n_hvgs)
    caf_adata = caf_adata[:, caf_adata.var["highly_variable"]].copy()
    print(f"Clustering CAFs with top {caf_adata.n_vars} CAF-specific HVGs")

    sc.pp.pca(caf_adata, n_comps=min(30, caf_adata.n_obs - 1, caf_adata.n_vars - 1))
    sc.pp.neighbors(
        caf_adata,
        n_neighbors=min(20, caf_adata.n_obs - 1),
        n_pcs=min(20, caf_adata.obsm["X_pca"].shape[1]),
    )
    sc.tl.umap(caf_adata, random_state=seed)
    sc.tl.leiden(caf_adata, resolution=0.35, key_added="caf_subcluster")

    caf_adata_full.obs["caf_subcluster"] = caf_adata.obs["caf_subcluster"].astype(str)
    caf_adata_full.obsm["X_umap"] = caf_adata.obsm["X_umap"]

    caf_palette = dict(
        zip(
            sorted(caf_adata_full.obs["caf_subcluster"].astype(str).unique()),
            sns.color_palette("tab10", caf_adata_full.obs["caf_subcluster"].nunique()).as_hex(),
        )
    )

    for color in ["caf_subcluster", "EMILIN1", "C3"]:
        if color in caf_adata_full.obs or color in caf_adata_full.var_names:
            sc.pl.umap(
                caf_adata_full,
                color=color,
                palette=caf_palette if color == "caf_subcluster" else None,
                show=False,
            )
            plt.savefig(
                os.path.join(figures_dir, f"atera_breast_caf_umap_{color}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(figures_dir, f"atera_breast_caf_umap_{color}.svg"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.show()

    caf_spatial_df = pd.DataFrame(
        caf_adata_full.obsm["spatial"],
        index=caf_adata_full.obs_names,
        columns=["X", "Y"],
    )
    caf_spatial_df["caf_subcluster"] = caf_adata_full.obs["caf_subcluster"].astype(str).values

    plt.figure(figsize=(10, 8))
    for cluster, cluster_df in caf_spatial_df.groupby("caf_subcluster"):
        plt.scatter(
            cluster_df["X"],
            cluster_df["Y"],
            s=4,
            linewidths=0,
            alpha=0.85,
            color=caf_palette[cluster],
            label=f"CAF {cluster}",
        )
    plt.gca().invert_yaxis()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("CAF expression subclusters")
    plt.legend(markerscale=3, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.savefig(
        os.path.join(figures_dir, "atera_breast_caf_subclusters_spatial.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(figures_dir, "atera_breast_caf_subclusters_spatial.svg"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    for gene in ["EMILIN1", "C3"]:
        if gene in caf_adata_full.var_names:
            plot_spatial_feature(
                caf_adata_full,
                _get_gene_expression(caf_adata_full, gene),
                title=f"CAF {gene} spatial expression",
                colorbar_label=f"{gene} expression",
                save_prefix=f"atera_breast_caf_{gene}_spatial_expression",
                point_size=4,
                cmap="viridis",
            )

    caf_marker_genes = [gene for gene in ["EMILIN1", "C3"] if gene in caf_adata_full.var_names]
    if caf_marker_genes:
        sc.pl.violin(
            caf_adata_full,
            keys=caf_marker_genes,
            groupby="caf_subcluster",
            stripplot=False,
            rotation=45,
            show=False,
        )
        plt.savefig(
            os.path.join(figures_dir, "atera_breast_caf_subcluster_emilin1_c3_violin.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(figures_dir, "atera_breast_caf_subcluster_emilin1_c3_violin.svg"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    c3_emilin1_clusters = {"0", "5"}
    caf_adata_full.obs["caf_emilin1_c3_group"] = caf_adata_full.obs[
        "caf_subcluster"
    ].astype(str).map(
        lambda cluster: "C3+EMILIN1+"
        if cluster in c3_emilin1_clusters
        else "C3-EMILIN1+"
    )
    caf_group_palette = {
        "C3+EMILIN1+": "#1f77b4",
        "C3-EMILIN1+": "#d62728",
    }
    print("CAF EMILIN1/C3 group counts:")
    print(caf_adata_full.obs["caf_emilin1_c3_group"].value_counts())

    sc.pl.umap(
        caf_adata_full,
        color="caf_emilin1_c3_group",
        palette=caf_group_palette,
        show=False,
    )
    plt.savefig(
        os.path.join(figures_dir, "atera_breast_caf_umap_emilin1_c3_group.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(figures_dir, "atera_breast_caf_umap_emilin1_c3_group.svg"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    caf_spatial_df["caf_emilin1_c3_group"] = caf_adata_full.obs[
        "caf_emilin1_c3_group"
    ].values
    plt.figure(figsize=(10, 8))
    for caf_group, group_df in caf_spatial_df.groupby("caf_emilin1_c3_group"):
        plt.scatter(
            group_df["X"],
            group_df["Y"],
            s=4,
            linewidths=0,
            alpha=0.85,
            color=caf_group_palette[caf_group],
            label=caf_group,
        )
    plt.gca().invert_yaxis()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("CAF EMILIN1/C3 groups")
    plt.legend(markerscale=3, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.savefig(
        os.path.join(figures_dir, "atera_breast_caf_emilin1_c3_group_spatial.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(figures_dir, "atera_breast_caf_emilin1_c3_group_spatial.svg"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # %% Compare T-cell senders highly attended by each CAF EMILIN1/C3 group
    sender_type = "T_Cells"
    caf_group_a = "C3+EMILIN1+"
    caf_group_b = "C3-EMILIN1+"

    attention_df = attention_patterns._attention_patterns_df.copy()
    nn_df = attention_patterns._nn_idxs_df
    neighbor_cols = [col for col in attention_df.columns if col.startswith("neighbor_")]
    t_cell_ids = adata.obs_names[adata.obs[labels_key].astype(str) == sender_type]
    sender_attention = pd.DataFrame(0.0, index=t_cell_ids, columns=[caf_group_a, caf_group_b])

    caf_receiver_groups = caf_adata_full.obs["caf_emilin1_c3_group"].astype(str)
    caf_receiver_ids = caf_receiver_groups.index
    receiver_attention_df = attention_df[attention_df["cell_idx"].isin(caf_receiver_ids)]
    for _, row in receiver_attention_df.iterrows():
        receiver_idx = row["cell_idx"]
        receiver_group = caf_receiver_groups.loc[receiver_idx]
        if receiver_group not in [caf_group_a, caf_group_b]:
            continue

        nn_ids = nn_df.loc[receiver_idx, neighbor_cols].values
        nn_labels = adata.obs.loc[nn_ids, labels_key].astype(str).values
        attention_values = row[neighbor_cols].astype(float).values
        t_sender_ids = nn_ids[nn_labels == sender_type]
        t_sender_attention = attention_values[nn_labels == sender_type]

        for sender_idx, attention_value in zip(t_sender_ids, t_sender_attention):
            sender_attention.loc[sender_idx, receiver_group] += attention_value

    threshold_a = sender_attention.loc[
        sender_attention[caf_group_a] > 0,
        caf_group_a,
    ].quantile(0.75)
    threshold_b = sender_attention.loc[
        sender_attention[caf_group_b] > 0,
        caf_group_b,
    ].quantile(0.75)

    sender_attention["caf_sender_group"] = "Other T-cell senders"
    sender_attention.loc[
        (sender_attention[caf_group_a] >= threshold_a)
        & (sender_attention[caf_group_a] > sender_attention[caf_group_b]),
        "caf_sender_group",
    ] = f"High attention to {caf_group_a} CAFs"
    sender_attention.loc[
        (sender_attention[caf_group_b] >= threshold_b)
        & (sender_attention[caf_group_b] > sender_attention[caf_group_a]),
        "caf_sender_group",
    ] = f"High attention to {caf_group_b} CAFs"
    sender_attention.to_csv(
        os.path.join(figures_dir, "atera_breast_t_cell_sender_caf_group_attention_scores.csv")
    )

    de_group_a = f"High attention to {caf_group_a} CAFs"
    de_group_b = f"High attention to {caf_group_b} CAFs"
    selected_t_cells = sender_attention.index[
        sender_attention["caf_sender_group"].isin([de_group_a, de_group_b])
    ]
    group_counts = sender_attention.loc[selected_t_cells, "caf_sender_group"].value_counts()
    print("T-cell sender attention group counts:")
    print(group_counts)

    if len(group_counts) == 2 and group_counts.min() >= 3:
        t_de_adata = adata[selected_t_cells].copy()
        t_de_adata.obs["caf_sender_group"] = sender_attention.loc[
            selected_t_cells,
            "caf_sender_group",
        ].values

        sc.tl.rank_genes_groups(
            t_de_adata,
            groupby="caf_sender_group",
            groups=[de_group_a],
            reference=de_group_b,
            method="wilcoxon",
        )
        t_cell_de_df = sc.get.rank_genes_groups_df(t_de_adata, group=de_group_a)
        t_cell_de_df.to_csv(
            os.path.join(figures_dir, "atera_breast_t_cell_sender_caf_group_de.csv"),
            index=False,
        )

        heatmap_genes = t_cell_de_df.sort_values("pvals_adj").head(20)["names"].tolist()
        heatmap_logfc = t_cell_de_df.set_index("names").loc[
            heatmap_genes,
            "logfoldchanges",
        ]
        heatmap_df = pd.DataFrame(
            [heatmap_logfc.values, -heatmap_logfc.values],
            index=[de_group_a, de_group_b],
            columns=heatmap_genes,
        )
        max_abs_logfc = float(np.nanmax(np.abs(heatmap_df.values)))
        max_abs_logfc = max(max_abs_logfc, 1.0)

        plt.figure(figsize=(max(7, 0.35 * len(heatmap_genes)), 3.2))
        sns.heatmap(
            heatmap_df,
            cmap="bwr",
            center=0,
            vmin=-max_abs_logfc,
            vmax=max_abs_logfc,
            cbar_kws={"label": "log fold change"},
            linewidths=0.4,
            linecolor="white",
        )
        plt.xlabel("Gene")
        plt.ylabel("")
        plt.title("T-cell sender DEGs by attended CAF group")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.savefig(
            os.path.join(figures_dir, "atera_breast_t_cell_sender_caf_group_de_heatmap.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(figures_dir, "atera_breast_t_cell_sender_caf_group_de_heatmap.svg"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
    else:
        print("Skipping T-cell DEG: one CAF-attended sender group has fewer than 3 cells.")

# %%
