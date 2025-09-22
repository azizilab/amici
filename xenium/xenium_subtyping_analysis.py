# %% Import libraries
from pathlib import Path
import os
import scanpy as sc
import pytorch_lightning as pl
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from sklearn.decomposition import PCA
from einops import rearrange

from scipy.stats import ttest_ind
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
    "Stromal": "#968253",
    "B_Cells": "#c5a9e8",
    "Mast_Cells": "#947b79",
    "Perivascular-Like": "#872727",
    "Endothelial": "#277987",
}

SUBTYPING_PALETTE = {
    "Low-interacting": "#73fb88",
    "High-interacting": "#009e1d",
    "Near-interacting-cell": "#1192dc",
    "Far-interacting-cell": "#91dcff",
}

# %% Seed everything
seed = 38
pl.seed_everything(seed)

# %% Load data
model_seed = 18
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

saved_models_dir = f"./saved_models/xenium_sample1_proseg_sweep_{data_date}_model_{model_date}"
wandb_run_id = "te7pkv3z"
wandb_sweep_id = "g3mucw4s"
model_path = os.path.join(
    saved_models_dir,
    f"xenium_{model_seed}_sweep_{wandb_sweep_id}_{wandb_run_id}_params_{model_date}",
)
# %%
# correct spatial coordinates
adata.obs["x"] = adata.obsm["spatial"]["X"]
adata.obs["y"] = adata.obsm["spatial"]["Y"]

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
# %%
# Set Analysis Parameters
distance = 15
receiver_ct = "Invasive_Tumor"
sender_ct = "CD8+_T_Cells"
quantile = 0.9

# %%
# Compute Counterfactual Attention Patterns between sender and receiver at distance
sender_indices = np.where(adata.obs[labels_key] == sender_ct)[0]
figures_dir = Path(f"figures/{receiver_ct}_from_{sender_ct}")
figures_dir.mkdir(parents=True, exist_ok=True)
counterfactual_attention_patterns = model.get_counterfactual_attention_patterns(
    cell_type=receiver_ct,
    adata=adata,
    head_idxs=None,
    indices=sender_indices,
)

counterfactual_attention_dfs = []
for head_idx in range(model.module.n_heads):
    counterfactual_attention_dfs.append(
        counterfactual_attention_patterns.calculate_counterfactual_attention_at_distances(
            head_idx=head_idx,
            distances=[distance],
        )[
            ["neighbor_idx", f"head_{head_idx}"]
        ].set_index(
            "neighbor_idx"
        )
    )
counterfactual_attention_df = pd.concat(counterfactual_attention_dfs, axis=1)

# %%
# Select head with highest variance
head_col_of_note = counterfactual_attention_df.std(axis=0).idxmax()
# head_col_of_note = 
head_idx_of_note = int(head_col_of_note.split("_")[1])
print(f"Highest variance head: {head_col_of_note}")
# %%
# Plot histogram of attention scores for selected head
counterfactual_attention_df[head_col_of_note].hist(bins=100)
# %%
# Select threshold for high attention
head_attention_threshold = counterfactual_attention_df[head_col_of_note].quantile(
    quantile
)
print(f"Head attention threshold: {head_attention_threshold}")
high_attention_df = counterfactual_attention_df[
    counterfactual_attention_df[head_col_of_note] > head_attention_threshold
]

# %%
attention_matrix = counterfactual_attention_df.values
pca = PCA(n_components=2)
pca_result = pca.fit_transform(attention_matrix)

pca_df = pd.DataFrame(
    data=pca_result, columns=["PC1", "PC2"], index=counterfactual_attention_df.index
)

pca_df["cell_type"] = adata.obs[labels_key].iloc[pca_df.index].values

plt.figure(figsize=(10, 8))
cell_types = pca_df["cell_type"].unique()

for cell_type in cell_types:
    subset = pca_df[pca_df["cell_type"] == cell_type]
    plt.scatter(subset["PC1"], subset["PC2"], label=cell_type, alpha=0.7)

plt.title(
    f"PCA of Counterfactual Attention Patterns\n{receiver_ct} receiving at distance {distance}"
)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.grid(alpha=0.3)
plt.savefig(
    figures_dir / f"counterfactual_attention_pca_{receiver_ct}_{distance}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
pca_df["cluster"] = pca_df.index.isin(high_attention_df.index).astype(int)
print(pca_df["cluster"].value_counts())

plt.figure(figsize=(12, 10))
for cluster_id in range(2):
    cluster_data = pca_df[pca_df["cluster"] == cluster_id]
    plt.scatter(
        cluster_data["PC1"],
        cluster_data["PC2"],
        label=f"Cluster {cluster_id}",
        alpha=0.7,
    )

plt.title(f"PCA (2 Clusters)\n{receiver_ct} receiving at distance {distance}")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.grid(alpha=0.3)
plt.savefig(
    figures_dir / f"counterfactual_attention_pca_kmeans_{receiver_ct}_{distance}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
# %%
plt.figure(figsize=(15, 10))

head_columns = [
    col for col in counterfactual_attention_df.columns if col.startswith("head_")
]
n_heads = len(head_columns)

fig, axes = plt.subplots(1, n_heads, figsize=(5 * n_heads, 6))
if n_heads == 1:
    axes = [axes]

counterfactual_with_clusters = counterfactual_attention_df.copy()
counterfactual_with_clusters["cluster"] = counterfactual_with_clusters.index.map(
    dict(zip(pca_df.index, pca_df["cluster"]))
)

counterfactual_with_clusters = counterfactual_with_clusters.dropna(subset=["cluster"])

for i, head_col in enumerate(head_columns):
    head_idx = head_col.split("_")[1]

    plot_data = counterfactual_with_clusters.melt(
        id_vars=["cluster"],
        value_vars=[head_col],
        var_name="head",
        value_name="attention_score",
    )

    sns.boxplot(
        hue="cluster",
        y="attention_score",
        data=plot_data,
        ax=axes[i],
        palette=["#3498db", "#e74c3c"],
    )

    cluster0 = counterfactual_with_clusters[
        counterfactual_with_clusters["cluster"] == 0
    ][head_col]
    cluster1 = counterfactual_with_clusters[
        counterfactual_with_clusters["cluster"] == 1
    ][head_col]

    if len(cluster0) > 0 and len(cluster1) > 0:
        t_stat, p_val = ttest_ind(cluster0, cluster1, equal_var=False)
        axes[i].set_title(f"Head {head_idx}\np-value: {p_val:.4f}")
    else:
        axes[i].set_title(f"Head {head_idx}")

    axes[i].set_xlabel("Cluster")

    if i == 0:
        axes[i].set_ylabel("Attention Score")
    else:
        axes[i].set_ylabel("")

plt.suptitle(
    f"Comparison of Attention Scores Between Clusters\n{receiver_ct} receiving at distance {distance}",
    fontsize=16,
)
plt.tight_layout()
plt.savefig(
    figures_dir
    / f"counterfactual_attention_pca_kmeans_boxplot_{receiver_ct}_{distance}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
# %%
# Boxplot of head of note (for use in figure)
fig, ax = plt.subplots(figsize=(4, 8))
plot_data = counterfactual_with_clusters.melt(
    id_vars=["cluster"],
    value_vars=[head_col_of_note],
    var_name="head",
    value_name="attention_score",
)

sns.boxplot(
    hue="cluster",
    y="attention_score",
    data=plot_data,
    ax=ax,
    palette=[
        SUBTYPING_PALETTE["Low-interacting"],
        SUBTYPING_PALETTE["High-interacting"],
    ],
)
# Update legend labels
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ["Low-interacting", "High-interacting"], title="Cluster")

ax.set_title(f"Boxplot of {head_col_of_note} by Cluster")
ax.set_xlabel("Neighbor Category")
ax.set_ylabel(
    f"Counterfactual Attention Score for {receiver_ct} at distance={distance}"
)
ax.set_ylim(0, 1)
plt.savefig(
    figures_dir
    / f"counterfactual_attention_pca_kmeans_boxplot_{receiver_ct}_{distance}_head_{head_col_of_note}.svg",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
# %%
cluster_0_indices = counterfactual_with_clusters[
    counterfactual_with_clusters["cluster"] == 0
].index
cluster_1_indices = counterfactual_with_clusters[
    counterfactual_with_clusters["cluster"] == 1
].index
# %%
# Spatial plot showing cell types and proximity scores to high-attention cluster
plot_df = adata.obsm["spatial"].copy()
plot_df[labels_key] = adata.obs[labels_key]
plot_df["sample"] = adata.obs["sample"]
cluster_1_df = plot_df.iloc[cluster_1_indices.values, :]
cluster_0_df = plot_df.iloc[cluster_0_indices.values, :]
plot_df = plot_df[plot_df["sample"] == "0"]  # only sample 0 for viz
cluster_1_df = cluster_1_df[cluster_1_df["sample"] == "0"]
cluster_0_df = cluster_0_df[cluster_0_df["sample"] == "0"]

receiver_adata = adata[adata.obs[labels_key] == receiver_ct].copy()
receiver_neighbor_idxs = receiver_adata.obsm["_nn_idx"]
receiver_adata.obs["dist_to_cluster_1"] = np.where(
    np.isin(np.array(receiver_neighbor_idxs), cluster_1_indices),
    receiver_adata.obsm["_nn_dist"],
    np.inf,
).min(axis=1)
receiver_adata.obs[f"in_dist_cluster_1"] = (
    receiver_adata.obs["dist_to_cluster_1"] <= distance
).astype(str)
receiver_adata.obs[f"in_dist_cluster_1"] = receiver_adata.obs[
    f"in_dist_cluster_1"
].astype("category")
receiver_df = receiver_adata.obsm["spatial"].copy()
receiver_df["sample"] = receiver_adata.obs["sample"]
receiver_df[f"in_dist_cluster_1"] = receiver_adata.obs[f"in_dist_cluster_1"]
receiver_df = receiver_df[receiver_df["sample"] == "0"]


def distance_to_scale(dist):
    if dist == np.inf:
        return 0
    else:
        a = 8 - dist / 25
        return np.exp(a) / (np.exp(a) + np.exp(3))


# Apply the mapping to create a continuous scale
receiver_adata.obs["proximity_score"] = receiver_adata.obs["dist_to_cluster_1"].apply(
    distance_to_scale
)
receiver_df["proximity_score"] = receiver_adata.obs["proximity_score"]
sns.histplot(receiver_df["proximity_score"], bins=50)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    cluster_0_df,
    x="X",
    y="Y",
    alpha=1.0,
    s=5,
    marker="o",
    facecolor=SUBTYPING_PALETTE["Low-interacting"],
)
sns.scatterplot(
    cluster_1_df,
    x="X",
    y="Y",
    alpha=1.0,
    s=6,
    marker="o",
    facecolor=SUBTYPING_PALETTE["High-interacting"],
)
sns.scatterplot(
    receiver_df,
    x="X",
    y="Y",
    hue="proximity_score",
    palette=LinearSegmentedColormap.from_list(
        "custom_palette",
        [
            SUBTYPING_PALETTE["Far-interacting-cell"],
            SUBTYPING_PALETTE["Near-interacting-cell"],
        ],
        N=100,
    ),
    alpha=1.0,
    s=5,
    marker="o",
)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.5)
plt.title(f"Spatial Distribution Interacting Neighbors of {receiver_ct} in Tissue")
plt.xlabel("X ($\mu m$)")
plt.ylabel("Y ($\mu m$)")
plt.legend(
    [
        f"Low-interacting {sender_ct} (Bottom 90% Attention)",
        f"High-interacting {sender_ct} (Top 10% Attention)",
    ],
    title="Neighbor Category",
    loc="upper left",
    bbox_to_anchor=(1.25, 1),
    borderaxespad=0.5,
    markerscale=3.0,
)
# Create colorbar for proximity score
sm = plt.cm.ScalarMappable(
    cmap=LinearSegmentedColormap.from_list(
        "custom_palette",
        [
            SUBTYPING_PALETTE["Far-interacting-cell"],
            SUBTYPING_PALETTE["Near-interacting-cell"],
        ],
        N=100,
    )
)
sm.set_array([])
cbar = plt.colorbar(
    sm, ax=ax, label="Proximity Score", orientation="vertical", pad=0.01
)
cbar.set_label(f"Proximity Score\n(Distance to {sender_ct})")
distances = np.linspace(0, 200, 5)
proximity_scores = [distance_to_scale(dist) for dist in distances]
cbar.set_ticks(proximity_scores)
cbar.set_ticklabels([f"{dist:.0f} μm" for dist in distances])

plt.savefig(
    figures_dir / f"sender_receiver_spatial_plot_{receiver_ct}_{distance}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    figures_dir / f"sender_receiver_spatial_plot_{receiver_ct}_{distance}.svg",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
# %%
# Ablation of cluster 1 (Sender-Receiver gene analysis)
ablation_module = model.get_neighbor_ablation_scores(
    cell_type=receiver_ct,
    head_idx=head_idx_of_note,
    ablated_neighbor_indices=cluster_1_indices,
    compute_z_value=True,
)


# %%
# Hacked dotplot to show ablated_indices scores
def plot_featurewise_contributions_dotplot(
    self,
    cell_type=None,
    color_by="diff",
    size_by="nl10_pval_adj",
    n_top_genes=20,
    step=0.2,
    dot_max=1.0,
    dot_min=0.0,
    show=True,
    save_png=False,
    save_svg=False,
):
    assert color_by in [
        "z_value",
        "ablation",
        "diff",
        "pval_adj",
        "nl10_pval_adj",
    ], "color_by must be one of 'z_value', 'ablation', 'diff', 'pval_adj', or 'nl10_pval_adj'"
    assert size_by in [
        "z_value",
        "ablation",
        "diff",
        "pval_adj",
        "nl10_pval_adj",
    ], "size_by must be one of 'z_value', 'ablation', 'diff', 'pval_adj', or 'nl10_pval_adj'"
    assert (
        len(self._ablation_scores_df["cell_type"].unique()) == 1
    ) or cell_type is not None, (
        "More than one cell type found. Please pass in cell_type."
    )
    assert (
        len(self._ablation_scores_df["head_idx"].unique()) == 1
    ), "More than one head index found"
    cell_type = (
        self._ablation_scores_df["cell_type"].unique()[0]
        if cell_type is None
        else cell_type
    )
    head_idx = self._ablation_scores_df["head_idx"].unique()[0]

    top_genes_per_ct = {}

    ct_name = "ablated_indices"
    size_by_cols = [f"{ct_name}_{size_by}"]
    color_by_cols = [f"{ct_name}_{color_by}"]
    ct_scores_df = self._ablation_scores_df[
        self._ablation_scores_df["cell_type"] == cell_type
    ]
    for colname in color_by_cols:
        ct_name = colname.replace(f"_{color_by}", "")
        # Filter for positive neighbor contribution scores
        diff_filter = ct_scores_df[f"{ct_name}_{size_by}"] > 0
        top_genes_idx = (
            ct_scores_df[diff_filter][colname].astype(float).nlargest(n_top_genes).index
        )
        top_genes_per_ct[ct_name] = top_genes_idx

    all_top_genes = []
    for genes in top_genes_per_ct.values():
        for gene in genes:
            if gene not in all_top_genes:
                all_top_genes.append(gene)

    # Create a dotplot dataframe
    dotplot_data = ct_scores_df.loc[all_top_genes, color_by_cols + size_by_cols]
    gene_names = ct_scores_df.loc[all_top_genes, "gene"]
    dotplot_data = dotplot_data.set_index(gene_names)
    dot_size_df = dotplot_data[size_by_cols]
    # scale dot sizes to 0-1 based on dot_max, dot_min
    dot_size_df[size_by_cols[0]] = np.clip(
        dot_size_df[size_by_cols[0]] - dot_min, a_min=0, a_max=dot_max
    ) / (dot_max - dot_min)

    for col in dot_size_df.columns:
        dot_size_df.rename(columns={col: col.replace(f"_{size_by}", "")}, inplace=True)
    dot_color_df = dotplot_data[color_by_cols]
    for col in dot_color_df.columns:
        dot_color_df.rename(
            columns={col: col.replace(f"_{color_by}", "")}, inplace=True
        )

    head_idx_str = f"head {head_idx}" if head_idx is not None else "all heads"
    score_titles = {
        "z_value": "Z-value",
        "ablation": "Ablation",
        "diff": "Neighbor Contribution",
        "pval_adj": "Adjusted P-value",
        "nl10_pval_adj": "Neg Log10 Adjusted P-value",
    }
    mock_adata = self._adata.copy()
    mock_adata.obs[self._labels_key] = "ablated_indices"
    fig = sc.pl.dotplot(
        mock_adata,
        dot_size_df=dot_size_df.T.astype(float),
        dot_color_df=dot_color_df.T.astype(float),
        var_names=gene_names,
        groupby=self._labels_key,
        vmin=-dot_color_df.max().max(),
        vmax=dot_color_df.max().max(),
        dot_min=0.0,
        dot_max=1.0,
        vcenter=0,
        cmap="RdBu_r",
        title=f"Dotplot of {score_titles[size_by]} by {score_titles[color_by]} Scores for {cell_type} for {head_idx_str}",
        size_title=score_titles[size_by],
        colorbar_title=score_titles[color_by],
        return_fig=True,
    )

    def _plot_size_legend(self, size_legend_ax, step=20, dot_max=1.0, dot_min=0.0):
        size_legend_ax.clear()
        # a descending range that is afterwards inverted is used
        # to guarantee that dot_max is in the legend.
        size_range = np.arange(dot_max, dot_min, step * -1)[::-1]
        if dot_min != 0 or dot_max != 1:
            dot_range = dot_max - dot_min
            size_values = (size_range - self.dot_min) / dot_range
        else:
            size_values = size_range

        size = size_values**self.size_exponent
        size = size * (self.largest_dot - self.smallest_dot) + self.smallest_dot

        # plot size bar
        size_legend_ax.scatter(
            np.arange(len(size)) + 0.5,
            np.repeat(0, len(size)),
            s=size,
            color="gray",
            edgecolor="black",
            linewidth=self.dot_edge_lw,
            zorder=100,
        )
        size_legend_ax.set_xticks(np.arange(len(size)) + 0.5)
        labels = [f"{np.round((x), decimals=0).astype(int)}" for x in size_range]
        size_legend_ax.set_xticklabels(labels, fontsize="small")

        # remove y ticks and labels
        size_legend_ax.tick_params(
            axis="y", left=False, labelleft=False, labelright=False
        )

        # remove surrounding lines
        size_legend_ax.spines["right"].set_visible(False)
        size_legend_ax.spines["top"].set_visible(False)
        size_legend_ax.spines["left"].set_visible(False)
        size_legend_ax.spines["bottom"].set_visible(False)
        size_legend_ax.grid(visible=False)

        ymax = size_legend_ax.get_ylim()[1]
        size_legend_ax.set_ylim(-1.05 - self.largest_dot * 0.003, 4)
        size_legend_ax.set_title(self.size_title, y=ymax + 0.45, size="small")

        xmin, xmax = size_legend_ax.get_xlim()
        size_legend_ax.set_xlim(xmin - 0.15, xmax + 0.5)

    _plot_size_legend(
        fig,
        fig.get_axes()["size_legend_ax"],
        step=step,
        dot_max=dot_max,
        dot_min=dot_min,
    )
    if save_png:
        plt.savefig(
            figures_dir
            / f"dotplot_{cell_type}_{head_idx_str}_{size_by}_{color_by}.png",
            dpi=300,
            bbox_inches="tight",
        )
    if save_svg:
        plt.savefig(
            figures_dir
            / f"dotplot_{cell_type}_{head_idx_str}_{size_by}_{color_by}.svg",
            dpi=300,
            bbox_inches="tight",
        )
    if show:
        plt.show()


fig = plot_featurewise_contributions_dotplot(
    ablation_module,
    cell_type=receiver_ct,
    color_by="diff",
    size_by="nl10_pval_adj",
    n_top_genes=20,
    step=40,
    dot_max=200,
    dot_min=0,
    save_svg=True,
    save_png=True,
    show=False,
)
# %%
# Plot top sender genes representative of cluster 1
sender_adata = adata[adata.obs[labels_key] == sender_ct].copy()
sender_adata.obs["cluster"] = "-1"
sender_adata.obs.loc[adata.obs_names[cluster_0_indices], "cluster"] = "0"
sender_adata.obs.loc[adata.obs_names[cluster_1_indices], "cluster"] = "1"
sc.tl.rank_genes_groups(sender_adata, groupby="cluster", method="t-test")
fig, ax = plt.subplots(figsize=(10, 3))
sc.pl.rank_genes_groups_dotplot(
    sender_adata,
    n_genes=5,
    groups=["1"],
    values_to_plot="logfoldchanges",
    cmap="bwr",
    vmin=-3,
    vmax=3,
    min_logfoldchange=0.5,
    ax=ax,
    show=False,
)
ax.set_title(
    f"Top 20 Effect LFC Genes for Cluster 1 ({sender_ct}) clustered based on attention to {receiver_ct} at distance {distance}"
)
ax.set_xlabel("Cluster")
ax.set_ylabel("Effect LFC")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(
    figures_dir
    / f"sender_de_genes_between_cluster_1_and_cluster_0_{receiver_ct}_{distance}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    figures_dir
    / f"sender_de_genes_between_cluster_1_and_cluster_0_{receiver_ct}_{distance}.svg",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
###############################################
# Receiver overall attention pattern analysis #
###############################################
receiver_attention_patterns = model.get_attention_patterns(
    indices=adata.obs[labels_key] == receiver_ct,
)
# %%
# Sum across all neighbor columns for each cell and head
neighbor_cols = [
    col
    for col in receiver_attention_patterns._attention_patterns_df.columns
    if col.startswith("neighbor_")
]
head_indices = receiver_attention_patterns._attention_patterns_df["head"].unique()

summed_attention_df = pd.DataFrame()
summed_attention_df["cell_idx"] = receiver_attention_patterns._attention_patterns_df[
    "cell_idx"
].unique()

for head_idx in head_indices:
    head_df = receiver_attention_patterns._attention_patterns_df[
        receiver_attention_patterns._attention_patterns_df["head"] == head_idx
    ]

    head_sums = (
        head_df.groupby("cell_idx")[neighbor_cols].sum().sum(axis=1).reset_index()
    )
    head_sums.columns = ["cell_idx", f"head_{head_idx}"]

    summed_attention_df = summed_attention_df.merge(
        head_sums, on="cell_idx", how="left"
    )

summed_attention_df.head()


# %%
# Sort heads by variance across all neighbors
head_variances = summed_attention_df.iloc[:, 1:].var(axis=0)
print(head_variances)

# %%
# Spatial plot of receiver cell types colored by head of note (all neighbors)
receiver_df[f"head_{head_idx_of_note}"] = summed_attention_df[
    ["cell_idx", f"head_{head_idx_of_note}"]
].set_index("cell_idx")
sns.scatterplot(
    receiver_df,
    x="X",
    y="Y",
    hue=f"head_{head_idx_of_note}",
    palette="viridis",
    alpha=1.0,
    s=5,
    marker="o",
)
plt.title(
    f"Total Head {head_idx_of_note} Attention from {receiver_ct} to all neighbors"
)
plt.savefig(
    figures_dir
    / f"receiver_attention_patterns_{receiver_ct}_head_{head_idx_of_note}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    figures_dir
    / f"receiver_attention_patterns_{receiver_ct}_head_{head_idx_of_note}.svg",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
# normal DE between high attention receiver and low attention receiver (receiver genes)
head_median_val = receiver_df[f"head_{head_idx_of_note}"].quantile(0.5)
receiver_adata.obs[f"high_head_{head_idx_of_note}"] = (
    receiver_df[f"head_{head_idx_of_note}"] > head_median_val
).astype(str)
sc.tl.rank_genes_groups(
    receiver_adata, groupby=f"high_head_{head_idx_of_note}", method="t-test"
)
fig, ax = plt.subplots(figsize=(10, 3))
sc.pl.rank_genes_groups_dotplot(
    receiver_adata,
    n_genes=20,
    groups=["True"],
    values_to_plot="logfoldchanges",
    cmap="bwr",
    vmin=-3,
    vmax=3,
    min_logfoldchange=0.5,
    ax=ax,
    show=False,
)
plt.savefig(
    figures_dir
    / f"receiver_de_genes_between_high_and_low_attention_{receiver_ct}_head_{head_idx_of_note}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
# Sum across all neighbor columns for each cell and head and sender cell
neighbor_cols = [
    col
    for col in receiver_attention_patterns._attention_patterns_df.columns
    if col.startswith("neighbor_")
]
head_indices = receiver_attention_patterns._attention_patterns_df["head"].unique()
neighbor_idxs = receiver_attention_patterns._nn_idxs_df.values
neighbor_cell_types = pd.DataFrame(
    data=rearrange(
        adata[rearrange(neighbor_idxs, "b n -> (b n)")]
        .obs[labels_key]
        .values.astype(str),
        "(b n) -> b n",
        b=receiver_attention_patterns._nn_idxs_df.shape[0],
    ),
    index=receiver_attention_patterns._nn_idxs_df.index,
    columns=receiver_attention_patterns._nn_idxs_df.columns,
)

# %%
# Sum attentions across cell type labels as well as for cluster 1
summed_attention_df = pd.DataFrame()
summed_attention_df["cell_idx"] = receiver_attention_patterns._attention_patterns_df[
    "cell_idx"
].unique()

head_df = receiver_attention_patterns._attention_patterns_df[
    receiver_attention_patterns._attention_patterns_df["head"] == head_idx_of_note
]
for label in np.unique(neighbor_cell_types.values.flatten()):
    head_ct_attention_df = pd.DataFrame(
        np.where(
            neighbor_cell_types.loc[head_df["cell_idx"]].to_numpy() == label,
            head_df[neighbor_cols].to_numpy(),
            0.0,
        ).sum(axis=1),
        columns=[f"head_{head_idx_of_note}_{label}"],
        index=head_df["cell_idx"],
    )
    head_ct_attention_df.reset_index(inplace=True)
    head_ct_attention_df.rename(columns={"index": "cell_idx"}, inplace=True)

    summed_attention_df = summed_attention_df.merge(
        head_ct_attention_df, on="cell_idx", how="left"
    )

# total attention (not to dummy)
summed_attention_df[f"head_{head_idx_of_note}"] = summed_attention_df.iloc[:, 1:].sum(
    axis=1
)
# add column for cluster 1
cluster_1_head_attention_df = pd.DataFrame(
    np.where(
        np.isin(neighbor_idxs, adata.obs_names[cluster_1_indices]),
        head_df[neighbor_cols].to_numpy(),
        0.0,
    ).sum(axis=1),
    columns=[f"head_{head_idx_of_note}_cluster_1"],
    index=head_df["cell_idx"],
)
cluster_1_head_attention_df.reset_index(inplace=True)
cluster_1_head_attention_df.rename(columns={"index": "cell_idx"}, inplace=True)
summed_attention_df = summed_attention_df.merge(
    cluster_1_head_attention_df, on="cell_idx", how="left"
)


summed_attention_df.head()

# %%
# Scatter plot of attention to different cell types for head of note
for label in np.unique(neighbor_cell_types.values.flatten()):
    sns.scatterplot(
        summed_attention_df,
        x=f"head_{head_idx_of_note}",
        y=f"head_{head_idx_of_note}_{label}",
        label=label,
        s=5,
        alpha=0.2,
    )
plt.ylabel("cell-type attention contribution")
plt.xlabel("total head attention")
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    markerscale=3.0,
)
plt.savefig(
    figures_dir
    / f"receiver_attention_patterns_{receiver_ct}_head_{head_idx_of_note}_per_cell_type_attention.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    figures_dir
    / f"receiver_attention_patterns_{receiver_ct}_head_{head_idx_of_note}_per_cell_type_attention.svg",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
# Show top neighbor cell types when head of note attention is high
summed_attention_df[(summed_attention_df[f"head_{head_idx_of_note}"] > 0.4)].iloc[
    :, 1:
].mean(axis=0)

# %%
# Spatial plot colored by attention to cluster 1
receiver_df[f"head_{head_idx_of_note}_cluster_1"] = summed_attention_df.set_index(
    "cell_idx"
)[f"head_{head_idx_of_note}_cluster_1"]
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    cluster_0_df,
    x="X",
    y="Y",
    alpha=1.0,
    s=3,
    marker="o",
    facecolor=SUBTYPING_PALETTE["Low-interacting"],
)
sns.scatterplot(
    cluster_1_df,
    x="X",
    y="Y",
    alpha=1.0,
    s=3,
    marker="o",
    facecolor=SUBTYPING_PALETTE["High-interacting"],
)
sns.scatterplot(
    receiver_df,
    x="X",
    y="Y",
    hue=f"head_{head_idx_of_note}_cluster_1",
    palette=LinearSegmentedColormap.from_list(
        "custom_palette",
        [
            "#FF4444",  # Red for low proximity (far distance)  
            "#4444FF",  # Blue for high proximity (close distance)
        ],
        N=100,
    ),
    alpha=1.0,
    s=3,
    marker="o",
)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.5)
plt.title(f"Spatial Distribution Interacting Neighbors of {receiver_ct} in Tissue")
plt.xlabel("X ($\mu m$)")
plt.ylabel("Y ($\mu m$)")
plt.legend(
    [
        f"Low-interacting {sender_ct} (Bottom 90% Attention)",
        f"High-interacting {sender_ct} (Top 10% Attention)",
    ],
    title="Neighbor Category",
    loc="upper left",
    bbox_to_anchor=(1.25, 1),
    borderaxespad=0.5,
    markerscale=3.0,
)
# Create colorbar for proximity score
sm = plt.cm.ScalarMappable(
    cmap=LinearSegmentedColormap.from_list(
        "custom_palette",
        [
            "#FF4444",  # Red for low proximity (far distance)  
            "#4444FF",  # Blue for high proximity (close distance)
        ],
        N=100,
    )
)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.01)
cbar.set_label(f"Attention to High-interacting {sender_ct}")

plt.savefig(
    figures_dir
    / f"sender_receiver_spatial_plot_{receiver_ct}_head_{head_idx_of_note}_with_attention_to_cluster_1.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    figures_dir
    / f"sender_receiver_spatial_plot_{receiver_ct}_head_{head_idx_of_note}_with_attention_to_cluster_1.svg",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
