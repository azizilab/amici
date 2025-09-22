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
from einops import rearrange
from scipy.stats import spearmanr

from scipy.stats import ttest_ind
from amici import AMICI

from libpysal.weights import KNN
from esda.moran import Moran_Local_BV
SPATIAL_STATS_AVAILABLE = True

from matplotlib.colors import Normalize

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
adata_full = sc.read_h5ad(f"./data/xenium_sample1/xenium_sample1_filtered_{data_date}.h5ad")
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
adata_full.obs["x"] = adata_full.obsm["spatial"]["X"]
adata_full.obs["y"] = adata_full.obsm["spatial"]["Y"]

# %% Load model
model = AMICI.load(
    model_path,
    adata=adata_full,
)
AMICI.setup_anndata(
    adata_full,
    labels_key=labels_key,
    coord_obsm_key="spatial",
    n_neighbors=50,
)
# %%
# Set Analysis Parameters
receiver_ct = "Invasive_Tumor"
sender_ct = "CD8+_T_Cells"
sample_id = '0'

# %% 
figures_dir = Path(f"figures/")
adata = adata_full[adata_full.obs["sample"] == sample_id].copy()

# %%
# Compute Empirical Attention Patterns for clustering
receiver_attention_patterns = model.get_attention_patterns(
    indices=(adata_full.obs[labels_key] == receiver_ct) & (adata_full.obs["sample"] == sample_id),
)

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

# %% Sum attentions across cell type labels for all heads
summed_attention_df = pd.DataFrame()
summed_attention_df["cell_idx"] = receiver_attention_patterns._attention_patterns_df[
    "cell_idx"
].unique()

heads_df = receiver_attention_patterns._attention_patterns_df

head_ct_attention_df = pd.DataFrame(
    np.where(
        neighbor_cell_types.loc[heads_df["cell_idx"]].to_numpy() == sender_ct,
        heads_df[neighbor_cols].to_numpy(),
        0.0,
    ).sum(axis=1),
    columns=[f"all_heads_{sender_ct}"],
)
# Preserve both cell_idx and head information
head_ct_attention_df["cell_idx"] = heads_df["cell_idx"].values
head_ct_attention_df["head"] = heads_df["head"].values

summed_attention_df = summed_attention_df.merge(
    head_ct_attention_df, on="cell_idx", how="left"
)

# Get max attention score per cell (since some cells appear multiple times in different heads)
empirical_attention_df = summed_attention_df.groupby("cell_idx").max()[f"all_heads_{sender_ct}"]

print(f"Empirical attention scores summary:")
print(empirical_attention_df.describe())

receiver_adata = adata[adata.obs[labels_key] == receiver_ct].copy()
receiver_neighbor_idxs = receiver_adata.obsm["_nn_idx"]
receiver_df = receiver_adata.obsm["spatial"].copy()
receiver_df["sample"] = receiver_adata.obs["sample"]
receiver_df = receiver_df[receiver_df["sample"] == sample_id]

# Map empirical attention scores to receiver cells
receiver_df[f"empirical_attention_{sender_ct}"] = receiver_df.index.map(empirical_attention_df).fillna(0)

# %% Plot the max attention received by receivers from any sender type
plt.figure(figsize=(15, 8))
scatter = sns.scatterplot(
    receiver_df,
    x="X",
    y="Y",
    hue=f"empirical_attention_{sender_ct}",
    palette=LinearSegmentedColormap.from_list(
        "custom_palette",
        [
            "#4444FF",  # Blue for high attention
            "#FF4444",  # Red for low attention
        ],
        N=100,
    ),
    alpha=1.0,
    s=8,
    marker="o",
)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.5)
plt.title(f"Spatial Distribution of {receiver_ct}\nEmpirical Attention to {sender_ct}")
plt.xlabel("X (μm)")
plt.ylabel("Y (μm)")

# Create colorbar for attention score
sm = plt.cm.ScalarMappable(
    cmap=LinearSegmentedColormap.from_list(
        "custom_palette",
        [
            "#4444FF",  # Blue for high attention
            "#FF4444",  # Red for low attention
        ],
        N=100,
    )
)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), label="Empirical Attention Score")

plt.savefig(
    figures_dir / f"empirical_attention_spatial_plot_{receiver_ct}_{sender_ct}.png",
    dpi=300,
    bbox_inches="tight",
)
scatter.collections[0].set_visible(False)
plt.savefig(
    figures_dir / f"empirical_attention_spatial_plot_{receiver_ct}_{sender_ct}.svg",
    dpi=300,
    bbox_inches="tight",
)
scatter.collections[0].set_visible(True)
plt.show()

# %% Spatial plot colored by empirical attention by attention paid to a sender from any receiver type
sender_adata = adata[adata.obs[labels_key] == sender_ct].copy() 
sender_df = sender_adata.obsm["spatial"].copy()
sender_df["sample"] = sender_adata.obs["sample"]
sender_df = sender_df[sender_df["sample"] == sample_id]

# For each sender cell, find the maximum attention it receives from any receiver
heads_df = receiver_attention_patterns._attention_patterns_df
neighbor_idxs_df = receiver_attention_patterns._nn_idxs_df
neighbor_cols = [col for col in heads_df.columns if col.startswith("neighbor_")]

# Convert neighbor indices to long format with receiver and position info
neighbor_long = neighbor_idxs_df.reset_index().melt(
    id_vars=['index'], 
    var_name='neighbor_position', 
    value_name='neighbor_cell_idx'
)
neighbor_long['neighbor_pos'] = neighbor_long['neighbor_position'].str.extract('(\d+)').astype(int)

# Convert attention scores to long format
attention_long = heads_df.melt(
    id_vars=['cell_idx', 'head'],
    value_vars=neighbor_cols,
    var_name='neighbor_position',
    value_name='attention_score'
)
attention_long['neighbor_pos'] = attention_long['neighbor_position'].str.extract('(\d+)').astype(int)

# Merge attention scores with neighbor cell indices
attention_with_neighbors = pd.merge(
    attention_long,
    neighbor_long[['index', 'neighbor_pos', 'neighbor_cell_idx']],
    left_on=['cell_idx', 'neighbor_pos'],
    right_on=['index', 'neighbor_pos'],
    how='left'
)

# Filter for sender cell types only
sender_cell_indices = adata[adata.obs[labels_key] == sender_ct].obs_names
attention_to_senders = attention_with_neighbors[
    attention_with_neighbors['neighbor_cell_idx'].isin(sender_cell_indices)
].copy()

# Group by sender cell and find maximum attention received
empirical_sender_attention_df = attention_to_senders.groupby('neighbor_cell_idx')['attention_score'].max()

# Ensure all sender cells are represented (fill missing with 0)
all_sender_indices = pd.Index(sender_cell_indices)
empirical_sender_attention_df = empirical_sender_attention_df.reindex(all_sender_indices, fill_value=0.0)

print(f"Sender attention received summary:")
print(empirical_sender_attention_df.describe())

# Map attention scores to sender spatial dataframe
sender_df[f"max_attention_received"] = sender_df.index.map(empirical_sender_attention_df).fillna(0)

# %% Plot the spatial distribution of the max attention received by senders from any receiver type
plt.figure(figsize=(15, 8))
sns.scatterplot(
    sender_df,
    x="X",
    y="Y",
    hue=f"max_attention_received",
    palette=LinearSegmentedColormap.from_list(
        "custom_palette",
        [
            "#4444FF",  # Blue for high attention
            "#FF4444",  # Red for low attention
        ],
        N=100,
    ),
    alpha=1.0,
    s=8,
    marker="o",
)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.5)
plt.title(f"Spatial Distribution of {receiver_ct}\nEmpirical Attention to {sender_ct}")
plt.xlabel("X (μm)")
plt.ylabel("Y (μm)")

# Create colorbar for attention score
sm = plt.cm.ScalarMappable(
    cmap=LinearSegmentedColormap.from_list(
        "custom_palette",
        [
            "#4444FF",  # Blue for high attention
            "#FF4444",  # Red for low attention
        ],
        N=100,
    )
)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), label="Empirical Attention Score")

plt.savefig(
    figures_dir / f"sender_empirical_attention_spatial_plot_{receiver_ct}_{sender_ct}.png",
    dpi=300,
    bbox_inches="tight",
)
scatter.collections[0].set_visible(False)
plt.savefig(
    figures_dir / f"sender_empirical_attention_spatial_plot_{receiver_ct}_{sender_ct}.svg",
    dpi=300,
    bbox_inches="tight",
)
scatter.collections[0].set_visible(True)
plt.show()

# %%
# Compare distributions: attention paid vs attention received
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Attention paid by receivers to senders
empirical_attention_df.hist(bins=50, alpha=0.7, color='lightcoral', ax=ax1)
ax1.set_title(f"Attention Paid by {receiver_ct}\nto {sender_ct}")
ax1.set_xlabel("Attention Score")
ax1.set_ylabel("Frequency")
ax1.grid(alpha=0.3)

# Max attention received by senders from receivers
empirical_sender_attention_df.hist(bins=50, alpha=0.7, color='skyblue', ax=ax2)
ax2.set_title(f"Max Attention Received by {sender_ct}\nfrom {receiver_ct}")
ax2.set_xlabel("Max Attention Score")
ax2.set_ylabel("Frequency")
ax2.grid(alpha=0.3)

plt.suptitle("Comparison: Attention Paid vs Attention Received", fontsize=16)
plt.tight_layout()
plt.savefig(
    figures_dir / f"attention_comparison_{receiver_ct}_{sender_ct}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %% Cluster the sender attention scores based on a quantile threshold
# Get the 95% quantile of the senders that have positive attention
quantile = 0.95
sender_attention_threshold = empirical_sender_attention_df[empirical_sender_attention_df > 0].astype(float).quantile(quantile)

print(f"Sender attention threshold: {sender_attention_threshold}")

high_empirical_sender_attention_cells = empirical_sender_attention_df[
    empirical_sender_attention_df > sender_attention_threshold
].index

sender_cell_indices = adata[adata.obs[labels_key] == sender_ct].obs_names
empirical_clusters = pd.DataFrame(0, columns=["empirical_cluster"], index=sender_cell_indices)
empirical_clusters.loc[high_empirical_sender_attention_cells, "empirical_cluster"] = 1

print(f"Number of high attention cells: {empirical_clusters['empirical_cluster'].sum()}")
print(f"Number of low attention cells: {len(empirical_clusters) - empirical_clusters['empirical_cluster'].sum()}")

# %% Boxplot of empirical attention scores by empirical clusters
empirical_clusters_attention_df = pd.DataFrame(empirical_sender_attention_df).merge(empirical_clusters, left_index=True, right_index=True)

fig, ax = plt.subplots(figsize=(8, 6))

# Create dataframe for empirical attention plotting
empirical_sender_attention_with_clusters = pd.DataFrame({
    'cell_idx': empirical_clusters_attention_df.index,
    'empirical_attention': empirical_clusters_attention_df["attention_score"],
    'empirical_cluster': empirical_clusters_attention_df["empirical_cluster"]
})

plot_data = empirical_sender_attention_with_clusters.melt(
    id_vars=["empirical_cluster"],
    value_vars=["empirical_attention"],
    var_name="attention_type",
    value_name="attention_score",
)

sns.violinplot(
    hue="empirical_cluster",
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

# Statistical test
cluster0_emp = empirical_sender_attention_with_clusters[
    empirical_sender_attention_with_clusters["empirical_cluster"] == 0
]["empirical_attention"]
cluster1_emp = empirical_sender_attention_with_clusters[
    empirical_sender_attention_with_clusters["empirical_cluster"] == 1
]["empirical_attention"]

if len(cluster0_emp) > 0 and len(cluster1_emp) > 0:
    t_stat, p_val = ttest_ind(a=np.array(cluster0_emp, dtype=float), b=np.array(cluster1_emp, dtype=float), equal_var=False)
    ax.set_title(f"Empirical Attention by Clusters\np-value: {p_val:.2e}")
else:
    ax.set_title("Empirical Attention by Clusters")

ax.set_xlabel("Empirical Attention Cluster")
ax.set_ylabel(f"Empirical Attention Score: {receiver_ct} to {sender_ct}")
plt.savefig(
    figures_dir / f"empirical_attention_by_clusters_{receiver_ct}_{sender_ct}.svg",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
# Get cluster indices using empirical clustering
cluster_0_indices = empirical_sender_attention_with_clusters[empirical_sender_attention_with_clusters["empirical_cluster"] == 0].index
cluster_1_indices = empirical_sender_attention_with_clusters[empirical_sender_attention_with_clusters["empirical_cluster"] == 1].index

print(f"Cluster 0 (Low-interacting): {len(cluster_0_indices)} cells")
print(f"Cluster 1 (High-interacting): {len(cluster_1_indices)} cells")

# %% Spatial plot showing cell types and proximity scores to high-attention cluster (empirical)
plot_df = adata.obsm["spatial"].copy()
plot_df[labels_key] = adata.obs[labels_key]
plot_df["sample"] = adata.obs["sample"]
cluster_1_df = plot_df.loc[cluster_1_indices, :]
cluster_0_df = plot_df.loc[cluster_0_indices, :]

receiver_adata = adata[adata.obs[labels_key] == receiver_ct].copy()
receiver_neighbor_idxs = receiver_adata.obsm["_nn_idx"]
sender_cell_indices = adata[adata.obs[labels_key] == sender_ct].obs_names
receiver_adata.obs["dist_to_cluster_1"] = np.where(
    np.isin(np.array(receiver_neighbor_idxs), np.where(np.isin(adata_full.obs_names, cluster_1_indices))[0]),
    receiver_adata.obsm["_nn_dist"],
    np.inf,
).min(axis=1)
proximity_receiver_df = receiver_adata.obsm["spatial"].copy()
proximity_receiver_df["sample"] = receiver_adata.obs["sample"]
proximity_receiver_df = proximity_receiver_df[proximity_receiver_df["sample"] == sample_id]


def distance_to_scale(dist):
    if dist == np.inf:
        return 0
    else:
        return 1 / (1 + np.exp(dist/25 - 5))


# Apply the mapping to create a continuous scale
receiver_adata.obs["proximity_score"] = receiver_adata.obs["dist_to_cluster_1"].apply(
    distance_to_scale
)
proximity_receiver_df["proximity_score"] = receiver_adata.obs["proximity_score"]
sns.histplot(proximity_receiver_df["proximity_score"], bins=50)
plt.title(f"Distribution of Proximity Scores to High-Interacting {sender_ct}")
plt.xlabel("Proximity Score")
plt.ylabel("Frequency")
plt.show()

# %% Plot the spatial proximity scores
fig, ax = plt.subplots(figsize=(15, 8))
scatter = sns.scatterplot(
    proximity_receiver_df,
    x="X",
    y="Y",
    hue="proximity_score",
    palette=LinearSegmentedColormap.from_list(
        "custom_palette",
        [
            "#4444FF",  # Blue for high proximity (close distance)
            "#FF4444",  # Red for low proximity (far distance)  
        ],
        N=500,
    ),
    alpha=1.0,
    s=5,
    marker="o",
)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.5)

# Get the actual data range
vmin = proximity_receiver_df["proximity_score"].min()
vmax = proximity_receiver_df["proximity_score"].max()

sm = plt.cm.ScalarMappable(
    cmap=LinearSegmentedColormap.from_list(
        "custom_palette",
        [
            "#4444FF",  # Blue for high proximity (close distance)
            "#FF4444",  # Red for low proximity (far distance)  
        ],
        N=500,
    ),
    norm=Normalize(vmin=vmin, vmax=vmax)  # Use the actual data range
)
sm.set_array([])
cbar = plt.colorbar(
    sm, ax=ax, label="Proximity Score", orientation="vertical", pad=0.01, 
)
cbar.set_label(f"Proximity Score\n(Distance to High-Interacting {sender_ct})")
distances = np.linspace(0, 200, 5)
proximity_scores = [distance_to_scale(dist) for dist in distances]
cbar.set_ticks(proximity_scores)
cbar.set_ticklabels([f"{dist:.0f} μm" for dist in distances])
scatter.get_legend().remove()
plt.title(f"Spatial Distribution of {receiver_ct}\nProximity to High-Interacting {sender_ct}")
plt.xlabel("X (μm)")
plt.ylabel("Y (μm)")
plt.savefig(
    figures_dir / f"proximity_spatial_plot_{receiver_ct}_high_senders.png",
    dpi=300,
    bbox_inches="tight",
)
scatter.collections[0].set_visible(False)
plt.savefig(
    figures_dir / f"proximity_spatial_plot_{receiver_ct}_high_senders.svg",
    dpi=300,
    bbox_inches="tight",
)
scatter.collections[0].set_visible(True)
plt.show()

# %% Proximity scores to any sender of the corresponding cell type
plot_df = adata.obsm["spatial"].copy()
plot_df[labels_key] = adata.obs[labels_key]
plot_df["sample"] = adata.obs["sample"]
cluster_1_df = plot_df.loc[cluster_1_indices, :]
cluster_0_df = plot_df.loc[cluster_0_indices, :]

receiver_adata = adata[adata.obs[labels_key] == receiver_ct].copy()
receiver_neighbor_idxs = receiver_adata.obsm["_nn_idx"]
sender_cell_indices = adata[adata.obs[labels_key] == sender_ct].obs_names
receiver_adata.obs["dist_to_sender_type"] = np.where(
    np.isin(np.array(receiver_neighbor_idxs), np.where(np.isin(adata_full.obs_names, sender_cell_indices))[0]),
    receiver_adata.obsm["_nn_dist"],
    np.inf,
).min(axis=1)
proximity_receiver_df = receiver_adata.obsm["spatial"].copy()
proximity_receiver_df["sample"] = receiver_adata.obs["sample"]
proximity_receiver_df = proximity_receiver_df[proximity_receiver_df["sample"] == sample_id]

# Apply the mapping to create a continuous scale
receiver_adata.obs["proximity_score"] = receiver_adata.obs["dist_to_sender_type"].apply(
    distance_to_scale
)
proximity_receiver_df["proximity_score"] = receiver_adata.obs["proximity_score"]
sns.histplot(proximity_receiver_df["proximity_score"], bins=50)
plt.title(f"Distribution of Proximity Scores to Any {sender_ct} Type")
plt.xlabel("Proximity Score")
plt.ylabel("Frequency")
plt.show()

# %% Plot the spatial proximity scores
fig, ax = plt.subplots(figsize=(15, 8))
scatter = sns.scatterplot(
    proximity_receiver_df,
    x="X",
    y="Y",
    hue="proximity_score",
    palette=LinearSegmentedColormap.from_list(
        "custom_palette",
        [
            "#4444FF",  # Blue for high proximity (close distance)
            "#FF4444",  # Red for low proximity (far distance)  
        ],
        N=500,
    ),
    alpha=1.0,
    s=5,
    marker="o",
)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.5)

# Get the actual data range
vmin = proximity_receiver_df["proximity_score"].min()
vmax = proximity_receiver_df["proximity_score"].max()

sm = plt.cm.ScalarMappable(
    cmap=LinearSegmentedColormap.from_list(
        "custom_palette",
        [
            "#4444FF",  # Blue for high proximity (close distance)
            "#FF4444",  # Red for low proximity (far distance)  
        ],
        N=500,
    ),
    norm=Normalize(vmin=vmin, vmax=vmax)  # Use the actual data range
)
sm.set_array([])
cbar = plt.colorbar(
    sm, ax=ax, label="Proximity Score", orientation="vertical", pad=0.01, 
)
cbar.set_label(f"Proximity Score\n(Distance to Any {sender_ct} Type)")
distances = np.linspace(0, 200, 5)
proximity_scores = [distance_to_scale(dist) for dist in distances]
cbar.set_ticks(proximity_scores)
cbar.set_ticklabels([f"{dist:.0f} μm" for dist in distances])
scatter.get_legend().remove()
plt.title(f"Spatial Distribution of {receiver_ct}\nProximity to Any {sender_ct} Type")
plt.xlabel("X (μm)")
plt.ylabel("Y (μm)")
plt.savefig(
    figures_dir / f"proximity_spatial_plot_{receiver_ct}_any_sender_type.png",
    dpi=300,
    bbox_inches="tight",
)
scatter.collections[0].set_visible(False)
plt.savefig(
    figures_dir / f"proximity_spatial_plot_{receiver_ct}_any_sender_type.svg",
    dpi=300,
    bbox_inches="tight",
)
scatter.collections[0].set_visible(True)
plt.show()

# %% Choose the head with the highest variance in attention scores
attention_to_senders = attention_to_senders.set_index("neighbor_cell_idx")
max_var_head_idx = attention_to_senders[["head", "attention_score"]].groupby("head").var().idxmax().values[0]
head_attention_to_senders_df = attention_to_senders[attention_to_senders["head"] == max_var_head_idx]["attention_score"].copy()
print(f"Head with highest variance in attention scores: {max_var_head_idx}")

# %% Exclude the highest variance head and replot the spatial plot of the max attention on the receivers
summed_attention_exclude_max_var_head_df = summed_attention_df[summed_attention_df["head"] != max_var_head_idx].copy()

receiver_adata = adata[adata.obs[labels_key] == receiver_ct].copy()
receiver_neighbor_idxs = receiver_adata.obsm["_nn_idx"]
exclude_max_var_head_receiver_df = receiver_adata.obsm["spatial"].copy()
exclude_max_var_head_receiver_df["sample"] = receiver_adata.obs["sample"]
exclude_max_var_head_receiver_df = exclude_max_var_head_receiver_df[exclude_max_var_head_receiver_df["sample"] == sample_id]

# Map empirical attention scores to receiver cells
exclude_max_var_head_receiver_df[f"empirical_attention_{sender_ct}"] = exclude_max_var_head_receiver_df.index.map(empirical_attention_df).fillna(0)

# %% Plot the max attention received by receivers from any sender type
plt.figure(figsize=(15, 8))
scatter = sns.scatterplot(
    exclude_max_var_head_receiver_df,
    x="X",
    y="Y",
    hue=f"empirical_attention_{sender_ct}",
    palette=LinearSegmentedColormap.from_list(
        "custom_palette",
        [
            "#4444FF",  # Blue for high attention
            "#FF4444",  # Red for low attention
        ],
        N=100,
    ),
    alpha=1.0,
    s=8,
    marker="o",
)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.5)
plt.title(f"Spatial Distribution of {receiver_ct}\nEmpirical Attention to {sender_ct} (Excluding Max Variance Head)")
plt.xlabel("X (μm)")
plt.ylabel("Y (μm)")

# Create colorbar for attention score
sm = plt.cm.ScalarMappable(
    cmap=LinearSegmentedColormap.from_list(
        "custom_palette",
        [
            "#4444FF",  # Blue for high attention
            "#FF4444",  # Red for low attention
        ],
        N=100,
    )
)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), label="Empirical Attention Score")
scatter.get_legend().remove()
plt.savefig(
    figures_dir / f"empirical_attention_spatial_plot_{receiver_ct}_{sender_ct}_exclude_max_var_head.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    figures_dir / f"empirical_attention_spatial_plot_{receiver_ct}_{sender_ct}_exclude_max_var_head.svg",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %% Only plot the attention scores on the max variance head for senders that are within distance of the receiver
summed_attention_max_var_head_df = summed_attention_df[summed_attention_df["head"] == max_var_head_idx].copy()

receiver_adata = adata[adata.obs[labels_key] == receiver_ct].copy()
receiver_neighbor_idxs = receiver_adata.obsm["_nn_idx"]
max_var_head_receiver_df = receiver_adata.obsm["spatial"].copy()
max_var_head_receiver_df["sample"] = receiver_adata.obs["sample"]

max_var_head_receiver_df = max_var_head_receiver_df[max_var_head_receiver_df["sample"] == sample_id]

# Map empirical attention scores to receiver cells
max_var_head_receiver_df[f"empirical_attention_{sender_ct}"] = max_var_head_receiver_df.index.map(summed_attention_max_var_head_df.groupby("cell_idx").max()[f"all_heads_{sender_ct}"]).fillna(0)

# %% Plot the max attention received by receivers from the max variance head
plt.figure(figsize=(15, 8))
scatter = sns.scatterplot(
    max_var_head_receiver_df,
    x="X",
    y="Y",
    hue=f"empirical_attention_{sender_ct}",
    palette=LinearSegmentedColormap.from_list(
        "custom_palette",
        [
            "#4444FF",  # Blue for high attention
            "#FF4444",  # Red for low attention
        ],
        N=100,
    ),
    alpha=1.0,
    s=8,
    marker="o",
)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.5)
plt.title(f"Spatial Distribution of {receiver_ct}\nEmpirical Attention to {sender_ct} (Max Variance Head)")
plt.xlabel("X (μm)")
plt.ylabel("Y (μm)")

# Create colorbar for attention score
sm = plt.cm.ScalarMappable(
    cmap=LinearSegmentedColormap.from_list(
        "custom_palette",
        [
            "#4444FF",  # Blue for high attention
            "#FF4444",  # Red for low attention
        ],
        N=100,
    )
)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), label="Empirical Attention Score")
scatter.get_legend().remove()
plt.savefig(
    figures_dir / f"empirical_attention_spatial_plot_{receiver_ct}_{sender_ct}_max_var_head.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    figures_dir / f"empirical_attention_spatial_plot_{receiver_ct}_{sender_ct}_max_var_head.svg",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %% Compute the spearman correlation between the empirical attention scores and the proximity scores
merged_scores = pd.merge(receiver_df, proximity_receiver_df, left_index=True, right_index=True)
spearman_corr, p_val = spearmanr(merged_scores[f"empirical_attention_{sender_ct}"], merged_scores["proximity_score"])
print(f"Spearman correlation between empirical attention and proximity scores: {spearman_corr}")
print(f"p-value: {p_val}")

# %% Compute the LISA score between the empirical attention and proximity scores
# Create spatial coordinates from merged scores
coords = merged_scores[['X_x', 'Y_x']].values

# Create spatial weights matrix using k-nearest neighbors (k=8 for spatial neighbors)
w = KNN.from_array(coords, k=8)
w.transform = 'r'  # Row-standardize the weights

# Extract the variables for local bivariate analysis
attention_scores = merged_scores[f"empirical_attention_{sender_ct}"].values
proximity_scores = merged_scores["proximity_score"].values

# Compute Local Bivariate Moran's I (LISA)
lisa_bv = Moran_Local_BV(attention_scores, proximity_scores, w)

# Add LISA scores to merged dataframe
merged_scores['lisa_I'] = lisa_bv.Is
merged_scores['lisa_pvalue'] = lisa_bv.p_sim
significant_lisa_scores = merged_scores[merged_scores['lisa_pvalue'] < 0.05]
print(f"Number of significant LISA scores: {len(significant_lisa_scores)}")
print(f"Average LISA score: {significant_lisa_scores['lisa_I'].mean()}")

# %% Plot the LISA scores on the spatial plot
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    merged_scores, x="X_x", y="Y_x", hue="lisa_I", alpha=0.7, s=8, palette="viridis"
)
ax = plt.gca()
norm = plt.Normalize(merged_scores["lisa_I"].min(), merged_scores["lisa_I"].max())
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label="LISA Score")
ax.set_title(f"LISA scores between empirical attention and proximity scores")
ax.set_xlabel("X (μm)")
ax.set_ylabel("Y (μm)")
scatter.get_legend().remove()
plt.savefig(
    figures_dir / f"lisa_scores_spatial_plot_{receiver_ct}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    figures_dir / f"lisa_scores_spatial_plot_{receiver_ct}.svg",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %% Plot the expression of a gene of interest and for a cell type of interest on the spatial plot
gene = "AGR3"
adata_subset = adata[adata.obs[labels_key] == receiver_ct].copy() 
plot_df = adata_subset.obsm["spatial"].copy()
plot_df["expression"] = adata_subset.X[:, adata_subset.var_names.get_loc(gene)].toarray()

plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    plot_df, x="X", y="Y", hue="expression", alpha=0.7, s=8, palette="viridis"
)
ax = plt.gca()
ax.set_title(f"Expression of {gene} in {receiver_ct}")
norm = plt.Normalize(plot_df["expression"].min(), plot_df["expression"].max())
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label=f"{gene} expression")
ax.set_xlabel("X (μm)")
ax.set_ylabel("Y (μm)")
scatter.get_legend().remove()
plt.savefig(
    figures_dir / f"gene_expression_{gene}_{receiver_ct}.png",
    dpi=300,
    bbox_inches="tight",
)
scatter.collections[0].set_visible(False)
plt.savefig(
    figures_dir / f"gene_expression_{gene}_{receiver_ct}.svg",
    dpi=300,
    bbox_inches="tight",
)
scatter.collections[0].set_visible(True)
plt.show()
# %% Compute the spearman correlation between the empirical attention scores and the expression of a gene of interest
merged_scores = pd.merge(max_var_head_receiver_df, plot_df, left_index=True, right_index=True)
spearman_corr, p_val = spearmanr(merged_scores[f"empirical_attention_{sender_ct}"], merged_scores["expression"])
print(f"Spearman correlation between empirical attention and expression of {gene}: {spearman_corr}")
print(f"p-value: {p_val}")

# %%
