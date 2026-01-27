# %% Import libraries
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pytorch_lightning as pl
from amici import AMICI

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from tqdm import tqdm
import numpy as np
import pandas as pd
import plotly.graph_objects as go

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

saved_models_dir = saved_models_dir = f"saved_models/xenium_sample1_proseg_sweep_{data_date}_model_{model_date}"
wandb_run_id = "te7pkv3z"
wandb_sweep_id = "g3mucw4s"
model_path = os.path.join(
    saved_models_dir,
    f"xenium_{seed}_sweep_{wandb_sweep_id}_{wandb_run_id}_params_{model_date}",
)

# %% Select subset of cell types for interpretation and visualize
def visualize_spatial_distribution(
    adata, labels_key="celltype_train_grouped", palette=CELL_TYPE_PALETTE, x_lim=None, y_lim=None, show_test=False
):
    plt.figure(figsize=(20, 6))
    plot_df = adata.obsm["spatial"].copy()
    plot_df["sample"] = adata.obs["sample"]
    plot_df[labels_key] = adata.obs[labels_key]
    plot_df["train_test"] = adata.obs["train_test_split"]
    
    # Create scatter plot
    scatter = sns.scatterplot(
        plot_df, x="X", y="Y", hue=labels_key, alpha=0.7, s=8, palette=palette
    )

    # Add two boxes around test observations for each sample
    if show_test:
        for sample in adata[adata.obs["train_test_split"] == "test"].obs["sample"].unique():
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
    plt.savefig(f"figures/xenium_sample1_proseg_spatial_hub_subset_analysis_{labels_key}.png")
    scatter.collections[0].set_visible(False)
    plt.savefig(f"figures/xenium_sample1_proseg_spatial_hub_subset_analysis_{labels_key}.svg")
    scatter.collections[0].set_visible(True)
    plt.show()

# %% Load model
visualize_spatial_distribution(adata)
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

# %% Get the attention patterns and compute hub analysis
attention_patterns = model.get_attention_patterns()

# %%
#############################################################################
# HUB ANALYSIS WITH COMPOSITION OF HIGH ATTENTION CELL TYPES AS INTERACTION COMPOSITION VECTOR
#############################################################################
quantile = 0.9

def find_optimal_clusters(data, min_clusters=2, max_clusters=12, random_state=42):
    """Find optimal number of clusters using silhouette analysis."""
    silhouette_scores = []
    cluster_range = range(min_clusters, max_clusters + 1)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For k={k}, silhouette score = {silhouette_avg:.4f}")

    # Find optimal k
    optimal_k = cluster_range[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)

    print(f"\nOptimal number of clusters: {optimal_k} (silhouette score: {best_score:.4f})")

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for Optimal k')
    plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return optimal_k, best_score

# First compute hub analysis without clustering to get composition vectors for optimal k search
high_interacting_counts_norm_no_cluster = attention_patterns.compute_communication_hubs(
    attention_quantile_threshold=quantile,
    n_clusters=2,  # temporary, will recompute with optimal k
    random_state=seed
).drop(columns=["hub_cluster"])

# Find optimal number of clusters for hub clustering
print("Finding optimal number of clusters for hub clustering...")
optimal_k_hub, _ = find_optimal_clusters(high_interacting_counts_norm_no_cluster, random_state=seed)

# Use the factored-out hub analysis method from AMICIAttentionModule with optimal k
high_interacting_counts_norm = attention_patterns.compute_communication_hubs(
    attention_quantile_threshold=quantile,
    n_clusters=optimal_k_hub,
    random_state=seed
)
adata.obs["hub_cluster"] = high_interacting_counts_norm.loc[adata.obs_names, "hub_cluster"]

# %% Plot the cells on spatial plot colored by the hub clusters
cluster_palette = {
    0: "#2E5BBA",  # Deep Blue
    1: "#FF8C42",  # Warm Orange
    2: "#228B22",  # Forest Green
    3: "#8E44AD",  # Plum Purple
    4: "#B22222",  # Fire Brick (cooler red with blue undertones)
    5: "#FFD700",  # Gold/Pure Yellow (brighter, more yellow)
    6: "#5D6D7E",  # Slate Gray
    7: "#16A085",  # Teal
    8: "#FF6347",  # Tomato (warmer red with orange undertones)
    9: "#8B4513",  # Chocolate Brown
    10: "#FF1493", # Deep Pink
    11: "#32CD32"  # Lime Green
}
visualize_spatial_distribution(adata, labels_key="hub_cluster", palette=cluster_palette, )

# %% Get hub cluster labels and create composition plot
# Calculate the mean composition for each hub cluster
sender_cell_type_composition = high_interacting_counts_norm.groupby("hub_cluster").mean()
sender_cell_type_composition_norm = sender_cell_type_composition.div(sender_cell_type_composition.sum(axis=1), axis=0)
sender_cell_type_composition_norm = sender_cell_type_composition_norm.fillna(0)

# Create stacked bar plot
plt.figure(figsize=(12, 8))

# Create the stacked bar plot
sender_cell_type_composition_norm.plot(
    kind="bar",
    stacked=True,
    color=[CELL_TYPE_PALETTE.get(cell_type, '#888888') for cell_type in sender_cell_type_composition_norm.columns],
    figsize=(12, 8)
)

plt.title("Sender Cell Type Composition by Hub Cluster", fontsize=16)
plt.xlabel("Hub Cluster", fontsize=14)
plt.ylabel("Proportion of High-Interacting Neighbors", fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/xenium_hub_cluster_composition.png", dpi=300, bbox_inches='tight')
plt.show()

# %% Look at the receiver cell type compositions
high_interacting_counts_norm["receiver_label"] = adata.obs[labels_key].loc[high_interacting_counts.index]

# Calculate the mean composition for each hub cluster
receiver_cell_type_composition = pd.DataFrame(high_interacting_counts_norm[["receiver_label", "hub_cluster"]].groupby("hub_cluster").value_counts()).reset_index()
receiver_cell_type_composition = receiver_cell_type_composition.pivot(columns="receiver_label", index="hub_cluster", values="count")
receiver_cell_type_composition_norm = receiver_cell_type_composition.div(receiver_cell_type_composition.sum(axis=1), axis=0)
receiver_cell_type_composition_norm = receiver_cell_type_composition_norm.fillna(0)

# Plot the composition
receiver_cell_type_composition_norm.plot(
    kind="bar",
    stacked=True,
    color=[CELL_TYPE_PALETTE.get(cell_type, '#888888') for cell_type in receiver_cell_type_composition_norm.columns],
    figsize=(12, 8)
)
plt.title("Receiver Cell Type Composition by Hub Cluster", fontsize=16)
plt.xlabel("Hub Cluster", fontsize=14)
plt.ylabel("Proportion of High-Interacting Neighbors", fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/xenium_hub_cluster_composition_receiver.png", dpi=300, bbox_inches='tight')
plt.show()

# %% Compute regular cell type composition for the dataset to compare to the interacting hubs
nn_labels = attention_patterns._nn_idxs_df.map(
    lambda x: adata.obs[labels_key][x]
)

# Convert to composition vector format using vectorized operations
# Melt the dataframe to long format
nn_labels_melted = nn_labels.reset_index().melt(
    id_vars='index', 
    var_name='neighbor_position', 
    value_name='cell_type'
)

# Count cell types per target cell using groupby and value_counts
neighbor_composition = (nn_labels_melted
                       .groupby(['index', 'cell_type'])
                       .size()
                       .unstack(fill_value=0))

# Ensure all cell types are represented as columns
all_cell_types = adata.obs[labels_key].unique()
missing_cell_types = set(all_cell_types) - set(neighbor_composition.columns)
for cell_type in missing_cell_types:
    neighbor_composition[cell_type] = 0

# Reorder columns to match original cell types order
neighbor_composition = neighbor_composition[all_cell_types]

# Convert to proportions instead of counts
neighbor_composition_norm = neighbor_composition.div(neighbor_composition.sum(axis=1), axis=0)
neighbor_composition_norm = neighbor_composition_norm.fillna(0)

# %% Cluster the composition vectors
kmeans_composition = KMeans(n_clusters=optimal_k_hub, random_state=seed)
kmeans_composition.fit(neighbor_composition_norm.values)
neighbor_composition_norm["composition_cluster"] = kmeans_composition.labels_
adata.obs["composition_cluster"] = neighbor_composition_norm.loc[adata.obs_names, "composition_cluster"]

# %% Plot the cells on spatial plot colored by the composition clusters
visualize_spatial_distribution(adata, labels_key="composition_cluster", palette=cluster_palette)

# %% Plot the composition of the composition clusters

# Calculate the mean composition for each composition cluster
cell_type_composition = neighbor_composition_norm.groupby("composition_cluster").mean()

# Create stacked bar plot
plt.figure(figsize=(12, 8))

# Create the stacked bar plot
cell_type_composition.plot(
    kind="bar",
    stacked=True,
    color=[CELL_TYPE_PALETTE.get(cell_type, '#888888') for cell_type in cell_type_composition.columns],
    figsize=(12, 8)
)

plt.title("Sender Cell Type Composition by Composition Cluster", fontsize=16)
plt.xlabel("Composition Cluster", fontsize=14)
plt.ylabel("Proportion of Neighbors", fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/xenium_composition_cluster_composition.png", dpi=300, bbox_inches='tight')
plt.savefig("figures/xenium_composition_cluster_composition.svg", dpi=300, bbox_inches='tight')
plt.show()

# %% Calculate the ARI between the hub clusters and the composition clusters

# Get the cluster labels for both methods
hub_labels = adata.obs["hub_cluster"].values
composition_labels = adata.obs["composition_cluster"].values

# Calculate ARI
ari_comp_score = adjusted_rand_score(hub_labels, composition_labels)

print(f"Adjusted Rand Index between hub clusters and composition clusters: {ari_comp_score:.4f}")

# %% Calculate the adjusted mutual information between the hub clusters and the composition clusters

# Get the cluster labels for both methods
hub_labels = adata.obs["hub_cluster"].values
composition_labels = adata.obs["composition_cluster"].values

# Calculate AMI
ami_comp_score = adjusted_mutual_info_score(hub_labels, composition_labels)

print(f"Adjusted Mutual Information between hub clusters and composition clusters: {ami_comp_score:.4f}")

# %% Calculate the adjusted rand index between the composition clusters and the cell type labels
# Get the cluster labels for both methods
hub_labels = adata.obs["hub_cluster"].values
cell_type_labels = adata.obs[labels_key].values

# Calculate ARI
ari_cell_type_score = adjusted_rand_score(composition_labels, cell_type_labels)

print(f"Adjusted Rand Index between composition clusters and cell type labels: {ari_cell_type_score:.4f}")

# %% Calculate the adjusted mutual information between the hub clusters and the cell type labels
hub_labels = adata.obs["hub_cluster"].values
cell_type_labels = adata.obs[labels_key].values

# Calculate AMI
ami_cell_type_score = adjusted_mutual_info_score(hub_labels, cell_type_labels)

print(f"Adjusted Mutual Information between hub clusters and cell type labels: {ami_cell_type_score:.4f}")

# %% Plot the ARI and AMI scores between the hub clusters and composition and cell type labels

# Create a dataframe with all the scores
scores_df = pd.DataFrame({
    'Metric': ['ARI\n(Hub vs Composition)', 'AMI\n(Hub vs Composition)', 
               'ARI\n(Hub vs Cell Type)', 'AMI\n(Hub vs Cell Type)'],
    'Score': [ari_comp_score, ami_comp_score, ari_cell_type_score, ami_cell_type_score],
    'Comparison': ['Hub vs Composition', 'Hub vs Composition', 
                   'Hub vs Cell Type', 'Hub vs Cell Type']
})

plt.figure(figsize=(12, 6))
bars = sns.barplot(data=scores_df, x='Metric', y='Score', hue='Comparison', palette='Set2')
plt.ylim(0, 1)
plt.title('Clustering Comparison Metrics', fontsize=16)
plt.ylabel('Score', fontsize=14)
plt.xlabel('Metric', fontsize=14)
plt.legend(title='Comparison', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("figures/xenium_clustering_comparison_metrics.png", dpi=300, bbox_inches='tight')
plt.savefig("figures/xenium_clustering_comparison_metrics.svg", dpi=300, bbox_inches='tight')
plt.show()

# %% Zooming into two specific hub communication clusters
hub_cluster = 9
high_interacting_counts_cluster = high_interacting_counts_norm[high_interacting_counts_norm["hub_cluster"] == hub_cluster]
high_interacting_counts_cluster_melted = pd.melt(
    high_interacting_counts_cluster.reset_index(),
    id_vars=["receiver_label", "index"],
    value_vars=list(adata.obs[labels_key].unique()),
    var_name="neighbor_label",
    value_name="proportion",
)
high_interacting_ct_props = high_interacting_counts_cluster_melted[["receiver_label", "neighbor_label", "proportion"]].groupby(["receiver_label", "neighbor_label"]).mean()
high_interacting_ct_props = high_interacting_ct_props.reset_index()
high_interacting_ct_props = high_interacting_ct_props[high_interacting_ct_props["receiver_label"] != high_interacting_ct_props["neighbor_label"]]

# Weight by receiver cell proportions in the cluster
receiver_weights = receiver_cell_type_composition_norm.loc[hub_cluster]
high_interacting_ct_props["receiver_weight"] = high_interacting_ct_props["receiver_label"].map(receiver_weights).astype(float)
high_interacting_ct_props["weighted_proportion"] = high_interacting_ct_props["proportion"] * high_interacting_ct_props["receiver_weight"]

# Calculate weighted average for each sender type
weighted_props = high_interacting_ct_props.groupby("neighbor_label").agg({
    "weighted_proportion": "sum",
    "receiver_weight": "sum"
}).reset_index()
weighted_props["final_proportion"] = weighted_props["weighted_proportion"] / weighted_props["receiver_weight"]

print("Weighted sender proportions:")
for _, row in weighted_props.sort_values("final_proportion", ascending=False).iterrows():
    print(f"  {row['neighbor_label']}: {row['final_proportion']:.4f}")

# Also keep the detailed receiver-sender pairs for the Sankey plot
high_interacting_ct_props_detailed = high_interacting_ct_props.copy()

# %% Create a sankey plot of the hub cluster
# Get the receiver and sender proportions for the specific hub cluster
receiver_proportions = receiver_cell_type_composition_norm.loc[hub_cluster].dropna()
sender_proportions = sender_cell_type_composition_norm.loc[hub_cluster].dropna()

# Filter proportion data to only include cell types present in this hub cluster AND positive proportions
filtered_proportions = high_interacting_ct_props_detailed[
    (high_interacting_ct_props_detailed['receiver_label'].isin(receiver_proportions.index)) &
    (high_interacting_ct_props_detailed['neighbor_label'].isin(sender_proportions.index)) &
    (high_interacting_ct_props_detailed['weighted_proportion'] > 0)
].copy()

# Get unique cell types for senders and receivers
sender_types = sender_proportions.index.tolist()
receiver_types = receiver_proportions.index.tolist()

# Create node labels: senders first, then receivers
node_labels = [f"Sender_{ct}" for ct in sender_types] + [f"Receiver_{ct}" for ct in receiver_types]

# Create node colors using the cell type palette
sender_colors = [CELL_TYPE_PALETTE.get(ct, '#888888') for ct in sender_types]
receiver_colors = [CELL_TYPE_PALETTE.get(ct, '#888888') for ct in receiver_types]
node_colors = sender_colors + receiver_colors

# Prepare data for links
source_indices = []
target_indices = []
link_values = []
link_colors = []

for _, row in filtered_proportions.iterrows():
    sender_idx = sender_types.index(row['neighbor_label'])
    receiver_idx = len(sender_types) + receiver_types.index(row['receiver_label'])
    
    source_indices.append(sender_idx)
    target_indices.append(receiver_idx)
    link_values.append(row['weighted_proportion'])  # Use weighted proportion
    
    # Create link color based on sender color with transparency
    sender_color = CELL_TYPE_PALETTE.get(row['neighbor_label'], '#888888')
    # Convert hex to rgba with transparency
    hex_color = sender_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    link_colors.append(f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.4)')

# Create the Sankey plot
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=0,
        thickness=20,  # Single thickness for all nodes
        line=dict(color="black", width=0),
        color=node_colors,
    ),
    link=dict(
        source=source_indices,
        target=target_indices,
        value=link_values,
        color=link_colors
    )
)])

fig.update_layout(
    title_text=f"Weighted High-Interacting Cell Type Proportions in Hub Cluster {hub_cluster}<br><sup>Weighted by receiver cell abundance</sup>",
    font_size=12,
    width=500,
    height=400
)
fig.show()

# %% Grid search over quantile thresholds and number of clusters
# Define parameter ranges
quantile_thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
n_clusters_range = [4, 6, 8, 10, 12, 14, 16]

# File paths for cached results
grid_search_cache_dir = "data/xenium_sample1/grid_search_cache"
hub_results_cache_dir = os.path.join(grid_search_cache_dir, "hub_results")
os.makedirs(grid_search_cache_dir, exist_ok=True)
os.makedirs(hub_results_cache_dir, exist_ok=True)
ari_cache_path = os.path.join(grid_search_cache_dir, "hub_grid_search_ari.csv")
ami_cache_path = os.path.join(grid_search_cache_dir, "hub_grid_search_ami.csv")

# Dictionary to store hub results (loaded from cache or computed)
hub_results_dict = {}

# Check if cached results exist
if os.path.exists(ari_cache_path) and os.path.exists(ami_cache_path):
    print("Loading cached grid search results...")
    ari_df = pd.read_csv(ari_cache_path, index_col=0)
    ami_df = pd.read_csv(ami_cache_path, index_col=0)
    ari_results = ari_df.values
    ami_results = ami_df.values

    # Load hub results from cache
    print("Loading cached hub results...")
    for n_clusters in n_clusters_range:
        for q_threshold in quantile_thresholds:
            hub_cache_path = os.path.join(
                hub_results_cache_dir,
                f"hub_q{q_threshold}_k{n_clusters}.csv"
            )
            if os.path.exists(hub_cache_path):
                hub_results_dict[(q_threshold, n_clusters)] = pd.read_csv(hub_cache_path, index_col=0)

    print("Cached results loaded!")
else:
    # Store results
    ari_results = np.zeros((len(n_clusters_range), len(quantile_thresholds)))
    ami_results = np.zeros((len(n_clusters_range), len(quantile_thresholds)))

    # Get cell type labels for comparison
    cell_type_labels = adata.obs[labels_key].values

    print("Running grid search over quantile thresholds and number of clusters...")
    total_combinations = len(n_clusters_range) * len(quantile_thresholds)
    with tqdm(total=total_combinations, desc="Grid search") as pbar:
        for i, n_clusters in enumerate(n_clusters_range):
            for j, q_threshold in enumerate(quantile_thresholds):
                pbar.set_postfix(n_clusters=n_clusters, quantile=q_threshold)

                # Compute hub analysis with current parameters
                hub_result = attention_patterns.compute_communication_hubs(
                    attention_quantile_threshold=q_threshold,
                    n_clusters=n_clusters,
                    random_state=seed
                )

                # Store hub result
                hub_results_dict[(q_threshold, n_clusters)] = hub_result

                # Save hub result to cache
                hub_cache_path = os.path.join(
                    hub_results_cache_dir,
                    f"hub_q{q_threshold}_k{n_clusters}.csv"
                )
                hub_result.to_csv(hub_cache_path)

                # Get hub cluster labels
                hub_labels = hub_result.loc[adata.obs_names, "hub_cluster"].values

                # Compute ARI and AMI against cell type labels
                ari_results[i, j] = adjusted_rand_score(hub_labels, cell_type_labels)
                ami_results[i, j] = adjusted_mutual_info_score(hub_labels, cell_type_labels)

                pbar.update(1)

    print("Grid search complete!")

    # Convert to DataFrames for easier plotting and saving
    ari_df = pd.DataFrame(
        ari_results,
        index=[str(k) for k in n_clusters_range],
        columns=[str(q) for q in quantile_thresholds]
    )
    ami_df = pd.DataFrame(
        ami_results,
        index=[str(k) for k in n_clusters_range],
        columns=[str(q) for q in quantile_thresholds]
    )

    # Save metrics results to cache
    ari_df.to_csv(ari_cache_path)
    ami_df.to_csv(ami_cache_path)
    print(f"Results saved to {grid_search_cache_dir}/")

# %%
# plot ARI and AMI between every pair of parameters
ari_matrix = np.zeros((len(hub_results_dict), len(hub_results_dict)))
ami_matrix = np.zeros((len(hub_results_dict), len(hub_results_dict)))
for i, k1 in enumerate(hub_results_dict.keys()):
    for j, k2 in enumerate(hub_results_dict.keys()):
        ari = adjusted_rand_score(hub_results_dict[k1].loc[adata.obs_names, "hub_cluster"].values, hub_results_dict[k2].loc[adata.obs_names, "hub_cluster"].values)
        ami = adjusted_mutual_info_score(hub_results_dict[k1].loc[adata.obs_names, "hub_cluster"].values, hub_results_dict[k2].loc[adata.obs_names, "hub_cluster"].values)
        ari_matrix[i, j] = ari
        ami_matrix[i, j] = ami

# %%
# Plot heatmaps for a fixed quantile threshold (0.9): rows/columns = cluster numbers, values = ARI/AMI between clusterings
import matplotlib.ticker as mticker

fixed_quantile = 0.9
fixed_q_key = [k for k in hub_results_dict.keys() if np.isclose(k[0], fixed_quantile)]
if not fixed_q_key:
    print(f"No clustering results for quantile threshold {fixed_quantile}")
else:
    # Make a list of available cluster numbers for this quantile
    q_k_set = sorted([k[1] for k in fixed_q_key])
    n = len(q_k_set)
    ari_mat = np.zeros((n, n))
    ami_mat = np.zeros((n, n))
    for i, k1 in enumerate(q_k_set):
        for j, k2 in enumerate(q_k_set):
            hub_labels_1 = hub_results_dict[(fixed_quantile, k1)].loc[adata.obs_names, "hub_cluster"].values
            hub_labels_2 = hub_results_dict[(fixed_quantile, k2)].loc[adata.obs_names, "hub_cluster"].values
            ari_mat[i, j] = adjusted_rand_score(hub_labels_1, hub_labels_2)
            ami_mat[i, j] = adjusted_mutual_info_score(hub_labels_1, hub_labels_2)

    fig, ax = plt.subplots(1,2, figsize=(14, 6))
    im0 = sns.heatmap(ari_mat, annot=True, fmt=".2f", xticklabels=q_k_set, yticklabels=q_k_set, ax=ax[0], cmap="Blues", vmin=0, vmax=1)
    ax[0].set_title(f"ARI between cluster numbers (q={fixed_quantile})")
    ax[0].set_xlabel("n_clusters")
    ax[0].set_ylabel("n_clusters")
    im1 = sns.heatmap(ami_mat, annot=True, fmt=".2f", xticklabels=q_k_set, yticklabels=q_k_set, ax=ax[1], cmap="Purples", vmin=0, vmax=1)
    ax[1].set_title(f"AMI between cluster numbers (q={fixed_quantile})")
    ax[1].set_xlabel("n_clusters")
    ax[1].set_ylabel("n_clusters")
    plt.tight_layout()
    plt.savefig(f"figures/xenium_hub_grid_search_heatmaps_q{fixed_quantile}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"figures/xenium_hub_grid_search_heatmaps_q{fixed_quantile}.svg", dpi=300, bbox_inches='tight')
    plt.show()

# For a fixed number of clusters (e.g. 10), plot heatmap with quantile thresholds as rows/columns
fixed_n_clusters = 10
fixed_k_key = [k for k in hub_results_dict.keys() if k[1] == fixed_n_clusters]
q_set = sorted([k[0] for k in fixed_k_key])
n = len(q_set)
ari_mat_q = np.zeros((n, n))
ami_mat_q = np.zeros((n, n))
for i, q1 in enumerate(q_set):
    for j, q2 in enumerate(q_set):
        hub_labels_1 = hub_results_dict[(q1, fixed_n_clusters)].loc[adata.obs_names, "hub_cluster"].values
        hub_labels_2 = hub_results_dict[(q2, fixed_n_clusters)].loc[adata.obs_names, "hub_cluster"].values
        ari_mat_q[i, j] = adjusted_rand_score(hub_labels_1, hub_labels_2)
        ami_mat_q[i, j] = adjusted_mutual_info_score(hub_labels_1, hub_labels_2)

fig, ax = plt.subplots(1,2, figsize=(14, 6))
im0 = sns.heatmap(ari_mat_q, annot=True, fmt=".2f", xticklabels=np.round(q_set,3), yticklabels=np.round(q_set,3), ax=ax[0], cmap="Blues", vmin=0, vmax=1)
ax[0].set_title(f"ARI for varying quantile thresholds (n_clusters={fixed_n_clusters})")
ax[0].set_xlabel("quantile_threshold")
ax[0].set_ylabel("quantile_threshold")
im1 = sns.heatmap(ami_mat_q, annot=True, fmt=".2f", xticklabels=np.round(q_set,3), yticklabels=np.round(q_set,3), ax=ax[1], cmap="Purples", vmin=0, vmax=1)
ax[1].set_title(f"AMI for varying quantile thresholds (n_clusters={fixed_n_clusters})")
ax[1].set_xlabel("quantile_threshold")
ax[1].set_ylabel("quantile_threshold")
plt.tight_layout()
plt.savefig(f"figures/xenium_hub_grid_search_heatmaps_k{fixed_n_clusters}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"figures/xenium_hub_grid_search_heatmaps_k{fixed_n_clusters}.svg", dpi=300, bbox_inches='tight')
plt.show()