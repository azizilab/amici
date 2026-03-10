# %% Import libraries
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import seaborn as sns
import os
import json
from amici import AMICI

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from tqdm import tqdm
import numpy as np
import pandas as pd

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

ROI_PALETTE = {
    "DCIS_1": "#E69F00",
    "DCIS_2": "#1a476e",
    "Invasive": "#cf4242",
    "Other": "#cccccc",
}

# %% Seed everything
seed = 18

# %% Load data
labels_key = "celltype_train_grouped"
data_date = "2025-05-01"
model_date = "2025-05-02"
adata = sc.read_h5ad(f"./data/xenium_sample1/xenium_sample1_filtered_{data_date}.h5ad")
adata = adata[adata.obs["sample"] == "0"]

saved_models_dir = (
    f"saved_models/xenium_sample1_proseg_sweep_{data_date}_model_{model_date}"
)
wandb_run_id = "te7pkv3z"
wandb_sweep_id = "g3mucw4s"
model_path = os.path.join(
    saved_models_dir,
    f"xenium_{seed}_sweep_{wandb_sweep_id}_{wandb_run_id}_params_{model_date}",
)

# %% Load geojson + alignment matrix, convert polygons to micron space
with open(
    "./data/xenium_sample1/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.geojson"
) as f:
    geojson_data = json.load(f)

affine_he = np.loadtxt(
    "./data/xenium_sample1/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_imagealignment.csv",
    delimiter=",",
)
XENIUM_PIXEL_SIZE = 0.2125  # microns per Xenium instrument pixel

# Convert each ROI polygon from HE pixel coords to adata micron coords
roi_polygons = []  # list of (name, Path, micron_coords_array)
for feature in geojson_data["features"]:
    name = feature["properties"]["name"]
    pixel_coords = np.array(feature["geometry"]["coordinates"][0])  # (N, 2)

    ones = np.ones((pixel_coords.shape[0], 1))
    pixel_homogeneous = np.hstack([pixel_coords, ones])  # (N, 3)
    xenium_pixels = (pixel_homogeneous @ affine_he.T)[:, :2]  # (N, 2)
    micron_coords = xenium_pixels * XENIUM_PIXEL_SIZE

    path = Path(micron_coords)
    roi_polygons.append((name, path, micron_coords))

print(f"Loaded {len(roi_polygons)} ROI polygons:")
for name, _, coords in roi_polygons:
    print(
        f"  {name}: {len(coords)} vertices, "
        f"X range [{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}], "
        f"Y range [{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}]"
    )

# %% Coordinate verification plot
plot_df = adata.obsm["spatial"].copy()
plot_df["sample"] = adata.obs["sample"]
plot_df[labels_key] = adata.obs[labels_key]

fig, ax = plt.subplots(figsize=(14, 10))
scatter = sns.scatterplot(
    data=plot_df,
    x="X",
    y="Y",
    hue=labels_key,
    alpha=0.5,
    s=4,
    palette=CELL_TYPE_PALETTE,
    ax=ax,
)

# Overlay ROI polygon boundaries
for name, _, micron_coords in roi_polygons:
    color = ROI_PALETTE.get(name, "#000000")
    polygon = plt.Polygon(
        micron_coords,
        fill=False,
        edgecolor=color,
        linewidth=2,
        linestyle="--",
        label=name,
    )
    ax.add_patch(polygon)

ax.set_xlabel("X (microns)")
ax.set_ylabel("Y (microns)")
ax.set_title("ROI Coordinate Verification — Rep1 cells with geojson polygons")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=2)
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig(
    "figures/xenium_roi_coordinate_verification.png", dpi=300, bbox_inches="tight"
)
plt.show()

# %% Assign ROI labels to cells
spatial_coords = adata.obsm["spatial"][["X", "Y"]].values
roi_labels = np.full(len(adata), "Other", dtype=object)

# Process polygons in geojson order (DCIS_1 first, then DCIS_2, then Invasive)
# Last-match wins for overlapping regions
for name, path, _ in roi_polygons:
    inside = path.contains_points(spatial_coords)
    roi_labels[inside] = name

adata.obs["roi_label"] = pd.Categorical(
    roi_labels, categories=["DCIS_1", "DCIS_2", "Invasive", "Other"]
)

print("ROI label value counts:")
print(adata.obs["roi_label"].value_counts())

# Mask for cells inside an ROI (exclude "Other")
roi_mask = adata.obs["roi_label"] != "Other"
roi_obs_names = adata.obs_names[roi_mask]
roi_labels_masked = adata.obs.loc[roi_obs_names, "roi_label"].values
print(f"\nCells with ROI label (excluding Other): {roi_mask.sum()}")

# Spatial scatter colored by ROI label
fig, ax = plt.subplots(figsize=(14, 10))
plot_df_roi = adata.obsm["spatial"].copy()
plot_df_roi["roi_label"] = adata.obs["roi_label"].values
sns.scatterplot(
    data=plot_df_roi,
    x="X",
    y="Y",
    hue="roi_label",
    alpha=0.5,
    s=4,
    palette=ROI_PALETTE,
    ax=ax,
)
ax.set_xlabel("X (microns)")
ax.set_ylabel("Y (microns)")
ax.set_title("Cells colored by ROI label")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=2)
plt.tight_layout()
plt.savefig("figures/xenium_roi_labels_spatial.png", dpi=300, bbox_inches="tight")
plt.show()

# %% Load model, compute attention patterns, and load cached hub results
model = AMICI.load(model_path, adata=adata)
AMICI.setup_anndata(
    adata,
    labels_key=labels_key,
    coord_obsm_key="spatial",
    n_neighbors=50,
)
attention_patterns = model.get_attention_patterns()

# Grid search parameters
quantile_thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
n_clusters_range = [4, 6, 8, 10, 12, 14, 16]

# Load cached hub results
grid_search_cache_dir = "data/xenium_sample1/grid_search_cache"
hub_results_cache_dir = os.path.join(grid_search_cache_dir, "hub_results")
hub_results_dict = {}

print("Loading cached hub results...")
missing = []
for q_threshold in quantile_thresholds:
    for n_clusters in n_clusters_range:
        hub_cache_path = os.path.join(
            hub_results_cache_dir, f"hub_q{q_threshold}_k{n_clusters}.csv"
        )
        if os.path.exists(hub_cache_path):
            hub_results_dict[(q_threshold, n_clusters)] = pd.read_csv(
                hub_cache_path, index_col=0
            )
        else:
            missing.append((q_threshold, n_clusters))

if missing:
    print(f"Computing {len(missing)} missing hub results...")
    for q_threshold, n_clusters in tqdm(missing, desc="Computing hubs"):
        hub_result = attention_patterns.compute_communication_hubs(
            attention_quantile_threshold=q_threshold,
            n_clusters=n_clusters,
            random_state=seed,
        )
        hub_results_dict[(q_threshold, n_clusters)] = hub_result
        os.makedirs(hub_results_cache_dir, exist_ok=True)
        hub_result.to_csv(
            os.path.join(hub_results_cache_dir, f"hub_q{q_threshold}_k{n_clusters}.csv")
        )
else:
    print(f"All {len(hub_results_dict)} hub results loaded from cache.")

# %% Compute AMI and ARI (hub clusters vs ROI labels, masked to ROI cells only)
hub_ami_cache_path = os.path.join(grid_search_cache_dir, "hub_grid_search_ami_roi.csv")
hub_ari_cache_path = os.path.join(grid_search_cache_dir, "hub_grid_search_ari_roi.csv")

if os.path.exists(hub_ami_cache_path) and os.path.exists(hub_ari_cache_path):
    print("Loading cached AMI/ARI results...")
    hub_ami_df = pd.read_csv(hub_ami_cache_path, index_col=0)
    hub_ari_df = pd.read_csv(hub_ari_cache_path, index_col=0)
else:
    ami_results = np.zeros((len(n_clusters_range), len(quantile_thresholds)))
    ari_results = np.zeros((len(n_clusters_range), len(quantile_thresholds)))

    print("Computing AMI and ARI for hub clusters vs ROI labels (ROI cells only)...")
    for i, n_clusters in enumerate(tqdm(n_clusters_range, desc="Grid search")):
        for j, q_threshold in enumerate(quantile_thresholds):
            hub_result = hub_results_dict[(q_threshold, n_clusters)]
            hub_labels_masked = hub_result.loc[
                roi_obs_names, "hub_cluster"
            ].values.astype(int)

            ami_results[i, j] = adjusted_mutual_info_score(
                roi_labels_masked, hub_labels_masked
            )
            ari_results[i, j] = adjusted_rand_score(
                roi_labels_masked, hub_labels_masked
            )

    hub_ami_df = pd.DataFrame(
        ami_results,
        index=[str(k) for k in n_clusters_range],
        columns=[str(q) for q in quantile_thresholds],
    )
    hub_ari_df = pd.DataFrame(
        ari_results,
        index=[str(k) for k in n_clusters_range],
        columns=[str(q) for q in quantile_thresholds],
    )

    hub_ami_df.to_csv(hub_ami_cache_path)
    hub_ari_df.to_csv(hub_ari_cache_path)
    print(f"AMI/ARI results saved to {grid_search_cache_dir}/")

# %% Visualization — Hub cluster AMI and ARI heatmaps
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, df, metric in zip(axes, [hub_ami_df, hub_ari_df], ["AMI", "ARI"]):
    sns.heatmap(
        df.astype(float).values,
        annot=True,
        fmt=".3f",
        xticklabels=[str(q) for q in quantile_thresholds],
        yticklabels=[str(k) for k in n_clusters_range],
        cmap="YlOrRd",
        ax=ax,
        vmin=0,
    )
    ax.set_title(f"{metric}: Hub Clusters vs ROI Labels")
    ax.set_xlabel("Quantile Threshold")
    ax.set_ylabel("Number of Clusters")

# Print best parameter combination (based on AMI)
ami_vals = hub_ami_df.astype(float).values
best_idx = np.unravel_index(np.argmax(ami_vals), ami_vals.shape)
print(
    f"Best AMI: {ami_vals[best_idx]:.4f} at n_clusters={n_clusters_range[best_idx[0]]}, "
    f"quantile={quantile_thresholds[best_idx[1]]}"
)
ari_vals = hub_ari_df.astype(float).values
best_ari_idx = np.unravel_index(np.argmax(ari_vals), ari_vals.shape)
print(
    f"Best ARI: {ari_vals[best_ari_idx]:.4f} at n_clusters={n_clusters_range[best_ari_idx[0]]}, "
    f"quantile={quantile_thresholds[best_ari_idx[1]]}"
)

plt.tight_layout()
plt.savefig("figures/xenium_hub_ami_ari_roi_heatmap.png", dpi=300, bbox_inches="tight")
plt.savefig("figures/xenium_hub_ami_ari_roi_heatmap.svg", dpi=300, bbox_inches="tight")
plt.show()

# %% Visualize best hub clusters spatially
best_hub_k = n_clusters_range[best_idx[0]]
best_hub_q = quantile_thresholds[best_idx[1]]
print(f"Visualizing hub clusters for best params: k={best_hub_k}, q={best_hub_q}")

best_hub_result = hub_results_dict[(best_hub_q, best_hub_k)]
adata.obs["hub_cluster"] = best_hub_result.loc[
    adata.obs_names, "hub_cluster"
].astype(int)

cluster_palette = {
    0: "#2E5BBA",
    1: "#FF8C42",
    2: "#228B22",
    3: "#8E44AD",
    4: "#B22222",
    5: "#FFD700",
    6: "#5D6D7E",
    7: "#16A085",
    8: "#FF6347",
    9: "#8B4513",
    10: "#FF1493",
    11: "#32CD32",
    12: "#00CED1",
    13: "#DC143C",
    14: "#6B8E23",
    15: "#4169E1",
}

fig, axes = plt.subplots(1, 2, figsize=(28, 10))

# Left: hub clusters
plot_df_hub = adata.obsm["spatial"].copy()
plot_df_hub["hub_cluster"] = adata.obs["hub_cluster"].values
sns.scatterplot(
    data=plot_df_hub,
    x="X",
    y="Y",
    hue="hub_cluster",
    alpha=0.5,
    s=4,
    palette=cluster_palette,
    ax=axes[0],
)
for name, _, micron_coords in roi_polygons:
    color = ROI_PALETTE.get(name, "#000000")
    polygon = plt.Polygon(
        micron_coords,
        fill=False,
        edgecolor=color,
        linewidth=2,
        linestyle="--",
    )
    axes[0].add_patch(polygon)
axes[0].set_xlabel("X (microns)")
axes[0].set_ylabel("Y (microns)")
axes[0].set_title(f"Hub Clusters (k={best_hub_k}, q={best_hub_q})")
axes[0].legend(
    bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=2, title="Cluster"
)

# Right: ROI labels for comparison
plot_df_roi2 = adata.obsm["spatial"].copy()
plot_df_roi2["roi_label"] = adata.obs["roi_label"].values
sns.scatterplot(
    data=plot_df_roi2,
    x="X",
    y="Y",
    hue="roi_label",
    alpha=0.5,
    s=4,
    palette=ROI_PALETTE,
    ax=axes[1],
)
for name, _, micron_coords in roi_polygons:
    color = ROI_PALETTE.get(name, "#000000")
    polygon = plt.Polygon(
        micron_coords,
        fill=False,
        edgecolor=color,
        linewidth=2,
        linestyle="--",
    )
    axes[1].add_patch(polygon)
axes[1].set_xlabel("X (microns)")
axes[1].set_ylabel("Y (microns)")
axes[1].set_title("ROI Labels (ground truth)")
axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=2)

plt.tight_layout()
plt.savefig(
    "figures/xenium_hub_clusters_spatial.png", dpi=300, bbox_inches="tight"
)
plt.savefig(
    "figures/xenium_hub_clusters_spatial.svg", dpi=300, bbox_inches="tight"
)
plt.show()


# %% Composition cluster grid search (raw neighbor composition — no attention filtering)
composition_results_cache_dir = os.path.join(
    grid_search_cache_dir, "composition_results"
)
os.makedirs(composition_results_cache_dir, exist_ok=True)

comp_ami_cache_path = os.path.join(
    grid_search_cache_dir, "composition_grid_search_ami_roi.csv"
)
comp_ari_cache_path = os.path.join(
    grid_search_cache_dir, "composition_grid_search_ari_roi.csv"
)

if os.path.exists(comp_ami_cache_path) and os.path.exists(comp_ari_cache_path):
    print("Loading cached composition AMI/ARI results...")
    comp_ami_df = pd.read_csv(comp_ami_cache_path, index_col=0)
    comp_ari_df = pd.read_csv(comp_ari_cache_path, index_col=0)
else:
    # Compute raw neighbor composition vectors
    print("Computing raw neighbor composition vectors...")
    nn_labels = attention_patterns._nn_idxs_df.map(lambda x: adata.obs[labels_key][x])

    nn_labels_melted = nn_labels.reset_index().melt(
        id_vars="index",
        var_name="neighbor_position",
        value_name="cell_type",
    )

    neighbor_composition = (
        nn_labels_melted.groupby(["index", "cell_type"]).size().unstack(fill_value=0)
    )

    all_cell_types = adata.obs[labels_key].unique()
    missing_cell_types = set(all_cell_types) - set(neighbor_composition.columns)
    for cell_type in missing_cell_types:
        neighbor_composition[cell_type] = 0
    neighbor_composition = neighbor_composition[all_cell_types]

    neighbor_composition_norm = neighbor_composition.div(
        neighbor_composition.sum(axis=1), axis=0
    ).fillna(0)

    # Grid search over n_clusters
    comp_ami_results = []
    comp_ari_results = []

    print("Running composition cluster grid search...")
    for n_clusters in tqdm(n_clusters_range, desc="Composition clusters"):
        comp_cache_path = os.path.join(
            composition_results_cache_dir, f"composition_k{n_clusters}.csv"
        )

        if os.path.exists(comp_cache_path):
            comp_result = pd.read_csv(comp_cache_path, index_col=0)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
            kmeans_labels = kmeans.fit_predict(neighbor_composition_norm.values)
            comp_result = pd.DataFrame(
                {"composition_cluster": kmeans_labels},
                index=neighbor_composition_norm.index,
            )
            comp_result.to_csv(comp_cache_path)

        comp_labels_masked = comp_result.loc[
            roi_obs_names, "composition_cluster"
        ].values.astype(int)

        comp_ami_results.append(
            adjusted_mutual_info_score(roi_labels_masked, comp_labels_masked)
        )
        comp_ari_results.append(
            adjusted_rand_score(roi_labels_masked, comp_labels_masked)
        )

    comp_ami_df = pd.DataFrame(
        {"n_clusters": n_clusters_range, "AMI": comp_ami_results}
    ).set_index("n_clusters")
    comp_ari_df = pd.DataFrame(
        {"n_clusters": n_clusters_range, "ARI": comp_ari_results}
    ).set_index("n_clusters")

    comp_ami_df.to_csv(comp_ami_cache_path)
    comp_ari_df.to_csv(comp_ari_cache_path)
    print(f"Composition results saved to {grid_search_cache_dir}/")

# %% Visualization — Composition cluster AMI and ARI bar charts
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, df, metric, color in zip(
    axes,
    [comp_ami_df, comp_ari_df],
    ["AMI", "ARI"],
    ["#E69F00", "#56B4E9"],
):
    col = df.columns[0]
    ax.bar(
        [str(k) for k in df.index],
        df[col].values,
        color=color,
        edgecolor="black",
    )
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric}: Composition Clusters vs ROI Labels")
    ax.set_ylim(0, max(df[col].max() * 1.15, 0.05))

print(
    f"Best composition AMI: {comp_ami_df['AMI'].max():.4f} at k={comp_ami_df['AMI'].idxmax()}"
)
print(
    f"Best composition ARI: {comp_ari_df['ARI'].max():.4f} at k={comp_ari_df['ARI'].idxmax()}"
)

plt.tight_layout()
plt.savefig("figures/xenium_composition_ami_ari_roi.png", dpi=300, bbox_inches="tight")
plt.savefig("figures/xenium_composition_ami_ari_roi.svg", dpi=300, bbox_inches="tight")
plt.show()

# %% Visualize composition clusters spatially (best k by AMI)
best_comp_k = int(comp_ami_df["AMI"].idxmax())
print(f"Visualizing composition clusters for best k={best_comp_k}")

best_comp_cache_path = os.path.join(
    composition_results_cache_dir, f"composition_k{best_comp_k}.csv"
)
best_comp_result = pd.read_csv(best_comp_cache_path, index_col=0)
adata.obs["composition_cluster"] = best_comp_result.loc[
    adata.obs_names, "composition_cluster"
].astype(int)

fig, axes = plt.subplots(1, 2, figsize=(28, 10))

# Left: composition clusters
plot_df_comp = adata.obsm["spatial"].copy()
plot_df_comp["composition_cluster"] = adata.obs["composition_cluster"].values
sns.scatterplot(
    data=plot_df_comp,
    x="X",
    y="Y",
    hue="composition_cluster",
    alpha=0.5,
    s=4,
    palette=cluster_palette,
    ax=axes[0],
)
for name, _, micron_coords in roi_polygons:
    color = ROI_PALETTE.get(name, "#000000")
    polygon = plt.Polygon(
        micron_coords,
        fill=False,
        edgecolor=color,
        linewidth=2,
        linestyle="--",
    )
    axes[0].add_patch(polygon)
axes[0].set_xlabel("X (microns)")
axes[0].set_ylabel("Y (microns)")
axes[0].set_title(f"Composition Clusters (k={best_comp_k})")
axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=2, title="Cluster")

# Right: ROI labels for comparison
plot_df_roi2 = adata.obsm["spatial"].copy()
plot_df_roi2["roi_label"] = adata.obs["roi_label"].values
sns.scatterplot(
    data=plot_df_roi2,
    x="X",
    y="Y",
    hue="roi_label",
    alpha=0.5,
    s=4,
    palette=ROI_PALETTE,
    ax=axes[1],
)
for name, _, micron_coords in roi_polygons:
    color = ROI_PALETTE.get(name, "#000000")
    polygon = plt.Polygon(
        micron_coords,
        fill=False,
        edgecolor=color,
        linewidth=2,
        linestyle="--",
    )
    axes[1].add_patch(polygon)
axes[1].set_xlabel("X (microns)")
axes[1].set_ylabel("Y (microns)")
axes[1].set_title("ROI Labels (ground truth)")
axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=2)

plt.tight_layout()
plt.savefig(
    "figures/xenium_composition_clusters_spatial.png", dpi=300, bbox_inches="tight"
)
plt.savefig(
    "figures/xenium_composition_clusters_spatial.svg", dpi=300, bbox_inches="tight"
)
plt.show()

# %% Compare best AMI and ARI: interaction hubs vs composition clusters
best_hub_ami = hub_ami_df.astype(float).values.max()
best_hub_ari = hub_ari_df.astype(float).values.max()
best_comp_ami = comp_ami_df["AMI"].max()
best_comp_ari = comp_ari_df["ARI"].max()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, hub_val, comp_val, metric in zip(
    axes,
    [best_hub_ami, best_hub_ari],
    [best_comp_ami, best_comp_ari],
    ["AMI", "ARI"],
):
    bars = ax.bar(
        ["Interaction Hubs", "Composition"],
        [hub_val, comp_val],
        color=["#56B4E9", "#E69F00"],
        edgecolor="black",
        width=0.5,
    )
    for bar, val in zip(bars, [hub_val, comp_val]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_ylabel(metric)
    ax.set_title(f"Best {metric} vs ROI Labels")
    ax.set_ylim(0, max(hub_val, comp_val) * 1.25)

axes[0].text(
    0, -0.08, f"k={best_hub_k}, q={best_hub_q}",
    ha="center", va="top", fontsize=9, color="gray",
    transform=axes[0].get_xaxis_transform(),
)
axes[0].text(
    1, -0.08, f"k={best_comp_k}",
    ha="center", va="top", fontsize=9, color="gray",
    transform=axes[0].get_xaxis_transform(),
)

# Best ARI params may differ from best AMI params
best_hub_ari_idx = np.unravel_index(
    np.argmax(hub_ari_df.astype(float).values), hub_ari_df.shape
)
best_hub_ari_k = n_clusters_range[best_hub_ari_idx[0]]
best_hub_ari_q = quantile_thresholds[best_hub_ari_idx[1]]
best_comp_ari_k = int(comp_ari_df["ARI"].idxmax())
axes[1].text(
    0, -0.08, f"k={best_hub_ari_k}, q={best_hub_ari_q}",
    ha="center", va="top", fontsize=9, color="gray",
    transform=axes[1].get_xaxis_transform(),
)
axes[1].text(
    1, -0.08, f"k={best_comp_ari_k}",
    ha="center", va="top", fontsize=9, color="gray",
    transform=axes[1].get_xaxis_transform(),
)

plt.tight_layout()
plt.savefig(
    "figures/xenium_ami_ari_comparison_hub_vs_composition.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    "figures/xenium_ami_ari_comparison_hub_vs_composition.svg",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

print(
    f"Interaction Hubs — Best AMI: {best_hub_ami:.4f} (k={best_hub_k}, q={best_hub_q}), "
    f"Best ARI: {best_hub_ari:.4f} (k={best_hub_ari_k}, q={best_hub_ari_q})\n"
    f"Composition      — Best AMI: {best_comp_ami:.4f} (k={best_comp_k}), "
    f"Best ARI: {best_comp_ari:.4f} (k={best_comp_ari_k})"
)

# %%
