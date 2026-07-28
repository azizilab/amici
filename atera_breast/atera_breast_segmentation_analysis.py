# %% Import necessary libraries
import os

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
from scipy import sparse, stats
from scipy.spatial.distance import cdist

# %% Load the anndata object
data_date = "2026-07-22"
adata = sc.read_h5ad(f"./data/atera_breast_filtered_{data_date}.h5ad")
adata_train = sc.read_h5ad(f"./data/atera_breast_filtered_train_{data_date}.h5ad")
adata_test = sc.read_h5ad(f"./data/atera_breast_filtered_test_{data_date}.h5ad")
labels_key = "celltype_train_grouped"

# %% Define the cell type pairs and receiver genes of interest
cell_type_pairs = [
    {
        "sender_type": "Myoepithelial",
        "receiver_type": "CAFs",
        "genes": ["COL15A1", "SPRY2", "COL1A1", "COL3A1", "COL6A1", "FN1"],
        "label": "Myoepithelial -> CAFs",
    },
    {
        "sender_type": "CAFs",
        "receiver_type": "Endothelial",
        "genes": ["COL15A1", "SPRY1", "LAMB1", "HSPG2", "COL4A1", "VWF", "GAS6", "PLXNB", "TIE1", "SPRY2"],
        "label": "CAFs -> Endothelial",
    },
    {
        "sender_type": "Pericytes",
        "receiver_type": "Endothelial",
        "genes": ["ENG", "PECAM1", "ENTPD1", "PLVAP", "SELP", "ACKR1", "STOM"],
        "label": "Pericytes -> Endothelial",
    },
]
near_threshold = 20.0
far_threshold = 100.0

# %% Analyze each cell type pair
all_gene_stats_data = []

for pair_idx, pair_config in enumerate(cell_type_pairs):
    sender_type = pair_config["sender_type"]
    receiver_type = pair_config["receiver_type"]
    genes = pair_config["genes"]
    pair_label = pair_config["label"]

    print(f"\n{'=' * 60}")
    print(f"Analyzing pair {pair_idx + 1}: {pair_label}")
    print(f"{'=' * 60}")

    sender_mask = adata.obs[labels_key] == sender_type
    receiver_mask = adata.obs[labels_key] == receiver_type

    sender_ids = adata.obs_names[sender_mask]
    receiver_ids = adata.obs_names[receiver_mask]

    if len(sender_ids) == 0:
        print(f"WARNING: No cells found for sender type: {sender_type}")
        continue
    if len(receiver_ids) == 0:
        print(f"WARNING: No cells found for receiver type: {receiver_type}")
        continue

    print(f"Found {len(sender_ids)} {sender_type} cells and {len(receiver_ids)} {receiver_type} cells")

    sender_coords = adata.obsm["spatial"][sender_mask]
    receiver_coords = adata.obsm["spatial"][receiver_mask]

    distances = cdist(receiver_coords, sender_coords)
    min_distances_to_senders = distances.min(axis=1)

    near_receiver_mask = min_distances_to_senders <= near_threshold
    near_receiver_ids = receiver_ids[near_receiver_mask]
    near_distances = min_distances_to_senders[near_receiver_mask]

    distances_senders_to_receivers = cdist(sender_coords, receiver_coords)
    min_distances_to_receivers = distances_senders_to_receivers.min(axis=1)
    far_sender_mask = min_distances_to_receivers >= far_threshold
    far_sender_ids = sender_ids[far_sender_mask]
    far_sender_distances = min_distances_to_receivers[far_sender_mask]

    print(f"Near receivers (<={near_threshold} units from any sender): {len(near_receiver_ids)}")
    print(f"Far senders (>={far_threshold} units from all receivers): {len(far_sender_ids)}")

    if len(near_receiver_ids) == 0 or len(far_sender_ids) == 0:
        print(f"WARNING: Insufficient cells for analysis in pair: {pair_label}")
        continue

    np.random.seed(42 + pair_idx)
    min_sample_size = min(len(near_receiver_ids), len(far_sender_ids))

    if len(near_receiver_ids) > min_sample_size:
        near_receiver_sample_idx = np.random.choice(len(near_receiver_ids), min_sample_size, replace=False)
        near_receiver_ids_sampled = near_receiver_ids[near_receiver_sample_idx]
        near_distances_sampled = near_distances[near_receiver_sample_idx]
    else:
        near_receiver_ids_sampled = near_receiver_ids
        near_distances_sampled = near_distances

    if len(far_sender_ids) > min_sample_size:
        far_sender_sample_idx = np.random.choice(len(far_sender_ids), min_sample_size, replace=False)
        far_sender_ids_sampled = far_sender_ids[far_sender_sample_idx]
        far_sender_distances_sampled = far_sender_distances[far_sender_sample_idx]
    else:
        far_sender_ids_sampled = far_sender_ids
        far_sender_distances_sampled = far_sender_distances

    print("\nSubsampled sizes for statistical comparison:")
    print(f"  Near receivers: {len(near_receiver_ids_sampled)}")
    print(f"  Far senders: {len(far_sender_ids_sampled)}")

    if isinstance(genes, str):
        genes = [genes]

    missing_genes = [g for g in genes if g not in adata.var_names]
    if missing_genes:
        print(f"Warning: Genes not found in data: {missing_genes}")
        genes = [g for g in genes if g in adata.var_names]

    if len(genes) == 0:
        print(f"WARNING: No valid genes found in the data for pair: {pair_label}")
        continue

    X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    near_indices = adata.obs_names.get_indexer(near_receiver_ids_sampled)
    far_sender_indices = adata.obs_names.get_indexer(far_sender_ids_sampled)
    gene_indices = [adata.var_names.get_loc(g) for g in genes]

    near_expr = X[near_indices][:, gene_indices]
    far_sender_expr = X[far_sender_indices][:, gene_indices]

    pair_gene_stats_data = []

    for i, gene in enumerate(genes):
        plt.figure(figsize=(20, 5))
        ax = plt.gca()

        near_values = near_expr[:, i] if len(near_expr) > 0 else []
        far_sender_values = far_sender_expr[:, i] if len(far_sender_expr) > 0 else []

        has_data = False

        if len(near_values) > 1:
            sns.kdeplot(
                near_values,
                ax=ax,
                color="blue",
                fill=True,
                alpha=0.3,
                label=f"Near receivers (<={near_threshold})",
                linewidth=2,
                common_norm=False,
            )
            near_mean = np.mean(near_values)
            ax.axvline(near_mean, color="blue", linestyle="--", alpha=0.8, label=f"Near mean: {near_mean:.2f}")
            has_data = True
        elif len(near_values) == 1:
            ax.axvline(near_values[0], color="blue", linestyle="-", alpha=0.8, label=f"Near (n=1): {near_values[0]:.2f}")
            has_data = True

        if len(far_sender_values) > 1:
            sns.kdeplot(
                far_sender_values,
                ax=ax,
                color="orange",
                fill=True,
                alpha=0.3,
                label=f"Far senders (>={far_threshold})",
                linewidth=2,
                common_norm=False,
            )
            far_mean = np.mean(far_sender_values)
            ax.axvline(far_mean, color="orange", linestyle="--", alpha=0.8, label=f"Far sender mean: {far_mean:.2f}")
            has_data = True
        elif len(far_sender_values) == 1:
            ax.axvline(
                far_sender_values[0],
                color="orange",
                linestyle="-",
                alpha=0.8,
                label=f"Far (n=1): {far_sender_values[0]:.2f}",
            )
            has_data = True

        if has_data:
            ax.set_xlabel("Expression")
            ax.set_ylabel("Density")
            ax.set_title(
                f"{gene} Expression Distribution ({pair_label})\n"
                f"{receiver_type} near {sender_type} vs far {sender_type}"
            )
            ax.legend()

            if len(near_values) > 1 and len(far_sender_values) > 1:
                u_statistic, mannwhitney_pval = stats.mannwhitneyu(
                    near_values,
                    far_sender_values,
                    alternative="greater",
                )
            else:
                u_statistic = np.nan
                mannwhitney_pval = np.nan

            pair_gene_stats_data.append(
                {
                    "gene": gene,
                    "pair_label": pair_label,
                    "sender_type": sender_type,
                    "receiver_type": receiver_type,
                    "mannwhitney_pval": mannwhitney_pval,
                    "mannwhitney_statistic": u_statistic,
                    "near_mean": np.mean(near_values) if len(near_values) > 0 else np.nan,
                    "far_sender_mean": np.mean(far_sender_values) if len(far_sender_values) > 0 else np.nan,
                    "near_distance_mean": np.mean(near_distances_sampled),
                    "far_sender_distance_mean": np.mean(far_sender_distances_sampled),
                }
            )
        else:
            ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{gene} - No data ({pair_label})")

        plt.tight_layout()
        plt.show()

    all_gene_stats_data.extend(pair_gene_stats_data)
    print(f"Completed analysis for {len(pair_gene_stats_data)} genes in {pair_label}")

print(f"\n{'=' * 60}")
print(f"Analysis complete. Total genes analyzed: {len(all_gene_stats_data)}")
print(f"{'=' * 60}")


# %% Create segmentation test plot using Mann-Whitney U test results
def plot_segmentation_test_results(
    gene_stats_data,
    pval_threshold=0.05,
    show=True,
    save_png=False,
    save_svg=False,
    save_dir="./figures",
):
    """Plot Mann-Whitney U p-values for directional segmentation checks."""
    import pandas as pd
    from matplotlib.patches import Patch

    df = pd.DataFrame(gene_stats_data)
    df = df.dropna(subset=["mannwhitney_pval"])

    if len(df) == 0:
        print("No genes with valid Mann-Whitney U test p-value data")
        return None

    df_grouped = []
    for pair_label in df["pair_label"].unique():
        pair_df = df[df["pair_label"] == pair_label].copy()
        pair_df = pair_df.sort_values("mannwhitney_pval", ascending=True)
        df_grouped.append(pair_df)

    df_sorted = pd.concat(df_grouped, ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, max(8, len(df_sorted) * 0.5)))
    y_positions = np.arange(len(df_sorted))
    colors = ["red" if p < pval_threshold else "lightblue" for p in df_sorted["mannwhitney_pval"]]

    ax.scatter(
        df_sorted["mannwhitney_pval"],
        y_positions,
        c=colors,
        s=120,
        marker="o",
        edgecolors="black",
        linewidth=0.8,
        alpha=0.8,
    )

    gene_labels = [f"{row['gene']} ({row['pair_label']})" for _, row in df_sorted.iterrows()]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(gene_labels, fontsize=9)

    ax.axvline(
        x=pval_threshold,
        color="red",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        label=f"Significance threshold (p = {pval_threshold})",
    )
    ax.set_xlabel("Mann-Whitney U Test P-value", fontsize=12)
    ax.set_ylabel("Genes by Cell Type Pair", fontsize=12)
    ax.set_title(
        "Atera Segmentation Test Results\nMann-Whitney U Test: Near Receivers > Far Senders",
        fontsize=14,
        pad=20,
    )
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis="x")

    legend_elements = [
        Patch(facecolor="red", alpha=0.8, label=f"Significant (p < {pval_threshold})"),
        Patch(facecolor="lightblue", alpha=0.8, label=f"Non-significant (p >= {pval_threshold})"),
        plt.Line2D([0], [0], color="red", linestyle="--", alpha=0.7, label=f"Significance threshold (p = {pval_threshold})"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    n_total = len(df_sorted)
    significant_genes = df_sorted[df_sorted["mannwhitney_pval"] < pval_threshold]
    n_significant = len(significant_genes)
    mean_pval = df_sorted["mannwhitney_pval"].mean()

    pair_stats = []
    for pair_label in df_sorted["pair_label"].unique():
        pair_data = df_sorted[df_sorted["pair_label"] == pair_label]
        n_pair = len(pair_data)
        n_sig_pair = len(pair_data[pair_data["mannwhitney_pval"] < pval_threshold])
        pair_stats.append(f"{pair_label}: {n_sig_pair}/{n_pair}")

    stats_text = f"""Total genes: {n_total}
Significant: {n_significant} ({n_significant / n_total * 100:.1f}%)
Mean p-value: {mean_pval:.3f}

Per pair:
{chr(10).join(pair_stats)}"""

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if save_png:
        plt.savefig(f"{save_dir}/atera_segmentation_test_results.png", dpi=300, bbox_inches="tight")
    if save_svg:
        plt.savefig(f"{save_dir}/atera_segmentation_test_results.svg", bbox_inches="tight")

    if show:
        plt.show()

    return fig


# %% Generate the segmentation test results plot
if len(all_gene_stats_data) > 0:
    os.makedirs("./figures/segmentation_analysis", exist_ok=True)
    plot_segmentation_test_results(
        gene_stats_data=all_gene_stats_data,
        pval_threshold=0.05,
        show=True,
        save_png=True,
        save_svg=True,
        save_dir="./figures/segmentation_analysis",
    )
else:
    print("No gene statistics data collected for segmentation test plot")

# %%
