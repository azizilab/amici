"""Empirical distance-binned expression profiles for the Atera breast cancer dataset.

Model-free check of AMICI's monotonic distance assumption: receiver cells are binned by
distance to their nearest sender and mean expression of the AMICI-implicated genes is
plotted per bin. Uses only cell positions, cell-type labels, and measured expression.
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data/atera_breast_filtered_2026-07-22.h5ad")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "figures/empirical_distance_expression")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS_KEY = "celltype_train_grouped"
# AMICI._compute_nn subtracts both cell radii from the centroid distance, so distances
# here are surface-to-surface to match what the model sees.
CELL_RADIUS_KEY = "cell_radius"

BIN_WIDTH = 1.0
MAX_DISTANCE = 50.0
MIN_CELLS_PER_BIN = 10

CAF_GENES = ["C3", "COL6A2", "MMP2", "CXCL12", "ABI3BP", "EMILIN1", "COL15A1"]
LUMINAL_DCIS_GENES = ["MUC1", "BMPR1B", "ITGB5", "TFF3", "GASK1B"]

INTERACTIONS = [
    {"sender": "Pericytes", "receiver": "Endothelial", "genes": ["PECAM1", "ADAM15", "SELP", "PRCP"]},
    {"sender": "Macrophages", "receiver": "CAFs", "genes": CAF_GENES},
    {"sender": "T_Cells", "receiver": "CAFs", "genes": CAF_GENES},
    {"sender": "CAFs", "receiver": "Luminal_DCIS", "genes": LUMINAL_DCIS_GENES},
    {"sender": "Myoepithelial", "receiver": "Luminal_DCIS", "genes": LUMINAL_DCIS_GENES},
]


def get_coords(adata):
    """Spatial coordinates in µm.

    Verified as micrometres rather than pixels: median nearest-neighbour distance is
    8.2 µm and median cell_area is 65 µm², matching the Xenium sample.
    """
    return np.asarray(adata.obsm["spatial"])[:, :2].astype(float)


def build_profile(adata, sender, receiver, genes):
    """Per-receiver distance to nearest sender, paired with receiver expression."""
    labels = adata.obs[LABELS_KEY].astype(str).to_numpy()
    radii = adata.obs[CELL_RADIUS_KEY].to_numpy(dtype=float)
    coords = get_coords(adata)
    sender_mask = labels == sender
    receiver_mask = labels == receiver
    if sender_mask.sum() == 0 or receiver_mask.sum() == 0:
        return pd.DataFrame(columns=[*genes, "distance"])
    distance, nn_index = cKDTree(coords[sender_mask]).query(coords[receiver_mask], k=1)
    distance = np.clip(distance - radii[receiver_mask] - radii[sender_mask][nn_index], 0, None)
    expression = adata[receiver_mask, genes].X
    if sparse.issparse(expression):
        expression = expression.toarray()
    frame = pd.DataFrame(np.asarray(expression, dtype=float), columns=genes)
    frame["distance"] = distance
    return frame


def summarize_profile(profile, genes):
    """Per-bin mean expression with SEM, plus a Spearman monotonicity test per gene."""
    profile = profile[profile["distance"] <= MAX_DISTANCE].copy()
    bins = np.arange(0, MAX_DISTANCE + BIN_WIDTH, BIN_WIDTH)
    profile["bin"] = pd.cut(profile["distance"], bins=bins, right=False)

    records = []
    stats = []
    for gene in genes:
        values = profile[gene].to_numpy(dtype=float)
        sd = values.std()
        z_values = (values - values.mean()) / sd if sd > 0 else np.zeros_like(values)
        rho, pvalue = spearmanr(profile["distance"].to_numpy(dtype=float), values)
        stats.append({"gene": gene, "spearman_rho": rho, "spearman_pvalue": pvalue, "n_cells": len(values)})

        binned = pd.DataFrame({"bin": profile["bin"].to_numpy(), "raw": values, "z": z_values})
        for interval, group in binned.groupby("bin", observed=True):
            if len(group) < MIN_CELLS_PER_BIN:
                continue
            records.append(
                {
                    "gene": gene,
                    "bin_center": (interval.left + interval.right) / 2,
                    "n_cells": len(group),
                    "mean_expression": group["raw"].mean(),
                    "sem_expression": group["raw"].std(ddof=1) / np.sqrt(len(group)),
                    "mean_z": group["z"].mean(),
                    "sem_z": group["z"].std(ddof=1) / np.sqrt(len(group)),
                }
            )
    return pd.DataFrame(records), pd.DataFrame(stats)


def plot_profiles(
    summaries,
    value_col="mean_expression",
    sem_col="sem_expression",
    ylabel="Mean log1p-normalized expression",
    suffix="",
    zero_line=False,
):
    """One panel per interaction, one line per gene, shaded 95% CI."""
    n_panels = len(summaries)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0 * n_cols, 4.6 * n_rows), squeeze=False)

    for panel_idx, (label, summary) in enumerate(summaries):
        ax = axes[panel_idx // n_cols][panel_idx % n_cols]
        genes = list(summary["gene"].drop_duplicates())
        colors = plt.get_cmap("tab10" if len(genes) <= 10 else "tab20")(np.linspace(0, 1, len(genes), endpoint=False))
        for gene, color in zip(genes, colors, strict=False):
            gene_df = summary[summary["gene"] == gene].sort_values("bin_center")
            ax.plot(gene_df["bin_center"], gene_df[value_col], marker="o", ms=3.5, lw=1.6, color=color, label=gene)
            ax.fill_between(
                gene_df["bin_center"],
                gene_df[value_col] - 1.96 * gene_df[sem_col],
                gene_df[value_col] + 1.96 * gene_df[sem_col],
                color=color,
                alpha=0.18,
                linewidth=0,
            )
        if zero_line:
            ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Surface-to-surface distance to nearest sender (µm)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax.grid(alpha=0.25)

    for empty_idx in range(n_panels, n_rows * n_cols):
        axes[empty_idx // n_cols][empty_idx % n_cols].axis("off")

    fig.suptitle(
        "Atera breast cancer: receiver expression vs surface-to-surface distance to nearest sender", y=1.0, fontsize=13
    )
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(
            os.path.join(OUTPUT_DIR, f"atera_breast_empirical_distance_expression{suffix}.{ext}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main():
    """Bin receiver cells by distance to the nearest sender and plot mean expression of the AMICI-implicated genes."""
    adata = sc.read_h5ad(DATA_PATH)
    adata.obs_names_make_unique()

    summaries = []
    all_bins = []
    all_stats = []
    for interaction in INTERACTIONS:
        sender, receiver, genes = interaction["sender"], interaction["receiver"], interaction["genes"]
        label = f"{sender} → {receiver}"
        profile = build_profile(adata, sender, receiver, genes)
        summary, stats = summarize_profile(profile, genes)
        summaries.append((label, summary))

        summary.insert(0, "interaction", label)
        stats.insert(0, "interaction", label)
        all_bins.append(summary)
        all_stats.append(stats)
        print(f"{label}: {len(profile)} receiver cells, {len(genes)} genes", flush=True)

    pd.concat(all_bins, ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, "atera_breast_distance_binned_expression.csv"), index=False
    )
    monotonicity = pd.concat(all_stats, ignore_index=True)
    monotonicity.to_csv(os.path.join(OUTPUT_DIR, "atera_breast_distance_monotonicity.csv"), index=False)

    plot_profiles(summaries)
    plot_profiles(
        summaries,
        value_col="mean_z",
        sem_col="sem_z",
        ylabel="Mean expression (z-scored log1p)",
        suffix="_zscore",
        zero_line=True,
    )
    print("\n=== Spearman correlation of expression with distance ===")
    print(monotonicity.to_string(index=False))
    print(f"\nWrote figures and tables to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
