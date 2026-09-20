"""Empirical distance-binned expression profiles for the MERFISH cortex dataset.

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
DATA_PATH = os.path.join(SCRIPT_DIR, "data/cortex_processed_2025-04-28.h5ad")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "figures/empirical_distance_expression")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS_KEY = "subclass"
SECTION_KEY = "slice_id"
# This dataset has no cell_radius column, so radii are derived from the segmentation
# polygons of the middle z-plane (z3 is the only plane with no missing boundaries).
BOUNDARY_Z_PLANE = "z3"

# Wider bins than the Xenium/Atera scripts because this tissue is sampled ~1.7x more
# sparsely (median nearest-neighbour distance 14.7 um vs 8.2-8.6 um).
BIN_WIDTH = 2.0
MAX_DISTANCE = 50.0
MIN_CELLS_PER_BIN = 10

INTERACTIONS = [
    {"sender": "L2/3 IT", "receiver": "Astro", "genes": ["Cux2", "Lama3", "Lamp5"]},
    {
        "sender": "Oligo",
        "receiver": "Astro",
        "genes": ["Igfbp5", "Gfap", "Bgn", "Ccdc3", "Sox10", "Adamts4", "Serpinf1"],
    },
    {"sender": "L2/3 IT", "receiver": "Sst", "genes": ["Cux2", "Col25a1", "Pcdh8", "Lamp5"]},
    {
        "sender": "L4/5 IT",
        "receiver": "Sst",
        "genes": ["Npnt", "Olfm3", "Syndig1", "Hpse", "Tox", "Unc13c", "Rerg"],
    },
]


def get_coords(adata):
    """Spatial coordinates; this dataset stores centroids in obs rather than obsm."""
    return adata.obs[["centroid_x", "centroid_y"]].to_numpy(dtype=float)


def get_cell_radii(adata):
    """Effective radius per cell from its segmentation polygon, as sqrt(area / pi).

    Matches the cell_radius definition used in the Xenium and Atera preprocessing so
    that the radius-corrected distances are comparable across datasets.
    """
    xs = adata.obs[f"boundaryX_{BOUNDARY_Z_PLANE}"].to_numpy()
    ys = adata.obs[f"boundaryY_{BOUNDARY_Z_PLANE}"].to_numpy()
    areas = np.full(len(xs), np.nan)
    for i in range(len(xs)):
        if not isinstance(xs[i], str) or not isinstance(ys[i], str):
            continue
        x = np.fromstring(xs[i], sep=",")
        y = np.fromstring(ys[i], sep=",")
        n = min(len(x), len(y))
        if n < 3:
            continue
        x, y = x[:n], y[:n]
        areas[i] = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    radii = np.sqrt(areas / np.pi)
    return np.nan_to_num(radii, nan=float(np.nanmedian(radii)))


def build_profile(adata, sender, receiver, genes):
    """Per-receiver distance to nearest sender, paired with receiver expression.

    Distances are computed within each tissue section and pooled afterwards, since
    coordinates are not comparable across sections.
    """
    labels = adata.obs[LABELS_KEY].astype(str).to_numpy()
    sections = adata.obs[SECTION_KEY].astype(str).to_numpy()
    radii = get_cell_radii(adata)
    coords = get_coords(adata)
    frames = []
    for section in np.unique(sections):
        in_section = sections == section
        sender_mask = in_section & (labels == sender)
        receiver_mask = in_section & (labels == receiver)
        if sender_mask.sum() == 0 or receiver_mask.sum() == 0:
            continue
        distance, nn_index = cKDTree(coords[sender_mask]).query(coords[receiver_mask], k=1)
        distance = np.clip(distance - radii[receiver_mask] - radii[sender_mask][nn_index], 0, None)
        expression = adata[receiver_mask, genes].X
        if sparse.issparse(expression):
            expression = expression.toarray()
        frame = pd.DataFrame(np.asarray(expression, dtype=float), columns=genes)
        frame["distance"] = distance
        frame["section"] = section
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


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
        "MERFISH cortex: receiver expression vs surface-to-surface distance to nearest sender", y=1.0, fontsize=13
    )
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(
            os.path.join(OUTPUT_DIR, f"cortex_empirical_distance_expression{suffix}.{ext}"),
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
        os.path.join(OUTPUT_DIR, "cortex_distance_binned_expression.csv"), index=False
    )
    monotonicity = pd.concat(all_stats, ignore_index=True)
    monotonicity.to_csv(os.path.join(OUTPUT_DIR, "cortex_distance_monotonicity.csv"), index=False)

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
