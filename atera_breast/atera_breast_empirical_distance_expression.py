"""Empirical distance-binned expression profiles for the Atera breast cancer dataset.

Model-free check of AMICI's monotonic distance assumption: receiver cells are binned by
distance to their nearest sender and mean expression of the AMICI-implicated genes is
plotted per bin. Uses only cell positions, cell-type labels, and measured expression.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "../benchmarks"))

from distance_profile_utils import (  # noqa: E402
    add_monotone_calls,
    monotone_decay_stats,
    plot_monotonicity_summary,
    plot_profiles,
)

DATA_PATH = os.path.join(SCRIPT_DIR, "data/atera_breast_filtered_2026-07-22.h5ad")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "figures/empirical_distance_expression")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS_KEY = "celltype_train_grouped"
# Both cell radii are subtracted from the centroid distance, so distances here are
# surface-to-surface.
CELL_RADIUS_KEY = "cell_radius"

BIN_WIDTH = 1.0
MAX_DISTANCE = 50.0
MIN_CELLS_PER_BIN = 10
# Gaussian kernel SD for the local-linear smoother drawn over the bin means, in um.
SMOOTHING_BANDWIDTH = 5.0

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
    """Per-bin mean expression with SEM, plus per-gene monotone-decay statistics.

    Two kinds of statistic are returned per gene. The cell-level Spearman correlation is
    kept for continuity with earlier versions of this figure, but it is driven by sample
    size: with tens of thousands of receiver cells a rho of -0.01 is "significant" while
    describing a flat profile. The monotone-decay statistics are computed on the binned
    profile instead, and calibrate the isotonic fit against a null in which the bin means
    are pure sampling noise, so they separate a real decay from bin-to-bin jitter.
    """
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

        gene_records = []
        binned = pd.DataFrame({"bin": profile["bin"].to_numpy(), "raw": values, "z": z_values})
        for interval, group in binned.groupby("bin", observed=True):
            if len(group) < MIN_CELLS_PER_BIN:
                continue
            gene_records.append(
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
        records.extend(gene_records)

        gene_bins = pd.DataFrame(gene_records)
        gene_stats = {"gene": gene, "spearman_rho": rho, "spearman_pvalue": pvalue, "n_cells": len(values)}
        if len(gene_bins):
            gene_stats.update(
                monotone_decay_stats(
                    gene_bins["bin_center"].to_numpy(),
                    gene_bins["mean_z"].to_numpy(),
                    gene_bins["sem_z"].to_numpy(),
                    gene_bins["n_cells"].to_numpy(),
                    max_distance=MAX_DISTANCE,
                    bandwidth=SMOOTHING_BANDWIDTH,
                )
            )
        stats.append(gene_stats)
    return pd.DataFrame(records), pd.DataFrame(stats)


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
        summary.insert(0, "interaction", label)
        stats.insert(0, "interaction", label)
        summaries.append((label, summary))
        all_bins.append(summary)
        all_stats.append(stats)
        print(f"{label}: {len(profile)} receiver cells, {len(genes)} genes", flush=True)

    pd.concat(all_bins, ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, "atera_breast_distance_binned_expression.csv"), index=False
    )
    # Multiple testing is controlled across every gene x interaction in this dataset.
    monotonicity = add_monotone_calls(pd.concat(all_stats, ignore_index=True))
    monotonicity.to_csv(os.path.join(OUTPUT_DIR, "atera_breast_distance_monotonicity.csv"), index=False)

    plot_kwargs = dict(
        output_dir=OUTPUT_DIR,
        output_prefix="atera_breast_empirical_distance_expression",
        suptitle="Atera breast cancer: receiver expression vs surface-to-surface distance to nearest sender",
        bandwidth=SMOOTHING_BANDWIDTH,
        max_distance=MAX_DISTANCE,
        stats=monotonicity,
    )
    plot_profiles(summaries, **plot_kwargs)
    plot_profiles(
        summaries,
        value_col="mean_z",
        sem_col="sem_z",
        ylabel="Mean expression (z-scored log1p)",
        suffix="_zscore",
        zero_line=True,
        **plot_kwargs,
    )
    plot_monotonicity_summary(
        monotonicity,
        output_dir=OUTPUT_DIR,
        output_prefix="atera_breast_distance_monotonicity_summary",
        suptitle="Atera breast cancer: monotone decay with distance to nearest sender",
    )

    columns = [
        "interaction",
        "gene",
        "spearman_rho",
        "spearman_qvalue",
        "spearman_bins_rho",
        "amplitude_z",
        "mono_score",
        "mono_qvalue",
        "snr",
        "decay_length_um",
        "monotone_decay",
        "testable",
    ]
    testable = monotonicity[monotonicity["testable"]]
    n_called = int(testable["monotone_decay"].sum())
    print("\n=== Monotone decay of the binned profile (BH-adjusted across all genes) ===")
    print(monotonicity[columns].to_string(index=False))
    print(
        f"\n{n_called}/{len(testable)} testable gene-interaction pairs pass the monotonic decay test "
        f"(per-cell Spearman q < 0.05, rho < 0); "
        f"{int(testable['shallow'].sum())} of those have a total drop below 0.05 z-units."
    )
    if len(testable) < len(monotonicity):
        print(
            f"{len(monotonicity) - len(testable)} pairs could not be tested: too few cells "
            f"to fill the minimum number of distance bins."
        )
    print(f"\nWrote figures and tables to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
