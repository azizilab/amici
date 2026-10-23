"""Empirical distance-binned expression profiles for the CosMx human tonsil dataset.

Model-free check of AMICI's monotonic distance assumption: receiver cells are binned by
distance to their nearest sender and mean expression of the AMICI-implicated genes is
plotted per bin. Uses only cell positions, cell-type labels, and measured expression.

The interactions and downstream genes are the ones shown in the tonsil downstream-impact
dot plot produced by human_tonsil_analysis.py.
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

DATA_PATH = os.path.join(SCRIPT_DIR, "data/human_tonsil_filtered_2026-07-25.h5ad")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "figures/empirical_distance_expression")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS_KEY = "celltype_manual_fine"
# obsm["spatial"] holds CosMx global pixel coordinates; obs["cell_radius"] is already in
# micrometres (sqrt(Area.um2 / pi), set in human_tonsil_preprocess.py). Coordinates are
# converted here so that distances, radii and the x axis are all micrometres, giving the
# surface-to-surface distances the Xenium, Atera and cortex versions of this figure use.
CELL_RADIUS_KEY = "cell_radius"
COSMX_PIXEL_SIZE_UM = 0.12028
# Single slide and single core, so all cells share one coordinate frame and no
# per-section splitting is needed.

BIN_WIDTH = 1.0
MAX_DISTANCE = 50.0
MIN_CELLS_PER_BIN = 10
# Gaussian kernel SD for the local-linear smoother drawn over the bin means, in um.
SMOOTHING_BANDWIDTH = 5.0

GC_B_GENES = [
    "CLU",
    "MARCKSL1",
    "CD22",
    "MS4A1",
    "RGS13",
    "BASP1",
    "SRGN",
    "MALAT1",
    "CXCL13",
    "RPL32",
    "VCAM1",
    "PTGDS",
]
CD4_T_GENES = ["CCL21", "CCL19", "VIM", "TPT1", "RPL32", "RPL34", "DCN", "PTGDS", "CXCL14", "CXCL12"]

INTERACTIONS = [
    {"sender": "Follicular dendritic cell", "receiver": "GC B cell (resting)", "genes": GC_B_GENES},
    {"sender": "Tfh cell", "receiver": "GC B cell (resting)", "genes": GC_B_GENES},
    {"sender": "Fibroblast", "receiver": "CD4 T cell", "genes": CD4_T_GENES},
]


def get_coords(adata):
    """Spatial coordinates converted from CosMx global pixels to µm."""
    return np.asarray(adata.obsm["spatial"])[:, :2].astype(float) * COSMX_PIXEL_SIZE_UM


def build_profile(adata, sender, receiver, genes):
    """Per-receiver distance to nearest sender, paired with receiver expression."""
    labels = adata.obs[LABELS_KEY].astype(str).to_numpy()
    radii = adata.obs[CELL_RADIUS_KEY].to_numpy(dtype=float)
    coords = get_coords(adata)
    sender_mask = labels == sender
    receiver_mask = labels == receiver
    if sender_mask.sum() == 0 or receiver_mask.sum() == 0:
        raise ValueError(f"No cells for {sender} -> {receiver} under {LABELS_KEY}")

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
    kept for continuity with the other datasets' versions of this figure, but it is
    driven by sample size: with tens of thousands of receiver cells a rho of -0.01 is
    "significant" while describing a flat profile. The monotone-decay statistics are
    computed on the binned profile instead, and calibrate the isotonic fit against a null
    in which the bin means are pure sampling noise, so they separate a real decay from
    bin-to-bin jitter.
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
        os.path.join(OUTPUT_DIR, "human_tonsil_distance_binned_expression.csv"), index=False
    )
    # Multiple testing is controlled across every gene x interaction in this dataset.
    monotonicity = add_monotone_calls(pd.concat(all_stats, ignore_index=True))
    monotonicity.to_csv(os.path.join(OUTPUT_DIR, "human_tonsil_distance_monotonicity.csv"), index=False)

    plot_kwargs = dict(
        output_dir=OUTPUT_DIR,
        output_prefix="human_tonsil_empirical_distance_expression",
        suptitle="CosMx human tonsil: receiver expression vs surface-to-surface distance to nearest sender",
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
        output_prefix="human_tonsil_distance_monotonicity_summary",
        suptitle="CosMx human tonsil: monotone decay with distance to nearest sender",
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
