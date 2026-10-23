"""Two-panel summary figure for the monotonic decay analysis.

Panel (a) shows the distance profiles of one sender-receiver pair with their smoothed
trends. Panel (b) covers every sender-receiver pair in all four datasets at once, one row
per pair and one dot per gene at its bin-level Spearman correlation with distance, so the
reader can see how general the monotone behaviour is without a figure per dataset.

Run the four ``*_empirical_distance_expression.py`` scripts first; this script only reads
the CSVs they write.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from distance_profile_utils import (  # noqa: E402
    MIN_BINS_FOR_STATS,
    smooth_summary,
)

REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "figures", "monotonicity")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset display name -> (results directory, file prefix, smoother bandwidth in um).
DATASETS = [
    ("MERFISH cortex", "cortex/figures/empirical_distance_expression", "cortex", 6.0),
    ("Xenium breast cancer", "xenium/figures/empirical_distance_expression", "xenium", 5.0),
    ("CosMx human tonsil", "human_tonsil/figures/empirical_distance_expression", "human_tonsil", 5.0),
    ("Atera breast cancer", "atera_breast/figures/empirical_distance_expression", "atera_breast", 5.0),
]

# The pair shown in full in panels (a) and (b). This one carries both clear monotone
# decays and three genes the test declines, so the example shows the call working in
# both directions rather than only where the assumption holds.
EXAMPLE_DATASET = "CosMx human tonsil"
EXAMPLE_INTERACTION = "Fibroblast → CD4 T cell"

MAX_DISTANCE = 50.0
CALLED_COLOR = "#2c6fb5"
UNCALLED_COLOR = "#b8b8b8"


def load_dataset(results_dir, prefix):
    """Binned profiles and per-gene statistics for one dataset."""
    base = os.path.join(REPO_ROOT, results_dir)
    bins = pd.read_csv(os.path.join(base, f"{prefix}_distance_binned_expression.csv"))
    stats = pd.read_csv(os.path.join(base, f"{prefix}_distance_monotonicity.csv"))
    return bins, stats


def plot_example_profile(ax, bins, stats, interaction, bandwidth):
    """Distance profile for one interaction: faint bin means under a smoothed trend."""
    sub = bins[bins["interaction"] == interaction]
    gene_stats = stats[stats["interaction"] == interaction].set_index("gene")
    genes = list(sub["gene"].drop_duplicates())
    colors = plt.get_cmap("tab10" if len(genes) <= 10 else "tab20")(np.linspace(0, 1, len(genes), endpoint=False))
    smoothed = smooth_summary(sub, "mean_z", "sem_z", bandwidth=bandwidth, max_distance=MAX_DISTANCE)

    for gene, color in zip(genes, colors, strict=False):
        gene_df = sub[sub["gene"] == gene].sort_values("bin_center")
        curve = smoothed[smoothed["gene"] == gene]
        called = bool(gene_stats.loc[gene, "monotone_decay"]) if gene in gene_stats.index else False
        ax.plot(
            gene_df["bin_center"],
            gene_df["mean_z"],
            marker="o",
            ms=1.6,
            mew=0,
            lw=0.6,
            color=color,
            alpha=0.25,
            zorder=1,
        )
        if curve.empty:
            continue
        ax.fill_between(
            curve["distance"],
            curve["fit"] - 1.96 * curve["se"],
            curve["fit"] + 1.96 * curve["se"],
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )
        # Genes the test declines are drawn dashed, so the two panels agree at a glance.
        ax.plot(
            curve["distance"],
            curve["fit"],
            lw=2.2,
            ls="-" if called else "--",
            color=color,
            label=gene,
            zorder=3,
            solid_capstyle="round",
        )

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.45, zorder=0)
    ax.set_xlim(0, MAX_DISTANCE)
    ax.set_xlabel("Surface-to-surface distance to nearest sender (µm)")
    ax.set_ylabel("Mean expression (z-scored log1p)")
    ax.set_title(interaction, fontsize=10)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper right")
    ax.grid(alpha=0.22)
    ax.set_axisbelow(True)


def plot_all_pairs(ax, per_dataset):
    """Every sender-receiver pair in every dataset: one row, one dot per gene.

    Dot position is the gene's monotone R^2, filled dots are genes called as monotone
    decays. The row label carries the fraction called, so the panel answers both "is this
    pair monotone" and "how strong is the evidence per gene" without a figure per dataset.
    """
    rows = []
    for name, (_, stats) in per_dataset.items():
        for interaction, sub in stats.groupby("interaction", sort=False):
            testable = sub[sub["testable"]]
            rows.append(
                {
                    "dataset": name,
                    "interaction": interaction,
                    "n_testable": len(testable),
                    "n_called": int(testable["monotone_decay"].sum()) if len(testable) else 0,
                    "scores": testable["spearman_rho"].to_numpy(),
                    "called": testable["monotone_decay"].to_numpy(),
                    "qvalues": testable["spearman_qvalue"].to_numpy(),
                    "n_untestable": len(sub) - len(testable),
                }
            )
    frame = pd.DataFrame(rows)
    # Pairs with no testable gene are dropped rather than drawn as an empty row; they are
    # still counted in the untestable total reported in the text.
    frame = frame[frame["n_testable"] > 0].reset_index(drop=True)

    # Datasets in the declared order, pairs within a dataset ordered by median score.
    order = []
    for name, *_ in DATASETS:
        block = frame[frame["dataset"] == name].copy()
        block["median_score"] = [np.median(s) if len(s) else np.nan for s in block["scores"]]
        order.append(block.sort_values("median_score", ascending=True, na_position="last"))
    frame = pd.concat(order, ignore_index=True)

    positions = np.arange(len(frame))[::-1]
    dataset_colors = dict(
        zip(
            [name for name, *_ in DATASETS],
            plt.get_cmap("Dark2")(np.linspace(0, 1, len(DATASETS), endpoint=False)),
            strict=False,
        )
    )

    labels = []
    row_lines = []
    for pos, (_, row) in zip(positions, frame.iterrows(), strict=False):
        color = dataset_colors[row["dataset"]]
        jitter = np.linspace(-0.16, 0.16, row["n_testable"]) if row["n_testable"] > 1 else np.zeros(1)
        for score, called, offset in zip(row["scores"], row["called"], jitter, strict=False):
            ax.plot(
                score,
                pos + offset,
                marker="o",
                ms=5.0,
                color=color if called else "white",
                markeredgecolor=color,
                markeredgewidth=1.1,
                alpha=0.95 if called else 0.85,
                zorder=3,
            )
        row_lines.append(pos)
        labels.append(f"{row['interaction']}   ({row['n_called']}/{row['n_testable']})")

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_ylim(-0.8, len(frame) - 0.2)
    lo = float(min(np.min(r["scores"]) for _, r in frame.iterrows()))
    hi = float(max(np.max(r["scores"]) for _, r in frame.iterrows()))
    pad = 0.08 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.axvline(0, color="black", lw=0.9, alpha=0.6, zorder=2)
    for pos in row_lines:
        ax.plot(ax.get_xlim(), [pos, pos], color="0.9", lw=0.7, zorder=1)
    ax.set_xlabel("Spearman correlation of expression with distance to nearest sender")
    ax.set_title(
        "All signalling pairs, all datasets: one dot per downstream gene, filled if it passes the monotonic decay test",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)

    handles = [plt.Line2D([], [], color=dataset_colors[name], lw=5, label=name) for name, *_ in DATASETS]
    handles += [
        plt.Line2D([], [], color="0.35", marker="o", ls="", ms=5.5, label="passes monotonic decay test"),
        plt.Line2D(
            [],
            [],
            color="0.35",
            marker="o",
            ls="",
            ms=5.5,
            markerfacecolor="white",
            label="fails; no significant decreasing trend",
        ),
    ]
    # Anchored to the figure rather than the axis so it clears the x-axis label whatever
    # the panel height ends up being.
    ax.figure.legend(
        handles=handles,
        fontsize=8,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.012),
        ncol=6,
        columnspacing=1.6,
        handletextpad=0.5,
    )
    return frame


def main():
    """Assemble the example pair and the across-dataset summary into one figure."""
    per_dataset = {}
    bandwidths = {}
    for name, results_dir, prefix, bandwidth in DATASETS:
        per_dataset[name] = load_dataset(results_dir, prefix)
        bandwidths[name] = bandwidth

    example_bins, example_stats = per_dataset[EXAMPLE_DATASET]

    n_pairs = sum(
        int((frame["testable"].groupby(frame["interaction"], sort=False).any()).sum())
        for _, frame in per_dataset.values()
    )
    fig = plt.figure(figsize=(12.0, 4.6 + 0.30 * n_pairs))
    grid = fig.add_gridspec(2, 1, height_ratios=[4.4, 0.30 * n_pairs], hspace=0.30)

    ax_a = fig.add_subplot(grid[0, 0])
    plot_example_profile(ax_a, example_bins, example_stats, EXAMPLE_INTERACTION, bandwidths[EXAMPLE_DATASET])
    ax_b = fig.add_subplot(grid[1, 0])
    frame = plot_all_pairs(ax_b, per_dataset)

    for ax, letter in ((ax_a, "a"), (ax_b, "b")):
        ax.text(-0.07, 1.02, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")

    total_called = int(frame["n_called"].sum())
    total_testable = int(frame["n_testable"].sum())
    fig.suptitle(
        f"Model-free test of the monotone distance assumption: "
        f"{total_called}/{total_testable} testable gene-pair combinations pass the monotonic decay test",
        fontsize=12,
        y=0.995,
    )
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"monotonicity_summary.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    frame.drop(columns=["scores", "called", "qvalues"]).to_csv(
        os.path.join(OUTPUT_DIR, "monotonicity_by_pair.csv"), index=False
    )
    print(frame.drop(columns=["scores", "called", "qvalues"]).to_string(index=False))
    print(f"\n{total_called}/{total_testable} testable gene-pair combinations pass the monotonic decay test")
    print(f"Example panel: {EXAMPLE_DATASET}, {EXAMPLE_INTERACTION}")
    print(f"Minimum bins required to test: {MIN_BINS_FOR_STATS}")
    print(f"\nWrote figure and table to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
