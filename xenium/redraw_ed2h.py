"""Redraw Extended Data Fig. 2h from the deposited segmentation-test Source Data.

Why this exists
---------------
The published panel disagreed with its own caption and with Fig. 4c.

The caption says the dashed line marks p = 0.05, which is the threshold the Methods describe and
the one `plot_segmentation_test_results` in `segmentation_analysis.py` has drawn in every revision
of that file. The published panel nevertheless draws and labels the line at p = 0.1, so it is a
render that predates the threshold being settled rather than anything the current code produces.

The same render also disagrees about a gene. It shows C15orf48 (CD8 T Cells -> M1 Macrophages) in
red, passing the segmentation test, while the deposited Source Data gives it p_max = 1.0 and
`bold_in_figure = False`. Fig. 4c already draws C15orf48 unbolded, so the two panels of the
submission contradicted each other. That is a data change, not a threshold effect: no gene in this
panel changes class between p < 0.05 and p < 0.1, and the counts are 7 of 19 either way.

Styling
-------
The panel was restyled by hand when the figure was assembled, so it does not look like its sibling
Extended Data Fig. 2h-equivalent for GITIII (Extended Data Fig. 3a), which is the plotting
function's default output. The assembled panel drops the figure title and the summary statistics
box, shortens the y tick labels to the bare gene name, and carries the sender-receiver pair on
brackets drawn beside the axis instead. Those brackets, the panel title and the panel letter are
separate objects in the figure file and are not redrawn here.

This script reproduces that style, so the committed code, the figure file and the caption agree.
Gene order is fixed to the published order so the brackets still line up against the rows.

The legend is emitted as its own file because it sits outside the axes in the assembled panel and
is a separate object there too.

Run from the xenium directory:  python redraw_ed2h.py
"""

# %% Imports
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# %% Configuration
PVAL_THRESHOLD = 0.05

SOURCE_DATA = "../source_data/panels/ED2/ED2h_segmentation_test.csv"
OUT_DIR = "./figures/segmentation_analysis"

# The published row order, grouped by sender-receiver pair. The brackets beside the axis are drawn
# against these groups, so the order is fixed here rather than sorted by p-value: a sort would
# reshuffle the ties (eleven of the nineteen genes sit at p = 1.0) and break the alignment.
ROW_ORDER = [
    ("CD8 T Cells", "Invasive Tumor", ["SERPINA3", "AGR3", "ESR1"]),
    ("CD8 T Cells", "M1 Macrophages", ["SLAMF7", "C15orf48", "WARS"]),
    ("Invasive Tumor", "M1 Macrophages", ["FASN", "FOXA1", "KRT7", "SCD", "PPARG", "APOC1", "FCGR3A"]),
    ("CD4 T Cells", "CD8 T Cells", ["SELL", "LTB", "CCR7", "IL7R", "GPR183", "TCF7"]),
]

# The panel occupies a 164.67 x 116.34 pt frame in the figure file, with the axes inset from it by
# the fractions below. Emitting at exactly that size keeps one SVG unit equal to one point, so the
# file drops into the frame without being rescaled.
FRAME_W_PT, FRAME_H_PT = 164.67, 116.34
AXES_LEFT, AXES_RIGHT, AXES_TOP, AXES_BOTTOM = 0.2454, 0.9917, 1.0, 0.0673

# Measured off the assembled panel at 600 dpi: gene labels have a 21 px cap height, the x tick
# digits 12 px, the markers an 18 px diameter and the threshold line a 3 px stroke.
FS_TICK = 3.5
FS_LABEL = 3.5
FS_XTICK = 2.0
MARKER_AREA = 4.3
LINE_WIDTH = 0.36
DASHES = (1.3, 0.7)  # points; see lines.scale_dashes below
YLIM = (-0.913, 18.739)
YLABEL_COORDS = (-0.3151, 0.448)

SIG_COLOR = "red"
NONSIG_COLOR = "lightblue"


# Matplotlib multiplies an explicit dash pattern by the line width unless this is off, which at a
# 0.36 pt stroke turns the 1.3/0.7 pt pattern into a fine dotted line rather than the published dash.
mpl.rcParams["lines.scale_dashes"] = False


# %% Load the deposited per-gene test results
def load_segmentation_rows(path=SOURCE_DATA):
    """Return the nineteen panel rows, in the published order, from the Source Data."""
    df = pd.read_csv(path)
    df = df[df["segmentation_p_max"].notna()]
    df = df.set_index(["gene", "sender", "receiver"])

    rows = []
    for sender, receiver, genes in ROW_ORDER:
        for gene in genes:
            key = (gene, sender, receiver)
            if key not in df.index:
                raise KeyError(f"{gene} ({sender} -> {receiver}) is not in {path}")
            rows.append(
                {
                    "gene": gene,
                    "sender": sender,
                    "receiver": receiver,
                    "pval": float(df.loc[key, "segmentation_p_max"]),
                }
            )

    expected = sum(len(genes) for _, _, genes in ROW_ORDER)
    if len(rows) != expected or len(df) != expected:
        raise ValueError(f"expected {expected} rows, got {len(rows)} ordered of {len(df)} in file")
    return pd.DataFrame(rows)


# %% Draw the panel
def plot_panel(rows, out_stem, pval_threshold=PVAL_THRESHOLD):
    """Draw the scatter in the assembled panel's style: no title, no stats box, gene-only labels."""
    fig = plt.figure(figsize=(FRAME_W_PT / 72.0, FRAME_H_PT / 72.0))
    ax = fig.add_axes(
        [
            AXES_LEFT,
            AXES_BOTTOM,
            AXES_RIGHT - AXES_LEFT,
            AXES_TOP - AXES_BOTTOM,
        ]
    )

    # Row 0 at the top, matching the published panel.
    y = range(len(rows))[::-1]
    colors = [SIG_COLOR if p < pval_threshold else NONSIG_COLOR for p in rows["pval"]]

    ax.scatter(
        rows["pval"],
        y,
        c=colors,
        s=MARKER_AREA,
        marker="o",
        edgecolors="black",
        linewidth=0.15,
        alpha=0.8,
        zorder=3,
    )
    ax.axvline(
        x=pval_threshold,
        color="red",
        linestyle="--",
        alpha=0.7,
        linewidth=LINE_WIDTH,
        dashes=DASHES,
        zorder=2,
    )

    ax.set_yticks(list(y))
    ax.set_yticklabels(rows["gene"], fontsize=FS_TICK)
    ax.set_xlim(0, 1)
    ax.set_ylim(*YLIM)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="x", labelsize=FS_XTICK, length=0.5, width=0.25, pad=0.8)
    ax.tick_params(axis="y", length=0.5, width=0.25, pad=0.8)
    ax.set_xlabel("Mann-Whitney U Test P-value", fontsize=FS_LABEL, labelpad=0.8)
    ax.set_ylabel("Genes by Cell Type Pair", fontsize=FS_LABEL)
    # The assembled panel carries the sender-receiver brackets in the left margin, so the axis
    # label sits hard against the frame edge rather than beside the tick labels where matplotlib
    # would put it; these are the published coordinates, in axes fractions.
    ax.yaxis.set_label_coords(YLABEL_COORDS[0], YLABEL_COORDS[1])
    ax.grid(True, alpha=0.3, axis="x", linewidth=0.3)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.25)

    for ext in ("svg", "png"):
        path = f"{out_stem}.{ext}"
        fig.savefig(path, format=ext, dpi=600, transparent=True)
        print(f"[panel] wrote {path}")
    plt.close(fig)


# %% Draw the legend as its own file
def plot_legend(out_stem, pval_threshold=PVAL_THRESHOLD):
    """The legend sits outside the axes in the assembled panel, so it is emitted separately."""
    fig = plt.figure(figsize=(67.84 / 72.0, 16.38 / 72.0))
    handles = [
        Patch(facecolor=SIG_COLOR, alpha=0.8, label=f"Significant (p < {pval_threshold})"),
        Patch(facecolor=NONSIG_COLOR, alpha=0.8, label=f"Non-significant (p ≥ {pval_threshold})"),
        Line2D(
            [0],
            [0],
            color="red",
            linestyle="--",
            alpha=0.7,
            linewidth=LINE_WIDTH,
            dashes=DASHES,
            label=f"Significance threshold (p = {pval_threshold})",
        ),
    ]
    legend = fig.legend(
        handles=handles,
        loc="center",
        fontsize=3.4,
        frameon=True,
        borderpad=0.4,
        labelspacing=0.45,
        handlelength=1.6,
        handletextpad=0.6,
    )
    legend.get_frame().set_linewidth(0.3)
    legend.get_frame().set_edgecolor("0.6")

    for ext in ("svg", "png"):
        path = f"{out_stem}.{ext}"
        fig.savefig(path, format=ext, dpi=600, transparent=True)
        print(f"[legend] wrote {path}")
    plt.close(fig)


# %% Run
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_segmentation_rows()

    n_sig = int((rows["pval"] < PVAL_THRESHOLD).sum())
    print(f"{len(rows)} genes, {n_sig} significant at p < {PVAL_THRESHOLD}")
    for _, r in rows[rows["pval"] < PVAL_THRESHOLD].iterrows():
        print(f"  {r['gene']:10s} {r['sender']} -> {r['receiver']}  p = {r['pval']:.4g}")

    plot_panel(rows, f"{OUT_DIR}/ED2h_panel_p0.05")
    plot_legend(f"{OUT_DIR}/ED2h_legend_p0.05")
