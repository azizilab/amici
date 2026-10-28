"""Redraw Fig. 6c with the gene bolding taken from the segmentation-artifact test.

Why this exists
---------------
Two separate problems met in this panel.

First, the distance thresholds. `adata.obsm["spatial"]` for the tonsil is `CenterX/Y_global_px`,
CosMx global *pixels*, while the library's `segmentation_near_threshold` and
`segmentation_far_threshold` default to the 20 um and 100 um of the Methods. Run unconverted they
select receivers within 2.4 um and senders beyond 12 um, which leaves 1, 3 and 9 interacting
receivers for the three sender-receiver pairs and makes every p-value meaningless. The scale is
0.120281 um/px, read from `uns["fov_positions"]`, which carries both pixel and millimetre
coordinates for every FOV; `human_tonsil_analysis.py` now passes the converted thresholds.

Second, the marking. `mark_segmentation_passing_genes` draws a white diamond per *sender-gene pair*,
but the published panel carries bold gene labels instead, applied by hand when the figure was
assembled. A label spans a gene's whole column, so it follows the same any-sender rule as Fig. 4c
and Fig. 5c: bold when the gene passes for at least one of the senders drawn above it.

This redraws both dot plots with the bolding recomputed at the corrected thresholds, so the panel
matches its own legend and the Methods.

The test
--------
Both one-sided Mann-Whitney U comparisons of the Methods -- interacting receivers against distant
senders, and against distant receivers -- at p < 0.05, using every cell in each group rather than
subsampling to the smallest. Mann-Whitney does not need equal group sizes, and subsampling here
would throw away thousands of receivers to match a sender group that is small for reasons of tissue
architecture: only 3 follicular dendritic cells lie more than 100 um from a GC B cell, so that
pair's sender comparison is weak however it is run.

Distances use a KD-tree; the cell x cell matrix `scipy.spatial.distance.cdist` would build for
41,497 fibroblasts against 55,622 CD4 T cells is 18 GB.

Run from the human_tonsil directory:  python redraw_fig6c.py
"""

# %% Imports
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats
from scipy.spatial import cKDTree

from amici.interpretation._ablation_module import AMICIAblationModule

# %% Paths and the two dot plots, as published
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = "./data/human_tonsil_filtered_2026-07-25.h5ad"
CACHE_PATH = "./figures/human_tonsil_analysis/cached_ablation_scores/all_cell_types_ablation_scores.pkl"
OUT_DIR = "./figures/fig6c_redraw"
LABELS_KEY = "celltype_manual_fine"

# uns["fov_positions"] gives both px and mm per FOV; their ratio is this, to six places.
UM_PER_PX = 0.120281
NEAR_PX = 20.0 / UM_PER_PX
FAR_PX = 100.0 / UM_PER_PX

# Receiver -> senders, in the order human_tonsil_analysis.py draws them.
DOTPLOTS = [
    ("GC B cell (resting)", ["Tfh cell", "Follicular dendritic cell"]),
    ("CD4 T cell", ["Fibroblast"]),
]
# plot_featurewise_contributions_dotplot arguments of human_tonsil_analysis.py:372-384.
DOT_N_TOP_GENES = 10
DOT_MIN_SIZE_BY = -10
DOT_STEP = 20


def segmentation_tests(adata, pairs):
    """Both one-sided Mann-Whitney comparisons per sender-receiver pair, all cells, no subsampling."""
    genes = sorted({g for _, _, gs in pairs for g in gs})
    columns = [adata.var_names.get_loc(g) for g in genes]
    block = adata.X[:, columns]
    expression = pd.DataFrame(
        np.asarray(block.todense()) if sparse.issparse(block) else np.asarray(block), columns=genes
    )
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    labels = adata.obs[LABELS_KEY].to_numpy()

    rows = []
    for sender, receiver, sender_genes in pairs:
        sender_idx = np.flatnonzero(labels == sender)
        receiver_idx = np.flatnonzero(labels == receiver)
        if not len(sender_idx) or not len(receiver_idx):
            continue
        to_sender = cKDTree(coords[sender_idx]).query(coords[receiver_idx], k=1)[0]
        to_receiver = cKDTree(coords[receiver_idx]).query(coords[sender_idx], k=1)[0]
        near_receivers = receiver_idx[to_sender <= NEAR_PX]
        far_receivers = receiver_idx[to_sender >= FAR_PX]
        far_senders = sender_idx[to_receiver >= FAR_PX]
        if not (len(near_receivers) and len(far_receivers) and len(far_senders)):
            continue
        for gene in sender_genes:
            values = expression[gene].to_numpy()
            p_sender = stats.mannwhitneyu(values[near_receivers], values[far_senders], alternative="greater")[1]
            p_receiver = stats.mannwhitneyu(values[near_receivers], values[far_receivers], alternative="greater")[1]
            rows.append(
                {
                    "receiver": receiver,
                    "sender": sender,
                    "gene": gene,
                    "n_near_receivers": len(near_receivers),
                    "n_far_senders": len(far_senders),
                    "n_far_receivers": len(far_receivers),
                    "segmentation_p_near_vs_far_sender": p_sender,
                    "segmentation_p_near_vs_far_receiver": p_receiver,
                    "segmentation_p_max": max(p_sender, p_receiver),
                    "passes_segmentation_test_p_lt_0.05": bool(max(p_sender, p_receiver) < 0.05),
                }
            )
    return pd.DataFrame(rows)


# %% Load the tissue and the cached ablation scores, then pick the genes each panel draws
adata = sc.read_h5ad(DATA)
gene_names = set(adata.var_names)

ablation = AMICIAblationModule.load_object(CACHE_PATH)
published_cell_types = list(ablation._cell_types)

drawn = {}
for receiver, senders in DOTPLOTS:
    ablation._cell_types = [receiver] + senders
    ablation.plot_featurewise_contributions_dotplot(
        cell_type=receiver,
        color_by="diff",
        size_by="z_value",
        n_top_genes=DOT_N_TOP_GENES,
        min_size_by=DOT_MIN_SIZE_BY,
        step=DOT_STEP,
        save_svg=False,
        save_png=False,
        show=False,
    )
    figure = plt.gcf()
    # The size and colour legends are axes too, so keep only labels that name a gene.
    genes = []
    for axis in figure.get_axes():
        for label in axis.get_xticklabels():
            if label.get_text() in gene_names:
                genes.append(label.get_text())
    drawn[receiver] = list(dict.fromkeys(genes))
    plt.close(figure)
    print(f"{receiver}: {len(drawn[receiver])} genes -> {drawn[receiver]}", flush=True)

# %% Run the test for every sender-gene pair the panels show
pairs = [(s, r, drawn[r]) for r, senders in DOTPLOTS for s in senders]
print(f"testing {len(pairs)} sender-receiver pairs at near {NEAR_PX:.1f} px / far {FAR_PX:.1f} px", flush=True)
tests = segmentation_tests(adata, pairs)

# A label spans the gene's column, so it is bold when any sender drawn above it passes.
bold = tests.groupby(["receiver", "gene"])["passes_segmentation_test_p_lt_0.05"].any()
tests["bold_in_figure"] = tests.set_index(["receiver", "gene"]).index.map(bold)

# %% Redraw each dot plot with the bolding applied
os.makedirs(OUT_DIR, exist_ok=True)
for receiver, senders in DOTPLOTS:
    ablation._cell_types = [receiver] + senders
    ablation.plot_featurewise_contributions_dotplot(
        cell_type=receiver,
        color_by="diff",
        size_by="z_value",
        n_top_genes=DOT_N_TOP_GENES,
        min_size_by=DOT_MIN_SIZE_BY,
        step=DOT_STEP,
        save_svg=False,
        save_png=False,
        show=False,
    )
    figure = plt.gcf()
    bold_genes = {gene for (recv, gene), passes in bold.items() if recv == receiver and passes}
    marked = []
    for axis in figure.get_axes():
        for label in axis.get_xticklabels():
            gene = label.get_text()
            if not gene:
                continue
            label.set_fontweight("bold" if gene in bold_genes else "normal")
            if gene in bold_genes:
                marked.append(gene)
    print(f"{receiver}: bolded {len(marked)}/{len(drawn[receiver])} -> {sorted(set(marked))}", flush=True)
    stem = os.path.join(OUT_DIR, f"Fig6c_{receiver.replace(' ', '_').replace('(', '').replace(')', '')}_redraw")
    for extension in ("png", "svg", "pdf"):
        figure.savefig(f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)

ablation._cell_types = published_cell_types

# %% Record the test and the bolding it implies
tests_path = os.path.join(OUT_DIR, "fig6c_segmentation_tests.csv")
tests.to_csv(tests_path, index=False)
print(f"\nWrote {tests_path}")
