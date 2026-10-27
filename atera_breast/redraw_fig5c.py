"""Redraw Fig. 5c with the gene bolding taken from the segmentation-artifact test.

Why this exists
---------------
The Fig. 5c legend says that "genes not flagged as potential segmentation artifacts are in bold",
but no code in this repository produces the published bolding. `atera_breast_analysis.py:411-412`
calls `plot_featurewise_contributions_dotplot` with `flag_segmentation_artifacts` and
`segmentation_adata` commented out, so the published dot plots carry no segmentation marking at all;
the bold labels were applied by hand when the figure was assembled.

They do not agree with the test. Over the 53 gene x sender rows of
`source_data/panels/Fig5/Fig5c.csv`, the published bolding matches the test at p < 0.05 for only 20.
The Luminal DCIS columns are inverted: CCND1, SLC12A2, EVL, TFF1, DUSP4, GATA3 and C6orf141 all pass
overwhelmingly (p down to 6e-102) and are not bold, while MUC1 fails (p = 0.15 and 0.35) and is bold.

This script redraws the three dot plots with the bolding recomputed, so the figure matches both its
own legend and the Source Data sheet shipped with the paper.

The bold rule
-------------
The test is per sender-gene pair, but a bold label applies to a gene across every sender drawn in its
panel, so a gene is bold when it passes for *all* of its senders (both one-sided Mann-Whitney U
p-values below 0.05 against distant senders and distant receivers). The three CAF genes that pass for
one sender but not the other (MMP2 and ABI3BP for Macrophages, EGR1 for T Cells) are therefore not
bold; `fig5c_bolding.csv` records the per-pair result either way.

The p-values are read from `source_data/panels/Fig5/Fig5c.csv` rather than recomputed, so the panel
and the Source Data cannot drift apart. Note that the library's own
`_get_segmentation_artifact_gene_pairs` seeds its subsample with `abs(hash((sender, receiver)))`,
which is not stable across processes unless PYTHONHASHSEED is set; the exporter's deterministic
`seed 42 + pair_idx` is used instead.

Dot colour, size, gene selection and layout all come from the same library call as the published
panel, driven by the cached ablation scores, so no model and no GPU are needed.

Run from the atera_breast directory:  python redraw_fig5c.py
"""

# %% Imports
import os

import matplotlib.pyplot as plt
import pandas as pd

from amici.interpretation._ablation_module import AMICIAblationModule

# %% Paths and the three dot plots, as published
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_CSV = os.path.join(REPO, "source_data/panels/Fig5/Fig5c.csv")
CACHE_PATH = "./figures/atera_breast_analysis/cached_ablation_scores/all_cell_types_ablation_scores.pkl"
OUT_DIR = "./figures/atera_breast_analysis"

# Receiver -> senders, in the sender order atera_breast_analysis.py used.
DOTPLOTS = [
    ("Endothelial", ["Pericytes"]),
    ("CAFs", ["Macrophages", "T_Cells"]),
    ("Luminal_DCIS", ["Myoepithelial", "CAFs"]),
]
# plot_featurewise_contributions_dotplot arguments of atera_breast_analysis.py:409-421.
DOT_N_TOP_GENES = 8
DOT_MIN_SIZE_BY = -10
DOT_STEP = 15

# Display names of source_data/fig5_atera.py, which the exported panel uses.
DISPLAY_NAMES = {"Luminal_DCIS": "Luminal DCIS", "T_Cells": "T Cells"}


def display(label):
    return DISPLAY_NAMES.get(label, label)


# %% Read the segmentation test from the exported panel and derive the bold set
panel = pd.read_csv(PANEL_CSV)
pass_col = "passes_segmentation_test_p_lt_0.05"

bold_rows = []
for receiver, senders in DOTPLOTS:
    rows = panel[panel["receiver"] == display(receiver)]
    for gene, gene_rows in rows.groupby("gene", sort=False):
        tested = gene_rows.dropna(subset=[pass_col])
        bold_rows.append(
            {
                "receiver": display(receiver),
                "gene": gene,
                "n_senders_tested": len(tested),
                "n_senders_passing": int(tested[pass_col].sum()),
                "bold_published": bool(gene_rows["bold_in_figure"].iloc[0]),
                "bold_redrawn": bool(len(tested) > 0 and tested[pass_col].all()),
                "max_p_over_senders": tested["segmentation_p_max"].max(),
            }
        )
bolding = pd.DataFrame(bold_rows)
bolding["changed"] = bolding["bold_published"] != bolding["bold_redrawn"]

print(f"Genes whose bolding changes: {int(bolding['changed'].sum())} of {len(bolding)}")
print(bolding[bolding["changed"]].to_string(index=False))

bold_by_receiver = {
    receiver: set(group[group["bold_redrawn"]]["gene"]) for receiver, group in bolding.groupby("receiver")
}

# %% Load the cached ablation scores (all cell types, all heads)
ablation = AMICIAblationModule.load_object(CACHE_PATH)
published_cell_types = list(ablation._cell_types)

# %% Redraw each dot plot
os.makedirs(OUT_DIR, exist_ok=True)
for receiver, senders in DOTPLOTS:
    # The published panels restricted the senders through ablated_neighbor_ct_sub; the cached object
    # holds every cell type, so restrict the same way before drawing.
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
    fig = plt.gcf()

    bold_genes = bold_by_receiver.get(display(receiver), set())
    drawn, bolded = [], []
    for ax in fig.get_axes():
        labels = ax.get_xticklabels()
        if not labels or not any(label.get_text() in set(panel["gene"]) for label in labels):
            continue
        for label in labels:
            gene = label.get_text()
            if not gene:
                continue
            drawn.append(gene)
            if gene in bold_genes:
                label.set_fontweight("bold")
                bolded.append(gene)
            else:
                label.set_fontweight("normal")

    print(f"{receiver}: drew {len(drawn)} genes, bolded {len(bolded)} -> {sorted(bolded)}")
    stem = os.path.join(OUT_DIR, f"Fig5c_{receiver}_redraw")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

ablation._cell_types = published_cell_types

# %% Record what changed
bolding_path = os.path.join(OUT_DIR, "fig5c_bolding.csv")
bolding.to_csv(bolding_path, index=False)
print(f"\nWrote {bolding_path}")
