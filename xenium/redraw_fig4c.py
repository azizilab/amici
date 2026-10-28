"""Redraw Fig. 4c with the gene bolding taken from the segmentation-artifact test.

Why this exists
---------------
The Fig. 4c legend says that "genes not flagged as potential segmentation artifacts are in bold", but
no code in this repository produces bold gene labels: `xenium_analysis.py:351-360` calls
`plot_featurewise_contributions_dotplot` without `mark_segmentation_passing_genes`, so the published
dot plots carry no segmentation marking at all and the bold labels were applied by hand when the
figure was assembled.

The published bolding is reproduced exactly (19 of 19 sender-gene pairs) by the *single* comparison of
interacting receivers against distant senders. The Methods require both comparisons -- also against
distant receivers -- and the two rules disagree on one gene: C15orf48 (CD8 T cells -> M1 macrophages)
is far higher in interacting receivers than in distant senders (p = 9.3e-9) but no higher than in
distant receivers (p = 0.98), so it fails the Methods' test and should not be bold. It is not
mentioned anywhere in the manuscript text.

This script redraws the three dot plots with the bolding recomputed under the Methods' two-test rule,
so the panel matches both its own legend and the Source Data sheet shipped with the paper.

The bold rule
-------------
A bold label spans a gene's whole column, so a gene is bold when it passes for any of the senders it
was selected for (both one-sided Mann-Whitney U p-values below 0.05). Every gene of this panel was
selected for exactly one sender, so the rule makes no difference here; it is written the same way as
in `atera_breast/redraw_fig5c.py`, where genes are selected for two senders and it does matter.
The p-values are read from `source_data/panels/Fig4/Fig4c.csv` rather than recomputed, so the panel
and the Source Data cannot drift apart.

Dot colour, size, gene selection and layout all come from the same library call as the published
panel, driven by the cached ablation scores, so no model and no GPU are needed.

Run from the xenium directory:  python redraw_fig4c.py
"""

# %% Imports
import os

import matplotlib.pyplot as plt
import pandas as pd

from amici.interpretation._ablation_module import AMICIAblationModule

# %% Paths and the three dot plots, as published
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_CSV = os.path.join(REPO, "source_data/panels/Fig4/Fig4c.csv")
CACHE_PATH = "./figures/cached_ablation_scores/all_cell_types_ablation_scores.pkl"
OUT_DIR = "./figures/fig4c_redraw"

# Every receiver shows the same three sender rows (fig4_xenium.py DOT_SENDERS).
DOT_SENDERS = ["CD4+_T_Cells", "CD8+_T_Cells", "Invasive_Tumor"]
RECEIVERS = ["Macrophages_1", "CD8+_T_Cells", "Invasive_Tumor"]
# plot_featurewise_contributions_dotplot arguments of xenium_analysis.py:351-360.
DOT_N_TOP_GENES = 10
DOT_MIN_SIZE_BY = -10
DOT_STEP = 5

# Display names of source_data/fig4_xenium.py, which the exported panel uses.
DISPLAY_NAMES = {
    "CD8+_T_Cells": "CD8 T Cells",
    "CD4+_T_Cells": "CD4 T Cells",
    "Macrophages_1": "M1 Macrophages",
    "Invasive_Tumor": "Invasive Tumor",
}


def display(label):
    return DISPLAY_NAMES.get(label, label)


# %% Read the segmentation test from the exported panel and derive the bold set
panel = pd.read_csv(PANEL_CSV)
pass_col = "passes_segmentation_test_p_lt_0.05"

bold_rows = []
for receiver in RECEIVERS:
    rows = panel[panel["receiver"] == display(receiver)]
    for gene, gene_rows in rows.groupby("gene", sort=False):
        tested = gene_rows[gene_rows["gene_selected_for_sender"]].dropna(subset=[pass_col])
        bold_rows.append(
            {
                "receiver": display(receiver),
                "gene": gene,
                "n_senders_tested": len(tested),
                "n_senders_passing": int(tested[pass_col].sum()),
                "bold_published": bool(gene_rows["bold_in_figure"].iloc[0]),
                "bold_redrawn": bool(len(tested) > 0 and tested[pass_col].any()),
                "max_p_over_senders": tested["segmentation_p_max"].max(),
            }
        )
bolding = pd.DataFrame(bold_rows)
bolding["changed"] = bolding["bold_published"] != bolding["bold_redrawn"]

print(f"Genes whose bolding changes: {int(bolding['changed'].sum())} of {len(bolding)}")
if bolding["changed"].any():
    print(bolding[bolding["changed"]].to_string(index=False))

bold_by_receiver = {
    receiver: set(group[group["bold_redrawn"]]["gene"]) for receiver, group in bolding.groupby("receiver")
}

# %% Load the cached ablation scores (all cell types, all heads)
ablation = AMICIAblationModule.load_object(CACHE_PATH)
published_cell_types = list(ablation._cell_types)

# %% Redraw each dot plot
os.makedirs(OUT_DIR, exist_ok=True)
for receiver in RECEIVERS:
    # The published panels restricted the senders through ablated_neighbor_ct_sub; the cached object
    # holds every cell type, so restrict the same way before drawing.
    senders = [s for s in DOT_SENDERS if s != receiver]
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
    stem = os.path.join(OUT_DIR, f"Fig4c_{receiver}_redraw")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

ablation._cell_types = published_cell_types

# %% Record what changed
bolding_path = os.path.join(OUT_DIR, "fig4c_bolding.csv")
bolding.to_csv(bolding_path, index=False)
print(f"\nWrote {bolding_path}")
