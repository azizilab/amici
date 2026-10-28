"""Compute the gene labels and the bolding of the Extended Data Fig. 2c volcano plots.

Why this exists
---------------
ED 2c draws one volcano per immune/tumor receiver, labels "the top 10 genes by neighbor
contribution" and prints in bold those "not flagged as potential segmentation artifacts". The
volcanoes themselves come from `xenium_analysis.py:583-635`, which plots the dots and nothing else:
it writes no labels, applies no bolding and saves no table, so the panel had no Source Data and its
bold labels were applied by hand when the figure was assembled.

This recomputes both from the cached ablation scores. The labels are the top `N_TOP` dots of each
receiver by neighbor contribution, across all of its senders -- a gene can therefore be labeled twice
in one subplot when it is a top dot for two different senders, which is what the published panel
shows. The bolding is the test of the Methods: a gene is bold when the interacting receivers express
it significantly above both the distant senders and the distant receivers (both one-sided
Mann-Whitney U p-values below 0.05), which is the same rule Fig. 4c and ED 2h use. A gene that is
labeled twice is bold on both labels when it passes for either sender, so that the least powered
pair cannot veto the gene's label.

A caveat worth keeping in view: the test has very little power for the rarest receivers. LAMP3+ DCs
leave 2 to 9 cells per group and IRF7+ DCs 16 to 33, so for those subplots a plain label means the
comparison could not resolve a difference, not that the gene looks like a segmentation artifact.
`segmentation_n_cells_per_group` is written out so a reader can see this.

Outputs, under `./figures/ed2c_volcano_bolding/`:

``ed2c_volcano_all_genes.csv``
    Every dot of every subplot: gene x sender x receiver, with the neighbor contribution and Wald
    statistic the axes show.
``ed2c_labeled_genes.csv``
    The labeled dots only, with the segmentation test for each and the bolding it implies.

No model and no GPU are needed; the ablation scores come from the cache.

Run from the xenium directory:  python ed2c_volcano_bolding.py
"""

# %% Imports
import os
import sys

import pandas as pd
import scanpy as sc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "source_data"))

import _common as C  # noqa: E402

from amici.interpretation._ablation_module import AMICIAblationModule  # noqa: E402

# %% Paths and the panel's contents, as published
DATA = "./data/xenium_sample1/xenium_sample1_filtered_2025-05-01.h5ad"
CACHE_PATH = "./figures/cached_ablation_scores/all_cell_types_ablation_scores.pkl"
OUT_DIR = "./figures/ed2c_volcano_bolding"
LABELS_KEY = "celltype_train_grouped"
N_TOP = 10

# The immune/tumor subset of Fig. 4b (right), in the order source_data/fig4_xenium.py defines it.
SUBSET = [
    "Invasive_Tumor", "DCIS_1", "DCIS_2", "Macrophages_2", "Macrophages_1",
    "CD8+_T_Cells", "CD4+_T_Cells", "B_Cells", "IRF7+_DCs", "LAMP3+_DCs",
]

# %% Load the tissue and the cached ablation scores
print("loading adata", flush=True)
adata = sc.read_h5ad(DATA)

print("loading cached ablation scores", flush=True)
ablation = AMICIAblationModule.load_object(CACHE_PATH)
abl_df = ablation._ablation_scores_df

# %% Every dot of every subplot
dots = pd.concat(
    [C.receiver_gene_table(abl_df, receiver, [s for s in SUBSET if s != receiver]) for receiver in SUBSET],
    ignore_index=True,
)
print(f"volcano dots: {len(dots):,} over {dots['receiver'].nunique()} receivers", flush=True)

# %% The labeled dots: the top N of each receiver, across its senders
labeled = (
    dots.sort_values("neighbor_contribution", ascending=False)
    .groupby("receiver", sort=False)
    .head(N_TOP)
    .copy()
)
labeled["rank_in_panel"] = labeled.groupby("receiver")["neighbor_contribution"].rank(
    ascending=False, method="first"
)
labeled = labeled.sort_values(["receiver", "rank_in_panel"])
print(f"labeled dots: {len(labeled)}", flush=True)

# %% The segmentation test, one call per sender-receiver pair
# The pairs are sorted so that the subsample seeds, which are seeded by position, are reproducible.
pairs = [
    (sender, receiver, sorted(group["gene"].tolist()))
    for (receiver, sender), group in labeled.groupby(["receiver", "sender"], sort=True)
]
print(f"running the segmentation test for {len(pairs)} sender-receiver pairs", flush=True)
seg = C.segmentation_tests(adata, LABELS_KEY, pairs)

labeled = labeled.merge(seg, on=["receiver", "sender", "gene"], how="left")
labeled["bold_in_figure"] = (
    labeled.groupby(["receiver", "gene"])["passes_segmentation_test_p_lt_0.05"].transform("any").fillna(False)
)

# %% Write the tables
os.makedirs(OUT_DIR, exist_ok=True)
dots.to_csv(os.path.join(OUT_DIR, "ed2c_volcano_all_genes.csv"), index=False)
labeled.to_csv(os.path.join(OUT_DIR, "ed2c_labeled_genes.csv"), index=False)

# %% Report what the bolding should be
print("\n================ ED 2c: the bolding implied by the test ================")
for receiver, group in labeled.groupby("receiver", sort=False):
    bold = group[group["bold_in_figure"]]["gene"].tolist()
    counts = group["segmentation_n_cells_per_group"].dropna()
    span = f"{int(counts.min())}-{int(counts.max())}" if len(counts) else "n/a"
    print(f"{receiver:16s} (n per group {span:>9s})  bold {len(bold):2d}/{len(group):2d}: {bold}")

print(f"\nWrote {OUT_DIR}/ed2c_labeled_genes.csv and {OUT_DIR}/ed2c_volcano_all_genes.csv")
