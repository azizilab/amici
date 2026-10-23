"""Source Data for Figure 5 (Atera whole-transcriptome breast cancer).

Panels
  5a  Spatial cell-type map with the held-out test region.
  5b  Clustered interaction-strength heatmap for all sender x receiver pairs.
  5c  Downstream gene dot plots for Endothelial, CAF and Luminal DCIS receivers, with the
      segmentation-artifact test that decides which genes are bold.

Checkpoint: wandb sweep 25xpxkuk, run zhtraubt (see source_data/checkpoints.yaml).
The computations mirror atera_breast/atera_breast_analysis.py and
atera_breast/atera_breast_segmentation_analysis.py.
Run from the repository root:  python source_data/fig5_atera.py
"""

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc

import _common as C

NAME = "atera_breast"
FIGURE = "Fig5"

DISPLAY_NAMES = {
    "High_Grade_DCIS": "High Grade DCIS",
    "Luminal_DCIS": "Luminal DCIS",
    "Basal_DCIS": "Basal DCIS",
    "T_Cells": "T Cells",
    "Mast_Cells": "Mast Cells",
    "Plasma_Cells": "Plasma Cells",
    "Plasma_Mast_Mixture": "Plasma/Mast Cells",
}

# Panel 5b: row (sender) and column (receiver) order of the published heatmap.
PUBLISHED_ROWS = [
    "CAFs", "Endothelial", "Myoepithelial", "Apocrine Cells", "Mast_Cells", "Fibroblasts", "Myeloid",
    "Plasma_Cells", "Plasma_Mast_Mixture", "High_Grade_DCIS", "Basal_DCIS", "Pericytes", "Luminal_DCIS",
    "Dendritic Cells", "Macrophages", "T_Cells",
]
PUBLISHED_COLS = [
    "T_Cells", "Macrophages", "Apocrine Cells", "Mast_Cells", "Myeloid", "Plasma_Cells", "Fibroblasts",
    "Plasma_Mast_Mixture", "CAFs", "High_Grade_DCIS", "Pericytes", "Basal_DCIS", "Luminal_DCIS",
    "Dendritic Cells", "Endothelial", "Myoepithelial",
]

# Panel 5c, exactly as drawn. Bold genes are those not flagged by the segmentation-artifact test.
DOTPLOTS = [
    {
        "receiver": "Endothelial",
        "senders": ["Pericytes"],
        "genes": ["PECAM1", "ENTPD1", "STOM", "ENG", "ADAM15", "SELP", "PRCP"],
        "bold": ["PECAM1", "ADAM15", "SELP", "PRCP"],
    },
    {
        "receiver": "CAFs",
        "senders": ["Macrophages", "T_Cells"],
        "genes": ["C3", "COL3A1", "COL6A1", "COL6A2", "COL6A3", "MMP2", "CXCL12", "EGR1", "ABI3BP", "EMILIN1", "COL15A1"],
        "bold": ["C3", "COL6A2", "MMP2", "CXCL12", "ABI3BP", "EMILIN1", "COL15A1"],
    },
    {
        "receiver": "Luminal_DCIS",
        # Sender order of atera_breast_analysis.py; it sets the column order of the union of top genes.
        "senders": ["Myoepithelial", "CAFs"],
        "genes": [
            "CCND1", "SLC12A2", "EVL", "TFF1", "DUSP4", "GATA3", "C6orf141", "MUC1", "BMPR1B", "ITGB5", "TFF3", "GASK1B",
        ],
        "bold": ["MUC1", "BMPR1B", "ITGB5", "TFF3", "GASK1B"],
    },
]
DOT_N_TOP_GENES = 8
DOT_MIN_SIZE_BY = -10


def display(label):
    return DISPLAY_NAMES.get(label, label)


def main():
    C.ensure_dirs()
    entry = C.CHECKPOINTS[NAME]
    C.verify_checkpoint(NAME)
    pl.seed_everything(entry["seed"])
    labels_key = entry["labels_key"]

    adata = sc.read_h5ad(C.REPO / entry["data"])
    model = C.load_model(NAME, adata)

    # ---- 5a ----------------------------------------------------------------------------------------
    coords = np.asarray(adata.obsm["spatial"])
    spatial = pd.DataFrame(
        {
            "cell_id": adata.obs_names,
            "x_um": coords[:, 0],
            "y_um": coords[:, 1],
            "cell_type_model_label": adata.obs[labels_key].astype(str).values,
            "cell_type_display": adata.obs[labels_key].astype(str).map(display).values,
            "train_test_split": adata.obs["train_test_split"].astype(str).values,
        }
    )

    # ---- 5b ----------------------------------------------------------------------------------------
    ablation = C.load_or_compute_all_ablation(NAME, model, adata)
    abl_df = ablation._ablation_scores_df
    weight_df = ablation._get_interaction_weight_matrix()
    clustered = C.clustered_matrix(weight_df)
    C.check_gene_list("5b heatmap rows (senders)", list(clustered.index), PUBLISHED_ROWS)
    C.check_gene_list("5b heatmap columns (receivers)", list(clustered.columns), PUBLISHED_COLS)

    # ---- 5c ----------------------------------------------------------------------------------------
    seg_pairs = [(s, spec["receiver"], spec["genes"]) for spec in DOTPLOTS for s in spec["senders"]]
    seg = C.segmentation_tests(adata, labels_key, seg_pairs)
    dot_tables = []
    for spec in DOTPLOTS:
        receiver, senders = spec["receiver"], spec["senders"]
        selected = C.select_dotplot_genes(abl_df, receiver, senders, DOT_N_TOP_GENES, DOT_MIN_SIZE_BY)
        union = []
        for sender in senders:
            union += [g for g in selected[sender] if g not in union]
        print(f"  5c {receiver}: selection per sender = {selected}")
        C.check_gene_list(f"5c receiver {receiver} (union of senders)", union, spec["genes"])

        fresh = model.get_neighbor_ablation_scores(
            adata=adata, cell_type=receiver, ablated_neighbor_ct_sub=senders, compute_z_value=True
        )
        cached = C.receiver_gene_table(abl_df, receiver, senders)
        recomputed = C.receiver_gene_table(fresh._ablation_scores_df, receiver, senders)
        merged = cached.merge(recomputed, on=["receiver", "sender", "gene"], suffixes=("", "_fresh"))
        print(
            f"  5c {receiver}: max |diff| cache vs fresh = "
            f"{np.nanmax(np.abs(merged['neighbor_contribution'] - merged['neighbor_contribution_fresh'])):.2e}, "
            f"max |z| = {np.nanmax(np.abs(merged['wald_z'] - merged['wald_z_fresh'])):.2e}"
        )

        table = C.dotplot_panel(recomputed, receiver, senders, spec["genes"], f"5c (Receiver: {display(receiver)})")
        table["gene_selected_for_sender"] = [
            g in selected.get(s, []) for s, g in zip(table["sender"], table["gene"], strict=True)
        ]
        table = table.merge(seg, on=["receiver", "sender", "gene"], how="left")
        table["bold_in_figure"] = table["gene"].isin(spec["bold"])
        dot_tables.append(table)
    dots = pd.concat(dot_tables, ignore_index=True)
    C.report_bold_agreement(dots, "5c")
    for col in ["receiver", "sender"]:
        dots[col] = dots[col].map(display)

    clustered_display = clustered.rename(index=display, columns=display)
    panels = [
        {
            "sheet": "Fig5a",
            "description": "Spatial cell-type map: one row per cell with coordinates (um), cell type and "
            "train/test split (test cells form the dashed box).",
            "df": spatial,
        },
        {
            "sheet": "Fig5b",
            "description": "Interaction strength = number of receiver genes with BH-adjusted p < 0.05 and positive "
            "neighbor contribution when the sender type is ablated. Rows are senders and columns receivers, in "
            "the order of the Ward hierarchical clustering drawn in the figure.",
            "df": clustered_display,
            "index": True,
        },
        {
            "sheet": "Fig5c",
            "description": "Dot plots of downstream genes. Colour = neighbor contribution (normalized expression "
            "units, scale clipped to +/-0.5); size = Wald test statistic (wald_z, labelled Z-value). p_value / "
            "p_adj_BH are the one-sided p of wald_z and its BH adjustment over all genes of the receiver/sender "
            "pair. segmentation_* columns: one-sided Mann-Whitney U segmentation-artifact test "
            "(segmentation_p_max = larger of the two p-values; passes when below 0.05 per the Methods, or 0.1 as in "
            "the analysis scripts). bold_in_figure records the published bolding, which this test does not "
            "fully reproduce (see README).",
            "df": dots,
        },
    ]
    readme = [
        ("Figure", "Figure 5: AMICI on Atera whole-transcriptome breast cancer"),
        ("Fig. 5a", "Sheet Fig5a"),
        ("Fig. 5b", "Sheet Fig5b"),
        ("Fig. 5c", "Sheet Fig5c"),
    ]
    C.save_panels(FIGURE, panels, readme, C.provenance_for(NAME))


if __name__ == "__main__":
    main()
