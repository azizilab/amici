"""Source Data for Figure 3 (MERFISH mouse cortex).

Panels
  3a  Spatial cell-type map of the six slices with the held-out test slice.
  3b  Directed interaction graph for Oligodendrocytes, L4/5 IT, L2/3 IT, Astrocytes and Sst.
  3c  Downstream gene dot plots for Astrocyte and Sst receivers.

Checkpoint: wandb sweep plm73bmg, run xrtcnlt0 (see source_data/checkpoints.yaml).
Run from the repository root:  python source_data/fig3_cortex.py
"""

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc

import _common as C

NAME = "cortex"
FIGURE = "Fig3"

# Display names used in the published legend for labels that differ from the model's labels.
DISPLAY_NAMES = {
    "Astro": "Astrocytes",
    "Oligo": "Oligodendrocytes",
    "Endo": "Endothelial",
    "Peri": "Pericytes",
    "Micro": "Microglial",
    "other": "Other",
}

# Panel 3b: node set of the published subgraph and the edge threshold quantile of cortex_analysis.py.
GRAPH_CELL_TYPES = ["Oligo", "L4/5 IT", "L2/3 IT", "Astro", "Sst"]
GRAPH_QUANTILE = 0.86

# Panel 3c: receivers, senders and genes exactly as drawn in the manuscript.
DOTPLOTS = [
    {
        "panel": "3c (Receiver: Astrocytes)",
        "receiver": "Astro",
        "senders": ["L2/3 IT", "Oligo"],
        "genes": {
            "L2/3 IT": ["Cux2", "Lama3", "Lamp5"],
            "Oligo": ["Igfbp5", "Gfap", "Bgn", "Ccdc3", "Sox10", "Adamts4", "Serpinf1"],
        },
        "n_top_genes": 8,
        "min_size_by": 15,
    },
    {
        "panel": "3c (Receiver: Sst)",
        "receiver": "Sst",
        "senders": ["L2/3 IT", "L4/5 IT"],
        "genes": {
            "L2/3 IT": ["Cux2", "Col25a1", "Pcdh8", "Lamp5"],
            "L4/5 IT": ["Npnt", "Olfm3", "Syndig1", "Hpse", "Tox", "Unc13c", "Rerg"],
        },
        # cortex_analysis.py only contains the Astrocyte call; these settings reproduce the published Sst
        # gene selection exactly (as do n_top_genes=9 and min_size_by=6).
        "n_top_genes": 10,
        "min_size_by": 5,
    },
]


def main():
    C.ensure_dirs()
    entry = C.CHECKPOINTS[NAME]
    C.verify_checkpoint(NAME)
    pl.seed_everything(entry["seed"])

    adata = sc.read_h5ad(C.REPO / entry["data"])
    adata.obsm["spatial"] = adata.obs[["centroid_x", "centroid_y"]].values
    model = C.load_model(NAME, adata)
    labels_key = entry["labels_key"]

    panels = []

    # ---- 3a: spatial map ---------------------------------------------------------------------------
    spatial = pd.DataFrame(
        {
            "cell_id": adata.obs_names,
            "slice_id": adata.obs["slice_id"].values,
            "x_um": adata.obs["centroid_x"].values,
            "y_um": adata.obs["centroid_y"].values,
            "cell_type_model_label": adata.obs[labels_key].astype(str).values,
            "cell_type_display": adata.obs[labels_key].astype(str).map(lambda s: DISPLAY_NAMES.get(s, s)).values,
            "held_out_test": adata.obs["in_test"].astype(bool).values,
        }
    )
    panels.append(
        {
            "sheet": "Fig3a",
            "description": "Spatial cell-type map: one row per cell with centroid coordinates (um), slice, "
            "cell type and whether the cell is in the held-out test slice (dashed box).",
            "df": spatial,
        }
    )

    # ---- 3b: directed interaction graph ------------------------------------------------------------
    ablation = C.load_or_compute_all_ablation(NAME, model, adata)
    weight_df = ablation._get_interaction_weight_matrix()
    edges = C.interaction_edges(weight_df, GRAPH_CELL_TYPES, GRAPH_QUANTILE)
    for col in ["sender", "receiver"]:
        edges[f"{col}_display"] = edges[col].map(lambda s: DISPLAY_NAMES.get(s, s))
    print("  Fig 3b drawn edges:")
    print(edges.loc[edges["edge_drawn"], ["sender", "receiver", "n_significant_genes"]].to_string(index=False))
    panels.append(
        {
            "sheet": "Fig3b",
            "description": "Directed interaction graph. Edge weight = number of receiver genes with BH-adjusted "
            "p < 0.05 and positive neighbor contribution when the sender type is ablated. Edges are drawn when "
            f"the weight is at least the {GRAPH_QUANTILE} quantile of the full {weight_df.shape[0]}x"
            f"{weight_df.shape[1]} matrix; line width and opacity are proportional to relative_edge_width.",
            "df": edges,
        }
    )
    full_matrix = weight_df.copy()
    full_matrix.index.name = "sender \\ receiver"
    panels.append(
        {
            "sheet": "Fig3b_full_matrix",
            "description": "Full sender x receiver interaction weight matrix from which the Fig. 3b subgraph and "
            "its threshold are taken (also underlies Fig. S4).",
            "df": full_matrix,
            "index": True,
        }
    )

    # ---- 3c: dot plots -----------------------------------------------------------------------------
    dot_tables = []
    for spec in DOTPLOTS:
        receiver = spec["receiver"]
        published = [g for s in spec["senders"] for g in spec["genes"][s]]
        selected = C.select_dotplot_genes(
            ablation._ablation_scores_df, receiver, spec["senders"], spec["n_top_genes"], spec["min_size_by"]
        )
        for sender in spec["senders"]:
            C.check_gene_list(f"{spec['panel']} sender {sender}", selected.get(sender, []), spec["genes"][sender])

        # Independent recompute for this receiver, to confirm the cached scores came from this checkpoint.
        fresh = model.get_neighbor_ablation_scores(
            adata=adata, cell_type=receiver, ablated_neighbor_ct_sub=spec["senders"], compute_z_value=True
        )
        cached = C.receiver_gene_table(ablation._ablation_scores_df, receiver, spec["senders"])
        recomputed = C.receiver_gene_table(fresh._ablation_scores_df, receiver, spec["senders"])
        merged = cached.merge(recomputed, on=["receiver", "sender", "gene"], suffixes=("", "_fresh"))
        max_diff = np.nanmax(np.abs(merged["neighbor_contribution"] - merged["neighbor_contribution_fresh"]))
        max_z = np.nanmax(np.abs(merged["wald_z"] - merged["wald_z_fresh"]))
        print(f"  {spec['panel']}: max |diff| cache vs fresh = {max_diff:.2e}, max |z| = {max_z:.2e}")

        table = C.dotplot_panel(recomputed, receiver, spec["senders"], published, spec["panel"])
        table["gene_selected_for_sender"] = [
            g in spec["genes"].get(s, []) for s, g in zip(table["sender"], table["gene"], strict=True)
        ]
        table["receiver"] = DISPLAY_NAMES.get(receiver, receiver)
        table["sender"] = table["sender"].map(lambda s: DISPLAY_NAMES.get(s, s))
        dot_tables.append(table)

    panels.append(
        {
            "sheet": "Fig3c",
            "description": "Dot plots of downstream genes. Colour = neighbor contribution (mean change in predicted "
            "expression when the sender type is ablated, normalized expression units; colour scale clipped to "
            "+/-0.5). Dot size = Wald test statistic (wald_z). p_value is the one-sided p of wald_z and p_adj_BH "
            "its Benjamini-Hochberg adjustment across all genes of the receiver/sender pair. "
            "gene_selected_for_sender marks the sender whose top genes contributed the column.",
            "df": pd.concat(dot_tables, ignore_index=True),
        }
    )

    readme = [
        ("Figure", "Figure 3: AMICI on MERFISH mouse cortex"),
        ("Fig. 3a", "Sheet Fig3a"),
        ("Fig. 3b", "Sheets Fig3b (edges) and Fig3b_full_matrix"),
        ("Fig. 3c", "Sheet Fig3c"),
    ]
    C.save_panels(FIGURE, panels, readme, C.provenance_for(NAME, {"graph_quantile": GRAPH_QUANTILE}))


if __name__ == "__main__":
    main()
