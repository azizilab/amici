"""Source Data for Figure 6 (CosMx 1k human tonsil).

Panels
  6a  Spatial cell-type map with the held-out test region.
  6b  Interaction network for the cell types of interest and the germinal-center heatmap.
  6c  Downstream gene dot plots for GC B cell (resting) and CD4 T cell receivers.
  6d  Head-resolved dot plots (heads 3 and 6) for GC B cell (resting) and the matching length-scale
      distributions.

Checkpoint: wandb sweep 0f9rk0na, run 4303v0zu (see source_data/checkpoints.yaml).
obsm["spatial"] is in CosMx pixels; lengths are converted with 0.12028 um/px as in
human_tonsil/human_tonsil_analysis.py.
Run from the repository root:  python source_data/fig6_tonsil.py
"""

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc

import _common as C

NAME = "human_tonsil"
FIGURE = "Fig6"
PIXEL_SIZE_UM = 0.12028

GRAPH_QUANTILE = 0.80
GRAPH_CELL_TYPES = [
    "Tfh cell",
    "CD4 T cell",
    "GC B cell (resting)",
    "Fibroblast",
    "Follicular dendritic cell",
    "Macrophage (GPNMB+)",
    "Macrophage (C1Q+)",
]
GC_SUBSET = ["GC B cell (resting)", "GC B cell (cycling)", "Follicular dendritic cell", "Macrophage (GPNMB+)"]
PUBLISHED_GC_ROWS = ["Follicular dendritic cell", "Macrophage (GPNMB+)", "GC B cell (resting)", "GC B cell (cycling)"]
PUBLISHED_GC_COLS = ["GC B cell (resting)", "GC B cell (cycling)", "Follicular dendritic cell", "Macrophage (GPNMB+)"]

GC_B = "GC B cell (resting)"
# Sender order of the dot-plot calls; it sets the column order of the union of top genes.
GC_SENDERS = ["Tfh cell", "Follicular dendritic cell"]
DOTPLOTS = [
    {
        "receiver": GC_B,
        "senders": GC_SENDERS,
        "genes": ["CLU", "MARCKSL1", "CD22", "MS4A1", "RGS13", "BASP1", "SRGN", "MALAT1", "CXCL13", "RPL32", "VCAM1", "PTGDS"],
    },
    {
        "receiver": "CD4 T cell",
        "senders": ["Fibroblast"],
        "genes": ["CCL21", "CCL19", "VIM", "TPT1", "RPL32", "RPL34", "DCN", "PTGDS", "CXCL14", "CXCL12"],
    },
]
DOT_N_TOP_GENES = 10
DOT_MIN_SIZE_BY = -10

HEAD_DOTPLOTS = {
    3: ["CLU", "MARCKSL1", "CD74", "SRGN", "BASP1", "MS4A1", "PFN1", "HLA-DRA", "CXCL13", "CD22"],
    6: ["IL32", "CXCL13", "MALAT1", "CLU", "SRGN", "VCAM1", "VIM"],
}
HEAD_LABELS = {3: "Long-range interaction (head 3)", 6: "Short-range interaction (head 6)"}
HIGHLIGHTED_RED = {"CXCL13", "VCAM1"}
LENGTH_SCALE_HEADS = [3, 6]
LENGTH_SCALE_CACHE = "human_tonsil/figures/human_tonsil_analysis/human_tonsil_all_heads_length_scale_distribution_um.csv"


def main():
    C.ensure_dirs()
    entry = C.CHECKPOINTS[NAME]
    C.verify_checkpoint(NAME)
    pl.seed_everything(entry["seed"])
    labels_key = entry["labels_key"]

    adata = sc.read_h5ad(C.REPO / entry["data"])
    model = C.load_model(NAME, adata)
    present = set(adata.obs[labels_key].astype(str).unique())
    missing = [ct for ct in GRAPH_CELL_TYPES + GC_SUBSET if ct not in present]
    assert not missing, f"cell types not found: {missing}"

    # ---- 6a ----------------------------------------------------------------------------------------
    coords = np.asarray(adata.obsm["spatial"])
    spatial = pd.DataFrame(
        {
            "cell_id": adata.obs_names,
            "x_px": coords[:, 0],
            "y_px": coords[:, 1],
            "x_um": coords[:, 0] * PIXEL_SIZE_UM,
            "y_um": coords[:, 1] * PIXEL_SIZE_UM,
            "cell_type": adata.obs[labels_key].astype(str).values,
            "train_test_split": adata.obs["train_test_split"].astype(str).values,
        }
    )

    # ---- 6b ----------------------------------------------------------------------------------------
    ablation = C.load_or_compute_all_ablation(NAME, model, adata)
    abl_df = ablation._ablation_scores_df
    weight_df = ablation._get_interaction_weight_matrix()
    edges = C.interaction_edges(weight_df, GRAPH_CELL_TYPES, GRAPH_QUANTILE)
    print("  Fig 6b drawn edges:")
    print(edges.loc[edges["edge_drawn"], ["sender", "receiver", "n_significant_genes"]].to_string(index=False))
    gc = C.clustered_matrix(weight_df.loc[GC_SUBSET, GC_SUBSET])
    C.check_gene_list("6b GC heatmap rows (senders)", list(gc.index), PUBLISHED_GC_ROWS)
    C.check_gene_list("6b GC heatmap columns (receivers)", list(gc.columns), PUBLISHED_GC_COLS)
    full = C.clustered_matrix(weight_df)

    # ---- 6c ----------------------------------------------------------------------------------------
    dot_tables = []
    for spec in DOTPLOTS:
        receiver, senders = spec["receiver"], spec["senders"]
        selected = C.select_dotplot_genes(abl_df, receiver, senders, DOT_N_TOP_GENES, DOT_MIN_SIZE_BY)
        union = []
        for sender in senders:
            union += [g for g in selected[sender] if g not in union]
        print(f"  6c {receiver}: selection per sender = {selected}")
        C.check_gene_list(f"6c receiver {receiver} (union of senders)", union, spec["genes"])

        fresh = model.get_neighbor_ablation_scores(
            adata=adata, cell_type=receiver, ablated_neighbor_ct_sub=senders, compute_z_value=True
        )
        cached = C.receiver_gene_table(abl_df, receiver, senders)
        recomputed = C.receiver_gene_table(fresh._ablation_scores_df, receiver, senders)
        merged = cached.merge(recomputed, on=["receiver", "sender", "gene"], suffixes=("", "_fresh"))
        print(
            f"  6c {receiver}: max |diff| cache vs fresh = "
            f"{np.nanmax(np.abs(merged['neighbor_contribution'] - merged['neighbor_contribution_fresh'])):.2e}, "
            f"max |z| = {np.nanmax(np.abs(merged['wald_z'] - merged['wald_z_fresh'])):.2e}"
        )
        table = C.dotplot_panel(recomputed, receiver, senders, spec["genes"], f"6c (Receiver: {receiver})")
        table["gene_selected_for_sender"] = [
            g in selected.get(s, []) for s, g in zip(table["sender"], table["gene"], strict=True)
        ]
        dot_tables.append(table)
    dots = pd.concat(dot_tables, ignore_index=True)

    # ---- 6d: head-resolved dot plots ---------------------------------------------------------------
    head_tables = []
    for head, genes in HEAD_DOTPLOTS.items():
        head_abl = model.get_neighbor_ablation_scores(
            adata=adata, cell_type=GC_B, head_idx=head, ablated_neighbor_ct_sub=GC_SENDERS, compute_z_value=True
        )
        head_df = head_abl._ablation_scores_df
        for n_top in [5, 8, 10, 12]:
            for min_size in [-10, 0, 5, 10]:
                sel = C.select_dotplot_genes(head_df, GC_B, GC_SENDERS, n_top, min_size)
                union = []
                for sender in GC_SENDERS:
                    union += [g for g in sel[sender] if g not in union]
                if union == genes:
                    print(f"  6d head {head}: published genes reproduced with n_top_genes={n_top}, min_size_by={min_size}")
        tab = C.receiver_gene_table(head_df, GC_B, GC_SENDERS, head=head)
        table = C.dotplot_panel(tab, GC_B, GC_SENDERS, genes, HEAD_LABELS[head])
        table["highlighted_red_in_figure"] = table["gene"].isin(HIGHLIGHTED_RED)
        head_tables.append(table)
    head_dots = pd.concat(head_tables, ignore_index=True).rename(
        columns={"neighbor_contribution": "head_specific_neighbor_contribution"}
    )

    # ---- 6d: length-scale distributions ------------------------------------------------------------
    cf = model.get_counterfactual_attention_patterns(cell_type=GC_B, adata=adata)
    ls = cf._calculate_length_scales(
        head_idxs=LENGTH_SCALE_HEADS, sender_types=GC_SENDERS, attention_threshold=0.1, sample_threshold=0.01
    )
    ls = ls.rename(columns={"length_scale": "length_scale_px", "neighbor_idx": "sender_cell_id", "sender_type": "sender"})
    ls["length_scale_um"] = ls["length_scale_px"] * PIXEL_SIZE_UM
    ls.insert(0, "receiver", GC_B)
    cache = C.REPO / LENGTH_SCALE_CACHE
    if cache.exists():
        old = pd.read_csv(cache).rename(columns={"neighbor_idx": "sender_cell_id", "sender_type": "sender"})
        both = ls.merge(old, on=["head_idx", "sender", "sender_cell_id"], suffixes=("", "_cached"))
        print(
            f"  6d length scales vs {cache.name}: {len(both)}/{len(ls)} rows matched, max |diff| = "
            f"{np.abs(both['length_scale_um'] - both['length_scale_um_cached']).max():.2e} um"
        )
    ls_summary = (
        ls.groupby(["head_idx", "sender"])["length_scale_um"]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .reset_index()
        .rename(columns={"count": "n_sender_cells", "50%": "median", "25%": "q1", "75%": "q3"})
    )
    print(ls_summary.to_string(index=False))

    panels = [
        {
            "sheet": "Fig6a",
            "description": "Spatial cell-type map: one row per cell with CosMx pixel coordinates and the same in um "
            "(0.12028 um/px), cell type, and train/test split (test cells form the dashed box).",
            "df": spatial,
        },
        {
            "sheet": "Fig6b_network",
            "description": "Interaction network. Edge weight = number of receiver genes with BH-adjusted p < 0.05 "
            "and positive neighbor contribution when the sender type is ablated; edges are drawn when the "
            f"weight is at least the {GRAPH_QUANTILE} quantile of the full matrix; arrow thickness follows "
            "relative_edge_width.",
            "df": edges,
        },
        {
            "sheet": "Fig6b_GC_heatmap",
            "description": "Germinal-center interaction heatmap (rows senders, columns receivers) in the order of "
            "the Ward clustering drawn in the figure.",
            "df": gc,
            "index": True,
        },
        {
            "sheet": "Fig6b_full_heatmap",
            "description": "Full clustered sender x receiver interaction heatmap shown as the inset in Fig. 6b.",
            "df": full,
            "index": True,
        },
        {
            "sheet": "Fig6c",
            "description": "Dot plots of downstream genes. Colour = neighbor contribution (normalized expression "
            "units, scale clipped to +/-0.5); size = Wald test statistic (wald_z, labelled Z-value). p_value / "
            "p_adj_BH are the one-sided p of wald_z and its BH adjustment over all genes of the receiver/sender "
            "pair.",
            "df": dots,
        },
        {
            "sheet": "Fig6d_head_dotplots",
            "description": "Head-resolved dot plots for GC B cell (resting): neighbor contribution with only the "
            "given head active (colour) and its Wald statistic (size). highlighted_red_in_figure marks genes "
            "set in red in the figure.",
            "df": head_dots,
        },
        {
            "sheet": "Fig6d_length_scales",
            "description": "Per-sender-cell length scale for GC B cell (resting) receivers on heads 3 and 6 "
            "(attention threshold 0.1), in CosMx pixels and um.",
            "df": ls,
        },
        {
            "sheet": "Fig6d_length_scale_summary",
            "description": "Box-plot statistics (um) of each box in Fig. 6d.",
            "df": ls_summary,
        },
    ]
    readme = [
        ("Figure", "Figure 6: AMICI on CosMx human tonsil"),
        ("Fig. 6a", "Sheet Fig6a"),
        ("Fig. 6b", "Sheets Fig6b_network, Fig6b_GC_heatmap, Fig6b_full_heatmap"),
        ("Fig. 6c", "Sheet Fig6c"),
        ("Fig. 6d", "Sheets Fig6d_head_dotplots, Fig6d_length_scales, Fig6d_length_scale_summary"),
    ]
    C.save_panels(FIGURE, panels, readme, C.provenance_for(NAME, {"pixel_size_um": PIXEL_SIZE_UM}))


if __name__ == "__main__":
    main()
