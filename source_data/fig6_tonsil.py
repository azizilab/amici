"""Source Data for Figure 6 (CosMx 1k human tonsil).

Panels
  6a  Spatial cell-type map with the held-out test region.
  6b  Interaction network for the cell types of interest and the germinal-center heatmap.
  6c  Downstream gene dot plots for GC B cell (resting) and CD4 T cell receivers.
  6d  Ablation dot plot for head 4 alone, fibroblast senders to CD4 T cell receivers, and the
      effective pairwise radius distribution for that pair on that head.

Checkpoint: the seed-40 retrain on 992 genes, after the five ribosomal-protein genes, UBA52, FAU and
MALAT1 were dropped and the counts renormalized. Every panel of the published Fig. 6 comes from it
(see the figure legend and source_data/checkpoints.yaml); the 1000-gene model the earlier version of
this figure used is still recorded, because ED 6a sets the two against each other.

Panel 6d is not the 6d of the earlier version. That one was two head-resolved dot plots, heads 3 and
6, for GC B cell (resting) receivers. The retrain's 6d is a single head: head 4 is the one head whose
fibroblast -> CD4 T cell radius sits in the tens of microns, and it is the head AMICI assigns CCR7,
which is the pairing the earlier two-head version could not support.
obsm["spatial"] is in CosMx pixels; lengths are converted with 0.12028 um/px as in
human_tonsil/human_tonsil_analysis.py.
Run from the repository root:  python source_data/fig6_tonsil.py
"""

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
from scipy import sparse, stats
from scipy.spatial import cKDTree

import _common as C

NAME = "human_tonsil_no_ribo"
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
# Read off the published panel: the column labels are all four in the text layer, the row labels all
# but "GC B cell (cycling)", whose slot between the first and third rows leaves it the only option.
PUBLISHED_GC_ROWS = ["GC B cell (resting)", "GC B cell (cycling)", "Follicular dendritic cell", "Macrophage (GPNMB+)"]
PUBLISHED_GC_COLS = ["GC B cell (resting)", "GC B cell (cycling)", "Follicular dendritic cell", "Macrophage (GPNMB+)"]

GC_B = "GC B cell (resting)"
CD4_T = "CD4 T cell"
# Sender order of the dot-plot calls; it sets the column order of the union of top genes.
GC_SENDERS = ["Tfh cell", "Follicular dendritic cell"]
CD4_T_SENDERS = ["Fibroblast"]
# adata.obsm["spatial"] is CenterX/Y_global_px, CosMx global pixels rather than microns, so the
# segmentation test's 20/100um thresholds have to be converted. The scale is the ratio of the pixel
# and millimetre coordinates uns["fov_positions"] carries for every FOV.
UM_PER_PX = 0.120281
# Gene order as the published panels draw it, left to right along each dot plot's gene axis, read
# out of the text layer of figures/Figure 6.pdf. MALAT1, RPL32 and RPL34, which the earlier lists
# carried, are not in the 992-gene panel at all.
DOTPLOTS = [
    {
        "receiver": GC_B,
        "senders": GC_SENDERS,
        "genes": ["MARCKSL1", "BASP1", "MS4A1", "CD22", "RGS13", "HLA-DRA", "P2RX5", "RACK1", "CD37",
                  "CSK", "TPT1", "VPREB3", "CD74"],
    },
    {
        "receiver": CD4_T,
        "senders": CD4_T_SENDERS,
        "genes": ["TPT1", "KLF2", "TCF7", "RACK1", "PLAC8", "CCR7", "BTG1", "NACA", "SELL", "CXCR4"],
    },
]
DOT_N_TOP_GENES = 10
DOT_MIN_SIZE_BY = -10

# 6d is one head over one pair, so the head dot plot and the length scales share a receiver and a
# sender with each other rather than with 6c.
HEAD_DOTPLOT_RECEIVER = CD4_T
HEAD_DOTPLOT_SENDERS = CD4_T_SENDERS
HEAD_DOTPLOTS = {
    4: ["KLF2", "CCR7", "TPT1", "PLAC8", "SELL", "TCF7", "CXCR4", "NACA", "NOSIP", "NELL2"],
}
HEAD_LABELS = {4: "Chemokine-range interaction (head 4)"}
HIGHLIGHTED_RED = {"CCR7"}
LENGTH_SCALE_HEADS = [4]
LENGTH_SCALE_CACHE = "human_tonsil/figures/human_tonsil_no_ribo_analysis/fig6d/fig6d_head_4_length_scale_distribution_um.csv"


def select_dotplot_genes_drawn(adata, labels_key, ablation_df, receiver, senders, n_top, min_size_by, panel):
    """The gene selection the panel actually draws: the top `n_top` that survive the segmentation filter.

    `C.select_dotplot_genes` takes the top genes by neighbour contribution, which is what
    `plot_featurewise_contributions_dotplot` does on its own. The analysis calls it with
    `exclude_segmentation_flagged_genes=True` and `segmentation_compare_near_senders=True`, so flagged
    genes are dropped before the top `n_top` is taken, and the published lists cannot be reproduced
    without that step. Three one-sided comparisons have to pass at p < 0.05: receivers bordering a
    sender against distant senders and against distant receivers, which are the two the Methods
    describe, and against the senders those receivers border, which is the third.

    The third is what keeps fibroblast *CCL19* and *CCL21* out of the CD4 T cell panel. Both beat the
    two distant groups -- T-zone fibroblasts express them and distant fibroblasts do not -- but not
    the fibroblasts the receivers touch, so the panel reads them as spillover from the sender rather
    than as a receiver program, and the whole of Fig. 6c is receiver genes.
    """
    labels = adata.obs[labels_key].astype(str).to_numpy()
    coords = np.asarray(adata.obsm["spatial"], dtype=float) * UM_PER_PX
    sub = ablation_df[ablation_df["cell_type"] == receiver]
    selected, dropped = {}, {}

    for sender in senders:
        candidates = sub[
            (sub[f"{sender}_diff"] > 0)
            & (sub[f"{sender}_z_value"] > min_size_by)
            & (sub[f"{sender}_nl10_pval_adj"] > -np.log10(0.05))
        ].sort_values(f"{sender}_diff", ascending=False)
        genes = candidates["gene"].astype(str).tolist()

        r_idx, s_idx = np.flatnonzero(labels == receiver), np.flatnonzero(labels == sender)
        r_dist = cKDTree(coords[s_idx]).query(coords[r_idx], k=1)[0]
        s_dist = cKDTree(coords[r_idx]).query(coords[s_idx], k=1)[0]
        near_r, far_r = r_idx[r_dist <= 20], r_idx[r_dist >= 100]
        near_s, far_s = s_idx[s_dist <= 20], s_idx[s_dist >= 100]
        block = adata[:, genes].X
        block = block.toarray() if sparse.issparse(block) else np.asarray(block)

        keep = []
        for j, gene in enumerate(genes):
            values = block[:, j]
            p = [
                stats.mannwhitneyu(values[near_r], values[group], alternative="greater")[1]
                for group in (far_s, far_r, near_s)
            ]
            if max(p) < 0.05:
                keep.append(gene)
        selected[sender] = keep[:n_top]
        dropped[sender] = [g for g in genes[:n_top] if g not in keep]
        print(
            f"  {panel} {receiver} <- {sender}: {len(genes)} candidates, {len(keep)} pass all three "
            f"comparisons; the unfiltered top {n_top} loses {dropped[sender]}"
        )
    return selected


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
    seg_pairs = []
    for spec in DOTPLOTS:
        receiver, senders = spec["receiver"], spec["senders"]
        selected = select_dotplot_genes_drawn(
            adata, labels_key, abl_df, receiver, senders, DOT_N_TOP_GENES, DOT_MIN_SIZE_BY, "6c"
        )
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
        seg_pairs += [(sender, receiver, spec["genes"]) for sender in senders]
    dots = pd.concat(dot_tables, ignore_index=True)

    seg = C.segmentation_tests(
        adata,
        labels_key,
        seg_pairs,
        near_threshold=20.0 / UM_PER_PX,
        far_threshold=100.0 / UM_PER_PX,
    )
    dots = dots.merge(seg, on=["receiver", "sender", "gene"], how="left")
    # A bold label spans the gene's whole column, so the gene is bold when it passes for any of the
    # senders drawn above it, as in Fig. 4c and Fig. 5c.
    dots["bold_in_figure"] = (
        dots.groupby(["receiver", "gene"])["passes_segmentation_test_p_lt_0.05"].transform("any").fillna(False)
    )
    for receiver, group in dots.groupby("receiver", sort=False):
        bold = sorted(set(group.loc[group["bold_in_figure"], "gene"]))
        print(f"  6c {receiver}: bold {len(bold)}/{group['gene'].nunique()} -> {bold}")

    # ---- 6d: head-resolved dot plots ---------------------------------------------------------------
    head_tables = []
    for head, genes in HEAD_DOTPLOTS.items():
        head_abl = model.get_neighbor_ablation_scores(
            adata=adata,
            cell_type=HEAD_DOTPLOT_RECEIVER,
            head_idx=head,
            ablated_neighbor_ct_sub=HEAD_DOTPLOT_SENDERS,
            compute_z_value=True,
        )
        head_df = head_abl._ablation_scores_df
        # The analysis draws this one by spreading the same dotplot_kwargs, so the segmentation
        # exclusion applies here as well as in 6c.
        selected = select_dotplot_genes_drawn(
            adata, labels_key, head_df, HEAD_DOTPLOT_RECEIVER, HEAD_DOTPLOT_SENDERS,
            DOT_N_TOP_GENES, DOT_MIN_SIZE_BY, f"6d head {head}",
        )
        union = []
        for sender in HEAD_DOTPLOT_SENDERS:
            union += [g for g in selected[sender] if g not in union]
        C.check_gene_list(f"6d head {head} (union of senders)", union, genes)
        tab = C.receiver_gene_table(head_df, HEAD_DOTPLOT_RECEIVER, HEAD_DOTPLOT_SENDERS, head=head)
        table = C.dotplot_panel(tab, HEAD_DOTPLOT_RECEIVER, HEAD_DOTPLOT_SENDERS, genes, HEAD_LABELS[head])
        table["highlighted_red_in_figure"] = table["gene"].isin(HIGHLIGHTED_RED)
        head_tables.append(table)
    head_dots = pd.concat(head_tables, ignore_index=True).rename(
        columns={"neighbor_contribution": "head_specific_neighbor_contribution"}
    )

    # ---- 6d: length-scale distributions ------------------------------------------------------------
    cf = model.get_counterfactual_attention_patterns(cell_type=HEAD_DOTPLOT_RECEIVER, adata=adata)
    ls = cf._calculate_length_scales(
        head_idxs=LENGTH_SCALE_HEADS,
        sender_types=HEAD_DOTPLOT_SENDERS,
        attention_threshold=0.1,
        sample_threshold=0.01,
    )
    ls = ls.rename(columns={"length_scale": "length_scale_px", "neighbor_idx": "sender_cell_id", "sender_type": "sender"})
    ls["length_scale_um"] = ls["length_scale_px"] * PIXEL_SIZE_UM
    ls.insert(0, "receiver", HEAD_DOTPLOT_RECEIVER)
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
            "pair. segmentation_* columns: one-sided Mann-Whitney U segmentation-artifact test "
            "(segmentation_p_max = larger of the two p-values; passes when below 0.05 per the Methods, or 0.1 as "
            "in the analysis scripts), run at 20/100um converted to the CosMx pixel coordinates this dataset "
            "stores. bold_in_figure is the bolding the test implies: a gene is bold when it passes for any of "
            "the senders drawn above it.",
            "df": dots,
        },
        {
            "sheet": "Fig6d_head_dotplots",
            "description": "Head-resolved dot plot for CD4 T cell receivers and fibroblast senders: neighbor "
            "contribution with only head 4 active (colour) and its Wald statistic (size). "
            "highlighted_red_in_figure marks CCR7, the gene set in red in the figure.",
            "df": head_dots,
        },
        {
            "sheet": "Fig6d_length_scales",
            "description": "Per-sender-cell effective pairwise radius for CD4 T cell receivers and fibroblast "
            "senders on head 4 (attention threshold 0.1), in CosMx pixels and um. One row per sender cell, "
            "which is one point of the box plot.",
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
