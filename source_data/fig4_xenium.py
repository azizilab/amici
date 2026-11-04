"""Source Data for Figure 4 (Xenium breast cancer, both replicates).

Panels
  4a  Spatial cell-type maps of replicates 1 and 2 with the held-out test regions.
  4b  Directed interaction graphs: all cell types (left) and the immune/tumor subset (right).
  4c  Downstream gene dot plots for M1 macrophage, CD8 T cell and invasive tumor receivers, with the
      segmentation-artifact test that decides which genes are bold.
  4d  Length-scale distributions per sender -> receiver (head) and the head-resolved CD4 -> CD8 dot plot.
  4e  Invasive tumor cells of replicate 1: proximity to any / attending CD8 T cell, total CD8 attention on
      the maximum-variance head, and ESR1 expression.
  4f  Communication hub assignments and the hub sender/receiver compositions shown as Sankey diagrams.

Checkpoint: wandb sweep g3mucw4s, run te7pkv3z (see source_data/checkpoints.yaml).
The computations mirror xenium/xenium_analysis.py, xenium/xenium_spatial_analysis.py,
xenium/xenium_niche_analysis.py and xenium/segmentation_analysis.py.
Run from the repository root:  python source_data/fig4_xenium.py
"""

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
from einops import rearrange
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

import _common as C

NAME = "xenium"
FIGURE = "Fig4"

DISPLAY_NAMES = {
    "CD8+_T_Cells": "CD8 T Cells",
    "CD4+_T_Cells": "CD4 T Cells",
    "DCIS_1": "DCIS 1",
    "DCIS_2": "DCIS 2",
    "IRF7+_DCs": "IRF7+ DCs",
    "LAMP3+_DCs": "LAMP3+ DCs",
    "Macrophages_1": "M1 Macrophages",
    "Macrophages_2": "M2 Macrophages",
    "Myoepi_ACTA2+": "Myoepi ACTA2+",
    "Myoepi_KRT15+": "Myoepi KRT15+",
    "Invasive_Tumor": "Invasive Tumor",
    "B_Cells": "B Cells",
    "Mast_Cells": "Mast Cells",
    "Perivascular-Like": "Perivascular Like",
}
REPLICATE_NAMES = {"0": "Replicate 1", "1": "Replicate 2"}

GRAPH_QUANTILE = 0.80
GRAPH_SUBSET = [
    "Invasive_Tumor",
    "DCIS_1",
    "DCIS_2",
    "Macrophages_2",
    "Macrophages_1",
    "CD8+_T_Cells",
    "CD4+_T_Cells",
    "B_Cells",
    "IRF7+_DCs",
    "LAMP3+_DCs",
]

# Panel 4c, exactly as drawn: every receiver shows the same three sender rows.
DOT_SENDERS = ["CD4+_T_Cells", "CD8+_T_Cells", "Invasive_Tumor"]
DOTPLOTS = [
    {
        "receiver": "Macrophages_1",
        "genes": {
            "CD8+_T_Cells": ["WARS", "SLAMF7", "C15orf48"],
            "Invasive_Tumor": ["APOC1", "FCGR3A", "SCD", "KRT7", "PPARG", "FOXA1", "FASN"],
        },
        "bold": ["WARS", "APOC1", "FCGR3A", "PPARG"],  # C15orf48 fails the receiver comparison (p = 0.98)
    },
    {
        "receiver": "CD8+_T_Cells",
        "genes": {
            "CD4+_T_Cells": ["GPR183", "SELL", "TCF7", "CCR7", "LTB", "IL7R"],
            "Invasive_Tumor": ["APOC1"],
        },
        "bold": [],
    },
    {
        "receiver": "Invasive_Tumor",
        "genes": {"CD8+_T_Cells": ["AGR3", "ESR1", "SERPINA3"]},
        "bold": ["AGR3", "ESR1", "SERPINA3"],
    },
]
DOT_N_TOP_GENES = 10
DOT_MIN_SIZE_BY = -10

# Segmentation-artifact test of xenium/segmentation_analysis.py (same pair order, so the same seeds).
SEGMENTATION_PAIRS = [
    ("CD4+_T_Cells", "CD8+_T_Cells", ["GPR183", "SELL", "TCF7", "CCR7", "LTB", "IL7R"]),
    ("Invasive_Tumor", "Macrophages_1", ["APOC1", "FCGR3A", "SCD", "KRT7", "PPARG", "FOXA1", "FASN"]),
    ("CD8+_T_Cells", "Macrophages_1", ["WARS", "SLAMF7", "C15orf48"]),
    ("CD8+_T_Cells", "Invasive_Tumor", ["AGR3", "ESR1", "SERPINA3"]),
]

# Panel 4d (top): interactions and heads of xenium_analysis.py.
LENGTH_SCALE_INTERACTIONS = {
    "CD8+_T_Cells": {"senders": ["Macrophages_2", "CD4+_T_Cells"], "head_idxs": [4, 5], "sample_threshold": 0.02},
    "Macrophages_1": {
        "senders": ["Invasive_Tumor", "CD8+_T_Cells", "DCIS_1", "Macrophages_2"],
        "head_idxs": [4, 5],
        "sample_threshold": 0.02,
    },
    "B_Cells": {"senders": ["CD4+_T_Cells"], "head_idxs": [4, 5], "sample_threshold": 0.02},
    "Invasive_Tumor": {"senders": ["CD8+_T_Cells"], "head_idxs": [5], "sample_threshold": 0.001},
    "CD4+_T_Cells": {"senders": ["B_Cells", "Macrophages_2"], "head_idxs": [5], "sample_threshold": 0.02},
}
PUBLISHED_VIOLIN_ORDER = [
    "CD4+_T_Cells -> CD8+_T_Cells (head 5)",
    "CD4+_T_Cells -> CD8+_T_Cells (head 4)",
    "CD4+_T_Cells -> B_Cells (head 4)",
    "CD8+_T_Cells -> Macrophages_1 (head 4)",
    "B_Cells -> CD4+_T_Cells (head 5)",
    "CD8+_T_Cells -> Invasive_Tumor (head 5)",
    "CD8+_T_Cells -> Macrophages_1 (head 5)",
    "Macrophages_2 -> CD4+_T_Cells (head 5)",
    "CD4+_T_Cells -> B_Cells (head 5)",
    "Macrophages_2 -> CD8+_T_Cells (head 5)",
    "DCIS_1 -> Macrophages_1 (head 5)",
    "Invasive_Tumor -> Macrophages_1 (head 5)",
    "Macrophages_2 -> Macrophages_1 (head 5)",
]

# Panel 4d (bottom): head-resolved CD4 -> CD8 contributions.
HEAD_DOT_RECEIVER, HEAD_DOT_SENDER = "CD8+_T_Cells", "CD4+_T_Cells"
HEAD_DOT_HEADS = [4, 5]
HEAD_DOT_GENES = ["GNLY", "CAVIN2", "SELL", "LTB"]

# Panel 4e: settings of xenium_spatial_analysis.py.
SPATIAL_RECEIVER, SPATIAL_SENDER, SPATIAL_SAMPLE = "Invasive_Tumor", "CD8+_T_Cells", "0"
SPATIAL_SENDER_QUANTILE = 0.95
SPATIAL_GENE = "ESR1"
PUBLISHED_MAX_VAR_HEAD = 2

# Panel 4f: settings of xenium_niche_analysis.py (k = 10 is the silhouette optimum reported in Fig. S13a).
HUB_QUANTILE, HUB_K = 0.9, 10
HUB_CACHE = "xenium/data/xenium_sample1/hub_analysis_cache/hub_cluster_assignments.csv"
# The two Sankey diagrams are titled with the legend numbering (index + 1): hub 6 has B-cell-dominated
# senders and hub 9 M1-macrophage senders into invasive tumor (xenium_niche_analysis.py sets hub_cluster = 9,
# which would be legend hub 10).
SANKEY_HUB_INDICES = {5: "Communication hub 6", 8: "Communication hub 9"}


def display(label):
    return DISPLAY_NAMES.get(label, label)


def distance_to_scale(dist):
    """Proximity score of xenium_spatial_analysis.py: logistic falloff in distance, 0 when unreachable."""
    if dist == np.inf:
        return 0
    return 1 / (1 + np.exp(dist / 25 - 5))


def main():
    C.ensure_dirs()
    entry = C.CHECKPOINTS[NAME]
    C.verify_checkpoint(NAME)
    pl.seed_everything(entry["seed"])
    labels_key = entry["labels_key"]

    adata = sc.read_h5ad(C.REPO / entry["data"])
    model = C.load_model(NAME, adata)
    print(adata.obs["sample"].value_counts())

    panels = []

    # ---- 4a: spatial maps --------------------------------------------------------------------------
    coords = adata.obsm["spatial"]
    spatial = pd.DataFrame(
        {
            "cell_id": adata.obs_names,
            "replicate": adata.obs["sample"].astype(str).map(REPLICATE_NAMES).values,
            "x_um": coords["X"].values,
            "y_um": coords["Y"].values,
            "cell_type_model_label": adata.obs[labels_key].astype(str).values,
            "cell_type_display": adata.obs[labels_key].astype(str).map(display).values,
            "train_test_split": adata.obs["train_test_split"].astype(str).values,
        }
    )

    # ---- 4b: interaction graphs --------------------------------------------------------------------
    ablation = C.load_or_compute_all_ablation(NAME, model, adata)
    abl_df = ablation._ablation_scores_df
    weight_df = ablation._get_interaction_weight_matrix()
    edges_all = C.interaction_edges(weight_df, list(weight_df.index), GRAPH_QUANTILE)
    edges_all.insert(0, "graph", "4b left: all cell types")
    edges_sub = C.interaction_edges(weight_df, GRAPH_SUBSET, GRAPH_QUANTILE)
    edges_sub.insert(0, "graph", "4b right: immune and tumor subset")
    edges = pd.concat([edges_all, edges_sub], ignore_index=True)
    for col in ["sender", "receiver"]:
        edges[f"{col}_display"] = edges[col].map(display)
    print(f"  Fig 4b: {edges_all['edge_drawn'].sum()} edges (all), {edges_sub['edge_drawn'].sum()} edges (subset)")
    full_matrix = weight_df.copy()
    full_matrix.index.name = "sender \\ receiver"

    # ---- 4c: dot plots and segmentation test -------------------------------------------------------
    seg = C.segmentation_tests(adata, labels_key, SEGMENTATION_PAIRS)
    dot_tables = []
    for spec in DOTPLOTS:
        receiver = spec["receiver"]
        published = [g for s in DOT_SENDERS for g in spec["genes"].get(s, [])]
        selected = C.select_dotplot_genes(abl_df, receiver, DOT_SENDERS, DOT_N_TOP_GENES, DOT_MIN_SIZE_BY)
        for sender, genes in spec["genes"].items():
            C.check_gene_list(f"4c receiver {receiver} sender {sender}", selected.get(sender, []), genes)

        senders = [s for s in DOT_SENDERS if s != receiver]
        fresh = model.get_neighbor_ablation_scores(
            adata=adata, cell_type=receiver, ablated_neighbor_ct_sub=senders, compute_z_value=True
        )
        cached = C.receiver_gene_table(abl_df, receiver, senders)
        recomputed = C.receiver_gene_table(fresh._ablation_scores_df, receiver, senders)
        merged = cached.merge(recomputed, on=["receiver", "sender", "gene"], suffixes=("", "_fresh"))
        print(
            f"  4c {receiver}: max |diff| cache vs fresh = "
            f"{np.nanmax(np.abs(merged['neighbor_contribution'] - merged['neighbor_contribution_fresh'])):.2e}, "
            f"max |z| = {np.nanmax(np.abs(merged['wald_z'] - merged['wald_z_fresh'])):.2e}"
        )

        table = C.dotplot_panel(recomputed, receiver, DOT_SENDERS, published, f"4c (Receiver: {display(receiver)})")
        table["gene_selected_for_sender"] = [
            g in spec["genes"].get(s, []) for s, g in zip(table["sender"], table["gene"], strict=True)
        ]
        table = table.merge(seg, on=["receiver", "sender", "gene"], how="left")
        table["bold_in_figure"] = table["gene"].isin(spec["bold"])
        dot_tables.append(table)
    dots = pd.concat(dot_tables, ignore_index=True)

    C.report_bold_agreement(dots, "4c")
    for col in ["receiver", "sender"]:
        dots[col] = dots[col].map(display)

    # ---- 4d (top): length-scale violins ------------------------------------------------------------
    ls_dfs = []
    for receiver, cfg in LENGTH_SCALE_INTERACTIONS.items():
        cf = model.get_counterfactual_attention_patterns(cell_type=receiver, adata=adata)
        ls = cf._calculate_length_scales(
            head_idxs=cfg["head_idxs"],
            sender_types=cfg["senders"],
            attention_threshold=0.1,
            sample_threshold=cfg["sample_threshold"],
        )
        kept = []
        for sender in cfg["senders"]:
            sender_df = ls[ls["sender_type"] == sender].copy()
            medians = sender_df.groupby("head_idx")["length_scale"].median()
            heads = medians[(medians > 0) & (medians <= 50)].index.tolist()
            kept.append(sender_df[sender_df["head_idx"].isin(heads)])
        kept = pd.concat(kept)
        kept["receiver_type"] = receiver
        kept["interaction"] = (
            kept["sender_type"] + " -> " + kept["receiver_type"] + " (head " + kept["head_idx"].astype(str) + ")"
        )
        ls_dfs.append(kept)
    ls_df = pd.concat(ls_dfs, ignore_index=True)
    order = ls_df.groupby("interaction")["length_scale"].median().sort_values().index.tolist()
    C.check_gene_list("4d violin rows (ascending median)", order, PUBLISHED_VIOLIN_ORDER)
    rank = {k: i + 1 for i, k in enumerate(order)}
    ls_long = pd.DataFrame(
        {
            "row_order_top_to_bottom": ls_df["interaction"].map(rank),
            "interaction": ls_df["interaction"].map(
                lambda s: " -> ".join(display(p) for p in s.split(" (")[0].split(" -> ")) + " (" + s.split(" (")[1]
            ),
            "sender": ls_df["sender_type"].map(display),
            "receiver": ls_df["receiver_type"].map(display),
            "head": ls_df["head_idx"],
            "sender_cell_id": ls_df["neighbor_idx"].values,
            "length_scale_um": ls_df["length_scale"].values,
        }
    ).sort_values(["row_order_top_to_bottom", "sender_cell_id"])
    ls_summary = (
        ls_long.groupby(["row_order_top_to_bottom", "interaction"])["length_scale_um"]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .reset_index()
        .rename(columns={"count": "n_sender_cells", "50%": "median", "25%": "q1", "75%": "q3"})
    )

    # ---- 4d (bottom): head-resolved CD4 -> CD8 dot plot --------------------------------------------
    head_tables = []
    for head in HEAD_DOT_HEADS:
        head_abl = model.get_neighbor_ablation_scores(
            adata=adata,
            cell_type=HEAD_DOT_RECEIVER,
            head_idx=head,
            ablated_neighbor_ct_sub=[HEAD_DOT_SENDER],
            compute_z_value=True,
        )
        tab = C.receiver_gene_table(head_abl._ablation_scores_df, HEAD_DOT_RECEIVER, [HEAD_DOT_SENDER], head=head)
        head_tables.append(tab[tab["gene"].isin(HEAD_DOT_GENES)])
    head_dots = pd.concat(head_tables, ignore_index=True)
    head_dots["gene"] = pd.Categorical(head_dots["gene"], HEAD_DOT_GENES, ordered=True)
    head_dots = head_dots.sort_values(["head", "gene"]).rename(
        columns={"neighbor_contribution": "head_specific_neighbor_contribution"}
    )
    head_dots["receiver"] = head_dots["receiver"].map(display)
    head_dots["sender"] = head_dots["sender"].map(display)

    # ---- 4e: replicate-1 invasive tumor spatial panels (xenium_spatial_analysis.py) ----------------
    adata_s = adata[adata.obs["sample"] == SPATIAL_SAMPLE].copy()
    receiver_attention = model.get_attention_patterns(
        indices=(adata.obs[labels_key] == SPATIAL_RECEIVER) & (adata.obs["sample"] == SPATIAL_SAMPLE),
    )
    heads_df = receiver_attention._attention_patterns_df
    nn_idxs_df = receiver_attention._nn_idxs_df
    neighbor_cols = [c for c in heads_df.columns if c.startswith("neighbor_")]
    neighbor_cell_types = pd.DataFrame(
        data=rearrange(
            adata_s[rearrange(nn_idxs_df.values, "b n -> (b n)")].obs[labels_key].values.astype(str),
            "(b n) -> b n",
            b=nn_idxs_df.shape[0],
        ),
        index=nn_idxs_df.index,
        columns=nn_idxs_df.columns,
    )
    head_ct_attention = pd.DataFrame(
        np.where(
            neighbor_cell_types.loc[heads_df["cell_idx"]].to_numpy() == SPATIAL_SENDER,
            heads_df[neighbor_cols].to_numpy(),
            0.0,
        ).sum(axis=1),
        columns=["attention_from_sender"],
    )
    head_ct_attention["cell_idx"] = heads_df["cell_idx"].values
    head_ct_attention["head"] = heads_df["head"].values

    # Max attention each CD8 T cell receives from any invasive tumor receiver, used to define the
    # "attending" (high-interacting) senders.
    neighbor_long = nn_idxs_df.reset_index().melt(
        id_vars=["index"], var_name="neighbor_position", value_name="neighbor_cell_idx"
    )
    neighbor_long["neighbor_pos"] = neighbor_long["neighbor_position"].str.extract(r"(\d+)").astype(int)
    attention_long = heads_df.melt(
        id_vars=["cell_idx", "head"], value_vars=neighbor_cols, var_name="neighbor_position", value_name="attention_score"
    )
    attention_long["neighbor_pos"] = attention_long["neighbor_position"].str.extract(r"(\d+)").astype(int)
    attention_with_neighbors = attention_long.merge(
        neighbor_long[["index", "neighbor_pos", "neighbor_cell_idx"]],
        left_on=["cell_idx", "neighbor_pos"],
        right_on=["index", "neighbor_pos"],
        how="left",
    )
    sender_ids = adata_s[adata_s.obs[labels_key] == SPATIAL_SENDER].obs_names
    attention_to_senders = attention_with_neighbors[attention_with_neighbors["neighbor_cell_idx"].isin(sender_ids)]
    sender_max_attention = (
        attention_to_senders.groupby("neighbor_cell_idx")["attention_score"].max().reindex(sender_ids, fill_value=0.0)
    )
    sender_threshold = sender_max_attention[sender_max_attention > 0].astype(float).quantile(SPATIAL_SENDER_QUANTILE)
    attending_senders = sender_max_attention[sender_max_attention > sender_threshold].index

    max_var_head = (
        attention_to_senders.set_index("neighbor_cell_idx")[["head", "attention_score"]]
        .groupby("head")
        .var()
        .idxmax()
        .values[0]
    )
    print(f"  4e max-variance head = {max_var_head} (published panel: head {PUBLISHED_MAX_VAR_HEAD})")

    receiver_adata = adata_s[adata_s.obs[labels_key] == SPATIAL_RECEIVER].copy()
    nn_idx = np.array(receiver_adata.obsm["_nn_idx"])
    nn_dist = receiver_adata.obsm["_nn_dist"]
    dist_any = np.where(
        np.isin(nn_idx, np.where(np.isin(adata.obs_names, sender_ids))[0]), nn_dist, np.inf
    ).min(axis=1)
    dist_attending = np.where(
        np.isin(nn_idx, np.where(np.isin(adata.obs_names, attending_senders))[0]), nn_dist, np.inf
    ).min(axis=1)
    head_attention = (
        head_ct_attention[head_ct_attention["head"] == max_var_head].groupby("cell_idx")["attention_from_sender"].max()
    )
    esr1 = receiver_adata.X[:, receiver_adata.var_names.get_loc(SPATIAL_GENE)]
    esr1 = np.asarray(esr1.toarray() if sparse.issparse(esr1) else esr1).reshape(-1)
    rcoords = receiver_adata.obsm["spatial"]
    spatial_4e = pd.DataFrame(
        {
            "cell_id": receiver_adata.obs_names,
            "x_um": rcoords["X"].values,
            "y_um": rcoords["Y"].values,
            "distance_to_nearest_CD8_um": dist_any,
            "proximity_score_any_CD8": [distance_to_scale(d) for d in dist_any],
            "distance_to_nearest_attending_CD8_um": dist_attending,
            "proximity_score_attending_CD8": [distance_to_scale(d) for d in dist_attending],
            f"total_CD8_attention_head_{max_var_head}": receiver_adata.obs_names.map(head_attention).fillna(0).values,
            f"{SPATIAL_GENE}_expression": esr1,
        }
    )
    spatial_4e = spatial_4e.replace({np.inf: np.nan})
    attending_df = pd.DataFrame(
        {
            "cd8_cell_id": sender_ids,
            "max_attention_from_invasive_tumor": sender_max_attention.values,
            "attending_CD8": sender_ids.isin(attending_senders),
        }
    )

    # ---- 4f: communication hubs ----------------------------------------------------------------------
    attention_all = model.get_attention_patterns()
    hubs = attention_all.compute_communication_hubs(
        attention_quantile_threshold=HUB_QUANTILE, n_clusters=HUB_K, random_state=entry["seed"]
    )
    cache_path = C.REPO / HUB_CACHE
    if cache_path.exists():
        cached = pd.read_csv(cache_path, index_col=0)["amici_hub_cluster"]
        common = cached.index.intersection(hubs.index)
        same = (cached.loc[common].values == hubs.loc[common, "hub_cluster"].values).mean()
        ari = adjusted_rand_score(cached.loc[common], hubs.loc[common, "hub_cluster"])
        print(f"  4f hubs vs cached assignments: identical labels {same:.4f}, ARI {ari:.4f} ({len(common)} cells)")

    hubs["receiver_label"] = adata.obs[labels_key].loc[hubs.index]
    sender_comp = hubs.drop(columns=["receiver_label"]).groupby("hub_cluster").mean()
    sender_comp = sender_comp.div(sender_comp.sum(axis=1), axis=0).fillna(0)
    receiver_comp = (
        pd.DataFrame(hubs[["receiver_label", "hub_cluster"]].groupby("hub_cluster").value_counts())
        .reset_index()
        .pivot(columns="receiver_label", index="hub_cluster", values="count")
    )
    receiver_comp = receiver_comp.div(receiver_comp.sum(axis=1), axis=0).fillna(0)

    flows = []
    cell_types = list(adata.obs[labels_key].unique())
    for hub in sorted(hubs["hub_cluster"].unique()):
        sub = hubs[hubs["hub_cluster"] == hub]
        melted = pd.melt(
            sub.reset_index(),
            id_vars=["receiver_label", "index"],
            value_vars=cell_types,
            var_name="neighbor_label",
            value_name="proportion",
        )
        props = melted[["receiver_label", "neighbor_label", "proportion"]].groupby(
            ["receiver_label", "neighbor_label"]
        ).mean().reset_index()
        props = props[props["receiver_label"] != props["neighbor_label"]]
        props["receiver_weight"] = props["receiver_label"].map(receiver_comp.loc[hub]).astype(float)
        props["weighted_proportion"] = props["proportion"] * props["receiver_weight"]
        props = props[
            props["receiver_label"].isin(receiver_comp.loc[hub].dropna().index)
            & props["neighbor_label"].isin(sender_comp.loc[hub].dropna().index)
            & (props["weighted_proportion"] > 0)
        ]
        props.insert(0, "hub_cluster_index", int(hub))
        flows.append(props)
    flows = pd.concat(flows, ignore_index=True)
    flows.insert(1, "hub_legend_label", "Hub " + (flows["hub_cluster_index"] + 1).astype(str))
    flows = flows.rename(columns={"neighbor_label": "sender", "receiver_label": "receiver"})

    # Identify the hubs drawn as Sankey diagrams by their composition.
    summary = []
    for hub, grp in flows.groupby("hub_cluster_index"):
        top_sender = grp.groupby("sender")["weighted_proportion"].sum().idxmax()
        top_receiver = receiver_comp.loc[hub].idxmax()
        summary.append(
            {
                "hub_cluster_index": hub,
                "hub_legend_label": f"Hub {hub + 1}",
                "n_cells": int((hubs["hub_cluster"] == hub).sum()),
                "top_sender": top_sender,
                "top_sender_share": grp.groupby("sender")["weighted_proportion"].sum().max()
                / grp["weighted_proportion"].sum(),
                "top_receiver": top_receiver,
                "top_receiver_share": receiver_comp.loc[hub].max(),
            }
        )
    summary = pd.DataFrame(summary)
    summary["shown_in_fig4f_sankey"] = summary["hub_cluster_index"].map(SANKEY_HUB_INDICES)
    flows["shown_in_fig4f_sankey"] = flows["hub_cluster_index"].map(SANKEY_HUB_INDICES)
    print(summary.to_string(index=False))
    for col in ["sender", "receiver"]:
        flows[f"{col}_display"] = flows[col].map(display)

    hub_spatial = spatial[["cell_id", "replicate", "x_um", "y_um", "cell_type_model_label"]].copy()
    hub_spatial["hub_cluster_index"] = hubs.loc[adata.obs_names, "hub_cluster"].values
    hub_spatial["hub_legend_label"] = "Hub " + (hub_spatial["hub_cluster_index"] + 1).astype(str)

    panels += [
        {
            "sheet": "Fig4a",
            "description": "Spatial cell-type maps of both replicates: one row per cell with coordinates (um), "
            "replicate, cell type and train/test split (test cells form the dashed boxes).",
            "df": spatial,
        },
        {
            "sheet": "Fig4b",
            "description": "Directed interaction graphs. Edge weight = number of receiver genes with BH-adjusted "
            "p < 0.05 and positive neighbor contribution when the sender type is ablated; edges are drawn when "
            f"the weight is at least the {GRAPH_QUANTILE} quantile of the full matrix (weight_threshold).",
            "df": edges,
        },
        {
            "sheet": "Fig4b_full_matrix",
            "description": "Full sender x receiver interaction weight matrix underlying Fig. 4b.",
            "df": full_matrix,
            "index": True,
        },
        {
            "sheet": "Fig4c",
            "description": "Dot plots of downstream genes. Colour = neighbor contribution (normalized expression "
            "units, scale clipped to +/-0.5); size = Wald test statistic (wald_z). p_value / p_adj_BH are the "
            "one-sided p of wald_z and its BH adjustment over all genes of the receiver/sender pair. The "
            "segmentation_* columns are the one-sided Mann-Whitney U test of xenium/segmentation_analysis.py; "
            "segmentation_p_max is the larger of its two one-sided p-values; a gene passes (is not attributed to segmentation artifacts) when it is below 0.05 (Methods) or 0.1 (threshold in the analysis scripts). bold_in_figure records the published bolding.",
            "df": dots,
        },
        {
            "sheet": "Fig4d_length_scales",
            "description": "Per-sender-cell effective pairwise radius (length scale, um) for each sender -> receiver "
            "(head) row of the violin plot, attention threshold 0.1, heads with median in (0, 50] um. "
            "row_order_top_to_bottom follows ascending median.",
            "df": ls_long,
        },
        {
            "sheet": "Fig4d_length_scale_summary",
            "description": "Summary statistics of each violin in Fig. 4d (um).",
            "df": ls_summary,
        },
        {
            "sheet": "Fig4d_head_dotplot",
            "description": "Head-resolved CD4 T cell -> CD8 T cell dot plot: neighbor contribution with only the "
            "given head active (colour, scale +/-0.01) and its Wald statistic (size).",
            "df": head_dots,
        },
        {
            "sheet": "Fig4e",
            "description": "Invasive tumor cells of replicate 1. Proximity score = 1/(1+exp(d/25-5)) of the "
            "distance d (um) to the nearest CD8 T cell among the 50 nearest neighbors (0 if none); 'attending' "
            f"CD8 T cells receive max attention above the {SPATIAL_SENDER_QUANTILE} quantile (sheet "
            f"Fig4e_attending_CD8). Total attention is summed over CD8 neighbors on head {max_var_head}, the "
            "head with the largest variance of attention to CD8 T cells. ESR1 is normalized expression.",
            "df": spatial_4e,
        },
        {
            "sheet": "Fig4e_attending_CD8",
            "description": "CD8 T cells of replicate 1 with the maximum attention they receive from any invasive "
            "tumor receiver, and whether they pass the attending threshold.",
            "df": attending_df,
        },
        {
            "sheet": "Fig4f_hubs",
            "description": f"Communication hub of every cell (KMeans, k = {HUB_K}, attention quantile "
            f"{HUB_QUANTILE}). hub_legend_label = hub_cluster_index + 1, as in the figure legend.",
            "df": hub_spatial,
        },
        {
            "sheet": "Fig4f_sankey_flows",
            "description": "Sender -> receiver flows of every hub: weighted_proportion = mean fraction of "
            "high-attention neighbors of the sender type among receivers of the receiver type, weighted by the "
            "receiver's share of the hub. The bottom of Fig. 4f shows two of these hubs (sheet Fig4f_hub_summary).",
            "df": flows,
        },
        {
            "sheet": "Fig4f_hub_summary",
            "description": "Size and dominant sender / receiver of each hub, used to identify the hubs drawn in "
            "the Sankey diagrams.",
            "df": summary,
        },
    ]
    readme = [
        ("Figure", "Figure 4: AMICI on Xenium breast cancer"),
        ("Fig. 4a", "Sheet Fig4a"),
        ("Fig. 4b", "Sheets Fig4b and Fig4b_full_matrix"),
        ("Fig. 4c", "Sheet Fig4c"),
        ("Fig. 4d", "Sheets Fig4d_length_scales, Fig4d_length_scale_summary, Fig4d_head_dotplot"),
        ("Fig. 4e", "Sheets Fig4e and Fig4e_attending_CD8"),
        ("Fig. 4f", "Sheets Fig4f_hubs, Fig4f_sankey_flows, Fig4f_hub_summary"),
    ]
    C.save_panels(
        FIGURE,
        panels,
        readme,
        C.provenance_for(
            NAME,
            {
                "graph_quantile": GRAPH_QUANTILE,
                "max_variance_head": int(max_var_head),
                "hub_k": HUB_K,
                "hub_quantile": HUB_QUANTILE,
            },
        ),
    )


if __name__ == "__main__":
    main()
