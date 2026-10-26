"""Shared helpers for the Source Data exporters.

Every exporter loads the exact dataset and AMICI checkpoint that produced its figure (listed in
``CHECKPOINTS`` below and in ``checkpoints.yaml``), verifies both by md5 before use, recomputes the
quantities drawn in each panel, and writes one Excel workbook per figure with one sheet per panel.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "source_data" / "output"
PANEL_DIR = REPO / "source_data" / "panels"

# Dataset and checkpoint used for each real-data figure. Paths are relative to the repository root.
# md5 sums were taken from the files that generated the manuscript panels; the remote copies are the
# Figshare record (doi:10.6084/m9.figshare.31927956, v1) and the lab bucket gs://justin-amici.
CHECKPOINTS = {
    "cortex": {
        "figures": ["Fig. 3a-c", "Fig. S4"],
        "data": "cortex/data/cortex_processed_2025-04-28.h5ad",
        "data_md5": "5f9081deef1132c89c09ae7f6819264a",
        "model": "cortex/saved_models/cortex_sweep_2025-04-28_model_2025-05-05/"
        "cortex_18_sweep_plm73bmg_xrtcnlt0_params_2025-05-05",
        "model_md5": "76eadfaa20920ae7b9a4044d9094512e",
        "wandb_sweep": "plm73bmg",
        "wandb_run": "xrtcnlt0",
        "config": "reproducibility/cortex_config.yaml",
        "labels_key": "subclass",
        "seed": 18,
        "figshare_model_file_id": 63409695,
        "figshare_data_file_id": 63409731,
        "gcs_model": "gs://justin-amici/cortex/saved_models/cortex_18_sweep_plm73bmg_xrtcnlt0_params_2025-05-05/model.pt",
        "gcs_data": "gs://justin-amici/cortex/data/cortex_processed_2025-04-28.h5ad",
    },
    "xenium": {
        "figures": ["Fig. 4a-f", "Figs. S5c, S7, S9, S13-S18"],
        "data": "xenium/data/xenium_sample1/xenium_sample1_filtered_2025-05-01.h5ad",
        "data_md5": "670168c3a8a8816a075d0d619372dd5b",
        "model": "xenium/saved_models/xenium_sample1_proseg_sweep_2025-05-01_model_2025-05-02/"
        "xenium_18_sweep_g3mucw4s_te7pkv3z_params_2025-05-02",
        "model_md5": "c118e838b4ff896245d21000a4a04f7e",
        "wandb_sweep": "g3mucw4s",
        "wandb_run": "te7pkv3z",
        "config": "reproducibility/xenium_full_config.yaml",
        "labels_key": "celltype_train_grouped",
        "seed": 18,
        "figshare_model_file_id": 63409710,
        "figshare_data_file_id": 63409728,
        "gcs_model": "gs://justin-amici/xenium/saved_models/xenium_18_sweep_g3mucw4s_te7pkv3z_params_2025-05-02/model.pt",
        "gcs_data": "gs://justin-amici/xenium/data/xenium_sample1_filtered_2025-05-01.h5ad",
    },
    "atera_breast": {
        "figures": ["Fig. 5a-c", "Fig. S38"],
        "data": "atera_breast/data/atera_breast_filtered_2026-07-22.h5ad",
        "data_md5": "aba4c26e3119d816f55b32a7e9e00fd1",
        "model": "atera_breast/saved_models/atera_breast_sweep_2026-07-22_model_2026-07-28/"
        "atera_breast_33_sweep_25xpxkuk_zhtraubt_params_2026-07-28",
        "model_md5": "f53e6ab53c322735193998ecc2db959a",
        "wandb_sweep": "25xpxkuk",
        "wandb_run": "zhtraubt",
        "config": "reproducibility/atera_breast_config.yaml",
        "labels_key": "celltype_train_grouped",
        "seed": 38,
        "figshare_model_file_id": None,
        "figshare_data_file_id": None,
        "gcs_model": None,
        "gcs_data": "gs://justin-amici/atera_breast/data/atera_breast_filtered_2026-07-22.h5ad",
    },
    "human_tonsil": {
        "figures": ["Fig. 6a-d"],
        "data": "human_tonsil/data/human_tonsil_filtered_2026-07-25.h5ad",
        "data_md5": "8a7e318a95e517942fda74a584eeb597",
        "model": "human_tonsil/saved_models/human_tonsil_sweep_2026-07-25_model_2026-07-26/"
        "human_tonsil_40_sweep_0f9rk0na_4303v0zu_params_2026-07-26",
        "model_md5": "e12cc39256feed3781ef711cc463b07a",
        "wandb_sweep": "0f9rk0na",
        "wandb_run": "4303v0zu",
        "config": "reproducibility/human_tonsil_config.yaml",
        "labels_key": "celltype_manual_fine",
        "seed": 40,
        "figshare_model_file_id": None,
        "figshare_data_file_id": None,
        "gcs_model": None,
        "gcs_data": None,
    },
}

ABLATION_CACHES = {
    "xenium": "xenium/figures/cached_ablation_scores/all_cell_types_ablation_scores.pkl",
    "atera_breast": "atera_breast/figures/atera_breast_analysis/cached_ablation_scores/all_cell_types_ablation_scores.pkl",
    "human_tonsil": "human_tonsil/figures/human_tonsil_analysis/cached_ablation_scores/all_cell_types_ablation_scores.pkl",
    "cortex": "cortex/figures/cached_ablation_scores/all_cell_types_ablation_scores.pkl",
}


def md5sum(path, chunk_size=1 << 24):
    """Return the hex md5 of a file."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(name, check_data=True):
    """Assert that the dataset and model on disk are the ones recorded for this figure."""
    entry = CHECKPOINTS[name]
    model_file = REPO / entry["model"] / "model.pt"
    if not model_file.exists():
        raise FileNotFoundError(
            f"{model_file} is missing. Download it from {entry['gcs_model'] or 'the Figshare record'} "
            "(see source_data/checkpoints.yaml)."
        )
    got = md5sum(model_file)
    if got != entry["model_md5"]:
        raise ValueError(f"{model_file} has md5 {got}, expected {entry['model_md5']}")
    if check_data:
        got = md5sum(REPO / entry["data"])
        if got != entry["data_md5"]:
            raise ValueError(f"{entry['data']} has md5 {got}, expected {entry['data_md5']}")
    print(f"[{name}] checkpoint and data verified by md5")


def load_model(name, adata):
    """Load the recorded AMICI checkpoint for a dataset, mirroring the analysis scripts."""
    from amici import AMICI

    entry = CHECKPOINTS[name]
    model = AMICI.load(str(REPO / entry["model"]), adata=adata)
    AMICI.setup_anndata(
        adata,
        labels_key=entry["labels_key"],
        coord_obsm_key="spatial",
        n_neighbors=50,
    )
    return model


def load_or_compute_all_ablation(name, model, adata):
    """Load the cached all-cell-type ablation scores, computing (and caching) them if absent."""
    from amici.interpretation._ablation_module import AMICIAblationModule

    cache = REPO / ABLATION_CACHES[name]
    if cache.exists():
        print(f"[{name}] loading cached ablation scores from {cache}")
        return AMICIAblationModule.load_object(str(cache))
    print(f"[{name}] computing all-cell-type ablation scores (no cache at {cache})")
    module = model.get_neighbor_ablation_scores(adata=adata, compute_z_value=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    module.save_object(str(cache))
    return module


def add_pvalues(df, z_col="wald_z", group_cols=("receiver", "sender", "head")):
    """Add the one-sided p-value of the Wald z and its BH adjustment within each receiver/sender/head."""
    df = df.copy()
    df["p_value"] = norm.sf(df[z_col].astype(float))
    df["p_adj_BH"] = np.nan
    group_cols = [c for c in group_cols if c in df.columns]
    for _, idx in df.groupby(group_cols, dropna=False).groups.items():
        p = df.loc[idx, "p_value"].to_numpy()
        ok = np.isfinite(p)
        if ok.any():
            adj = np.full_like(p, np.nan)
            adj[ok] = multipletests(p[ok], method="fdr_bh")[1]
            df.loc[idx, "p_adj_BH"] = adj
    return df


def receiver_gene_table(ablation_df, receiver, senders, head=None):
    """Long table of neighbor contribution and Wald z for every gene of one receiver and its senders.

    The BH adjustment runs over all genes of a receiver/sender pair, exactly as in
    ``AMICIAblationModule.compute``.
    """
    sub = ablation_df[ablation_df["cell_type"] == receiver].set_index("gene")
    rows = []
    for sender in senders:
        if sender == receiver or f"{sender}_diff" not in sub.columns:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "receiver": receiver,
                    "sender": sender,
                    "head": "all heads" if head is None else head,
                    "gene": sub.index,
                    "neighbor_contribution": sub[f"{sender}_diff"].to_numpy(),
                    "wald_z": sub[f"{sender}_z_value"].to_numpy(),
                }
            )
        )
    table = add_pvalues(pd.concat(rows, ignore_index=True))
    table["significant_positive"] = (table["neighbor_contribution"] > 0) & (table["p_adj_BH"] < 0.05)
    positive = table["neighbor_contribution"].where(table["neighbor_contribution"] > 0)
    table["rank_by_neighbor_contribution"] = positive.groupby([table["receiver"], table["sender"]]).rank(
        ascending=False, method="first"
    )
    return table


def select_dotplot_genes(ablation_df, receiver, senders, n_top_genes, min_size_by, size_by="z_value"):
    """Reproduce the gene selection of ``AMICIAblationModule.plot_featurewise_contributions_dotplot``."""
    sub = ablation_df[ablation_df["cell_type"] == receiver]
    selected = {}
    for sender in senders:
        if sender == receiver:
            continue
        positive = sub[sub[f"{sender}_diff"] > 0]
        top = positive[f"{sender}_diff"].astype(float).nlargest(n_top_genes)
        keep = top.index[
            (sub.loc[top.index, f"{sender}_{size_by}"] > min_size_by)
            & (sub.loc[top.index, f"{sender}_nl10_pval_adj"] > -np.log10(0.05))
        ]
        selected[sender] = sub.loc[keep, "gene"].astype(str).tolist()
    return selected


def dotplot_panel(gene_table, receiver, senders, genes, panel_label, extra=None):
    """Subset a receiver gene table to the sender x gene grid drawn in a dot plot."""
    grid = pd.MultiIndex.from_product([senders, genes], names=["sender", "gene"]).to_frame(index=False)
    sub = gene_table[gene_table["receiver"] == receiver]
    out = grid.merge(sub, on=["sender", "gene"], how="left")
    out["receiver"] = receiver
    out.insert(0, "panel", panel_label)
    cols = ["panel", "receiver", "sender", "head", "gene", "neighbor_contribution", "wald_z", "p_value", "p_adj_BH"]
    out = out[[c for c in cols if c in out.columns] + [c for c in out.columns if c not in cols]]
    if extra is not None:
        out = out.merge(extra, on=["receiver", "sender", "gene"], how="left")
    return out


def check_gene_list(label, computed, published):
    """Report whether a recomputed gene selection matches the published panel."""
    match = list(computed) == list(published)
    status = "MATCH" if match else "MISMATCH"
    print(f"  [{status}] {label}")
    if not match:
        print(f"      computed : {list(computed)}")
        print(f"      published: {list(published)}")
    return match


def segmentation_tests(adata, labels_key, pairs, near_threshold=20.0, far_threshold=100.0):
    """One-sided Mann-Whitney segmentation-artifact test of xenium/segmentation_analysis.py.

    For each (sender, receiver, genes) pair, in order, receivers within ``near_threshold`` of a sender are
    compared with receivers and senders at least ``far_threshold`` away, after subsampling all three groups
    to the smallest size with ``np.random.seed(42 + pair_idx)``. The reported p is the larger of the two
    one-sided p-values. Per the Methods, a gene passes (is not attributed to segmentation artifacts, and is
    printed bold) when both p-values are below 0.05. The analysis scripts used 0.1 (``significant_both``)
    until 2026-10-26, when every segmentation threshold was set to the 0.05 of the Methods; both calls are
    still exported, since the published panels were bolded under the older code.
    """
    from scipy import sparse, stats
    from scipy.spatial.distance import cdist

    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    coords_all = np.asarray(adata.obsm["spatial"])
    rows = []
    for pair_idx, (sender_type, receiver_type, genes) in enumerate(pairs):
        sender_mask = (adata.obs[labels_key] == sender_type).to_numpy()
        receiver_mask = (adata.obs[labels_key] == receiver_type).to_numpy()
        sender_ids = adata.obs_names[sender_mask]
        receiver_ids = adata.obs_names[receiver_mask]
        sender_coords = coords_all[sender_mask]
        receiver_coords = coords_all[receiver_mask]

        min_to_senders = cdist(receiver_coords, sender_coords).min(axis=1)
        near_receiver_ids = receiver_ids[min_to_senders <= near_threshold]
        far_receiver_ids = receiver_ids[min_to_senders >= far_threshold]
        min_to_receivers = cdist(sender_coords, receiver_coords).min(axis=1)
        far_sender_ids = sender_ids[min_to_receivers >= far_threshold]
        if min(len(near_receiver_ids), len(far_sender_ids), len(far_receiver_ids)) == 0:
            continue

        np.random.seed(42 + pair_idx)
        n = min(len(near_receiver_ids), len(far_sender_ids), len(far_receiver_ids))
        if len(near_receiver_ids) > n:
            near_receiver_ids = near_receiver_ids[np.random.choice(len(near_receiver_ids), n, replace=False)]
        if len(far_sender_ids) > n:
            far_sender_ids = far_sender_ids[np.random.choice(len(far_sender_ids), n, replace=False)]
        if len(far_receiver_ids) > n:
            far_receiver_ids = far_receiver_ids[np.random.choice(len(far_receiver_ids), n, replace=False)]

        gene_idx = [adata.var_names.get_loc(g) for g in genes]
        near = X[adata.obs_names.get_indexer(near_receiver_ids)][:, gene_idx]
        far_s = X[adata.obs_names.get_indexer(far_sender_ids)][:, gene_idx]
        far_r = X[adata.obs_names.get_indexer(far_receiver_ids)][:, gene_idx]
        for i, gene in enumerate(genes):
            if n > 1:
                p_s = stats.mannwhitneyu(near[:, i], far_s[:, i], alternative="greater")[1]
                p_r = stats.mannwhitneyu(near[:, i], far_r[:, i], alternative="greater")[1]
            else:
                p_s = p_r = np.nan
            p = max(p_s, p_r)
            rows.append(
                {
                    "receiver": receiver_type,
                    "sender": sender_type,
                    "gene": gene,
                    "segmentation_test_order": pair_idx,
                    "segmentation_n_cells_per_group": n,
                    "segmentation_p_near_vs_far_sender": p_s,
                    "segmentation_p_near_vs_far_receiver": p_r,
                    "segmentation_p_max": p,
                    "passes_segmentation_test_p_lt_0.05": bool(p < 0.05),
                    "passes_segmentation_test_p_lt_0.1": bool(p < 0.1),
                }
            )
    return pd.DataFrame(rows)


def report_bold_agreement(dots, label):
    """Compare the recomputed segmentation test with the genes printed bold in the figure.

    A gene column is bold when it passes for the sender(s) whose top genes contributed it.
    """
    tested = dots[dots["gene_selected_for_sender"] & dots["segmentation_p_max"].notna()]
    for col in ["passes_segmentation_test_p_lt_0.05", "passes_segmentation_test_p_lt_0.1"]:
        passes = tested.groupby(["receiver", "gene"])[col].all()
        bold = tested.groupby(["receiver", "gene"])["bold_in_figure"].first()
        agree = passes == bold
        print(f"  {label}: test at p < {col.split('_lt_')[1]} reproduces bolding for {agree.sum()}/{len(agree)} genes")
        if not agree.all():
            print(pd.DataFrame({"passes": passes, "bold_in_figure": bold})[~agree].to_string())
    return tested


def interaction_edges(weight_df, cell_types, quantile):
    """Edge list of the directed interaction graph, mirroring ``plot_interaction_directed_graph``.

    ``weight_df`` has senders as rows and receivers as columns; an edge sender -> receiver is drawn when
    its weight (number of genes with BH-adjusted p < 0.05 and positive neighbor contribution) is at least
    the given quantile of all weights in the full matrix.
    """
    threshold = float(np.quantile(weight_df.values.flatten(), quantile))
    rows = []
    for sender in cell_types:
        for receiver in cell_types:
            if sender == receiver:
                continue
            weight = float(weight_df.loc[sender, receiver])
            rows.append(
                {
                    "sender": sender,
                    "receiver": receiver,
                    "n_significant_genes": weight,
                    "edge_drawn": weight >= threshold,
                }
            )
    edges = pd.DataFrame(rows)
    max_weight = edges.loc[edges["edge_drawn"], "n_significant_genes"].max()
    edges["relative_edge_width"] = np.where(edges["edge_drawn"], edges["n_significant_genes"] / max_weight, np.nan)
    edges["weight_threshold"] = threshold
    edges["threshold_quantile"] = quantile
    return edges


def clustered_matrix(weight_df):
    """Return the matrix in the row/column order drawn by ``sns.clustermap(method='ward')``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    grid = sns.clustermap(weight_df, method="ward")
    rows = [weight_df.index[i] for i in grid.dendrogram_row.reordered_ind]
    cols = [weight_df.columns[i] for i in grid.dendrogram_col.reordered_ind]
    plt.close("all")
    out = weight_df.loc[rows, cols].copy()
    out.index.name = "sender \\ receiver"
    return out


def save_panels(figure, panels, readme_rows, provenance):
    """Write per-panel CSVs and the panel manifest consumed by build_workbooks.py.

    panels: list of dicts with keys sheet, description, df (and optional index=True).
    """
    fig_dir = PANEL_DIR / figure
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"figure": figure, "provenance": provenance, "readme": readme_rows, "sheets": []}
    for panel in panels:
        path = fig_dir / f"{panel['sheet']}.csv"
        panel["df"].to_csv(path, index=panel.get("index", False))
        manifest["sheets"].append(
            {
                "sheet": panel["sheet"],
                "csv": path.name,
                "description": panel["description"],
                "index": panel.get("index", False),
                "rows": int(len(panel["df"])),
            }
        )
        print(f"  wrote {path.relative_to(REPO)} ({len(panel['df'])} rows)")
    with open(fig_dir / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2, default=str)


def provenance_for(name, extra=None):
    """Provenance block recorded in each workbook's README sheet."""
    entry = dict(CHECKPOINTS[name])
    entry["dataset"] = name
    if extra:
        entry.update(extra)
    return entry


def ensure_dirs():
    """Create the output directories."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONHASHSEED", "0")
