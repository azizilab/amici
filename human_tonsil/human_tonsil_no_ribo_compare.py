"""Compare the no-ribosomal retrain with the main tonsil model, to decide whether to switch.

Four comparisons, each between the main model (human_tonsil_40_sweep_0f9rk0na_4303v0zu, 1000
genes) and the seed-40 retrain without RPL21/22/32/34/37, UBA52, FAU and MALAT1 (992 genes):

  1. fit: early-stopping epoch and validation loss, and per-gene variance explained beyond the
     cell-type mean on the test region, over the 992 shared genes
  2. Fig. 6b: the interaction weight matrix (significant positive genes per sender-receiver pair)
  3. Fig. 6c: top genes per pair after removing the eight genes and genes flagged by the
     segmentation checks (Methods test plus the near-sender comparison)
  4. Fig. 6d: median effective pairwise radius per valid head, FDC / Tfh -> GC B cell (resting),
     set against the 20-seed spread of the length-scale seed sweep

Run from the human_tonsil directory:  python human_tonsil_no_ribo_compare.py
"""

import json
import os

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats
from scipy.spatial import cKDTree

from amici import AMICI
from amici.interpretation._ablation_module import AMICIAblationModule

# %% Config
LABELS_KEY = "celltype_manual_fine"
COORD_KEY = "spatial"
N_NEIGHBORS = 50
UM_PER_PX = 0.120281
REMOVED_GENES = ["RPL21", "RPL22", "RPL32", "RPL34", "RPL37", "UBA52", "FAU", "MALAT1"]
OLD = {
    "data": "data/human_tonsil_filtered_2026-07-25.h5ad",
    "test": "data/human_tonsil_filtered_test_2026-07-25.h5ad",
    "model": "saved_models/human_tonsil_sweep_2026-07-25_model_2026-07-26/human_tonsil_40_sweep_0f9rk0na_4303v0zu_params_2026-07-26",
    "ablation": "figures/human_tonsil_analysis/cached_ablation_scores/all_cell_types_ablation_scores.pkl",
    "wandb_summary": "wandb/run-20260726_221010-4303v0zu/files/wandb-summary.json",
}
NEW = {
    "data": "data/human_tonsil_filtered_no_ribo_2026-07-25.h5ad",
    "test": "data/human_tonsil_filtered_no_ribo_test_2026-07-25.h5ad",
    "model": "saved_models/human_tonsil_no_ribo_retrain/seed_40",
    "ablation": "results/human_tonsil_no_ribo_retrain/all_cell_types_ablation_scores.pkl",
    "history": "results/human_tonsil_no_ribo_retrain/seed_40_history.csv",
}
RESULTS_DIR = "results/human_tonsil_no_ribo_retrain"
SEED_SWEEP_BY_RANK = (
    "results/human_tonsil_length_scale_seed_sweep_reanalysis/human_tonsil_length_scale_seed_reanalysis_by_rank.csv"
)

FIG6B_CELL_TYPES = ["GC B cell (resting)", "GC B cell (cycling)", "Follicular dendritic cell", "Macrophage (GPNMB+)"]
FIG6C_PAIRS = [
    ("CD4 T cell", "Fibroblast"),
    ("GC B cell (resting)", "Tfh cell"),
    ("GC B cell (resting)", "Follicular dendritic cell"),
]
WATCH_GENES = ["CCR7", "KLF2", "TCF7", "CXCL13", "CXCR5", "VCAM1", "CCL19", "CCL21"]
N_CANDIDATES = 60  # genes per pair run through the segmentation checks
N_TOP = 10
FIG6D_RECEIVER = "GC B cell (resting)"
FIG6D_SENDERS = ["Follicular dendritic cell", "Tfh cell"]
FIG6D_PUBLISHED_HEADS = [3, 6]
ATTENTION_THRESHOLD = 0.1
SAMPLE_THRESHOLD = 0.01
VALID_RANGE_UM = (0, 50)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)


def load(cfg):
    """Read one model's data and load its model, set up for 50-neighbour inference."""
    adata = sc.read_h5ad(cfg["data"])
    model = AMICI.load(cfg["model"], adata=adata)
    AMICI.setup_anndata(adata, labels_key=LABELS_KEY, coord_obsm_key=COORD_KEY, n_neighbors=N_NEIGHBORS)
    return adata, model


def neighbour_r2(model, test_path, genes):
    """Per-gene variance explained beyond the model's cell-type mean, on the test region."""
    adata_test = sc.read_h5ad(test_path)
    AMICI.setup_anndata(adata_test, labels_key=LABELS_KEY, coord_obsm_key=COORD_KEY, n_neighbors=N_NEIGHBORS)
    pred = model.get_predictions(adata_test, batch_size=512, prog_bar=False)
    resid = model.get_predictions(adata_test, batch_size=512, get_residuals=True, prog_bar=False)
    x = adata_test.X.toarray() if sparse.issparse(adata_test.X) else np.asarray(adata_test.X)
    baseline = pred - resid  # the cell-type mean the residual is added to
    cols = [adata_test.var_names.get_loc(g) for g in genes]
    sse = ((x[:, cols] - pred[:, cols]) ** 2).sum(axis=0)
    sse0 = ((x[:, cols] - baseline[:, cols]) ** 2).sum(axis=0)
    return pd.Series(1 - sse / np.maximum(sse0, 1e-12), index=genes)


def ablation_scores(model, adata, path):
    """All-cell-type neighbour ablation scores with z-values, cached at path."""
    if os.path.exists(path):
        return AMICIAblationModule.load_object(path)
    scores = model.get_neighbor_ablation_scores(adata=adata, compute_z_value=True)
    scores.save_object(path)
    return scores


def weight_matrix(df, drop_genes=()):
    """Significant positive genes per sender (rows) -> receiver (columns), as _get_interaction_weight_matrix."""
    df = df[~df["gene"].astype(str).isin(drop_genes)]
    cts = np.sort(df["cell_type"].unique())
    w = pd.DataFrame(0.0, index=cts, columns=cts)
    for receiver in cts:
        sub = df[df["cell_type"] == receiver]
        for sender in cts:
            if sender != receiver:
                w.loc[sender, receiver] = (
                    (sub[f"{sender}_nl10_pval_adj"] > -np.log10(0.05)) & (sub[f"{sender}_diff"] > 0)
                ).sum()
    return w


def segmentation_flags(adata, receiver, sender, genes):
    """Methods test (vs distant senders and distant receivers) plus the near-sender comparison."""
    labels = adata.obs[LABELS_KEY].astype(str).to_numpy()
    coords = np.asarray(adata.obsm[COORD_KEY], dtype=float) * UM_PER_PX
    r_idx, s_idx = np.flatnonzero(labels == receiver), np.flatnonzero(labels == sender)
    r_dist = cKDTree(coords[s_idx]).query(coords[r_idx], k=1)[0]
    s_dist = cKDTree(coords[r_idx]).query(coords[s_idx], k=1)[0]
    near_r, far_r = r_idx[r_dist <= 20], r_idx[r_dist >= 100]
    near_s, far_s = s_idx[s_dist <= 20], s_idx[s_dist >= 100]
    X = adata[:, genes].X
    X = X.toarray() if sparse.issparse(X) else np.asarray(X)
    out = {}
    for j, g in enumerate(genes):
        v = X[:, j]
        p = [stats.mannwhitneyu(v[near_r], v[grp], alternative="greater")[1] for grp in (far_s, far_r, near_s)]
        out[g] = "passes" if max(p) < 0.05 else ("Methods test" if max(p[:2]) >= 0.05 else "near check")
    return out


def ranked_pair(df, receiver, sender):
    """Significant positive genes for one pair, ordered by neighbour contribution (the dot plot's order)."""
    sub = df[df["cell_type"] == receiver]
    sig = sub[
        (sub[f"{sender}_diff"] > 0)
        & (sub[f"{sender}_nl10_pval_adj"] > -np.log10(0.05))
        & (sub[f"{sender}_z_value"] > -10)
    ]
    return sig.sort_values(f"{sender}_diff", ascending=False)["gene"].astype(str).tolist()


def radii(model, adata, n_heads):
    """Median effective pairwise radius (um) per head for FIG6D_SENDERS -> FIG6D_RECEIVER."""
    cf = model.get_counterfactual_attention_patterns(cell_type=FIG6D_RECEIVER, adata=adata)
    rows = []
    for sender in FIG6D_SENDERS:
        ls = cf._calculate_length_scales(
            head_idxs=list(range(n_heads)),
            sender_types=[sender],
            attention_threshold=ATTENTION_THRESHOLD,
            sample_threshold=SAMPLE_THRESHOLD,
        ).copy()
        ls["length_scale_um"] = ls["length_scale"] * UM_PER_PX
        for head, grp in ls.groupby("head_idx"):
            rows.append(
                {
                    "sender": sender,
                    "head_idx": int(head),
                    "n_senders": len(grp),
                    "median_um": grp["length_scale_um"].median(),
                    "q25_um": grp["length_scale_um"].quantile(0.25),
                    "q75_um": grp["length_scale_um"].quantile(0.75),
                }
            )
    out = pd.DataFrame(rows)
    lo, hi = VALID_RANGE_UM
    out["valid"] = (out["median_um"] > lo) & (out["median_um"] <= hi)
    return out


# %% Load both models
old_adata, old_model = load(OLD)
new_adata, new_model = load(NEW)
shared_genes = [g for g in old_adata.var_names if g in set(new_adata.var_names)]
summary = {}

# %% 1. Fit
with open(OLD["wandb_summary"]) as f:
    old_wandb = json.load(f)
history = pd.read_csv(NEW["history"], index_col=0)
print("=== 1. FIT ===")
print(
    f"old: stopped after epoch {old_wandb['epoch']}, last validation loss {old_wandb['validation_loss']:.2f} (1000 genes)"
)
print(
    f"new: stopped after epoch {len(history) - 1}, best validation loss {history['elbo_validation'].min():.2f} (992 genes, renormalized)"
)
r2 = pd.DataFrame(
    {
        "old": neighbour_r2(old_model, OLD["test"], shared_genes),
        "new": neighbour_r2(new_model, NEW["test"], shared_genes),
    }
)
r2.to_csv(os.path.join(RESULTS_DIR, "compare_test_neighbour_r2.csv"))
print("test-region variance explained beyond the cell-type mean, 992 shared genes:")
print(r2.describe().round(4).to_string())
print(
    f"Spearman across genes {stats.spearmanr(r2['old'], r2['new'])[0]:.3f}; "
    f"total explained (sum over genes) old {r2['old'].clip(lower=0).sum():.2f}, new {r2['new'].clip(lower=0).sum():.2f}"
)

# %% 2. Fig. 6b interaction weights
old_abl = AMICIAblationModule.load_object(OLD["ablation"])
new_abl = ablation_scores(new_model, new_adata, NEW["ablation"])
w_old = weight_matrix(old_abl._ablation_scores_df)
w_old_drop = weight_matrix(old_abl._ablation_scores_df, REMOVED_GENES)
w_new = weight_matrix(new_abl._ablation_scores_df)
off = ~np.eye(len(w_old), dtype=bool)
print("\n=== 2. FIG. 6b INTERACTION WEIGHTS (significant positive genes per sender -> receiver) ===")
for name, w in [("old", w_old), ("old minus the 8 genes", w_old_drop)]:
    a, b = w.values[off], w_new.loc[w.index, w.columns].values[off]
    top_a = set(zip(*np.where((w.values >= np.quantile(a, 0.8)) & off), strict=True))
    top_b = set(zip(*np.where((w_new.loc[w.index, w.columns].values >= np.quantile(b, 0.8)) & off), strict=True))
    print(
        f"new vs {name}: Pearson {stats.pearsonr(a, b)[0]:.3f}, Spearman {stats.spearmanr(a, b)[0]:.3f}; "
        f"edges above the 0.8-quantile threshold: {len(top_a)} vs {len(top_b)}, shared {len(top_a & top_b)} "
        f"(Jaccard {len(top_a & top_b) / len(top_a | top_b):.2f})"
    )
sub = [c for c in FIG6B_CELL_TYPES if c in w_old.index]
print("\nFig. 6b subset, old (rows = sender, columns = receiver):")
print(w_old.loc[sub, sub].astype(int).to_string())
print("Fig. 6b subset, new:")
print(w_new.loc[sub, sub].astype(int).to_string())
w_old.to_csv(os.path.join(RESULTS_DIR, "compare_weights_old.csv"))
w_new.to_csv(os.path.join(RESULTS_DIR, "compare_weights_new.csv"))

# %% 3. Fig. 6c gene lists after both filters
print("\n=== 3. FIG. 6c TOP GENES AFTER REMOVING THE 8 GENES AND SEGMENTATION-FLAGGED GENES ===")
rows = []
for receiver, sender in FIG6C_PAIRS:
    lists = {}
    for name, abl, adata in [("old", old_abl, old_adata), ("new", new_abl, new_adata)]:
        ranked = [g for g in ranked_pair(abl._ablation_scores_df, receiver, sender) if g not in REMOVED_GENES]
        flags = segmentation_flags(adata, receiver, sender, ranked[:N_CANDIDATES])
        kept = [g for g in ranked[:N_CANDIDATES] if flags[g] == "passes"]
        lists[name] = kept
        for g in WATCH_GENES:
            rows.append(
                {
                    "receiver": receiver,
                    "sender": sender,
                    "model": name,
                    "gene": g,
                    "rank_raw": ranked.index(g) + 1 if g in ranked else None,
                    "flag": flags.get(g, "beyond candidates" if g in ranked else "not significant"),
                    "rank_after_filter": kept.index(g) + 1 if g in kept else None,
                }
            )
    top_old, top_new = lists["old"][:N_TOP], lists["new"][:N_TOP]
    print(f"\n{sender} -> {receiver}")
    print(f"  old top {N_TOP}: {top_old}")
    print(f"  new top {N_TOP}: {top_new}")
    print(
        f"  shared: {len(set(top_old) & set(top_new))} of {N_TOP}; "
        f"top-20 shared: {len(set(lists['old'][:20]) & set(lists['new'][:20]))} of 20"
    )
watch = pd.DataFrame(rows)
watch.to_csv(os.path.join(RESULTS_DIR, "compare_fig6c_watch_genes.csv"), index=False)
print("\nwatched genes (rank among significant positive genes; after filtering):")
print(
    watch.pivot_table(
        index=["receiver", "sender", "gene"], columns="model", values=["rank_raw", "rank_after_filter"], aggfunc="first"
    ).to_string()
)

# %% 4. Fig. 6d radii
print("\n=== 4. FIG. 6d MEDIAN EFFECTIVE PAIRWISE RADIUS (um), FDC / Tfh -> GC B cell (resting) ===")
r_old = radii(old_model, old_adata, old_model.module.n_heads).assign(model="old")
r_new = radii(new_model, new_adata, new_model.module.n_heads).assign(model="new")
r_all = pd.concat([r_old, r_new])
r_all.to_csv(os.path.join(RESULTS_DIR, "compare_fig6d_radii.csv"), index=False)
print("published heads in the old model:")
print(r_old[r_old["head_idx"].isin(FIG6D_PUBLISHED_HEADS)].round(2).to_string(index=False))
seed = pd.read_csv(SEED_SWEEP_BY_RANK) if os.path.exists(SEED_SWEEP_BY_RANK) else None
for sender in FIG6D_SENDERS:
    print(f"\n{sender}: valid heads, ascending by median radius")
    for name, r in [("old", r_old), ("new", r_new)]:
        v = r[(r["sender"] == sender) & r["valid"]].sort_values("median_um")
        print(
            f"  {name}: " + ", ".join(f"head {h} {m:.1f}" for h, m in zip(v["head_idx"], v["median_um"], strict=True))
        )
    if seed is not None:
        s = seed[seed["sender"] == sender].groupby("head_rank")["length_scale"]
        print(
            "  20-seed sweep by head rank, median [min, max]: "
            + "; ".join(f"rank {k}: {g.median():.1f} [{g.min():.1f}, {g.max():.1f}] (n={len(g)})" for k, g in s)
        )
