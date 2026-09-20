"""Spatial block-jackknife robustness check for AMICI's downstream-gene significance test.

The per-gene test in AMICIAblationModule.compute correlates |Delta_cg| (the per-cell sender contribution) with the
observed expression X_cg across receiver cells and scales the correlation by sqrt(n_t - 2), treating cells as
independent. Here we keep the same statistic but replace its null standard error with a delete-a-block jackknife
standard error, where the tissue is tiled into square spatial blocks (within each section). Blocks are chosen to be
larger than the length scales AMICI learns, so the jackknife accounts for dependence among nearby cells.

For each (receiver, sender, gene) test:
    theta   = atanh(r)                                  (Fisher transform of the full-data correlation)
    theta_b = atanh(r_{-b})                             (correlation with block b removed)
    Var_J   = delete-a-group jackknife variance with unequal block sizes (Busing et al., 1999)
    n_eff   = 3 + 1 / Var_J                             (since Var(atanh r) ~ 1 / (n - 3) under independence)
    z_eff   = r sqrt(n_eff - 2) / sqrt(1 - r^2)         (the original statistic with n_t replaced by n_eff)
    p_eff   = 1 - Phi(z_eff)                            (one-sided, as in the main analysis)
BH correction and the positive-mean-contribution filter are then applied exactly as in the main analysis.
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import norm, spearmanr
from statsmodels.stats.multitest import multipletests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AMICI_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, os.path.join(AMICI_DIR, "src"))

from amici import AMICI  # noqa: E402
from amici.interpretation._ablation_module import AMICIAblationModule  # noqa: E402

ALPHA = 0.05
EDGE_QUANTILE = 0.80

DATASETS = {
    "xenium": {
        "title": "Xenium breast cancer",
        "data": os.path.join(AMICI_DIR, "xenium/data/xenium_sample1/xenium_sample1_filtered_2025-05-01.h5ad"),
        "model": os.path.join(
            AMICI_DIR,
            "xenium/saved_models/xenium_sample1_proseg_sweep_2025-05-01_model_2025-05-02",
            "xenium_18_sweep_g3mucw4s_te7pkv3z_params_2025-05-02",
        ),
        "cache": os.path.join(AMICI_DIR, "xenium/figures/cached_ablation_scores/all_cell_types_ablation_scores.pkl"),
        "labels_key": "celltype_train_grouped",
        "section_key": "sample",
        "um_per_unit": 1.0,
    },
    "atera_breast": {
        "title": "Atera breast cancer (DCIS)",
        "data": os.path.join(AMICI_DIR, "atera_breast/data/atera_breast_filtered_2026-07-22.h5ad"),
        "model": os.path.join(
            AMICI_DIR,
            "atera_breast/saved_models/atera_breast_sweep_2026-07-22_model_2026-07-28",
            "atera_breast_33_sweep_25xpxkuk_zhtraubt_params_2026-07-28",
        ),
        "cache": os.path.join(
            AMICI_DIR,
            "atera_breast/figures/atera_breast_analysis/cached_ablation_scores/all_cell_types_ablation_scores.pkl",
        ),
        "labels_key": "celltype_train_grouped",
        "section_key": None,
        "um_per_unit": 1.0,
    },
    "human_tonsil": {
        "title": "CosMx human tonsil",
        "data": os.path.join(AMICI_DIR, "human_tonsil/data/human_tonsil_filtered_2026-07-25.h5ad"),
        "model": os.path.join(
            AMICI_DIR,
            "human_tonsil/saved_models/human_tonsil_sweep_2026-07-25_model_2026-07-26",
            "human_tonsil_40_sweep_0f9rk0na_4303v0zu_params_2026-07-26",
        ),
        "cache": os.path.join(
            AMICI_DIR,
            "human_tonsil/figures/human_tonsil_analysis/cached_ablation_scores/all_cell_types_ablation_scores.pkl",
        ),
        "labels_key": "celltype_manual_fine",
        "section_key": None,
        "um_per_unit": 0.12028,
        "fov_key": "fov",
    },
}

OUT_DIR = os.path.join(SCRIPT_DIR, "figures", "spatial_block_jackknife")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "spatial_block_jackknife")


# --------------------------------------------------------------------------------------------------------------------
# Per-cell sender contributions
# --------------------------------------------------------------------------------------------------------------------
def per_cell_contributions(model, adata, receiver, senders):
    """Yield (sender, Delta, X) for one receiver cell type.

    Delta[c, g] = yhat_cg - yhat_cg^{-sender}, the per-cell quantity that AMICIAblationModule.compute averages over
    cells (`_diff`) and correlates with X (`_z_value`). Rows follow the receiver's order in `adata`.
    """
    base_residuals, _ = AMICIAblationModule._get_ct_residuals_for_ablated_head(
        model, receiver, adata=adata, ablate_heads=False, ablated_neighbor_ct=None, head_idx=None
    )
    base = base_residuals.values.astype(np.float64)
    for sender in senders:
        ablated_residuals, gene_exp = AMICIAblationModule._get_ct_residuals_for_ablated_head(
            model, receiver, adata=adata, ablate_heads=False, ablated_neighbor_ct=sender, head_idx=None
        )
        yield sender, base - ablated_residuals.values.astype(np.float64), gene_exp.values.astype(np.float64)


# --------------------------------------------------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------------------------------------------------
def square_blocks(coords_um, block_um, section=None):
    """Tile the tissue into square spatial blocks of `block_um`, keeping sections separate."""
    ix = np.floor((coords_um[:, 0] - coords_um[:, 0].min()) / block_um).astype(np.int64)
    iy = np.floor((coords_um[:, 1] - coords_um[:, 1].min()) / block_um).astype(np.int64)
    keys = pd.Series(ix.astype(str)) + "_" + pd.Series(iy.astype(str))
    if section is not None:
        keys = pd.Series(np.asarray(section).astype(str)) + "|" + keys
    return pd.factorize(keys)[0]


# --------------------------------------------------------------------------------------------------------------------
# Correlation and jackknife from block sufficient statistics
# --------------------------------------------------------------------------------------------------------------------
def _corr(n, sx, sy, sxx, syy, sxy):
    cov = n * sxy - sx * sy
    vx = n * sxx - sx**2
    vy = n * syy - sy**2
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / np.sqrt(vx * vy)


def module_z(r, n):
    """The z-value exactly as computed in AMICIAblationModule.compute."""
    r = np.where(np.isfinite(r), r, 0.0)
    z = np.zeros_like(r)
    valid = (np.abs(r) < 1) & (n > 2)
    z[valid] = r[valid] * np.sqrt(n - 2) / np.sqrt(1 - r[valid] ** 2)
    return z


def block_jackknife(x, y, block_ids):
    """Return full-data r, jackknife z (Fisher scale) and n_eff for every column of x and y."""
    n = x.shape[0]
    blocks, inv = np.unique(block_ids, return_inverse=True)
    B = len(blocks)
    if B < 2:
        r_full = _corr(n, x.sum(0), y.sum(0), (x * x).sum(0), (y * y).sum(0), (x * y).sum(0))
        return r_full, np.zeros(x.shape[1]), np.full(x.shape[1], np.nan), B
    onehot = sparse.csr_matrix((np.ones(n), (inv, np.arange(n))), shape=(B, n))
    m = np.asarray(onehot.sum(axis=1)).ravel()

    stats_b = [onehot @ a for a in (x, y, x * x, y * y, x * y)]
    stats_full = [s.sum(axis=0) for s in stats_b]
    r_full = _corr(n, *stats_full)

    stats_loo = [full[None, :] - sb for full, sb in zip(stats_full, stats_b, strict=True)]
    r_loo = _corr((n - m)[:, None], *stats_loo)

    clip = 1 - 1e-12
    theta = np.arctanh(np.clip(r_full, -clip, clip))
    theta_loo = np.arctanh(np.clip(r_loo, -clip, clip))

    h = (n / m)[:, None]
    theta_j = B * theta - np.nansum((1 - m / n)[:, None] * theta_loo, axis=0)
    pseudo = h * theta - (h - 1) * theta_loo
    var_j = np.nanmean((pseudo - theta_j) ** 2 / (h - 1), axis=0)

    undefined = ~np.isfinite(r_full) | (var_j <= 0) | ~np.isfinite(var_j)
    z_j = np.where(undefined, 0.0, theta / np.sqrt(np.where(undefined, 1.0, var_j)))
    n_eff = np.where(undefined, np.nan, 3 + 1 / np.where(undefined, 1.0, var_j))
    return r_full, z_j, n_eff, B


# --------------------------------------------------------------------------------------------------------------------
# Spatial autocorrelation diagnostic
# --------------------------------------------------------------------------------------------------------------------
def morans_i_correlogram(adata, coords_um, section, bands_um, n_sample=20000, n_pcs=5, seed=0):
    """Moran's I of the top expression PCs as a function of distance (distance-band weights)."""
    from sklearn.neighbors import BallTree

    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, size=min(n_sample, adata.n_obs), replace=False)
    sub = adata[idx].copy()
    sc.pp.pca(sub, n_comps=n_pcs)
    z = sub.obsm["X_pca"]
    z = (z - z.mean(0)) / z.std(0)
    xy = coords_um[idx]
    sec = np.zeros(len(idx)) if section is None else pd.factorize(np.asarray(section)[idx])[0]

    tree = BallTree(xy)
    rows = []
    for lo, hi in zip(bands_um[:-1], bands_um[1:], strict=True):
        inner = tree.query_radius(xy, r=hi)
        num, w_sum = np.zeros(n_pcs), 0
        for i, nbrs in enumerate(inner):
            d = np.linalg.norm(xy[nbrs] - xy[i], axis=1)
            nbrs = nbrs[(d > lo) & (d <= hi) & (sec[nbrs] == sec[i])]
            num += (z[i][None, :] * z[nbrs]).sum(axis=0)
            w_sum += len(nbrs)
        moran = num / max(w_sum, 1)  # z is standardised, so denominator is 1 per cell
        rows.append({"band_lo_um": lo, "band_hi_um": hi, **{f"PC{k + 1}": moran[k] for k in range(n_pcs)}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------------------------------------------------
def weight_matrix(tests, cell_types, col):
    """Count the tests flagged significant in `col` for each (sender, receiver) pair."""
    counts = tests.groupby(["sender", "receiver"])[col].sum()
    mat = pd.DataFrame(0.0, index=cell_types, columns=cell_types)
    for (s, r), v in counts.items():
        mat.loc[s, r] = v
    return mat


def ward_order(mat):
    """Order rows and columns by Ward linkage so related cell types sit together in the heatmap."""
    rows = leaves_list(linkage(mat.values, method="ward"))
    cols = leaves_list(linkage(mat.values.T, method="ward"))
    return list(mat.index[rows]), list(mat.columns[cols])


def top_edges(mat, q):
    """Return the off-diagonal (sender, receiver) pairs above the `q` quantile of the weight matrix."""
    thr = np.quantile(mat.values.flatten(), q)
    return {(s, r) for s in mat.index for r in mat.columns if s != r and mat.loc[s, r] > thr}


def _heatmap(ax, mat, vmin, vmax, cmap, title, labels=None):
    n = len(mat)
    im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
    ax.set_xticks(range(n), mat.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(n), mat.index, fontsize=8)
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Receiver Cell Type")
    ax.set_ylabel("Sender Cell Type")
    ax.set_title(title, fontsize=11)
    if labels is not None:
        for i in range(n):
            for j in range(n):
                v, label = mat.values[i, j], labels.values[i, j]
                if isinstance(label, str):
                    color = "black" if np.isfinite(v) and v >= 0.5 else "white"
                    ax.text(
                        j, i, label, ha="center", va="center", fontsize=5.5 if len(label) <= 7 else 4.0, color=color
                    )
    return im


def plot_comparison(name, title, per_pair, cell_types, block_label, stats):
    """Paired interaction-weight heatmaps and a per-pair retention heatmap, built from the per-pair table."""

    def pivot(col):
        return per_pair.pivot(index="sender", columns="receiver", values=col).reindex(
            index=cell_types, columns=cell_types
        )

    mat_orig = pivot("n_sig_current").fillna(0)
    mat_jk = pivot("n_sig_jackknife").fillna(0)
    frac = pivot("frac_retained")
    n_both, n_cur = pivot("n_sig_both"), pivot("n_sig_current")
    labels = pd.DataFrame(
        [
            [f"{int(b)}/{int(c)}" if np.isfinite(c) and c > 0 else None for b, c in zip(rb, rc, strict=True)]
            for rb, rc in zip(n_both.values, n_cur.values, strict=True)
        ],
        index=cell_types,
        columns=cell_types,
    )

    rows, cols = ward_order(mat_orig)
    mat_orig, mat_jk = mat_orig.loc[rows, cols], mat_jk.loc[rows, cols]
    frac, labels = frac.loc[rows, cols], labels.loc[rows, cols]
    vmax = max(mat_orig.values.max(), 1)
    n = len(mat_orig)
    side = max(6, 0.32 * n + 3)

    fig, axes = plt.subplots(1, 2, figsize=(2 * side + 1.5, side), constrained_layout=True)
    _heatmap(axes[0], mat_orig, 0, vmax, "Reds", "Current: cells treated as independent")
    im = _heatmap(axes[1], mat_jk, 0, vmax, "Reds", f"Spatial block jackknife ({block_label} blocks)")
    fig.colorbar(im, ax=axes, shrink=0.6, label="Interaction Weight (# significant genes)")
    fig.suptitle(
        f"{title}\n"
        f"{100 * stats['frac_calls_retained']:.1f}% of significant gene calls retained   ·   "
        f"median n_eff / n (significant genes) = {stats['median_neff_ratio_sig_genes']:.2f}   ·   "
        f"Spearman ρ (edge weights) = {stats['spearman_edge_weights']:.3f}   ·   "
        f"Jaccard (top {100 * (1 - EDGE_QUANTILE):.0f}% edges) = {stats['jaccard_top_edges']:.2f}",
        fontsize=12,
    )
    for ext in ("png", "svg"):
        fig.savefig(
            os.path.join(OUT_DIR, f"{name}_jackknife_{block_label}_interaction_weight_comparison.{ext}"), dpi=300
        )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(side + 1.5, side), constrained_layout=True)
    im = _heatmap(
        ax,
        frac,
        0,
        1,
        "viridis",
        f"{title}: significant genes retained per pair ({block_label} blocks)\n"
        "cell label = retained / current; blank = no significant genes under the current test",
        labels=labels,
    )
    fig.colorbar(im, ax=ax, shrink=0.6, label="Fraction of current significant genes retained")
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}_jackknife_{block_label}_fraction_retained.{ext}"), dpi=300)
    plt.close(fig)


def replot(name, primary_block):
    """Re-render one dataset's comparison figure from its saved per-pair results."""
    stats = (
        pd.read_csv(os.path.join(RESULTS_DIR, f"{name}_summary.csv")).set_index("blocks").loc[primary_block].to_dict()
    )
    per_pair = pd.read_csv(os.path.join(RESULTS_DIR, f"{name}_jackknife_{primary_block}_per_pair.csv"))
    cell_types = np.sort(pd.unique(per_pair[["sender", "receiver"]].values.ravel()))
    plot_comparison(name, DATASETS[name]["title"], per_pair, cell_types, primary_block, stats)


# --------------------------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------------------------
def run_dataset(name, cfg, block_sizes_um, primary_block_um):
    """Run the block-jackknife significance check for one dataset across the requested block sizes."""
    print(f"=== {name}", flush=True)
    adata = sc.read_h5ad(cfg["data"])
    model = AMICI.load(cfg["model"], adata=adata)
    AMICI.setup_anndata(adata, labels_key=cfg["labels_key"], coord_obsm_key="spatial", n_neighbors=50)

    coords_um = np.asarray(adata.obsm["spatial"], dtype=np.float64)[:, :2] * cfg["um_per_unit"]
    section = adata.obs[cfg["section_key"]].values if cfg.get("section_key") else None
    labels = adata.obs[cfg["labels_key"]].astype(str).values

    schemes = {f"{b:g}um": square_blocks(coords_um, b, section) for b in block_sizes_um}
    if cfg.get("fov_key"):
        schemes["fov"] = pd.factorize(adata.obs[cfg["fov_key"]].astype(str).values)[0]

    correlogram = morans_i_correlogram(adata, coords_um, section, np.array([0, 25, 50, 100, 200, 400, 800, 1600]))
    correlogram.to_csv(os.path.join(RESULTS_DIR, f"{name}_morans_i_correlogram.csv"), index=False)
    print(correlogram.round(3).to_string(), flush=True)

    reference = AMICIAblationModule.load_object(cfg["cache"])._ablation_scores_df
    cell_types = np.sort(reference["cell_type"].unique())

    records = []
    for receiver in cell_types:
        receiver_mask = labels == receiver
        ref_ct = reference[reference["cell_type"] == receiver]
        n_t = int(receiver_mask.sum())
        senders = [s for s in cell_types if s != receiver]
        for sender, delta, x_obs in per_cell_contributions(model, adata, receiver, senders):
            abs_delta = np.abs(delta)
            ref_z = ref_ct[f"{sender}_z_value"].values.astype(float)
            rec = {
                "sender": sender,
                "receiver": receiver,
                "gene": ref_ct["gene"].values,
                "n_t": n_t,
                "diff": delta.mean(axis=0),
                "z_ref": ref_z,
                "sig_ref": (ref_ct[f"{sender}_nl10_pval_adj"].values > -np.log10(ALPHA))
                & (ref_ct[f"{sender}_diff"].values > 0),
            }
            for scheme, block_ids in schemes.items():
                r_full, z_j, n_eff, n_blocks = block_jackknife(abs_delta, x_obs, block_ids[receiver_mask])
                rec["r"] = r_full
                rec[f"z_jk_{scheme}"] = z_j
                rec[f"neff_{scheme}"] = n_eff
                rec[f"nblocks_{scheme}"] = n_blocks
            # Sanity check: the per-cell contributions reproduce the cached module z-values.
            z_check = module_z(r_full, n_t)
            assert np.allclose(z_check, ref_z, rtol=1e-3, atol=1e-2), (receiver, sender, np.abs(z_check - ref_z).max())
            records.append(pd.DataFrame(rec))
        print(f"  receiver {receiver} (n={n_t}) done", flush=True)

    tests = pd.concat(records, ignore_index=True)
    tests.to_parquet(os.path.join(RESULTS_DIR, f"{name}_jackknife_tests.parquet"), index=False)
    return summarize(name, tests, cell_types, list(schemes), f"{primary_block_um:g}um")


def neff_statistic(tests, scheme):
    """The original test statistic with n_t replaced by the block-jackknife effective sample size."""
    n_t = tests["n_t"].values.astype(float)
    if "r" in tests:
        r = tests["r"].values.astype(float)
    else:  # older result files: invert the original statistic
        z_ref = tests["z_ref"].values.astype(float)
        r = z_ref / np.sqrt(n_t - 2 + z_ref**2)
    r = np.where(np.isfinite(r), r, 0.0)
    if f"neff_{scheme}" in tests:
        n_eff = tests[f"neff_{scheme}"].values.astype(float)
    else:  # older result files: z_J = artanh(r) / sigma_J, so n_eff = 3 + (z_J / artanh(r))^2
        theta = np.arctanh(np.clip(r, -1 + 1e-12, 1 - 1e-12))
        z_j = tests[f"z_jk_{scheme}"].values.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            n_eff = np.where((z_j != 0) & (theta != 0), 3 + (z_j / theta) ** 2, np.nan)
    valid = np.isfinite(n_eff) & (n_eff > 2) & (np.abs(r) < 1)
    z = np.zeros_like(r)
    z[valid] = r[valid] * np.sqrt(n_eff[valid] - 2) / np.sqrt(1 - r[valid] ** 2)
    return z, n_eff


def summarize(name, tests, cell_types, schemes, primary_block):
    """Summarize how many originally significant genes survive each jackknife scheme, and write the results."""
    summary_rows = []
    for scheme in schemes:
        z_eff, n_eff = neff_statistic(tests, scheme)
        tests[f"neff_{scheme}"] = n_eff
        p = norm.sf(z_eff)
        p_adj = np.empty_like(p)
        for _, idx in tests.groupby(["sender", "receiver"]).indices.items():
            p_adj[idx] = multipletests(p[idx], method="fdr_bh")[1]
        tests[f"sig_jk_{scheme}"] = (p_adj <= ALPHA) & (tests["diff"].values > 0)

        mat_orig = weight_matrix(tests, cell_types, "sig_ref")
        mat_jk = weight_matrix(tests, cell_types, f"sig_jk_{scheme}")
        off = ~np.eye(len(cell_types), dtype=bool)
        e_orig, e_jk = top_edges(mat_orig, EDGE_QUANTILE), top_edges(mat_jk, EDGE_QUANTILE)
        n_ref = int(tests["sig_ref"].sum())
        n_both = int((tests["sig_ref"] & tests[f"sig_jk_{scheme}"]).sum())
        neff_ratio = tests[f"neff_{scheme}"] / tests["n_t"]
        stats = {
            "dataset": name,
            "blocks": scheme,
            "median_n_blocks_per_receiver": float(tests.groupby("receiver")[f"nblocks_{scheme}"].first().median()),
            "median_neff_ratio": float(np.nanmedian(neff_ratio)),
            "median_neff_ratio_sig_genes": float(np.nanmedian(neff_ratio[tests["sig_ref"]])),
            "n_sig_current": n_ref,
            "n_sig_jackknife": int(tests[f"sig_jk_{scheme}"].sum()),
            "n_sig_both": n_both,
            "frac_calls_retained": n_both / max(n_ref, 1),
            "spearman_edge_weights": spearmanr(mat_orig.values[off], mat_jk.values[off]).statistic,
            "jaccard_top_edges": len(e_orig & e_jk) / max(len(e_orig | e_jk), 1),
            "n_nonzero_edges_current": int((mat_orig.values[off] > 0).sum()),
            "n_nonzero_edges_jackknife": int((mat_jk.values[off] > 0).sum()),
        }
        summary_rows.append(stats)
        print(pd.Series(stats).to_string(), flush=True)

        per_pair = tests.groupby(["sender", "receiver"]).apply(
            lambda d, s=scheme: pd.Series(
                {
                    "n_receivers": d["n_t"].iloc[0],
                    "n_blocks": d[f"nblocks_{s}"].iloc[0],
                    "n_sig_current": int(d["sig_ref"].sum()),
                    "n_sig_jackknife": int(d[f"sig_jk_{s}"].sum()),
                    "n_sig_both": int((d["sig_ref"] & d[f"sig_jk_{s}"]).sum()),
                    "median_neff_ratio": float(np.nanmedian(d[f"neff_{s}"] / d["n_t"])),
                }
            )
        )
        per_pair["frac_retained"] = per_pair["n_sig_both"] / per_pair["n_sig_current"].replace(0, np.nan)
        per_pair.reset_index().sort_values("n_sig_current", ascending=False).to_csv(
            os.path.join(RESULTS_DIR, f"{name}_jackknife_{scheme}_per_pair.csv"), index=False
        )

        if scheme == primary_block:
            plot_comparison(name, DATASETS[name]["title"], per_pair.reset_index(), cell_types, scheme, stats)

    return summary_rows


def main():
    """Run the spatial block-jackknife robustness check for each requested dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["xenium", "atera_breast", "human_tonsil"])
    parser.add_argument("--block-sizes-um", nargs="+", type=float, default=[250, 500, 1000])
    parser.add_argument("--primary-block-um", type=float, default=500)
    parser.add_argument("--replot", action="store_true", help="Re-render figures from saved per-pair results.")
    parser.add_argument(
        "--from-tests", action="store_true", help="Recompute significance and figures from saved per-test results."
    )
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name in args.datasets:
        if args.replot:
            replot(name, f"{args.primary_block_um:g}um")
            continue
        if args.from_tests:
            tests = pd.read_parquet(os.path.join(RESULTS_DIR, f"{name}_jackknife_tests.parquet"))
            cell_types = np.sort(pd.unique(tests[["sender", "receiver"]].values.ravel()))
            schemes = [c.removeprefix("z_jk_") for c in tests.columns if c.startswith("z_jk_")]
            rows = summarize(name, tests, cell_types, schemes, f"{args.primary_block_um:g}um")
        else:
            rows = run_dataset(name, DATASETS[name], args.block_sizes_um, args.primary_block_um)
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, f"{name}_summary.csv"), index=False)


if __name__ == "__main__":
    main()
