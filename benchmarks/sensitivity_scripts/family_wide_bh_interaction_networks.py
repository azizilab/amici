"""Compare interaction-strength matrices under per-pair vs. family-wide Benjamini-Hochberg correction.

The ablation module applies BH across genes separately for each (receiver, sender) pair. Here we instead pool
every off-diagonal (sender, receiver, gene) test into a single family, re-apply BH, and recount the number of
significant genes with a positive mean contribution for each pair.
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import norm, spearmanr
from statsmodels.stats.multitest import multipletests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AMICI_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, os.path.join(AMICI_DIR, "src"))

from amici.interpretation._ablation_module import AMICIAblationModule  # noqa: E402

ALPHA = 0.05
EDGE_QUANTILE = 0.80  # threshold used for the directed interaction graphs in the analysis scripts

DATASETS = {
    "xenium": {
        "title": "Xenium breast cancer",
        "cache": os.path.join(AMICI_DIR, "xenium/figures/cached_ablation_scores/all_cell_types_ablation_scores.pkl"),
        "data": os.path.join(AMICI_DIR, "xenium/data/xenium_sample1/xenium_sample1_filtered_2025-05-01.h5ad"),
        "model": os.path.join(
            AMICI_DIR,
            "xenium/saved_models/xenium_sample1_proseg_sweep_2025-05-01_model_2025-05-02",
            "xenium_18_sweep_g3mucw4s_te7pkv3z_params_2025-05-02",
        ),
        "labels_key": "celltype_train_grouped",
    },
    "atera_breast": {
        "title": "Atera breast cancer (DCIS)",
        "cache": os.path.join(
            AMICI_DIR,
            "atera_breast/figures/atera_breast_analysis/cached_ablation_scores/all_cell_types_ablation_scores.pkl",
        ),
    },
    "human_tonsil": {
        "title": "CosMx human tonsil",
        "cache": os.path.join(
            AMICI_DIR,
            "human_tonsil/figures/human_tonsil_analysis/cached_ablation_scores/all_cell_types_ablation_scores.pkl",
        ),
    },
}

OUT_DIR = os.path.join(SCRIPT_DIR, "figures", "family_wide_bh")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "family_wide_bh")


def load_or_compute_ablation(cfg):
    """Load this dataset's cached ablation scores, recomputing them from the saved model when the cache is absent."""
    if os.path.exists(cfg["cache"]):
        return AMICIAblationModule.load_object(cfg["cache"])

    import scanpy as sc

    from amici import AMICI

    adata = sc.read_h5ad(cfg["data"])
    model = AMICI.load(cfg["model"], adata=adata)
    AMICI.setup_anndata(adata, labels_key=cfg["labels_key"], coord_obsm_key="spatial", n_neighbors=50)
    ablation = model.get_neighbor_ablation_scores(adata=adata, compute_z_value=True)
    os.makedirs(os.path.dirname(cfg["cache"]), exist_ok=True)
    ablation.save_object(cfg["cache"])
    return ablation


def long_form_tests(ablation):
    """One row per off-diagonal (sender, receiver, gene) test with its z-value and mean contribution."""
    df = ablation._ablation_scores_df
    cell_types = np.sort(df["cell_type"].unique())
    rows = []
    for receiver in cell_types:
        ct_df = df[df["cell_type"] == receiver]
        for sender in cell_types:
            if sender == receiver:
                continue
            rows.append(
                pd.DataFrame(
                    {
                        "sender": sender,
                        "receiver": receiver,
                        "gene": ct_df["gene"].values,
                        "z": ct_df[f"{sender}_z_value"].values.astype(float),
                        "diff": ct_df[f"{sender}_diff"].values.astype(float),
                        "nl10_pval_adj_pair": ct_df[f"{sender}_nl10_pval_adj"].values.astype(float),
                    }
                )
            )
    tests = pd.concat(rows, ignore_index=True)
    # Survival function avoids the 1 - cdf underflow for large z; identical otherwise.
    tests["p"] = norm.sf(tests["z"].values)
    tests["sig_pair"] = (tests["nl10_pval_adj_pair"] > -np.log10(ALPHA)) & (tests["diff"] > 0)
    _, p_adj_family, _, _ = multipletests(tests["p"].values, method="fdr_bh")
    tests["p_adj_family"] = p_adj_family
    tests["sig_family"] = (tests["p_adj_family"] <= ALPHA) & (tests["diff"] > 0)
    return tests, cell_types


def weight_matrix(tests, cell_types, col):
    """Count the tests flagged significant in `col` for each (sender, receiver) pair."""
    counts = tests.groupby(["sender", "receiver"])[col].sum()
    mat = pd.DataFrame(0.0, index=cell_types, columns=cell_types)
    for (s, r), v in counts.items():
        mat.loc[s, r] = v
    return mat


def ward_order(mat):
    """Order rows and columns by Ward linkage so related cell types sit together in the heatmap."""
    if len(mat) < 3:
        return list(mat.index), list(mat.columns)
    rows = leaves_list(linkage(mat.values, method="ward"))
    cols = leaves_list(linkage(mat.values.T, method="ward"))
    return list(mat.index[rows]), list(mat.columns[cols])


def top_edges(mat, threshold):
    """Return the off-diagonal (sender, receiver) pairs whose weight exceeds `threshold`."""
    vals = mat.values.copy()
    np.fill_diagonal(vals, np.nan)
    return {(s, r) for i, s in enumerate(mat.index) for j, r in enumerate(mat.columns) if vals[i, j] > threshold}


def summarize(name, tests, mat_pair, mat_family):
    """Compare the per-pair and family-wide matrices by rank correlation, top-edge overlap and significant-test counts."""
    off = ~np.eye(len(mat_pair), dtype=bool)
    w_pair, w_family = mat_pair.values[off], mat_family.values[off]
    rho = spearmanr(w_pair, w_family).statistic

    thr_pair = np.quantile(mat_pair.values.flatten(), EDGE_QUANTILE)
    thr_family = np.quantile(mat_family.values.flatten(), EDGE_QUANTILE)
    e_pair, e_family = top_edges(mat_pair, thr_pair), top_edges(mat_family, thr_family)
    jaccard = len(e_pair & e_family) / max(len(e_pair | e_family), 1)

    n_pair, n_family = int(tests["sig_pair"].sum()), int(tests["sig_family"].sum())
    return {
        "dataset": name,
        "n_cell_types": len(mat_pair),
        "n_genes": tests["gene"].nunique(),
        "n_tests_family": len(tests),
        "n_sig_per_pair_bh": n_pair,
        "n_sig_family_bh": n_family,
        "frac_calls_retained": n_family / max(n_pair, 1),
        "frac_calls_retained_if_sig_pair": float((tests["sig_pair"] & tests["sig_family"]).sum() / max(n_pair, 1)),
        "n_nonzero_edges_per_pair_bh": int((w_pair > 0).sum()),
        "n_nonzero_edges_family_bh": int((w_family > 0).sum()),
        "spearman_edge_weights": rho,
        "n_top_edges_per_pair_bh": len(e_pair),
        "n_top_edges_family_bh": len(e_family),
        "jaccard_top_edges": jaccard,
    }


def plot_pair(name, title, mat_pair, mat_family, stats):
    """Plot the per-pair and family-wide interaction matrices side by side on a shared color scale."""
    row_order, col_order = ward_order(mat_pair)
    mat_pair, mat_family = mat_pair.loc[row_order, col_order], mat_family.loc[row_order, col_order]
    vmax = max(mat_pair.values.max(), 1)

    n = len(mat_pair)
    side = max(6, 0.32 * n + 3)
    fig, axes = plt.subplots(1, 2, figsize=(2 * side + 1.5, side), constrained_layout=True)
    for ax, mat, subtitle in [
        (axes[0], mat_pair, "Current: BH within each sender–receiver pair"),
        (axes[1], mat_family, "BH across all sender–receiver–gene tests"),
    ]:
        im = ax.imshow(mat.values, cmap="Reds", vmin=0, vmax=vmax, aspect="equal", interpolation="nearest")
        ax.set_xticks(range(n), mat.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(n), mat.index, fontsize=8)
        ax.set_xticks(np.arange(-0.5, n), minor=True)
        ax.set_yticks(np.arange(-0.5, n), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="minor", length=0)
        ax.set_xlabel("Receiver Cell Type")
        ax.set_ylabel("Sender Cell Type")
        ax.set_title(subtitle, fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.6, label="Interaction Weight (# significant genes)")
    fig.suptitle(
        f"{title}\n"
        f"significant gene calls: {stats['n_sig_per_pair_bh']:,} → {stats['n_sig_family_bh']:,} "
        f"({100 * stats['frac_calls_retained_if_sig_pair']:.1f}% of current calls remain significant)"
        f"   ·   Spearman ρ (edge weights) = {stats['spearman_edge_weights']:.3f}",
        fontsize=12,
    )
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}_interaction_weight_family_bh_comparison.{ext}"), dpi=300)
    plt.close(fig)


def main():
    """Run the per-pair vs family-wide comparison for each requested dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_stats = []
    for name in args.datasets:
        cfg = DATASETS[name]
        print(f"=== {name}", flush=True)
        ablation = load_or_compute_ablation(cfg)
        tests, cell_types = long_form_tests(ablation)

        mat_pair = weight_matrix(tests, cell_types, "sig_pair")
        mat_family = weight_matrix(tests, cell_types, "sig_family")
        # Sanity check: per-pair counts must reproduce the module's interaction weight matrix.
        reference = ablation._get_interaction_weight_matrix(significance_threshold=ALPHA)
        assert np.array_equal(reference.loc[cell_types, cell_types].values, mat_pair.values), name

        stats = summarize(name, tests, mat_pair, mat_family)
        print(pd.Series(stats).to_string(), flush=True)
        all_stats.append(stats)

        per_pair = pd.DataFrame(
            {
                "n_sig_per_pair_bh": mat_pair.stack(),
                "n_sig_family_bh": mat_family.stack(),
            }
        )
        per_pair.index.names = ["sender", "receiver"]
        per_pair = per_pair.reset_index()
        per_pair = per_pair[per_pair["sender"] != per_pair["receiver"]]
        per_pair["frac_retained"] = per_pair["n_sig_family_bh"] / per_pair["n_sig_per_pair_bh"].replace(0, np.nan)
        per_pair.sort_values("n_sig_per_pair_bh", ascending=False).to_csv(
            os.path.join(RESULTS_DIR, f"{name}_per_pair_counts.csv"), index=False
        )

        plot_pair(name, cfg["title"], mat_pair, mat_family, stats)
        del ablation, tests

    pd.DataFrame(all_stats).to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)


if __name__ == "__main__":
    main()
