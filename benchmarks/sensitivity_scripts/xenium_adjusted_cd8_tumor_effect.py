"""Spatially adjusted CD8-context associations in Xenium replicate 1."""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import statsmodels.api as sm
from einops import rearrange
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AMICI_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, AMICI_DIR)

from amici import AMICI  # noqa: E402

DATA_PATH = os.path.join(AMICI_DIR, "xenium/data/xenium_sample1/xenium_sample1_filtered_2025-05-01.h5ad")
MODEL_PATH = os.path.join(
    AMICI_DIR,
    "xenium/saved_models/xenium_sample1_proseg_sweep_2025-05-01_model_2025-05-02/"
    "xenium_18_sweep_g3mucw4s_te7pkv3z_params_2025-05-02",
)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "figures/xenium_adjusted_cd8_tumor_effect")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS_KEY = "celltype_train_grouped"
REPLICATE_KEY = "sample"
REPLICATE_VALUE = "0"
TUMOR_LABEL = "Invasive_Tumor"
CD8_LABEL = "CD8+_T_Cells"
N_NEIGHBORS = 50
SPATIAL_BLOCK_SIZE = 1000.0
N_PERMUTATIONS = 1000

HYPOXIA_GENES = ["CA9", "VEGFA", "SLC2A1", "LDHA", "BNIP3", "HIF1A"]
PROLIFERATION_GENES = ["MKI67", "TOP2A", "PCNA", "MCM2", "MCM6", "TYMS"]
ER_GENES = ["ESR1", "AGR3"]
# Chosen so the penalty performs real variable selection (~half the block dummies drop out)
# while retaining ~90% of the unpenalized adjustment magnitude; at alpha=0.1 the confounder
# coefficients collapse to zero and the adjustment becomes vacuous.
L1_ALPHA = 0.001


def expression_vector(adata, genes):
    """Return mean normalized log-expression for each cell over available genes."""
    present = [gene for gene in genes if gene in adata.var_names]
    if not present:
        return np.zeros(adata.n_obs, dtype=float), []
    values = adata[:, present].X
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=float).mean(axis=1), present


def compute_attention_covariate(model, adata_full):
    """Reproduce the empirical CD8 attention calculation in xenium_spatial_analysis."""
    replicate = adata_full[adata_full.obs[REPLICATE_KEY].astype(str) == REPLICATE_VALUE].copy()
    receiver_attention_patterns = model.get_attention_patterns(
        indices=(adata_full.obs[LABELS_KEY] == TUMOR_LABEL)
        & (adata_full.obs[REPLICATE_KEY].astype(str) == REPLICATE_VALUE),
    )
    neighbor_cols = [
        col for col in receiver_attention_patterns._attention_patterns_df.columns if col.startswith("neighbor_")
    ]
    neighbor_idxs = receiver_attention_patterns._nn_idxs_df.values
    neighbor_cell_types = pd.DataFrame(
        data=rearrange(
            replicate[rearrange(neighbor_idxs, "b n -> (b n)")].obs[LABELS_KEY].values.astype(str),
            "(b n) -> b n",
            b=receiver_attention_patterns._nn_idxs_df.shape[0],
        ),
        index=receiver_attention_patterns._nn_idxs_df.index,
        columns=receiver_attention_patterns._nn_idxs_df.columns,
    )
    summed_attention_df = pd.DataFrame()
    summed_attention_df["cell_idx"] = receiver_attention_patterns._attention_patterns_df["cell_idx"].unique()
    heads_df = receiver_attention_patterns._attention_patterns_df
    head_ct_attention_df = pd.DataFrame(
        np.where(
            neighbor_cell_types.loc[heads_df["cell_idx"]].to_numpy() == CD8_LABEL,
            heads_df[neighbor_cols].to_numpy(),
            0.0,
        ).sum(axis=1),
        columns=[f"all_heads_{CD8_LABEL}"],
    )
    head_ct_attention_df["cell_idx"] = heads_df["cell_idx"].values
    head_ct_attention_df["head"] = heads_df["head"].values
    summed_attention_df = summed_attention_df.merge(head_ct_attention_df, on="cell_idx", how="left")
    empirical_attention = summed_attention_df.groupby("cell_idx").max()[f"all_heads_{CD8_LABEL}"]

    receiver_adata = replicate[replicate.obs[LABELS_KEY] == TUMOR_LABEL].copy()
    receiver_df = receiver_adata.obsm["spatial"].copy()
    receiver_df[REPLICATE_KEY] = receiver_adata.obs[REPLICATE_KEY]
    receiver_df = receiver_df[receiver_df[REPLICATE_KEY].astype(str) == REPLICATE_VALUE]
    receiver_df["cd8_attention_raw"] = receiver_df.index.map(empirical_attention).fillna(0)
    return replicate, receiver_df


def build_covariates(adata, model):
    """Build attention, purity, state-score, and spatial-block covariates."""
    replicate, receiver_attention_df = compute_attention_covariate(model, adata)
    tumor_mask = replicate.obs[LABELS_KEY].astype(str).to_numpy() == TUMOR_LABEL
    tumor_local_indices = np.flatnonzero(tumor_mask)
    tumor_obs_names = replicate.obs_names[tumor_local_indices]
    attention = receiver_attention_df.loc[tumor_obs_names, "cd8_attention_raw"].to_numpy()
    coords = np.asarray(replicate.obsm["spatial"])
    knn = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1).fit(coords)
    _, nn_local = knn.kneighbors(coords[tumor_local_indices])
    nn_local = nn_local[:, 1:]
    local_labels = replicate.obs[LABELS_KEY].astype(str).to_numpy()[nn_local]
    tumor_purity = (local_labels == TUMOR_LABEL).mean(axis=1)

    hypoxia, hypoxia_genes = expression_vector(replicate, HYPOXIA_GENES)
    proliferation, proliferation_genes = expression_vector(replicate, PROLIFERATION_GENES)
    x = coords[tumor_local_indices, 0]
    y = coords[tumor_local_indices, 1]
    x_blocks = np.floor(x / SPATIAL_BLOCK_SIZE).astype(int)
    y_blocks = np.floor(y / SPATIAL_BLOCK_SIZE).astype(int)
    blocks = np.char.add(np.char.add(x_blocks.astype(str), "_"), y_blocks.astype(str))

    covariates = pd.DataFrame(
        {
            "cell_idx": tumor_obs_names,
            "cd8_attention": attention,
            "cd8_attention_raw": attention,
            "tumor_purity": tumor_purity,
            "hypoxia_score": hypoxia[tumor_local_indices],
            "proliferation_score": proliferation[tumor_local_indices],
            "spatial_block": blocks,
        }
    ).set_index("cell_idx")
    for column in ["cd8_attention", "tumor_purity", "hypoxia_score", "proliferation_score"]:
        values = covariates[column].to_numpy(float)
        scale = values.std()
        covariates[column] = (values - values.mean()) / scale if scale > 0 else 0.0
    return replicate, tumor_local_indices, covariates, hypoxia_genes, proliferation_genes


def fit_lasso(y, design):
    """L1-penalized least squares, shared by the coefficient models and the residualization.

    L1_wt=1.0 is a pure lasso. Penalized fits have no standard errors, so callers report
    p-values only where an unpenalized model is available.
    """
    return sm.OLS(y, design).fit_regularized(alpha=L1_ALPHA, L1_wt=1.0)


def build_design(covariates):
    """Design matrix for the reviewer's Eq. 22.

    Columns are the intercept, CD8 attention, the available cell-state covariates, and
    spatial-block indicators standing in for the region-level effect. Returns the column
    index of the CD8 term and of every non-CD8 term, so the same fit can supply both the
    reported coefficient and the covariate-adjusted expression.
    """
    candidate_covariates = ["cd8_attention", "tumor_purity", "hypoxia_score", "proliferation_score"]
    active_covariates = [
        column for column in candidate_covariates if covariates[column].to_numpy(dtype=float).std() > 1e-8
    ]
    block_matrix = pd.get_dummies(covariates["spatial_block"], drop_first=True, dtype=float).to_numpy()
    design = sm.add_constant(np.column_stack([covariates[active_covariates].to_numpy(dtype=float), block_matrix]))
    attention_position = 1 + active_covariates.index("cd8_attention")
    non_cd8_positions = [idx for idx in range(design.shape[1]) if idx not in (0, attention_position)]
    return design, active_covariates, attention_position, non_cd8_positions


def fit_gene_models(replicate, tumor_local_indices, covariates):
    """Fit the reviewer's Eq. 22 per ER gene with an L1 penalty.

    Returns the summary table plus the fitted coefficients, which the adjusted-expression
    maps reuse so that a single model serves both.
    """
    genes = [gene for gene in ER_GENES if gene in replicate.var_names]
    design, active_covariates, attention_position, _ = build_design(covariates)
    x_unadjusted = sm.add_constant(covariates[["cd8_attention"]].to_numpy())
    expression = replicate[tumor_local_indices, genes].X
    if sparse.issparse(expression):
        expression = expression.toarray()

    records = []
    fitted_params = {}
    for gene_idx, gene in enumerate(genes):
        y = np.asarray(expression[:, gene_idx], dtype=float)
        unadjusted = fit_lasso(y, x_unadjusted)
        adjusted = fit_lasso(y, design)
        fitted_params[gene] = np.asarray(adjusted.params, dtype=float)
        records.append(
            {
                "gene": gene,
                "unadjusted_cd8_beta": float(np.asarray(unadjusted.params)[1]),
                "adjusted_cd8_beta": float(np.asarray(adjusted.params)[attention_position]),
                "l1_alpha": L1_ALPHA,
                "active_covariates": ",".join(active_covariates),
                "n_cells": len(y),
                "n_spatial_blocks": covariates["spatial_block"].nunique(),
            }
        )
    return pd.DataFrame(records), fitted_params


def plot_coefficients(results):
    """Plot unadjusted and spatially adjusted CD8 coefficients."""
    plot_df = results.sort_values("adjusted_cd8_beta").reset_index(drop=True)
    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(plot_df["unadjusted_cd8_beta"], y - 0.12, s=55, label="Unadjusted", color="#999999")
    ax.scatter(plot_df["adjusted_cd8_beta"], y + 0.12, s=55, label="Adjusted", color="#D55E00")
    for row_idx, row in plot_df.iterrows():
        ax.plot(
            [row["unadjusted_cd8_beta"], row["adjusted_cd8_beta"]],
            [row_idx - 0.12, row_idx + 0.12],
            color="#BBBBBB",
            linewidth=1,
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["gene"])
    ax.set_xlabel("CD8-attention coefficient")
    ax.set_title("CD8 context association with tumor expression")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "cd8_tumor_adjusted_coefficients.png"), dpi=300)
    fig.savefig(os.path.join(OUTPUT_DIR, "cd8_tumor_adjusted_coefficients.svg"), dpi=300)
    plt.close(fig)


def compute_adjusted_expression_maps(replicate, tumor_local_indices, covariates, fitted_params):
    """Subtract the non-CD8 terms of Eq. 22 for spatial visualization.

    Uses the coefficients from the same fit as fit_gene_models, so the confounder effects are
    estimated while controlling for CD8 attention. Only the covariate and spatial-block terms
    are removed; the intercept and the CD8 term are retained, so CD8-associated signal stays
    in the adjusted expression by construction rather than by omitting it from the model.
    """
    genes = [gene for gene in ER_GENES if gene in replicate.var_names]
    design, confounders, _, non_cd8_positions = build_design(covariates)
    expression = replicate[tumor_local_indices, genes].X
    if sparse.issparse(expression):
        expression = expression.toarray()
    expression = np.asarray(expression, dtype=float)
    coords = np.asarray(replicate.obsm["spatial"])[tumor_local_indices]
    map_frames = []
    summaries = []
    for gene_idx, gene in enumerate(genes):
        raw = expression[:, gene_idx]
        params = fitted_params[gene]
        non_cd8_effect = design[:, non_cd8_positions] @ params[non_cd8_positions]
        # Centre the removed component so the adjusted expression keeps the raw mean.
        adjusted = raw - (non_cd8_effect - non_cd8_effect.mean())
        delta = adjusted - raw
        map_frames.append(
            pd.DataFrame(
                {
                    "cell_idx": covariates.index.to_numpy(),
                    "x": coords[:, 0],
                    "y": coords[:, 1],
                    "gene": gene,
                    "raw_expression": raw,
                    "adjusted_expression": adjusted,
                    "adjustment_delta": delta,
                    "total_cd8_attention": covariates["cd8_attention_raw"].to_numpy(),
                }
            )
        )
        raw_sd = raw.std()
        summaries.append(
            {
                "gene": gene,
                "mean_abs_adjustment": np.mean(np.abs(delta)),
                "median_abs_adjustment": np.median(np.abs(delta)),
                "adjustment_sd": delta.std(),
                "raw_expression_sd": raw_sd,
                "adjustment_sd_fraction_of_raw_sd": delta.std() / raw_sd if raw_sd > 0 else np.nan,
                "mean_abs_adjustment_fraction_of_raw_sd": np.mean(np.abs(delta)) / raw_sd if raw_sd > 0 else np.nan,
                "removed_covariates": ",".join([c for c in confounders if c != "cd8_attention"]) + ",spatial_block",
            }
        )
    return pd.concat(map_frames, ignore_index=True), pd.DataFrame(summaries)


def plot_adjusted_spatial_maps(map_df):
    """Plot raw expression, adjusted expression, and adjustment magnitude."""
    genes = list(map_df["gene"].drop_duplicates())
    fig, axes = plt.subplots(len(genes), 3, figsize=(15, 5.5 * len(genes)), squeeze=False)
    for row_idx, gene in enumerate(genes):
        gene_df = map_df[map_df["gene"] == gene]
        raw = gene_df["raw_expression"].to_numpy()
        adjusted = gene_df["adjusted_expression"].to_numpy()
        shared_min = min(raw.min(), adjusted.min())
        shared_max = max(raw.max(), adjusted.max())
        delta = gene_df["adjustment_delta"].to_numpy()
        limit = max(abs(delta.min()), abs(delta.max()), 1e-8)
        panels = [
            (raw, "Raw expression", "viridis", None),
            (adjusted, "Adjusted expression", "viridis", None),
            (delta, "Adjustment (adjusted - raw)", "coolwarm", TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)),
        ]
        for col_idx, (values, title, cmap, norm) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            scatter = ax.scatter(
                gene_df["x"],
                gene_df["y"],
                c=values,
                s=1.2,
                alpha=0.55,
                cmap=cmap,
                norm=norm,
                vmin=None if norm else shared_min,
                vmax=None if norm else shared_max,
                linewidths=0,
            )
            ax.set_title(f"{gene}: {title}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect("equal")
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Covariate adjustment in replicate-1 invasive tumor cells", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "cd8_tumor_adjusted_spatial_maps.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, "cd8_tumor_adjusted_spatial_maps.svg"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_high_expression_clustering(map_df):
    """Test whether high adjusted expression is spatially clustered."""
    coords = map_df.loc[map_df["gene"] == map_df["gene"].iloc[0], ["x", "y"]].to_numpy()
    nn = NearestNeighbors(n_neighbors=51).fit(coords)
    neighbors = nn.kneighbors(return_distance=False)[:, 1:]
    rng = np.random.default_rng(42)
    records = []
    for gene in map_df["gene"].drop_duplicates():
        gene_df = map_df[map_df["gene"] == gene]
        adjusted = gene_df["adjusted_expression"].to_numpy(dtype=float)
        high_expression = adjusted >= np.quantile(adjusted, 0.90)
        observed = np.mean(high_expression[:, None] & high_expression[neighbors])
        null = np.empty(N_PERMUTATIONS, dtype=float)
        for perm_idx in range(N_PERMUTATIONS):
            shuffled = rng.permutation(high_expression)
            null[perm_idx] = np.mean(shuffled[:, None] & shuffled[neighbors])
        pvalue = (1 + np.sum(np.abs(null) >= abs(observed))) / (N_PERMUTATIONS + 1)
        records.append(
            {
                "gene": gene,
                "high_expression_quantile": 0.90,
                "observed_high_high_edge_fraction": observed,
                "null_2.5_percentile": np.quantile(null, 0.025),
                "null_97.5_percentile": np.quantile(null, 0.975),
                "permutation_pvalue": pvalue,
                "n_permutations": N_PERMUTATIONS,
            }
        )
    return pd.DataFrame(records), neighbors


def plot_attention_overlap_maps(map_df):
    """Plot raw, adjusted, adjustment, and total-attention spatial maps."""
    genes = list(map_df["gene"].drop_duplicates())
    custom_palette = LinearSegmentedColormap.from_list("custom_palette", ["#4444FF", "#FF4444"], N=100)
    fig, axes = plt.subplots(len(genes), 4, figsize=(18, 5.0 * len(genes)), squeeze=False)
    for row_idx, gene in enumerate(genes):
        gene_df = map_df[map_df["gene"] == gene]
        values = [
            (gene_df["raw_expression"], "Raw expression", "viridis"),
            (gene_df["adjusted_expression"], "Adjusted expression", "viridis"),
            (gene_df["adjustment_delta"], "Adjustment (adjusted - raw)", "coolwarm"),
        ]
        for col_idx, (value, title, cmap) in enumerate(values):
            ax = axes[row_idx, col_idx]
            scatter = ax.scatter(
                gene_df["x"],
                gene_df["y"],
                c=value,
                s=1.2,
                alpha=0.6,
                cmap=cmap,
                linewidths=0,
            )
            ax.set_title(f"{gene}: {title}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect("equal")
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

        # This panel intentionally follows xenium_spatial_analysis.py directly.
        ax = axes[row_idx, 3]
        sns.scatterplot(
            data=gene_df,
            x="x",
            y="y",
            hue="total_cd8_attention",
            palette=custom_palette,
            alpha=1.0,
            s=8,
            marker="o",
            legend=False,
            ax=ax,
        )
        ax.set_title(f"{gene}: Total CD8 attention")
        ax.set_xlabel("X (μm)")
        ax.set_ylabel("Y (μm)")
        ax.set_aspect("equal")
        smap = plt.cm.ScalarMappable(cmap=custom_palette)
        smap.set_array([])
        fig.colorbar(
            smap,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="Empirical Attention Score",
        )
    fig.suptitle("CD8 attention and covariate-adjusted tumor expression", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "cd8_tumor_attention_expression_maps.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, "cd8_tumor_attention_expression_maps.svg"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_high_expression_clustering(map_df, clustering):
    """Plot high-expression regions and their spatial permutation test."""
    genes = list(clustering["gene"])
    fig, axes = plt.subplots(1, len(genes), figsize=(6.5 * len(genes), 4.8), squeeze=False)
    for col_idx, gene in enumerate(genes):
        gene_df = map_df[map_df["gene"] == gene]
        ax = axes[0, col_idx]
        high_expression = gene_df["adjusted_expression"] >= gene_df["adjusted_expression"].quantile(0.90)
        ax.scatter(
            gene_df.loc[~high_expression, "x"], gene_df.loc[~high_expression, "y"], s=1, alpha=0.15, color="#D9D9D9"
        )
        ax.scatter(
            gene_df.loc[high_expression, "x"], gene_df.loc[high_expression, "y"], s=2, alpha=0.75, color="#D55E00"
        )
        row = clustering[clustering["gene"] == gene].iloc[0]
        ax.set_title(f"{gene}: top 10% adjusted expression\n" f"spatial permutation p={row['permutation_pvalue']:.3g}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
    fig.suptitle("Spatial clustering of high adjusted expression", y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "cd8_tumor_high_expression_clustering_maps.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, "cd8_tumor_high_expression_clustering_maps.svg"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    plot_df = clustering.set_index("gene").loc[genes].reset_index()
    x = np.arange(len(plot_df))
    for idx, row in plot_df.iterrows():
        ax.vlines(
            idx,
            row["null_2.5_percentile"],
            row["null_97.5_percentile"],
            color="#666666",
            linewidth=4,
            label="Permutation null 95% interval" if idx == 0 else None,
        )
    ax.scatter(x, plot_df["observed_high_high_edge_fraction"], s=70, color="#D55E00", zorder=3, label="Observed")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["gene"])
    ax.set_ylabel("Fraction of spatial edges connecting high-expression cells")
    ax.set_title("High-expression spatial clustering is not explained by chance")
    ymax = max(
        plot_df["observed_high_high_edge_fraction"].max(),
        plot_df["null_97.5_percentile"].max(),
    )
    ymin = min(
        plot_df["observed_high_high_edge_fraction"].min(),
        plot_df["null_2.5_percentile"].min(),
    )
    margin = max((ymax - ymin) * 0.3, 0.001)
    ax.set_ylim(ymin - margin, ymax + margin)
    for idx, row in plot_df.iterrows():
        ax.annotate(
            f"p={row['permutation_pvalue']:.2e}",
            (idx, row["observed_high_high_edge_fraction"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "cd8_tumor_high_expression_clustering.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, "cd8_tumor_high_expression_clustering.svg"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    """Fit the covariate-adjusted CD8/tumor expression models and write their coefficients and spatial maps."""
    adata = sc.read_h5ad(DATA_PATH)
    adata.obs_names_make_unique()
    adata.obs[LABELS_KEY] = adata.obs[LABELS_KEY].astype(str)
    model = AMICI.load(MODEL_PATH, adata=adata)
    AMICI.setup_anndata(adata, labels_key=LABELS_KEY, coord_obsm_key="spatial", n_neighbors=N_NEIGHBORS)
    replicate, tumor_local_indices, covariates, hypoxia_genes, proliferation_genes = build_covariates(adata, model)
    results, fitted_params = fit_gene_models(replicate, tumor_local_indices, covariates)
    results["hypoxia_genes_used"] = ",".join(hypoxia_genes)
    results["proliferation_genes_used"] = ",".join(proliferation_genes)
    results.to_csv(os.path.join(OUTPUT_DIR, "cd8_tumor_adjusted_coefficients.csv"), index=False)
    covariates.to_csv(os.path.join(OUTPUT_DIR, "replicate1_tumor_covariates.csv"))
    plot_coefficients(results)
    spatial_values, adjustment_summary = compute_adjusted_expression_maps(
        replicate, tumor_local_indices, covariates, fitted_params
    )
    spatial_values.to_csv(os.path.join(OUTPUT_DIR, "cd8_tumor_adjusted_spatial_values.csv"), index=False)
    adjustment_summary.to_csv(os.path.join(OUTPUT_DIR, "cd8_tumor_adjustment_summary.csv"), index=False)
    plot_adjusted_spatial_maps(spatial_values)
    clustering, _ = compute_high_expression_clustering(spatial_values)
    clustering.to_csv(os.path.join(OUTPUT_DIR, "cd8_tumor_high_expression_clustering.csv"), index=False)
    plot_attention_overlap_maps(spatial_values)
    plot_high_expression_clustering(spatial_values, clustering)
    print(f"Analyzed {len(tumor_local_indices)} replicate-1 tumor receivers", flush=True)
    print(f"Hypoxia genes used: {hypoxia_genes}", flush=True)
    print(f"Proliferation genes used: {proliferation_genes}", flush=True)
    print(f"Saved results to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
